#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gcsfs",
#     "huggingface_hub",
#     "pyarrow",
# ]
# ///
"""
Upload GIK GEFS parquet files from GCS to HuggingFace.

Mirrors the ECMWF version (`ecmwf/upload_parquets_to_hf.py`) for the GEFS
ensemble (30 members: gep01-gep30).

Copies the nested parquet structure from:
    gs://${GCS_BUCKET}/run_par_gefs/{YYYY}/{MM}/{YYYYMMDD}/{run}z/*.parquet

To HuggingFace dataset (default: E4DRR/gik-gefs-par):
    run_par_gefs/{YYYY}/{MM}/{YYYYMMDD}/{run}z/

Aggregate mode (--aggregate) concatenates all parquet files for a given
month + run hour into a single parquet and uploads to:
    run_par_gefs_agg/monthly_agg/{YYYY}/{MM}_{run}z.parquet

Catalog mode (--catalog) builds a lightweight index of every file:
    run_par_gefs_agg/catalog.parquet

NOAA's GEFS realtime/reforecast archive on s3://noaa-gefs-pds/ starts on
2020-09-25; earlier dates have no upstream data.

Env vars
--------
    GCS_BUCKET        Required. GCS bucket name (no gs:// prefix).
    GCS_SA_FILE       Optional. Path to GCS service-account JSON. If
                      unset, falls back to GOOGLE_APPLICATION_CREDENTIALS,
                      then to gcsfs Application Default Credentials.
    HF_REPO           Optional. HuggingFace dataset repo id.
                      Default: "E4DRR/gik-gefs-par".
    HF_TOKEN          Required for upload (read by huggingface_hub).

Usage
-----
    export GCS_BUCKET=your-bucket-name
    export HF_TOKEN=hf_xxx

    uv run upload_parquets_to_hf.py                          # all default years
    uv run upload_parquets_to_hf.py --year 2024              # single year
    uv run upload_parquets_to_hf.py --year 2024 --month 03   # single month
    uv run upload_parquets_to_hf.py --run 00                 # single run hour
    uv run upload_parquets_to_hf.py --sync                   # upload only missing dates
    uv run upload_parquets_to_hf.py --sync --run 18 --dry-run # show missing 18z dates
    uv run upload_parquets_to_hf.py --aggregate --year 2024  # monthly aggregates
    uv run upload_parquets_to_hf.py --catalog                # build & upload catalog
    uv run upload_parquets_to_hf.py --catalog --dry-run      # preview catalog
    uv run upload_parquets_to_hf.py --dry-run                # list without uploading
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_SA_FILE = (
    os.environ.get("GCS_SA_FILE")
    or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
)
HF_REPO = os.environ.get("HF_REPO", "E4DRR/gik-gefs-par")
GCS_PREFIX = "run_par_gefs"
AGG_PREFIX = "run_par_gefs_agg"
DEFAULT_YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]


def get_gcs_fs():
    """Return a gcsfs filesystem.

    Honours GCS_SA_FILE / GOOGLE_APPLICATION_CREDENTIALS env vars; falls
    back to Application Default Credentials when neither is set.
    """
    if GCS_SA_FILE and os.path.exists(GCS_SA_FILE):
        return gcsfs.GCSFileSystem(token=GCS_SA_FILE)
    return gcsfs.GCSFileSystem()


def get_hf_dates(api, run_filter: str = None) -> dict[str, set[str]]:
    """List all (run_hour -> set of date strings) already on HuggingFace.

    Returns e.g. {'00z': {'20240301', '20240302', ...}, '12z': {...}}.
    """
    print("  Scanning HuggingFace repo for existing dates...", flush=True)
    t0 = time.time()
    files = list(api.list_repo_tree(
        repo_id=HF_REPO, repo_type="dataset", recursive=True,
    ))

    dates_by_run: dict[str, set[str]] = {}
    for f in files:
        path = f.path if hasattr(f, "path") else str(f)
        if not path.endswith(".parquet"):
            continue
        parts = path.split("/")
        if len(parts) >= 5 and parts[0] == GCS_PREFIX:
            run_hour = parts[4]  # e.g. "00z"
            date_str = parts[3]  # e.g. "20240301"
            if run_filter and run_hour != f"{run_filter}z":
                continue
            dates_by_run.setdefault(run_hour, set()).add(date_str)

    elapsed = time.time() - t0
    for run in sorted(dates_by_run):
        print(f"    HF {run}: {len(dates_by_run[run])} dates", flush=True)
    print(f"  HF scan done in {elapsed:.0f}s", flush=True)
    return dates_by_run


def build_catalog(fs, api, years: list[str], dry_run: bool = False,
                  run_filter: str = None, month_filter: str = None):
    """Scan GCS and build a lightweight catalog parquet listing every file.

    Columns: year, month, date, run, member, filename, hf_path, size_bytes.
    Uploaded to: {AGG_PREFIX}/catalog.parquet
    """
    print("Scanning GCS for all parquet files...", flush=True)
    t0 = time.time()

    rows = []
    for year in years:
        if month_filter:
            months_to_scan = [month_filter]
        else:
            try:
                month_dirs = sorted(fs.ls(f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/"))
                months_to_scan = [m.split("/")[-1] for m in month_dirs]
            except FileNotFoundError:
                print(f"  Year {year} not found on GCS, skipping")
                continue

        for month in months_to_scan:
            gcs_month = f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/{month}"
            try:
                date_dirs = sorted(fs.ls(gcs_month))
            except FileNotFoundError:
                continue

            for date_dir in date_dirs:
                date_name = date_dir.split("/")[-1]
                try:
                    run_dirs = fs.ls(date_dir)
                except FileNotFoundError:
                    continue

                for run_dir in run_dirs:
                    run_name = run_dir.split("/")[-1]  # e.g. 00z
                    if run_filter and run_name != f"{run_filter}z":
                        continue

                    try:
                        entries = fs.ls(run_dir, detail=True)
                    except FileNotFoundError:
                        continue

                    for entry in entries:
                        name = entry["name"]
                        if not name.endswith(".parquet"):
                            continue
                        fname = name.split("/")[-1]
                        # Extract member from filename: {date}{run}z-{member}.parquet
                        base = fname.rsplit(".parquet", 1)[0]
                        parts = base.split("-", 1)
                        member = parts[1] if len(parts) == 2 else base

                        hf_path = (f"{GCS_PREFIX}/{year}/{month}/"
                                   f"{date_name}/{run_name}/{fname}")
                        rows.append({
                            "year": year,
                            "month": month,
                            "date": date_name,
                            "run": run_name,
                            "member": member,
                            "filename": fname,
                            "hf_path": hf_path,
                            "size_bytes": entry.get("size", 0),
                        })

            print(f"  {year}/{month}: {sum(1 for r in rows if r['year']==year and r['month']==month)} files",
                  flush=True)

    elapsed = time.time() - t0
    print(f"GCS scan: {len(rows)} files found in {elapsed:.0f}s", flush=True)

    if not rows:
        print("No files found, nothing to upload.")
        return

    table = pa.table({
        "year":       pa.array([r["year"] for r in rows], type=pa.string()),
        "month":      pa.array([r["month"] for r in rows], type=pa.string()),
        "date":       pa.array([r["date"] for r in rows], type=pa.string()),
        "run":        pa.array([r["run"] for r in rows], type=pa.string()),
        "member":     pa.array([r["member"] for r in rows], type=pa.string()),
        "filename":   pa.array([r["filename"] for r in rows], type=pa.string()),
        "hf_path":    pa.array([r["hf_path"] for r in rows], type=pa.string()),
        "size_bytes": pa.array([r["size_bytes"] for r in rows], type=pa.int64()),
    })

    dates = set(r["date"] for r in rows)
    runs = set(r["run"] for r in rows)
    members = set(r["member"] for r in rows)
    total_size = sum(r["size_bytes"] for r in rows)
    print(f"\nCatalog summary:")
    print(f"  Files:   {len(rows):,}")
    print(f"  Dates:   {len(dates)} ({min(dates)}..{max(dates)})")
    print(f"  Runs:    {sorted(runs)}")
    print(f"  Members: {len(members)} ({sorted(members)[:5]}...)")
    print(f"  Total:   {total_size / (1024**3):.2f} GB")

    hf_path = f"{AGG_PREFIX}/catalog.parquet"

    if dry_run:
        print(f"\n  [DRY RUN] Would upload catalog to {hf_path}")
        return

    tmpdir = tempfile.mkdtemp()
    try:
        out_path = os.path.join(tmpdir, "catalog.parquet")
        pq.write_table(table, out_path)
        size_kb = os.path.getsize(out_path) / 1024
        api.upload_file(
            path_or_fileobj=out_path,
            path_in_repo=hf_path,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print(f"\n  Uploaded {hf_path} ({size_kb:.0f} KB, {len(rows):,} entries)",
              flush=True)
    except Exception as e:
        print(f"  Catalog upload FAILED — {e}")
    finally:
        shutil.rmtree(tmpdir)


def aggregate_month(fs, api, year: str, month: str, run_hour: str,
                    dry_run: bool = False):
    """Aggregate all parquets for one month+run into a single parquet file.

    Downloads every parquet for {year}/{month}/*//{run_hour}z/ from GCS,
    reads them with pyarrow, adds 'date' and 'source_file' columns,
    concatenates into one table, and uploads as:
        {AGG_PREFIX}/monthly_agg/{year}/{month}_{run_hour}z.parquet

    Returns the number of source parquet files aggregated.
    """
    gcs_month_path = f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/{month}"

    try:
        date_dirs = sorted(fs.ls(gcs_month_path))
    except FileNotFoundError:
        print(f"  {year}/{month}: not found on GCS, skipping")
        return 0

    all_parquets = []  # list of (date_name, gcs_path)
    for date_dir in date_dirs:
        date_name = date_dir.split("/")[-1]  # e.g. 20240301
        run_dir = f"{date_dir}/{run_hour}z"
        try:
            entries = fs.ls(run_dir)
        except FileNotFoundError:
            continue
        parquets = [f for f in entries if f.endswith(".parquet")]
        for pf in parquets:
            all_parquets.append((date_name, pf))

    if not all_parquets:
        print(f"  {year}/{month}/{run_hour}z: no parquets found")
        return 0

    hf_path = f"{AGG_PREFIX}/monthly_agg/{year}/{month}_{run_hour}z.parquet"

    if dry_run:
        dates_found = sorted(set(d for d, _ in all_parquets))
        print(f"  [AGG] {hf_path}: {len(all_parquets)} parquets "
              f"from {len(dates_found)} dates "
              f"({dates_found[0]}..{dates_found[-1]})")
        return len(all_parquets)

    tmpdir = tempfile.mkdtemp()
    try:
        tables = []
        for date_name, pf in all_parquets:
            fname = os.path.basename(pf)
            local_path = os.path.join(tmpdir, f"{date_name}_{fname}")
            fs.get(pf, local_path)

            table = pq.read_table(local_path)
            n_rows = len(table)
            date_col = pa.array([date_name] * n_rows, type=pa.string())
            source_col = pa.array([fname] * n_rows, type=pa.string())
            table = table.append_column("date", date_col)
            table = table.append_column("source_file", source_col)
            tables.append(table)
            os.remove(local_path)

        combined = pa.concat_tables(tables, promote_options="default")
        out_path = os.path.join(tmpdir, "aggregated.parquet")
        pq.write_table(combined, out_path)

        api.upload_file(
            path_or_fileobj=out_path,
            path_in_repo=hf_path,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {hf_path}: {len(all_parquets)} parquets → "
              f"{len(combined)} rows, {size_mb:.1f} MB uploaded",
              flush=True)
    except Exception as e:
        print(f"  {hf_path}: FAILED — {e}")
    finally:
        shutil.rmtree(tmpdir)

    return len(all_parquets)


def upload_month(fs, api, year: str, month: str, dry_run: bool = False,
                 run_filter: str = None, skip_dates: dict[str, set[str]] = None):
    """Download all parquets for one month from GCS, upload batch to HF.

    skip_dates: if provided, {run_hour -> set of date_strs} already on HF.
    """
    gcs_month_path = f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/{month}"

    try:
        date_dirs = sorted(fs.ls(gcs_month_path))
    except FileNotFoundError:
        print(f"  {year}/{month}: not found on GCS, skipping")
        return 0, 0

    total_files = 0
    skipped_dates = 0

    for date_dir in date_dirs:
        date_name = date_dir.split("/")[-1]  # e.g. 20240301
        run_dirs = fs.ls(date_dir)

        for run_dir in run_dirs:
            run_name = run_dir.split("/")[-1]  # e.g. 00z
            if run_filter and run_name != f"{run_filter}z":
                continue

            if skip_dates and date_name in skip_dates.get(run_name, set()):
                skipped_dates += 1
                continue

            parquets = [f for f in fs.ls(run_dir) if f.endswith(".parquet")]
            if not parquets:
                continue

            hf_path = f"{GCS_PREFIX}/{year}/{month}/{date_name}/{run_name}"

            if dry_run:
                print(f"  [MISSING] {hf_path}: {len(parquets)} parquets")
                total_files += len(parquets)
                continue

            tmpdir = tempfile.mkdtemp()
            try:
                for pf in parquets:
                    local_path = os.path.join(tmpdir, os.path.basename(pf))
                    fs.get(pf, local_path)

                api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=HF_REPO,
                    path_in_repo=hf_path,
                    repo_type="dataset",
                )
                total_files += len(parquets)
                print(f"  {hf_path}: {len(parquets)} parquets uploaded",
                      flush=True)
            except Exception as e:
                print(f"  {hf_path}: FAILED — {e}")
            finally:
                shutil.rmtree(tmpdir)

    return total_files, skipped_dates


def main():
    parser = argparse.ArgumentParser(
        description="Upload GIK GEFS parquets from GCS to HuggingFace")
    parser.add_argument("--year", type=str, default=None,
                        help=f"Single year to upload (default: {','.join(DEFAULT_YEARS)})")
    parser.add_argument("--month", type=str, default=None,
                        help="Single month MM to upload (requires --year)")
    parser.add_argument("--run", type=str, default=None,
                        help="Run hour filter: 00, 06, 12, 18 (default: all)")
    parser.add_argument("--sync", action="store_true",
                        help="Only upload dates missing from HF (diff-based)")
    parser.add_argument("--catalog", action="store_true",
                        help="Build a lightweight catalog/index parquet")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate parquets by month+run into single files")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    args = parser.parse_args()

    if not GCS_BUCKET:
        print("ERROR: set GCS_BUCKET env var to your bucket name", file=sys.stderr)
        sys.exit(1)

    fs = get_gcs_fs()
    api = HfApi()

    if args.year:
        years = [args.year]
    else:
        years = DEFAULT_YEARS

    print("=" * 70)
    print("Upload GIK GEFS Parquets: GCS → HuggingFace")
    print("=" * 70)
    print(f"GCS source: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print(f"HF target:  {HF_REPO}")
    print(f"Years:      {years}")
    if args.month:
        print(f"Month:      {args.month}")
    if args.run:
        print(f"Run:        {args.run}z")
    if args.catalog:
        print(f"Mode:       CATALOG (index → {AGG_PREFIX}/catalog.parquet)")
    if args.aggregate:
        print(f"Mode:       AGGREGATE (monthly parquets → {AGG_PREFIX}/)")
    if args.sync:
        print("Mode:       SYNC (upload missing dates only)")
    if args.dry_run:
        print("Mode:       DRY RUN")
    print("=" * 70)

    if args.catalog:
        build_catalog(fs, api, years, dry_run=args.dry_run,
                      run_filter=args.run, month_filter=args.month)
        return

    skip_dates = None
    if args.sync and not args.aggregate:
        skip_dates = get_hf_dates(api, run_filter=args.run)

    run_hours = [args.run] if args.run else ["00", "06", "12", "18"]

    overall_t0 = time.time()
    grand_total = 0
    grand_skipped = 0

    for year in years:
        print(f"\n{'='*70}")
        print(f"Year: {year}")
        print(f"{'='*70}")

        if args.month:
            months = [args.month]
        else:
            try:
                month_dirs = sorted(fs.ls(f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/"))
                months = [m.split("/")[-1] for m in month_dirs]
            except FileNotFoundError:
                print(f"  Year {year} not found on GCS, skipping")
                continue

        year_total = 0
        year_skipped = 0

        if args.aggregate:
            for month in months:
                for run_hour in run_hours:
                    t0 = time.time()
                    n = aggregate_month(fs, api, year, month, run_hour,
                                        dry_run=args.dry_run)
                    elapsed = time.time() - t0
                    year_total += n
                    if not args.dry_run and n > 0:
                        print(f"  Month {month}/{run_hour}z: "
                              f"{n} parquets aggregated in {elapsed:.0f}s")
        else:
            for month in months:
                t0 = time.time()
                n, skipped = upload_month(fs, api, year, month,
                                          dry_run=args.dry_run,
                                          run_filter=args.run,
                                          skip_dates=skip_dates)
                elapsed = time.time() - t0
                year_total += n
                year_skipped += skipped
                if not args.dry_run and n > 0:
                    print(f"  Month {month}: {n} files in {elapsed:.0f}s")

        msg = f"  Year {year} total: {year_total} parquets"
        if skip_dates:
            msg += f" ({year_skipped} dates skipped — already on HF)"
        print(msg)
        grand_total += year_total
        grand_skipped += year_skipped

    total_time = time.time() - overall_t0
    print(f"\n{'='*70}")
    msg = f"COMPLETE: {grand_total} parquets in {total_time:.0f}s ({total_time/60:.1f} min)"
    if skip_dates:
        msg += f"\n  Skipped: {grand_skipped} dates already on HF"
    print(msg)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
