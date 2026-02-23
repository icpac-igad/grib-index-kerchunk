#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gcsfs",
#     "pandas",
#     "pyarrow",
#     "huggingface_hub",
# ]
# ///
"""
Consolidate GIK ECMWF parquets into combined files for HuggingFace.

Reads per-member parquets from GCS:
    gs://gik-ecmwf-aws-tf/run_par_ecmwf/{YYYY}/{MM}/{YYYYMMDD}/{HH}z/*.parquet

Produces one combined parquet per run hour:
    combined/ecmwf_gik_{HH}z.parquet

Each combined file contains all dates and members with columns:
    date (str YYYYMMDD), member (str), key (str zarr path), value (bytes)

Sorted by (date, member) for efficient row-group filtering via pyarrow
predicate pushdown.

Usage:
    uv run consolidate_parquets_to_hf.py                     # all 4 run hours
    uv run consolidate_parquets_to_hf.py --run 00             # 00z only
    uv run consolidate_parquets_to_hf.py --run 00 --dry-run   # count files only
    uv run consolidate_parquets_to_hf.py --run 00 --local-only /tmp/combined  # write locally, skip HF upload
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gcsfs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
GCS_SA_FILE = str(SCRIPT_DIR / "coiled-data.json")
GCS_BUCKET = "gik-ecmwf-aws-tf"
GCS_PREFIX = "run_par_ecmwf"
HF_REPO = "E4DRR/gik-ecmwf-par"

# Member name mapping: parquet filename stem -> canonical member id
# e.g. "2024030100z-control.parquet" -> "control"
#      "2024030100z-ens01.parquet"   -> "ens_01"
YEARS = ["2024", "2025", "2026"]
RUN_HOURS = ["00", "06", "12", "18"]


def parquet_stem_to_member_id(stem: str) -> str:
    """Extract canonical member id from parquet filename stem.

    '2024030100z-control' -> 'control'
    '2024030100z-ens01'   -> 'ens_01'
    """
    # stem is like "2024030100z-control" or "2024030100z-ens01"
    member_part = stem.split("-", 1)[1]  # "control" or "ens01"
    if member_part.startswith("ens") and member_part != "ens":
        # "ens01" -> "ens_01"
        num = member_part[3:]
        return f"ens_{num}"
    return member_part


def get_gcs_fs():
    return gcsfs.GCSFileSystem(token=GCS_SA_FILE)


def discover_dates(fs, years: list[str],
                   month_filter: str | None = None) -> list[tuple[str, str, str]]:
    """Discover all (year, month, date_str) tuples on GCS.

    Does NOT check per-date run hour availability — that is handled at
    process_date() time.  This avoids O(n_dates) extra GCS API calls.

    Returns sorted list of (year, month, date_str) where date_str is YYYYMMDD.
    """
    results = []
    for year in years:
        try:
            month_dirs = sorted(fs.ls(f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/"))
        except FileNotFoundError:
            continue

        months = [m.split("/")[-1] for m in month_dirs]
        if month_filter:
            months = [m for m in months if m == month_filter]

        for month in months:
            try:
                date_dirs = sorted(fs.ls(f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/{month}"))
            except FileNotFoundError:
                continue

            for date_dir in date_dirs:
                date_name = date_dir.split("/")[-1]
                results.append((year, month, date_name))

    return sorted(results, key=lambda x: x[2])


def process_date(fs, year: str, month: str, date_str: str,
                 run_hour: str) -> pd.DataFrame | None:
    """Read all member parquets for one date/run and return combined DataFrame.

    Adds 'date' and 'member' columns. Returns None if no files found.
    """
    run_path = f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/{month}/{date_str}/{run_hour}z"
    try:
        parquets = sorted([f for f in fs.ls(run_path) if f.endswith(".parquet")])
    except FileNotFoundError:
        return None

    if not parquets:
        return None

    dfs = []
    for pf in parquets:
        basename = os.path.basename(pf)
        stem = basename.replace(".parquet", "")
        member_id = parquet_stem_to_member_id(stem)

        try:
            with fs.open(pf, "rb") as f:
                df = pd.read_parquet(f)
        except Exception as e:
            print(f"    WARN: failed to read {pf}: {e}")
            continue

        df.insert(0, "date", date_str)
        df.insert(1, "member", member_id)
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def consolidate_run_hour(fs, run_hour: str, years: list[str],
                         month_filter: str | None, dry_run: bool,
                         output_dir: str | None) -> str | None:
    """Consolidate all dates for one run hour into a single parquet file.

    Returns path to the local combined parquet file, or None if dry_run.
    """
    print(f"\n{'='*70}")
    print(f"Run hour: {run_hour}z")
    print(f"{'='*70}")

    dates = discover_dates(fs, years, month_filter)
    print(f"  Discovered {len(dates)} dates")

    if not dates:
        print("  No dates found, skipping")
        return None

    if dry_run:
        print(f"  DRY RUN: would consolidate {len(dates)} dates × ~51 members")
        for y, m, d in dates[:5]:
            print(f"    {d}/{run_hour}z")
        if len(dates) > 5:
            print(f"    ... and {len(dates) - 5} more")
        return None

    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"ecmwf_gik_{run_hour}z.parquet")
    else:
        out_path = os.path.join(tempfile.mkdtemp(), f"ecmwf_gik_{run_hour}z.parquet")

    # Stream dates directly to parquet via ParquetWriter to avoid OOM.
    # Each date produces ~51 members × ~2500 rows ≈ 128k rows (~10 MB),
    # written immediately then discarded.
    writer = None
    current_month = None
    month_count = 0
    total_rows = 0
    dates_written = 0

    t0 = time.time()
    for i, (year, month, date_str) in enumerate(dates):
        month_key = f"{year}/{month}"
        if month_key != current_month:
            if current_month is not None:
                print(f"    {current_month}: {month_count} dates processed",
                      flush=True)
            current_month = month_key
            month_count = 0

        df = process_date(fs, year, month, date_str, run_hour)
        if df is not None:
            # Sort within this date for row-group locality
            df.sort_values(["date", "member"], inplace=True, ignore_index=True)
            table = pa.Table.from_pandas(df, preserve_index=False)
            del df

            if writer is None:
                writer = pq.ParquetWriter(
                    out_path,
                    table.schema,
                    compression="snappy",
                    use_dictionary=["date", "member"],
                )
            writer.write_table(table)
            total_rows += table.num_rows
            del table
            month_count += 1
            dates_written += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {i+1}/{len(dates)} dates, "
                  f"{dates_written} written, {total_rows:,} rows ({elapsed:.0f}s)",
                  flush=True)

    if current_month is not None:
        print(f"    {current_month}: {month_count} dates processed", flush=True)

    if writer is None:
        print("  No data collected, skipping")
        return None

    writer.close()

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  Written: {out_path}")
    print(f"    Dates: {dates_written}, Rows: {total_rows:,}")
    print(f"    Size: {file_size_mb:.0f} MB, Time: {elapsed:.0f}s")

    return out_path


def upload_to_hf(api, local_path: str, run_hour: str):
    """Upload combined parquet to HuggingFace."""
    hf_path = f"combined/ecmwf_gik_{run_hour}z.parquet"
    print(f"  Uploading to {HF_REPO}/{hf_path} ...")

    t0 = time.time()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_path,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    elapsed = time.time() - t0
    print(f"  Uploaded in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate GIK ECMWF parquets into combined files for HuggingFace")
    parser.add_argument("--run", type=str, default=None,
                        help="Single run hour: 00, 06, 12, 18 (default: all)")
    parser.add_argument("--year", type=str, default=None,
                        help="Single year (default: 2024,2025,2026)")
    parser.add_argument("--month", type=str, default=None,
                        help="Single month MM (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without downloading or uploading")
    parser.add_argument("--local-only", type=str, default=None,
                        help="Write combined parquet locally and skip HF upload")
    parser.add_argument("--no-upload", action="store_true",
                        help="Write to temp dir but skip HF upload")
    args = parser.parse_args()

    fs = get_gcs_fs()

    if args.run:
        run_hours = [args.run]
    else:
        run_hours = RUN_HOURS

    years = [args.year] if args.year else YEARS

    print("=" * 70)
    print("Consolidate GIK ECMWF Parquets → Combined Files")
    print("=" * 70)
    print(f"GCS source:  gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print(f"HF target:   {HF_REPO}/combined/")
    print(f"Run hours:   {run_hours}")
    print(f"Years:       {years}")
    if args.month:
        print(f"Month:       {args.month}")
    if args.dry_run:
        print("Mode:        DRY RUN")
    if args.local_only:
        print(f"Output:      {args.local_only} (local only, no HF upload)")
    print("=" * 70)

    skip_upload = args.dry_run or args.no_upload or args.local_only

    if not skip_upload:
        from huggingface_hub import HfApi
        api = HfApi()
    else:
        api = None

    overall_t0 = time.time()

    for run_hour in run_hours:
        out_path = consolidate_run_hour(
            fs, run_hour, years, args.month, args.dry_run,
            output_dir=args.local_only,
        )

        if out_path and not skip_upload:
            upload_to_hf(api, out_path, run_hour)

            # Clean up temp file if not using --local-only
            if not args.local_only:
                try:
                    parent = os.path.dirname(out_path)
                    shutil.rmtree(parent)
                except Exception:
                    pass

    total_time = time.time() - overall_t0
    print(f"\n{'='*70}")
    print(f"COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
