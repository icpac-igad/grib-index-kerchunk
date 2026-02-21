#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gcsfs",
#     "huggingface_hub",
# ]
# ///
"""
Upload GIK ECMWF parquet files from GCS to HuggingFace.

Copies the nested parquet structure from:
    gs://gik-ecmwf-aws-tf/run_par_ecmwf/{YYYY}/{MM}/{YYYYMMDD}/00z/*.parquet

To HuggingFace dataset:
    E4DRR/gik-ecmwf-par  (under run_par_ecmwf/{YYYY}/{MM}/{YYYYMMDD}/00z/)

Processes year-by-year, month-by-month to manage memory. Downloads each
parquet from GCS to a temp directory, then uploads the batch to HF.

Usage:
    uv run upload_parquets_to_hf.py                     # all 3 years
    uv run upload_parquets_to_hf.py --year 2025          # single year
    uv run upload_parquets_to_hf.py --dry-run             # list without uploading
    uv run upload_parquets_to_hf.py --year 2024 --month 03  # single month
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gcsfs
from huggingface_hub import HfApi

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
GCS_SA_FILE = str(SCRIPT_DIR / "coiled-data.json")
GCS_BUCKET = "gik-ecmwf-aws-tf"
GCS_PREFIX = "run_par_ecmwf"
HF_REPO = "E4DRR/gik-ecmwf-par"


def get_gcs_fs():
    return gcsfs.GCSFileSystem(token=GCS_SA_FILE)


def upload_month(fs, api, year: str, month: str, dry_run: bool = False,
                 run_filter: str = None):
    """Download all parquets for one month from GCS, upload batch to HF."""
    gcs_month_path = f"{GCS_BUCKET}/{GCS_PREFIX}/{year}/{month}"

    try:
        date_dirs = sorted(fs.ls(gcs_month_path))
    except FileNotFoundError:
        print(f"  {year}/{month}: not found on GCS, skipping")
        return 0

    total_files = 0

    for date_dir in date_dirs:
        date_name = date_dir.split("/")[-1]  # e.g. 20240301
        run_dirs = fs.ls(date_dir)

        for run_dir in run_dirs:
            run_name = run_dir.split("/")[-1]  # e.g. 00z
            if run_filter and run_name != f"{run_filter}z":
                continue
            parquets = [f for f in fs.ls(run_dir) if f.endswith(".parquet")]

            if not parquets:
                continue

            hf_path = f"{GCS_PREFIX}/{year}/{month}/{date_name}/{run_name}"

            if dry_run:
                print(f"  {hf_path}: {len(parquets)} parquets")
                total_files += len(parquets)
                continue

            # Download to temp dir, then upload as folder
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
                print(f"  {hf_path}: {len(parquets)} parquets uploaded")
            except Exception as e:
                print(f"  {hf_path}: FAILED — {e}")
            finally:
                shutil.rmtree(tmpdir)

    return total_files


def main():
    parser = argparse.ArgumentParser(
        description="Upload GIK ECMWF parquets from GCS to HuggingFace")
    parser.add_argument("--year", type=str, default=None,
                        help="Single year to upload (default: 2024,2025,2026)")
    parser.add_argument("--month", type=str, default=None,
                        help="Single month MM to upload (requires --year)")
    parser.add_argument("--run", type=str, default=None,
                        help="Run hour filter: 00, 06, 12, 18 (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    args = parser.parse_args()

    fs = get_gcs_fs()
    api = HfApi()

    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        years = ["2024", "2025", "2026"]

    print("=" * 70)
    print("Upload GIK ECMWF Parquets: GCS → HuggingFace")
    print("=" * 70)
    print(f"GCS source: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print(f"HF target:  {HF_REPO}")
    print(f"Years: {years}")
    if args.month:
        print(f"Month: {args.month}")
    if args.run:
        print(f"Run: {args.run}z")
    if args.dry_run:
        print("Mode: DRY RUN")
    print("=" * 70)

    overall_t0 = time.time()
    grand_total = 0

    for year in years:
        print(f"\n{'='*70}")
        print(f"Year: {year}")
        print(f"{'='*70}")

        # Get months for this year
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
        for month in months:
            t0 = time.time()
            n = upload_month(fs, api, year, month, dry_run=args.dry_run,
                            run_filter=args.run)
            elapsed = time.time() - t0
            year_total += n
            if not args.dry_run and n > 0:
                print(f"  Month {month}: {n} files in {elapsed:.0f}s")

        print(f"  Year {year} total: {year_total} parquets")
        grand_total += year_total

    total_time = time.time() - overall_t0
    print(f"\n{'='*70}")
    print(f"COMPLETE: {grand_total} parquets in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
