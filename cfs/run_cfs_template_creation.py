#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas<2.2",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "kerchunk==0.2.7",
#     "h5py",
#     "zarr>=2.18,<3",
#     "xarray==2024.10.0",
#     "numcodecs<0.13",
# ]
# ///
"""
CFS Index Preprocessing — One-time Template Creation
=====================================================

Creates reusable parquet templates for fast index-based processing.
This is the CFS equivalent of ecmwf/ecmwf_index_preprocessing.py.

Two modes:
  1. Mapping parquets (default): per-timestep build_idx_grib_mapping output
  2. Zarr template (--zarr-template): zarr skeleton from scan_grib + grib_tree

CFS ensemble = multi-date ensemble: 31 init_dates x 4 runs = up to 124 members
per target month. Init dates come from the PRECEDING month.

Usage:
    # Create mapping parquets locally (test with 48h = 9 timesteps)
    uv run run_cfs_template_creation.py --init-date 20251101 --run 00 \
        --local-dir ./cfs_templates --max-forecast-hours 48

    # Create mapping parquets to GCS
    uv run run_cfs_template_creation.py --init-date 20251101 --run 00 \
        --bucket gik-fmrc --max-forecast-hours 48

    # Create zarr skeleton template (for use by run_lithops_cfs.py)
    uv run run_cfs_template_creation.py --zarr-template --init-date 20251101 \
        --run 00 --local-dir ./cfs_templates --max-forecast-hours 48

    # Full month (124 members)
    uv run run_cfs_template_creation.py --init-month 202511 \
        --local-dir ./cfs_templates --max-forecast-hours 5160
"""

import argparse
import copy
import json
import logging
import os
import time
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fsspec
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Anonymous S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# CFS configuration
S3_BUCKET = "noaa-cfs-pds"
CFS_MEMBER = "01"  # 6hrly_grib_01
DEFAULT_MAX_FORECAST_HOURS = 5160  # ~215 days


def _patch_kerchunk_for_cfs():
    """
    CFS .idx files have non-integer idx values (e.g. "36.1") which causes
    kerchunk's parse_grib_idx line `result["idx"].astype(int)` to fail.
    Monkey-patch pd.Series.astype to handle this gracefully.
    """
    import kerchunk._grib_idx as _grib_idx

    if getattr(_grib_idx, '_cfs_patched', False):
        return

    _orig_astype = pd.Series.astype

    def _safe_astype(self, dtype, **kwargs):
        if dtype == int or dtype is int:
            try:
                return _orig_astype(self, dtype, **kwargs)
            except (ValueError, TypeError):
                return pd.to_numeric(self, errors="coerce").fillna(0).astype(int)
        return _orig_astype(self, dtype, **kwargs)

    pd.Series.astype = _safe_astype
    _grib_idx._cfs_patched = True


def cfs_grib_url(date_str: str, run: str, forecast_hour: int,
                 member: str = CFS_MEMBER) -> str:
    """Build S3 URL for a single CFS flux GRIB file."""
    init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
    forecast_dt = init_dt + timedelta(hours=forecast_hour)
    init_time = init_dt.strftime("%Y%m%d%H")
    forecast_time = forecast_dt.strftime("%Y%m%d%H")
    return (
        f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{member}/"
        f"flxf{forecast_time}.{member}.{init_time}.grb2"
    )


def cfs_forecast_hours(max_hours: int = DEFAULT_MAX_FORECAST_HOURS) -> List[int]:
    """Generate list of 6-hourly forecast hours."""
    return list(range(0, max_hours + 1, 6))


def generate_init_dates(init_month: str) -> List[Tuple[str, str]]:
    """
    Generate all (init_date, run) pairs for a given month.

    Returns:
        List of (date_str, run) tuples -- up to 31 days x 4 runs = 124 members
    """
    year = int(init_month[:4])
    month = int(init_month[4:6])
    _, last_day = monthrange(year, month)

    members = []
    for day in range(1, last_day + 1):
        date_str = f"{year}{month:02d}{day:02d}"
        for run in ["00", "06", "12", "18"]:
            members.append((date_str, run))

    logger.info(f"Generated {len(members)} init_date+run pairs for {init_month}")
    return members


# ==============================================================================
# MODE 1: MAPPING PARQUETS (like ecmwf_index_preprocessing.py)
# ==============================================================================

def process_single_cfs_hour(
    date_str: str,
    run: str,
    forecast_hour: int,
    local_dir: Optional[str],
    bucket: Optional[str],
    gcs_fs,
    storage_options: Dict
) -> Optional[str]:
    """
    Process a single CFS forecast hour: build_idx_grib_mapping -> save.

    Saves to local_dir if specified, otherwise to GCS bucket.
    """
    try:
        fname = cfs_grib_url(date_str, run, forecast_hour)
        logger.info(f"Processing {date_str} run {run} f{forecast_hour:04d}: {fname}")
        start_time = time.time()

        from kerchunk._grib_idx import build_idx_grib_mapping

        _patch_kerchunk_for_cfs()

        # validate=False because CFS has ~2 messages that kerchunk can't decode
        grib_mapping = build_idx_grib_mapping(
            basename=fname,
            storage_options=storage_options,
            validate=False
        )

        # Deduplicate on attrs
        deduped_mapping = grib_mapping.loc[
            ~grib_mapping["attrs"].duplicated(keep="first"), :
        ]

        parquet_filename = f"cfs-time-{date_str}-{run}-rt{forecast_hour:04d}.parquet"

        if local_dir:
            # Save locally
            output_path = Path(local_dir) / "mapping" / date_str / run
            output_path.mkdir(parents=True, exist_ok=True)
            parquet_path = output_path / parquet_filename
            deduped_mapping.to_parquet(parquet_path)
            logger.info(f"  Saved locally: {parquet_path}")
            result_path = str(parquet_path)
        else:
            # Save to GCS
            year = date_str[:4]
            gcs_path = (
                f"gs://{bucket}/cfs/time_idx/{year}/{date_str}/{run}/"
                f"{parquet_filename}"
            )
            logger.info(f"  Saving to GCS: {gcs_path}")
            deduped_mapping.to_parquet(gcs_path, filesystem=gcs_fs)
            result_path = gcs_path

        elapsed = time.time() - start_time
        logger.info(f"  Hour {forecast_hour:04d} complete in {elapsed:.1f}s")
        return result_path

    except Exception as e:
        logger.error(f"  Failed hour {forecast_hour:04d}: {e}")
        return None


def process_cfs_member(
    date_str: str,
    run: str,
    local_dir: Optional[str] = None,
    bucket: Optional[str] = None,
    gcp_service_account: Optional[str] = None,
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS
):
    """Process all forecast hours for a single CFS init_date+run (one member)."""

    logger.info("=" * 80)
    logger.info(f"Processing CFS member: {date_str} run {run}")
    if local_dir:
        logger.info(f"Output: {local_dir}")
    else:
        logger.info(f"GCS bucket: {bucket}")
    logger.info("=" * 80)

    # Set up GCS filesystem (only needed for GCS upload)
    gcs_fs = None
    if not local_dir:
        import gcsfs
        if gcp_service_account:
            gcs_fs = gcsfs.GCSFileSystem(token=gcp_service_account)
        else:
            gcs_fs = gcsfs.GCSFileSystem()

    storage_options = {"anon": True}
    hours = cfs_forecast_hours(max_forecast_hours)

    logger.info(f"Will process {len(hours)} forecast hours (6-hourly, 0-{max_forecast_hours}h)")

    successful = 0
    failed = 0

    for hour in hours:
        result = process_single_cfs_hour(
            date_str=date_str,
            run=run,
            forecast_hour=hour,
            local_dir=local_dir,
            bucket=bucket,
            gcs_fs=gcs_fs,
            storage_options=storage_options
        )
        if result:
            successful += 1
        else:
            failed += 1

    logger.info("=" * 80)
    logger.info(f"Member {date_str} {run}z processing complete!")
    logger.info(f"  Successful: {successful}/{len(hours)}")
    logger.info(f"  Failed: {failed}/{len(hours)}")

    if successful == len(hours):
        logger.info(f"  All hours processed successfully!")
    elif failed > 0:
        logger.warning(f"  Some hours failed. Re-run to retry failed hours.")


# ==============================================================================
# MODE 2: ZARR SKELETON TEMPLATE (for run_lithops_cfs.py)
# ==============================================================================

def create_zarr_template(
    date_str: str,
    run: str,
    local_dir: str,
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
) -> Optional[str]:
    """
    Create zarr skeleton template using scan_grib + grib_tree.

    Scans the first 2 GRIB files (f000 + f006) to capture the full
    variable/dimension schema, then strips data references to create
    a deflated store. Saved as a parquet with columns ['key', 'value'].

    This template can be loaded by run_lithops_cfs.py's
    build_deflated_store_from_template().
    """
    from kerchunk.grib2 import scan_grib, grib_tree

    logger.info("=" * 80)
    logger.info(f"Creating zarr skeleton template for {date_str} run {run}")
    logger.info("=" * 80)

    start_time = time.time()
    storage_options = {"anon": True}

    # Scan first 2 GRIB files to get full variable schema
    grib_files = [
        cfs_grib_url(date_str, run, 0),
        cfs_grib_url(date_str, run, 6),
    ]

    all_groups = []
    for gurl in grib_files:
        logger.info(f"  Scanning: {gurl}")
        groups = scan_grib(gurl, storage_options=storage_options)
        all_groups.extend(groups)
        logger.info(f"  Found {len(groups)} messages")

    # Build grib_tree → zarr store
    logger.info(f"  Building grib_tree from {len(all_groups)} messages...")
    tree_store = grib_tree(all_groups, remote_options={"anon": True})
    logger.info(f"  Tree store: {len(tree_store.get('refs', {}))} keys")

    # Strip data references → deflated store (metadata only)
    deflated_store = copy.deepcopy(tree_store)
    refs = deflated_store.get('refs', {})

    # Remove data chunk references (keep .zattrs, .zarray, .zgroup)
    keys_to_remove = [
        k for k in refs
        if not k.startswith('.') and '/' in k
        and not any(k.endswith(ext) for ext in ['.zattrs', '.zarray', '.zgroup'])
    ]
    for k in keys_to_remove:
        del refs[k]

    logger.info(f"  Deflated store: {len(refs)} metadata keys "
                f"(removed {len(keys_to_remove)} data refs)")

    # Save as parquet (key-value format matching Lithops script expectations)
    output_dir = Path(local_dir) / "template"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"cfs-zarr-template-{date_str}-{run}.parquet"

    data = []
    for key, value in refs.items():
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value).encode('utf-8')
        elif isinstance(value, str):
            encoded = value.encode('utf-8')
        else:
            encoded = str(value).encode('utf-8')
        data.append((key, encoded))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(parquet_path)

    elapsed = time.time() - start_time
    logger.info(f"  Zarr template saved: {parquet_path} ({len(df)} keys, {elapsed:.1f}s)")

    return str(parquet_path)


# ==============================================================================
# CLI + MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CFS Index Preprocessing - Create templates for fast processing"
    )
    parser.add_argument(
        "--init-date",
        help="Single init date (YYYYMMDD)"
    )
    parser.add_argument(
        "--init-month",
        help="Full month of init dates (YYYYMM) — processes all days x 4 runs"
    )
    parser.add_argument(
        "--run",
        default="00",
        choices=["00", "06", "12", "18"],
        help="Run hour (default: 00, used with --init-date)"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all 4 runs (00, 06, 12, 18) for --init-date"
    )

    # Output destination (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--local-dir",
        help="Save output to local directory"
    )
    output_group.add_argument(
        "--bucket",
        help="GCS bucket name for storing templates"
    )

    parser.add_argument(
        "--gcp-service-account",
        help="Path to GCP service account JSON file"
    )
    parser.add_argument(
        "--max-forecast-hours",
        type=int,
        default=DEFAULT_MAX_FORECAST_HOURS,
        help=f"Max forecast hours (default: {DEFAULT_MAX_FORECAST_HOURS})"
    )

    # Zarr template mode
    parser.add_argument(
        "--zarr-template",
        action="store_true",
        help="Create zarr skeleton template (scan_grib + grib_tree). "
             "Use with --local-dir."
    )

    args = parser.parse_args()

    if not args.init_date and not args.init_month:
        parser.error("Either --init-date or --init-month is required")

    if args.zarr_template and not args.local_dir:
        parser.error("--zarr-template requires --local-dir")

    # Zarr template mode
    if args.zarr_template:
        date_str = args.init_date
        if not date_str:
            parser.error("--zarr-template requires --init-date")
        result = create_zarr_template(
            date_str, args.run, args.local_dir, args.max_forecast_hours
        )
        if result:
            print(f"\nZarr template created: {result}")
            print(f"\nUse with run_lithops_cfs.py:")
            print(f"  uv run run_lithops_cfs.py --init-date {date_str} --run {args.run} "
                  f"--sequential --local-template {result} "
                  f"--output-dir ./cfs_output --max-forecast-hours {args.max_forecast_hours}")
        return

    # Mapping parquets mode
    if args.init_month:
        members = generate_init_dates(args.init_month)
        logger.info(f"Processing ALL {len(members)} members for month {args.init_month}")
        for i, (date_str, run) in enumerate(members, 1):
            logger.info(f"\n[{i}/{len(members)}] Member: {date_str} run {run}")
            process_cfs_member(
                date_str=date_str,
                run=run,
                local_dir=args.local_dir,
                bucket=args.bucket,
                gcp_service_account=args.gcp_service_account,
                max_forecast_hours=args.max_forecast_hours
            )
    elif args.all_runs:
        for run in ["00", "06", "12", "18"]:
            process_cfs_member(
                date_str=args.init_date,
                run=run,
                local_dir=args.local_dir,
                bucket=args.bucket,
                gcp_service_account=args.gcp_service_account,
                max_forecast_hours=args.max_forecast_hours
            )
    else:
        process_cfs_member(
            date_str=args.init_date,
            run=args.run,
            local_dir=args.local_dir,
            bucket=args.bucket,
            gcp_service_account=args.gcp_service_account,
            max_forecast_hours=args.max_forecast_hours
        )


if __name__ == "__main__":
    main()
