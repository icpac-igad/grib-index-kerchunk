#!/usr/bin/env python3
"""
ECMWF Index Preprocessing - Stage 0
Creates reusable GCS parquet templates for fast index-based processing.

This is the ONE-TIME expensive preprocessing step that creates GCS templates
that can be reused for any date with the same ECMWF structure.

Usage:
    # Process single member
    python ecmwf_index_preprocessing.py --date 20240529 --member ens01 --bucket gik-ecmwf-aws-tf

    # Process all members
    python ecmwf_index_preprocessing.py --date 20240529 --bucket gik-ecmwf-aws-tf --all-members
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fsspec
import gcsfs
import pandas as pd
from kerchunk._grib_idx import build_idx_grib_mapping, parse_grib_idx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ECMWF configuration
ECMWF_BUCKET = "ecmwf-forecasts"
ECMWF_FORECAST_HOURS_3H = list(range(0, 145, 3))  # 0-144h at 3h intervals (49 steps)
ECMWF_FORECAST_HOURS_6H = list(range(150, 361, 6))  # 150-360h at 6h intervals (36 steps)
ECMWF_ALL_HOURS = ECMWF_FORECAST_HOURS_3H + ECMWF_FORECAST_HOURS_6H  # Total: 85 steps

# Ensemble members
ECMWF_MEMBERS = ["control"] + [f"ens{i:02d}" for i in range(1, 51)]


def process_single_ecmwf_hour(
    date_str: str,
    hour: int,
    member: str,
    bucket: str,
    gcs_fs: gcsfs.GCSFileSystem,
    storage_options: Dict
) -> Optional[str]:
    """
    Process a single forecast hour for ECMWF data to create GCS template.

    This is the expensive operation that runs scan_grib to build complete
    mapping of GRIB structure.
    """
    try:
        # Build ECMWF S3 URL
        # Format: s3://ecmwf-forecasts/20240529/00z/ifs/0p25/enfo/20240529000000-{hour}h-enfo-ef.grib2
        fname = (
            f"s3://{ECMWF_BUCKET}/{date_str}/00z/ifs/0p25/enfo/"
            f"{date_str}000000-{hour}h-enfo-ef.grib2"
        )

        logger.info(f"Processing {member} hour {hour:03d}: {fname}")
        start_time = time.time()

        # Step 1: Parse GRIB index file
        logger.info(f"  Parsing index for hour {hour:03d}...")
        idxdf = parse_grib_idx(basename=fname, storage_options=storage_options)

        # Filter for specific member if needed
        if member != "all":
            # ECMWF uses 'number' field: 0 for control, 1-50 for perturbed
            if member == "control":
                member_number = 0
            else:
                member_number = int(member.replace("ens", ""))

            # Filter index for specific member
            idxdf = idxdf[idxdf['attrs'].str.contains(f"number={member_number}")]

        # Step 2: Build complete idx-grib mapping (EXPENSIVE!)
        logger.info(f"  Building idx-grib mapping for hour {hour:03d} (this takes time)...")

        # Create mapper for the GRIB file
        fs = fsspec.filesystem("s3", anon=True)
        mapper = fs.get_mapper(fname)

        grib_mapping = build_idx_grib_mapping(
            basename=fname,
            mapper=mapper,
            storage_options=storage_options,
            validate=True
        )

        # Step 3: Deduplicate and clean
        deduped_mapping = grib_mapping.loc[~grib_mapping["attrs"].duplicated(keep="first"), :]

        # Step 4: Save to GCS
        gcs_path = f"gs://{bucket}/ecmwf/{member}/ecmwf-time-{date_str}-{member}-rt{hour:03d}.parquet"

        logger.info(f"  Saving to GCS: {gcs_path}")
        deduped_mapping.to_parquet(gcs_path, filesystem=gcs_fs)

        elapsed = time.time() - start_time
        logger.info(f"  ✅ Hour {hour:03d} complete in {elapsed:.1f}s")

        return gcs_path

    except Exception as e:
        logger.error(f"  ❌ Failed hour {hour:03d}: {e}")
        return None


def process_ecmwf_member(
    date_str: str,
    member: str,
    bucket: str,
    gcp_service_account: Optional[str] = None,
    hours_to_process: Optional[List[int]] = None
):
    """Process all forecast hours for a single ECMWF member."""

    logger.info("="*80)
    logger.info(f"Processing ECMWF member: {member}")
    logger.info(f"Reference date: {date_str}")
    logger.info(f"GCS bucket: {bucket}")
    logger.info("="*80)

    # Set up GCS filesystem
    if gcp_service_account:
        gcs_fs = gcsfs.GCSFileSystem(token=gcp_service_account)
    else:
        gcs_fs = gcsfs.GCSFileSystem()

    # Storage options for S3 access (anonymous)
    storage_options = {"anon": True}

    # Hours to process
    if hours_to_process is None:
        hours_to_process = ECMWF_ALL_HOURS

    logger.info(f"Will process {len(hours_to_process)} forecast hours")

    # Process each hour
    successful = 0
    failed = 0

    for hour in hours_to_process:
        result = process_single_ecmwf_hour(
            date_str=date_str,
            hour=hour,
            member=member,
            bucket=bucket,
            gcs_fs=gcs_fs,
            storage_options=storage_options
        )

        if result:
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("="*80)
    logger.info(f"Member {member} processing complete!")
    logger.info(f"  Successful: {successful}/{len(hours_to_process)}")
    logger.info(f"  Failed: {failed}/{len(hours_to_process)}")

    if successful == len(hours_to_process):
        logger.info(f"  ✅ All hours processed successfully!")
        logger.info(f"  Templates saved to: gs://{bucket}/ecmwf/{member}/")
    else:
        logger.warning(f"  ⚠️ Some hours failed. Re-run to retry failed hours.")


def main():
    parser = argparse.ArgumentParser(
        description="ECMWF Index Preprocessing - Create GCS templates for fast processing"
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Reference date (YYYYMMDD) to create templates from"
    )
    parser.add_argument(
        "--member",
        default="ens01",
        help="Member to process (control, ens01-ens50, or 'all')"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name for storing templates"
    )
    parser.add_argument(
        "--gcp-service-account",
        help="Path to GCP service account JSON file"
    )
    parser.add_argument(
        "--hours",
        help="Comma-separated list of hours to process (default: all 85 hours)"
    )
    parser.add_argument(
        "--all-members",
        action="store_true",
        help="Process all ensemble members"
    )

    args = parser.parse_args()

    # Parse hours if specified
    hours_to_process = None
    if args.hours:
        hours_to_process = [int(h) for h in args.hours.split(",")]

    # Process members
    if args.all_members:
        logger.info(f"Processing ALL {len(ECMWF_MEMBERS)} ensemble members")
        for member in ECMWF_MEMBERS:
            process_ecmwf_member(
                date_str=args.date,
                member=member,
                bucket=args.bucket,
                gcp_service_account=args.gcp_service_account,
                hours_to_process=hours_to_process
            )
    else:
        process_ecmwf_member(
            date_str=args.date,
            member=args.member,
            bucket=args.bucket,
            gcp_service_account=args.gcp_service_account,
            hours_to_process=hours_to_process
        )


if __name__ == "__main__":
    main()