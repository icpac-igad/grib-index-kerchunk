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
#     "xarray==2024.10.0",
#     "zarr>=2.18,<3",
#     "numcodecs<0.13",
#     "h5py",
#     "dask",
#     "distributed",
#     "google-cloud-storage",
# ]
# ///
"""
GEFS v10 (pre-2020-09-25) pgrb2b 0.5° Index Pre-processing Module + Driver

NOAA's GEFS reforecast/realtime archive on s3://noaa-gefs-pds/ switched
layout on 2020-09-25 (GEFS v12 upgrade). The current production routine
(gefs_index_preprocessing_fixed.py + run_gefs_preprocessing.py) targets
the post-2020-09-25 v12 layout: gefs.{date}/{run}/atmos/pgrb2sp25/ at
0.25° with 30 perturbed members. That routine does not work for earlier
dates because the v10 archive differs in three ways:

  - path:     gefs.{date}/{run}/pgrb2b/...           (no `atmos/` prefix)
  - product:  pgrb2b at 0.5° (no 0.25° `pgrb2sp25` product exists)
  - members:  gec00 + gep01..gep20 (21 total; v12 has gec00 + gep01..gep30)
  - filename: {member}.t{run}z.pgrb2bf{HH[H]}        (no `.0p25.f` suffix)
  - cadence:  6h step, f000 → f384 (65 timesteps per member)

This module is the v10 sibling of gefs_index_preprocessing_fixed.py: it
creates idx→grib mapping parquets for the pgrb2b 0.5° product and uploads
them to GCS under a separate prefix (`time_idx/gefs_v10_pgrb2b/...`) so
v10 templates do not collide with the v12 archive at `time_idx/gefs/...`.

The processing driver (formerly run_gefs_preprocessing.py) is merged into
this same file — invoke directly with --all-members instead of via a
subprocess loop.

Usage (uv resolves the PEP 723 deps block on first run):
    # Single member
    uv run gefs_pgrb2b05_preprocessing.py --date 20190101 --run 00 --member gep01

    # All v10 members for one date/run
    uv run gefs_pgrb2b05_preprocessing.py --date 20190101 --run 00 --all-members
"""

import logging
import os
import pathlib
import tempfile
import time
from typing import List

import dask
import fsspec
import gcsfs
from distributed import get_worker
from google.cloud import storage

from kerchunk._grib_idx import build_idx_grib_mapping

logger = logging.getLogger("gefs-v10-pgrb2b-preprocessing")

V12_CUTOFF_DATE = "20200925"
V10_MEMBERS: List[str] = ["gec00"] + [f"gep{i:02d}" for i in range(1, 21)]
V10_MAX_FORECAST_HOUR = 384


def setup_gefs_logging(log_level: int = logging.INFO,
                       log_file: str = "gefs_v10_pgrb2b_preprocessing.log"):
    root = logging.getLogger()
    root.setLevel(log_level)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(sh)


def _validate_v10_date(date_str: str) -> None:
    if date_str >= V12_CUTOFF_DATE:
        raise ValueError(
            f"Date {date_str} is on or after the GEFS v12 cutoff "
            f"({V12_CUTOFF_DATE}). Use gefs_index_preprocessing_fixed.py "
            f"for v12 (0.25° pgrb2sp25) dates."
        )


def get_gefs_v10_details(url: str):
    """Extract (date, run, member, forecast_hour) from a v10 pgrb2b URL."""
    import re
    pattern = (
        r"s3://noaa-gefs-pds/gefs\.(\d{8})/(\d{2})/pgrb2b/"
        r"(gec00|gep\d{2})\.t(\d{2})z\.pgrb2bf(\d{2,3})$"
    )
    match = re.match(pattern, url)
    if not match:
        logger.warning(f"No match for v10 pgrb2b URL pattern: {url}")
        return None, None, None, None
    return match.group(1), match.group(2), match.group(3), match.group(5)


def gefs_v10_s3_url_maker(date_str: str, run: str = "00",
                          ensemble_member: str = "gep01",
                          max_forecast_hour: int = V10_MAX_FORECAST_HOUR) -> List[str]:
    """List all v10 pgrb2b GRIB URLs for one (date, run, member).

    Globs S3 directly so we pick up whatever forecast hours NOAA actually
    published for that date — robust against archive gaps. Excludes the
    `pgrb2banl` analysis file and any `.idx` companions.
    """
    _validate_v10_date(date_str)
    fs_s3 = fsspec.filesystem("s3", anon=True)
    glob_pattern = (
        f"noaa-gefs-pds/gefs.{date_str}/{run}/pgrb2b/"
        f"{ensemble_member}.t{run}z.pgrb2bf*"
    )
    matches = fs_s3.glob(glob_pattern)
    grib_only = [m for m in matches if not m.endswith(".idx")]

    urls = []
    for m in grib_only:
        full_url = "s3://" + m
        _, _, _, fhh = get_gefs_v10_details(full_url)
        if fhh is None:
            continue
        if int(fhh) <= max_forecast_hour:
            urls.append(full_url)

    urls = sorted(urls)
    print(f"Generated {len(urls)} v10 pgrb2b URLs for "
          f"{date_str} {run}z {ensemble_member}")
    return urls


def _get_worker_creds_path(dask_worker) -> str:
    return str(pathlib.Path(dask_worker.local_directory) / "coiled-data-key.json")


def upload_to_gcs(bucket_name: str, source_file_name: str,
                  destination_blob_name: str) -> None:
    """Upload via Dask worker credentials. Raises ValueError if no worker."""
    worker = get_worker()
    creds_path = _get_worker_creds_path(worker)
    print(f"Using GCS creds at: {creds_path}")
    client = storage.Client.from_service_account_json(creds_path)
    blob = client.bucket(bucket_name).blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} → gs://{bucket_name}/{destination_blob_name}")


def noncluster_upload_to_gcs(bucket_name: str, source_file_name: str,
                             destination_blob_name: str,
                             credentials_path: str) -> None:
    """Upload without a Dask worker (local testing)."""
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
    print(f"Using GCS creds at: {credentials_path}")
    client = storage.Client.from_service_account_json(credentials_path)
    blob = client.bucket(bucket_name).blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} → gs://{bucket_name}/{destination_blob_name}")


@dask.delayed
def process_v10_time_idx_data(s3url: str, bucket_name: str) -> bool:
    """Build idx→grib mapping for one v10 pgrb2b file and upload to GCS."""
    try:
        date_str, runz, member, fhh = get_gefs_v10_details(s3url)
        if not all([date_str, runz, member, fhh]):
            logger.error(f"Invalid v10 URL: {s3url}")
            return False

        print(f"Processing v10 pgrb2b: {date_str} {runz}z {member} f{fhh}")
        mapping = build_idx_grib_mapping(s3url, storage_options={"anon": True})
        deduped = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
        deduped.set_index("attrs", inplace=True)

        out_dir = f"gefs_v10_pgrb2b_mapping_{date_str}_{member}"
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(
            out_dir,
            f"gefs-v10-pgrb2b-time-{date_str}-{member}-rt{int(fhh):03}.parquet",
        )
        deduped.to_parquet(parquet_path, index=True)

        year = date_str[:4]
        dest = (
            f"time_idx/gefs_v10_pgrb2b/{year}/{date_str}/{member}/"
            f"{os.path.basename(parquet_path)}"
        )
        try:
            upload_to_gcs(bucket_name, parquet_path, dest)
        except ValueError:
            noncluster_upload_to_gcs(bucket_name, parquet_path, dest,
                                     "/tmp/coiled-data.json")

        os.remove(parquet_path)
        print(f"Done v10 pgrb2b {date_str} {member} f{fhh}")
        return True
    except Exception as e:
        logger.error(f"Failed v10 pgrb2b for {s3url}: {e}")
        raise


def logged_process_v10_time_idx_data(s3url: str, bucket_name: str) -> bool:
    """Same as process_v10_time_idx_data but with per-message log upload."""
    date_str, runz, member, fhh = get_gefs_v10_details(s3url)

    suffix = f"_gefs_v10_{date_str}_{member}_{fhh}.log"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=suffix, delete=False) as lf:
        log_filename = lf.name
    member_logger = logging.getLogger(f"gefs_v10_{date_str}_{member}")
    success = False

    try:
        member_logger.info(f"Processing v10 pgrb2b: {s3url}")
        if not all([date_str, runz, member, fhh]):
            member_logger.error(f"Invalid v10 URL: {s3url}")
            return False

        mapping = build_idx_grib_mapping(s3url, storage_options={"anon": True})
        deduped = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
        deduped.set_index("attrs", inplace=True)

        out_dir = f"gefs_v10_pgrb2b_mapping_{date_str}_{member}"
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(
            out_dir,
            f"gefs-v10-pgrb2b-time-{date_str}-{member}-rt{int(fhh):03}.parquet",
        )
        deduped.to_parquet(parquet_path, index=True)

        year = date_str[:4]
        dest = (
            f"time_idx/gefs_v10_pgrb2b/{year}/{date_str}/{member}/"
            f"{os.path.basename(parquet_path)}"
        )
        try:
            upload_to_gcs(bucket_name, parquet_path, dest)
        except ValueError:
            noncluster_upload_to_gcs(bucket_name, parquet_path, dest,
                                     "/tmp/coiled-data.json")
        os.remove(parquet_path)
        member_logger.info(f"Done v10 pgrb2b {date_str} {member} f{fhh}")
        success = True
    except Exception as e:
        member_logger.error(f"Failed v10 pgrb2b for {s3url}: {e}")
        success = False
    finally:
        year = date_str[:4] if date_str else "unknown"
        gcs_log = (
            f"time_idx/gefs_v10_pgrb2b/{year}/logs/{date_str}/{member}/"
            f"{os.path.basename(log_filename)}"
        )
        try:
            upload_to_gcs(bucket_name, log_filename, gcs_log)
        except Exception:
            pass
        os.remove(log_filename)
    return success


def create_v10_member_mappings(date_str: str, run: str, ensemble_member: str,
                               bucket_name: str,
                               max_forecast_hour: int = V10_MAX_FORECAST_HOUR,
                               use_dask: bool = True):
    """Build idx mappings for every forecast hour of one v10 member."""
    _validate_v10_date(date_str)
    print(f"Creating v10 pgrb2b mappings for {date_str} {run}z {ensemble_member}")
    urls = gefs_v10_s3_url_maker(date_str, run, ensemble_member, max_forecast_hour)

    if use_dask:
        delayed = [process_v10_time_idx_data(u, bucket_name) for u in urls]
        return dask.compute(*delayed)

    results = []
    for u in urls:
        try:
            results.append(logged_process_v10_time_idx_data(u, bucket_name))
        except Exception as e:
            print(f"Error on {u}: {e}")
            results.append(False)
    return results


def create_v10_full_ensemble_mappings(date_str: str, run: str, bucket_name: str,
                                      ensemble_members: List[str] = None,
                                      max_forecast_hour: int = V10_MAX_FORECAST_HOUR):
    """Build idx mappings for all v10 members for one date/run."""
    _validate_v10_date(date_str)
    if ensemble_members is None:
        ensemble_members = list(V10_MEMBERS)
    print(f"v10 pgrb2b mappings for {date_str} {run}z, "
          f"{len(ensemble_members)} members")

    all_results = {}
    start_total = time.time()
    for i, member in enumerate(ensemble_members, 1):
        print(f"\n{'='*72}")
        print(f"Member {member} ({i}/{len(ensemble_members)})")
        print(f"{'='*72}")
        start = time.time()
        try:
            all_results[member] = create_v10_member_mappings(
                date_str, run, member, bucket_name, max_forecast_hour, use_dask=True
            )
            elapsed = time.time() - start
            print(f"✅ {member} done in {elapsed:.1f}s")
        except Exception as e:
            print(f"❌ {member} failed: {e}")
            all_results[member] = None
        avg = (time.time() - start_total) / i
        remaining = (len(ensemble_members) - i) * avg
        print(f"Progress {i}/{len(ensemble_members)} — "
              f"avg {avg:.1f}s/member, ETA {remaining/60:.1f} min")
    return all_results


def verify_v10_mappings_in_gcs(bucket_name: str, date_str: str,
                               ensemble_member: str, credentials_path: str):
    gcs_fs = gcsfs.GCSFileSystem(token=credentials_path)
    year = date_str[:4]
    pattern = (
        f"gs://{bucket_name}/time_idx/gefs_v10_pgrb2b/{year}/{date_str}/"
        f"{ensemble_member}/gefs-v10-pgrb2b-time-{date_str}-"
        f"{ensemble_member}-rt*.parquet"
    )
    try:
        found = gcs_fs.glob(pattern)
        print(f"Found {len(found)} v10 pgrb2b mapping files for "
              f"{ensemble_member} on {date_str}")
        return found
    except Exception as e:
        print(f"Error checking v10 mappings: {e}")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GEFS v10 pgrb2b 0.5° index preprocessing "
                    "(dates before 2020-09-25)"
    )
    parser.add_argument("--date", required=True,
                        help="Date in YYYYMMDD format (must be < 20200925)")
    parser.add_argument("--run", default="00",
                        help="Run hour (00, 06, 12, 18) — default 00")
    parser.add_argument("--bucket", default="gik-gefs-aws-tf",
                        help="GCS bucket for mapping parquets")
    parser.add_argument("--member",
                        help="Process a single member (gec00 or gep01..gep20)")
    parser.add_argument("--all-members", action="store_true",
                        help="Process all 21 v10 members "
                             "(gec00 + gep01..gep20)")
    parser.add_argument("--max-forecast-hour", type=int,
                        default=V10_MAX_FORECAST_HOUR,
                        help=f"Cap on forecast hour — default {V10_MAX_FORECAST_HOUR}")
    args = parser.parse_args()

    if not args.member and not args.all_members:
        parser.error("Specify either --member or --all-members")
    if args.member and args.all_members:
        parser.error("Specify --member OR --all-members, not both")
    if args.run not in ("00", "06", "12", "18"):
        parser.error("--run must be one of: 00, 06, 12, 18")

    _validate_v10_date(args.date)
    setup_gefs_logging()

    print(f"GEFS v10 pgrb2b preprocessing — date {args.date} run {args.run}z "
          f"bucket {args.bucket}")
    start = time.time()
    if args.member:
        results = create_v10_member_mappings(
            args.date, args.run, args.member, args.bucket,
            max_forecast_hour=args.max_forecast_hour,
        )
    else:
        results = create_v10_full_ensemble_mappings(
            args.date, args.run, args.bucket,
            max_forecast_hour=args.max_forecast_hour,
        )
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Results: {results}")
