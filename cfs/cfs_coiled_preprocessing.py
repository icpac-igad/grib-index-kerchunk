#!/usr/bin/env python3
"""
CFS Coiled preprocessing driver — adapted from
`ecmwf/dev-test/ecmwf_50r1_coiled_preprocessing.py`.

This is Step 1 ("preprocessing") of the CFS template build: in parallel on a
Coiled cluster, run `build_idx_grib_mapping` on EVERY (member, forecast-hour)
CFS flux GRIB of a reference date and store a per-(member, hour) idx->grib
mapping parquet to GCS:
    gs://{bucket}/cfs/time_idx/{year}/{date}/{run}/{member}/
        cfs-time-{date}-{run}-rt{hhhh}.parquet

Why Coiled (see cfs/GIK_TEMPLATE_BUILD_CORPUS_SCAN.md): the GIK template is the
corpus of per-file idx->grib mappings, NOT a 2-file scan. CFS is the GEFS-style
model — `build_idx_grib_mapping` decodes each file via `scan_grib`, so building
the template means thousands of scans (861 6-hourly hours/member × N members for
a 215-day run). That is a distributed job, exactly like the ECMWF per-(stream,
hour) Coiled scan. This driver is the parallel replacement for the sequential
`process_cfs_member` loop in `run_cfs_template_creation.py`.

Difference from the ECMWF driver:
  - scan op is CFS `build_idx_grib_mapping` (idx->grib mapping), not the raw
    `dynamic_zarr_store._map_grib_file_by_group` per-message dump. The mapping
    parquet (deduped on `attrs`) is what the CFS Stage-2 merge consumes.
  - applies the CFS `.idx` patch (fractional record ids like `36.1`/`36.2`) and
    `validate=False` (a few CFS messages don't decode), matching
    `run_cfs_template_creation.py`.

  CFS task grid:  members 01[,02,03,04] × forecast hours 0..MAX step 6.
  One reference (date, run); the mapping is date-independent and reused for all
  forecast dates (same as the GEFS/ECMWF templates).

Must be launched by a user with Coiled auth in `cfs/` (`coiled login` done;
`coiled-data.json` GCS key present). It cannot be launched by the assistant.
`--dry-run` prints the plan with no Coiled / no cost.

Coiled config follows the ECMWF driver: software=gik-coiled-pinned (fallback
gik-coiled-v6), vm_type=n2-standard-2, region=us-east1, arm=False,
idle_timeout=30m, cluster.adapt(min=1,max=N), workspace=gcp-sewaa-nka. All
overridable via flags.

Usage:
    python cfs_coiled_preprocessing.py --date 20251101 --run 00 --dry-run
    python cfs_coiled_preprocessing.py --date 20251101 --run 00                 # paid
    python cfs_coiled_preprocessing.py --date 20251101 --run 00 \
        --members 01,02,03,04 --max-forecast-hours 2160 --max-workers 20
"""
import os
import re
import argparse
import sys
from datetime import datetime, timedelta

# CFS flux files are 6-hourly. Default 5160h ~= 215 days -> 861 files per member.
FORECAST_INTERVAL = 6
DEFAULT_MAX_FORECAST_HOURS = 5160

S3_BUCKET = "noaa-cfs-pds"
GCS_BUCKET = "gik-cfs-aws-tf"


def cfs_grib_url(date_str, run, forecast_hour, member):
    """Build the S3 URL for one CFS flux GRIB (matches run_cfs_template_creation.py)."""
    init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
    forecast_dt = init_dt + timedelta(hours=forecast_hour)
    init_time = init_dt.strftime("%Y%m%d%H")
    forecast_time = forecast_dt.strftime("%Y%m%d%H")
    return (
        f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{member}/"
        f"flxf{forecast_time}.{member}.{init_time}.grb2"
    )


def build_task_plan(date, run, members, max_forecast_hours):
    """Build the (member × forecast-hour) task plan for one reference date/run."""
    hours = list(range(0, max_forecast_hours + 1, FORECAST_INTERVAL))
    tasks = []
    for member in members:
        for hour in hours:
            tasks.append({
                "member": member,
                "hour": hour,
                "url": cfs_grib_url(date, run, hour, member),
            })
    return tasks


def print_plan(tasks, date, run, members):
    n_hours = len({t["hour"] for t in tasks})
    print(f"CFS preprocessing plan: date={date} run={run}z  {len(tasks)} tasks "
          f"({len(members)} member(s) {','.join(members)} × {n_hours} hours)")
    print(f"GCS dest: gs://{GCS_BUCKET}/cfs/time_idx/{date[:4]}/{date}/{run}/"
          f"<member>/cfs-time-{date}-{run}-rt<hhhh>.parquet")
    for t in tasks[:2]:
        print(f"  [m{t['member']}]  {t['hour']:04d}h  {t['url']}")
    if len(tasks) > 2:
        print(f"  ... ({len(tasks) - 2} more)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="reference init date YYYYMMDD")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--members", default="01",
                    help="comma-separated 6hrly_grib members, or 'all' (01,02,03,04). "
                         "Default 01 (matches the current CFS pipeline).")
    ap.add_argument("--max-forecast-hours", type=int, default=DEFAULT_MAX_FORECAST_HOURS,
                    help=f"max forecast hour, 6-hourly (default {DEFAULT_MAX_FORECAST_HOURS})")
    ap.add_argument("--max-workers", type=int, default=20)
    # Coiled config — defaults follow the ECMWF driver.
    ap.add_argument("--software", default="gik-coiled-pinned",
                    help="Coiled software env. Default gik-coiled-pinned (reproducible "
                         "build); fallback with identical pins: --software gik-coiled-v6.")
    ap.add_argument("--workspace", default="gcp-sewaa-nka",
                    help="Coiled workspace/account")
    ap.add_argument("--vm-type", default="n2-standard-2")
    ap.add_argument("--region", default="us-east1")
    ap.add_argument("--gcs-key", default="coiled-data.json",
                    help="GCS service-account key shipped to workers (default coiled-data.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the task plan; no Coiled, no cost")
    args = ap.parse_args()

    if not re.match(r"^\d{8}$", args.date):
        sys.exit(f"--date must be YYYYMMDD; got {args.date!r}")

    if args.members.lower() == "all":
        members = ["01", "02", "03", "04"]
    else:
        members = [m.strip() for m in args.members.split(",") if m.strip()]
        bad = [m for m in members if not re.match(r"^0[1-4]$", m)]
        if bad:
            sys.exit(f"--members entries must be 01..04; got {bad}")

    tasks = build_task_plan(args.date, args.run, members, args.max_forecast_hours)
    print_plan(tasks, args.date, args.run, members)

    if args.dry_run:
        return 0

    if not os.path.exists(args.gcs_key):
        sys.exit(f"GCS key not found: {args.gcs_key} (run from cfs/ with coiled-data.json present)")

    import coiled
    from distributed import Client

    date = args.date
    run = args.run
    gcs_key_name = os.path.basename(args.gcs_key)

    @coiled.function(
        vm_type=args.vm_type,
        software=args.software,
        workspace=args.workspace,
        name="func-cfs-preproc",
        region=args.region,
        arm=False,
        idle_timeout="30 minutes",
    )
    def scan_one(task):
        # The gik-coiled image bakes stale AWS creds; force ANONYMOUS S3 for the
        # public CFS bucket (same fix as the ECMWF driver).
        import os
        import fsspec
        import s3fs
        import pandas as pd
        fsspec.config.conf["s3"] = {"anon": True}
        s3fs.S3FileSystem.clear_instance_cache()
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/nonexistent"

        # CFS .idx has fractional record ids (e.g. 36.1/36.2) -> kerchunk's
        # parse_grib_idx `.astype(int)` raises. Patch pd.Series.astype to coerce.
        _orig_astype = pd.Series.astype

        def _safe_astype(self, dtype, **kwargs):
            if dtype is int or dtype == int:
                try:
                    return _orig_astype(self, dtype, **kwargs)
                except (ValueError, TypeError):
                    return pd.to_numeric(self, errors="coerce").fillna(0).astype(int)
            return _orig_astype(self, dtype, **kwargs)
        pd.Series.astype = _safe_astype

        # build_idx_grib_mapping lives in kerchunk._grib_idx (new) or the image's
        # baked-in dynamic_zarr_store (kerchunk 0.2.6). Try both.
        try:
            from kerchunk._grib_idx import build_idx_grib_mapping
        except ImportError:
            from dynamic_zarr_store import build_idx_grib_mapping

        # validate=False: a few CFS messages don't decode (matches the local driver)
        try:
            mapping = build_idx_grib_mapping(
                basename=task["url"], storage_options={"anon": True}, validate=False)
        except TypeError:
            # older signature without validate=
            mapping = build_idx_grib_mapping(
                basename=task["url"], storage_options={"anon": True})

        deduped = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]

        out = f'cfs-time-{date}-{run}-rt{task["hour"]:04d}.parquet'
        deduped.to_parquet(out, engine="pyarrow")

        # locate the GCS key (uploaded to workers via client.upload_file)
        creds = None
        cands = []
        try:
            from distributed import get_worker
            cands.append(get_worker().local_directory)
        except Exception:
            pass
        cands += [os.environ.get("DASK_WORKER_LOCAL_DIRECTORY"), os.getcwd(), "."]
        for c in cands:
            if c and os.path.exists(os.path.join(c, gcs_key_name)):
                creds = os.path.join(c, gcs_key_name)
                break
        if creds is None:
            raise FileNotFoundError(f"{gcs_key_name} not found on worker")

        from google.oauth2 import service_account
        from google.cloud import storage
        cobj = service_account.Credentials.from_service_account_file(creds)
        cl = storage.Client(credentials=cobj, project=cobj.project_id)
        dest = f'cfs/time_idx/{date[:4]}/{date}/{run}/{task["member"]}/{out}'
        cl.bucket(GCS_BUCKET).blob(dest).upload_from_filename(out)
        return f'm{task["member"]}/{task["hour"]:04d}h ok ({len(deduped)} attrs)'

    cluster = scan_one.cluster
    cluster.adapt(min=1, max=args.max_workers)
    # ship the GCS service-account key to every worker (image doesn't bake it in)
    Client(cluster).upload_file(args.gcs_key)

    results = list(scan_one.map(tasks))
    ok = [r for r in results if isinstance(r, str) and "ok" in r]
    print(f"completed {len(ok)}/{len(tasks)}")
    for r in results:
        print("  ", r)
    return 0 if len(ok) == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
