#!/usr/bin/env python3
"""
ECMWF 0.4deg (`0p4-beta`) Coiled preprocessing driver — archive cousin of
`ecmwf_49r1_coiled_preprocessing.py`.

This is Step 1 ("preprocessing", per `ecmwf/README.md`) of a template
rebuild for the pre-0.25deg era: in parallel on Coiled, scan_grib every
forecast-hour GRIB of the 0.4deg enfo/ef stream and store per-hour
all-member dumps to GCS
`gs://{bucket}/fmrc/scan_grib{date}/e_sg_mdt_{date}_enfo_{h}h.parquet`.

Why this exists separately from the 49r1 / 50r1 drivers:
  - Before 2024-02-29, ECMWF open data was published at 0.4deg under a
    DIFFERENT path: `{date}/{run}z/0p4-beta/{stream}/...` (no `ifs`
    segment), grid 451x900 instead of 721x1440.
  - Structurally it matches the EARLY (9-level) 49r1 era otherwise:
    SINGLE stream (enfo/ef) carrying 51 members in one file (1 bundled
    control as `number=0` + 50 perturbed `number=1..50`), 9 pressure
    levels [50,200,250,300,500,700,850,925,1000], 85 forecast hours
    (0-144 every 3h, 150-360 every 6h). The era is homogeneous, so a
    single reference date covers it.
  - Empirical archive bounds: 0p4-beta runs 2023-01-18 .. 2024-02-28;
    `ifs/0p25` begins 2024-02-29. See
    [`../lithops-cr-gik-ecmwf/2026-06-01-ecmwf-0p4-beta-archive-evaluation.md`](../lithops-cr-gik-ecmwf/2026-06-01-ecmwf-0p4-beta-archive-evaluation.md).

The ONLY differences from the 49r1 driver are the source URL path prefix
(`0p4-beta` vs `ifs/0p25`) and the date guidance. Output naming is
identical (`e_sg_mdt_{date}_enfo_{h}h.parquet`) so the dumps flow into the
same realigner (`ecmwf_par_to_ensemble_members.py`) with no branching —
the realigner recovers the 0.4deg grid shape from each dump's `uri`
column (`field_shape_for_uri` -> [1,451,900]), so no downstream change is
needed.

Calls fmrc_utils.s3_ecmwf_scan_grib_storing per task with stream="enfo"
so the dumps are stream-tagged on disk, consistent with the 49r1/50r1
drivers.

Must be launched by a user with Coiled auth in `ecmwf/dev-test/`
(`coiled login` done; `coiled-data.json` present). It cannot be launched
from the assistant. `--dry-run` prints the plan with no Coiled/cost.

Coiled config follows the 49r1 driver (which followed the notebook):
software=gik-coiled-v6, vm_type=n2-standard-2, region=us-east1,
arm=False, idle_timeout=30m, cluster.adapt(min=1,max=N). Workspace
defaults to gcp-sewaa-nka (overridable via flags).

Date guidance: in scope = 2023-01-18 (archive start) through 2024-02-28
(last 0.4deg day; 2024-02-29 onward is 0.25deg). The era is homogeneous,
so any 00z date works as the template reference; a clean mid-era choice
is 20230601.

Usage:
    python ecmwf_0p4_coiled_preprocessing.py --date 20230601 --run 00 --dry-run
    python ecmwf_0p4_coiled_preprocessing.py --date 20230601 --run 00            # ~1 h, paid
    python ecmwf_0p4_coiled_preprocessing.py --date 20230601 --run 00 \
        --software gik-coiled-v6 --workspace gcp-sewaa-nka --max-workers 9
"""
import os
import re
import argparse
import sys

# Forecast hours: 0-144 every 3h, then 150-360 every 6h (85 total) — same
# cadence as the 0.25deg era (verified on real 0p4-beta enfo listings).
FORECAST_HOURS = list(range(0, 145, 3)) + list(range(150, 361, 6))
assert len(FORECAST_HOURS) == 85, f"expected 85 hours, got {len(FORECAST_HOURS)}"

GCS_BUCKET = "gik-ecmwf-aws-tf"


def build_task_plan(date, run):
    """Build the 85-task plan: 85 enfo-only tasks for one date/run.

    0.4deg open data lives at `{date}/{run}z/0p4-beta/enfo/...` (no `ifs`
    segment) and bundles the control inside the enfo/ef file as
    `number=0`, so one task per forecast hour (no oper/fc path needed).
    """
    tasks = []
    ts = f"{date}{run}0000"
    for hour in FORECAST_HOURS:
        h = str(hour)
        tasks.append({
            "stream": "enfo",
            "hour": h,
            "url": f"s3://ecmwf-forecasts/{date}/{run}z/0p4-beta/enfo/{ts}-{h}h-enfo-ef.grib2",
        })
    return tasks


def print_plan(tasks, date, run):
    print(f"0p4-beta (0.4deg) preprocessing plan: date={date} run={run}z  "
          f"{len(tasks)} tasks (enfo only)")
    print(f"GCS dest: gs://{GCS_BUCKET}/fmrc/scan_grib{date}/"
          f"e_sg_mdt_{date}_enfo_<h>h.parquet")
    for t in tasks[:2]:
        print(f"  [{t['stream']}]  {t['hour']}h  {t['url']}")
    if len(tasks) > 2:
        print(f"  ... ({len(tasks) - 2} more)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True,
                    help="preprocessing date YYYYMMDD "
                         "(0.4deg era: 2023-01-18 to 2024-02-28; "
                         "suggested ref date: 20230601)")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--max-workers", type=int, default=9)
    # Coiled config — defaults follow the 49r1/50r1 drivers (which followed
    # the notebook). Workspace is made explicit so the ~1 h paid run can't
    # accidentally land in the wrong workspace.
    ap.add_argument("--software", default="gik-coiled-pinned",
                    help="Coiled software env. Default gik-coiled-pinned = the "
                         "reproducible build (devops/Dockerfile.gik-coiled-pinned); "
                         "build+register it first. Existing fallback with identical "
                         "pins: --software gik-coiled-v6.")
    ap.add_argument("--workspace", default="gcp-sewaa-nka",
                    help="Coiled workspace/account (49r1 driver: gcp-sewaa-nka)")
    ap.add_argument("--vm-type", default="n2-standard-2")
    ap.add_argument("--region", default="us-east1")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the task plan; no Coiled, no cost")
    args = ap.parse_args()

    if not re.match(r"^\d{8}$", args.date):
        sys.exit(f"--date must be YYYYMMDD; got {args.date!r}")
    if not ("20230118" <= args.date <= "20240228"):
        print(f"WARNING: --date {args.date} is outside the 0.4deg era "
              f"(2023-01-18 .. 2024-02-28). 2024-02-29 onward is 0.25deg — "
              f"use ecmwf_49r1_coiled_preprocessing.py instead.", file=sys.stderr)

    tasks = build_task_plan(args.date, args.run)
    print_plan(tasks, args.date, args.run)

    if args.dry_run:
        return 0

    import coiled
    from distributed import Client

    date = args.date

    @coiled.function(
        vm_type=args.vm_type,          # 49r1 driver: n2-standard-2
        software=args.software,        # gik-coiled-v6 (kerchunk 0.2.6 container)
        workspace=args.workspace,      # 49r1 driver: gcp-sewaa-nka
        name="func-ecmwf-0p4",         # distinct from func-ecmwf-49r1 / -50r1
        region=args.region,            # 49r1 driver: us-east1
        arm=False,                     # 49r1 driver: arm=False
        idle_timeout="30 minutes",     # 49r1 driver: 30 minutes
    )
    def scan_one(task):
        # gik-coiled-v6 ships kerchunk 0.2.6 (no kerchunk._grib_idx), so we use
        # the image's BAKED-IN dynamic_zarr_store primitives rather than the
        # newer fmrc_utils refactor. The image also bakes in stale AWS creds, so
        # force ANONYMOUS s3 (public ECMWF bucket) via the fsspec protocol
        # default — scan_grib() opens the url with no storage_options.
        import os
        import fsspec
        import s3fs
        fsspec.config.conf["s3"] = {"anon": True}
        s3fs.S3FileSystem.clear_instance_cache()
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/nonexistent"
        import dynamic_zarr_store as dz

        # locate coiled-data.json (uploaded to workers via client.upload_file)
        creds = None
        cands = []
        try:
            from distributed import get_worker
            cands.append(get_worker().local_directory)
        except Exception:
            pass
        cands += [os.environ.get("DASK_WORKER_LOCAL_DIRECTORY"), os.getcwd(), "."]
        for c in cands:
            if c and os.path.exists(os.path.join(c, "coiled-data.json")):
                creds = os.path.join(c, "coiled-data.json")
                break
        if creds is None:
            raise FileNotFoundError("coiled-data.json not found on worker")

        out = f'e_sg_mdt_{date}_{task["stream"]}_{task["hour"]}h.parquet'
        df = dz._map_grib_file_by_group(task["url"])     # scan_grib -> per-message dump
        df.to_parquet(out, engine="pyarrow")

        from google.oauth2 import service_account
        from google.cloud import storage
        cobj = service_account.Credentials.from_service_account_file(creds)
        cl = storage.Client(credentials=cobj, project=cobj.project_id)
        dest = f'fmrc/scan_grib{date}/{out}'
        cl.bucket(GCS_BUCKET).blob(dest).upload_from_filename(out)
        return f"{task['stream']}/{task['hour']}h ok ({len(df)} msgs)"

    cluster = scan_one.cluster
    cluster.adapt(min=1, max=args.max_workers)
    # ship the GCS service-account key to every worker (image doesn't bake it in)
    Client(cluster).upload_file("coiled-data.json")
    results = list(scan_one.map(tasks))
    print(f"completed {len(results)}/{len(tasks)}")
    for r in results:
        print("  ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
