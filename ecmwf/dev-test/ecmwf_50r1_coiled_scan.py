#!/usr/bin/env python3
"""
ECMWF 50r1 Coiled scan driver — replaces the hardcoded notebook
`99o-coiled-function-ecmwf-scan_grib_store_fmrc.ipynb`.

Step 1 of the template build: scan every forecast-hour GRIB of BOTH 50r1
streams in parallel on Coiled and store per-(stream,hour) all-member dumps
to GCS `gs://{bucket}/fmrc/scan_grib{date}/e_sg_mdt_{date}_{stream}_{h}h.parquet`.

  enfo/ef -> perturbed members 1..50   (85 files)
  oper/fc -> control (number -1)       (85 files)
  total 170 tasks @ ~2 h on 9-worker Coiled cluster (paid).

Calls fmrc_utils.s3_ecmwf_scan_grib_storing per task with stream tag so the
two streams' same-hour dumps don't collide.

Must be launched by a user with Coiled auth in `ecmwf/dev-test/`
(`coiled login` done; `coiled-data.json` present). It cannot be launched
from the assistant. `--dry-run` prints the plan with no Coiled/cost.

Coiled config follows the notebook: software=gik-coiled-v6,
vm_type=n2-standard-2, region=us-east1, arm=False, idle_timeout=30m,
cluster.adapt(min=1,max=N). Workspace defaults to gcp-sewaa-nka (the
notebook used the `coiled login` default; made explicit here). All
overridable via flags.

Usage:
    python ecmwf_50r1_coiled_scan.py --date 20260513 --run 00 --dry-run
    python ecmwf_50r1_coiled_scan.py --date 20260513 --run 00            # ~2 h, paid
    python ecmwf_50r1_coiled_scan.py --date 20260513 --run 00 \
        --software gik-coiled-v6 --workspace gcp-sewaa-nka --max-workers 9
"""
import os
import re
import argparse
import sys

# Forecast hours: 0-144 every 3h, then 150-360 every 6h (85 total)
FORECAST_HOURS = list(range(0, 145, 3)) + list(range(150, 361, 6))
assert len(FORECAST_HOURS) == 85, f"expected 85 hours, got {len(FORECAST_HOURS)}"

GCS_BUCKET = "gik-ecmwf-aws-tf"


def build_task_plan(date, run):
    """Build the 170-task plan: 85 enfo + 85 oper tasks for one date/run."""
    tasks = []
    ts = f"{date}{run}0000"
    for hour in FORECAST_HOURS:
        h = str(hour)
        tasks.append({
            "stream": "enfo",
            "hour": h,
            "url": f"s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/enfo/{ts}-{h}h-enfo-ef.grib2",
        })
        tasks.append({
            "stream": "oper",
            "hour": h,
            "url": f"s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/oper/{ts}-{h}h-oper-fc.grib2",
        })
    return tasks


def print_plan(tasks, date, run):
    n_enfo = sum(t["stream"] == "enfo" for t in tasks)
    n_oper = sum(t["stream"] == "oper" for t in tasks)
    print(f"50r1 scan plan: date={date} run={run}z  {len(tasks)} tasks "
          f"({n_enfo} enfo + {n_oper} oper)")
    print(f"GCS dest: gs://{GCS_BUCKET}/fmrc/scan_grib{date}/"
          f"e_sg_mdt_{date}_<stream>_<h>h.parquet")
    for t in tasks[:2]:
        print(f"  [{t['stream']}]  {t['hour']}h  {t['url']}")
    if len(tasks) > 2:
        print(f"  ... ({len(tasks) - 2} more)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="scan date YYYYMMDD (>=20260513)")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--max-workers", type=int, default=9)
    # Coiled config — defaults follow the notebook
    # (99o-coiled-function-ecmwf-scan_grib_store_fmrc.ipynb). The notebook
    # did NOT set a workspace (it used the `coiled login` default); the run
    # the user referenced was workspace gcp-sewaa-nka — made explicit here so
    # the scan can't accidentally land in the wrong workspace.
    ap.add_argument("--software", default="gik-coiled-v6",
                    help="Coiled software environment (notebook: gik-coiled-v6)")
    ap.add_argument("--workspace", default="gcp-sewaa-nka",
                    help="Coiled workspace/account (notebook used login default)")
    ap.add_argument("--vm-type", default="n2-standard-2")
    ap.add_argument("--region", default="us-east1")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the task plan; no Coiled, no cost")
    args = ap.parse_args()

    if not re.match(r"^\d{8}$", args.date):
        sys.exit(f"--date must be YYYYMMDD; got {args.date!r}")

    tasks = build_task_plan(args.date, args.run)
    print_plan(tasks, args.date, args.run)

    if args.dry_run:
        return 0

    import coiled
    import fsspec

    @coiled.function(
        vm_type=args.vm_type,          # notebook: n2-standard-2
        software=args.software,        # notebook: gik-coiled-v6
        workspace=args.workspace,      # notebook: login default (gcp-sewaa-nka)
        name="func-ecmwf-50r1",        # notebook: func-ecmwf3
        region=args.region,            # notebook: us-east1
        arm=False,                     # notebook: arm=False
        idle_timeout="30 minutes",     # notebook: 30 minutes
    )
    def scan_one(task):
        from fmrc_utils import s3_ecmwf_scan_grib_storing
        fs = fsspec.filesystem("s3", anon=True)
        creds_path = os.path.join(
            os.environ.get("DASK_WORKER_LOCAL_DIRECTORY", "."),
            "coiled-data.json",
        )
        s3_ecmwf_scan_grib_storing(
            fs, task["url"], args.date, "index", f"{task['hour']}h",
            GCS_BUCKET, creds_path, stream=task["stream"],
        )
        return f"{task['stream']}/{task['hour']}h ok"

    cluster = scan_one.cluster
    cluster.adapt(min=1, max=args.max_workers)
    results = list(scan_one.map(tasks))
    print(f"completed {len(results)}/{len(tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
