#!/usr/bin/env python3
"""
ECMWF 49r1 Coiled preprocessing driver — single-stream cousin of
`ecmwf_50r1_coiled_preprocessing.py`.

This is Step 1 ("preprocessing", per `ecmwf/README.md`) of the 49r1
template rebuild: in parallel on Coiled, scan_grib every forecast-hour
GRIB of the 49r1 enfo/ef stream and store per-hour all-member dumps to
GCS `gs://{bucket}/fmrc/scan_grib{date}/e_sg_mdt_{date}_enfo_{h}h.parquet`.

Why this exists separately from the 50r1 driver:
  - 49r1 used a SINGLE stream (enfo/ef) carrying 51 members in one file
    (1 bundled control as `number=0` + 50 perturbed as `number=1..50`).
  - 50r1 (live 2026-05-12 06z) split the open-data ENS into two streams
    (enfo/ef perturbed 1..50 + oper/fc control), so its driver runs 170
    tasks; this 49r1 driver runs 85 (one per forecast hour).
The underlying Coiled config, scan operation, and output naming scheme
are identical so artifacts produced by either driver flow into the same
realigner (`ecmwf_par_to_ensemble_members.py`) without further branching.

  enfo/ef -> 51 members (control + 1..50)   (85 files)
  total 85 tasks @ ~1 h on 9-worker Coiled cluster (paid).

Calls fmrc_utils.s3_ecmwf_scan_grib_storing per task with stream="enfo"
so the dumps are stream-tagged on disk — keeps them distinguishable
from the future 50r1 dumps in shared GCS prefixes and lets the realigner
use the same 50r1-aware parse_filename path without a legacy branch.

The driver inherits the per-pressure-level keys fix from `4ca1c21`
(in `utils_ecmwf_step1_scangrib.py`'s `fixed_ensemble_grib_tree`) — the
preprocessing dump it produces is row-level per GRIB message, so all
13 isobaric levels per (var, hour) survive into the dump for the
aggregator (run later) to keep as separate zarr groups. See
[`2026-05-29-49r1-perlevel-reprocess-plan.md`](../lithops-cr-gik-ecmwf/2026-05-29-49r1-perlevel-reprocess-plan.md)
for the full chain and gates.

Must be launched by a user with Coiled auth in `ecmwf/dev-test/`
(`coiled login` done; `coiled-data.json` present). It cannot be launched
from the assistant. `--dry-run` prints the plan with no Coiled/cost.

Coiled config follows the 50r1 driver (which followed the notebook):
software=gik-coiled-v6, vm_type=n2-standard-2, region=us-east1,
arm=False, idle_timeout=30m, cluster.adapt(min=1,max=N). Workspace
defaults to gcp-sewaa-nka (overridable via flags).

Date guidance: in scope per the 49r1 plan = 2024-03-01 onwards (start
of 49r1 on ECMWF Open Data) through 2026-05-12 00z (last 49r run before
the 50r1 cutover at 2026-05-12 06z). The existing 49r1 template was
built on 20240529; reusing that reference date makes the new tar.gz
trivially comparable to the legacy one.

Usage:
    python ecmwf_49r1_coiled_preprocessing.py --date 20240529 --run 00 --dry-run
    python ecmwf_49r1_coiled_preprocessing.py --date 20240529 --run 00            # ~1 h, paid
    python ecmwf_49r1_coiled_preprocessing.py --date 20240529 --run 00 \
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
    """Build the 85-task plan: 85 enfo-only tasks for one date/run.

    49r1 has no oper/fc path — the control rides inside the enfo/ef file
    as `number=0`. So one task per forecast hour, not two.
    """
    tasks = []
    ts = f"{date}{run}0000"
    for hour in FORECAST_HOURS:
        h = str(hour)
        tasks.append({
            "stream": "enfo",
            "hour": h,
            "url": f"s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/enfo/{ts}-{h}h-enfo-ef.grib2",
        })
    return tasks


def print_plan(tasks, date, run):
    print(f"49r1 preprocessing plan: date={date} run={run}z  {len(tasks)} tasks "
          f"(enfo only)")
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
                         "(49r1 era: 2024-03-01 to 2026-05-12 00z; "
                         "rebuild ref date: 20240529)")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--max-workers", type=int, default=9)
    # Coiled config — defaults follow the 50r1 driver (which followed the
    # notebook). Workspace is made explicit so the ~1 h paid run can't
    # accidentally land in the wrong workspace.
    ap.add_argument("--software", default="gik-coiled-v6",
                    help="Coiled software environment (50r1 driver: gik-coiled-v6)")
    ap.add_argument("--workspace", default="gcp-sewaa-nka",
                    help="Coiled workspace/account (50r1 driver: gcp-sewaa-nka)")
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
    from distributed import Client

    date = args.date

    @coiled.function(
        vm_type=args.vm_type,
        software=args.software,        # gik-coiled-v6 (kerchunk 0.2.6 container)
        workspace=args.workspace,
        name="func-ecmwf-49r1",        # distinct cluster per era
        region=args.region,
        arm=False,
        idle_timeout="30 minutes",
    )
    def scan_one(task):
        # gik-coiled-v6 ships kerchunk 0.2.6 (no kerchunk._grib_idx), so we use
        # the image's BAKED-IN dynamic_zarr_store primitives rather than the
        # newer fmrc_utils refactor. The image also bakes in stale AWS creds, so
        # force ANONYMOUS s3 (public ECMWF bucket) via the fsspec protocol
        # default -- scan_grib() opens the url with no storage_options.
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
