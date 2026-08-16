# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "requests",
#   "google-cloud-storage",
# ]
# ///
"""Sequential par -> Icechunk backfill: one date, one subprocess, one commit.

The counterpart to `backfill_parallel.py`, and the driver to reach for when the
frisky/Dask cluster is unavailable or untrusted. It needs nothing but this host:
no scheduler, no worker VMs, no pushed code, no shared key path.

    per date:  stage 51 pars (GCS)  ->  build_ecmwf_icechunk.py  ->  commit  ->  rm pars

Each date runs in its own subprocess, so the coordinator's memory is flat and no
recycle loop is needed -- the ~14 MB/date creep that forces `LIMIT` on the
parallel driver dies with the child. The pars for the NEXT date are fetched on a
background thread while the current one builds; that overlaps the only phase
that is pure waiting, and changes nothing about the commit path.

Cost of being sequential: one commit per date, and commits are the bottleneck
(~12 MB/s to GCS, and a 49r1 date is a ~72 MB changeset). Measured ~60 s/date on
the first full run, against ~15-29 s/date for the batched parallel driver. Choose
accordingly -- for 800 dates that is 13 h versus 4 h.

Resume is exact and free: the group's time axis is preallocated, so every date
has a fixed index and `date_written` reads the manifest to see which indices
already hold refs. Killing this at any point loses at most the date in flight.

Usage
-----
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/frisky-ea/gcs-key.json

    # finish 06z, then all of 12z and 18z, smallest era first
    uv run backfill_all_eras.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
        --era 50r1,0p4,49r1 --run 06,12,18

    # one group, and look before leaping
    uv run backfill_all_eras.py --store gs://... --era 49r1 --run 06 --dry-run

The group must already exist with a preallocated time axis; that is what fixes
each date's index. Create it once with:

    build_ecmwf_icechunk.py --era 49r1 --run 12 --create-schema \
        --dates-file dates/dates-49r1-12z.txt --store gs://...

Par source: `gcs` is authoritative and the default. **HuggingFace still holds the
defective pars for 20260319 / 20260322-27 / 20260329-31** (collapsed pl levels),
so `--par-source hf` reintroduces a known bug -- see published-ecmwf-store-defects.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ecmwf_icechunk as B  # noqa: E402

HF = "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par-v2"
BUILDER = Path(__file__).resolve().parent / "build_ecmwf_icechunk.py"


# ---------------------------------------------------------------------------
# par staging
# ---------------------------------------------------------------------------
def stage_pars(date: str, run: str, args, dest: Path) -> Path:
    """Fetch one date's 51 member pars and return the directory holding them.

    Paths are derived from (date, run) rather than looked up: the published
    catalog.parquet lists 00z only, while the bucket holds all four runs.
    """
    if args.par_source == "local":
        d = Path(args.pars_root) / date
        n = len(list(d.glob("*.parquet")))
        if n != B.N_MEMBERS:
            raise RuntimeError(f"{d}: found {n} pars, expected {B.N_MEMBERS}")
        return d

    d = dest / f"{date}{run}z"
    if len(list(d.glob("*.parquet"))) == B.N_MEMBERS:
        return d                       # already staged (prefetch, or a retry)
    d.mkdir(parents=True, exist_ok=True)

    if args.par_source == "gcs":
        from google.cloud import storage
        path = f"{args.par_root_gcs}/{date[:4]}/{date[4:6]}/{date}/{run}z"
        bucket, _, prefix = path[5:].partition("/")
        blobs = [b for b in storage.Client().list_blobs(bucket, prefix=prefix + "/")
                 if b.name.endswith(".parquet")]
        if len(blobs) != B.N_MEMBERS:
            raise RuntimeError(f"{path}: {len(blobs)} pars, expected {B.N_MEMBERS}")

        def get(b):
            b.download_to_filename(d / b.name.split("/")[-1])
    else:
        import requests
        hf_path = f"par/{date[:4]}/{date[4:6]}/{date}/{run}z"
        tree = requests.get(
            f"https://huggingface.co/api/datasets/E4DRR/gik-ecmwf-par-v2/"
            f"tree/main/{hf_path}", timeout=60).json()
        blobs = sorted(x["path"].split("/")[-1] for x in tree
                       if x["path"].endswith(".parquet"))
        if len(blobs) != B.N_MEMBERS:
            raise RuntimeError(f"{hf_path}: {len(blobs)} pars, "
                               f"expected {B.N_MEMBERS}")

        def get(n):
            for attempt in range(3):
                try:
                    r = requests.get(f"{HF}/resolve/main/{hf_path}/{n}", timeout=120)
                    r.raise_for_status()
                    (d / n).write_bytes(r.content)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))

    # 51 small, latency-bound objects: overlap them
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, blobs))
    return d


# ---------------------------------------------------------------------------
# what is left to do
# ---------------------------------------------------------------------------
def open_group(session, grp: str, era: str):
    try:
        g = zarr.open_group(store=session.store, path=grp, mode="r",
                            zarr_format=3)
    except Exception:
        raise SystemExit(
            f"group {grp} does not exist. Create it first, which also freezes "
            f"the time axis:\n"
            f"    build_ecmwf_icechunk.py --era {era} --run {grp[-3:-1]} "
            f"--create-schema --dates-file <dates> --store <store>")
    if B.group_mode(g) != B.PREALLOC:
        raise SystemExit(
            f"REFUSING: group {grp} is in '{B.group_mode(g)}' mode. This driver "
            f"resumes by reading the manifest at each date's fixed index, which "
            f"an append-mode group does not have. Build it with --create-schema.")
    B.check_coords(g, era)          # the longitude guard
    return g


def pending(session, grp: str, g, args) -> list[tuple[str, int]]:
    """(date, time index) for every slot the axis wants that has no refs yet."""
    tvals = g["time"][:]
    dates = [((B.EPOCH + pd.Timedelta(hours=int(t))).strftime("%Y%m%d"), i)
             for i, t in enumerate(tvals)]
    if args.start:
        dates = [(d, i) for d, i in dates if d >= args.start]
    if args.end:
        dates = [(d, i) for d, i in dates if d <= args.end]
    return [(d, i) for d, i in dates if not B.date_written(session, grp, g, i)]


# ---------------------------------------------------------------------------
def build_group(era: str, run: str, args) -> tuple[int, list[tuple[str, str]]]:
    grp = f"{era}/{run}z"
    repo = B.open_or_create_repo(B.resolve_storage(args.store, args.sa_key))
    session = repo.readonly_session("main")
    g = open_group(session, grp, era)

    t_scan = time.time()
    todo = pending(session, grp, g, args)
    n_axis = int(g["time"].shape[0])
    print(f"{grp}: {n_axis} dates on the axis, {n_axis - len(todo)} already "
          f"written, {len(todo)} to build (scan {time.time()-t_scan:.0f}s)"
          + (f", limited to {args.limit}" if args.limit else ""), flush=True)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("  nothing to do", flush=True)
        return 0, []
    if args.dry_run:
        print(f"  would build {todo[0][0]}..{todo[-1][0]}", flush=True)
        return 0, []

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    done, fails, times = 0, [], []
    # one date of lookahead: the fetch of date N+1 overlaps the build of date N,
    # which is the only free parallelism a sequential driver has
    fetcher = ThreadPoolExecutor(1)
    ahead = fetcher.submit(stage_pars, todo[0][0], run, args, staging)

    try:
        for i, (date, ti) in enumerate(todo, 1):
            t0 = time.time()
            pars = None
            try:
                pars = ahead.result()
                if i < len(todo):       # start the next fetch before building
                    ahead = fetcher.submit(stage_pars, todo[i][0], run,
                                           args, staging)
                cmd = [sys.executable, "-u", str(BUILDER), "--era", era,
                       "--run", run, "--date", date, "--pars-dir", str(pars),
                       "--store", args.store]
                if args.sa_key:
                    cmd += ["--sa-key", args.sa_key]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError((r.stdout[-400:] + r.stderr[-400:]).strip()
                                       .replace("\n", " | "))
            except Exception as e:
                fails.append((date, str(e).splitlines()[-1][:200]))
                print(f"[{i}/{len(todo)}] {grp} {date}: FAILED -- "
                      f"{str(e).splitlines()[-1][:200]}", flush=True)
                continue
            finally:
                if pars is not None and args.par_source != "local" \
                        and not args.keep_pars:
                    shutil.rmtree(pars, ignore_errors=True)
            dt = time.time() - t0
            times.append(dt)
            done += 1
            eta = (len(todo) - i) * (sum(times[-30:]) / len(times[-30:])) / 3600
            print(f"[{i}/{len(todo)}] {grp} {date} (t={ti}): ok in {dt:.0f}s "
                  f"(ETA {eta:.1f} h)", flush=True)
    finally:
        fetcher.shutdown(wait=False, cancel_futures=True)

    el = sum(times)
    print(f"{grp}: {done} built, {len(fails)} failed"
          + (f" ({el/max(1,done):.0f} s/date)" if done else ""), flush=True)
    return done, fails


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None, help="GCS service-account json")
    ap.add_argument("--era", default="50r1,0p4,49r1",
                    help="comma list; default is smallest era first")
    ap.add_argument("--run", default="00", help="comma list of 00,06,12,18")
    ap.add_argument("--par-source", default="gcs", choices=["gcs", "local", "hf"],
                    help="gcs is authoritative; hf still holds the defective "
                         "March 2026 pars")
    ap.add_argument("--par-root-gcs",
                    default="gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf")
    ap.add_argument("--pars-root", default=None,
                    help="directory of <date>/ subdirs (--par-source local)")
    ap.add_argument("--staging", default="/tmp/gik_backfill_pars",
                    help="scratch for downloaded pars, deleted per date")
    ap.add_argument("--start", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--limit", type=int, default=None, help="max dates per group")
    ap.add_argument("--keep-pars", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eras = [e for e in args.era.split(",") if e]
    runs = [r for r in args.run.split(",") if r]
    bad = [e for e in eras if e not in B.ERAS] + [r for r in runs
                                                  if r not in ("00", "06", "12", "18")]
    if bad:
        raise SystemExit(f"unknown era/run: {bad}")
    if args.par_source == "hf":
        print("WARNING: HuggingFace holds defective pars for 20260319 / "
              "20260322-27 / 20260329-31 (collapsed pl levels)", flush=True)

    t_start = time.time()
    total, all_fails = 0, {}
    # run-outer, so a whole forecast run finishes before the next one starts
    for run in runs:
        for era in eras:
            done, fails = build_group(era, run, args)
            total += done
            if fails:
                all_fails[f"{era}/{run}z"] = fails
            print(flush=True)

    print(f"done: {total} dates built in {(time.time()-t_start)/3600:.2f} h")
    if all_fails:
        for grp, fails in all_fails.items():
            print(f"{grp}: {len(fails)} failed -- "
                  f"{[d for d, _ in fails][:20]}")
            for d, why in fails[:5]:
                print(f"    {d}: {why}")
        # nonzero so a wrapper stops instead of marching on through every
        # remaining group repeating the same failure
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
