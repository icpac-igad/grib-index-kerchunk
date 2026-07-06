# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "gcsfs",
# ]
# ///
"""Backfill the GEFS Icechunk store from the GCS par archive -- the GEFS
counterpart of ecmwf/icechunk-par/backfill_all_eras.py.

  gs://gik-gefs-aws-tf/run_par_gefs/{YYYY}/{MM}/{date}/00z/*.parquet  (30 pars)
      -> build_gefs_icechunk.py per date -> commit -> delete local pars

Dates are enumerated from the GCS tree itself (authoritative; the HF mirror
E4DRR/gik-gefs-par only holds 2020-2021). Resumable at any interruption: dates
already in the store's 0p25/00z time axis are skipped, and appends must stay
chronological (start from an empty store prefix or the true next date).

All progress goes to stdout AND gefs/backfill_gefs.log. Estimated full run:
1,926+ dates x ~12-15 s (fetch+build+commit) ~= 7-8 h sequential.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
  uv run backfill_gefs_icechunk.py --store gs://gik-gefs-aws-tf/icechunk/gefs-ens
  # options: --start 20200923 --end 20211231 --limit 5 --dry-run
  # full run: nohup uv run backfill_gefs_icechunk.py \
  #     --store gs://gik-gefs-aws-tf/icechunk/gefs-ens > /dev/null 2>&1 &
  #     (tail -f backfill_gefs.log to watch)
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GCS_PARS = "gik-gefs-aws-tf/run_par_gefs"
BUILDER = Path(__file__).parent / "build_gefs_icechunk.py"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
N_MEMBERS = 30

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(Path(__file__).parent / "backfill_gefs.log"),
              logging.StreamHandler()])
log = logging.getLogger("backfill_gefs")


def list_gcs_dates(fs) -> pd.DataFrame:
    """Walk the par tree -> DataFrame [date, path, n_pars], chronological."""
    rows = []
    for ydir in sorted(fs.ls(GCS_PARS)):
        for mdir in sorted(fs.ls(ydir)):
            for ddir in sorted(fs.ls(mdir)):
                run = f"{ddir}/00z"
                date = ddir.rsplit("/", 1)[-1]
                if date.isdigit() and len(date) == 8:
                    rows.append((date, run))
    return pd.DataFrame(rows, columns=["date", "gcs_path"])


def dates_done(store: str, sa_key: str | None) -> set[str]:
    import icechunk
    import zarr
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        storage = icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    else:
        storage = icechunk.local_filesystem_storage(store)
    if not icechunk.Repository.exists(storage):
        return set()
    auth = icechunk.containers_credentials(
        {"s3://noaa-gefs-pds/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    try:
        hours = zarr.open_array(repo.readonly_session("main").store,
                                path="0p25/00z/time", mode="r")[:]
        return {(EPOCH + pd.Timedelta(hours=int(h))).strftime("%Y%m%d")
                for h in np.asarray(hours)}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--start", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work-dir", default="/tmp/gefs_backfill_pars")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import gcsfs
    fs = gcsfs.GCSFileSystem(
        token=args.sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    cat = list_gcs_dates(fs)
    if args.start:
        cat = cat[cat.date >= args.start]
    if args.end:
        cat = cat[cat.date <= args.end]

    done = dates_done(args.store, args.sa_key)
    todo = cat[~cat.date.isin(done)]
    log.info(f"GCS corpus: {len(cat)} dates in scope | in store: "
             f"{len(cat) - len(todo)} | to build: {len(todo)}"
             + (f" (limited to {args.limit})" if args.limit else ""))
    if args.limit:
        todo = todo.head(args.limit)
    if args.dry_run:
        log.info(f"first: {todo.date.min()}  last: {todo.date.max()}")
        return

    t_start, times, fails = time.time(), [], []
    for i, row in enumerate(todo.itertuples(), 1):
        pars = Path(args.work_dir) / row.date
        t0 = time.time()
        try:
            pars.mkdir(parents=True, exist_ok=True)
            fs.get(row.gcs_path + "/*.parquet", str(pars) + "/")
            n = len(list(pars.glob("*.parquet")))
            if n != N_MEMBERS:
                raise RuntimeError(f"expected {N_MEMBERS} pars, got {n}")
            cmd = [sys.executable, str(BUILDER), "--date", row.date,
                   "--pars-dir", str(pars), "--store", args.store]
            if args.sa_key:
                cmd += ["--sa-key", args.sa_key]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
        except Exception as e:
            fails.append(row.date)
            log.error(f"[{i}/{len(todo)}] {row.date}: FAILED -- {e}")
            continue
        finally:
            subprocess.run(["rm", "-rf", str(pars)])
        dt = time.time() - t0
        times.append(dt)
        eta_h = (len(todo) - i) * (sum(times[-50:]) / len(times[-50:])) / 3600
        log.info(f"[{i}/{len(todo)}] {row.date}: ok in {dt:.0f}s (ETA {eta_h:.1f} h)")

    log.info(f"done: {len(times)} built, {len(fails)} failed in "
             f"{(time.time()-t_start)/3600:.2f} h")
    if fails:
        log.info(f"failed dates (re-run to retry): {fails}")


if __name__ == "__main__":
    main()
