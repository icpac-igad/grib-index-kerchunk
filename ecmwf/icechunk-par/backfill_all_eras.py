# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "requests",
# ]
# ///
"""Backfill the unified ECMWF Icechunk store from the HF par archive.

Drives build_ecmwf_icechunk.py over the whole catalog (1,256 dates, four
schema regimes -> three era groups), one commit per date, resumable:

  catalog.parquet (HF)  ->  per date: download 51 pars -> build -> commit -> delete pars

Estimated sequential wall time on GCS from a small VM: ~13 h
(0p4 ~26 s/date x 401, 49r1 ~35-48 s x 804, 50r1 ~50 s x 51) -- see
CONVERSION_ESTIMATE.md. Safe to interrupt at ANY point: already-committed
dates are read from the store's per-group time axes and skipped, so a
restart continues where it stopped. Disk use is one date of pars (~25 MB).

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
  uv run backfill_all_eras.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens
  # options: --era 0p4  --start 20230118 --end 20240228  --limit 10  --dry-run
  # long run: nohup uv run backfill_all_eras.py --store gs://... > backfill.log 2>&1 &
"""
import argparse
import io
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HF = "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par-v2"
BUILDER = Path(__file__).parent / "build_ecmwf_icechunk.py"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def load_catalog() -> pd.DataFrame:
    r = requests.get(f"{HF}/resolve/main/catalog.parquet", timeout=60)
    r.raise_for_status()
    c = pd.read_parquet(io.BytesIO(r.content))
    return c.sort_values("date").reset_index(drop=True)


def dates_done(store: str, sa_key: str | None) -> dict[str, set]:
    """Read each era group's time axis -> set of YYYYMMDD already committed."""
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
        return {}
    auth = icechunk.containers_credentials(
        {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    ro = repo.readonly_session("main").store
    done = {}
    for era in ("0p4", "49r1", "50r1"):
        try:
            hours = zarr.open_array(ro, path=f"{era}/00z/time", mode="r")[:]
            done[era] = {(EPOCH + pd.Timedelta(hours=int(h))).strftime("%Y%m%d")
                         for h in np.asarray(hours)}
        except Exception:
            done[era] = set()
    return done


def download_date(row, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    listing = requests.get(
        f"https://huggingface.co/api/datasets/E4DRR/gik-ecmwf-par-v2/tree/main/{row.hf_path}",
        timeout=60).json()
    names = [x["path"].split("/")[-1] for x in listing if x["path"].endswith(".parquet")]

    def get(name, tries=3):
        for i in range(tries):
            try:
                r = requests.get(f"{HF}/resolve/main/{row.hf_path}/{name}", timeout=120)
                r.raise_for_status()
                (dest / name).write_bytes(r.content)
                return
            except Exception:
                if i == tries - 1:
                    raise
                time.sleep(2 * (i + 1))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, names))
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--era", choices=["0p4", "49r1", "50r1"], default=None)
    ap.add_argument("--start", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--limit", type=int, default=None, help="max dates this run")
    ap.add_argument("--work-dir", default="/tmp/gik_backfill_pars")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cat = load_catalog()
    if args.era:
        cat = cat[cat.era == args.era]
    if args.start:
        cat = cat[cat.date.astype(str) >= args.start]
    if args.end:
        cat = cat[cat.date.astype(str) <= args.end]

    done = dates_done(args.store, args.sa_key)
    todo = cat[~cat.apply(lambda r: str(r.date) in done.get(r.era, set()), axis=1)]
    print(f"catalog: {len(cat)} dates in scope | already in store: "
          f"{len(cat) - len(todo)} | to build: {len(todo)}"
          + (f" (limited to {args.limit})" if args.limit else ""))
    if args.limit:
        todo = todo.head(args.limit)
    if args.dry_run:
        print(todo.groupby("era")["date"].agg(["count", "min", "max"]).to_string())
        return

    t_start, times, fails = time.time(), [], []
    for i, row in enumerate(todo.itertuples(), 1):
        date = str(row.date)
        pars = Path(args.work_dir) / date
        t0 = time.time()
        try:
            n = download_date(row, pars)
            if n != 51:
                raise RuntimeError(f"expected 51 pars, got {n}")
            cmd = [sys.executable, str(BUILDER), "--era", row.era, "--run", "00",
                   "--date", date, "--pars-dir", str(pars), "--store", args.store]
            if args.sa_key:
                cmd += ["--sa-key", args.sa_key]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stdout[-500:] + r.stderr[-500:])
        except Exception as e:
            fails.append(date)
            print(f"[{i}/{len(todo)}] {row.era} {date}: FAILED -- {e}", flush=True)
            continue
        finally:
            subprocess.run(["rm", "-rf", str(pars)])
        dt = time.time() - t0
        times.append(dt)
        eta_h = (len(todo) - i) * (sum(times[-50:]) / len(times[-50:])) / 3600
        print(f"[{i}/{len(todo)}] {row.era} {date}: ok in {dt:.0f}s "
              f"(ETA {eta_h:.1f} h)", flush=True)

    print(f"\ndone: {len(times)} built, {len(fails)} failed in "
          f"{(time.time()-t_start)/3600:.2f} h")
    if fails:
        print("failed dates (re-run to retry):", fails)


if __name__ == "__main__":
    main()
