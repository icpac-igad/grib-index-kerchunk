# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "pandas", "gribberish>=1.4", "gcsfs",
# ]
# ///
"""Health check for the GEFS Icechunk store -- GEFS counterpart of
ecmwf/icechunk-par/check_store_health.py.

  H1 repo opens; commits count/rate; latest commit message
  H2 0p25/00z time axis unique (+sorted, WARN if gap-filled out of order)
  H3 every data array sized to the time axis (zarr-level; catches the
     schema-drift divergence class found on the ECMWF store)
  H4 progress vs the AUTHORITATIVE GCS par tree (counted live, not a
     hardcoded catalog) -> missing dates listed if any
  H5 (--decode) GEFS-specific spot reads on the NEWEST date:
     t2m (instant) decodes finite; tp f000 is all-NaN (accumulated vars
     publish no f000 message -- NaN there is CORRECT) while tp f003 decodes
  H6 (--log) scan backfill_gefs.log for FAILED dates; process liveness

Exit code 0 = all PASS (WARNs allowed), 1 = any FAIL.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
  uv run check_gefs_store_health.py --decode --log backfill_gefs.log
"""
import argparse
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import icechunk
import zarr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

GROUP = "0p25/00z"
GCS_PARS = "gik-gefs-aws-tf/run_par_gefs"
COORD_NAMES = {"time", "number", "step", "latitude", "longitude"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
FAILURES, WARNINGS = [], []


def check(name, ok, detail="", warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        (WARNINGS if warn else FAILURES).append(name)


def gcs_corpus_dates(sa_key):
    import gcsfs
    fs = gcsfs.GCSFileSystem(token=sa_key)
    dates = set()
    for ydir in fs.ls(GCS_PARS):
        for mdir in fs.ls(ydir):
            for ddir in fs.ls(mdir):
                d = ddir.rsplit("/", 1)[-1]
                if d.isdigit() and len(d) == 8:
                    dates.add(d)
    return dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="gs://gik-gefs-aws-tf/icechunk/gefs-ens")
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--skip-corpus", action="store_true",
                    help="skip the GCS par-tree walk (faster)")
    args = ap.parse_args()
    sa_key = args.sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    print(f"== H1 repository: {args.store} ==")
    bucket, _, prefix = args.store[5:].partition("/")
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix.rstrip("/"),
                                   service_account_file=sa_key)
    auth = icechunk.containers_credentials(
        {"s3://noaa-gefs-pds/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    anc = list(repo.ancestry(branch="main"))
    check("repo opens, main branch resolvable", True,
          f"{len(anc)} commits, tip {anc[0].id[:12]}..")
    print(f"       tip: '{anc[0].message.strip()}' at {anc[0].written_at}")

    print(f"== H2/H3 group {GROUP} ==")
    ro = repo.readonly_session("main").store
    g = zarr.open_group(store=ro, path=GROUP, mode="r", zarr_format=3)
    hours = g["time"][:]
    n = len(hours)
    dates_in = {(EPOCH + pd.Timedelta(hours=int(h))).strftime("%Y%m%d")
                for h in np.asarray(hours)}
    uniq = len(dates_in) == n
    mono = bool(np.all(np.diff(hours) > 0)) if n > 1 else True
    d0 = (EPOCH + pd.Timedelta(hours=int(hours.min()))).strftime("%Y%m%d")
    d1 = (EPOCH + pd.Timedelta(hours=int(hours.max()))).strftime("%Y%m%d")
    check("time axis unique", uniq, f"{n} dates: {d0}..{d1}")
    if not mono:
        check("time axis sorted", False,
              "gap-filled out of order -- consumers should sortby('time')", warn=True)
    data_arrays = [k for k in g.array_keys() if k not in COORD_NAMES]
    bad = {k: g[k].shape[0] for k in data_arrays if g[k].shape[0] != n}
    check(f"all {len(data_arrays)} arrays sized to time={n}", not bad,
          f"lagging: {bad}" if bad else "")
    check("member axis = 30 (gep01..gep30)", g["number"].shape[0] == 30)
    check("step axis = 81 (0..240h/3h)", g["step"].shape[0] == 81)

    if not args.skip_corpus:
        print("== H4 progress vs GCS par tree ==")
        corpus = gcs_corpus_dates(sa_key)
        missing = sorted(corpus - dates_in)
        extra = sorted(dates_in - corpus)
        check(f"store covers corpus ({len(corpus)} dates)", not missing,
              f"missing {len(missing)}: {missing[:8]}" if missing
              else f"{len(dates_in)}/{len(corpus)} (100%)")
        if extra:
            check("no unexpected dates", False, f"in store but not corpus: {extra[:5]}",
                  warn=True)

    if args.decode:
        print("== H5 spot decode (newest date; GEFS accum semantics) ==")
        ti = int(np.argmax(hours))
        newest = (EPOCH + pd.Timedelta(hours=int(hours[ti]))).date()
        t0 = time.time()
        t2m = g["t2m"][ti, 0, 0]          # instant var, f000 exists
        check(f"t2m @ {newest} f000 decodes", np.isfinite(t2m).mean() > 0.99,
              f"mean={np.nanmean(t2m):.2f} K ({time.time()-t0:.1f}s)")
        tp0 = g["tp"][ti, 0, 0]           # accum var: NO f000 message -> NaN correct
        check("tp f000 all-NaN (accum var, expected)", bool(np.isnan(tp0).all()))
        tp3 = g["tp"][ti, 0, 1]
        check("tp f003 decodes", np.isfinite(tp3).mean() > 0.99,
              f"mean={np.nanmean(tp3):.4f} kg m-2")

    if args.log:
        print(f"== H6 backfill log: {args.log} ==")
        try:
            lines = open(args.log).read().splitlines()
            fails = [l for l in lines if "FAILED" in l]
            check("no FAILED dates in log", not fails,
                  f"{len(fails)} failed, last: {fails[-1][:90]}" if fails else "",
                  warn=True)
            done = [l for l in lines if l.rstrip().startswith("done:") or " done:" in l]
            if done:
                print(f"       {done[-1].split('INFO')[-1].strip()}")
        except FileNotFoundError:
            check("log file readable", False, args.log, warn=True)

    print(f"\n{'HEALTH OK' if not FAILURES else 'UNHEALTHY: ' + ', '.join(FAILURES)}"
          + (f" ({len(WARNINGS)} warnings)" if WARNINGS else ""))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
