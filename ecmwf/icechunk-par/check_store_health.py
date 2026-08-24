# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "gribberish>=1.4",
# ]
# ///
"""Health check for a GIK Icechunk store while (or after) a backfill runs.

Checks, without disturbing the writer (read-only session on the current tip):
  H1 repo opens; branch tip + total commits; commit RATE from snapshot
     timestamps (last 20) -> ETA for the remaining catalog dates
  H2 per group: time axis monotonic + duplicate-free, date range, count
  H3 store metadata consistency: every group array opens, shapes agree with
     the time axis (no torn/partial commit visible on the tip)
  H3b geolocation: each era group's latitude/longitude axes equal the canonical
     ones in grib-index-kerchunk/ecmwf/grids.py. Shapes agreeing with the time
     axis says nothing about WHERE the field is; a store written before the
     longitude fix carries 0..359.75 and every reader silently gets the wrong
     hemisphere. See grids.py and HANDOVER_LONGITUDE_FIX.md
  H4 (--decode) spot-read one chunk of the LATEST date through the
     gribberish codec -> proves refs written by the newest commit resolve
  H5 (--log FILE) scan the backfill log for FAILED dates; report process

Exit code: 0 = all PASS, 1 = any FAIL (WARNs don't fail).

Usage (ECMWF store, from the directory holding this file):
  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
  uv run check_store_health.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens \
      --expected-dates 1256 --decode --log backfill.log
GEFS store:
  uv run check_store_health.py --store gs://gik-gefs-aws-tf/icechunk/gefs-ens \
      --container s3://noaa-gefs-pds/ --expected-dates 2031 --decode
Run regularly:  watch -n 600 ...  or a cron/loop.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import icechunk
import zarr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
FAILURES, WARNINGS = [], []


# --- the canonical grid, for H3b. Same resolution as build_ecmwf_icechunk.py.
# Deliberately copied rather than imported from it: that module pulls in the
# whole builder to answer a question about sys.path. What must never be copied
# is the grid itself, and that comes from grids.py either way. ---
def _locate_grids() -> Path | None:
    """Find grids.py, or None.

    Unlike the builder this does NOT exit when the module is missing: this
    script also checks non-ECMWF stores (GEFS), which have no era groups to
    check against it. H3b turns a None into a FAIL only for an ECMWF store,
    where the check is not optional.

    Disagreement between reachable copies IS fatal here too -- checking a store
    against a grids.py you cannot identify proves nothing.
    """
    env = os.environ.get("GIK_ECMWF_DIR")
    here = Path(__file__).resolve()
    cands = ([Path(env)] if env else []) + [here.parent, here.parent.parent]
    for anc in here.parents:
        cands += [anc / "grib-index-kerchunk" / "ecmwf", anc / "ecmwf"]
    seen, found = set(), []
    for c in cands:
        f = c / "grids.py"
        if f.is_file() and f.resolve() not in seen:
            seen.add(f.resolve())
            found.append(c)
    if len(found) > 1:
        body = (found[0] / "grids.py").read_bytes()
        bad = [d for d in found[1:] if (d / "grids.py").read_bytes() != body]
        if bad:
            raise SystemExit(
                "FATAL: two reachable grids.py disagree:\n"
                + "\n".join(f"    {d / 'grids.py'}" for d in [found[0]] + bad)
                + "\n  The vendored copy and the grib-index-kerchunk original must\n"
                  "  stay byte-identical. Re-copy from the original, or set\n"
                  "  GIK_ECMWF_DIR to the one you mean.")
    return found[0] if found else None


GRIDS_DIR = _locate_grids()
if GRIDS_DIR is not None:
    sys.path.insert(0, str(GRIDS_DIR))
    from grids import ERAS, latitudes, longitudes  # noqa: E402
else:
    ERAS = {}


def coord_problems(g, era: str) -> list[str]:
    """Ways this group's spatial axes disagree with grids.py. Empty == correct."""
    bad = []
    for name, want in (("latitude", latitudes(era)),
                       ("longitude", longitudes(era))):
        if name not in g:
            bad.append(f"{name} missing")
            continue
        have = np.asarray(g[name][:])
        if have.shape != want.shape:
            bad.append(f"{name} {have.shape[0]} pts, expected {want.shape[0]}")
        elif not np.allclose(have, want, atol=1e-9):
            bad.append(f"{name} {have[0]:+.4f}..{have[-1]:+.4f}, expected "
                       f"{want[0]:+.4f}..{want[-1]:+.4f} "
                       f"(offset {float(have[0] - want[0]):+.2f} deg)")
    return bad


def check(name, ok, detail="", warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        (WARNINGS if warn else FAILURES).append(name)


def resolve_storage(store, sa_key):
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        return icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    return icechunk.local_filesystem_storage(store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens")
    ap.add_argument("--container", default="s3://ecmwf-forecasts/")
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--expected-dates", type=int, default=1256,
                    help="total catalog dates, for progress/ETA")
    ap.add_argument("--decode", action="store_true",
                    help="spot-decode one chunk of the latest date")
    ap.add_argument("--log", default=None, help="backfill log to scan for FAILED")
    args = ap.parse_args()

    print(f"== H1 repository: {args.store} ==")
    auth = icechunk.containers_credentials(
        {args.container: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(resolve_storage(args.store, args.sa_key),
                                    authorize_virtual_chunk_access=auth)
    anc = list(repo.ancestry(branch="main"))
    check("repo opens, main branch resolvable", True,
          f"{len(anc)} commits, tip {anc[0].id[:12]}..")
    print(f"       tip: '{anc[0].message.strip()}' at {anc[0].written_at}")
    ts = [a.written_at for a in anc[:20]]
    if len(ts) >= 3:
        dt = (ts[0] - ts[-1]).total_seconds() / (len(ts) - 1)
        age = (datetime.now(timezone.utc) - ts[0]).total_seconds()
        check("commits advancing", age < max(4 * dt, 600),
              f"last commit {age:.0f}s ago; recent rate 1/{dt:.0f}s", warn=True)
    else:
        dt = None

    print("== H2/H3 groups (zarr-level, robust to torn groups) ==")
    # An ECMWF store must have checkable axes; a GEFS one has no era groups.
    is_ecmwf = args.container.startswith("s3://ecmwf-forecasts")
    ro = repo.readonly_session("main").store
    root = zarr.open_group(store=ro, mode="r", zarr_format=3)
    coord_names = {"time", "number", "step", "isobaricInhPa",
                   "latitude", "longitude"}
    groups = []
    for era_name, era_grp in root.groups():
        groups += [f"{era_name}/{run}" for run, _ in era_grp.groups()]
    total_dates = 0
    latest = None  # (group, zarr_group, last_hours)
    for grp in sorted(groups):
        g = zarr.open_group(store=ro, path=grp, mode="r", zarr_format=3)
        hours = g["time"][:]
        n = len(hours)
        total_dates += n
        mono = bool(np.all(np.diff(hours) > 0)) if n > 1 else True
        uniq = len(np.unique(hours)) == n
        d0 = (EPOCH + pd.Timedelta(hours=int(hours[0]))).strftime("%Y%m%d") if n else "-"
        d1 = (EPOCH + pd.Timedelta(hours=int(hours.max()))).strftime("%Y%m%d") if n else "-"
        check(f"{grp} time axis unique", uniq, f"{n} dates: {d0}..{d1}")
        if not mono:  # unsorted = legitimate gap fill; readers sortby("time")
            check(f"{grp} time axis sorted", False,
                  "gap-filled out of order -- consumers should sortby('time')",
                  warn=True)
        data_arrays = [k for k in g.array_keys() if k not in coord_names]
        bad_shape = {k: g[k].shape[0] for k in data_arrays
                     if g[k].shape[0] != n}
        check(f"{grp} all {len(data_arrays)} arrays sized to time={n}",
              not bad_shape,
              f"lagging (self-heals on next append): {bad_shape}" if bad_shape else "")

        # H3b -- where the field actually is. This is a FAIL, not a WARN: a
        # mislabelled axis is not a lag that self-heals, it is every date in the
        # group pointing at the wrong part of the world, and it cannot be
        # repaired in place (see check_coords in build_ecmwf_icechunk.py).
        era = grp.split("/")[0]
        if GRIDS_DIR is None:
            if is_ecmwf:   # silent for a GEFS store: nothing to check it against
                check(f"{grp} lat/lon axes match grids.py", False,
                      "cannot find grib-index-kerchunk/ecmwf/grids.py -- set "
                      "GIK_ECMWF_DIR; an unverifiable store is not a healthy one")
        elif era in ERAS:
            bad_axes = coord_problems(g, era)
            lon = longitudes(era)
            check(f"{grp} lat/lon axes match grids.py", not bad_axes,
                  "; ".join(bad_axes) if bad_axes else
                  f"lon {lon[0]:+.2f}..{lon[-1]:+.2f}, {len(lon)} pts")
        elif is_ecmwf:
            check(f"{grp} era known to grids.py", False,
                  f"{era!r} is not in ERAS ({', '.join(sorted(ERAS))}) -- axes "
                  f"unchecked", warn=True)

        if n and (latest is None or hours[-1] > latest[2]):
            latest = (grp, g, int(hours[-1]))
    pct = 100 * total_dates / args.expected_dates
    eta = ((args.expected_dates - total_dates) * dt / 3600) if dt else float("nan")
    check("progress", True,
          f"{total_dates}/{args.expected_dates} dates ({pct:.1f}%), "
          f"ETA ~{eta:.1f} h at current rate")

    if args.decode and latest:
        grp, g, h = latest
        print(f"== H4 spot decode: newest date in {grp} ==")
        var = "t2m" if "t2m" in g else next(
            k for k in g.array_keys() if k not in coord_names)
        arr = g[var]
        t0 = time.time()
        idx = (arr.shape[0] - 1,) + (0,) * (arr.ndim - 3)  # last date, first of the rest
        field = arr[idx]
        finite = np.isfinite(field).mean()
        check(f"{var} @ {(EPOCH + pd.Timedelta(hours=h)).date()} decodes",
              finite > 0.99, f"finite={finite:.3f}, mean={np.nanmean(field):.2f} "
              f"({time.time()-t0:.1f}s)")

    if args.log:
        print(f"== H5 backfill log: {args.log} ==")
        try:
            txt = open(args.log).read()
            fails = [l for l in txt.splitlines() if "FAILED" in l]
            check("no FAILED dates in log", not fails,
                  f"{len(fails)} failed, last: {fails[-1][:90]}" if fails else "",
                  warn=True)
            alive = subprocess.run(["pgrep", "-f", "backfill_"],
                                   capture_output=True, text=True).stdout.strip()
            check("backfill process alive", bool(alive),
                  f"pids: {alive.split()[:3]}" if alive else "not running", warn=True)
        except FileNotFoundError:
            check("log file readable", False, args.log, warn=True)

    n_warn = len(WARNINGS)
    print(f"\n{'HEALTH OK' if not FAILURES else 'UNHEALTHY: ' + ', '.join(FAILURES)}"
          + (f" ({n_warn} warnings)" if n_warn else ""))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
