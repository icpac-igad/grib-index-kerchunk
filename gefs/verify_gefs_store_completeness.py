# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "pandas", "gribberish>=1.4", "gcsfs", "numpy",
# ]
# ///
"""Prove the GEFS Icechunk store holds every chunk the archive can supply.

The GEFS counterpart of ecmwf/icechunk-par/verify_store_completeness.py, and the
check that was missing when three all-NaN holes sat in the store for ~3 years:

    20231010 gep07 f201    20231013 gep10 f030    20231105 gep12 f069

`check_gefs_store_health.py` did not see them, and could not: its H4 compares
DATE coverage against the GCS par tree, and all three dates were present. The
loss was one member at one step inside an otherwise complete date. Date coverage
is the wrong granularity for that class; these checks work below it.

Three modes, cheapest first
---------------------------
    refs     commit log -> refs per date, grouped into contiguous eras.
             Seconds, no data read, covers the WHOLE corpus. This is the one
             that found the three holes -- run it in CI.
    dates    date coverage vs the authoritative GCS par tree (~60 s).
    chunks   exact manifest chunk census per (array, time) via
             session.chunk_coordinates -- no data fetched, but ~9 min PER ARRAY
             over the full corpus. The deep check; use after `refs` flags.

    uv run verify_gefs_store_completeness.py refs
    uv run verify_gefs_store_completeness.py dates
    uv run verify_gefs_store_completeness.py chunks --var t2m

Exit status is 0 only if every check passed, so this drops into CI.

Why `refs` works
----------------
build_gefs_icechunk.py asserts `not bad` on every set_virtual_refs and then
records the count in the commit message, so the committed number IS the number
of references written -- no sampling, no estimate. Within an era every date must
carry the identical count; NOAA's variable set only ever changes on a boundary,
and those boundaries are contiguous date ranges. A date below its era's plateau
lost references, full stop. Repair commits are summed into their date, so a
repaired date reads as whole.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import icechunk
import zarr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

GROUP = "0p25/00z"
N_MEMBERS, N_STEPS = 30, 81
GCS_PARS = "gik-gefs-aws-tf/run_par_gefs"
DEFAULT_STORE = "gs://gik-gefs-aws-tf/icechunk/gefs-ens"
CONTAINER_PREFIX = "s3://noaa-gefs-pds/"

BUILD_RE = re.compile(r"0p25/00z (\d{8}): (\d+) members, (\d+) refs")
REPAIR_RE = re.compile(r"0p25/00z (\d{8}): repair \S+ \S+ -- (\d+) refs restored")


def open_repo(a):
    bucket, _, prefix = a.store[5:].partition("/")
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix.rstrip("/"),
                                   service_account_file=a.sa_key)
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    return icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)


# --------------------------------------------------------------------------
def cmd_refs(a) -> bool:
    """Refs per date from the commit log, against each era's plateau."""
    repo = open_repo(a)
    per = collections.Counter()
    seen = collections.Counter()
    other = 0
    for s in repo.ancestry(branch="main"):
        msg = s.message or ""
        m = REPAIR_RE.search(msg)
        if m:
            per[m.group(1)] += int(m.group(2))
            continue
        m = BUILD_RE.search(msg)
        if m:
            per[m.group(1)] += int(m.group(3))
            seen[m.group(1)] += 1
        else:
            other += 1
    rows = sorted(per.items())
    print(f"{len(rows)} dates from the commit log ({other} non-date commits)")

    dup = {d: n for d, n in seen.items() if n > 1}
    if dup:
        print(f"  WARN dates built more than once: {dup}")

    # contiguous runs of equal totals = NOAA variable-set eras
    runs, start, cur, prev = [], rows[0][0], rows[0][1], rows[0][0]
    for d, n in rows[1:]:
        if n != cur:
            runs.append((start, prev, cur))
            start, cur = d, n
        prev = d
    runs.append((start, prev, cur))

    # an era is a run spanning many dates; short runs are suspects
    plateau = collections.Counter(n for _, _, n in runs for _ in [0])
    big = sorted({n for a_, b_, n in runs
                  if len([d for d, _ in rows if a_ <= d <= b_]) >= 30})
    print(f"\nrefs-per-date plateaus (>=30 dates): {sorted(big)}")
    print("\ncontiguous runs:")
    bad = []
    for a_, b_, n in runs:
        cnt = len([d for d, _ in rows if a_ <= d <= b_])
        tag = ""
        if n not in big:
            # below the nearest plateau => lost refs
            above = [p for p in big if p > n]
            if above:
                tag = f"   <-- SHORT by {min(above) - n} refs"
                bad += [d for d, v in rows if a_ <= d <= b_ and v == n]
            else:
                tag = "   <-- unexpected plateau"
        span = f"{a_} .. {b_}" if a_ != b_ else a_
        print(f"  {span:<22} {n:>7} refs  {cnt:>5} dates{tag}")

    if bad:
        print(f"\nFAIL: {len(bad)} date(s) short of their era plateau: {bad}")
        print("  -> classify with `upstream`, repair with repair_gefs_holes.py")
        return False
    print(f"\nPASS: every date sits on an era plateau "
          f"({sum(per.values()):,} refs total)")
    return True


# --------------------------------------------------------------------------
def cmd_dates(a) -> bool:
    """Store time axis vs the authoritative GCS par tree."""
    import gcsfs
    repo = open_repo(a)
    g = zarr.open_group(store=repo.readonly_session("main").store, path=GROUP,
                        mode="r", zarr_format=3)
    have = {datetime.fromtimestamp(int(t) * 3600, tz=timezone.utc).strftime("%Y%m%d")
            for t in np.asarray(g["time"][:])}

    fs = gcsfs.GCSFileSystem(token=a.sa_key)
    pat = re.compile(r"run_par_gefs/\d{4}/\d{2}/(\d{8})/(\d{2})z/")
    par = collections.defaultdict(set)
    for o in fs.find(GCS_PARS):
        m = pat.search(o)
        if m:
            par[m.group(2)].add(m.group(1))

    p00 = par.get("00", set())
    missing = sorted(p00 - have)
    extra = sorted(have - p00)
    print(f"par tree 00z dates : {len(p00):,}")
    print(f"store time axis    : {len(have):,}")
    print(f"in par, NOT stored : {len(missing)}  {missing[:10]}")
    print(f"stored, NOT in par : {len(extra)}  {extra[:10]}")
    for run in sorted(k for k in par if k != "00"):
        print(f"  note: {len(par[run]):>4} {run}z par dates exist and are NOT "
              f"ingested (store is a single 00z group)")
    ok = not missing and not extra
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
def cmd_chunks(a) -> bool:
    """Exact chunk census per (array, time) from the manifest -- no data read."""
    repo = open_repo(a)
    sess = repo.readonly_session("main")
    g = zarr.open_group(store=sess.store, path=GROUP, mode="r", zarr_format=3)
    coords = {"time", "number", "step", "latitude", "longitude"}
    arrays = ([a.var] if a.var != "all"
              else sorted(k for k in g.array_keys() if k not in coords))
    ntime = g["time"].shape[0]

    async def census(path):
        per = collections.Counter()
        steps_seen = collections.defaultdict(set)
        async for c in sess.chunk_coordinates(f"/{GROUP}/{path}"):
            per[c[0]] += 1
            steps_seen[c[0]].add(c[2])
        return per, steps_seen

    ok = True
    for name in arrays:
        per, steps_seen = asyncio.run(census(name))
        counts = collections.Counter(per.values())
        # a var absent at a date has 0 chunks (NOAA had not introduced it);
        # accumulated vars publish no f000, so they cap at 30*80
        full, accum = N_MEMBERS * N_STEPS, N_MEMBERS * (N_STEPS - 1)
        legit = {0, full, accum}
        bad = {t: n for t, n in per.items() if n not in legit}
        for t in range(ntime):
            if t not in per:
                bad.setdefault(t, 0)
                del bad[t]                      # 0 is legitimate; keep explicit
        print(f"\n{name}: {sum(per.values()):,} chunks over {len(per)}/{ntime} "
              f"time indices")
        print(f"   counts seen: {dict(sorted(counts.items()))}")
        if bad:
            ok = False
            print(f"   FAIL partial time indices ({len(bad)}):")
            for t, n in sorted(bad.items())[:15]:
                miss = full - n
                print(f"     time idx {t}: {n} chunks ({miss} missing)")
        else:
            print("   PASS every time index is empty, accumulated-full, or full")
    return ok


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--sa-key",
                    default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("refs", help="refs per date from the commit log (seconds)")
    p.set_defaults(fn=cmd_refs)

    p = sub.add_parser("dates", help="date coverage vs the GCS par tree (~60 s)")
    p.set_defaults(fn=cmd_dates)

    p = sub.add_parser("chunks", help="exact chunk census (~9 min per array)")
    p.add_argument("--var", default="t2m", help="array name, or 'all'")
    p.set_defaults(fn=cmd_chunks)

    a = ap.parse_args()
    ok = a.fn(a)
    print(f"\n{'=' * 62}\n{a.cmd}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
