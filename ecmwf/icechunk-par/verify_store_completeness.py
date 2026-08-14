# /// script
# requires-python = ">=3.11"
# dependencies = ["icechunk>=2.1", "zarr>=3.2", "numpy"]
# ///
"""Enumerate a group's manifest and name every (date, member, step) that is absent.

Why this exists: the store's step axis is a UNION across members, so if any one of
51 members carries a step, the group's axis looks complete and the build reports
success. A member whose par lost a step therefore lands as a silent NaN. Nothing
in the pipeline detects that -- check_chunks.py counts chunks written, not chunks
expected, and a per-date probe (build_ecmwf_icechunk.date_written) deliberately
stops at the first hit, so it cannot see a hole inside a date.

Do NOT try to infer this from par file sizes. That was tried and produced a
retracted 5,114-par figure: control is always the smallest par (its byte offsets
are the smallest numbers in the GRIB, so its JSON strings are shortest) and size
also tracks the pl-level regime. Size correlates with completeness AND with two
other things that dominate. The manifest is the ground truth.

One surface variable is enough. A par that lost a step lost that step for every
variable of that member, so `t2m` (or any var present on all dates) exposes the
same gaps as scanning all 59 channels, at 1/59 the cost.

    uv run verify_store_completeness.py --store gs://bucket/icechunk/ecmwf-ens-v4 \
        --era 49r1 --run 00 --sa-key /tmp/frisky-ea/gcs-key.json
    ... --var skt --max-report 50      # different probe var, longer listing

Exit status is 1 if any gap is found, so this can gate a publish.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys
from datetime import datetime, timedelta, timezone

import icechunk
import numpy as np
import zarr

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def open_group(store: str, grp: str, sa_key: str | None):
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        st = icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    else:
        st = icechunk.local_filesystem_storage(store)
    auth = icechunk.containers_credentials(
        {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(st, authorize_virtual_chunk_access=auth)
    s = repo.readonly_session("main")
    return s, zarr.open_group(store=s.store, path=grp, mode="r", zarr_format=3)


async def present_chunks(session, path: str) -> set:
    """Every initialised chunk coordinate for one array."""
    # chunk_coordinates yields the FULL coordinate -- (time, number, step, lat, lon)
    # for a surface array, 6-long for a pl one. The spatial dims are a single chunk
    # each, so only the leading three identify a (date, member, step). Keeping the
    # whole tuple makes every membership test fail and reports 100% missing while
    # simultaneously reporting every chunk present.
    out = set()
    async for c in session.chunk_coordinates(path):
        out.add(tuple(c)[:3])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--era", required=True)
    ap.add_argument("--run", default="00")
    ap.add_argument("--var", default="t2m",
                    help="surface variable to probe; must exist on every date")
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--max-report", type=int, default=25)
    a = ap.parse_args()

    grp = f"{a.era}/{a.run}z"
    session, g = open_group(a.store, grp, a.sa_key)
    if a.var not in g:
        print(f"{a.var!r} not in {grp}; have: {sorted(n for n in g.array_keys())[:12]} ...")
        return 2

    tvals = g["time"][:]
    steps = [int(s) for s in g["step"][:]]
    n_time, n_num, n_step = g[a.var].shape[:3]
    print(f"{a.store} {grp}  probe={a.var}")
    print(f"  {n_time} dates x {n_num} members x {n_step} steps = "
          f"{n_time * n_num * n_step:,} expected chunks")

    have = asyncio.run(present_chunks(session, f"/{grp}/{a.var}"))
    print(f"  {len(have):,} present in the manifest")

    # A date with no chunks at all is an unwritten slot, not a hole -- report
    # those separately, since they mean "not built yet" rather than "lost data".
    by_date = collections.Counter(c[0] for c in have)
    empty = [ti for ti in range(n_time) if by_date[ti] == 0]
    holes = collections.defaultdict(list)          # ti -> [(member, step_h)]
    for ti in range(n_time):
        if by_date[ti] == 0:
            continue
        for m in range(n_num):
            for si in range(n_step):
                if (ti, m, si) not in have:
                    holes[ti].append((m, steps[si]))

    def date_of(ti):
        return (EPOCH + timedelta(hours=int(tvals[ti]))).strftime("%Y-%m-%d")

    n_holes = sum(len(v) for v in holes.values())
    print(f"\n  unwritten dates      : {len(empty)}"
          + (f"  e.g. {', '.join(date_of(t) for t in empty[:5])}" if empty else ""))
    print(f"  dates with holes     : {len(holes)} of {n_time - len(empty)} written")
    print(f"  missing (member,step): {n_holes:,}"
          f"  ({100 * n_holes / max(1, n_time * n_num * n_step):.4f}% of chunks)")

    if holes:
        print(f"\n  {'date':12s} {'n':>4s}  members(steps)")
        for ti in sorted(holes)[:a.max_report]:
            v = holes[ti]
            mem = collections.defaultdict(list)
            for m, sh in v:
                mem[m].append(sh)
            desc = "; ".join(
                f"{'control' if m == 0 else f'ens_{m:02d}'}@{','.join(str(x) for x in sh[:6])}"
                f"{'...' if len(sh) > 6 else ''}"
                for m, sh in sorted(mem.items())[:4])
            print(f"  {date_of(ti):12s} {len(v):>4d}  {desc}"
                  f"{'  ...' if len(mem) > 4 else ''}")
        if len(holes) > a.max_report:
            print(f"  ... and {len(holes) - a.max_report} more dates")

    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main())
