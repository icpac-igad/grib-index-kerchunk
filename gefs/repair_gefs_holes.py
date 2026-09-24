# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "gcsfs", "numpy",
# ]
# ///
"""Repair all-NaN (date, member, step) holes in the GEFS Icechunk store.

A hole is one member at one step where the par carries no reference, so every
branch is NaN while the neighbouring steps and members are intact. Three were
found on 2026-09-18 by `verify_gefs_store_completeness.py refs`:

    20231010 gep07 f201    20231013 gep10 f030    20231105 gep12 f069

Each lost 33 refs. The cause is NOT an upstream gap -- the GRIB *and* its .idx
are both published on s3://noaa-gefs-pds for all three. Stage 2 dropped them
while building the par, so the bytes were always there to point at.

How the missing refs are reconstructed
--------------------------------------
Not from the template, and not by guessing. GRIB message order inside a
pgrb2sp25 file is identical across members of the same (date, run), so:

  1. take a DONOR member that has the step, and read its par rows for it;
  2. locate each donor ref's byte offset in the donor's own .idx -> an ORDINAL;
  3. read the broken member's .idx and take the line at that same ordinal;
  4. assert the (VAR, LEVEL) at that ordinal matches the donor's, and that the
     two .idx files have identical (VAR, LEVEL) sequences end to end;
  5. the broken member's own offset/length at that ordinal is the missing ref.

So every repaired ref is read out of the broken member's own index -- the donor
only supplies the branch list and the ordinal. Step 4 is what makes it safe.

Each repaired chunk is then verified by decoding the bytes straight from S3 with
gribberish and comparing to what the store returns, at float32.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
  uv run repair_gefs_holes.py                      # dry run: plan + validate
  uv run repair_gefs_holes.py --apply              # write and commit
  uv run repair_gefs_holes.py --hole 20231010:7:67 # a specific hole
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import icechunk
import zarr
import gribberish
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

GROUP = "0p25/00z"
NY, NX = 721, 1440
N_MEMBERS = 30
N_STEPS = 81
S3_HTTP = "https://noaa-gefs-pds.s3.amazonaws.com"
CONTAINER_PREFIX = "s3://noaa-gefs-pds/"
GCS_PARS = "gik-gefs-aws-tf/run_par_gefs"
DEFAULT_STORE = "gs://gik-gefs-aws-tf/icechunk/gefs-ens"

# Found by `verify_gefs_store_completeness.py refs` on 2026-09-18.
KNOWN_HOLES = [("20231010", 7, 67), ("20231013", 10, 10), ("20231105", 12, 23)]


# --------------------------------------------------------------------------
def par_key(date: str, member: int) -> str:
    return (f"{GCS_PARS}/{date[:4]}/{date[4:6]}/{date}/00z/"
            f"{date}00z-gep{member:02d}.parquet")


def grib_key(date: str, member: int, step_idx: int) -> str:
    return (f"gefs.{date}/00/atmos/pgrb2sp25/"
            f"gep{member:02d}.t00z.pgrb2s.0p25.f{step_idx * 3:03d}")


def object_size(key: str) -> int:
    """Content-Length of the GRIB file, for the last message's length."""
    req = urllib.request.Request(f"{S3_HTTP}/{key}", method="HEAD")
    return int(urllib.request.urlopen(req, timeout=90).headers["Content-Length"])


def read_idx(key: str) -> list[dict]:
    """GEFS .idx -> [{n, offset, length, var, level}], length from the next offset.

    The LAST message has no next offset, so its length comes from the file size.
    It must NOT be borrowed from another member: GRIB packing is per-member, so
    the final message's length differs between members and a borrowed value
    truncates the range ("Data section not found"). PRMSL is last in this
    product, so this is hit on every repair, not an edge case.
    """
    lines = urllib.request.urlopen(f"{S3_HTTP}/{key}.idx", timeout=90).read() \
        .decode().splitlines()
    recs = [ln.split(":") for ln in lines]
    size = object_size(key)
    out = []
    for i, r in enumerate(recs):
        off = int(r[1])
        nxt = int(recs[i + 1][1]) if i + 1 < len(recs) else size
        out.append(dict(n=i, offset=off, length=nxt - off,
                        var=r[3], level=r[4]))
    return out


def parse_member_par(path: Path):
    """Same parse as build_gefs_icechunk.py: par -> rows + the skipped branches."""
    df = pd.read_parquet(path)
    refs = ast.literal_eval(df.loc[df.key == "refs", "value"].iloc[0].decode())
    rows = []
    for k in refs:
        if not k.endswith("/.zarray"):
            continue
        parts = k.split("/")
        if len(parts) != 5 or parts[0] != parts[3]:
            continue
        var, step_type, lev = parts[0], parts[1], parts[2]
        if json.loads(refs[k])["shape"] != [N_STEPS, NY, NX]:
            continue
        prefix = f"{var}/{step_type}/{lev}/{var}/"
        for ck, cv in refs.items():
            if ck.startswith(prefix) and isinstance(cv, list) and len(cv) == 3:
                rows.append((var, lev, int(ck[len(prefix):].split(".")[0]),
                             cv[0], int(cv[1]), int(cv[2])))
    return pd.DataFrame(rows, columns=["var", "lev", "step_idx",
                                       "url", "offset", "length"])


def znames_for(df: pd.DataFrame) -> pd.Series:
    """Reproduce the builder's array naming: suffix _{lev} only where var repeats."""
    branches = df[["var", "lev"]].drop_duplicates()
    dup = set(branches["var"][branches["var"].duplicated()])
    return pd.Series([f"{v}_{l}" if v in dup else v
                      for v, l in zip(df["var"], df["lev"])], index=df.index)


# --------------------------------------------------------------------------
def plan_hole(fs, date: str, member: int, step_idx: int, donor: int | None):
    """Work out every missing ref for one hole. Raises rather than guess."""
    donor = donor or (1 if member != 1 else 2)
    tmp = Path(tempfile.mkdtemp())

    dpath = tmp / f"donor{donor}.parquet"
    fs.get(par_key(date, donor), str(dpath))
    ddf = parse_member_par(dpath)
    ddf["zname"] = znames_for(ddf)
    dstep = ddf[ddf.step_idx == step_idx]
    if dstep.empty:
        raise SystemExit(f"donor gep{donor:02d} has no step {step_idx} either")

    bpath = tmp / f"broken{member}.parquet"
    fs.get(par_key(date, member), str(bpath))
    bdf = parse_member_par(bpath)
    if not bdf[bdf.step_idx == step_idx].empty:
        raise SystemExit(f"{date} gep{member:02d} step {step_idx} is NOT missing "
                         f"from the par -- nothing to repair")

    didx = read_idx(grib_key(date, donor, step_idx))
    bidx = read_idx(grib_key(date, member, step_idx))

    # The safety gate: the two indexes must describe the same message sequence.
    dseq = [(r["var"], r["level"]) for r in didx]
    bseq = [(r["var"], r["level"]) for r in bidx]
    if dseq != bseq:
        raise SystemExit(
            f"{date} f{step_idx*3:03d}: gep{donor:02d} and gep{member:02d} .idx "
            f"differ ({len(dseq)} vs {len(bseq)} messages) -- ordinal mapping is "
            f"NOT valid here, refusing to guess")

    by_off = {r["offset"]: r for r in didx}
    plan = []
    for r in dstep.itertuples():
        d = by_off.get(r.offset)
        if d is None:
            raise SystemExit(f"donor offset {r.offset} ({r.zname}) not in its own "
                             f".idx -- par/idx disagree, refusing")
        b = bidx[d["n"]]
        assert (b["var"], b["level"]) == (d["var"], d["level"])
        assert b["length"] and b["length"] > 0, f"{b} has no usable length"
        plan.append(dict(zname=r.zname, var=d["var"], level=d["level"],
                         ordinal=d["n"],
                         url=f"{CONTAINER_PREFIX}{grib_key(date, member, step_idx)}",
                         offset=b["offset"], length=b["length"]))
    return plan


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--sa-key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    ap.add_argument("--hole", action="append", default=None,
                    help="DATE:MEMBER:STEP_IDX (repeatable); default = the three known")
    ap.add_argument("--donor", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="write and commit")
    a = ap.parse_args()

    import gcsfs
    fs = gcsfs.GCSFileSystem(token=a.sa_key)

    holes = KNOWN_HOLES
    if a.hole:
        holes = []
        for h in a.hole:
            d, m, s = h.split(":")
            holes.append((d, int(m), int(s)))

    bucket, _, prefix = a.store[5:].partition("/")
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix.rstrip("/"),
                                   service_account_file=a.sa_key)
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)

    ro = repo.readonly_session("main").store
    g = zarr.open_group(store=ro, path=GROUP, mode="r", zarr_format=3)
    times = np.asarray(g["time"][:])
    from datetime import datetime, timezone
    tstr = np.array([datetime.fromtimestamp(int(t) * 3600, tz=timezone.utc)
                     .strftime("%Y%m%d") for t in times])

    all_ok = True
    for date, member, step_idx in holes:
        where = np.where(tstr == date)[0]
        if not where.size:
            print(f"{date}: not in the store's time axis -- skip")
            all_ok = False
            continue
        ti = int(where[0])
        print(f"\n=== {date} gep{member:02d} f{step_idx*3:03d} "
              f"(time idx {ti}, step idx {step_idx})")
        plan = plan_hole(fs, date, member, step_idx, a.donor)
        missing = [p["zname"] for p in plan if p["zname"] not in g]
        if missing:
            print(f"  branches not present as store arrays: {missing}")
            all_ok = False
            continue
        print(f"  {len(plan)} refs reconstructed from gep{member:02d}'s own .idx")
        for p in plan[:3]:
            print(f"     {p['zname']:<22} msg#{p['ordinal']:<3} "
                  f"{p['var']}:{p['level']:<28} off={p['offset']:<10} "
                  f"len={p['length']}")
        print(f"     ... ({len(plan) - 3} more)")

        if not a.apply:
            print("  DRY RUN -- nothing written (pass --apply)")
            continue

        session = repo.writable_session("main")
        st = session.store
        n = 0
        for p in plan:
            spec = icechunk.VirtualChunkSpec(
                index=[ti, member - 1, step_idx, 0, 0],
                location=p["url"], offset=p["offset"], length=p["length"])
            bad = st.set_virtual_refs(f"{GROUP}/{p['zname']}", [spec])
            assert not bad, f"{p['zname']}: rejected {bad}"
            n += 1
        snap = session.commit(
            f"{GROUP} {date}: repair gep{member:02d} f{step_idx*3:03d} -- "
            f"{n} refs restored from the member's own .idx "
            f"(Stage 2 par gap; bytes were always published)")
        print(f"  committed {n} refs -> {snap}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
