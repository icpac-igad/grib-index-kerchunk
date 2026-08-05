"""Completeness from the MANIFEST, without reading a byte of data.

`check_complete.py` reads every block to find all-zero ones. That is the right
check on one date (1,590 blocks, 7.8 GB) and hopeless on a corpus: 30 dates is
47,700 blocks and 233 GB of reads to answer a yes/no question.

`Session.chunk_coordinates` lists the chunks that were actually written, out of
the manifest. A chunk that was never written simply is not there -- which is
exactly the failure `create_schema`'s zero-fill would otherwise disguise.

Costs a manifest read (~17 MB for June 2026) instead of the whole store.

Usage
-----
    .venv/bin/python check_chunks.py --prefix ea-cgan/v3-june2026
"""
from __future__ import annotations

import argparse
import asyncio
import os

EP_DEFAULT = "https://object-store.os-api.cci1.ecmwf.int"


async def coords_for(session, path):
    return {c async for c in session.chunk_coordinates(f"/{path}")}


async def main_async(args):
    import icechunk as ic
    import xarray as xr

    creds = {}
    with open(args.env) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                creds[k.strip()] = v.strip()

    st = ic.s3_storage(
        bucket="must-icechunk", prefix=args.prefix, region="RegionOne",
        endpoint_url=args.endpoint, access_key_id=creds["AK"],
        secret_access_key=creds["SK"], force_path_style=True, from_env=False)
    repo = ic.Repository.open(st)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

    n_t, n_s = ds.sizes["time"], ds.sizes["step"]
    expect = {(t, s, 0, 0, 0) for t in range(n_t) for s in range(n_s)}
    print(f"must-icechunk/{args.prefix}")
    print(f"  {len(ds.data_vars)} channels x {n_t} time x {n_s} steps")
    print(f"  {len(ds.data_vars) * n_t * n_s:,} chunks expected\n")

    total_missing, per_var = 0, []
    for v in sorted(ds.data_vars):
        got = await coords_for(session, v)
        missing = expect - got
        total_missing += len(missing)
        per_var.append((v, len(got), sorted(missing)[:4]))

    for v, n, miss in per_var:
        flag = "" if not miss else f"   MISSING {miss}"
        print(f"  {v:7s} {n:6,d} chunks{flag}")

    print(f"\n  total missing: {total_missing}")
    print(f"  VERDICT: {'complete' if not total_missing else 'INCOMPLETE'}")
    return 0 if not total_missing else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--env", default=".env")
    p.add_argument("--endpoint", default=EP_DEFAULT)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
