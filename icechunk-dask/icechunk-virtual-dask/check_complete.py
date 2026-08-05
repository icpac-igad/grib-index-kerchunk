"""Did every chunk actually get written?

`0 failed blocks` is not proof. `create_schema` pre-fills the arrays with
ZEROS, so a write that never happened is indistinguishable from data unless
you go looking: no error is raised, the shape is right, the dtype is right,
and the finite fraction is 1.000. This sweeps every (channel, step) block and
reports the ones that are entirely zero.

The one subtlety, and it bit on the first run: **accumulated fields are
legitimately zero at step 0**. `tp`, `ssr` and `ttr` accumulate from the
forecast start, so at +0h there is nothing to have accumulated. Flagging those
as missing writes is a false alarm -- the cross-check that settled it was that
two independently written stores (per-block forks and a single shared fork)
produced exactly the same three zero blocks.

Usage
-----
    .venv/bin/python check_complete.py --prefix ea-cgan/v2-7day
"""
from __future__ import annotations

import argparse
import os

EP_DEFAULT = "https://object-store.os-api.cci1.ecmwf.int"

# Zero at step 0 is correct for these -- they accumulate from the start.
ACCUMULATED = {"tp", "ssr", "ttr"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--env", default=".env")
    p.add_argument("--endpoint", default=EP_DEFAULT)
    args = p.parse_args()

    import numpy as np
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
    ds = xr.open_zarr(ic.Repository.open(st).readonly_session("main").store,
                      consolidated=False, zarr_format=3, decode_timedelta=True)

    n_t, n_s = ds.sizes["time"], ds.sizes["step"]
    total = len(ds.data_vars) * n_t * n_s
    print(f"must-icechunk/{args.prefix}")
    print(f"  {len(ds.data_vars)} channels x {n_t} time x {n_s} steps "
          f"= {total} blocks\n")

    missing, expected, nonfinite = [], [], []
    for v in ds.data_vars:
        for ti in range(n_t):
            a = ds[v].isel(time=ti).values
            for s in range(n_s):
                blk = a[s]
                if not np.any(blk):
                    (expected if (v in ACCUMULATED and s == 0)
                     else missing).append((v, ti, s))
                if not np.isfinite(blk).all():
                    nonfinite.append((v, ti, s))

    print(f"  all-zero, expected  {len(expected):5d}   "
          f"{sorted({v for v, _, _ in expected})}")
    print(f"  all-zero, MISSING   {len(missing):5d}   {missing[:8]}")
    print(f"  non-finite          {len(nonfinite):5d}   {nonfinite[:8]}")
    ok = not missing and not nonfinite
    print(f"\n  VERDICT: {'every block written' if ok else 'INCOMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
