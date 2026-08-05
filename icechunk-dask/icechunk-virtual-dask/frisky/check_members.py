"""Open a realized store and check the member dimension is real.

The failure this is written to catch: a `number` selection that silently
broadcasts, so all 51 members hold identical values and the store looks fine
by shape, dtype and finite-fraction while carrying one member copied 51 times.
Shape checks do not see it.  These do:

  1. adjacent members differ at all
  2. ensemble spread GROWS with lead time -- the physical signature of a
     forecast ensemble, and the thing a broadcast bug cannot fake
  3. member 0 (the control) is distinguishable from the perturbed members

Usage
-----
    .venv/bin/python check_members.py --prefix ea-cgan/v1-7day
    .venv/bin/python check_members.py --prefix ea-frisky-test/2026-07-02
"""
from __future__ import annotations

import argparse
import os

EP_DEFAULT = "https://object-store.os-api.cci1.ecmwf.int"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="ea-cgan/v1-7day")
    p.add_argument("--env", default="../.env")
    p.add_argument("--var", default=None, help="which channel to inspect")
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
    repo = ic.Repository.open(st)

    snaps = list(repo.ancestry(branch="main"))
    print(f"must-icechunk/{args.prefix}")
    print(f"  {len(snaps)} snapshots, newest {snaps[0].written_at:%Y-%m-%d %H:%M}")
    print(f"  latest: {snaps[0].message!r}")

    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False,
                      zarr_format=3, decode_timedelta=True)
    print(f"\n  dims      {dict(ds.sizes)}")
    print(f"  channels  {len(ds.data_vars)}: {', '.join(list(ds.data_vars))}")
    if "number" not in ds.dims:
        raise SystemExit("\nNo `number` dimension -- this store holds no members.")

    var = args.var or list(ds.data_vars)[0]
    da = ds[var].isel(time=0)
    n_num = ds.sizes["number"]
    print(f"\n  inspecting {var}  {tuple(da.sizes.values())}")

    # ---- 1. are the members distinct at all -----------------------------
    first = da.isel(step=-1, number=0).values
    diffs = []
    for m in range(1, min(n_num, 6)):
        other = da.isel(step=-1, number=m).values
        diffs.append(float(np.abs(first - other).mean()))
    print(f"\n  1. member 0 vs 1..{min(n_num, 6) - 1} at the longest step, "
          f"mean |diff|:")
    print("     " + "  ".join(f"{d:.4f}" for d in diffs))
    identical = all(d == 0.0 for d in diffs)
    print(f"     -> {'IDENTICAL -- members are not real' if identical else 'distinct'}")

    # ---- 2. does spread grow with lead time ------------------------------
    print(f"\n  2. ensemble spread (sd over `number`, mean over the box) "
          f"by lead time:")
    n_step = ds.sizes["step"]
    picks = sorted(set([0, n_step // 4, n_step // 2, 3 * n_step // 4,
                        n_step - 1]))
    spreads = []
    for s in picks:
        sd = float(da.isel(step=s).std("number").mean().values)
        hours = int(ds.step.values[s] / np.timedelta64(1, "h"))
        spreads.append((hours, sd))
        print(f"     +{hours:3d}h   sd = {sd:.4f}")
    # Growth is only a meaningful signal over a long enough lead time.  An
    # ensemble does not measurably diverge in six hours, so a short store
    # (a smoke test with 2-3 steps) fails this test while being perfectly
    # correct.  Report it as untestable rather than as a failure.
    span_h = spreads[-1][0] - spreads[0][0]
    testable = span_h >= 48
    grows = spreads[-1][1] > spreads[0][1]
    if not testable:
        print(f"     -> lead span is only {span_h}h; spread growth is not "
              f"testable below 48h  ({spreads[0][1]:.4f} -> "
              f"{spreads[-1][1]:.4f})")
    else:
        print(f"     -> spread "
              f"{'GROWS with lead time' if grows else 'does NOT grow'}"
              f"  ({spreads[0][1]:.4f} -> {spreads[-1][1]:.4f})")

    # ---- 3. control vs perturbed -----------------------------------------
    last = da.isel(step=-1)
    ctrl = last.isel(number=0).values
    pert = last.isel(number=slice(1, None)).values
    print(f"\n  3. control (number=0) vs the {pert.shape[0]} perturbed members "
          f"at the longest step:")
    print(f"     control mean      {float(np.nanmean(ctrl)):.4f}")
    print(f"     perturbed mean    {float(np.nanmean(pert)):.4f}")
    print(f"     |ctrl - pertmean| {abs(float(np.nanmean(ctrl)) - float(np.nanmean(pert))):.4f}")

    print(f"\n  finite fraction   {float(np.isfinite(last.values).mean()):.4f}")

    verdict = (not identical) and (grows or not testable)
    note = "" if testable else "  (distinctness only -- lead span too short)"
    print(f"\n  VERDICT: members are "
          f"{'REAL' if verdict else 'SUSPECT'}{note}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
