#!/usr/bin/env python
"""Open a realized store on EWC Ceph and print what is in it.

Credentials come from a `.env` via python-dotenv, so nothing is hardcoded and
nothing is exported into the shell:

    AK=<ewc access key>
    SK=<ewc secret key>

    $P inspect_store.py                                   # default store
    $P inspect_store.py --prefix ea-swio/v1-49r1-mar-may2026 --env .env
    $P inspect_store.py --var tp --step 240               # look at one field

Read-only throughout -- `readonly_session` cannot write even by accident.
"""
import argparse
import os

DEFAULT_PREFIX = "ea-swio/v1-49r1-mar-may2026"
ENDPOINT = "https://object-store.os-api.cci1.ecmwf.int"


def open_store(prefix, env_path, endpoint=ENDPOINT):
    """Return an xarray Dataset for must-icechunk/<prefix>."""
    import icechunk as ic
    import xarray as xr
    from dotenv import load_dotenv

    # override=True so a stale AK/SK already in the shell cannot win over the
    # file you just edited.  Silent precedence is the usual way this wastes an
    # afternoon.
    load_dotenv(env_path, override=True)
    ak, sk = os.environ.get("AK"), os.environ.get("SK")
    if not (ak and sk):
        raise SystemExit(f"AK/SK not found in {env_path}")

    storage = ic.s3_storage(
        bucket="must-icechunk", prefix=prefix, region="RegionOne",
        endpoint_url=endpoint, access_key_id=ak, secret_access_key=sk,
        force_path_style=True, from_env=False,
    )
    repo = ic.Repository.open(storage)
    return repo, xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False, zarr_format=3, decode_timedelta=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--env", default=".env")
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--var", default="tp", help="channel to sample")
    p.add_argument("--step", type=int, default=240, help="lead time, hours")
    args = p.parse_args()

    import numpy as np

    repo, ds = open_store(args.prefix, args.env, args.endpoint)

    snaps = list(repo.ancestry(branch="main"))
    print(f"must-icechunk/{args.prefix}")
    print(f"  {len(snaps)} snapshots, HEAD {snaps[0].id}")
    print(f"  latest   {snaps[0].message!r}")
    print(f"  written  {snaps[0].written_at:%Y-%m-%d %H:%M}")

    print(f"\n  dims     {dict(ds.sizes)}")
    print(f"  channels {len(ds.data_vars)}")
    print(f"  time     {np.datetime_as_string(ds.time.values, 'D').tolist()}")
    print(f"  step     +{int(ds.step.values[0] / np.timedelta64(1, 'h'))}h .. "
          f"+{int(ds.step.values[-1] / np.timedelta64(1, 'h'))}h "
          f"({ds.sizes['step']} steps)")
    print(f"  number   {int(ds.number.values.min())} .. "
          f"{int(ds.number.values.max())}  (0 = control)")
    print(f"  lat      {float(ds.latitude.values[0])} -> "
          f"{float(ds.latitude.values[-1])}")
    print(f"  lon      {float(ds.longitude.values[0])} -> "
          f"{float(ds.longitude.values[-1])}")

    # --- sample one field -------------------------------------------------
    da = ds[args.var].isel(time=0).sel(step=np.timedelta64(args.step, "h"))
    a = da.values                                   # (number, lat, lon)
    print(f"\n  {args.var} at +{args.step}h  shape {a.shape}")
    print(f"    finite      {float(np.isfinite(a).mean()):.4f}")
    print(f"    ens mean    {float(np.nanmean(a)):.6g}")
    print(f"    ens spread  {float(np.nanstd(a, axis=0).mean()):.6g}  "
          f"(sd over `number`, averaged over the box)")
    print(f"    control     {float(np.nanmean(a[0])):.6g}   "
          f"perturbed {float(np.nanmean(a[1:])):.6g}")
    print(f"    min / max   {float(np.nanmin(a)):.6g} / "
          f"{float(np.nanmax(a)):.6g}")

    # Probability of exceedance across the ensemble -- the shape the flood
    # scenario actually consumes.  Only meaningful for precipitation.
    if args.var == "tp":
        for thr in (0.020, 0.050, 0.100):            # metres
            frac = float((a > thr).mean())
            print(f"    P(tp > {thr * 1000:>3.0f} mm) = {frac:.4f} "
                  f"(over members x grid)")


if __name__ == "__main__":
    main()
