# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "s3fs",
# ]
# ///
"""Test routine for a GIK -> Icechunk virtual store (0p4 era).

Three test groups:
  T1 structure -- store opens natively with xr.open_zarr, expected dims/vars/coords
  T2 decode    -- sample fields decode through the gribberish codec to sane values
  T3 bit-exact -- N randomly sampled chunks match a direct S3 byte-range +
                  gribberish decode of the SAME ref taken from the source par
                  (ground truth independent of icechunk)

Works on both the single-member pilot store (dims step,lat,lon) and the
ensemble store (dims time,number,step,...): selectors adapt to the dims found.

Usage:
  uv run test_icechunk_read.py --store stores/ecmwf-0p4-ens \
      --par pars/20230627/2023062700z-ens_07.parquet --member 7 --time-index 0
  uv run test_icechunk_read.py --store stores/ecmwf-0p4-pilot \
      --par pars/20230627/2023062700z-control.parquet
Exit code 0 = all PASS, 1 = any FAIL.
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec
from gribberish import parse_grib_array

CONTAINER_PREFIX = "s3://ecmwf-forecasts/"
NY, NX = 451, 900
SFC_RENAME = {"2t": "t2m", "10u": "u10", "10v": "v10"}
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def parse_par(path):
    df = pd.read_parquet(path)
    df["v"] = df["value"].map(lambda b: b.decode() if isinstance(b, (bytes, bytearray)) else b)
    refs = df[df.key.str.startswith("step_")].copy()
    parts = refs.key.str.split("/")
    refs["step_h"] = parts.map(lambda p: int(p[0].split("_")[1]))
    refs["var"] = parts.map(lambda p: p[1])
    refs["levtype"] = parts.map(lambda p: p[2])
    refs["level"] = parts.map(lambda p: float(p[3]) if p[2] == "pl" else np.nan)
    loc = refs.v.map(json.loads)
    refs["url"] = loc.map(lambda x: x[0] if x[0].endswith(".grib2") else x[0] + ".grib2")
    refs["offset"] = loc.map(lambda x: x[1])
    refs["length"] = loc.map(lambda x: x[2])
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--par", help="source par for bit-exact ground truth (T3)")
    ap.add_argument("--member", type=int, default=0,
                    help="number-dim index matching --par (0=control)")
    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"== T1 structure: {args.store} ==")
    storage = icechunk.local_filesystem_storage(args.store)
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
    print(ds)

    check("opens with xr.open_zarr", True)
    check("19 data vars", len(ds.data_vars) == 19, f"got {len(ds.data_vars)}")
    for expect in ["t2m", "tp", "t", "gh"]:
        check(f"var {expect} present", expect in ds)
    check("85 steps", ds.sizes.get("step") == 85)
    check("9 levels", ds.sizes.get("isobaricInhPa") == 9)
    check("grid 451x900", ds.sizes.get("latitude") == NY and ds.sizes.get("longitude") == NX)
    ensemble = "number" in ds.dims
    if ensemble:
        check("51 members", ds.sizes["number"] == 51)
        check("time is datetime", np.issubdtype(ds.time.dtype, np.datetime64),
              str(ds.time.values[: min(3, ds.sizes['time'])]))
    lat = ds.latitude.values
    check("latitude 90..-90 monotonic", lat[0] == 90.0 and lat[-1] == -90.0
          and (np.diff(lat) < 0).all())

    print("== T2 decode (gribberish codec via icechunk virtual refs) ==")
    sel = {}
    if "time" in ds.dims:
        sel["time"] = args.time_index
    if ensemble:
        sel["number"] = args.member
    t1 = time.time()
    t2m = ds["t2m"].isel(step=0, **sel).values
    finite = np.isfinite(t2m).mean()
    check("t2m step0 decodes", finite > 0.99, f"finite={finite:.3f}")
    check("t2m plausible (180-340 K)", 180 < np.nanmean(t2m) < 340,
          f"mean={np.nanmean(t2m):.2f} K")
    t850 = ds["t"].isel(step=0, **sel).sel(isobaricInhPa=850).values
    check("t@850 plausible", 220 < np.nanmean(t850) < 320, f"mean={np.nanmean(t850):.2f} K")
    print(f"  (2 fields decoded in {time.time()-t1:.1f}s)")

    if args.par:
        print(f"== T3 bit-exact vs source par: {args.par} (member idx {args.member}) ==")
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        refs = parse_par(args.par)
        rng = random.Random(args.seed)
        samples = refs.sample(n=args.n_samples, random_state=rng.randint(0, 2**31))
        for r in samples.itertuples():
            zname = r.var if r.levtype == "pl" else SFC_RENAME.get(r.var, r.var)
            da = ds[zname].isel(step=list(ds.step.values).index(r.step_h), **sel)
            if r.levtype == "pl":
                da = da.sel(isobaricInhPa=r.level)
            got = da.values
            truth = parse_grib_array(fs.read_block(r.url, r.offset, r.length), 0, False)
            truth = truth.reshape(NY, NX).astype("float32")
            check(f"{zname} step={r.step_h}"
                  + (f" lev={r.level:.0f}" if r.levtype == "pl" else ""),
                  np.array_equal(got, truth, equal_nan=True),
                  f"mean={np.nanmean(got):.3f}")

    print(f"\n{'ALL TESTS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
