# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "s3fs",
# ]
# ///
"""Pilot: convert ONE GIK ECMWF 0p4-era par (single member) into an Icechunk
store with virtual chunk refs + the gribberish Zarr v3 codec, then prove the
native xarray read path with a bit-exact check against a direct S3 decode.

Store model (single date, single member):
  surface var  -> (step, latitude, longitude)                chunks (1, 451, 900)
  pressure var -> (step, isobaricInhPa, latitude, longitude) chunks (1, 1, 451, 900)

Usage:
  uv run pilot_0p4_to_icechunk.py --par pars/20230627/2023062700z-control.parquet \
      --store stores/ecmwf-0p4-pilot
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec
from gribberish.zarr.codec import GribberishCodec
from gribberish import parse_grib_array

CONTAINER_PREFIX = "s3://ecmwf-forecasts/"
NY, NX = 451, 900  # 0.4-deg beta grid
SFC_RENAME = {"2t": "t2m", "10u": "u10", "10v": "v10"}


def parse_par(path):
    """GIK par -> DataFrame of chunk refs with parsed key fields.

    Chunk keys: step_SSS/{var}/{sfc|pl}/[{level}/]{member}/0.0.0
    Values: JSON [url, offset, length]; URLs lack the .grib2 suffix.
    """
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
    attrs_by_var = {}
    for _, row in df[df.key.str.endswith(".zattrs") & ~df.key.str.startswith("step_")].iterrows():
        a = json.loads(row.v)
        attrs_by_var[a.get("varname", "")] = a
    return refs, attrs_by_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", required=True)
    ap.add_argument("--store", default="stores/ecmwf-0p4-pilot")
    ap.add_argument("--ref-datetime", default="2023-06-27T00:00:00Z")
    args = ap.parse_args()
    t0 = time.time()

    refs, attrs_by_var = parse_par(args.par)
    steps = np.array(sorted(refs.step_h.unique()), dtype="int32")
    levels = np.array(sorted(refs.loc[refs.levtype == "pl", "level"].unique()))
    step_idx = {h: i for i, h in enumerate(steps)}
    lev_idx = {v: i for i, v in enumerate(levels)}
    sfc_vars = sorted(refs.loc[refs.levtype == "sfc", "var"].unique())
    pl_vars = sorted(refs.loc[refs.levtype == "pl", "var"].unique())
    print(f"parsed {len(refs)} refs | {len(steps)} steps | {len(levels)} levels | "
          f"{len(sfc_vars)} sfc + {len(pl_vars)} pl vars")

    if Path(args.store).exists():
        shutil.rmtree(args.store)
    storage = icechunk.local_filesystem_storage(args.store)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        CONTAINER_PREFIX, icechunk.s3_store(region="eu-central-1", anonymous=True)))
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.create(storage, config, authorize_virtual_chunk_access=auth)
    session = repo.writable_session("main")
    store = session.store
    root = zarr.group(store=store, zarr_format=3)

    lat = np.linspace(90.0, -90.0, NY)
    lon = np.arange(NX) * 0.4
    for name, data, dim in [("step", steps, "step"), ("isobaricInhPa", levels, "isobaricInhPa"),
                            ("latitude", lat, "latitude"), ("longitude", lon, "longitude")]:
        arr = zarr.create_array(store, name=name, shape=data.shape, dtype=data.dtype,
                                chunks=data.shape, dimension_names=[dim], overwrite=True)
        arr[:] = data
    root.attrs["reference_datetime"] = args.ref_datetime
    root.attrs["description"] = "GIK ECMWF 0p4-beta pilot: virtual GRIB refs, single member"
    root["step"].attrs["units"] = "hours"
    root["isobaricInhPa"].attrs["units"] = "hPa"

    # gribberish gotcha: var must be non-None or codec metadata fails to round-trip
    n_set = 0
    for var in sfc_vars:
        zname = SFC_RENAME.get(var, var)
        sub = refs[(refs["var"] == var) & (refs.levtype == "sfc")]
        zarr.create_array(store, name=zname, shape=(len(steps), NY, NX), chunks=(1, NY, NX),
                          dtype="float32", fill_value=float("nan"),
                          serializer=GribberishCodec(var=zname), compressors=None, filters=None,
                          dimension_names=["step", "latitude", "longitude"],
                          attributes={"long_name": attrs_by_var.get(zname, {}).get("name", zname),
                                      "grib_shortName": var}, overwrite=True)
        specs = [icechunk.VirtualChunkSpec(index=[step_idx[r.step_h], 0, 0], location=r.url,
                                           offset=int(r.offset), length=int(r.length))
                 for r in sub.itertuples()]
        bad = store.set_virtual_refs(zname, specs)
        assert not bad, f"{zname}: rejected refs {bad}"
        n_set += len(specs)

    for var in pl_vars:
        sub = refs[(refs["var"] == var) & (refs.levtype == "pl")]
        zarr.create_array(store, name=var, shape=(len(steps), len(levels), NY, NX),
                          chunks=(1, 1, NY, NX), dtype="float32", fill_value=float("nan"),
                          serializer=GribberishCodec(var=var), compressors=None, filters=None,
                          dimension_names=["step", "isobaricInhPa", "latitude", "longitude"],
                          attributes={"long_name": attrs_by_var.get(var, {}).get("name", var),
                                      "grib_shortName": var}, overwrite=True)
        specs = [icechunk.VirtualChunkSpec(index=[step_idx[r.step_h], lev_idx[r.level], 0, 0],
                                           location=r.url, offset=int(r.offset), length=int(r.length))
                 for r in sub.itertuples()]
        bad = store.set_virtual_refs(var, specs)
        assert not bad, f"{var}: rejected refs {bad}"
        n_set += len(specs)

    snap = session.commit("pilot: single member, 0p4-beta era, virtual GRIB refs")
    print(f"committed {n_set} virtual refs as snapshot {snap}  ({time.time()-t0:.1f}s)")

    # READ PATH PROOF: cold reopen + native xarray + bit-exact vs direct S3 decode
    repo2 = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    ds = xr.open_zarr(repo2.readonly_session("main").store, consolidated=False, zarr_format=3)
    print(ds)
    t2m0 = ds["t2m"].isel(step=0).values
    print(f"decoded t2m step0: mean {np.nanmean(t2m0):.2f} K")

    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    r = refs[(refs["var"] == "2t") & (refs.step_h == 0)].iloc[0]
    truth = parse_grib_array(fs.read_block(r.url, r.offset, r.length), 0, False)
    truth = truth.reshape(NY, NX).astype("float32")
    match = np.array_equal(t2m0, truth, equal_nan=True)
    print(f"BIT-EXACT match (icechunk-native read vs direct S3 decode): {match}")
    print(f"total wall time: {time.time()-t0:.1f}s")
    raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()
