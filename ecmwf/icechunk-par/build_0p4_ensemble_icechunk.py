# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4",
# ]
# ///
"""Build the 0p4-era ensemble Icechunk store from GIK pars: ALL 51 members,
with `time` (forecast init) as the append dimension.

Store model (FMRC):
  surface var  -> (time, number, step, latitude, longitude)
  pressure var -> (time, number, step, isobaricInhPa, latitude, longitude)
One chunk per GRIB message: (1, 1, 1, [1,] 451, 900).

First call creates the store (time=1); subsequent calls with a new date append
along time (resize + set refs at the new index + one commit per date), which is
the mutable-store pattern from VirtualiZarr discussion #964.

Usage:
  uv run build_0p4_ensemble_icechunk.py --pars-dir pars/20230627 --date 20230627 \
      --store stores/ecmwf-0p4-ens
  uv run build_0p4_ensemble_icechunk.py --pars-dir pars/20230628 --date 20230628 \
      --store stores/ecmwf-0p4-ens          # appends
"""
import argparse
import json
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import icechunk
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec
from gribberish.zarr.codec import GribberishCodec

CONTAINER_PREFIX = "s3://ecmwf-forecasts/"
NY, NX = 451, 900  # 0.4-deg beta grid
N_MEMBERS = 51     # number 0 = control, 1..50 = ens_01..ens_50
SFC_RENAME = {"2t": "t2m", "10u": "u10", "10v": "v10"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def resolve_storage(store: str):
    """Local path, or s3://bucket/prefix (credentials + AWS_ENDPOINT_URL from env,
    e.g. source.coop: source .env with AWS_* keys and endpoint https://data.source.coop)."""
    if store.startswith("s3://"):
        import os
        bucket, _, prefix = store[5:].partition("/")
        return icechunk.s3_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            from_env=True, force_path_style=True)
    return icechunk.local_filesystem_storage(store)


def member_number(par_path: Path) -> int:
    m = re.search(r"-(control|ens_(\d+))\.parquet$", par_path.name)
    if not m:
        raise ValueError(f"cannot parse member from {par_path.name}")
    return 0 if m.group(1) == "control" else int(m.group(2))


def parse_par(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["v"] = df["value"].map(lambda b: b.decode() if isinstance(b, (bytes, bytearray)) else b)
    refs = df[df.key.str.startswith("step_")].copy()
    parts = refs.key.str.split("/")
    refs["step_h"] = parts.map(lambda p: int(p[0].split("_")[1]))
    refs["var"] = parts.map(lambda p: p[1])
    refs["levtype"] = parts.map(lambda p: p[2])
    refs["level"] = parts.map(lambda p: float(p[3]) if p[2] == "pl" else np.nan)
    loc = refs.v.map(json.loads)
    # par URLs omit the .grib2 suffix the real S3 objects carry
    refs["url"] = loc.map(lambda x: x[0] if x[0].endswith(".grib2") else x[0] + ".grib2")
    refs["offset"] = loc.map(lambda x: x[1])
    refs["length"] = loc.map(lambda x: x[2])
    return refs[["step_h", "var", "levtype", "level", "url", "offset", "length"]]


def load_date_refs(pars_dir: Path) -> pd.DataFrame:
    frames = []
    pars = sorted(pars_dir.glob("*.parquet"))
    if len(pars) != N_MEMBERS:
        raise SystemExit(f"expected {N_MEMBERS} pars in {pars_dir}, found {len(pars)}")
    for p in pars:
        r = parse_par(p)
        r["number"] = member_number(p)
        frames.append(r)
    return pd.concat(frames, ignore_index=True)


def open_or_create(store_path, steps, levels):
    storage = resolve_storage(str(store_path))
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth), False

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        CONTAINER_PREFIX, icechunk.s3_store(region="eu-central-1", anonymous=True)))
    repo = icechunk.Repository.create(storage, config, authorize_virtual_chunk_access=auth)

    session = repo.writable_session("main")
    store = session.store
    root = zarr.group(store=store, zarr_format=3)
    coords = [
        ("time", np.zeros(0, dtype="int64"), {"units": "hours since 1970-01-01",
                                              "calendar": "proleptic_gregorian",
                                              "standard_name": "time"}),
        ("number", np.arange(N_MEMBERS, dtype="int16"),
         {"long_name": "ensemble member number (0 = control)"}),
        ("step", steps, {"units": "hours"}),
        ("isobaricInhPa", levels, {"units": "hPa"}),
        ("latitude", np.linspace(90.0, -90.0, NY), {"units": "degrees_north"}),
        ("longitude", np.arange(NX) * 0.4, {"units": "degrees_east"}),
    ]
    for name, data, attrs in coords:
        shape = data.shape if data.size else (0,)
        arr = zarr.create_array(store, name=name, shape=shape, dtype=data.dtype,
                                chunks=(max(1, shape[0]),), dimension_names=[name],
                                attributes=attrs, overwrite=True)
        if data.size:
            arr[:] = data
    root.attrs["description"] = ("GIK ECMWF ensemble, 0p4-beta era (2023-01-18..2024-02-28): "
                                 "virtual GRIB refs into s3://ecmwf-forecasts, 51 members")
    root.attrs["era"] = "0p4-beta"
    session.commit("init 0p4-beta ensemble store: coords + attrs, no data arrays yet")
    return repo, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pars-dir", required=True)
    ap.add_argument("--date", required=True, help="YYYYMMDD (00z assumed)")
    ap.add_argument("--store", default="stores/ecmwf-0p4-ens")
    args = ap.parse_args()
    t0 = _time.time()

    refs = load_date_refs(Path(args.pars_dir))
    steps = np.array(sorted(refs.step_h.unique()), dtype="int32")
    levels = np.array(sorted(refs.loc[refs.levtype == "pl", "level"].unique()))
    step_idx = {h: i for i, h in enumerate(steps)}
    lev_idx = {v: i for i, v in enumerate(levels)}
    sfc_vars = sorted(refs.loc[refs.levtype == "sfc", "var"].unique())
    pl_vars = sorted(refs.loc[refs.levtype == "pl", "var"].unique())
    time_val = int((datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc)
                    - EPOCH).total_seconds() // 3600)
    print(f"{args.date}: {len(refs)} refs, {refs.number.nunique()} members, "
          f"{len(steps)} steps, {len(levels)} levels")

    repo, created = open_or_create(args.store, steps, levels)
    session = repo.writable_session("main")
    store = session.store
    root = zarr.open_group(store=store, mode="r+", zarr_format=3)

    # --- extend the time axis (append dim) ---
    tarr = root["time"]
    existing = tarr[:] if tarr.shape[0] else np.array([], dtype="int64")
    if time_val in existing:
        raise SystemExit(f"date {args.date} already in store (time={time_val}h)")
    if existing.size and time_val < existing[-1]:
        raise SystemExit("appends must be chronological")
    ti = int(tarr.shape[0])
    tarr.resize((ti + 1,))
    tarr[ti] = time_val

    # --- create (first date) or resize (append) each data array, then set refs ---
    n_set = 0
    for var in sfc_vars + pl_vars:
        is_pl = var in pl_vars
        zname = var if is_pl else SFC_RENAME.get(var, var)
        full = ((ti + 1, N_MEMBERS, len(steps), len(levels), NY, NX) if is_pl
                else (ti + 1, N_MEMBERS, len(steps), NY, NX))
        dims = (["time", "number", "step", "isobaricInhPa", "latitude", "longitude"] if is_pl
                else ["time", "number", "step", "latitude", "longitude"])
        if created and ti == 0:
            zarr.create_array(store, name=zname, shape=full,
                              chunks=(1, 1, 1) + ((1, NY, NX) if is_pl else (NY, NX)),
                              dtype="float32", fill_value=float("nan"),
                              serializer=GribberishCodec(var=zname),
                              compressors=None, filters=None, dimension_names=dims,
                              attributes={"grib_shortName": var}, overwrite=True)
        else:
            arr = root[zname]
            assert arr.shape[1:] == full[1:], f"{zname}: shape drift {arr.shape} vs {full}"
            arr.resize(full)

        sub = refs[refs["var"] == var]
        if is_pl:
            specs = [icechunk.VirtualChunkSpec(
                index=[ti, int(r.number), step_idx[r.step_h], lev_idx[r.level], 0, 0],
                location=r.url, offset=int(r.offset), length=int(r.length))
                for r in sub.itertuples()]
        else:
            specs = [icechunk.VirtualChunkSpec(
                index=[ti, int(r.number), step_idx[r.step_h], 0, 0],
                location=r.url, offset=int(r.offset), length=int(r.length))
                for r in sub.itertuples()]
        bad = store.set_virtual_refs(zname, specs)
        assert not bad, f"{zname}: rejected refs {bad}"
        n_set += len(specs)

    snap = session.commit(f"{'add' if ti else 'init'} {args.date} 00z: 51 members, "
                          f"{n_set} virtual refs (time index {ti})")
    print(f"committed {n_set} refs at time index {ti} -> snapshot {snap} "
          f"({_time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
