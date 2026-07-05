# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4",
# ]
# ///
"""ONE Icechunk store for the whole ECMWF ENS archive: era/run groups.

Hierarchy (zarr groups inside a single repo -> a single URL for users):

    <repo root>
    ├── 0p4/00z    (time, number, step, [9 levels],  451,  900)
    ├── 49r1/00z   (time, number, step, [13 levels], 721, 1440)  # 13L superset:
    │                               9-level dates leave 100/150/400/600 empty
    ├── 50r1/00z   (time, number, step, [14 levels], 721, 1440)
    └── {era}/{06,12,18}z ...      # same pattern when those pars exist;
                                   # each run group has its own time & step axes
                                   # (06z/18z are shorter forecasts)

Each group is an independent FMRC dataset: `time` (init date) is the append
dim, one commit per (era, run, date). Manifest splitting along `time`
(1 date/shard) keeps appends O(1) and store growth linear. Arrays that first
appear mid-era (schema drift) are created on the fly; earlier dates read NaN.

Storage: local path, s3://bucket/prefix (env creds), or gs://bucket/prefix
(GOOGLE_APPLICATION_CREDENTIALS service-account json, or --sa-key).

Usage:
  uv run build_ecmwf_icechunk.py --era 0p4 --run 00 --date 20230627 \
      --pars-dir pars/20230627 --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens
  uv run build_ecmwf_icechunk.py --era 49r1 --run 00 --date 20250515 \
      --pars-dir pars/20250515 --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens

Open one group:   xr.open_zarr(repo.readonly_session("main").store,
                               group="49r1/00z", consolidated=False)
Open everything:  xr.open_datatree(..., engine="zarr", consolidated=False)
"""
import argparse
import json
import os
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
N_MEMBERS = 51  # number 0 = control, 1..50 = ens_01..ens_50
SFC_RENAME = {"2t": "t2m", "10u": "u10", "10v": "v10"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MANIFEST_SPLIT_TIME = 1  # 1 date per manifest shard: O(1) appends, linear growth

# era table: grid + canonical pressure-level SUPERSET (dates with fewer levels
# simply leave those slots empty -> NaN on read, same as the template decision)
ERAS = {
    "0p4":  dict(ny=451, nx=900, dlon=0.4,
                 levels=[50, 200, 250, 300, 500, 700, 850, 925, 1000]),
    "49r1": dict(ny=721, nx=1440, dlon=0.25,
                 levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]),
    "50r1": dict(ny=721, nx=1440, dlon=0.25,
                 levels=[10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]),
}


def resolve_storage(store: str, sa_key: str | None):
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        key = sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        return icechunk.gcs_storage(bucket=bucket, prefix=prefix.rstrip("/"),
                                    service_account_file=key)
    if store.startswith("s3://"):
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
    refs["url"] = loc.map(lambda x: x[0] if x[0].endswith(".grib2") else x[0] + ".grib2")
    refs["offset"] = loc.map(lambda x: x[1])
    refs["length"] = loc.map(lambda x: x[2])
    return refs[["step_h", "var", "levtype", "level", "url", "offset", "length"]]


def load_date_refs(pars_dir: Path) -> pd.DataFrame:
    pars = sorted(pars_dir.glob("*.parquet"))
    if len(pars) != N_MEMBERS:
        raise SystemExit(f"expected {N_MEMBERS} pars in {pars_dir}, found {len(pars)}")
    frames = []
    for p in pars:
        r = parse_par(p)
        r["number"] = member_number(p)
        frames.append(r)
    return pd.concat(frames, ignore_index=True)


def open_or_create_repo(storage):
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        CONTAINER_PREFIX, icechunk.s3_store(region="eu-central-1", anonymous=True)))
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict({
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName("time"):
                    MANIFEST_SPLIT_TIME}}))
    return icechunk.Repository.create(storage, config,
                                      authorize_virtual_chunk_access=auth)


def ensure_group(store, grp: str, era: dict, steps: np.ndarray) -> bool:
    """Create the era/run group with its coordinate arrays if absent."""
    try:
        zarr.open_group(store=store, path=grp, mode="r", zarr_format=3)
        return False
    except Exception:
        pass
    zarr.create_group(store=store, path=grp, zarr_format=3, overwrite=False)
    levels = np.array(era["levels"], dtype="float64")
    coords = [
        ("time", np.zeros(0, dtype="int64"), {"units": "hours since 1970-01-01",
                                              "calendar": "proleptic_gregorian",
                                              "standard_name": "time"}),
        ("number", np.arange(N_MEMBERS, dtype="int16"),
         {"long_name": "ensemble member number (0 = control)"}),
        ("step", steps, {"units": "hours"}),
        ("isobaricInhPa", levels, {"units": "hPa"}),
        ("latitude", np.linspace(90.0, -90.0, era["ny"]), {"units": "degrees_north"}),
        ("longitude", np.arange(era["nx"]) * era["dlon"], {"units": "degrees_east"}),
    ]
    for name, data, attrs in coords:
        shape = data.shape if data.size else (0,)
        arr = zarr.create_array(store, name=f"{grp}/{name}", shape=shape,
                                dtype=data.dtype, chunks=(max(1, shape[0]),),
                                dimension_names=[name], attributes=attrs,
                                overwrite=True)
        if data.size:
            arr[:] = data
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=list(ERAS))
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--pars-dir", required=True)
    ap.add_argument("--store", required=True, help="local path | s3://... | gs://...")
    ap.add_argument("--sa-key", default=None, help="GCS service-account json")
    args = ap.parse_args()
    t0 = _time.time()

    era = ERAS[args.era]
    grp = f"{args.era}/{args.run}z"
    ny, nx = era["ny"], era["nx"]
    lev_idx = {float(v): i for i, v in enumerate(era["levels"])}
    n_levels = len(era["levels"])

    refs = load_date_refs(Path(args.pars_dir))
    steps = np.array(sorted(refs.step_h.unique()), dtype="int32")
    step_idx = {h: i for i, h in enumerate(steps)}
    bad_levels = set(refs.loc[refs.levtype == "pl", "level"].dropna()) - set(lev_idx)
    if bad_levels:
        raise SystemExit(f"refs contain levels {bad_levels} outside the {args.era} "
                         f"superset -- wrong --era?")
    # levtype "sol" (sot, vsw soil fields, 49r1+) has no level segment -> surface-like
    sfc_vars = sorted(refs.loc[refs.levtype.isin(["sfc", "sol"]), "var"].unique())
    pl_vars = sorted(refs.loc[refs.levtype == "pl", "var"].unique())
    other = set(refs.levtype.unique()) - {"sfc", "sol", "pl"}
    if other:
        raise SystemExit(f"unhandled levtypes {other} -- refusing to drop refs silently")
    time_val = int((datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc)
                    - EPOCH).total_seconds() // 3600) + int(args.run)
    print(f"{grp} {args.date}: {len(refs)} refs, {len(steps)} steps, "
          f"{len(sfc_vars)} sfc + {len(pl_vars)} pl vars")

    repo = open_or_create_repo(resolve_storage(args.store, args.sa_key))
    session = repo.writable_session("main")
    store = session.store
    created = ensure_group(store, grp, era, steps)
    g = zarr.open_group(store=store, path=grp, mode="r+", zarr_format=3)

    tarr = g["time"]
    existing = tarr[:] if tarr.shape[0] else np.array([], dtype="int64")
    if time_val in existing:
        raise SystemExit(f"{args.date} {args.run}z already in group {grp}")
    if existing.size and time_val < existing[-1]:
        raise SystemExit("appends must be chronological within a group")
    if not created:
        gsteps = g["step"][:]
        assert np.array_equal(gsteps, steps), \
            f"step axis mismatch: group has {len(gsteps)}, date has {len(steps)}"
    ti = int(tarr.shape[0])
    tarr.resize((ti + 1,))
    tarr[ti] = time_val

    n_set = 0
    for var in sfc_vars + pl_vars:
        is_pl = var in pl_vars
        zname = var if is_pl else SFC_RENAME.get(var, var)
        path = f"{grp}/{zname}"
        full = ((ti + 1, N_MEMBERS, len(steps), n_levels, ny, nx) if is_pl
                else (ti + 1, N_MEMBERS, len(steps), ny, nx))
        dims = (["time", "number", "step", "isobaricInhPa", "latitude", "longitude"]
                if is_pl else ["time", "number", "step", "latitude", "longitude"])
        if zname in g:
            arr = g[zname]
            assert arr.shape[1:] == full[1:], f"{path}: shape drift"
            arr.resize(full)
        else:
            # first appearance (group creation, or schema drift mid-era):
            # full time length, earlier dates read as NaN
            zarr.create_array(store, name=path, shape=full,
                              chunks=(1, 1, 1) + ((1, ny, nx) if is_pl else (ny, nx)),
                              dtype="float32", fill_value=float("nan"),
                              serializer=GribberishCodec(var=zname),
                              compressors=None, filters=None, dimension_names=dims,
                              attributes={"grib_shortName": var}, overwrite=True)
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
        bad = store.set_virtual_refs(path, specs)
        assert not bad, f"{path}: rejected refs {bad}"
        n_set += len(specs)

    snap = session.commit(f"{grp} {args.date}: 51 members, {n_set} refs "
                          f"(time index {ti})")
    print(f"committed {n_set} refs at {grp}[time={ti}] -> {snap} "
          f"({_time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
