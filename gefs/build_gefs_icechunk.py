# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4", "gcsfs",
# ]
# ///
"""GEFS GIK pars -> Icechunk virtual store (same pattern as the ECMWF
ecmwf/icechunk-par/build_ecmwf_icechunk.py, adapted to the GEFS par format).

GEFS par format (audited 2026-07-06): one parquet per member per date with two
rows -- `refs` holds the WHOLE grib_tree kerchunk dict as a python-repr string,
`version` = 1. Branches are `{var}/{stepType}/{typeOfLevel}/{var}/.zarray` with
chunk keys `.../{var}/S.0.0` (S = step index on a shared 81-step axis 0..240h
by 3h; accum/avg vars simply have no ref at S=0). URLs are complete
s3://noaa-gefs-pds paths. Corpus: 1,926 dates x 30 members (gep01..gep30),
2020-09-23..2025-12-31+ on gs://gik-gefs-aws-tf/run_par_gefs (HF mirror
E4DRR/gik-gefs-par has 2020-2021 so far).

Store model -- one repo, group `0p25/00z` (single schema era so far; room for
more groups if NOAA changes the grid or more runs are processed):

    var  (time, number=30, step=81, latitude=721, longitude=1440)
    one chunk per GRIB message, gribberish Zarr v3 codec, `time` append dim,
    manifest splitting 1 date/shard.

Skipped: branches whose data shape is not (81, 721, 1440) -- the layered
template stubs (st/soilw depthBelowLandLayer, hlcy, cin@pressureFromGroundLayer)
carry no usable time series in the pgrb2s pars. Duplicate var names across
kept branches get a _{typeOfLevel} suffix (e.g. sulwrf / sulwrf_nominalTop).

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
  uv run build_gefs_icechunk.py --date 20240301 --fetch-gcs \
      --store gs://gik-gefs-aws-tf/icechunk/gefs-ens
  # or with pre-downloaded pars:
  uv run build_gefs_icechunk.py --date 20240301 --pars-dir /tmp/pars/20240301 \
      --store stores/gefs-ens
"""
import argparse
import ast
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

CONTAINER_PREFIX = "s3://noaa-gefs-pds/"
GCS_PARS = "gik-gefs-aws-tf/run_par_gefs"
GROUP = "0p25/00z"
NY, NX = 721, 1440
N_MEMBERS = 30            # gep01..gep30 -> number coord 1..30
STEPS = np.arange(0, 241, 3, dtype="int32")   # shared 81-step axis
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MANIFEST_SPLIT_TIME = 1


def resolve_storage(store: str, sa_key: str | None):
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        return icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    return icechunk.local_filesystem_storage(store)


def parse_member_par(path: Path) -> pd.DataFrame:
    """One member par -> rows [branch, zname_var, step_idx, url, offset, length]."""
    df = pd.read_parquet(path)
    refs = ast.literal_eval(df.loc[df.key == "refs", "value"].iloc[0].decode())
    rows, skipped = [], []
    # find data-array branches: keys '{var}/{stepType}/{typeOfLevel}/{var}/.zarray'
    for k in refs:
        if not k.endswith("/.zarray"):
            continue
        parts = k.split("/")
        if len(parts) != 5 or parts[0] != parts[3]:
            continue                      # coordinate arrays etc.
        var, step_type, lev = parts[0], parts[1], parts[2]
        shape = json.loads(refs[k])["shape"]
        if shape != [81, NY, NX]:
            skipped.append((var, lev, tuple(shape)))
            continue
        prefix = f"{var}/{step_type}/{lev}/{var}/"
        for ck, cv in refs.items():
            if ck.startswith(prefix) and isinstance(cv, list) and len(cv) == 3:
                s = int(ck[len(prefix):].split(".")[0])
                rows.append((var, lev, s, cv[0], int(cv[1]), int(cv[2])))
    out = pd.DataFrame(rows, columns=["var", "lev", "step_idx", "url", "offset", "length"])
    return out, skipped


def member_number(p: Path) -> int:
    m = re.search(r"-gep(\d+)\.parquet$", p.name)
    if not m:
        raise ValueError(f"cannot parse member from {p.name}")
    return int(m.group(1))


def fetch_gcs_pars(date: str, dest: Path) -> None:
    import gcsfs
    fs = gcsfs.GCSFileSystem(token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    src = f"{GCS_PARS}/{date[:4]}/{date[4:6]}/{date}/00z"
    dest.mkdir(parents=True, exist_ok=True)
    fs.get(src + "/*.parquet", str(dest) + "/")


def open_or_create_repo(storage):
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        CONTAINER_PREFIX, icechunk.s3_store(region="us-east-1", anonymous=True)))
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict({
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName("time"):
                    MANIFEST_SPLIT_TIME}}))
    return icechunk.Repository.create(storage, config,
                                      authorize_virtual_chunk_access=auth)


def ensure_group(store) -> bool:
    try:
        zarr.open_group(store=store, path=GROUP, mode="r", zarr_format=3)
        return False
    except Exception:
        pass
    zarr.create_group(store=store, path=GROUP, zarr_format=3, overwrite=False)
    coords = [
        ("time", np.zeros(0, dtype="int64"), {"units": "hours since 1970-01-01",
                                              "calendar": "proleptic_gregorian",
                                              "standard_name": "time"}),
        ("number", np.arange(1, N_MEMBERS + 1, dtype="int16"),
         {"long_name": "ensemble member (gep01..gep30, no control in pgrb2sp25)"}),
        ("step", STEPS, {"units": "hours"}),
        ("latitude", np.linspace(90.0, -90.0, NY), {"units": "degrees_north"}),
        ("longitude", np.arange(NX) * 0.25, {"units": "degrees_east"}),
    ]
    for name, data, attrs in coords:
        shape = data.shape if data.size else (0,)
        arr = zarr.create_array(store, name=f"{GROUP}/{name}", shape=shape,
                                dtype=data.dtype, chunks=(max(1, shape[0]),),
                                dimension_names=[name], attributes=attrs,
                                overwrite=True)
        if data.size:
            arr[:] = data
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD (00z)")
    ap.add_argument("--pars-dir", default=None)
    ap.add_argument("--fetch-gcs", action="store_true",
                    help=f"download the date's 30 pars from gs://{GCS_PARS}")
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None)
    args = ap.parse_args()
    t0 = _time.time()

    pars_dir = Path(args.pars_dir or f"/tmp/gefs_pars/{args.date}")
    if args.fetch_gcs:
        fetch_gcs_pars(args.date, pars_dir)
    pars = sorted(pars_dir.glob("*.parquet"))
    if len(pars) != N_MEMBERS:
        raise SystemExit(f"expected {N_MEMBERS} pars in {pars_dir}, found {len(pars)}")

    frames, skipped = [], []
    for p in pars:
        r, sk = parse_member_par(p)
        r["number"] = member_number(p)
        frames.append(r)
        skipped = sk  # identical across members
    refs = pd.concat(frames, ignore_index=True)
    step_idx_ok = refs.step_idx.between(0, len(STEPS) - 1).all()
    assert step_idx_ok, "step index outside the shared 0..240h axis"

    # unique array names: suffix with typeOfLevel where a var spans branches
    branches = refs[["var", "lev"]].drop_duplicates()
    dup_vars = set(branches["var"][branches["var"].duplicated()])
    refs["zname"] = [f"{v}_{l}" if v in dup_vars else v
                     for v, l in zip(refs["var"], refs["lev"])]
    time_val = int((datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc)
                    - EPOCH).total_seconds() // 3600)
    print(f"{GROUP} {args.date}: {len(refs)} refs, {refs.zname.nunique()} vars, "
          f"{refs.number.nunique()} members | skipped non-81-step branches: "
          f"{sorted(set(skipped))}")

    repo = open_or_create_repo(resolve_storage(args.store, args.sa_key))
    session = repo.writable_session("main")
    store = session.store
    ensure_group(store)
    g = zarr.open_group(store=store, path=GROUP, mode="r+", zarr_format=3)

    tarr = g["time"]
    existing = tarr[:] if tarr.shape[0] else np.array([], dtype="int64")
    if time_val in existing:
        raise SystemExit(f"{args.date} already in store")
    if existing.size and time_val < existing[-1]:
        raise SystemExit("appends must be chronological")
    ti = int(tarr.shape[0])
    tarr.resize((ti + 1,))
    tarr[ti] = time_val

    n_set = 0
    for zname, sub in refs.groupby("zname"):
        path = f"{GROUP}/{zname}"
        full = (ti + 1, N_MEMBERS, len(STEPS), NY, NX)
        if zname in g:
            arr = g[zname]
            assert arr.shape[1:] == full[1:], f"{path}: shape drift"
            arr.resize(full)
        else:
            zarr.create_array(store, name=path, shape=full,
                              chunks=(1, 1, 1, NY, NX), dtype="float32",
                              fill_value=float("nan"),
                              serializer=GribberishCodec(var=zname),
                              compressors=None, filters=None,
                              dimension_names=["time", "number", "step",
                                               "latitude", "longitude"],
                              attributes={"grib_shortName": sub["var"].iloc[0],
                                          "typeOfLevel": sub["lev"].iloc[0]},
                              overwrite=True)
        specs = [icechunk.VirtualChunkSpec(
            index=[ti, int(r.number) - 1, int(r.step_idx), 0, 0],
            location=r.url, offset=r.offset, length=r.length)
            for r in sub.itertuples()]
        bad = store.set_virtual_refs(path, specs)
        assert not bad, f"{path}: rejected refs {bad}"
        n_set += len(specs)

    snap = session.commit(f"{GROUP} {args.date}: {N_MEMBERS} members, "
                          f"{n_set} refs (time index {ti})")
    print(f"committed {n_set} refs at {GROUP}[time={ti}] -> {snap} "
          f"({_time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
