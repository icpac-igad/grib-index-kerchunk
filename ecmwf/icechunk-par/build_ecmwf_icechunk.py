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

# --- grid definitions come from ONE place: grib-index-kerchunk/ecmwf/grids.py ---
# This builder used to carry its own copy of the era table and re-derived the
# longitude origin as 0 deg instead of -180 deg. Every store it wrote was
# labelled 180 deg out. Never inline a grid constant here again -- import it.
# See HANDOVER_LONGITUDE_FIX.md and grids.py.
import sys as _sys


def _locate_grids() -> Path:
    """Find the canonical ecmwf/grids.py. Fails loudly rather than guessing."""
    env = os.environ.get("GIK_ECMWF_DIR")
    here = Path(__file__).resolve()
    cands = ([Path(env)] if env else []) + [here.parent, here.parent.parent]
    for anc in here.parents:  # works from either repo, no absolute paths baked in
        cands += [anc / "grib-index-kerchunk" / "ecmwf", anc / "ecmwf"]
    for c in cands:
        if (c / "grids.py").is_file():
            return c
    raise SystemExit(
        "FATAL: cannot find the canonical ecmwf/grids.py.\n"
        "  It holds the ECMWF grid origin (-180 deg) that this builder must not\n"
        "  re-derive -- see HANDOVER_LONGITUDE_FIX.md.\n"
        "  Fix: set GIK_ECMWF_DIR=/path/to/grib-index-kerchunk/ecmwf, or symlink\n"
        f"  grids.py next to {Path(__file__).name}.\n"
        f"  Searched: {', '.join(str(c) for c in cands[:6])} ...")


_sys.path.insert(0, str(_locate_grids()))
from grids import ERAS, latitudes, longitudes  # noqa: E402

CONTAINER_PREFIX = "s3://ecmwf-forecasts/"
N_MEMBERS = 51  # number 0 = control, 1..50 = ens_01..ens_50
SFC_RENAME = {"2t": "t2m", "10u": "u10", "10v": "v10"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MANIFEST_SPLIT_TIME = 1  # 1 date per manifest shard: O(1) appends, linear growth


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
    # defective pars exist (e.g. 2026-03-19..31): pl keys missing the level
    # segment entirely, 13 levels collapsed onto one arbitrary message --
    # unrecoverable here, the date's pars must be regenerated upstream
    n_bad = int(((parts.map(len) == 5) & (refs.levtype == "pl")).sum())
    if n_bad:
        raise SystemExit(
            f"DEFECTIVE PAR {path.name}: {n_bad} pl chunk keys lack the level "
            f"segment (upstream par-generation bug) -- regenerate this date's "
            f"pars with run_lithops_ecmwf before retrying")
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


def check_coords(g, era_name: str) -> None:
    """Refuse to append into a group whose axes disagree with grids.py.

    `ensure_group` writes coordinates only when it CREATES a group, so a store
    built before the longitude fix keeps its 0-start axis and every date appended
    afterwards silently inherits the mislabelling. Checking on every append closes
    that: a stale store fails loudly on the first date instead of growing.

    There is deliberately no in-place repair. Mislabelled stores get rebuilt --
    relabelling would only ever be right for this global-field store and never for
    a realized subset, and having the escape hatch here invites using it on the
    wrong thing. See HANDOVER_LONGITUDE_FIX.md.
    """
    for name, want in (("latitude", latitudes(era_name)),
                       ("longitude", longitudes(era_name))):
        if name not in g:
            raise SystemExit(f"group has no {name!r} coordinate -- not an ECMWF "
                             f"era group?")
        have = np.asarray(g[name][:])
        if have.shape != want.shape:
            raise SystemExit(
                f"{name} has {have.shape[0]} points, era {era_name} expects "
                f"{want.shape[0]} -- wrong --era for this group?")
        if np.allclose(have, want, atol=1e-9):
            continue
        off = float(have[0] - want[0])
        raise SystemExit(
            f"REFUSING TO APPEND: this group's {name} disagrees with grids.py.\n"
            f"  stored:   {have[0]:+.4f} .. {have[-1]:+.4f}\n"
            f"  expected: {want[0]:+.4f} .. {want[-1]:+.4f}"
            + (f"   (offset {off:+.2f} deg)" if abs(off) > 1e-9 else "") + "\n"
            f"  The dates already in this group are mislabelled, so appending a\n"
            f"  correct one would mix two conventions in a single array -- worse\n"
            f"  than the original bug. Build a new store instead.\n"
            f"  See HANDOVER_LONGITUDE_FIX.md.")


TIME_UNITS = {"units": "hours since 1970-01-01",
              "calendar": "proleptic_gregorian", "standard_name": "time"}
APPEND, PREALLOC = "append", "preallocated"


def time_value(date: str, run: str) -> int:
    """Hours since epoch for an init datetime."""
    return int((datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
                - EPOCH).total_seconds() // 3600) + int(run)


def group_mode(g) -> str:
    """How this group's time axis grows. Legacy groups predate the marker."""
    return g.attrs.get("time_mode", APPEND)


def ensure_group(store, grp: str, era: dict, steps: np.ndarray,
                 era_name: str, prealloc_times: np.ndarray | None = None) -> bool:
    """Create the era/run group with its coordinate arrays if absent.

    `prealloc_times` writes the whole `time` axis up front, sorted, and marks the
    group `preallocated`: every date then writes at its own known index instead of
    extending the tail. That is what makes dates independent of each other -- both
    parallelisable and immune to the out-of-order axis that `append_dim="time"`
    produced in the published store. The cost is that the date range is frozen
    here; see --create-schema.
    """
    try:
        zarr.open_group(store=store, path=grp, mode="r", zarr_format=3)
        return False
    except Exception:
        pass
    zarr.create_group(store=store, path=grp, zarr_format=3, overwrite=False)
    levels = np.array(era["levels"], dtype="float64")
    tvals = (np.zeros(0, dtype="int64") if prealloc_times is None
             else np.asarray(prealloc_times, dtype="int64"))
    coords = [
        ("time", tvals, TIME_UNITS),
        ("number", np.arange(N_MEMBERS, dtype="int16"),
         {"long_name": "ensemble member number (0 = control)"}),
        ("step", steps, {"units": "hours"}),
        ("isobaricInhPa", levels, {"units": "hPa"}),
        # Axes come from grids.py -- the GRIB scans 90 -> -90 and -180 -> +179.75.
        # Do not inline these; a 0-start longitude here displaces every store by
        # 180 deg and nothing downstream can detect it (HANDOVER_LONGITUDE_FIX.md).
        ("latitude", latitudes(era_name), {"units": "degrees_north"}),
        ("longitude", longitudes(era_name), {"units": "degrees_east"}),
    ]
    for name, data, attrs in coords:
        shape = data.shape if data.size else (0,)
        arr = zarr.create_array(store, name=f"{grp}/{name}", shape=shape,
                                dtype=data.dtype, chunks=(max(1, shape[0]),),
                                dimension_names=[name], attributes=attrs,
                                overwrite=True)
        if data.size:
            arr[:] = data
    g = zarr.open_group(store=store, path=grp, mode="r+", zarr_format=3)
    g.attrs["time_mode"] = APPEND if prealloc_times is None else PREALLOC
    if prealloc_times is not None:
        g.attrs["n_dates"] = int(tvals.size)
    return True


def create_schema(args, storage) -> None:
    """Lay down a group with its whole time axis preallocated, and commit.

    Metadata only -- no refs, no pars read. Data arrays are still created on a
    variable's first appearance (schema drift is real across an era), but at FULL
    time length, so nothing is ever resized afterwards.
    """
    dates = read_dates(args.dates_file)
    grp = f"{args.era}/{args.run}z"
    era = ERAS[args.era]
    steps = np.array(read_steps(args.steps_file) if args.steps_file
                     else DEFAULT_STEPS_H, dtype="int32")
    tvals = np.array(sorted(time_value(d, args.run) for d in dates), dtype="int64")
    if tvals.size != len(set(dates)):
        raise SystemExit("duplicate dates in the list")

    repo = open_or_create_repo(storage)
    session = repo.writable_session("main")
    # The mode="w" hazard, guarded: laying a schema over a group that already
    # holds data makes every committed chunk unreachable from main in one commit
    # -- that cost 30 committed dates on 2026-08-07 (EWC_REALIZE_NOTES.md #6).
    try:
        existing = zarr.open_group(store=session.store, path=grp, mode="r",
                                   zarr_format=3)
    except Exception:
        existing = None
    if existing is not None:
        n = len([k for k in existing.array_keys() if k not in COORD_NAMES])
        raise SystemExit(
            f"REFUSING: group {grp} already exists in {args.store} with {n} data "
            f"array(s).\n  Writing a schema over it would make every committed "
            f"chunk unreachable from main.\n  Use a new --store, or delete that "
            f"group deliberately first.")

    ensure_group(session.store, grp, era, steps, args.era, prealloc_times=tvals)
    snap = session.commit(
        f"{grp}: schema, time axis preallocated to {tvals.size} dates "
        f"({min(dates)}..{max(dates)}), {len(steps)} steps")
    print(f"{grp}: preallocated {tvals.size} dates {min(dates)}..{max(dates)}, "
          f"{len(steps)} steps, {len(era['levels'])} levels -> {snap}")
    print(f"  time axis is sorted and frozen; dates may now be built in any "
          f"order, concurrently")


def channel_name(var: str, is_pl: bool, pl_vars) -> str:
    """Zarr array name for a GRIB shortName.

    A var can exist at BOTH sfc and pl (50r1 control: z = orography at surface,
    geopotential on levels) -- the surface instance gets a _sfc suffix.
    Shared by the serial builder and the parallel driver: this naming rule is
    exactly the kind of thing that must not exist in two places.
    """
    if is_pl:
        return var
    z = SFC_RENAME.get(var, var)
    return f"{z}_sfc" if var in pl_vars else z


def array_geometry(is_pl: bool, n_time: int, n_steps: int, n_levels: int,
                   ny: int, nx: int):
    """(shape, chunks, dimension_names) for a data array."""
    if is_pl:
        return ((n_time, N_MEMBERS, n_steps, n_levels, ny, nx),
                (1, 1, 1, 1, ny, nx),
                ["time", "number", "step", "isobaricInhPa", "latitude", "longitude"])
    return ((n_time, N_MEMBERS, n_steps, ny, nx),
            (1, 1, 1, ny, nx),
            ["time", "number", "step", "latitude", "longitude"])


def ref_specs(refs, var: str, is_pl: bool, ti: int, step_idx: dict,
              lev_idx: dict) -> list:
    """VirtualChunkSpecs for one (var, levtype) at time index `ti`.

    Every index is unique per (date, member, step, level), which is what makes
    concurrent dates chunk-disjoint and therefore legal to write from forks.
    """
    sub = refs[(refs["var"] == var)
               & (refs.levtype == "pl" if is_pl
                  else refs.levtype.isin(["sfc", "sol"]))]
    if is_pl:
        return [icechunk.VirtualChunkSpec(
            index=[ti, int(r.number), step_idx[r.step_h], lev_idx[r.level], 0, 0],
            location=r.url, offset=int(r.offset), length=int(r.length))
            for r in sub.itertuples()]
    return [icechunk.VirtualChunkSpec(
        index=[ti, int(r.number), step_idx[r.step_h], 0, 0],
        location=r.url, offset=int(r.offset), length=int(r.length))
        for r in sub.itertuples()]


def split_levtypes(refs):
    """(sfc_vars, pl_vars) present in a date's refs; raises on anything else."""
    # levtype "sol" (sot, vsw soil fields, 49r1+) has no level segment -> surface-like
    sfc = sorted(refs.loc[refs.levtype.isin(["sfc", "sol"]), "var"].unique())
    pl = sorted(refs.loc[refs.levtype == "pl", "var"].unique())
    other = set(refs.levtype.unique()) - {"sfc", "sol", "pl"}
    if other:
        raise SystemExit(f"unhandled levtypes {other} -- refusing to drop refs silently")
    return sfc, pl


def date_written(session, grp: str, g, ti: int) -> bool:
    """Has any date-`ti` chunk already been written into this group?

    A preallocated time axis is full from the start, so unlike append mode the
    axis cannot tell you which dates are done -- the manifest can. Probing one
    chunk of one data array is enough, since a date is committed atomically.
    Returns False for a group with no data arrays yet.
    """
    for name in sorted(g.array_keys()):
        if name in COORD_NAMES:
            continue
        idx = [ti] + [0] * (len(g[name].shape) - 1)
        try:
            # chunk_type wants an absolute path; anything other than
            # "uninitialized" (virtual/inline/native) means it was written.
            t = str(session.chunk_type(f"/{grp}/{name}", idx))
        except Exception:
            return False        # older icechunk without chunk_type -> don't block
        return t != "ChunkType.uninitialized"
    return False


def read_dates(path: str) -> list[str]:
    txt = Path(path).read_text().split()
    bad = [d for d in txt if not re.fullmatch(r"\d{8}", d)]
    if bad:
        raise SystemExit(f"not YYYYMMDD: {bad[:5]}")
    return sorted(txt)


def read_steps(path: str) -> list[int]:
    return sorted(int(s) for s in Path(path).read_text().split())


# 0-144h at 3h + 150-360h at 6h -- the 00z/12z forecast range. 06z/18z are
# shorter, so pass --steps-file for those.
DEFAULT_STEPS_H = list(range(0, 145, 3)) + list(range(150, 361, 6))
COORD_NAMES = {"time", "number", "step", "isobaricInhPa", "latitude", "longitude"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=list(ERAS))
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--date", help="YYYYMMDD (not needed with --create-schema)")
    ap.add_argument("--pars-dir", help="(not needed with --create-schema)")
    ap.add_argument("--store", required=True, help="local path | s3://... | gs://...")
    ap.add_argument("--sa-key", default=None, help="GCS service-account json")
    ap.add_argument("--create-schema", action="store_true",
                    help="create the group with its whole time axis preallocated "
                         "from --dates-file, then exit. Dates can afterwards be "
                         "built in any order and concurrently")
    ap.add_argument("--dates-file",
                    help="whitespace-separated YYYYMMDD list (--create-schema)")
    ap.add_argument("--steps-file",
                    help="whitespace-separated forecast hours; default is the "
                         "00z/12z range (--create-schema)")
    ap.add_argument("--overwrite-date", action="store_true",
                    help="in a preallocated group, rewrite a date whose refs are "
                         "already present instead of refusing")
    args = ap.parse_args()
    t0 = _time.time()

    if args.create_schema:
        if not args.dates_file:
            raise SystemExit("--create-schema needs --dates-file")
        return create_schema(args, resolve_storage(args.store, args.sa_key))
    if not (args.date and args.pars_dir):
        raise SystemExit("--date and --pars-dir are required "
                         "unless --create-schema")

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
    sfc_vars, pl_vars = split_levtypes(refs)
    time_val = time_value(args.date, args.run)
    print(f"{grp} {args.date}: {len(refs)} refs, {len(steps)} steps, "
          f"{len(sfc_vars)} sfc + {len(pl_vars)} pl vars")

    repo = open_or_create_repo(resolve_storage(args.store, args.sa_key))
    session = repo.writable_session("main")
    store = session.store
    created = ensure_group(store, grp, era, steps, args.era)
    g = zarr.open_group(store=store, path=grp, mode="r+", zarr_format=3)
    mode = group_mode(g)

    tarr = g["time"]
    existing = tarr[:] if tarr.shape[0] else np.array([], dtype="int64")
    if not created:
        # the group was written by an earlier run -- its axes are whatever that
        # run believed. A store predating the longitude fix fails here on its
        # first append rather than quietly growing (HANDOVER_LONGITUDE_FIX.md).
        check_coords(g, args.era)
        gsteps = g["step"][:]
        assert np.array_equal(gsteps, steps), \
            f"step axis mismatch: group has {len(gsteps)}, date has {len(steps)}"

    if mode == PREALLOC:
        # Region write at a known index. The time axis is already correct and
        # sorted, so nothing here can reorder it and no array is ever resized --
        # which is also what lets dates run concurrently.
        pos = int(np.searchsorted(existing, time_val))
        if pos >= existing.size or existing[pos] != time_val:
            raise SystemExit(
                f"{args.date} {args.run}z is not in group {grp}'s preallocated "
                f"time axis ({len(existing)} dates). The range is frozen at "
                f"--create-schema; rebuild the schema in a new store to widen it.")
        ti = pos
        n_time = int(existing.size)
        if not args.overwrite_date and date_written(session, grp, g, ti):
            raise SystemExit(f"{args.date} {args.run}z already has refs in {grp} "
                             f"at index {ti} -- pass --overwrite-date to rewrite")
        print(f"  preallocated group: writing index {ti} of {n_time}")
    else:
        if time_val in existing:
            raise SystemExit(f"{args.date} {args.run}z already in group {grp}")
        if existing.size and time_val < existing[-1]:
            # gap fill (e.g. retrying a date that failed on a transient error):
            # appended at the END, so the time axis becomes unsorted -- readers
            # should ds.sortby("time"); the health check reports it as a WARN.
            # --create-schema avoids this class of defect entirely.
            print(f"NOTE: {args.date} is earlier than the group tip -> out-of-order "
                  f"gap fill; time axis unsorted until consumers sortby('time')")
        ti = int(tarr.shape[0])
        n_time = ti + 1
        tarr.resize((ti + 1,))
        tarr[ti] = time_val

    if mode == APPEND:
        # resize EVERY existing data array, not just those with refs in this date:
        # a var that disappears mid-era (e.g. 49r1 cape -> mucape) must keep growing
        # (NaN for absent dates) or the group's time dims diverge and xarray
        # refuses to open it (found live by check_store_health.py).
        #
        # A preallocated group needs none of this -- every array is already full
        # length. Skipping it is what makes concurrent dates safe: this loop
        # mutates metadata on ~40 shared arrays, which no amount of chunk
        # disjointness would protect.
        for name in list(g.array_keys()):
            if name not in COORD_NAMES:
                arr = g[name]
                if arr.shape[0] != ti + 1:
                    arr.resize((ti + 1,) + arr.shape[1:])

    n_set = 0
    for var, is_pl in [(v, False) for v in sfc_vars] + [(v, True) for v in pl_vars]:
        zname = channel_name(var, is_pl, pl_vars)
        path = f"{grp}/{zname}"
        # n_time is the group's full time length: the preallocated count, or
        # ti + 1 while appending. A var appearing mid-era is created at full
        # length either way, so earlier dates read NaN.
        full, chunks, dims = array_geometry(is_pl, n_time, len(steps), n_levels,
                                            ny, nx)
        if zname in g:
            arr = g[zname]
            assert arr.shape[1:] == full[1:], f"{path}: shape drift"
            assert arr.shape[0] == n_time, \
                f"{path}: time length {arr.shape[0]} != {n_time}"
        else:
            # first appearance (group creation, or schema drift mid-era):
            # full time length, earlier dates read as NaN
            zarr.create_array(store, name=path, shape=full, chunks=chunks,
                              dtype="float32", fill_value=float("nan"),
                              serializer=GribberishCodec(var=zname),
                              compressors=None, filters=None, dimension_names=dims,
                              attributes={"grib_shortName": var}, overwrite=True)
        specs = ref_specs(refs, var, is_pl, ti, step_idx, lev_idx)
        bad = store.set_virtual_refs(path, specs)
        assert not bad, f"{path}: rejected refs {bad}"
        n_set += len(specs)

    snap = session.commit(f"{grp} {args.date}: 51 members, {n_set} refs "
                          f"(time index {ti})")
    print(f"committed {n_set} refs at {grp}[time={ti}] -> {snap} "
          f"({_time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
