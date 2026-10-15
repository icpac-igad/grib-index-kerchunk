"""Daily ECMWF IFS ensemble -> raw members over EA/SW-IO, as a Frisky futures DAG.

This is the Option-A-shaped design from NEXT_SESSION.md §5, but built on Frisky
futures instead of a Dask collection graph. The three rules it is built to obey:

  §4.2  one task = one GRIB message, and nothing oversubscribes a worker
  §4.3  NO `.compute()` anywhere inside a task -- leaves read numpy directly,
        so no zarr `sync()` event loop is ever nested inside a worker thread
  §4.4  every read is wrapped in 503/SlowDown backoff

DAG shape, per date.  Every leaf is independent; the only fan-in is over
`number`, and it is a STACK, not a reduction -- the store keeps every raw
ensemble member, control included (`number=0`), because the downstream model
trains on the spread itself:

    read(m=0,  step=s) ---.        m=0 is the CONTROL
    read(m=1,  step=s) ----+--> stack_members(step=s) --> write_block -> chunk
    ...                   /                                 (number, y, x)
    read(m=50, step=s) --'
    ... one such fan-in per (channel, step) ...

`reduce_members` (mean/sd) still exists but is a probe-only fold, reachable
only via `--reduce-members`, and refused with `--sink`.  Neither writer
(`build_corpus.py`, `parallel_write.py`) can reach it at all: both submit
`write_block` directly, which stacks.

Counts for the production shape (36 channels x 51 members x 53 steps):
97,308 leaves + 1,908 reductions + 36 stacks per date.  Dask pays ~1 ms of
scheduler overhead per task on that (~97 s/date of pure bookkeeping); Frisky
pays ~3 us (~0.25 s).  That is not the main reason to use it here -- see the
memory note in the header of `run()` -- but it is not nothing.

Usage
-----
    P=.venv/bin/python

    # exercise the DAG shape with no network at all
    $P frisky_daily_dag.py --synthetic --channels 4 --members 6 --steps 3

    # smallest real thing: one channel, few members/steps
    $P frisky_daily_dag.py --date 2026-07-02 --channels 1 --members 4 --steps 2

    # ramp the channel count to find where it breaks (NEXT_SESSION.md §4.5)
    $P frisky_daily_dag.py --date 2026-07-02 --ramp --members 4 --steps 2

    # DAG size only, nothing submitted -- the frisky column for DAG_METHOD.md
    $P frisky_daily_dag.py --count --channels 30 --members 51 --steps 53

    # against a standalone Frisky cluster instead of a local one
    #   frisky scheduler
    #   frisky worker <sched>:8786 --nthreads 1     (on each worker VM)
    $P frisky_daily_dag.py --scheduler tcp://10.0.0.1:8786 --date 2026-07-02
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# The AWS_* trap (README §2).  The virtual-chunk S3 client is built lazily, on
# the first chunk fetch, and icechunk caches object-store config process-wide.
# So AWS_* must be gone BEFORE anything touches the store -- in this process and
# in every worker process.  Frisky's LocalCluster spawns workers from this
# process's environment, so popping here covers both.  Standalone `frisky
# worker` processes must be launched with AWS_* already unset.
#
# Nothing here writes to must-icechunk, so nothing here needs the Ceph
# credentials back.  Keep the write in a separate process (see `--out`).
# ---------------------------------------------------------------------------
_STRIPPED_AWS = {k: os.environ.pop(k) for k in list(os.environ)
                 if k.startswith("AWS_")}

SRC_ENDPOINT = "https://data.source.coop"
SRC_BUCKET = "e4drr-project"
SRC_PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
CONTAINER = "s3://ecmwf-forecasts/"

# ---------------------------------------------------------------------------
# Domain.  `ea-sw-io` -- East Africa plus the south-west Indian Ocean, the
# basin the systems that flood East Africa actually come from.  Widened to
# 40N on 2026-08-16 so the northern part of the region is assessed too:
#
#   ea          lat  25.25 -> -15.25, lon 18.5 -> 55.0   163 x 147 =  23,961
#   ea-sw-io    lat  25.25 -> -40.0,  lon 15.0 -> 80.0   262 x 261 =  68,382
#   ea-sw-io40  lat  40.0  -> -40.0,  lon 15.0 -> 80.0   321 x 261 =  83,781
#
# Every edge lands exactly on the 0.25 deg grid, so no interpolation is
# implied (90 - 40 = 50, and 50 / 0.25 = 200 exactly).
#
# Widening costs nothing to READ.  A GRIB message is global, so the number of
# messages -- channels x members x steps -- is independent of the box; the box
# only decides what is kept after the fetch.  It costs on the WRITE side only:
# the chunk grows 13.95 -> 17.09 MB and a date 12.0 -> 14.8 GB stored.
#
# Changing these constants changes what every worker cuts out of a message, and
# the workers hold their own copy of this module -- redeploy with
# `deploy_workers.py --start` after any edit here, or they keep the old box
# and `write_block` fails on a shape mismatch (loudly, which is the point).
# ---------------------------------------------------------------------------
LAT_MAX, LAT_MIN = 40.0, -40.0         # store latitude DESCENDS
LON_MAX = 80.0

# The grid is a property of the ERA, not a constant.
#
#   49r1 / 50r1   0.25 deg, 721x1440 global   -> 321 x 261 over the box
#   0p4           0.4  deg, 451x900  global   -> 201 x 164 over the box
#
# The western edge differs and that is not cosmetic. 15.0E is NOT a 0p4 grid
# point: the grid runs -180 + 0.4k, and 15.0 needs k = 487.5. Asking for
# slice(15.0, 80.0) on a 0.4 deg axis silently returns the 15.2 edge and a
# 163-wide box, which then fails the write on a shape mismatch -- or worse,
# succeeds against a schema built the same wrong way. 14.8 IS on the grid, and
# 40.0, -40.0 and 80.0 are on both grids exactly.
#
# Every era therefore gets its own (deg, lon_min); nothing here is derived from
# another era's numbers. Changing these changes what each worker cuts out of a
# message, and workers hold their own copy of this module -- redeploy with
# `deploy_workers.py --start` after any edit, or they keep the old box.
_GRID = {
    "0p4":  (0.4,  14.8),
    "49r1": (0.25, 15.0),
    "50r1": (0.25, 15.0),
}


def grid_for(era):
    """-> (deg, lon_min, ny, nx) for one era. The single source of box truth."""
    try:
        deg, lon_min = _GRID[era]
    except KeyError:
        raise SystemExit(f"no grid defined for era {era!r}; known: {sorted(_GRID)}")
    ny = int(round((LAT_MAX - LAT_MIN) / deg)) + 1
    nx = int(round((LON_MAX - lon_min) / deg)) + 1
    return deg, lon_min, ny, nx


# Module-level defaults are the 0.25 deg eras, so anything that predates
# grid_for() keeps its old behaviour bit-for-bit.
GRID_DEG, LON_MIN, NY, NX = (lambda g: (g[0], g[1], g[2], g[3]))(grid_for("49r1"))

# (store_var, level|None, out_name) -- ordered heaviest-manifest-last so a small
# --channels N still exercises a mix of surface and pressure-level reads.
#
# 36 of the 39 draw on base variables common to `0p4`, `49r1` and `50r1`
# (ECMWF_VARIABLES_REFERENCE.md §2), and that portable 36 is the set to build a
# 0p4 season from -- unlike the previous 30-channel table, which used `ssr`,
# `ttr`, `tcw`, `mucape` and `w`, none of which exist in 0p4. The other three --
# w700, cape, d2m -- are 0.25-degree only; see NOT_IN_0P4 and has_dewpoint().
# Levels used are 200/500/700/850/925, all in the 9 levels common to the three.
# ECMWF renamed the CAPE field. Before 2024-11-13 it is `cape`; from that date
# it is `mucape` (most-unstable parcel). They are the same channel to the model
# and are stored under the single output name `cape`, so a season that straddles
# the rename still yields one continuous variable. Resolved per date at read
# time by _resolve_var(), because a channel table has no date in it.
#
# The date was 2025-01-01 here until 2026-08-27 and that was WRONG by seven
# weeks. Measured against the source: `cape` decodes on 2024-11-12 and returns
# all-NaN on 2024-11-13, `mucape` the reverse. 2024-11-13 is also the date
# `mn2t6`/`mx2t6` first appear, so ECMWF changed the published field set that
# day. Nothing built before 2026-08-27 was affected -- MAM 2024 and JJA 2024
# both end before 2024-11-13, and every later season starts after it, so no
# built season straddled the boundary. SOND 2024 is the first that does: under
# the old constant its last 49 dates would have resolved to `cape` and written
# an all-NaN channel, silently, because an unwritten chunk reads as NaN rather
# than raising.
CAPE = "@cape"                          # sentinel, never a real GRIB shortName
CAPE_RENAMED = "2024-11-13"


def _resolve_var(var, date):
    """Map a channel's source name for one date. Only CAPE is date-dependent."""
    if var != CAPE:
        return var
    return "cape" if str(date)[:10] < CAPE_RENAMED else "mucape"


# Source variables the 0.4-degree archive does not publish. Verified against the
# published store on 2026-09-18: 0p4/00z carries 19 variables and these are not
# among them, so a 0p4 season is a 36-channel store against 49r1/50r1's 39.
#
# Declared once, as data, because the guard in build_corpus.py used to encode it
# as `c[2] == "w700" or c[0] == CAPE`. Adding `2d` as channel 39 escaped that
# test silently, and a 0p4 season would then have raised KeyError inside a worker
# on message ~90,000 of a run 40 minutes in. A new channel needs no edit here
# unless it draws on a variable 0p4 also lacks.
NOT_IN_0P4 = {"w", "2d", CAPE}


def is_convective_era(era):
    """Whether `era` carries the NOT_IN_0P4 variables -- i.e. is not 0p4."""
    return era in ("49r1", "50r1")


def channels_absent_from(era, channels, date=None):
    """out_names in `channels` that (era, date) cannot supply.

    Empty means buildable. Pass the season's FIRST date: `2d` is limited by date
    as well as by era (has_dewpoint), and a 49r1 season starting before
    2024-03-06 would otherwise write an all-NaN d2m channel silently -- the same
    failure the CAPE constant caused, since an unwritten chunk reads as NaN
    rather than raising.
    """
    absent = [] if is_convective_era(era) else \
        [out for src, _lvl, out in channels if src in NOT_IN_0P4]
    if date is not None and not has_dewpoint(era, date):
        absent += [out for src, _lvl, out in channels
                   if src == "2d" and out not in absent]
    return absent


CHANNELS = [
    # -- surface: direct hazard / moisture / circulation ------------------
    ("tp", None, "tp"),                # precipitation
    ("tcwv", None, "tcwv"),            # column water vapour
    ("msl", None, "msl"),              # synoptic pressure systems
    ("sp", None, "sp"),                # surface pressure
    ("ro", None, "ro"),                # runoff
    ("t2m", None, "t2m"),              # near-surface temperature
    ("skt", None, "skt"),              # surface thermal state
    ("u10", None, "u10"),
    ("v10", None, "v10"),
    ("lsm", None, "lsm"),              # land-sea mask (constant in time)

    # -- low/mid/upper-level circulation ----------------------------------
    ("u", 925, "u925"), ("v", 925, "v925"),
    ("u", 850, "u850"), ("v", 850, "v850"),
    ("u", 700, "u700"), ("v", 700, "v700"),
    ("u", 500, "u500"), ("v", 500, "v500"),
    ("u", 200, "u200"), ("v", 200, "v200"),

    # -- specific humidity -------------------------------------------------
    ("q", 925, "q925"), ("q", 850, "q850"),
    ("q", 700, "q700"), ("q", 500, "q500"),

    # -- relative humidity -------------------------------------------------
    ("r", 850, "r850"), ("r", 700, "r700"),

    # -- thermodynamic profile ---------------------------------------------
    ("t", 850, "t850"), ("t", 700, "t700"), ("t", 500, "t500"),

    # -- mid-tropospheric circulation --------------------------------------
    ("gh", 500, "gh500"),

    # -- native relative vorticity -----------------------------------------
    ("vo", 850, "vo850"), ("vo", 700, "vo700"), ("vo", 500, "vo500"),

    # -- native divergence --------------------------------------------------
    ("d", 850, "d850"), ("d", 700, "d700"), ("d", 200, "d200"),

    # -- convection: the two cGAN channels ---------------------------------
    # Added 2026-08-22, so a season is built at 38 in one pass instead of 36
    # plus an append_channels.py run.  Both are absent from `0p4`, which is
    # why a 0p4 season is a 36-channel store -- see is_convective_era().
    ("w", 700, "w700"),                # vertical velocity: the updraught
    (CAPE, None, "cape"),              # convective available potential energy

    # ---- beyond the default 38 -------------------------------------------
    # Channel 39. Deliberately AFTER the 38, so `--channels 38` still selects
    # exactly the set every published 0.25 deg season was built with. Reach it
    # by name: `--vars d2m`. Availability is narrower than the rest, so check
    # has_dewpoint() before building -- see the note there.
    ("2d", None, "d2m"),               # 2 m dew point temperature
]

DEWPOINT_FROM = "2024-03-06"


def has_dewpoint(era, date):
    """`2d` is not available everywhere the other channels are.

    Measured by decode, not read off the era's variable union:

        era    date          2d
        0p4    any           absent   (not among its 19 variables)
        49r1   2024-03-05    all-NaN
        49r1   2024-03-06    ok       <- the 83 -> 102 field change
        50r1   any           ok

    2024-03-06 is the same boundary that limits `w700`/`cape` to 87/92 in
    MAM 2024, so the gap is identical and already precedented.
    """
    if era == "0p4":
        return False
    return str(date)[:10] >= DEWPOINT_FROM


N_CHANNELS = len(CHANNELS)             # 39 -- 0p4 can supply only 36

STEPS_ALL_H = list(range(0, 145, 3)) + list(range(150, 361, 6))


# ===========================================================================
# Worker-side.  Module-level functions only -- Frisky pickles them by
# reference, so nothing large rides along with each of the 97,308 tasks.
# ===========================================================================

_DS_CACHE: dict = {}          # per worker PROCESS, not per task
_SRC_SNAPSHOT: dict = {}      # era -> snapshot id of the source store as opened


def _open_era(era: str):
    """Open one era group, numpy-backed.  ~5 s, then cached for the process.

    `chunks=None` is the whole point: the returned DataArrays are NOT dask
    arrays, so `.values` on a single-message selection is a direct icechunk
    read.  There is no dask graph inside the task and therefore no nested
    event loop -- NEXT_SESSION.md §4.3.
    """
    if era in _DS_CACHE:
        return _DS_CACHE[era]

    import icechunk
    import xarray as xr
    import gribberish.zarr  # noqa: F401  registers the Zarr v3 codec

    storage = icechunk.s3_storage(
        bucket=SRC_BUCKET, prefix=SRC_PREFIX, endpoint_url=SRC_ENDPOINT,
        region="us-east-1", anonymous=True, from_env=False,
        force_path_style=True,
    )
    auth = icechunk.containers_credentials(
        {CONTAINER: icechunk.s3_anonymous_credentials()})
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg,
                                    authorize_virtual_chunk_access=auth)
    sess = repo.readonly_session("main")
    _SRC_SNAPSHOT[era] = sess.snapshot_id
    ds = xr.open_zarr(sess.store, group=f"{era}/00z", consolidated=False,
                      zarr_format=3, decode_timedelta=True, chunks=None)
    _DS_CACHE[era] = ds
    return ds


def source_provenance(era: str) -> str:
    """One line naming the GIK virtual store this build read, and its SNAPSHOT.

    A run log recorded the sink, the shape and the cost, but nothing tying the
    output to its input: provenance had to be inferred from a source-code
    constant that could since have changed. The snapshot id is the part that
    matters -- it pins the exact version of the virtual store, which the date
    axis only gestures at, and it is what distinguishes a realization built from
    the GIK manifests from one built any other way.

    Call it on the CLIENT. Cheap: the client already opens the source for
    subset_coords()/available_dates(), so this reuses that cached session.
    """
    _open_era(era)            # populates _SRC_SNAPSHOT; no-op if already open
    return (f"source   {SRC_ENDPOINT}/{SRC_BUCKET}/{SRC_PREFIX}\n"
            f"         group {era}/00z  snapshot {_SRC_SNAPSHOT.get(era, 'unknown')}\n"
            f"         virtual chunks -> {CONTAINER} (anonymous), decoded by gribberish")


def read_message(era, var, level, date, member, step_h, synthetic=False):
    """ONE GRIB message -> the East Africa box as float32.  The unit of work.

    Measured at 0.23-0.25 s warm on a single machine (NEXT_SESSION.md §2).
    Holds one global message (~4 MB) transiently and returns ~0.33 MB.
    """
    _, _lon_min, _ny, _nx = grid_for(era)
    if synthetic:
        rng = np.random.default_rng(abs(hash((var, member, step_h))) % 2**32)
        time.sleep(0.02)
        _, _, _ny, _nx = grid_for(era)
        return rng.standard_normal((_ny, _nx), dtype=np.float32)

    import numpy as _np
    import random as _random

    # _open_era used to sit OUTSIDE the retry below.  It fetches the store
    # manifest, so it is an S3 call like any other and gets throttled like any
    # other -- and with 12 worker processes starting at once, all of them open
    # the store simultaneously.  A SlowDown there killed the task outright.
    delay = 0.5
    for attempt in range(8):
        try:
            ds = _open_era(era)
            break
        except Exception as exc:
            if "SlowDown" not in str(exc) or attempt == 7:
                raise
            time.sleep(delay + _random.random() * delay)
            delay = min(delay * 2, 30)
    da = ds[_resolve_var(var, date)]
    # Level FIRST, while this is still lazily-indexed xarray -- DAG_METHOD.md
    # §5. There is no dask graph here to blow up, but the composed lazy index
    # is narrower for it and the rule costs nothing to follow.
    if level is not None:
        da = da.sel(isobaricInhPa=level)
    da = da.sel(
        time=_np.datetime64(date),
        number=member,
        step=_np.timedelta64(step_h, "h"),
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(_lon_min, LON_MAX),
    )

    # §4.4 -- icechunk surfaces AWS throttling as `unhandled error (SlowDown)`
    # and does not retry.  One 503 anywhere otherwise fails the whole read.
    #
    # This went untested through every run until 192 concurrent readers finally
    # provoked AWS, and then two things were wrong with it. Six attempts at
    # 0.5-16 s is only ~31 s of total backoff, which a sustained throttle
    # outruns; and with no jitter every throttled reader retried in lockstep,
    # so each wave arrived together and was refused together. Eight attempts
    # capped at 30 s with full jitter gives ~2 min of headroom and spreads the
    # retries out.
    #
    # Backoff is damage control, not a fix. The real lever is not exceeding
    # ~96 concurrent readers in the first place -- see SCALING.md.
    delay = 0.5
    for attempt in range(8):
        try:
            return _np.asarray(da.values, dtype=_np.float32)
        except Exception as exc:
            msg = str(exc)
            throttled = ("SlowDown" in msg or "503" in msg
                         or "ServiceUnavailable" in msg)
            if not throttled or attempt == 7:
                raise
            time.sleep(delay + _random.random() * delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("unreachable")


def reduce_members(*fields):
    """PROBE ONLY -- mean/sd fan-in over `number`.  Not the production path.

    Nothing that writes a store calls this.  It survives because the `--out`
    npz path and the throughput probes use it to keep client memory trivial
    while measuring read rate.  Collapsing 51 members to two moments would
    destroy exactly the signal the realized store exists to carry, so `--sink`
    refuses it and `build_and_submit` defaults to `stack_members`.

    Takes ~51 x 0.1 MB and returns 2 x 0.1 MB, so the peak a worker holds for
    one reduction is ~5 MB.  Memory is bounded by construction, which matters
    because neither Dask nor Frisky can see icechunk's Rust allocations.
    """
    stack = np.stack(fields, axis=0)
    return (stack.mean(axis=0, dtype=np.float32),
            stack.std(axis=0, dtype=np.float32))


def stack_members(*fields):
    """THE production fan-in: keep every member, (number, y, x) per step.

    `fields` arrives in the order the members were submitted -- 0..50 -- so
    axis 0 of the result is the `number` coordinate as written, with the
    control at index 0.  ~17.1 MB out instead of ~0.5 MB, still bounded.
    """
    return np.stack(fields, axis=0)


def write_block(fork, name, time_idx, step_idx, *fields):
    """Stack members and write THIS chunk straight to Ceph from the worker.

    The whole point of icechunk's fork/merge model
    (https://icechunk.io/en/latest/understanding/parallel/): the 259 MB of a
    channel never travels to the client.  Each worker writes its own region
    and returns only a changeset, so

      * the client stops being a funnel -- it was gathering 259 MB per channel
        into an 8 GiB cgroup and got SIGKILLed at channel 21
      * the 7.8 GB write happens on six VMs at once instead of serially
      * one commit covers the date, instead of one per channel

    Safe because our store chunk is (1, 1, number, lat, lon), so a
    (channel, step) block is EXACTLY one chunk: aligned, and no two workers
    can touch the same one.  Those are the two constraints the icechunk docs
    put on the caller.
    """
    import zarr

    block = np.stack(fields, axis=0)          # (number, y, x)
    arr = zarr.open_array(fork.store, path=name, mode="r+")
    arr[time_idx, step_idx] = block
    return fork


def stack_steps(*per_step):
    """(step, y, x) mean and sd for one channel on one date."""
    return (np.stack([m for m, _ in per_step], axis=0),
            np.stack([s for _, s in per_step], axis=0))


# ===========================================================================
# Client-side: build the graph, submit once, gather.
# ===========================================================================

def build_and_submit(client, date, channels, members, steps, era, synthetic,
                     keep_members=True):
    """One submission per date.  Members are KEPT unless explicitly told not to.

    Returns {out_name: [Future[(number, y, x)] per step]} -- a LIST,
    deliberately.  With keep_members=False (probe only) it collapses to
    {out_name: Future[(mean, sd)]}.

    The list matters for memory.  A whole channel with members kept is
    51 x 53 x 321 x 261 float32 = 906 MB, and 36 of those is 32.6 GB against an
    8 GiB client cgroup.  Returning per-step futures lets the caller gather one
    channel, write it, and drop it before touching the next.

    Note what is NOT here: no `client.compute()` called from inside a task, no
    graph materialised on the client beyond a list of futures, and no single
    task that holds every member x step at once.  That combination is what
    hung the Dask runs.
    """
    fold = stack_members if keep_members else reduce_members
    out = {}
    for var, level, name in channels:
        step_futures = []
        for step_h in steps:
            member_futures = [
                client.submit(read_message, era, var, level, date, m, step_h,
                              synthetic)
                for m in members
            ]
            step_futures.append(client.submit(fold, *member_futures))
        out[name] = step_futures if keep_members else \
            client.submit(stack_steps, *step_futures)
    return out


def subset_coords(era, date, members, steps, synthetic=False):
    """Coordinate values for the EA box, read once on the client.

    Metadata only -- `.sel()` on lazily-indexed xarray fetches no chunks, so
    this costs one store open (~5 s) and no data.
    """
    _, _lon_min, _ny, _nx = grid_for(era)
    import numpy as _np

    if synthetic:
        return {"step": _np.array([_np.timedelta64(h, "h") for h in steps]),
                "number": _np.array(members),
                "latitude": _np.linspace(LAT_MAX, LAT_MIN, _ny),
                "longitude": _np.linspace(_lon_min, LON_MAX, _nx)}

    ds = _open_era(era)
    box = ds["t2m"].sel(latitude=slice(LAT_MAX, LAT_MIN),
                        longitude=slice(_lon_min, LON_MAX))
    return {
        "step": _np.array([_np.timedelta64(h, "h") for h in steps]),
        "number": _np.array(members),
        "latitude": box.latitude.values,
        "longitude": box.longitude.values,
    }


def available_dates(era, dates):
    """Which of `dates` are actually on the source era's `time` axis.

    `49r1/00z` carries all 804 calendar days it spans as of the 2026-08-08
    re-mirror; the ten March 2026 holes (03-19, 03-22..27, 03-29..31) are
    closed.  This check stays because `0p4` still has six upstream holes
    (2023-04-27..05-02) and nothing guarantees a future era is dense.

    NOTE: a worker started before a re-mirror keeps serving the OLD snapshot --
    `_open_era` caches per process -- so it will report the recovered dates as
    absent.  Restart the workers after any source change.

    Without this check `build_corpus.py`
    submits 1,908 blocks for a date that cannot resolve, waits out four retry
    rounds at 45 s each, and only then calls it FAILED: ~3 minutes of pure
    waiting per absent date, plus 97,308 doomed reads.

    Membership is tested against the raw values, NOT a slice.  The 49r1 and
    0p4 axes are non-monotonic -- one date was appended after a later batch, so
    `t[0]`/`t[-1]` are not the min and max -- and pandas refuses partial slicing
    on a non-monotonic DatetimeIndex with `KeyError`.  Exact-label lookup is
    unaffected by the ordering, which is also why `read_message` keeps working.
    """
    import numpy as _np

    ds = _open_era(era)
    have = set(_np.datetime_as_string(ds.time.values, "D").tolist())
    return [d for d in dates if d in have]


def select_channels(args):
    """--vars by name, else the first --channels of the table."""
    if not getattr(args, "vars", None):
        return CHANNELS[:args.channels]
    by_name = {name: c for c in CHANNELS for name in (c[2],)}
    missing = [v for v in args.vars if v not in by_name]
    if missing:
        raise SystemExit(f"unknown channel(s) {missing}.  "
                         f"known: {', '.join(c[2] for c in CHANNELS)}")
    return [by_name[v] for v in args.vars]


def count_only(args):
    """DAG size, answered by arithmetic, before anything is opened.

    This is the point of the whole design, and the answer to DAG_METHOD.md §6
    ("a graph whose size tracks the REQUEST rather than the STORE is the goal.
    We are not there").  With futures there is no dask graph to construct, so
    the count is exactly what you asked for and is known in advance:

        leaves     = channels x members x steps
        reductions = channels x steps
        stacks     = channels

    Contrast, measured by graph_size.py on this store for ONE variable `u` at
    2 members x 1 step:

        chunks={} then .sel()   never finishes constructing   **HANG**
        .sel() then .chunk()    3,761,156 tasks, built in 2 s

    Frisky's number for that same request is 3.  The gap is not Frisky being
    fast -- it is that a dask-array graph is sized by `prod(ceil(shape/chunk))`
    over the whole array, so it tracks the store, while a futures DAG is
    written one task per thing you actually want.

    Frisky cannot rescue a dask collection from this: `frisky.hijack` and
    `client.compute` both take the graph AFTER the client has built it, so the
    3.76 M tasks (or the hang) happen before Frisky is involved.  Its
    `submit_expression` path does expand graphs scheduler-side in Rust, but it
    needs dask@main + the dask-array Rust backend, which would break the
    client/worker version parity README.md §1 depends on -- and it still
    expands to 3.76 M tasks, just somewhere else.  So: `graph_size.py` stays
    the right tool for measuring a dask graph.  Frisky lets you not build one.
    """
    channels = select_channels(args)
    n_c, n_m, n_s = len(channels), args.members, min(args.steps, len(STEPS_ALL_H))
    leaves = n_c * n_m * n_s
    reductions = n_c * n_s

    print(f"request  {n_c} channels x {n_m} members x {n_s} steps, "
          f"1 date, {args.era}")
    print(f"data     {leaves} GRIB messages wanted\n")
    print(f"  leaves      {leaves:12,d}   one per message, 0.25 s each")
    print(f"  reductions  {reductions:12,d}   fan-in over `number`")
    print(f"  stacks      {n_c:12,d}   one per channel")
    print(f"  TOTAL       {leaves + reductions + n_c:12,d}   tasks submitted\n")
    print(f"ratio    {(leaves + reductions + n_c) / leaves:.3f} tasks per message"
          f"  -- tracks the REQUEST, not the store")
    print(f"submit   ~{(leaves + reductions + n_c) * 3e-6:.2f}s at Frisky's 3 us/task"
          f"  (~{(leaves + reductions + n_c) * 1e-3:.0f}s at Dask's ~1 ms)")
    print(f"runtime  ~{leaves * 0.25 / max(args.workers, 1) / 60:.0f} min on "
          f"{args.workers} single-threaded workers at 0.25 s/message")
    print("\nfor the dask-graph column of the same request, run:")
    print(f"  graph_size.py --vars <...> --members {n_m} --steps {n_s}")
    return True


def run(args):
    """Why Frisky rather than the Dask scheduler, stated plainly.

    Frisky does NOT solve the root cause.  The root cause is that icechunk and
    gribberish allocate in Rust, so `managed` reads 0.00 GB against 33 GB
    resident, and any scheduler that sizes work from managed memory will
    oversubscribe.  Frisky sizes from managed memory too.

    What it does change, and these are worth having:

      * Worker sizing reads the cgroup limit.  On this VM Frisky sees the real
        8 GiB cap; psutil (and therefore Dask) reports the host's 31.3 GiB.
        That gap is the exit-137 client kills in BLOCKERS.md.
      * `--threads 1` is honoured per worker with no nanny in the loop, so the
        §4.2 fix is one flag rather than a cloud-init change.
      * A hijacked or standalone Frisky scheduler starts with no state, and
        `client.restart()` works -- no 236 stale tasks, no zombie clients, no
        `assert not self.tasks`.
      * ~3 us/task instead of ~1 ms, which is the difference between 0.25 s and
        80 s of scheduling per date at 81,090 tasks.

    The bounded-memory task shape above is what actually fixes the hang.
    Frisky just stops the scheduler from being a second problem.
    """
    import frisky

    channels = select_channels(args)
    members = list(range(args.members))
    steps = STEPS_ALL_H[:args.steps]
    n_leaf = len(channels) * len(members) * len(steps)

    print(f"date     {args.date}   era {args.era}"
          f"{'   [SYNTHETIC]' if args.synthetic else ''}")
    print(f"channels {len(channels)}  members {len(members)}  "
          f"steps {len(steps)}")
    print(f"tasks    {n_leaf} leaves + "
          f"{len(channels) * len(steps)} reductions + {len(channels)} stacks")

    cluster = None
    if args.scheduler:
        client = frisky.Client(args.scheduler)
        print(f"cluster  {args.scheduler} (external)")
    else:
        cluster = frisky.LocalCluster(
            n_workers=args.workers,
            threads_per_worker=args.threads,   # §4.2: 1, always
            processes=True,
            dashboard_address=args.dashboard,
        )
        client = cluster.get_client()
        print(f"cluster  {args.workers}x{args.threads} local, "
              f"dashboard {cluster.dashboard_link}")

    keep = not args.reduce_members
    repo = coords = None
    # Set here, not only under `if args.sink`: `first = not already` below runs
    # on every path, so a no-sink run (--synthetic, --out, the smoke tests in
    # this module's docstring) raised UnboundLocalError. Nothing caught it
    # because every production writer passes --sink.
    already = set()
    if args.sink:
        import sink_icechunk as sink
        here = os.path.dirname(os.path.abspath(__file__))
        ak, sk = sink.load_env(args.env or os.path.join(here, ".env"))
        repo, created = sink.open_sink(args.sink, ak, sk)
        already = sink.existing_channels(repo)
        print(f"sink     must-icechunk/{args.sink}"
              f"  ({'created' if created else 'existing'})")
        if already:
            print(f"already  {len(already)} channel(s) committed: "
                  f"{', '.join(sorted(already))}")
            if args.resume:
                before = len(channels)
                channels = [c for c in channels if c[2] not in already]
                print(f"resume   skipping {before - len(channels)}, "
                      f"{len(channels)} to go: "
                      f"{', '.join(c[2] for c in channels)}")
                if not channels:
                    print("\nnothing left to do")
                    client.close()
                    return True
                n_leaf = len(channels) * len(members) * len(steps)
        coords = subset_coords(args.era, args.date, members, steps,
                               args.synthetic)

    t0 = time.time()
    written, failed, results = [], [], {}
    try:
        futures = build_and_submit(client, args.date, channels, members, steps,
                                   args.era, args.synthetic, keep_members=keep)
        t_submit = time.time() - t0
        print(f"submitted in {t_submit:.2f}s -- now waiting\n")

        # mode="w" would wipe the channels a previous run committed.
        first = not already
        for name, fut in futures.items():
            try:
                if keep:
                    # Gather ONE channel into a PREALLOCATED buffer.
                    #
                    # The obvious `np.stack([f.result() for f in fut])` holds
                    # all 53 results (259 MB) AND the stacked copy (259 MB) at
                    # once.  Churning 518 MB per channel across 30 channels
                    # got the client SIGKILLed at channel 21 on the gateway's
                    # 8 GiB cgroup -- the exit-137 in BLOCKERS.md, self
                    # inflicted.  Filling slot by slot and dropping each
                    # result as it lands halves the peak.
                    arr = None
                    for i, f in enumerate(fut):
                        block = f.result()
                        if arr is None:
                            arr = np.empty((len(fut),) + block.shape,
                                           dtype=np.float32)
                        arr[i] = block
                        del block
                    fut.clear()          # drop the client's future refs too
                else:
                    arr = fut.result()

                if repo is not None and keep:
                    t1 = time.time()
                    snap = sink.write_channel(repo, name, arr, coords,
                                              args.date, first,
                                              first_date=args.first_date)
                    written.append((name, arr.nbytes, time.time() - t1))
                    print(f"  {name:7s} {arr.shape}  "
                          f"{arr.nbytes / 1e6:7.1f} MB  "
                          f"finite {float(np.isfinite(arr).mean()):.3f}  "
                          f"-> {str(snap)[:12]}  {time.time() - t1:5.1f}s")
                    first = False
                    del arr
                else:
                    results[name] = arr
            except Exception as exc:
                failed.append((name, f"{type(exc).__name__}: {exc}"))
        elapsed = time.time() - t0
    finally:
        client.close()
        if cluster is not None:
            cluster.close()

    # `results` holds whatever the fan-in produced: a (mean, sd) pair under
    # --reduce-members, otherwise the stacked members. Reporting assumed the
    # pair unconditionally, so a no-sink run without --reduce-members -- the
    # default, and what the docstring's smoke tests use -- died with
    # "too many values to unpack". Same cause as `already`: the no-sink path
    # is not exercised by any production writer, since all of them pass --sink.
    for name, res in results.items():
        if isinstance(res, tuple):
            mean, sd = res
            finite = float(np.isfinite(mean).mean())
            print(f"  {name:7s} mean{mean.shape}  finite {finite:.3f}  "
                  f"min/mean/max {np.nanmin(mean):9.2f} {np.nanmean(mean):9.2f} "
                  f"{np.nanmax(mean):9.2f}   sd_mean {np.nanmean(sd):8.3f}")
        else:
            finite = float(np.isfinite(res).mean())
            print(f"  {name:7s} stack{res.shape}  finite {finite:.3f}  "
                  f"min/mean/max {np.nanmin(res):9.2f} {np.nanmean(res):9.2f} "
                  f"{np.nanmax(res):9.2f}")
    for name, err in failed:
        print(f"  {name:7s} FAILED  {err}")

    ok = len(results) + len(written)
    rate = n_leaf / elapsed if elapsed else 0
    print(f"\n{ok}/{len(futures)} channels in {elapsed:.1f}s "
          f"({rate:.1f} messages/s, {elapsed / max(ok, 1):.1f}s per channel)")
    if written:
        total = sum(b for _, b, _ in written)
        wtime = sum(t for _, _, t in written)
        print(f"wrote    {total / 1e9:.2f} GB raw float32 to "
              f"must-icechunk/{args.sink} in {wtime:.0f}s "
              f"({total / 1e6 / max(wtime, 1e-9):.0f} MB/s)")

    if args.out and results:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.date}.npz")
        np.savez_compressed(path, **{f"{n}_mean": m for n, (m, _) in results.items()},
                            **{f"{n}_sd": s for n, (_, s) in results.items()})
        print(f"wrote {path}")

    return len(failed) == 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2026-07-02")
    p.add_argument("--era", default="50r1", choices=["0p4", "49r1", "50r1"])
    p.add_argument("--channels", type=int, default=1,
                   help=f"how many of the {N_CHANNELS} channels "
                        "(start at 1 -- §4.5)")
    p.add_argument("--vars", nargs="+", default=None,
                   help="pick channels by out_name (u925 v850 ...) instead of "
                        "taking the first --channels")
    p.add_argument("--members", type=int, default=4, help="0..N-1")
    p.add_argument("--steps", type=int, default=2, help="first N step hours")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=1,
                   help="per worker.  1, unless you enjoy nanny kills")
    p.add_argument("--scheduler", default=None,
                   help="tcp://host:8786 of a standalone Frisky scheduler")
    p.add_argument("--dashboard", default="127.0.0.1:8790",
                   help="8787 is taken by the Dask scheduler on this VM")
    p.add_argument("--out", default=None, help="directory for .npz output")
    p.add_argument("--sink", default=None,
                   help="prefix under must-icechunk to write, e.g. "
                        "ea-swio/one-date.  Always writes raw members")
    p.add_argument("--reduce-members", action="store_true",
                   help="PROBE ONLY: collapse the 51 members to mean/sd "
                        "instead of keeping them.  Refused with --sink.  "
                        "Members are kept by default: "
                        f"51x53x{NY}x{NX} f32 = 906 MB per channel, "
                        "32.6 GB/date")
    p.add_argument("--resume", action="store_true",
                   help="skip channels already committed to --sink.  Each is "
                        "its own commit, so a killed run loses nothing")
    p.add_argument("--first-date", action="store_true",
                   help="this date lays down the schema (mode=w); omit to "
                        "append along `time`")
    p.add_argument("--env", default=None,
                   help="path to the .env holding AK/SK (default ./.env)")
    p.add_argument("--synthetic", action="store_true",
                   help="no network: prove the DAG shape only")
    p.add_argument("--ramp", action="store_true",
                   help=f"run at 1, 2, 4, 8, 16, {N_CHANNELS} channels and "
                        "stop at the break")
    p.add_argument("--count", action="store_true",
                   help="print the DAG size and exit; opens nothing, submits "
                        "nothing")
    args = p.parse_args()

    if args.sink and args.reduce_members:
        p.error("--sink writes the raw member dimension; drop "
                "--reduce-members")

    if args.count:
        sys.exit(0 if count_only(args) else 1)

    if not args.ramp:
        sys.exit(0 if run(args) else 1)

    for n in (1, 2, 4, 8, 16, N_CHANNELS):
        if n > len(CHANNELS):
            break
        print(f"\n{'=' * 72}\n  RAMP: {n} channel(s)\n{'=' * 72}")
        args.channels = n
        if not run(args):
            print(f"\nbroke at {n} channels.  That number localises it.")
            sys.exit(1)
    print(f"\nramp clean to {N_CHANNELS} channels.")


if __name__ == "__main__":
    main()
