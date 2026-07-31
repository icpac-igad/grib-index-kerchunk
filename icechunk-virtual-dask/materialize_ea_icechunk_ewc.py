"""Materialise an East Africa cGAN predictor subset out of the *virtual* ECMWF
IFS ensemble Icechunk store (source.coop) into a *realized* Icechunk store on
the EWC object store, using the EWC Dask cluster.

    source.coop  e4drr-project/forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd
        (virtual: chunk refs point at s3://ecmwf-forecasts on AWS eu-central-1)
                                |
                    EWC dask workers: read -> crop -> reduce over `number`
                                |
    must-icechunk/<prefix>   (realized: real chunks in EWC Ceph RadosGW)

The write side follows `test-icechunk-long.py`: one `writable_session` per
forecast date, `to_icechunk(..., append_dim="time")`, one commit per date. That
pattern was already proven on this backend (conditional writes work on the Ceph
RGW behind object-store.os-api.cci1.ecmwf.int), so this script only changes
*what* is written, not *how*.

What comes out (docs/ecmwf_icechunk_dask_variable_extraction.md sec 5.3/5.8):

    <group>/<channel>   dims (time, step, stat, latitude, longitude)
                        stat = ["mean", "sd"]  -- the 2 channels per field the
                        cGAN's load_fcst() actually consumes
                        float32, one chunk per (date, all steps, both stats)

Design constraints, all of which come from the *source* store, not the sink:

  * THE WORKERS' AWS_* ENVIRONMENT POISONS THE SOURCE READ. The EWC workers
    carry AWS_ENDPOINT_URL=<ceph> and AWS_DEFAULT_REGION=RegionOne so they can
    talk to must-icechunk. The object-store client icechunk builds for the
    *virtual chunk container* picks those up too, and then tries to resolve
    `ecmwf-forecasts.s3.RegionOne.amazonaws.com` (or the bucket as a subdomain
    of the Ceph endpoint) -- neither of which exists in DNS:

        error fetching virtual reference -> dispatch failure -> io error
          -> client error (Connect) -> dns error
          -> failed to lookup address information: Name or service not known

    The repo's own stored container config is correct (region eu-central-1,
    anonymous) -- the environment overrides it. Popping AWS_* *after* the
    process has already built an S3 client does NOT help; the config is cached
    process-wide on first use. So the worker must be scrubbed BEFORE it touches
    icechunk, which is what scrub_aws_env() + --restart do here. The client
    keeps its AWS_* -- the client is what writes the sink.

  * Resolving ANY chunk of a source array loads that array's ENTIRE icechunk
    manifest into RAM. refs = dates x members x steps x levels for the WHOLE
    era, regardless of what you select, and the in-memory cost is ~2000 B/ref
    (measured -- the ~200 B/ref in the docs is the packed on-disk size). On
    this cluster's 13.9 GB workers that means AT MOST ONE pressure-level
    manifest per worker. So one store variable is pinned to one worker for the
    whole run and the opened era is cached there: the manifest is paid for once
    (11-215 s) instead of once per date. Use --vars to run in tiers.

  * Reads want to be BIG, not many. One task holding all 5 `u` levels measured
    3060 messages in 176 s (21.6 msg/s). The same work split into 5 concurrent
    single-level tasks on the same worker took the cluster down -- 5 decode
    pipelines on top of a 6.2 GB resident manifest. Hence one task per
    variable (--levels-per-task 0) and a staggered manifest warm-up.

  * Cropping to East Africa does NOT reduce bytes read: a virtual chunk is one
    whole *global* GRIB message. Only the write side shrinks (~500x). The read
    is therefore the entire cost, and it is cross-cloud from EWC to AWS.

  * The `49r1/00z` group is a UNION of two schema sub-eras. `cape` is finite
    only before 2025-01-14, `mucape`/`tcw`/`ptype` only after, `tcc`/`sf`
    never. Those arrays open, have the right shape, and read back ALL-NaN.
    A run must not silently write an all-NaN channel: the first date fixes the
    channel schema (dropping all-NaN ones loudly) and every later date must
    match it exactly, or the run aborts.

Prerequisites -- the EWC credentials must be in the environment of the *client*
as well as the workers (workers already have them; the client shell may not):

    export AWS_ENDPOINT_URL=https://object-store.os-api.cci1.ecmwf.int
    export AWS_DEFAULT_REGION=RegionOne
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...

The source store is read ANONYMOUSLY with from_env=False, so those variables do
not leak into the source path.

Usage
-----
    P=/opt/mamba/envs/dask/bin/python

    # 0. what would it cost? pure arithmetic, touches nothing.
    #    Defaults are the production shape: 30 dates x 7-day lead (53 steps)
    #    x 51 members -> 9.1 GB written, 2.43 M messages read, ~5 h.
    $P materialize_ea_icechunk_ewc.py plan --days 30 --lead-days 7
    $P materialize_ea_icechunk_ewc.py plan --days 30 --cheap-members

    # 1. what is in the store right now, and is every channel finite?
    $P materialize_ea_icechunk_ewc.py probe --eras 50r1

    # 2. one date, one variable -- calibrates msg/s and proves the write path
    $P materialize_ea_icechunk_ewc.py run --days 1 --vars t2m --prefix ea/cal

    # 3. one date, all 30 channels -- the first honest per-date cost
    $P materialize_ea_icechunk_ewc.py run --days 1 --prefix ea-cgan/v1-1day

    # 4. the month
    $P materialize_ea_icechunk_ewc.py run --days 30 --prefix ea-cgan/v1-30day

    # 5. how big did it get?
    $P materialize_ea_icechunk_ewc.py size --prefix ea-cgan/v1-30day
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

# ── source: the published virtual store ──────────────────────────────────────
SRC_ENDPOINT = "https://data.source.coop"
SRC_BUCKET = "e4drr-project"
SRC_PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
SRC_CONTAINER = "s3://ecmwf-forecasts/"      # where the virtual chunks live

# ── sink: the EWC object store ───────────────────────────────────────────────
DST_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL",
                              "https://object-store.os-api.cci1.ecmwf.int")
DST_REGION = os.environ.get("AWS_DEFAULT_REGION", "RegionOne")
DST_BUCKET = "must-icechunk"

# Era windows, from the store's own time axes.
ERA_WINDOWS = {
    "0p4":  ("2023-01-18", "2024-02-28"),
    "49r1": ("2024-02-29", "2026-05-12"),
    "50r1": ("2026-05-13", None),
}
BREAK_13LEVEL = np.datetime64("2025-01-14")   # sub-era break inside 49r1

# ── domain: the 0.25 deg superset box (docs sec 5.2) ─────────────────────────
# Strictly contains the 0.1 deg TF training frame (-13.65..24.65 N,
# 19.15..54.25 E) and the 0.25 deg PyTorch EP box (-15..25 N, 20..53 E), with a
# 2-cell halo so 0.25 -> 0.1 interpolation never extrapolates. 163 x 147.
LAT_MAX, LAT_MIN = 25.25, -15.25       # store latitude DESCENDS 90 -> -90
LON_MIN, LON_MAX = 18.5, 55.0          # store longitude is 0..359.75, no wrap
NLAT, NLON = 163, 147

# ── lead times ───────────────────────────────────────────────────────────────
# The store's step axis is 3-hourly to 144 h, then 6-hourly to 360 h (85
# values). `--lead-days N` takes every published step out to N*24 h.
#
#   lead-days 7  -> 0..144 by 3h (49) + 150,156,162,168 (4)  = 53 steps
#   lead-days 2  -> 0..48  by 3h                             = 17 steps
#
# The narrower 24..54 h band (11 steps) is what the extraction doc specs for
# the cGAN's own accumulation windows; the full 7-day set is a superset of it.
STEPS_ALL_H = list(range(0, 145, 3)) + list(range(150, 361, 6))


def steps_for_lead(lead_days: float) -> list[int]:
    return [h for h in STEPS_ALL_H if h <= lead_days * 24]


STEPS_H = steps_for_lead(7)            # 53 values, 0 .. 168 h
STEPS_CGAN_H = list(range(24, 57, 3))  # 11 values, the accumulation-window set

MEMBERS_FULL = list(range(51))         # 0 = control, 1..50 perturbed
MEMBERS_CHEAP = [0] + list(range(1, 51, 5))    # control + 10 perturbed

# ── channel spec: (store_var, level|None, out_name, eras_populated) ──────────
ALL_ERAS = ("0p4", "49r1", "50r1")
E_2425 = ("49r1", "50r1")

SURFACE_SPEC = [
    ("tp",     None, "tp",     ALL_ERAS),   # accumulated from step 0
    ("tcwv",   None, "pw",     ALL_ERAS),
    ("sp",     None, "sp",     ALL_ERAS),
    ("msl",    None, "msl",    ALL_ERAS),
    ("t2m",    None, "t2m",    ALL_ERAS),
    ("skt",    None, "skt",    ALL_ERAS),   # -> SST proxy over water
    ("ssr",    None, "ssr",    E_2425),     # accumulated
    ("ttr",    None, "ttr",    E_2425),     # -> OLR = -ttr / dt
    ("tcw",    None, "tcw",    E_2425),     # 49r1: only from 2025-01-14
    ("cape",   None, "cape",   ("49r1",)),  # 49r1: only BEFORE 2025-01-14
    ("mucape", None, "mucape", E_2425),     # 49r1: only from 2025-01-14
]

PRESSURE_SPEC = [
    ("u",  925, "u925",  ALL_ERAS), ("v", 925, "v925", ALL_ERAS),
    ("u",  850, "u850",  ALL_ERAS), ("v", 850, "v850", ALL_ERAS),
    ("u",  700, "u700",  ALL_ERAS), ("v", 700, "v700", ALL_ERAS),
    ("u",  500, "u500",  ALL_ERAS), ("v", 500, "v500", ALL_ERAS),
    ("u",  200, "u200",  ALL_ERAS), ("v", 200, "v200", ALL_ERAS),
    ("gh", 500, "gh500", ALL_ERAS),
    ("w",  925, "w925",  E_2425),   ("w", 850, "w850", E_2425),   # no w in 0p4
    ("w",  700, "w700",  E_2425),   ("w", 500, "w500", E_2425),
    ("r",  850, "r850",  ALL_ERAS),                # the "RH 800" ask: 800 hPa
    ("r",  700, "r700",  ALL_ERAS),                #   does not exist, 850 is it
    ("t",  850, "t850",  ALL_ERAS),                # t850/700/500 + r850/700
    ("t",  700, "t700",  ALL_ERAS),                #   feed K-index and theta-w
    ("t",  500, "t500",  ALL_ERAS),
]

SPEC = SURFACE_SPEC + PRESSURE_SPEC

# Order store variables heaviest-manifest-first, so that when they are dealt
# round-robin to workers the big pressure-level manifests land one per worker.
PL_VARS = ["u", "v", "w", "r", "t", "gh"]

# Measured in-memory cost of one manifest chunk-ref on this cluster (RSS delta
# across the first read of an array, divided by that array's ref count), from
# three arrays spanning 0.22 M to 3.44 M refs:
#     50r1/t2m  0.22 M refs -> 0.52 GB, 2195 B/ref, manifest load  11 s
#     50r1/u    3.10 M refs -> 6.25 GB, 2008 B/ref, manifest load 179 s
#     49r1/t2m  3.44 M refs -> 6.95 GB, 2006 B/ref, manifest load 155 s
# The ~200 B/ref quoted in the docs is the packed on-disk figure and is ~10x
# too optimistic for sizing workers.
BYTES_PER_REF = 2000
ERA_SHAPE = {                      # (dates, members, steps, levels) per era
    "0p4":  (401, 51, 85, 9),
    "49r1": (794, 51, 85, 13),
    "50r1": (51,  51, 85, 14),     # still growing; `probe` reports the truth
}

# EA box size per era -- READ from the store's coordinate arrays, not assumed.
# 0p4 is a 0.4 deg grid, so the same lat/lon window is a much smaller array.
EA_CELLS = {
    "0p4":  102 * 91,              # 9,282   @ 0.40 deg
    "49r1": 163 * 147,             # 23,961  @ 0.25 deg
    "50r1": 163 * 147,             # 23,961  @ 0.25 deg
}

# The 49r1 group is a union of two schema sub-eras, split at 2025-01-14.
# Counted from the store's own time axis: 320 dates before, 474 after.
SUBERA_DATES = {"49r1-pre": 320, "49r1-post": 474}

# Corpus rows: (label, era, dates, channel-count). Channel counts come from the
# per-era availability in SURFACE_SPEC/PRESSURE_SPEC plus the sub-era rules:
#   0p4       22 = 6 sfc (no ssr/ttr/tcw/cape/mucape, none published) + 16 pl
#             (no `w` at all in 0p4)
#   49r1-pre  29 = 9 sfc (cape, but no tcw/mucape yet)               + 20 pl
#   49r1-post 30 = 10 sfc (mucape + tcw, cape now all-NaN)           + 20 pl
#   50r1      30 = 10 sfc                                            + 20 pl
CORPUS = [
    ("0p4",       "0p4",  401, 22),
    ("49r1-pre",  "49r1", 320, 29),
    ("49r1-post", "49r1", 474, 30),
    ("50r1",      "50r1",  51, 30),
]


def manifest_gb(era: str, pressure: bool, dates: int | None = None) -> float:
    d, m, s, lv = ERA_SHAPE[era]
    d = dates or d
    return d * m * s * (lv if pressure else 1) * BYTES_PER_REF / 1e9

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ea-ewc")


# ─────────────────────────────────────────────────────────────────────────────
# Worker side
# ─────────────────────────────────────────────────────────────────────────────
_ERA_CACHE: dict = {}


def scrub_aws_env():
    """Remove every AWS_* variable from this process.

    Must run on a worker BEFORE anything builds an S3 client, otherwise the
    EWC Ceph endpoint/region leak into the virtual-chunk fetch and it dies on
    a DNS lookup (see the module docstring). Returns what it removed so the
    driver can log that it actually happened.
    """
    import os
    removed = sorted(k for k in os.environ if k.startswith("AWS_"))
    for k in removed:
        os.environ.pop(k)
    return removed


def scrub_plugin():
    """Worker plugin that scrubs AWS_* as each worker comes up, including
    workers that join later, so a scaled-up cluster does not start failing.

    Built lazily: `distributed` is only imported by the cluster subcommands,
    and duck-typed plugins are rejected, so it must really subclass
    WorkerPlugin.
    """
    from distributed.diagnostics.plugin import WorkerPlugin

    class ScrubAWS(WorkerPlugin):
        name = "scrub-aws-env"

        def setup(self, worker=None):
            scrub_aws_env()

    return ScrubAWS()


def open_source_era(era: str):
    """Open one era group of the virtual store anonymously, cached per worker.

    The cache is the point: the first read of an array pulls that array's whole
    manifest, and holding the repo object keeps icechunk's manifest cache warm
    so every later date on this worker is free of that cost.
    """
    if era in _ERA_CACHE:
        return _ERA_CACHE[era]
    scrub_aws_env()             # belt and braces; the plugin should have run
    import icechunk
    import gribberish.zarr  # noqa: F401  registers the Zarr v3 GRIB codec

    storage = icechunk.s3_storage(
        bucket=SRC_BUCKET, prefix=SRC_PREFIX, endpoint_url=SRC_ENDPOINT,
        region="us-east-1",
        anonymous=True,         # public read of the store metadata
        from_env=False,         # do NOT pick up the EWC AWS_* credentials
        force_path_style=True,  # source.coop needs path-style addressing
    )
    auth = icechunk.containers_credentials(
        {SRC_CONTAINER: icechunk.s3_anonymous_credentials()})
    cfg = icechunk.RepositoryConfig.default()
    # Eager manifest preload draws source.coop's sporadic HTTP 500s and would
    # turn a transient error into a failed open. Go lazy.
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg,
                                    authorize_virtual_chunk_access=auth)
    ds = xr.open_zarr(repo.readonly_session("main").store, group=f"{era}/00z",
                      consolidated=False, zarr_format=3, decode_timedelta=True)
    _ERA_CACHE[era] = (repo, ds)
    return _ERA_CACHE[era]


def fetch_coords(era, steps_h):
    """The driver holds AWS_* (it writes the sink) and therefore cannot read
    the source itself -- so even the coordinate arrays for the output template
    have to come off a worker."""
    import numpy as np
    _, ds = open_source_era(era)
    lat = np.asarray(ds.latitude.values)
    lon = np.asarray(ds.longitude.values)
    return {
        "latitude": lat[(lat <= LAT_MAX) & (lat >= LAT_MIN)],
        "longitude": lon[(lon >= LON_MIN) & (lon <= LON_MAX)],
        "number": np.asarray(ds.number.values),
    }


def extract_var_day(era, store_var, levels, date, steps_h, members,
                    reduce_members=True, retries=3):
    """Read one store variable for one forecast date, crop to the EA box and
    reduce over `number` to mean + sd.

    One task = (one store variable, ALL its levels, one date). That is the unit
    that amortises the manifest: selecting all levels in a single .compute()
    issues one graph over the already-resident manifest instead of one per
    level.

    Returns {out_name: DataArray(step, stat, latitude, longitude)}.
    """
    import socket
    t0 = time.time()
    cold = era not in _ERA_CACHE
    _, ds = open_source_era(era)
    host = socket.gethostname()
    if store_var not in ds:
        return {"var": store_var, "status": "absent", "host": host,
                "seconds": round(time.time() - t0, 1), "data": {}}

    sel = dict(time=np.datetime64(date),
               number=members,
               step=[np.timedelta64(h, "h") for h in steps_h],
               latitude=slice(LAT_MAX, LAT_MIN),
               longitude=slice(LON_MIN, LON_MAX))
    da = ds[store_var].sel(**sel)
    level_values = [lv for lv, _ in levels]
    if level_values[0] is not None:
        da = da.sel(isobaricInhPa=level_values)

    last = None
    for attempt in range(retries):
        try:
            cube = da.compute()          # (number, step[, isobaricInhPa], y, x)
            break
        except Exception as exc:         # noqa: BLE001  source.coop 500s, S3 blips
            last = exc
            time.sleep(2 ** attempt)
    else:
        return {"var": store_var, "status": f"error:{type(last).__name__}",
                "detail": str(last)[:200], "host": host,
                "seconds": round(time.time() - t0, 1), "data": {}}

    stat = xr.DataArray(["mean", "sd"], dims="stat", name="stat")
    out, finite = {}, {}
    for level, name in levels:
        c = cube.sel(isobaricInhPa=level) if level is not None else cube
        if reduce_members:
            # (step, stat, y, x) -- the 2 channels per field load_fcst() reads
            stacked = (xr.concat([c.mean("number"), c.std("number")], dim=stat)
                       .transpose("step", "stat", "latitude", "longitude"))
        else:
            # (step, number, y, x) -- every member kept, 25.5x the bytes
            stacked = c.transpose("step", "number", "latitude", "longitude")
        stacked = stacked.astype("float32")
        drop = ("isobaricInhPa", "time", "valid_time")
        if reduce_members:
            drop += ("number",)
        for coord in drop:
            if coord in stacked.coords:
                stacked = stacked.drop_vars(coord)
        probe = stacked.sel(stat="mean") if reduce_members else stacked.isel(number=0)
        f = float(np.isfinite(probe.values).mean())
        # The all-NaN trap: the array exists for this era but was never
        # populated in this sub-era window. Report it; the driver decides.
        out[name] = stacked.rename(name)
        finite[name] = round(f, 3)

    def _rss_gb():
        try:
            with open("/proc/self/statm") as f:
                return round(int(f.read().split()[1]) * 4096 / 1e9, 2)
        except OSError:
            return None

    return {"var": store_var, "status": "ok", "host": host, "cold": cold,
            "seconds": round(time.time() - t0, 1), "rss_gb": _rss_gb(),
            "finite": finite, "data": out,
            "n_messages": len(members) * len(steps_h) * len(levels)}


# ─────────────────────────────────────────────────────────────────────────────
# Driver helpers
# ─────────────────────────────────────────────────────────────────────────────
def era_for(date) -> str:
    d = np.datetime64(str(date)[:10])
    for era, (lo, hi) in ERA_WINDOWS.items():
        if d >= np.datetime64(lo) and (hi is None or d <= np.datetime64(hi)):
            return era
    raise ValueError(f"{d} is outside every era window")


def channels_for_era(era: str, only_vars=None):
    """{store_var: [(level, out_name), ...]} for the channels published in
    `era`, ordered pressure-level first so the heavy manifests are dealt to
    distinct workers."""
    per_var = defaultdict(list)
    for store_var, level, out_name, eras_ok in SPEC:
        if era not in eras_ok:
            continue
        if only_vars and store_var not in only_vars:
            continue
        per_var[store_var].append((level, out_name))
    return {v: per_var[v] for v in
            sorted(per_var, key=lambda v: (v not in PL_VARS, PL_VARS.index(v)
                                           if v in PL_VARS else 0, v))}


def open_sink(prefix: str, create=True):
    import icechunk as ic
    storage = ic.s3_storage(bucket=DST_BUCKET, prefix=prefix, region=DST_REGION,
                            endpoint_url=DST_ENDPOINT, from_env=True,
                            force_path_style=True)
    return (ic.Repository.open_or_create(storage) if create
            else ic.Repository.open(storage))


def source_time_axis(era: str):
    """Ask the store what dates it actually has -- do not assume."""
    _, ds = open_source_era(era)
    return pd.to_datetime(ds.time.values)


def prepare_cluster(client, restart: bool):
    """Connect, make sure every worker is AWS_*-free before it touches
    icechunk, and report the shape we have to plan around."""
    # The workers and the scheduler have no copy of this file on their path.
    # Run as __main__ that is fine -- plain pickle cannot handle __main__
    # functions so dask falls back to cloudpickle, which serialises them by
    # value. But when this module is IMPORTED (from a notebook, or a wrapper)
    # plain pickle succeeds by reference on the client and then the scheduler
    # dies with ModuleNotFoundError. Ship the file so both entry points work.
    mod = sys.modules[__name__]
    if mod.__name__ != "__main__" and getattr(mod, "__file__", None):
        import cloudpickle
        cloudpickle.register_pickle_by_value(mod)
        try:
            client.upload_file(mod.__file__)
        except Exception as exc:                                # noqa: BLE001
            log.warning("upload_file failed (%s); if tasks die with "
                        "ModuleNotFoundError, run this as a script instead "
                        "of importing it", type(exc).__name__)

    n_expected = len(client.scheduler_info()["workers"])
    if restart:
        log.info("restarting workers so the AWS_* scrub lands on a process "
                 "that has not yet built an S3 client")
        client.restart(timeout=300)
        client.wait_for_workers(n_expected, timeout=300)

    # Covers workers that join mid-run. NOTE: this registration is STICKY --
    # the scheduler re-applies it to every worker that starts afterwards,
    # including workers for unrelated jobs, which would silently strip their
    # EWC credentials. release_cluster() must undo it; every entry point that
    # calls prepare_cluster() wraps its work in a try/finally to guarantee it.
    client.register_plugin(scrub_plugin())

    # Belt and braces for workers that were already up when the plugin landed.
    # Workers can still be settling right after a restart, so a dropped comm
    # here is not fatal: the plugin's setup() has already run on each of them,
    # and open_source_era() scrubs again before it touches icechunk.
    for attempt in range(3):
        try:
            removed = client.run(scrub_aws_env)
            seen = {tuple(v) for v in removed.values()}
            log.info("scrubbed from %d workers: %s", len(removed),
                     seen if seen != {()} else "(already clean)")
            break
        except Exception as exc:                                # noqa: BLE001
            log.warning("scrub sweep %d/3 failed (%s); the worker plugin has "
                        "already run, continuing", attempt + 1,
                        type(exc).__name__)
            time.sleep(5)

    winfo = client.scheduler_info()["workers"]
    gb = min(w["memory_limit"] for w in winfo.values()) / 1e9
    log.info("cluster: %d workers, %d threads, %.1f GB/worker", len(winfo),
             sum(w["nthreads"] for w in winfo.values()), gb)
    return sorted(winfo), gb


def preallocate(repo, chans, dates, steps_td, coords, store_members, log_=log):
    """Create the full (time, step, depth, lat, lon) schema up front, filled
    with NaN, so each (date, variable) can be written straight into its own
    region afterwards.

    Needed for --store-members: at 51 members a single date is ~7.8 GB, far
    too much to gather into the driver and append as one Dataset the way the
    mean+sd path does. Region writes let each variable land independently.

    The NaN fill costs almost nothing on disk -- zarr does not write a chunk
    that is entirely fill_value -- but it does create every array's metadata,
    which is what makes the region writes legal.
    """
    import dask.array as dsa
    from icechunk.xarray import to_icechunk

    ny, nx = len(coords["latitude"]), len(coords["longitude"])
    depth_name = "number" if store_members else "stat"
    depth_vals = (coords["number"] if store_members
                  else np.array(["mean", "sd"], dtype="<U4"))
    shape = (len(dates), len(steps_td), len(depth_vals), ny, nx)
    chunks = (1, len(steps_td), len(depth_vals), ny, nx)

    names = [n for lv in chans.values() for _, n in lv]
    template = xr.Dataset(
        {n: ((("time", "step", depth_name, "latitude", "longitude")),
             dsa.full(shape, np.nan, dtype="float32", chunks=chunks))
         for n in names},
        coords={"time": [np.datetime64(d) for d in dates],
                "step": steps_td, depth_name: depth_vals,
                "latitude": coords["latitude"],
                "longitude": coords["longitude"]},
    )
    logical = sum(v.nbytes for v in template.data_vars.values())
    log_.info("preallocating %d channels x %d dates = %.2f GB logical "
              "(NaN chunks are not written)", len(names), len(dates),
              logical / 1e9)
    session = repo.writable_session("main")
    to_icechunk(template, session, mode="w",
                encoding={n: {"chunks": chunks} for n in names})
    snap = session.commit(f"preallocate {len(names)} channels x {len(dates)} "
                          f"dates ({depth_name}={len(depth_vals)})")
    log_.info("schema committed %s", snap[:12])
    return depth_name


def warm_manifests(client, era, chans, pin, date, batch, timeout):
    """Load each variable's manifest once, a few at a time.

    Firing all 16 variables at a date simultaneously means 16 concurrent
    multi-GB manifest loads. Observed effect: the workers stop answering
    heartbeats and the scheduler drops the client mid-run
    ("scheduler-connection-lost"), losing every result in flight. Warming in
    small batches costs one cheap read per variable and makes the main loop
    start from an all-warm cluster.
    """
    order = list(chans)
    log.info("warming %d manifests in batches of %d (one cheap read each; "
             "expect 10-180 s per variable)", len(order), batch)
    for i in range(0, len(order), batch):
        group = order[i:i + batch]
        t0 = time.time()
        futs = {v: client.submit(
            extract_var_day, era, v, chans[v][:1], date, [24], [0],
            workers=[pin[v]], allow_other_workers=False, pure=False)
            for v in group}
        for v, f in futs.items():
            try:
                r = f.result(timeout=timeout)
                log.info("  warm %-8s %6.1fs  %s", v, r.get("seconds", -1),
                         r.get("status"))
            except Exception as exc:                            # noqa: BLE001
                log.warning("  warm %-8s FAILED %s: %s", v,
                            type(exc).__name__, str(exc)[:120])
        log.info("  batch %d/%d done in %.0fs", i // batch + 1,
                 (len(order) + batch - 1) // batch, time.time() - t0)


def release_cluster(client, restore=True):
    """Undo prepare_cluster()'s sticky state.

    The scrub plugin stays registered on the SCHEDULER until removed, so every
    worker that starts later -- including workers belonging to somebody else's
    job -- would come up with its EWC credentials stripped and be unable to
    reach must-icechunk. Always call this, even on failure.
    """
    try:
        client.unregister_worker_plugin("scrub-aws-env")
        log.info("unregistered the AWS_* scrub plugin")
    except Exception as exc:                                    # noqa: BLE001
        log.warning("could NOT unregister the scrub plugin (%s). Workers "
                    "starting from now on will have AWS_* stripped -- clear "
                    "it by hand with "
                    "client.unregister_worker_plugin('scrub-aws-env')",
                    type(exc).__name__)
        return
    if restore:
        # The running workers are still scrubbed; only a restart brings the
        # service environment back.
        try:
            n = len(client.scheduler_info()["workers"])
            client.restart(timeout=300)
            client.wait_for_workers(n, timeout=300)
            got = sum(1 for v in client.run(
                lambda: bool(os.environ.get("AWS_ACCESS_KEY_ID"))).values() if v)
            log.info("workers restarted with credentials restored: %d/%d",
                     got, n)
        except Exception as exc:                                # noqa: BLE001
            log.warning("restart to restore worker credentials failed (%s); "
                        "restart the cluster before running anything that "
                        "writes to must-icechunk", type(exc).__name__)


def check_manifest_budget(era: str, chans: dict, n_workers: int, worker_gb: float):
    """Refuse a run whose manifests cannot physically fit.

    A worker can hold roughly `worker_gb * 0.7` of manifest before dask starts
    spilling and then killing it. Each store variable pinned to a worker costs
    one whole manifest for the era.
    """
    cost = {v: manifest_gb(era, pressure=(v in PL_VARS)) for v in chans}
    budget = worker_gb * 0.7
    too_big = {v: g for v, g in cost.items() if g > budget}
    per_worker = defaultdict(float)
    for i, v in enumerate(sorted(cost, key=lambda k: -cost[k])):
        per_worker[i % n_workers] += cost[v]
    worst = max(per_worker.values()) if per_worker else 0.0
    return cost, too_big, worst, budget


# ─────────────────────────────────────────────────────────────────────────────
# plan -- arithmetic only, touches nothing
# ─────────────────────────────────────────────────────────────────────────────
def cmd_plan(args):
    era = args.era
    chans = channels_for_era(era, args.vars)
    n_chan = sum(len(v) for v in chans.values())
    n_mem = len(args.members)
    n_step = len(args.steps)

    cell = NLAT * NLON
    per_field_kb = cell * 4 / 1024
    per_day_mb = n_chan * n_step * 2 * cell * 4 / 1e6
    msgs_per_day = n_chan * n_mem * n_step

    print(f"\nplan  era={era}  days={args.days}  members={n_mem}  "
          f"steps={n_step} ({args.steps[0]}..{args.steps[-1]} h)")
    print(f"box   {LAT_MAX} .. {LAT_MIN} N,  {LON_MIN} .. {LON_MAX} E   "
          f"-> {NLAT} x {NLON} = {cell:,} cells, {per_field_kb:.1f} KB/field")
    print(f"chans {n_chan} from {len(chans)} store variables: "
          f"{', '.join(f'{k}x{len(v)}' for k, v in chans.items())}")

    print(f"\nWRITE (what lands in must-icechunk, uncompressed float32)")
    print(f"  per date         {per_day_mb:8.1f} MB   "
          f"({n_chan} chan x {n_step} step x 2 stat)")
    print(f"  {args.days:>3} dates        {per_day_mb*args.days/1000:8.2f} GB   "
          f"-> ~{per_day_mb*args.days/1000*0.6:.2f} GB after zstd (~40% off "
          f"on smooth met fields; measure, do not trust this)")
    print(f"  sink chunk-refs  {args.days*n_chan:8,}   "
          f"(1 chunk per date per channel -> the sink manifest is trivial, "
          f"which is the whole point of materialising)")

    print(f"\nREAD (from AWS s3://ecmwf-forecasts, one whole GLOBAL GRIB "
          f"message per member/step/level)")
    print(f"  messages/date    {msgs_per_day:8,}")
    print(f"  messages total   {msgs_per_day*args.days:8,}")
    for mb in (0.8, 2.0):
        tb = msgs_per_day * args.days * mb / 1e6
        print(f"    @ {mb} MB/msg    {tb:8.2f} TB read   "
              f"(write is {per_day_mb*args.days/1000/1000/tb*1e6:.0f}x smaller)")

    # Wall clock is NOT total_messages / total_threads: every store variable is
    # pinned to one worker (that is the memory plan), so the critical path is
    # the busiest worker, and within a worker the busiest thread. Level
    # splitting is what lets a 5-level variable use more than one thread.
    print(f"\nWALL CLOCK ({args.workers} workers x {args.threads} threads, "
          f"one task per variable, pinned)")
    # Each variable is one task on its pinned worker, and a worker runs its
    # variables concurrently on separate threads. So the critical path per date
    # is the single heaviest TASK -- the variable with the most levels -- not
    # total messages over total threads.
    per_worker = defaultdict(list)
    for i, v in enumerate(chans):                 # same order cmd_run pins in
        per_worker[i % args.workers].append(v)
    def worker_secs(vs, rate):
        msgs = [len(chans[v]) * n_mem * n_step for v in vs]
        msgs.sort(reverse=True)
        # threads run concurrently; beyond that, tasks queue
        waves = [msgs[i::args.threads] for i in range(args.threads)]
        return max((sum(w) for w in waves), default=0) / rate
    heaviest_var = max(chans, key=lambda v: len(chans[v]))
    for rate in (args.msg_rate / 2, args.msg_rate, args.msg_rate * 2):
        per_date = max(worker_secs(vs, rate) for vs in per_worker.values())
        print(f"    @ {rate:4.1f} msg/s/task   {per_date/60:6.1f} min/date  "
              f"-> {per_date*args.days/3600:6.2f} h for {args.days} dates")
    crit = len(chans[heaviest_var]) * n_mem * n_step
    print(f"    critical path is `{heaviest_var}` "
          f"({len(chans[heaviest_var])} levels x {n_mem} members x {n_step} "
          f"steps = {crit:,} messages in one task)")
    agg = msgs_per_day / max(worker_secs(max(per_worker.values(),
                             key=lambda vs: worker_secs(vs, args.msg_rate)),
                             args.msg_rate), 1)
    print(f"    implies ~{agg:.0f} msg/s aggregate = ~{agg*1.5:.0f} MB/s "
          f"sustained EWC<-AWS.")
    print(f"    If the link caps below that, BANDWIDTH sets the time, not the "
          f"task model.")

    print(f"\nSOURCE MANIFEST RAM ({BYTES_PER_REF} B/ref measured -- the "
          f"constraint that shapes the whole task graph)")
    d, m, s, lv = ERA_SHAPE[era]
    for kind, nlev in (("surface (2-D)", 1), (f"pressure ({lv} levels)", lv)):
        refs = d * m * s * nlev
        print(f"    {kind:24s} {refs/1e6:7.2f} M refs -> "
              f"{refs*BYTES_PER_REF/1e9:6.2f} GB in the worker that holds it")

    cost, too_big, worst, budget = check_manifest_budget(
        era, chans, args.workers, args.worker_gb)
    print(f"    per-worker budget        {budget:6.2f} GB "
          f"(70% of {args.worker_gb} GB)")
    print(f"    {len(chans)} store vars over {args.workers} workers -> worst "
          f"worker holds {worst:.2f} GB")
    if too_big:
        print(f"\n    *** INFEASIBLE on this cluster: "
              f"{', '.join(f'{v} needs {g:.1f} GB' for v, g in too_big.items())}")
        print(f"    A single manifest exceeds one worker. No amount of task "
              f"splitting helps -- \n    the fix is upstream: rebuild the "
              f"source store with icechunk.ManifestSplittingConfig\n    split "
              f"along `time`, so a read loads one shard, not the whole array.")
    elif worst > budget:
        print(f"\n    *** OVER BUDGET: run in tiers with --vars, e.g.\n"
              f"        --vars {' '.join(list(chans)[:args.workers])}\n"
              f"        --vars {' '.join(list(chans)[args.workers:])}")
    else:
        print(f"    -> fits: one pass, {len(chans)} vars pinned one-per-worker")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# corpus -- the whole published record, all three eras, both storage modes
# ─────────────────────────────────────────────────────────────────────────────
def cmd_corpus(args):
    n_step = len(steps_for_lead(args.lead_days))
    n_mem = 51
    # What multiplies the cell count on the write side: 51 individual members,
    # or the 2-channel mean+sd reduction the cGAN actually consumes.
    depth = n_mem if args.store_members else 2
    mode = f"{n_mem} individual members" if args.store_members else "mean+sd"

    print(f"\nFULL CORPUS  --  {mode}, lead {args.lead_days:g} d "
          f"({n_step} steps), all 51 members READ either way")
    print(f"box {LAT_MAX}..{LAT_MIN} N, {LON_MIN}..{LON_MAX} E\n")

    hdr = (f"{'group':11s} {'dates':>5s} {'ch':>3s} {'cells':>7s} "
           f"{'GB/date':>8s} {'WRITE TB':>9s} {'msgs M':>8s} {'READ TB':>8s} "
           f"{'hours':>7s}")
    print(hdr); print("-" * len(hdr))

    tot_w = tot_msg = tot_read = tot_h = tot_d = 0.0
    for label, era, dates, n_chan in CORPUS:
        cells = EA_CELLS[era]
        per_date_b = n_chan * n_step * depth * cells * 4
        write_tb = per_date_b * dates / 1e12
        msgs = dates * n_chan * n_mem * n_step
        # 0p4 messages are a 451x900 global field, 0.25 deg eras are 721x1440
        mb_per_msg = 1.5 * (cells / EA_CELLS["49r1"]) ** 0 * (
            0.39 if era == "0p4" else 1.0)
        read_tb = msgs * mb_per_msg / 1e6
        # Critical path per date is the heaviest single task: `u`, 5 levels.
        rate = args.msg_rate * (2.0 if era == "0p4" else 1.0)   # smaller fields
        hours = dates * (5 * n_mem * n_step / rate) / 3600
        print(f"{label:11s} {dates:5d} {n_chan:3d} {cells:7,d} "
              f"{per_date_b/1e9:8.2f} {write_tb:9.3f} {msgs/1e6:8.2f} "
              f"{read_tb:8.2f} {hours:7.1f}")
        tot_w += write_tb; tot_msg += msgs; tot_read += read_tb
        tot_h += hours; tot_d += dates
    print("-" * len(hdr))
    print(f"{'TOTAL':11s} {int(tot_d):5d} {'':3s} {'':7s} {'':8s} "
          f"{tot_w:9.3f} {tot_msg/1e6:8.2f} {tot_read:8.2f} {tot_h:7.1f}")

    print(f"\nWRITE   {tot_w*1000:.0f} GB uncompressed "
          f"(~{tot_w*1000*0.6:.0f} GB after zstd -- measure, do not trust)")
    if args.store_members:
        print(f"        storing mean+sd instead would be "
              f"{tot_w*1000*2/n_mem:.0f} GB, i.e. {n_mem/2:.1f}x smaller")
    else:
        print(f"        storing all {n_mem} members instead would be "
              f"{tot_w*1000*n_mem/2/1000:.1f} TB, i.e. {n_mem/2:.1f}x bigger")
    print(f"READ    {tot_msg/1e6:.0f} M GRIB messages, ~{tot_read:.0f} TB "
          f"-- IDENTICAL in both modes. The read is what costs; the ensemble\n"
          f"        reduction only changes what lands on disk.")
    print(f"TIME    {tot_h:.0f} h = {tot_h/24:.1f} days of continuous cluster "
          f"at the measured {args.msg_rate} msg/s")

    print(f"\nOUTPUT-STORE MANIFEST (the materialised store's own cost)")
    for chunk_desc, per_date_chunks in (
            ("chunk=(1 date, all steps, all members)", 1),
            ("chunk=(1 date, 1 step, all members)", n_step)):
        refs = sum(d * c * per_date_chunks for _, _, d, c in CORPUS)
        print(f"    {chunk_desc:42s} {refs/1e6:6.2f} M refs -> "
              f"{refs*BYTES_PER_REF/1e9:5.2f} GB")
    print(f"    -> past ~1 M refs the OUTPUT store starts to have the same\n"
          f"       manifest problem as the input. Split it per era (or per\n"
          f"       year) and write it with icechunk.ManifestSplittingConfig.")

    print(f"\nFEASIBILITY ON THIS CLUSTER ({args.workers} x {args.worker_gb} GB)")
    budget = args.worker_gb * 0.7
    done = set()
    for _, era, _, _ in CORPUS:
        if era in done:
            continue
        done.add(era)
        sfc, pl = manifest_gb(era, False), manifest_gb(era, True)
        ok_s = "fits" if sfc <= budget else "INFEASIBLE"
        ok_p = "fits" if pl <= budget else "INFEASIBLE"
        print(f"    {era:6s} surface {sfc:6.2f} GB {ok_s:10s} "
              f"pressure {pl:6.2f} GB {ok_p}")
    reachable = sum(d for _, e, d, _ in CORPUS
                    if manifest_gb(e, True) <= budget)
    print(f"    budget {budget:.2f} GB/worker -> {reachable} of {int(tot_d)} "
          f"dates ({reachable/tot_d*100:.0f}%) are extractable today.\n")


# ─────────────────────────────────────────────────────────────────────────────
# probe -- what is actually in the store, and is it finite?
# ─────────────────────────────────────────────────────────────────────────────
def cmd_probe(args):
    from dask.distributed import Client
    client = Client(args.scheduler, timeout=60)
    try:
        return _probe(args, client)
    finally:
        release_cluster(client, restore=not args.leave_scrubbed)
        client.close()


def _probe(args, client):
    waddrs, _ = prepare_cluster(client, args.restart)

    def axis(era):
        _, ds = open_source_era(era)
        t = pd.to_datetime(ds.time.values)
        return dict(era=era, n=len(t), first=str(t[0])[:10], last=str(t[-1])[:10],
                    sizes={k: int(v) for k, v in ds.sizes.items()},
                    levels=[int(x) for x in ds.isobaricInhPa.values],
                    data_vars=sorted(map(str, ds.data_vars)))

    for era in (args.eras or list(ERA_WINDOWS)):
        try:
            a = client.submit(axis, era, pure=False).result(timeout=900)
        except Exception as exc:                                # noqa: BLE001
            print(f"\n=== {era}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
            continue
        print(f"\n=== {era}/00z   {a['n']} dates  {a['first']} .. {a['last']}")
        print(f"    sizes  {a['sizes']}")
        print(f"    levels {a['levels']}")

        era = a["era"]
        probe_date = str(pd.to_datetime(a["last"]))[:10]
        chans = channels_for_era(era, args.vars)
        pin = {v: waddrs[i % len(waddrs)] for i, v in enumerate(chans)}
        print(f"    finiteness on {probe_date}, member 0, +24 h:")
        # Batched, not all at once: concurrent manifest loads have knocked the
        # client off the scheduler here before.
        order = list(chans)
        for i in range(0, len(order), args.batch):
            futs = {v: client.submit(extract_var_day, era, v, chans[v],
                                     probe_date, [24], [0], workers=[pin[v]],
                                     allow_other_workers=False, pure=False)
                    for v in order[i:i + args.batch]}
            for v, f in futs.items():
                try:
                    r = f.result(timeout=args.timeout)
                except Exception as exc:                        # noqa: BLE001
                    print(f"      {v:8s} FAILED {type(exc).__name__}: "
                          f"{str(exc)[:110]}", flush=True)
                    continue
                if r["status"] != "ok":
                    print(f"      {v:8s} {r['status']}  {r.get('detail','')}",
                          flush=True)
                    continue
                bad = [n for n, x in r["finite"].items() if x == 0.0]
                good = [f"{n}={x:.2f}" for n, x in r["finite"].items() if x > 0]
                print(f"      {v:8s} {r['seconds']:6.1f}s  "
                      f"ok: {' '.join(good) or '-'}"
                      + (f"   ALL-NaN: {', '.join(bad)}" if bad else ""),
                      flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# run -- the materialisation
# ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args):
    from dask.distributed import Client
    from icechunk.xarray import to_icechunk

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        log.error("no AWS_ACCESS_KEY_ID in this shell -- the CLIENT writes the "
                  "sink, not just the workers. Export the EWC credentials.")
        return 2

    client = Client(args.scheduler, timeout=60)
    try:
        return _run(args, client, to_icechunk)
    finally:
        # Sticky scheduler state: leaving the scrub plugin registered would
        # strip AWS_* from every worker that starts later, including other
        # people's jobs. Undo it whatever happened above.
        release_cluster(client, restore=not args.leave_scrubbed)
        client.close()


def _run(args, client, to_icechunk):
    waddrs, worker_gb = prepare_cluster(client, args.restart)
    user = os.environ.get("JUPYTERHUB_USER", "<user>")
    log.info("dashboard: https://jupyter-ewc-must.e4drr-cloud.work"
             "/user/%s/proxy/8787/status", user)

    # ── dates ────────────────────────────────────────────────────────────────
    if args.dates:
        dates = [pd.Timestamp(d) for d in args.dates]
    else:
        avail = client.submit(lambda e: pd.to_datetime(open_source_era(e)[1]
                                                       .time.values),
                              args.era, pure=False).result(timeout=900)
        end = pd.Timestamp(args.end) if args.end else avail[-1]
        avail = avail[avail <= end]
        dates = list(avail[-args.days:])
    eras = {era_for(d) for d in dates}
    if len(eras) > 1:
        log.error("dates span %s -- one run must stay inside one era, and "
                  "inside one 49r1 sub-era (break %s), or the channel schema "
                  "changes mid-store", eras, BREAK_13LEVEL)
        return 2
    era = eras.pop()
    if era == "49r1":
        side = {d < pd.Timestamp(str(BREAK_13LEVEL)) for d in dates}
        if len(side) > 1:
            log.error("dates straddle the 49r1 sub-era break %s: `cape` is "
                      "finite before it and `mucape` after. Split the run.",
                      BREAK_13LEVEL)
            return 2

    chans = channels_for_era(era, args.vars)
    n_chan = sum(len(v) for v in chans.values())
    log.info("era=%s  %d dates %s .. %s  %d channels from %d store vars  "
             "%d members  %d steps", era, len(dates), dates[0].date(),
             dates[-1].date(), n_chan, len(chans), len(args.members),
             len(args.steps))

    # ── pin one store variable to one worker, for the whole run ──────────────
    # Deals pressure-level variables first, so with >=6 workers each heavy
    # manifest (up to 9 GB on 49r1) lands on a distinct worker and is loaded
    # exactly once rather than once per date.
    cost, too_big, worst, budget = check_manifest_budget(
        era, chans, len(waddrs), worker_gb)
    if too_big and not args.force:
        log.error("a single manifest exceeds one worker: %s",
                  {v: f"{g:.1f} GB" for v, g in too_big.items()})
        log.error("budget is %.1f GB/worker. This cannot be fixed by splitting "
                  "tasks -- the source store needs ManifestSplittingConfig "
                  "along `time`. Run `plan --era %s` for the numbers, or "
                  "--vars to pick variables that do fit.", budget, era)
        return 2
    if worst > budget and not args.force:
        log.error("manifests do not fit: worst worker would hold %.1f GB of a "
                  "%.1f GB budget. Split the run with --vars.", worst, budget)
        return 2

    pin = {v: waddrs[i % len(waddrs)] for i, v in enumerate(chans)}
    load = defaultdict(list)
    for v, w in pin.items():
        load[w].append(v)
    log.info("variable -> worker pinning (manifest GB in brackets, %.1f GB "
             "budget/worker):", budget)
    for w in waddrs:
        vs = load[w]
        log.info("  %-24s %-26s %5.2f GB", w.split("/")[-1],
                 " ".join(f"{v}[{cost[v]:.1f}]" for v in vs) or "-",
                 sum(cost[v] for v in vs))

    if args.warmup:
        warm_manifests(client, era, chans, pin, str(dates[0].date()),
                       args.batch, args.timeout)

    repo = open_sink(args.prefix)
    log.info("sink: s3://%s/%s", DST_BUCKET, args.prefix)

    steps_td = pd.to_timedelta([f"{h}h" for h in args.steps])
    depth_name = "number" if args.store_members else "stat"
    if args.store_members:
        # 51 members is ~7.8 GB per date -- far too much to funnel through the
        # driver as one appended Dataset. Preallocate and region-write instead.
        coords = client.submit(fetch_coords, era, args.steps, pure=False
                               ).result(timeout=args.timeout)
        preallocate(repo, chans, dates, steps_td, coords, True)
        per_date_gb = (sum(len(v) for v in chans.values()) * len(args.steps)
                       * len(args.members) * NLAT * NLON * 4 / 1e9)
        log.info("store-members mode: ~%.2f GB/date, written per variable "
                 "into preallocated regions", per_date_gb)
    schema = None            # fixed by the first date, enforced on the rest
    written_mb = tot_msgs = 0
    per_date_secs = []
    t_start = time.time()

    # Optionally split multi-level variables into several tasks on the same
    # pinned worker (they share the one resident manifest and would use more
    # of that worker's threads).
    #
    # DEFAULT IS OFF, on evidence. One task holding all 5 `u` levels measured
    # 3060 messages in 176 s = 21.6 msg/s -- large reads amortise well. The
    # same work as 5 concurrent single-level tasks on one worker took the
    # cluster down instead: 5 decode pipelines on top of a 6.2 GB resident
    # manifest, worker killed, "cancelled for reason: scheduler-restart".
    # Bigger tasks are both faster and safer here; --levels-per-task is left
    # as a knob for a cluster with more headroom per worker.
    tasks = []
    for v, lv in chans.items():
        n = args.levels_per_task
        pieces = ([lv] if n <= 0 or len(lv) <= n
                  else [lv[j:j + n] for j in range(0, len(lv), n)])
        tasks.extend((v, p) for p in pieces)
    log.info("%d read tasks per date (%d store vars split by level, "
             "%d levels/task)", len(tasks), len(chans), args.levels_per_task)

    # Sliding window: keep `--lookahead` dates of reads in flight while the
    # commits, which must stay ordered, drain behind them.
    inflight: dict = {}

    def submit(d):
        # allow_other_workers MUST stay False. The pinning is not a scheduling
        # hint, it is the memory plan: letting dask relocate a task would load
        # a second multi-GB manifest onto a worker that already holds one, and
        # kill it. A queued task on its own worker is correct; a stolen one is
        # an OOM.
        inflight[d] = [(v, client.submit(
            extract_var_day, era, v, lvs, str(d.date()), args.steps,
            args.members, not args.store_members,
            key=f"x-{'-'.join(n for _, n in lvs)}-{d.date()}",
            workers=[pin[v]], allow_other_workers=False, pure=False))
            for v, lvs in tasks]

    for d in dates[:args.lookahead]:
        submit(d)

    for i, d in enumerate(dates):
        t0 = time.time()

        if args.store_members:
            # Stream: take each variable as it lands, write it into its region,
            # release it. Never holds more than one variable (~1.3 GB for a
            # 5-level one) instead of the whole 7.8 GB date.
            session = repo.writable_session("main")
            names, dropped, failed = [], [], []
            mb = 0.0
            write_s = 0.0
            peak = 0.0
            cold = []
            for v, f in inflight.pop(d):
                r = f.result(timeout=args.timeout)
                if r["status"] != "ok":
                    failed.append(f"{v}({r['status']})")
                    continue
                tot_msgs += r["n_messages"]
                peak = max(peak, r.get("rss_gb") or 0)
                if r.get("cold"):
                    cold.append(v)
                for name, da in r["data"].items():
                    if r["finite"][name] == 0.0:
                        dropped.append(name)
                        continue
                    one = da.expand_dims(time=[np.datetime64(d)]).rename(name)
                    one = one.to_dataset()
                    one = one.drop_vars([c for c in one.coords])
                    t1 = time.time()
                    to_icechunk(one, session, region={"time": slice(i, i + 1)})
                    write_s += time.time() - t1
                    mb += da.nbytes / 1e6
                    names.append(name)
                del r
            if failed:
                log.error("date %s: tasks failed: %s", d.date(), failed)
                return 3
            got = tuple(sorted(names))
            if schema is None:
                schema = got
                if dropped:
                    log.warning("all-NaN, not written: %s",
                                ", ".join(sorted(set(dropped))))
                log.info("schema %d channels: %s", len(schema), " ".join(schema))
            snap = session.commit(f"{era} {d.date()} ({len(names)} channels, "
                                  f"{len(args.members)} members)")
            written_mb += mb
            per_date_secs.append(time.time() - t0)
            log.info("[%3d/%d] %s  total %6.1fs  write %5.1fs  %7.1f MB  "
                     "peak worker %4.1f GB  snapshot %s%s", i + 1, len(dates),
                     d.date(), time.time() - t0, write_s, mb, peak, snap[:12],
                     f"  (cold: {' '.join(cold)})" if cold else "")
            nxt = i + args.lookahead
            if nxt < len(dates):
                submit(dates[nxt])
            continue

        results = [(v, f.result(timeout=args.timeout))
                   for v, f in inflight.pop(d)]
        read_s = time.time() - t0

        arrays, dropped, failed = {}, [], []
        for v, r in results:
            if r["status"] != "ok":
                failed.append(f"{v}({r['status']})")
                continue
            tot_msgs += r["n_messages"]
            for name, da in r["data"].items():
                if r["finite"][name] == 0.0:
                    dropped.append(name)
                    continue
                arrays[name] = da.expand_dims(time=[np.datetime64(d)])
        if failed:
            log.error("date %s: tasks failed: %s", d.date(), failed)
            return 3

        got = tuple(sorted(arrays))
        if schema is None:
            schema = got
            if dropped:
                log.warning("dropping %d all-NaN channel(s) -- not published "
                            "in this sub-era: %s", len(dropped),
                            ", ".join(sorted(dropped)))
            log.info("schema fixed at %d channels: %s", len(schema),
                     " ".join(schema))
        elif got != schema:
            # Appending a different variable set would corrupt the store's
            # shape silently. Refuse.
            log.error("date %s has channels %s, first date had %s -- refusing "
                      "to append a schema change", d.date(),
                      sorted(set(got) ^ set(schema)), len(schema))
            return 3

        dsd = xr.Dataset(arrays).assign_coords(step=steps_td)
        dsd.attrs.update(
            source=f"s3://{SRC_BUCKET}/{SRC_PREFIX} [{era}/00z]",
            created_by="materialize_ea_icechunk_ewc.py",
            members=str(args.members), n_members=len(args.members),
            box=f"{LAT_MAX}/{LAT_MIN}N {LON_MIN}/{LON_MAX}E",
            note="stat=[mean,sd] over ensemble `number`; tp/ssr/ttr are "
                 "ACCUMULATED from step 0 -- difference consecutive steps "
                 "downstream")
        mb = sum(v.nbytes for v in dsd.data_vars.values()) / 1e6

        session = repo.writable_session("main")
        t1 = time.time()
        if i == 0:
            # One chunk per (date, all steps, both stats, whole box) ~ 2.1 MB.
            # Take the spatial size from the data, not the constants: if the
            # box ever lands differently the encoding must follow it.
            ny, nx = dsd.sizes["latitude"], dsd.sizes["longitude"]
            if (ny, nx) != (NLAT, NLON):
                log.warning("box is %dx%d, expected %dx%d -- check the slice "
                            "against the store's coordinate order", ny, nx,
                            NLAT, NLON)
            enc = {n: {"chunks": (1, len(args.steps), 2, ny, nx)}
                   for n in dsd.data_vars}
            to_icechunk(dsd, session, mode="w", encoding=enc)
        else:
            to_icechunk(dsd, session, append_dim="time")
        snap = session.commit(f"{era} {d.date()} "
                              f"({len(arrays)} channels, {len(args.members)} members)")
        write_s = time.time() - t1

        written_mb += mb
        per_date_secs.append(time.time() - t0)
        cold = sorted({v for v, r in results if r.get("cold")})
        peak = max((r.get("rss_gb") or 0) for _, r in results)
        log.info("[%3d/%d] %s  read %5.1fs  write %4.1fs  %5.1f MB  "
                 "peak worker %4.1f GB  snapshot %s%s", i + 1, len(dates),
                 d.date(), read_s, write_s, mb, peak, snap[:12],
                 f"  (cold manifests: {' '.join(cold)})" if cold else "")

        nxt = i + args.lookahead
        if nxt < len(dates):
            submit(dates[nxt])

    wall = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"dates          {len(dates)}   {dates[0].date()} .. {dates[-1].date()}")
    print(f"channels       {len(schema)}")
    print(f"members        {len(args.members)}   steps {len(args.steps)}")
    print(f"GRIB messages  {tot_msgs:,} decoded  "
          f"({tot_msgs/wall:.1f}/s, {wall/max(tot_msgs,1)*len(waddrs)*4:.2f} "
          f"s/msg/slot)")
    print(f"written        {written_mb/1000:.2f} GB uncompressed "
          f"({written_mb/len(dates):.1f} MB/date)")
    print(f"wall clock     {wall/60:.1f} min  ({np.mean(per_date_secs):.1f} "
          f"s/date, median {np.median(per_date_secs):.1f} s)")
    print(f"extrapolated   30 dates -> {np.median(per_date_secs)*30/3600:.2f} h,"
          f" {written_mb/len(dates)*30/1000:.2f} GB")
    print(f"               90 dates -> {np.median(per_date_secs)*90/3600:.2f} h,"
          f" {written_mb/len(dates)*90/1000:.2f} GB   (one MAM season)")
    print(f"               276 dates -> {np.median(per_date_secs)*276/3600:.2f} h,"
          f" {written_mb/len(dates)*276/1000:.2f} GB   (MAM x 3)")
    print(f"sink           s3://{DST_BUCKET}/{args.prefix}")
    print(f"{'='*72}\n")
    log.info("measure the on-disk size with:  size --prefix %s", args.prefix)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# size -- what the sink actually costs on the object store
# ─────────────────────────────────────────────────────────────────────────────
def cmd_size(args):
    import s3fs
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": DST_ENDPOINT})
    root = f"{DST_BUCKET}/{args.prefix}"
    by_kind, total, n = defaultdict(int), 0, 0
    for f in fs.find(root, detail=True).values():
        sz = f.get("size", 0) or 0
        rel = f["Key"].removeprefix(root + "/")
        by_kind[rel.split("/")[0]] += sz
        total += sz
        n += 1
    print(f"\ns3://{root}")
    for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v/1e6:10.1f} MB")
    print(f"  {'TOTAL':14s} {total/1e6:10.1f} MB in {n:,} objects")

    try:
        repo = open_sink(args.prefix, create=False)
        ds = xr.open_zarr(repo.readonly_session("main").store,
                          consolidated=False, zarr_format=3, chunks={})
        raw = sum(v.nbytes for v in ds.data_vars.values())
        print(f"\n  logical      {raw/1e6:10.1f} MB uncompressed float32")
        print(f"  ratio        {raw/max(total,1):10.2f} x")
        print(f"  sizes        {dict(ds.sizes)}")
        print(f"  channels     {len(ds.data_vars)}: "
              f"{' '.join(sorted(map(str, ds.data_vars)))}")
        print(f"  per date     {total/max(int(ds.sizes['time']),1)/1e6:.1f} MB "
              f"on disk\n")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  (could not open as a repo: {type(exc).__name__}: "
              f"{str(exc)[:120]})\n")


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--scheduler", default=os.environ.get(
            "DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"))
        sp.add_argument("--vars", nargs="*", default=None,
                        help="restrict to these STORE variables, e.g. u v w -- "
                             "use to run in manifest-sized tiers")
        sp.add_argument("--no-restart", dest="restart", action="store_false",
                        help="skip the worker restart. Only safe if the "
                             "workers have never touched icechunk in this "
                             "process -- see the AWS_* note in the docstring")
        sp.add_argument("--leave-scrubbed", action="store_true",
                        help="skip the closing restart that restores AWS_* on "
                             "the workers (the scrub plugin is unregistered "
                             "either way)")
        sp.add_argument("--batch", type=int, default=3,
                        help="how many manifests to load at once. Firing all "
                             "16 has dropped the scheduler connection")
        sp.add_argument("--timeout", type=int, default=3600)

    sp = sub.add_parser("plan", help="sizing arithmetic, touches nothing")
    sp.add_argument("--era", default="50r1", choices=list(ERA_WINDOWS))
    sp.add_argument("--days", type=int, default=30, help="forecast DATES")
    sp.add_argument("--lead-days", type=float, default=7,
                    help="lead time to cover; 7 -> 53 steps, 0..168 h")
    sp.add_argument("--steps", type=int, nargs="+", default=None)
    sp.add_argument("--members", type=int, nargs="+", default=MEMBERS_FULL)
    sp.add_argument("--cheap-members", action="store_true")
    sp.add_argument("--vars", nargs="*", default=None)
    sp.add_argument("--workers", type=int, default=6)
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--worker-gb", type=float, default=13.9)
    sp.add_argument("--msg-rate", type=float, default=21.6,
                    help="messages/s per task, measured warm (see the plan doc)")

    sp = sub.add_parser("corpus",
                        help="whole published record, all eras, both modes")
    sp.add_argument("--lead-days", type=float, default=7)
    sp.add_argument("--store-members", action="store_true",
                    help="store all 51 members instead of mean+sd")
    sp.add_argument("--workers", type=int, default=6)
    sp.add_argument("--worker-gb", type=float, default=13.9)
    sp.add_argument("--msg-rate", type=float, default=21.6)

    sp = sub.add_parser("probe", help="live store: dates, levels, finiteness")
    common(sp)
    sp.add_argument("--eras", nargs="*", default=None)

    sp = sub.add_parser("run", help="materialise into the EWC store")
    common(sp)
    sp.add_argument("--prefix", required=True,
                    help="path under must-icechunk, e.g. ea-cgan/v1-7day")
    sp.add_argument("--era", default="50r1", choices=list(ERA_WINDOWS))
    sp.add_argument("--days", type=int, default=7,
                    help="take the LAST n dates available in --era")
    sp.add_argument("--end", default=None, help="last date to consider")
    sp.add_argument("--dates", nargs="*", default=None,
                    help="explicit YYYY-MM-DD list, overrides --days/--end")
    sp.add_argument("--lead-days", type=float, default=7,
                    help="keep every published step out to N*24 h "
                         "(7 -> 53 steps, 0..168 h)")
    sp.add_argument("--steps", type=int, nargs="+", default=None,
                    help="explicit step hours, overrides --lead-days")
    sp.add_argument("--cgan-steps", action="store_true",
                    help="just the 24..54 h accumulation-window set (11 steps)")
    sp.add_argument("--members", type=int, nargs="+", default=MEMBERS_FULL)
    sp.add_argument("--cheap-members", action="store_true",
                    help="control + every 5th perturbed (11) instead of 51")
    sp.add_argument("--store-members", action="store_true",
                    help="store all 51 members (dim `number`) instead of "
                         "reducing to mean+sd. 25.5x the bytes, identical "
                         "read cost, and switches the write path from "
                         "append-per-date to preallocate + region writes")
    sp.add_argument("--levels-per-task", type=int, default=0,
                    help="levels per read task. 0 (default) = one task per "
                         "variable, all its levels together -- MEASURED at "
                         "21.6 msg/s. Splitting to 1 was measured to kill the "
                         "scheduler on this cluster; see the plan doc 4.1")
    sp.add_argument("--lookahead", type=int, default=2,
                    help="dates of reads kept in flight ahead of the commits")
    sp.add_argument("--no-warmup", dest="warmup", action="store_false",
                    help="skip the staggered manifest warm-up")
    sp.add_argument("--force", action="store_true",
                    help="run even if the manifest budget says it will not fit")

    sp = sub.add_parser("size", help="how big did the sink get")
    sp.add_argument("--prefix", required=True)

    args = p.parse_args(argv)
    if getattr(args, "cheap_members", False):
        args.members = MEMBERS_CHEAP
    # Only the subcommands that actually read a step axis carry --lead-days;
    # `probe` and `size` have neither and must not be given one.
    if getattr(args, "cgan_steps", False):
        args.steps = STEPS_CGAN_H
    elif getattr(args, "steps", None) is None and hasattr(args, "lead_days"):
        args.steps = steps_for_lead(args.lead_days)
    return {"plan": cmd_plan, "corpus": cmd_corpus, "probe": cmd_probe,
            "run": cmd_run, "size": cmd_size}[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
