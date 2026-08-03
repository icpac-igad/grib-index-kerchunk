"""End-to-end realization smoke test: published virtual store -> EWC S3.

Takes the exact opening recipe from OPENING_PUBLISHED_ECMWF_ICECHUNK.ipynb,
reads an East Africa subset of the real ECMWF IFS ensemble, and writes it back
as a *realized* Icechunk store on the EWC object store. Then reads it back and
verifies.

This is the calibration step the plan documents keep pointing at. It is what
turns three assumptions into measurements:

    * seconds per GRIB message, on the real access pattern
    * the compression ratio of the written store (assumed 40% everywhere)
    * whether the write path works at all on real data rather than synthetic

Credentials come from ./.env (AK / SK), which is gitignored. The source store
is read ANONYMOUSLY -- AWS_* must NOT leak into it, or the virtual chunk fetch
resolves the Ceph endpoint and dies on DNS. This script never exports AWS_*
into the process environment for that reason; it passes the sink credentials
explicitly instead.

Usage
-----
    P=/opt/mamba/envs/dask/bin/python

    # smallest useful: 1 date, all 30 channels, 4 members x 4 steps
    $P realize_smoke_test.py --date 2026-06-01 --members 4 --steps 4

    # a fuller single date -- this is the real per-date calibration
    $P realize_smoke_test.py --date 2026-06-01 --members 51 --steps 53

    # leave it behind for inspection instead of deleting
    $P realize_smoke_test.py --date 2026-06-01 --keep
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import xarray as xr

# ── source: exactly the notebook's recipe ────────────────────────────────────
SRC_ENDPOINT = "https://data.source.coop"
SRC_BUCKET = "e4drr-project"
SRC_PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
SRC_CONTAINER = "s3://ecmwf-forecasts/"

# ── sink: EWC Ceph ───────────────────────────────────────────────────────────
DST_ENDPOINT = "https://object-store.os-api.cci1.ecmwf.int"
DST_REGION = "RegionOne"
DST_BUCKET = "must-icechunk"

# ── East Africa superset box, 163 x 147 at 0.25 deg ──────────────────────────
LAT_MAX, LAT_MIN = 25.25, -15.25       # store latitude DESCENDS
LON_MIN, LON_MAX = 18.5, 55.0

# The 30 predictor channels: (store_var, level|None, out_name)
CHANNELS = [
    ("tp", None, "tp"), ("tcwv", None, "pw"), ("sp", None, "sp"),
    ("msl", None, "msl"), ("t2m", None, "t2m"), ("skt", None, "skt"),
    ("ssr", None, "ssr"), ("ttr", None, "ttr"), ("tcw", None, "tcw"),
    ("mucape", None, "mucape"),
    ("u", 925, "u925"), ("v", 925, "v925"),
    ("u", 850, "u850"), ("v", 850, "v850"),
    ("u", 700, "u700"), ("v", 700, "v700"),
    ("u", 500, "u500"), ("v", 500, "v500"),
    ("u", 200, "u200"), ("v", 200, "v200"),
    ("gh", 500, "gh500"),
    ("w", 925, "w925"), ("w", 850, "w850"),
    ("w", 700, "w700"), ("w", 500, "w500"),
    ("r", 850, "r850"), ("r", 700, "r700"),
    ("t", 850, "t850"), ("t", 700, "t700"), ("t", 500, "t500"),
]

# Full 7-day lead: 3-hourly to 144 h then 6-hourly. 53 values.
STEPS_ALL = list(range(0, 145, 3)) + [150, 156, 162, 168]


def load_env(path=".env"):
    """AK / SK for the sink. Never exported into os.environ -- see docstring."""
    env = {}
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        env[k.strip().replace("export ", "")] = v.strip().strip('"').strip("'")
    ak = env.get("AK") or env.get("AWS_ACCESS_KEY_ID")
    sk = env.get("SK") or env.get("AWS_SECRET_ACCESS_KEY")
    if not ak or not sk:
        sys.exit(f"{path}: need AK and SK (or AWS_ACCESS_KEY_ID/SECRET)")
    return ak, sk


def open_source():
    """Verbatim from the notebook, including the manifest-preload workaround."""
    import icechunk
    import gribberish.zarr  # noqa: F401  registers the Zarr v3 GRIB codec

    storage = icechunk.s3_storage(
        bucket=SRC_BUCKET, prefix=SRC_PREFIX, endpoint_url=SRC_ENDPOINT,
        region="us-east-1", anonymous=True, from_env=False,
        force_path_style=True)
    auth = icechunk.containers_credentials(
        {SRC_CONTAINER: icechunk.s3_anonymous_credentials()})
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg,
                                    authorize_virtual_chunk_access=auth)
    return xr.open_zarr(repo.readonly_session("main").store, group="50r1/00z",
                        consolidated=False, zarr_format=3,
                        decode_timedelta=True)


def read_channel_on_worker(var, lev, name, date, members, steps_h,
                           retries=8):
    """Runs ON A DASK WORKER. Opens the source, reads one channel, crops to
    East Africa, returns only the small cropped array.

    Why on a worker: the JupyterHub client is capped at 8 GiB (memory.max on
    jupyter-<user>.service) and one pressure-level read costs ~6.5 GB
    resident -- it gets SIGKILLed. Workers have 13.94 GB. What comes back
    over the wire is members x steps x 163 x 147 x 4 B.

    AWS_* handling, and it is fiddly:

      * It must be ABSENT for the WHOLE task, not just the open. The virtual
        chunk container's object-store client is not built until the first
        virtual chunk is actually fetched -- inside compute(), not inside
        open_source(). Restoring the environment right after the open (an
        earlier version) meant the fetch built its client against the EWC
        Ceph endpoint, and every read died with
        "error fetching virtual reference".

      * It must be PUT BACK when the task ends. An even earlier version
        popped it permanently, silently stripping the workers' Ceph
        credentials for every other job on the cluster.

    So: scrub for the duration, restore in a finally that wraps everything.
    """
    import os
    saved = {k: os.environ[k] for k in list(os.environ)
             if k.startswith("AWS_")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        import numpy as np, time
        g = globals()
        if "_SRC" not in g:
            g["_SRC"] = open_source()
        src = g["_SRC"]
        t0 = time.time()
        da = src[var].sel(time=np.datetime64(date), number=members,
                          step=[np.timedelta64(h, "h") for h in steps_h],
                          latitude=slice(LAT_MAX, LAT_MIN),
                          longitude=slice(LON_MIN, LON_MAX))
        if lev is not None:
            da = da.sel(isobaricInhPa=lev)
        # AWS returns 503 SlowDown under load and ICECHUNK DOES NOT RETRY IT --
        # it surfaces as `unhandled error (SlowDown)`. Retrying here with
        # exponential backoff plus jitter is what makes the read survive.
        import random
        slowdowns = 0
        for attempt in range(retries):
            try:
                cube = da.compute().transpose("number", "step",
                                              "latitude", "longitude")
                break
            except Exception as exc:                            # noqa: BLE001
                text = str(exc)
                if "SlowDown" not in text and "Connect" not in text:
                    raise
                slowdowns += 1
                if attempt == retries - 1:
                    raise
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
        for c in ("isobaricInhPa", "time", "valid_time"):
            if c in cube.coords:
                cube = cube.drop_vars(c)
        with open("/proc/self/statm") as f:
            rss = round(int(f.read().split()[1]) * 4096 / 1e9, 2)
        return {"name": name, "seconds": time.time() - t0, "rss_gb": rss,
                "slowdowns": slowdowns,
                "finite": float(np.isfinite(cube.values).mean()),
                "mean": float(np.nanmean(cube.values)),
                "cube": cube.astype("float32")}
    finally:
        os.environ.update(saved)      # leave the worker as we found it


def open_sink(ak, sk, prefix):
    """Create the realized store, splitting manifests one shard per date --
    the same layout the source store uses. Set it explicitly rather than
    relying on a default."""
    import icechunk
    storage = icechunk.s3_storage(
        bucket=DST_BUCKET, prefix=prefix, region=DST_REGION,
        endpoint_url=DST_ENDPOINT, access_key_id=ak, secret_access_key=sk,
        from_env=False, force_path_style=True)
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict(
            {icechunk.ManifestSplitCondition.AnyArray():
                {icechunk.ManifestSplitDimCondition.DimensionName("time"): 1}}))
    return icechunk.Repository.open_or_create(storage, config=cfg)


def index_message_bytes(date, step_h):
    """True bytes per GRIB message for our channels, from ECMWF's .index
    sidecar. Cheap (~2 MB) and it means the ingress figure is measured rather
    than assumed."""
    d = date.replace("-", "")
    url = (f"https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
           f"{d}/00z/ifs/0p25/enfo/{d}000000-{step_h}h-enfo-ef.index")
    sfc = {"tp", "tcwv", "sp", "msl", "2t", "skt", "ssr", "ttr", "tcw", "mucape"}
    pl = {"u": {925, 850, 700, 500, 200}, "v": {925, 850, 700, 500, 200},
          "w": {925, 850, 700, 500}, "r": {850, 700},
          "t": {850, 700, 500}, "gh": {500}}
    try:
        raw = urllib.request.urlopen(url, timeout=120).read()
    except Exception as e:                                      # noqa: BLE001
        return None, f"{type(e).__name__}"
    lens = []
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p, lev = r.get("param"), r.get("levelist")
        if (r.get("levtype") == "sfc" and p in sfc) or (
                p in pl and lev is not None and int(lev) in pl[p]):
            lens.append(int(r["_length"]))
    return (sum(lens) / len(lens) / 1e6, len(lens)) if lens else (None, "none")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-06-01")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4,
                    help="how many of the 53 lead times to take")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--keep", action="store_true",
                    help="leave the store behind instead of deleting it")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--dask", action="store_true",
                    help="run the reads on the dask workers instead of this "
                         "client. Strongly recommended: the client is capped "
                         "at 8 GiB and a pressure-level read needs ~6.5 GB")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-workers", type=int, default=6,
                    help="how many workers hit AWS at once. Lower it if the "
                         "bucket is returning 503 SlowDown")
    args = ap.parse_args()

    prefix = args.prefix or f"realize-smoke/{args.date}"
    members = list(range(args.members))
    steps_h = STEPS_ALL[:args.steps]
    ak, sk = load_env(args.env)

    print(f"date     {args.date}")
    print(f"channels {len(CHANNELS)}   members {len(members)}   "
          f"steps {len(steps_h)} ({steps_h[0]}..{steps_h[-1]} h)")
    n_msg = len(CHANNELS) * len(members) * len(steps_h)
    print(f"messages {n_msg:,}")
    print(f"sink     s3://{DST_BUCKET}/{prefix}\n")

    mb_per_msg, n_idx = index_message_bytes(args.date, steps_h[-1])
    if mb_per_msg:
        print(f"index    {n_idx} matching messages, mean {mb_per_msg:.3f} MB "
              f"-> expect ~{n_msg * mb_per_msg / 1000:.2f} GB read\n")

    arrays, timings = {}, []
    per_ch = len(members) * len(steps_h)

    if args.dask:
        import os as _os
        from dask.distributed import Client
        client = Client(_os.environ.get("DASK_SCHEDULER_ADDRESS",
                                        "tcp://127.0.0.1:8786"), timeout=60)
        import cloudpickle
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        w = sorted(client.scheduler_info()["workers"])
        print(f"dask: {len(w)} workers, "
              f"{sum(x['nthreads'] for x in client.scheduler_info()['workers'].values())}"
              f" threads -- reads run THERE, not on this 8 GiB client\n")
        t_read0 = time.time()
        # WAVES, one channel per worker at a time. A pressure-level read costs
        # ~6.5 GB resident and a worker has 4 threads against a 13.94 GB
        # limit -- submitting all 30 channels at once let dask run 4
        # concurrently per worker, which OOM-killed them, bounced the nanny,
        # and destroyed the run's futures. One heavy read per worker is the
        # only safe concurrency here.
        nw = min(len(w), args.max_workers)
        w = w[:nw]
        waves = [CHANNELS[i:i + nw] for i in range(0, len(CHANNELS), nw)]
        print(f"  {len(waves)} waves of <={nw} channels "
              f"(1 per worker; ~6.5 GB each, {nw * 6.5:.0f} GB across the "
              f"cluster)\n")
        done = 0
        for wi, wave in enumerate(waves, 1):
            futs = [(name, client.submit(
                        read_channel_on_worker, var, lev, name, args.date,
                        members, steps_h, workers=[w[j]],
                        allow_other_workers=False, pure=False))
                    for j, (var, lev, name) in enumerate(wave)]
            for name, f in futs:
                done += 1
                try:
                    r = f.result(timeout=args.timeout)
                except Exception as e:                          # noqa: BLE001
                    msg = " ".join(str(e).split())
                    print(f"  [{done:2d}/{len(CHANNELS)}] {name:7s} FAILED "
                          f"{type(e).__name__}: {msg[:280]}")
                    continue
                timings.append((name, r["seconds"], per_ch))
                arrays[name] = r["cube"].expand_dims(
                    time=[np.datetime64(args.date)]).rename(name)
                print(f"  [{done:2d}/{len(CHANNELS)}] {name:7s} "
                      f"{r['seconds']:6.1f}s  {per_ch/r['seconds']:5.1f} msg/s  "
                      f"RSS {r['rss_gb']:5.2f} GB  "
                      f"503s {r.get('slowdowns', 0):2d}  "
                      f"finite {r['finite']:.2f}  mean {r['mean']:9.2f}")
            print(f"    -- wave {wi}/{len(waves)} done "
                  f"({time.time()-t_read0:.0f}s elapsed)")
        read_s = time.time() - t_read0
        client.close()
    else:
        t_open = time.time()
        src = open_source()
        print(f"source opened in {time.time() - t_open:.1f}s  "
              f"({len(src.data_vars)} vars)\n")
        print("WARNING: client-side reads. This session is capped at 8 GiB and "
              "a pressure-level\n         read costs ~6.5 GB -- expect SIGKILL. "
              "Use --dask.\n")
        sel = dict(time=np.datetime64(args.date), number=members,
                   step=[np.timedelta64(h, "h") for h in steps_h],
                   latitude=slice(LAT_MAX, LAT_MIN),
                   longitude=slice(LON_MIN, LON_MAX))
        t_read0 = time.time()
        for i, (var, lev, name) in enumerate(CHANNELS, 1):
            t0 = time.time()
            da = src[var].sel(**sel)
            if lev is not None:
                da = da.sel(isobaricInhPa=lev)
            try:
                cube = da.compute()
            except Exception as e:                              # noqa: BLE001
                print(f"  [{i:2d}/{len(CHANNELS)}] {name:7s} FAILED "
                      f"{type(e).__name__}: {str(e)[:80]}")
                continue
            dt = time.time() - t0
            timings.append((name, dt, per_ch))
            cube = cube.transpose("number", "step", "latitude", "longitude")
            for c in ("isobaricInhPa", "time", "valid_time"):
                if c in cube.coords:
                    cube = cube.drop_vars(c)
            arrays[name] = cube.astype("float32").expand_dims(
                time=[np.datetime64(args.date)]).rename(name)
            print(f"  [{i:2d}/{len(CHANNELS)}] {name:7s} {dt:6.1f}s  "
                  f"{per_ch/dt:5.1f} msg/s  finite "
                  f"{float(np.isfinite(cube.values).mean()):.2f}  "
                  f"mean {float(np.nanmean(cube.values)):9.2f}")
        read_s = time.time() - t_read0

    if not arrays:
        sys.exit("\nnothing read -- is the AWS path throttled again?")

    ds = xr.Dataset(arrays)
    logical_mb = sum(v.nbytes for v in ds.data_vars.values()) / 1e6
    print(f"\nread   {len(arrays)}/{len(CHANNELS)} channels in {read_s:.0f}s "
          f"({sum(t[2] for t in timings)/read_s:.1f} msg/s aggregate)")
    print(f"cube   {dict(ds.sizes)}  ->  {logical_mb:.1f} MB float32")

    # ── write ────────────────────────────────────────────────────────────────
    repo = open_sink(ak, sk, prefix)
    from icechunk.xarray import to_icechunk
    ny, nx = ds.sizes["latitude"], ds.sizes["longitude"]
    enc = {n: {"chunks": (1, len(members), len(steps_h), ny, nx)}
           for n in ds.data_vars}
    t0 = time.time()
    session = repo.writable_session("main")
    to_icechunk(ds, session, mode="w", encoding=enc)
    snap = session.commit(f"realize smoke test {args.date} "
                          f"({len(arrays)} channels, {len(members)} members, "
                          f"{len(steps_h)} steps)")
    write_s = time.time() - t0
    print(f"write  {logical_mb:.1f} MB in {write_s:.1f}s "
          f"({logical_mb/write_s:.0f} MB/s)  snapshot {snap[:12]}")

    # ── read back and verify ─────────────────────────────────────────────────
    t0 = time.time()
    back = xr.open_zarr(repo.readonly_session("main").store,
                        consolidated=False, zarr_format=3)
    same = set(back.data_vars) == set(ds.data_vars)
    probe = sorted(ds.data_vars)[0]
    a = ds[probe].values
    b = back[probe].values
    match = bool(np.allclose(a, b, equal_nan=True))
    print(f"verify readback in {time.time()-t0:.1f}s  vars match {same}  "
          f"`{probe}` values match {match}")

    # ── on-disk size, and what it implies ────────────────────────────────────
    import s3fs
    fs = s3fs.S3FileSystem(key=ak, secret=sk,
                           client_kwargs={"endpoint_url": DST_ENDPOINT,
                                          "region_name": DST_REGION})
    found = fs.find(f"{DST_BUCKET}/{prefix}", detail=True)
    disk_mb = sum((f.get("size") or 0) for f in found.values()) / 1e6
    ratio = logical_mb / disk_mb if disk_mb else float("nan")
    print(f"\n{'='*70}")
    print(f"on disk        {disk_mb:8.1f} MB in {len(found):,} objects")
    print(f"logical        {logical_mb:8.1f} MB float32")
    print(f"COMPRESSION    {ratio:8.2f}x   "
          f"({100*(1-1/ratio):.0f}% saved)  <- was assumed 40%")
    if mb_per_msg:
        read_gb = n_msg * mb_per_msg / 1000
        print(f"read           {read_gb:8.2f} GB for {logical_mb/1000:.2f} GB "
              f"logical  ({read_gb*1000/logical_mb:.0f}x amplification)")

    # extrapolate to a full date and a month, at the measured message rate
    rate = sum(t[2] for t in timings) / read_s
    full_msg = len(CHANNELS) * 51 * 53
    full_mb = logical_mb * (51/len(members)) * (53/len(steps_h))
    print(f"\nEXTRAPOLATION at the measured {rate:.1f} msg/s aggregate")
    print(f"  one full date (51 members x 53 steps, {full_msg:,} msgs)")
    print(f"    read   {full_msg/rate/3600:6.2f} h    "
          f"store {full_mb/1000:6.2f} GB logical / "
          f"{full_mb/1000/ratio:5.2f} GB on disk")
    print(f"  June 2026 (30 dates)")
    print(f"    read   {30*full_msg/rate/3600:6.2f} h    "
          f"store {30*full_mb/1000:6.2f} GB logical / "
          f"{30*full_mb/1000/ratio:5.2f} GB on disk")
    print(f"  NOTE: single-process here. The dask cluster runs ~16 of these")
    print(f"        concurrently, so divide the read time accordingly -- if")
    print(f"        AWS does not throttle (HTTP 503 SlowDown).")
    print(f"{'='*70}")

    if not args.keep:
        fs.rm(f"{DST_BUCKET}/{prefix}", recursive=True)
        print(f"\ncleaned up s3://{DST_BUCKET}/{prefix}")
    else:
        print(f"\nkept s3://{DST_BUCKET}/{prefix}")


if __name__ == "__main__":
    main()
