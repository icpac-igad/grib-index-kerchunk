"""Daily ECMWF IFS ensemble -> East Africa mean/sd, as a Frisky futures DAG.

This is the Option-A-shaped design from NEXT_SESSION.md §5, but built on Frisky
futures instead of a Dask collection graph. The three rules it is built to obey:

  §4.2  one task = one GRIB message, and nothing oversubscribes a worker
  §4.3  NO `.compute()` anywhere inside a task -- leaves read numpy directly,
        so no zarr `sync()` event loop is ever nested inside a worker thread
  §4.4  every read is wrapped in 503/SlowDown backoff

DAG shape, per date.  Every leaf is independent; the only fan-in is the
reduction over `number`, which is exactly what the science asks for:

    read(m=0, step=s) ---.
    read(m=1, step=s) ----+--> reduce_members(step=s) --.
    ...                  /                               +--> stack_steps(channel)
    read(m=50, step=s) -'                               /          |
    ... one such fan-in per step ----------------------'           v
                                                              mean/sd arrays

Counts for the production shape (30 channels x 51 members x 53 steps):
81,090 leaves + 1,590 reductions + 30 stacks per date.  Dask pays ~1 ms of
scheduler overhead per task on that (~80 s/date of pure bookkeeping); Frisky
pays ~3 us (~0.25 s).  That is not the main reason to use it here -- see the
memory note in the header of `run()` -- but it is not nothing.

Usage
-----
    P=/var/lib/private/nishadhka/cGAN_tutorial/icechunk-virtual-dask/frisky/.venv/bin/python

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

LAT_MAX, LAT_MIN = 25.25, -15.25       # store latitude DESCENDS
LON_MIN, LON_MAX = 18.5, 55.0

# (store_var, level|None, out_name) -- ordered heaviest-manifest-last so a small
# --channels N still exercises a mix of surface and pressure-level reads.
CHANNELS = [
    ("t2m", None, "t2m"), ("tp", None, "tp"), ("msl", None, "msl"),
    ("tcwv", None, "pw"), ("sp", None, "sp"), ("skt", None, "skt"),
    ("ssr", None, "ssr"), ("ttr", None, "ttr"), ("tcw", None, "tcw"),
    ("mucape", None, "mucape"),
    ("u", 925, "u925"), ("v", 925, "v925"), ("u", 850, "u850"),
    ("v", 850, "v850"), ("u", 700, "u700"), ("v", 700, "v700"),
    ("u", 500, "u500"), ("v", 500, "v500"), ("u", 200, "u200"),
    ("v", 200, "v200"), ("gh", 500, "gh500"),
    ("w", 925, "w925"), ("w", 850, "w850"), ("w", 700, "w700"),
    ("w", 500, "w500"),
    ("r", 850, "r850"), ("r", 700, "r700"),
    ("t", 850, "t850"), ("t", 700, "t700"), ("t", 500, "t500"),
]

STEPS_ALL_H = list(range(0, 145, 3)) + list(range(150, 361, 6))


# ===========================================================================
# Worker-side.  Module-level functions only -- Frisky pickles them by
# reference, so nothing large rides along with each of the 81,090 tasks.
# ===========================================================================

_DS_CACHE: dict = {}          # per worker PROCESS, not per task


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
    ds = xr.open_zarr(sess.store, group=f"{era}/00z", consolidated=False,
                      zarr_format=3, decode_timedelta=True, chunks=None)
    _DS_CACHE[era] = ds
    return ds


def read_message(era, var, level, date, member, step_h, synthetic=False):
    """ONE GRIB message -> the East Africa box as float32.  The unit of work.

    Measured at 0.23-0.25 s warm on a single machine (NEXT_SESSION.md §2).
    Holds one global message (~4 MB) transiently and returns ~0.1 MB.
    """
    if synthetic:
        rng = np.random.default_rng(abs(hash((var, member, step_h))) % 2**32)
        time.sleep(0.02)
        return rng.standard_normal((163, 147), dtype=np.float32)

    import numpy as _np

    ds = _open_era(era)
    da = ds[var]
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
        longitude=slice(LON_MIN, LON_MAX),
    )

    # §4.4 -- icechunk surfaces AWS throttling as `unhandled error (SlowDown)`
    # and does not retry.  One 503 anywhere otherwise fails the whole read.
    delay = 0.5
    for attempt in range(6):
        try:
            return _np.asarray(da.values, dtype=_np.float32)
        except Exception as exc:
            msg = str(exc)
            throttled = ("SlowDown" in msg or "503" in msg
                         or "ServiceUnavailable" in msg)
            if not throttled or attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def reduce_members(*fields):
    """Fan-in over `number`.  The only reduction in the whole job.

    Takes ~51 x 0.1 MB and returns 2 x 0.1 MB, so the peak a worker holds for
    one reduction is ~5 MB.  Memory is bounded by construction, which matters
    because neither Dask nor Frisky can see icechunk's Rust allocations.
    """
    stack = np.stack(fields, axis=0)
    return (stack.mean(axis=0, dtype=np.float32),
            stack.std(axis=0, dtype=np.float32))


def stack_members(*fields):
    """Keep every member: (number, y, x) for one (channel, step).

    The alternative fan-in to `reduce_members`, used when the store is to hold
    individual members.  ~5 MB out instead of ~0.2 MB, still bounded.
    """
    return np.stack(fields, axis=0)


def stack_steps(*per_step):
    """(step, y, x) mean and sd for one channel on one date."""
    return (np.stack([m for m, _ in per_step], axis=0),
            np.stack([s for _, s in per_step], axis=0))


# ===========================================================================
# Client-side: build the graph, submit once, gather.
# ===========================================================================

def build_and_submit(client, date, channels, members, steps, era, synthetic,
                     keep_members=False):
    """One submission per date.

    Returns {out_name: Future[(mean, sd)]}, or with keep_members
    {out_name: [Future[(number, y, x)] per step]} -- a LIST, deliberately.

    The list matters for memory.  A whole channel with members kept is
    51 x 53 x 163 x 147 float32 = 259 MB, and 30 of those is 7.8 GB against an
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
    import numpy as _np

    if synthetic:
        return {"step": _np.array([_np.timedelta64(h, "h") for h in steps]),
                "number": _np.array(members),
                "latitude": _np.linspace(LAT_MAX, LAT_MIN, 163),
                "longitude": _np.linspace(LON_MIN, LON_MAX, 147)}

    ds = _open_era(era)
    box = ds["t2m"].sel(latitude=slice(LAT_MAX, LAT_MIN),
                        longitude=slice(LON_MIN, LON_MAX))
    return {
        "step": _np.array([_np.timedelta64(h, "h") for h in steps]),
        "number": _np.array(members),
        "latitude": box.latitude.values,
        "longitude": box.longitude.values,
    }


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

    keep = args.store_members
    repo = coords = None
    if args.sink:
        import sink_icechunk as sink
        here = os.path.dirname(os.path.abspath(__file__))
        ak, sk = sink.load_env(args.env or os.path.join(here, "..", ".env"))
        repo, created = sink.open_sink(args.sink, ak, sk)
        print(f"sink     must-icechunk/{args.sink}"
              f"  ({'created' if created else 'existing'})")
        coords = subset_coords(args.era, args.date, members, steps,
                               args.synthetic)

    t0 = time.time()
    written, failed, results = [], [], {}
    try:
        futures = build_and_submit(client, args.date, channels, members, steps,
                                   args.era, args.synthetic, keep_members=keep)
        t_submit = time.time() - t0
        print(f"submitted in {t_submit:.2f}s -- now waiting\n")

        first = True
        for name, fut in futures.items():
            try:
                if keep:
                    # Gather ONE channel: 53 futures of (number, y, x) ->
                    # (step, number, y, x) = 259 MB.  Written and dropped
                    # before the next channel is touched, so the client peak
                    # stays ~0.5 GB against the 8 GiB cgroup.
                    arr = np.stack([f.result() for f in fut], axis=0)
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

    for name, (mean, sd) in results.items():
        finite = float(np.isfinite(mean).mean())
        print(f"  {name:7s} mean{mean.shape}  finite {finite:.3f}  "
              f"min/mean/max {np.nanmin(mean):9.2f} {np.nanmean(mean):9.2f} "
              f"{np.nanmax(mean):9.2f}   sd_mean {np.nanmean(sd):8.3f}")
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
                   help="how many of the 30 channels (start at 1 -- §4.5)")
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
                        "ea-cgan/v1-7day.  Requires --store-members")
    p.add_argument("--store-members", action="store_true",
                   help="keep every member instead of reducing to mean/sd.  "
                        "51x53x163x147 f32 = 259 MB per channel, 7.8 GB/date")
    p.add_argument("--first-date", action="store_true",
                   help="this date lays down the schema (mode=w); omit to "
                        "append along `time`")
    p.add_argument("--env", default=None,
                   help="path to the .env holding AK/SK (default ../.env)")
    p.add_argument("--synthetic", action="store_true",
                   help="no network: prove the DAG shape only")
    p.add_argument("--ramp", action="store_true",
                   help="run at 1, 2, 4, 8, 16, 30 channels and stop at the break")
    p.add_argument("--count", action="store_true",
                   help="print the DAG size and exit; opens nothing, submits "
                        "nothing")
    args = p.parse_args()

    if args.sink and not args.store_members:
        p.error("--sink writes the member dimension; pass --store-members")

    if args.count:
        sys.exit(0 if count_only(args) else 1)

    if not args.ramp:
        sys.exit(0 if run(args) else 1)

    for n in (1, 2, 4, 8, 16, 30):
        if n > len(CHANNELS):
            break
        print(f"\n{'=' * 72}\n  RAMP: {n} channel(s)\n{'=' * 72}")
        args.channels = n
        if not run(args):
            print(f"\nbroke at {n} channels.  That number localises it.")
            sys.exit(1)
    print("\nramp clean to 30 channels.")


if __name__ == "__main__":
    main()
