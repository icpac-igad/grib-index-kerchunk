"""Single-date read test, built on the pattern that is already known to work.

Adapted from grib-index-kerchunk/ecmwf/icechunk-par/test_dask_read.py (local
cluster) and test_dask_read_ewc.py (EWC cluster), both of which pass. The point
here is to find out *where* that pattern stops working as the variable count
grows — because the working tests read ONE variable and our failing extraction
reads sixteen.

Why the pattern matters, and why our extraction diverged from it
---------------------------------------------------------------
The working tests do:

    ds  = xr.open_zarr(..., chunks={})        # lazy: 1 dask task per GRIB msg
    sub = ds["t2m"].isel(step=slice(0, n))
    client.compute([sub.mean("number"), sub.std("number")])

Dask schedules a task per chunk and a tree reduction over `number`; each chunk
is released as soon as it has been folded in. Peak worker RSS stays inside a
2 GB limit, which those tests assert.

`realize_smoke_test.py` instead did, inside one dask task per channel:

    da.sel(number=<all>, step=<all>).compute()   # materialise EVERYTHING
    then reduce

which holds members x steps x lat x lon plus every decode buffer live at once,
and issues that whole burst of chunk requests from a single task. Both the
~7 GB RSS and the 503 SlowDown are plausible consequences of *that*, not of
anything inherent to the store or the link. This script tests both readings
directly, on ONE date, with `--eager` to reproduce the bad pattern for
comparison.

What it reports
---------------
    chunks/s, wall time, per-worker PEAK RSS (ru_maxrss, not current), whether
    any worker was OOM-killed, and a count of 503 SlowDown responses seen.

Usage
-----
    P=/opt/mamba/envs/dask/bin/python

    # start here: 1 variable, tiny, on the EWC cluster
    $P test_single_date.py --vars t2m --members 4 --steps 2

    # ramp the variable count -- the actual hypothesis under test
    $P test_single_date.py --ramp --members 4 --steps 2

    # THREADED scheduler, in this process -- no pickling, no scheduler, no
    # nannies. Icechunk's docs say the multithreaded scheduler is the one that
    # works; this is the cheapest way to test that claim.
    $P test_single_date.py --cluster threads --threads 8 --ramp

    # local cluster instead (bounded by the 8 GiB jupyter cgroup, so small)
    $P test_single_date.py --cluster local --vars t2m --members 4 --steps 2

    # reproduce the broken eager pattern for comparison
    $P test_single_date.py --vars t2m --members 4 --steps 2 --eager
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

SRC_ENDPOINT = "https://data.source.coop"
SRC_BUCKET = "e4drr-project"
SRC_PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
SRC_CONTAINER = "s3://ecmwf-forecasts/"

LAT_MAX, LAT_MIN = 25.25, -15.25
LON_MIN, LON_MAX = 18.5, 55.0

# Ordered so the ramp adds surface variables first, then pressure-level ones.
RAMP = ["t2m", "sp", "msl", "skt", "tp", "tcwv", "u", "v"]

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def open_store():
    """The notebook's recipe. AWS_* must be absent before the first S3 client
    is built, or the EWC Ceph endpoint hijacks the virtual-chunk fetch."""
    for k in [k for k in list(os.environ) if k.startswith("AWS_")]:
        os.environ.pop(k)
    import icechunk
    import xarray as xr
    import gribberish.zarr  # noqa: F401

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
    # NOTE: deliberately NOT chunks={}.
    #
    # chunks={} turns the WHOLE array into a dask array, so the graph is built
    # over every chunk in the store *before* any selection is applied:
    #
    #     t2m  221,085 store chunks ->   665,966 graph tasks  (1.4 s)
    #     u  3,097,290 store chunks -> ~9,300,000 graph tasks (never finishes)
    #
    # That is the hang. It happens in the CLIENT, during graph construction,
    # before anything is submitted -- which is why the dashboard shows no
    # movement and why it made no difference whether the scheduler was
    # distributed or threaded.
    #
    # Opening without `chunks` gives lazily-indexed xarray (not dask), so
    # .sel() is a cheap metadata operation. Call .chunk() AFTER selecting to
    # parallelise just the subset -- see build_graph().
    return xr.open_zarr(repo.readonly_session("main").store, group="50r1/00z",
                        consolidated=False, zarr_format=3,
                        decode_timedelta=True)


def worker_peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def reset_peak_rss():
    """ru_maxrss is a high-water mark for the life of the process, so it can
    only be reset by restarting the worker. Report that we cannot reset it
    rather than silently comparing against a stale peak."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_graph(ds, names, date, members, steps_h, eager=False):
    """Lazy reduction per variable. Returns (graph_items, n_chunks)."""
    import xarray as xr
    sel = dict(time=np.datetime64(date), number=list(range(members)),
               step=[np.timedelta64(h, "h") for h in steps_h],
               latitude=slice(LAT_MAX, LAT_MIN),
               longitude=slice(LON_MIN, LON_MAX))
    items, n_chunks = [], 0
    for name in names:
        da = ds[name]
        # Select the LEVEL FIRST, while this is still lazy xarray and the
        # selection costs nothing. Doing it after .sel() built the graph
        # across all 14 levels.
        if "isobaricInhPa" in da.dims:
            da = da.isel(isobaricInhPa=0)
        da = da.sel(**sel)
        # Only now make it a dask array -- over the SUBSET, so the graph is
        # members x steps tasks rather than every chunk in the store.
        da = da.chunk({"number": 1, "step": 1})
        n_chunks += members * len(steps_h)
        if eager:
            items.append(("EAGER", name, da))
        else:
            # mean+std over `number` -- the shape the cGAN consumes, and the
            # shape that lets dask stream instead of materialising.
            items.append(("mean", name, da.mean("number")))
            items.append(("std", name, da.std("number")))
    return items, n_chunks


def run_one(client, ds, names, args):
    """One measurement: N variables, one date. Returns a dict of results."""
    items, n_chunks = build_graph(ds, names, args.date, args.members,
                                  args.steps_h, eager=args.eager)
    before = client.run(worker_peak_rss_mb)
    t0 = time.time()
    err = None
    try:
        if args.eager:
            # Deliberately the BAD pattern: materialise each variable whole.
            out = [da.compute() for _, _, da in items]
        else:
            out = client.compute([g for _, _, g in items], sync=True)
    except Exception as e:                                      # noqa: BLE001
        err = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"
        out = []
    wall = time.time() - t0
    after = client.run(worker_peak_rss_mb)
    alive = len(client.scheduler_info()["workers"])

    finite = None
    if out:
        try:
            finite = float(np.mean([np.isfinite(np.asarray(o)).mean()
                                    for o in out]))
        except Exception:                                       # noqa: BLE001
            finite = None
    return {
        "n_vars": len(names), "vars": ",".join(names), "n_chunks": n_chunks,
        "wall": wall, "chunks_s": n_chunks / wall if wall else 0,
        "peak_before_mb": max(before.values()) if before else 0,
        "peak_after_mb": max(after.values()) if after else 0,
        "alive": alive, "finite": finite, "err": err,
        "slowdown": bool(err and "SlowDown" in err),
    }


class ThreadedShim:
    """Stand-in for a distributed Client, for dask's THREADED scheduler.

    Why this mode exists: icechunk's own Dask guide says the distributed and
    multiprocessing schedulers do not work with the standard zarr write path
    and that "it is fine with the multithreaded scheduler". Reads are not
    documented either way, but the shape of that limitation points at one
    session per process being the assumption the library is built around --
    which is exactly what a client-side store pickled across six worker
    processes violates.

    Running in-process removes, in one move: pickling the store, the
    scheduler, the nannies, worker OOM kills, orphaned `processing` tasks and
    zombie clients. It keeps the parallelism, because the threaded scheduler
    still runs `num_workers` chunk reads at once and the GIL is released
    during the S3 fetch and the Rust GRIB decode.

    Exposes only the three methods run_one() needs, so that code is unchanged.
    """

    def __init__(self, nthreads):
        self.nthreads = nthreads

    def run(self, fn):                       # client.run(fn) -> {addr: result}
        return {"in-process": fn()}

    def scheduler_info(self):
        return {"workers": {"in-process": {"id": "in-process",
                                           "nthreads": self.nthreads,
                                           "memory_limit": _cgroup_limit_bytes(),
                                           "metrics": {}}}}

    def compute(self, graphs, sync=True):
        import dask
        with dask.config.set(scheduler="threads", num_workers=self.nthreads):
            return list(dask.compute(*graphs))

    def close(self):
        pass


def _cgroup_limit_bytes():
    """The cap that actually kills us. `free` reports the host, which is 32 GB
    and misleading; the JupyterHub session is capped far lower."""
    import pathlib
    for pat in ("/sys/fs/cgroup/memory.max",
                f"/sys/fs/cgroup/system.slice/jupyter-{os.environ.get('JUPYTERHUB_USER','')}.service/memory.max"):
        try:
            v = pathlib.Path(pat).read_text().strip()
            if v.isdigit():
                return int(v)
        except OSError:
            pass
    return 0


def make_client(args):
    if args.cluster == "threads":
        lim = _cgroup_limit_bytes()
        print(f"THREADED scheduler, in-process: {args.threads} threads, "
              f"no scheduler, no workers, nothing pickled")
        if lim:
            print(f"  process memory cap: {lim/2**30:.1f} GiB (cgroup)")
            if lim < 12 * 2**30:
                print(f"  ! this looks like the JupyterHub session cap. For a "
                      f"full 30-channel run,\n    run this ON A WORKER VM "
                      f"(16.77 GB) instead.")
        return ThreadedShim(args.threads), None

    from dask.distributed import Client
    if args.cluster == "local":
        from dask.distributed import LocalCluster
        # The whole jupyter session is capped at 8 GiB (memory.max), and that
        # cap covers the client AND any LocalCluster workers. Stay well under.
        cluster = LocalCluster(n_workers=args.workers,
                               threads_per_worker=args.threads,
                               processes=True, memory_limit=args.memory_limit,
                               dashboard_address=None)
        print(f"LocalCluster: {args.workers} procs x {args.threads} threads "
              f"x {args.memory_limit}  (jupyter cgroup is 8 GiB total)")
        return Client(cluster), cluster
    c = Client(args.scheduler, timeout=60)
    info = c.scheduler_info()["workers"]
    print(f"EWC cluster: {len(info)} workers, "
          f"{sum(w['nthreads'] for w in info.values())} threads, "
          f"{list(info.values())[0]['memory_limit']/1e9:.1f} GB each")
    return c, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-06-01")
    ap.add_argument("--vars", nargs="+", default=["t2m"])
    ap.add_argument("--ramp", action="store_true",
                    help="run 1, 2, 4, 8 variables in turn -- the hypothesis")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--eager", action="store_true",
                    help="reproduce the materialise-everything pattern that "
                         "realize_smoke_test.py used, for comparison")
    ap.add_argument("--cluster", choices=["ewc", "local", "threads"],
                    default="ewc",
                    help="ewc = distributed cluster (currently hangs); "
                         "local = LocalCluster processes; "
                         "threads = dask threaded scheduler IN THIS PROCESS "
                         "-- no pickling, no scheduler, no nannies")
    ap.add_argument("--scheduler", default=os.environ.get(
        "DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"))
    ap.add_argument("--workers", type=int, default=3, help="local only")
    ap.add_argument("--threads", type=int, default=2,
                    help="threads per worker (local), or total threads "
                         "(--cluster threads; try 8)")
    ap.add_argument("--memory-limit", default="1.5GB", help="local only")
    args = ap.parse_args()
    args.steps_h = list(range(0, 3 * args.steps, 3))

    print(f"date {args.date}   members {args.members}   steps {args.steps} "
          f"{args.steps_h}   pattern "
          f"{'EAGER (materialise all)' if args.eager else 'LAZY (chunks={}, '
             'reduce in graph)'}\n")

    client, cluster = make_client(args)
    try:
        ds = client.submit(open_store, pure=False).result(timeout=900) \
            if args.cluster == "ewc" else open_store()   # local/threads: here
    except Exception as e:                                      # noqa: BLE001
        # Opening on a worker returns an unpicklable handle in some setups;
        # fall back to opening client-side, which is fine for graph building
        # (the actual chunk reads still happen on workers).
        print(f"  (opening via worker failed: {type(e).__name__}; "
              f"opening client-side)")
        ds = open_store()

    sets = ([RAMP[:n] for n in (1, 2, 4, 8)] if args.ramp
            else [args.vars])

    print(f"\n{'vars':>4s} {'chunks':>7s} {'wall':>7s} {'chunk/s':>8s} "
          f"{'peakRSS':>9s} {'alive':>5s} {'finite':>7s}  status")
    print("-" * 88)
    rows = []
    for names in sets:
        r = run_one(client, ds, names, args)
        rows.append(r)
        status = ("503 SlowDown" if r["slowdown"]
                  else (r["err"][:40] if r["err"] else "ok"))
        print(f"{r['n_vars']:4d} {r['n_chunks']:7d} {r['wall']:6.1f}s "
              f"{r['chunks_s']:8.1f} {r['peak_after_mb']:8.0f}M "
              f"{r['alive']:5d} "
              f"{('%.2f' % r['finite']) if r['finite'] is not None else '  -  ':>7s}"
              f"  {status}", flush=True)

    print("-" * 88)
    ok = [r for r in rows if not r["err"]]
    check("at least one run completed", bool(ok),
          f"{len(rows)-len(ok)}/{len(rows)} runs errored")
    if ok:
        peak = max(r["peak_after_mb"] for r in ok)
        print(f"\npeak worker RSS across all successful runs: {peak:.0f} MB")
        if args.cluster == "local":
            lim = float(args.memory_limit.rstrip("GB")) * 1000
            check("worker peak RSS bounded", peak < 0.9 * lim,
                  f"{peak:.0f} MB vs {args.memory_limit} limit")
        elif args.cluster == "threads":
            lim = _cgroup_limit_bytes() / 1e6
            check("process peak RSS inside the cgroup cap",
                  bool(lim) and peak < 0.9 * lim,
                  f"{peak:.0f} MB vs {lim:.0f} MB cap")
        else:
            check("worker peak RSS under 12 GB", peak < 12_000,
                  f"{peak:.0f} MB")
        check("all workers survived", all(r["alive"] == rows[0]["alive"]
                                          for r in rows),
              f"{[r['alive'] for r in rows]}")
        check("results finite", all((r["finite"] or 0) > 0.99 for r in ok))
    n_503 = sum(1 for r in rows if r["slowdown"])
    if n_503:
        print(f"\n  {n_503}/{len(rows)} run(s) hit 503 SlowDown -- AWS is "
              f"rate-limiting.")
        print(f"  That is a request-RATE problem, not a bandwidth or memory "
              f"one. Reduce")
        print(f"  concurrency (fewer threads, fewer variables at once) rather "
              f"than retrying harder.")

    if cluster is not None:
        client.close(); cluster.close()
    else:
        client.close()
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
