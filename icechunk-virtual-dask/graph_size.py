"""How big a Dask graph does this Icechunk read actually build?

The question that took a whole session to think of asking. Everything else we
measured — worker RSS, 503 rates, egress addresses, scheduler states — was
downstream of one number nobody had looked at: **how many tasks are in the
graph before it is submitted**.

    t2m    221,085 store chunks ->   665,966 graph tasks   built in 1.4 s
    u    3,097,290 store chunks -> ~9,300,000 graph tasks   never finishes

Graph construction happens in the CLIENT, in Python, holding the GIL. Past a
few million tasks it simply never finishes: the dashboard shows nothing (no
tasks were submitted), there is no traceback, and a watchdog thread cannot even
report because it never gets the GIL. It looks exactly like a cluster hang and
is not one.

This script measures that number, for whatever store and variables you point it
at, WITHOUT computing anything.

Each measurement runs in its own subprocess with a timeout, because the thing
being measured is precisely "does this hang". A hang costs you one row, not the
tool.

Usage
-----
    P=/opt/mamba/envs/dask/bin/python

    # the default store, one surface and one pressure-level variable
    $P graph_size.py --vars t2m u

    # the shape you actually intend to read
    $P graph_size.py --vars u v w --members 51 --steps 53

    # a different store
    $P graph_size.py --store s3://bucket/prefix --group 49r1/00z --vars t

Reading the output
------------------
    store chunks   how many chunks exist in the WHOLE array
    raw tasks      tasks in the graph as built, BEFORE dask culls it
    culled         what the scheduler actually receives, after dask.optimize
    build          wall time to construct + optimize (not to read data)

Thresholds apply to the RAW count, because that is what the client has to
build, and building is where it hangs:

    < 10k     fine
    10k-100k  slow to build, works
    100k-1M   seconds to minutes of pure graph building, fragile
    > 1M      expect a hang

A large gap between raw and culled is the tell: the client is constructing
millions of tasks and then throwing nearly all of them away. That work is pure
waste and it is what freezes the process.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_STORE = "s3://e4drr-project/forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
DEFAULT_GROUP = "50r1/00z"
DEFAULT_ENDPOINT = "https://data.source.coop"
SRC_CONTAINER = "s3://ecmwf-forecasts/"

LAT_MAX, LAT_MIN = 25.25, -15.25
LON_MIN, LON_MAX = 18.5, 55.0

PATTERNS = {
    "chunks={} then .sel()": (
        "THE TRAP. The whole array becomes dask before the selection narrows "
        "it, so the graph covers every chunk in the store."),
    ".sel() then .chunk()": (
        "THE FIX. Open without `chunks`, so .sel() is a cheap metadata "
        "operation on lazily-indexed xarray; convert to dask afterwards, over "
        "the subset only."),
}


# ─────────────────────────────────────────────────────────────────────────────
# child: one measurement, one process, so a hang is contained
# ─────────────────────────────────────────────────────────────────────────────
def measure(store, group, endpoint, var, pattern, date, members, steps,
            cull=False):
    import numpy as np, xarray as xr, icechunk
    import gribberish.zarr  # noqa: F401

    for k in [k for k in list(os.environ) if k.startswith("AWS_")]:
        os.environ.pop(k)

    if store.startswith("s3://"):
        bucket, _, prefix = store[5:].partition("/")
        storage = icechunk.s3_storage(
            bucket=bucket, prefix=prefix.rstrip("/"), endpoint_url=endpoint,
            region="us-east-1", anonymous=True, from_env=False,
            force_path_style=True)
    else:
        storage = icechunk.local_filesystem_storage(store)
    auth = icechunk.containers_credentials(
        {SRC_CONTAINER: icechunk.s3_anonymous_credentials()})
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg,
                                    authorize_virtual_chunk_access=auth)
    sess = repo.readonly_session("main")

    kw = dict(group=group, consolidated=False, zarr_format=3,
              decode_timedelta=True)
    lazy = xr.open_zarr(sess.store, **kw)                 # no chunks -> lazy
    a = lazy[var]

    # how many chunks the whole array has, from its own encoding
    enc = a.encoding.get("chunks") or a.encoding.get("preferred_chunks")
    if isinstance(enc, dict):
        enc = tuple(enc[d] for d in a.dims)
    store_chunks = 1
    if enc:
        for size, cs in zip(a.shape, enc):
            store_chunks *= -(-size // cs)
    out = {"var": var, "dims": list(a.dims), "shape": list(a.shape),
           "chunk": list(enc) if enc else None, "store_chunks": store_chunks}

    sel = dict(time=np.datetime64(date), number=list(range(members)),
               step=[np.timedelta64(3 * i, "h") for i in range(steps)],
               latitude=slice(LAT_MAX, LAT_MIN),
               longitude=slice(LON_MIN, LON_MAX))

    t0 = time.time()
    if pattern == "chunks={} then .sel()":
        ds = xr.open_zarr(sess.store, chunks={}, **kw)
        da = ds[var].sel(**sel)
        if "isobaricInhPa" in da.dims:
            da = da.isel(isobaricInhPa=0)
        g = da.mean("number")
    else:
        da = a
        if "isobaricInhPa" in da.dims:
            da = da.isel(isobaricInhPa=0)          # level FIRST, still lazy
        da = da.sel(**sel)                          # metadata only
        g = da.chunk({"number": 1, "step": 1}).mean("number")
    out["tasks"] = len(g.__dask_graph__())
    out["build_s"] = round(time.time() - t0, 2)
    # The raw graph overstates what is SCHEDULED -- dask culls unreachable
    # tasks first. But culling a multi-million-task graph is itself slow
    # enough to time out, so it is opt-in (--cull). The raw number is the one
    # that matters anyway: it is what the client has to build, and building is
    # where it hangs.
    out["culled"], out["opt_s"] = None, 0.0
    if cull:
        import dask
        t1 = time.time()
        (opt,) = dask.optimize(g)
        out["culled"] = len(opt.__dask_graph__())
        out["opt_s"] = round(time.time() - t1, 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def verdict(tasks):
    if tasks is None:
        return "HANG"
    if tasks > 1_000_000:
        return "expect a hang"
    if tasks > 100_000:
        return "fragile"
    if tasks > 10_000:
        return "slow build"
    return "fine"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--group", default=DEFAULT_GROUP)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--vars", nargs="+", default=["t2m", "u"])
    ap.add_argument("--date", default="2026-06-01")
    ap.add_argument("--members", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=90,
                    help="per measurement; a hang costs one row")
    ap.add_argument("--cull", action="store_true",
                    help="also report the post-optimize task count. Slow: "
                         "culling a multi-million-task graph can itself "
                         "exceed the timeout")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:                      # subprocess entry point
        a = json.loads(args.child)
        try:
            print("__RESULT__" + json.dumps(measure(**a)))
        except Exception as e:          # noqa: BLE001
            print("__RESULT__" + json.dumps(
                {"err": f"{type(e).__name__}: {str(e)[:120]}"}))
        return

    print(f"store  {args.store}  group {args.group}")
    print(f"read   {args.members} members x {args.steps} steps on {args.date}"
          f"  ({args.members * args.steps} chunks of data wanted)\n")

    print(f"{'variable':9s} {'pattern':24s} {'store chunks':>13s} "
          f"{'raw tasks':>12s} {'culled':>9s} {'build':>7s}  verdict")
    print("-" * 96)

    shown = set()
    for var in args.vars:
        for pattern in PATTERNS:
            payload = json.dumps(dict(
                store=args.store, group=args.group, endpoint=args.endpoint,
                var=var, pattern=pattern, date=args.date,
                members=args.members, steps=args.steps, cull=args.cull))
            cmd = [sys.executable, os.path.abspath(__file__), "--child", payload]
            r = None
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=args.timeout)
                for line in p.stdout.splitlines():
                    if line.startswith("__RESULT__"):
                        r = json.loads(line[len("__RESULT__"):])
            except subprocess.TimeoutExpired:
                r = None
            if r and "err" in r:
                print(f"{var:9s} {pattern:24s} {'':>13s} {'':>13s} {'':>8s}  "
                      f"{r['err']}")
                continue
            if r is None:
                print(f"{var:9s} {pattern:24s} {'':>13s} {'(no result)':>12s} "
                      f"{'':>9s} {'>%ds' % args.timeout:>7s}  **HANG**")
                continue
            if var not in shown:
                shown.add(var)
                print(f"{'':9s} {'  dims ' + '×'.join(map(str, r['shape'])):24s} "
                      f"chunk {r['chunk']}")
            culled = f"{r['culled']:,}" if r.get("culled") is not None else "-"
            print(f"{var:9s} {pattern:24s} {r['store_chunks']:13,d} "
                  f"{r['tasks']:12,d} {culled:>9s} "
                  f"{r['build_s'] + r['opt_s']:6.2f}s  "
                  f"{verdict(r['tasks'])}")
    print("-" * 96)
    for name, why in PATTERNS.items():
        print(f"\n{name}\n  {why}")
    print("\nGraph construction is client-side Python holding the GIL. Past a "
          "few million\ntasks it never finishes, the dashboard stays empty "
          "because nothing was\nsubmitted, and there is no traceback. See "
          "DAG_METHOD.md.")


if __name__ == "__main__":
    main()
