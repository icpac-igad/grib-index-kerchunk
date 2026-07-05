# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "dask[distributed]",
#   "pandas", "gribberish>=1.4", "psutil",
# ]
# ///
"""Dask-cluster read test for a GIK -> Icechunk virtual store.

Context (VirtualiZarr #884 / CMORPH): the historical OOM was on the WRITE side
(coordinator accumulating kerchunk JSON; append_dim re-reading O(n) metadata).
This test evaluates the complementary question: can the store be CONSUMED from
a distributed dask cluster -- i.e. do the icechunk readonly session, the
virtual-chunk S3 fetches, and the gribberish Zarr v3 codec all survive
pickling to worker processes, and does worker memory stay bounded?

A multi-process LocalCluster exercises the same serialization path as Coiled
workers; on Coiled, replace LocalCluster with coiled.Cluster(...).get_client()
(workers need: icechunk, zarr, xarray, gribberish installed).

What it does:
  - open the store with xr.open_zarr(..., chunks={})  -> one dask task per
    GRIB message chunk
  - compute ensemble mean + std of t2m over `number` for the first --steps
    lead times, ALL time indices and members  -> (time*51*steps) S3 chunk reads
  - report wall time, decoded volume, per-worker PEAK RSS (ru_maxrss)

Usage:
  uv run test_dask_read.py --store stores/ecmwf-0p4-ens --steps 3
  source .env && uv run test_dask_read.py \
      --store s3://e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test
Exit code 0 = all PASS, 1 = any FAIL.
"""
import argparse
import os
import time

import numpy as np
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

CONTAINER_PREFIX = "s3://ecmwf-forecasts/"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def resolve_storage(store: str):
    if store.startswith("s3://"):
        bucket, _, prefix = store[5:].partition("/")
        anon = "AWS_ACCESS_KEY_ID" not in os.environ  # no creds -> public read
        return icechunk.s3_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            anonymous=anon, from_env=not anon, force_path_style=True)
    return icechunk.local_filesystem_storage(store)


def worker_peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--steps", type=int, default=3, help="lead times to include")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--memory-limit", default="2GB", help="per-worker limit")
    args = ap.parse_args()

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=2,
                           processes=True, memory_limit=args.memory_limit,
                           dashboard_address=None)
    client = Client(cluster)
    print(f"cluster: {args.workers} worker processes x {args.memory_limit} "
          f"(multi-process => real pickling, as on Coiled)")

    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(resolve_storage(args.store),
                                    authorize_virtual_chunk_access=auth)
    ds = xr.open_zarr(repo.readonly_session("main").store,
                      consolidated=False, zarr_format=3, chunks={})
    sub = ds["t2m"].isel(step=slice(0, args.steps))
    n_chunks = int(np.prod([sub.sizes[d] for d in ("time", "number", "step")]))
    print(f"t2m subset {dict(sub.sizes)} -> {n_chunks} virtual chunks "
          f"({n_chunks * 451 * 900 * 4 / 1e6:.0f} MB decoded)")
    check("dataset opens lazily as dask arrays", sub.chunks is not None,
          f"dask chunksize {sub.data.chunksize}")

    t0 = time.time()
    emean = sub.mean("number")
    estd = sub.std("number")
    emean_v, estd_v = client.compute([emean, estd], sync=True)
    wall = time.time() - t0
    print(f"ensemble mean+std over `number` computed in {wall:.1f}s "
          f"({n_chunks / wall:.0f} chunks/s incl. S3 fetch + GRIB decode)")

    check("results finite", bool(np.isfinite(emean_v.values).all()
                                 and np.isfinite(estd_v.values).all()))
    check("ens-mean t2m plausible (180-340 K)",
          180 < float(emean_v.mean()) < 340, f"mean={float(emean_v.mean()):.2f} K")
    check("ens spread positive", float(estd_v.mean()) > 0,
          f"mean std={float(estd_v.mean()):.3f} K")

    peaks = client.run(worker_peak_rss_mb)
    peak_max = max(peaks.values())
    print("  per-worker peak RSS (MB): " +
          ", ".join(f"{v:.0f}" for v in peaks.values()))
    limit_mb = float(args.memory_limit.rstrip("GB")) * 1000
    check("worker peak RSS bounded", peak_max < 0.9 * limit_mb,
          f"max {peak_max:.0f} MB < {args.memory_limit}/worker")
    alive = len(client.scheduler_info()["workers"])
    check("all workers alive (no OOM kills/restarts)", alive == args.workers,
          f"{alive}/{args.workers}")

    client.close(); cluster.close()
    print(f"\n{'ALL TESTS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
