# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "dask[distributed]",
#   "gribberish>=1.4", "s3fs", "psutil",
# ]
# ///
"""Materialize (realize) the East Africa subset of a GIK Icechunk VIRTUAL
store into a plain Zarr store on another path -- on a dask cluster.

This is the pending #884 test, and it separates the two memory questions:
  metadata plane -- writing VIRTUAL refs (where #884 OOM'd): already solved
    by the GIK builder (streamed set_virtual_refs, per-date commits;
    measured ~470 MB peak RSS per date, flat).
  data plane -- READING the virtual store and WRITING a realized dataset:
    what this script exercises. Workers stream chunk -> gribberish decode ->
    EA subset -> S3 put; nothing accumulates on the coordinator.

The write target follows ../dev-test/ecmwf_ea_tp_source_coop_zarr.py: plain
Zarr through the data.source.coop proxy (plain PUTs work there; icechunk
commits do not) -- transactions stay off the source.coop write path.

Usage (source .env first for the proxy write credentials):
  source .env && uv run materialize_ea_from_icechunk.py \
      --source s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test \
      --dest   s3://e4drr-project/forecasts/ecmwf_ea_tp_0p4_zarr_test \
      --workers 2
Local smoke test:
  uv run materialize_ea_from_icechunk.py --source stores/ecmwf-0p4-ens \
      --dest /tmp/ea_tp_test.zarr --steps 5
Exit code 0 = all PASS, 1 = any FAIL.
"""
import argparse
import os
import resource
import time

import numpy as np
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

CONTAINER_PREFIX = "s3://ecmwf-forecasts/"
LAT_MAX, LAT_MIN = 25.0, -14.0   # East Africa / ICPAC box
LON_MIN, LON_MAX = 19.0, 55.0
FAILURES = []

# capture the proxy endpoint for DEST writes, then scrub it from the env:
# icechunk's Rust S3 client auto-reads AWS_ENDPOINT_URL, which would wrongly
# route the (direct-AWS, anonymous) SOURCE reads through the proxy
DEST_ENDPOINT = os.environ.pop("AWS_ENDPOINT_URL", None)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def resolve_icechunk_storage(store: str):
    if store.startswith("s3://"):
        bucket, _, prefix = store[5:].partition("/")
        # always anonymous: the source store is public on the opendata bucket,
        # and any AWS_* creds in env are proxy creds for the DEST writes only
        return icechunk.s3_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            region="us-west-2", endpoint_url=None,
            anonymous=True, force_path_style=True)
    return icechunk.local_filesystem_storage(store)


def open_virtual(source: str) -> xr.Dataset:
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    # the public ecmwf-forecasts bucket throttles bursts of range-GETs
    # (SlowDown): retry with backoff and cap concurrent requests
    config = icechunk.RepositoryConfig.default()
    config.storage = icechunk.StorageSettings(
        retries=icechunk.StorageRetriesSettings(
            max_tries=10, initial_backoff_ms=300, max_backoff_ms=15000))
    config.max_concurrent_requests = 8
    repo = icechunk.Repository.open(resolve_icechunk_storage(source), config,
                                    authorize_virtual_chunk_access=auth)
    return xr.open_zarr(repo.readonly_session("main").store,
                        consolidated=False, zarr_format=3, chunks={})


def dest_store(dest: str, anonymous=False):
    """Plain-zarr target: local path, or s3://bucket/prefix via AWS_ENDPOINT_URL
    (write) / direct AWS (anonymous verify read)."""
    if not dest.startswith("s3://"):
        return dest
    import s3fs
    if anonymous:
        # public reads go via the opendata bucket, not the proxy bucket path
        fs = s3fs.S3FileSystem(anon=True)
        return s3fs.S3Map(root=f"us-west-2.opendata.source.coop/{dest[5:]}", s3=fs)
    else:
        fs = s3fs.S3FileSystem(
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            token=os.environ.get("AWS_SESSION_TOKEN"),
            client_kwargs={"endpoint_url": DEST_ENDPOINT})
    return s3fs.S3Map(root=dest[5:], s3=fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="icechunk virtual store (path or s3://)")
    ap.add_argument("--dest", required=True, help="realized zarr target (path or s3://)")
    ap.add_argument("--var", default="tp")
    ap.add_argument("--steps", type=int, default=None, help="limit lead times (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--memory-limit", default="1.5GB")
    args = ap.parse_args()

    os.environ["ZARR_ASYNC__CONCURRENCY"] = "6"  # bound per-task chunk fan-out
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=2,
                           processes=True, memory_limit=args.memory_limit,
                           dashboard_address=None)
    client = Client(cluster)
    print(f"cluster: {args.workers} worker processes x {args.memory_limit}")

    ds = open_virtual(args.source)
    da = ds[args.var].sel(latitude=slice(LAT_MAX, LAT_MIN),
                          longitude=slice(LON_MIN, LON_MAX))
    if args.steps:
        da = da.isel(step=slice(0, args.steps))
    nt, nm, nsteps = da.sizes["time"], da.sizes["number"], da.sizes["step"]
    ny, nx = da.sizes["latitude"], da.sizes["longitude"]
    n_msgs = nt * nm * nsteps
    print(f"EA subset {args.var}: {dict(da.sizes)} = {da.nbytes/1e6:.0f} MB realized, "
          f"from {n_msgs} GRIB messages (~{n_msgs*0.6:.0f} MB fetched)")

    # one output object per (date, member): chunks (1,1,steps,ny,nx), ~few MB each
    out = da.chunk({"time": 1, "number": 1, "step": nsteps,
                    "latitude": ny, "longitude": nx}).to_dataset()
    # drop inherited encodings: the source serializer is the read-only
    # gribberish codec, which must not be applied to the realized write
    for v in list(out.variables):
        out[v].encoding = {}
    out.attrs.update({
        "title": f"ECMWF IFS ensemble {args.var} -- East Africa, realized",
        "source": f"materialized from icechunk virtual store {args.source}",
        "region": f"lat {LAT_MIN}..{LAT_MAX}, lon {LON_MIN}..{LON_MAX}",
    })

    t0 = time.time()
    out.to_zarr(dest_store(args.dest), mode="w", consolidated=True)
    wall = time.time() - t0
    print(f"materialized + written in {wall:.1f}s ({n_msgs/wall:.1f} msgs/s)")

    coord_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    peaks = client.run(lambda: __import__("resource").getrusage(
        __import__("resource").RUSAGE_SELF).ru_maxrss / 1024)
    print(f"  coordinator peak RSS: {coord_peak:.0f} MB")
    print("  worker peak RSS (MB): " + ", ".join(f"{v:.0f}" for v in peaks.values()))
    limit_mb = float(args.memory_limit.rstrip("GB")) * 1000
    check("coordinator RSS bounded (metadata never accumulates)", coord_peak < 2000,
          f"{coord_peak:.0f} MB")
    check("worker peak RSS bounded", max(peaks.values()) < 0.95 * limit_mb,
          f"max {max(peaks.values()):.0f} MB < {args.memory_limit}")
    alive = len(client.scheduler_info()["workers"])
    check("all workers alive (no OOM kills)", alive == args.workers, f"{alive}/{args.workers}")
    client.close(); cluster.close()

    # verify: reopen realized store, compare one random (date, member) slab
    # against the virtual store read
    ro = xr.open_zarr(dest_store(args.dest, anonymous=args.dest.startswith("s3://")),
                      consolidated=True)
    check("realized store reopens", True, f"sizes {dict(ro[args.var].sizes)}")
    rng = np.random.default_rng(7)
    ti, mi = int(rng.integers(nt)), int(rng.integers(nm))
    got = ro[args.var].isel(time=ti, number=mi).values
    want = da.isel(time=ti, number=mi).values
    check(f"bit-exact vs virtual store (time={ti}, member={mi})",
          np.array_equal(got, want, equal_nan=True),
          f"mean={np.nanmean(got):.6f} {ds[args.var].attrs.get('units','')}")
    check("values plausible (tp >= 0)",
          bool(np.nanmin(got) >= 0) if args.var == "tp" else True,
          f"min={np.nanmin(got):.6f} max={np.nanmax(got):.6f}")

    print(f"\n{'ALL TESTS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
