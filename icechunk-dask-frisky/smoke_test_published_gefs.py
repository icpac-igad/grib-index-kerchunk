# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "numpy", "gribberish>=1.4", "s3fs",
# ]
# ///
"""Anonymous smoke test of the published GEFS Icechunk store on source.coop.

Proves that a reader with ZERO credentials can:
  1. open the store metadata anonymously over the source.coop S3 endpoint, and
  2. resolve a virtual chunk by anonymous byte-range read against the public
     noaa-gefs-pds AWS bucket, decoding it through the gribberish Zarr codec.

Run WITHOUT sourcing .env (no AWS creds in env) -- that is the whole point:
it simulates a public consumer of the published dataset.

Usage:
  uv run smoke_test_published_gefs.py
"""
import os
import time

import numpy as np
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

# --- published location (source.coop) ---
ENDPOINT = "https://data.source.coop"
BUCKET = "e4drr-project"
PREFIX = "forecasts/noaa_gefs_aws_s3_icechunk_vd"
# --- the GEFS arrays live under this group, not the root ---
GROUP = "0p25/00z"
# --- virtual chunks live here (public AWS, anonymous) ---
CONTAINER_PREFIX = "s3://noaa-gefs-pds/"


def main():
    assert "AWS_ACCESS_KEY_ID" not in os.environ, (
        "run this WITHOUT sourcing .env -- it must prove anonymous access")

    print(f"== opening s3://{BUCKET}/{PREFIX} (group {GROUP}) "
          f"anonymously via {ENDPOINT} ==")
    t0 = time.time()
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=PREFIX,
        endpoint_url=ENDPOINT, region="us-east-1",
        anonymous=True, from_env=False, force_path_style=True,
    )
    auth = icechunk.containers_credentials(
        {CONTAINER_PREFIX: icechunk.s3_anonymous_credentials()})
    # Disable eager manifest preload. source.coop returns sporadic HTTP 500s
    # ("service error") on a few % of GETs; icechunk's default open prefetches
    # thousands of manifests in parallel, so at least one critical fetch tends
    # to draw a 500 and the open raises. With preload off, manifests load
    # lazily on demand (with retry) -- slower open, but robust and quiet.
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg,
                                    authorize_virtual_chunk_access=auth)
    ds = xr.open_zarr(repo.readonly_session("main").store,
                      group=GROUP, consolidated=False, zarr_format=3)
    print(f"opened in {time.time()-t0:.1f}s\n")
    print(ds)

    print("\n== structure ==")
    print("  data vars :", list(ds.data_vars))
    print("  dims      :", dict(ds.sizes))
    print("  coords    :", list(ds.coords))
    # NB: ds.nbytes is the DENSE decoded-float32 size, assuming every cell exists
    # and is uncompressed. It is NOT the data volume. The GRIB actually referenced
    # is ~92 TB (packed), and the store's own objects are ~5.4 GB.
    print(f"  dense logical size (float32, if materialized): "
          f"{ds.nbytes/1e12:.1f} TB ({ds.nbytes:,} bytes)")
    print(f"  -- referenced GRIB (packed, on noaa-gefs-pds): ~92 TB")
    print(f"  -- store objects on source.coop:               ~5.4 GB")

    # newest forecast date; decode an instant var (t2m has an f000 message)
    ti = int(np.argmax(ds["time"].values))
    newest = np.datetime_as_string(ds["time"].values[ti], unit="D")
    print(f"\n== decoding t2m @ {newest}, member gep01, f000 ==")
    t1 = time.time()
    t2m = ds["t2m"].isel(time=ti, number=0, step=0).values
    dt = time.time() - t1
    finite = np.isfinite(t2m).mean()
    print(f"  shape {t2m.shape} decoded in {dt:.2f}s")
    print(f"  finite fraction {finite:.3f}")
    print(f"  min/mean/max = {np.nanmin(t2m):.4g} / "
          f"{np.nanmean(t2m):.4g} / {np.nanmax(t2m):.4g} K")

    ok = finite > 0.99 and 200 < np.nanmean(t2m) < 330
    print("\nRESULT:", "PASS -- anonymous open + virtual decode works" if ok
          else "FAIL -- decoded field looks wrong")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
