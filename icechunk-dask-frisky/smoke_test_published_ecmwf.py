# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "numpy", "gribberish>=1.4", "s3fs",
# ]
# ///
"""Anonymous smoke test of the published ECMWF IFS Icechunk store on source.coop.

Proves that a reader with ZERO credentials can open the store metadata over the
source.coop S3 endpoint and resolve virtual chunks by anonymous byte-range read
against the public ecmwf-forecasts AWS bucket, decoding through gribberish.

The ECMWF store is multi-era: three subgroups (0p4, 49r1, 50r1), each with a
`00z` group holding the arrays. This test opens all three and decodes a 2 m
temperature field from each.

Run WITHOUT sourcing .env (no AWS creds in env).

Usage:
  uv run smoke_test_published_ecmwf.py
"""
import os
import time

import numpy as np
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

ENDPOINT = "https://data.source.coop"
BUCKET = "e4drr-project"
PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
CONTAINER_PREFIX = "s3://ecmwf-forecasts/"   # virtual chunks (public AWS, anon)
ERAS = ["0p4", "49r1", "50r1"]               # each has a 00z group
T2M_CANDIDATES = ["t2m", "2t"]               # var name differs across eras
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    assert "AWS_ACCESS_KEY_ID" not in os.environ, (
        "run this WITHOUT sourcing .env -- it must prove anonymous access")

    print(f"== opening s3://{BUCKET}/{PREFIX} anonymously via {ENDPOINT} ==")
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
    sess = repo.readonly_session("main")
    print(f"repo opened in {time.time()-t0:.1f}s")

    for era in ERAS:
        group = f"{era}/00z"
        print(f"\n== era {group} ==")
        ds = xr.open_zarr(sess.store, group=group,
                          consolidated=False, zarr_format=3)
        ny, nx = ds.sizes["latitude"], ds.sizes["longitude"]
        check(f"{era} opens", True,
              f"{len(ds.data_vars)} vars, time={ds.sizes['time']}, "
              f"number={ds.sizes['number']}, step={ds.sizes['step']}, "
              f"levels={ds.sizes.get('isobaricInhPa')}, grid={ny}x{nx}")
        # NB: ds.nbytes is the DENSE decoded-float32 size (every cell, uncompressed).
        # It is NOT the data volume -- the packed GRIB referenced across all three
        # eras totals ~620 TB, and the store's own objects are ~15 GB.
        print(f"         dense logical size (float32, if materialized): "
              f"{ds.nbytes/1e12:.1f} TB ({ds.nbytes:,} bytes)")

        var = next((v for v in T2M_CANDIDATES if v in ds), None)
        if var is None:
            check(f"{era} has a 2m-temperature var", False,
                  f"none of {T2M_CANDIDATES} present")
            continue

        ti = int(np.argmax(ds["time"].values))
        newest = np.datetime_as_string(ds["time"].values[ti], unit="D")
        t1 = time.time()
        field = ds[var].isel(time=ti, number=0, step=0).values
        dt = time.time() - t1
        finite = np.isfinite(field).mean()
        mean = float(np.nanmean(field))
        ok = finite > 0.99 and 200 < mean < 330
        check(f"{era} {var} @ {newest} decodes (virtual chunk)", ok,
              f"shape {field.shape}, mean {mean:.1f} K, "
              f"finite {finite:.3f}, {dt:.2f}s")

    print("\n== size reality check (00z store, 1246 dates) ==")
    print("  store objects on source.coop : ~15 GB   (what is actually hosted)")
    print("  referenced GRIB (packed)     : ~620 TB  (97-100% of enfo published)")
    print("  enfo published, same dates   : ~627 TB")
    print("  dense float32 if materialized: ~2.79 PB (sum of ds.nbytes above)")

    print("\nRESULT:", "PASS -- anonymous open + virtual decode works for all eras"
          if not FAILURES else f"FAIL -- {FAILURES}")
    raise SystemExit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
