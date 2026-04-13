#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "icechunk>=0.1.0",
#     "dask>=2024.1.0",
#     "python-dotenv>=1.0.0",
#     "coiled>=1.0.0",
#     "distributed>=2024.1.0",
#     "s3fs>=2024.1.0",
# ]
# ///
"""
Rechunk ECMWF EA TP Icechunk store from slab to pencil-tile layout.

Reads the slab-chunked Icechunk store at source.coop (anonymous) and writes a
pencil-tile-chunked store — either as a new Icechunk store or as plain Zarr
(Coiled path).

  Slab   : (1, 1, 53, 157, 145)  — 1 date × 1 member × all leads × full EA spatial
            good for: spatial maps at one date/member
  Pencil : (1, 51, 53, 16, 16)   — 1 date × all members × all leads × 16×16 tile
            good for: full ensemble distribution at one pixel for one date (1 read)

Source  : s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ea_tp_icechunk
Target  : s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ea_tp_pencil_icechunk
          s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ea_tp_pencil_zarr  (Coiled)

Usage:
    # 1. Dry run — show source/target shapes and chunk sizes (no credentials needed):
    uv run ecmwf/ecmwf_ea_tp_pencil_zarr.py --dry-run

    # 2. Rechunk lat-band by lat-band (writes Icechunk pencil store):
    uv run ecmwf/ecmwf_ea_tp_pencil_zarr.py

    # 3. Rechunk using Coiled Dask P2P (fast, writes plain Zarr to source.coop):
    uv run ecmwf/ecmwf_ea_tp_pencil_zarr.py --coiled --n-workers 15

    # 4. Verify the pencil store:
    uv run ecmwf/ecmwf_ea_tp_pencil_zarr.py --verify
    uv run ecmwf/ecmwf_ea_tp_pencil_zarr.py --verify --coiled

Author: ICPAC GIK Team
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_BASE_PREFIX = "e4drr-project/forecasts"

SOURCE_PREFIX = "ecmwf_ea_tp_icechunk"
PENCIL_PREFIX = "ecmwf_ea_tp_pencil_icechunk"
PENCIL_ZARR_PREFIX = "ecmwf_ea_tp_pencil_zarr"


# ─── Credentials ──────────────────────────────────────────────────────────────


def _get_s3_credentials():
    """Get source.coop S3 credentials, falling back to AWS_* env vars."""
    access_key   = os.getenv("SOURCE_COOP_ACCESS_KEY_ID")   or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key   = os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("SOURCE_COOP_SESSION_TOKEN")   or os.getenv("AWS_SESSION_TOKEN")
    return access_key, secret_key, session_token


# ─── Storage helpers ──────────────────────────────────────────────────────────


def _s3_source_storage():
    """Anonymous S3 storage for reading slab Icechunk store from source.coop."""
    import icechunk

    prefix = f"{SOURCE_COOP_BASE_PREFIX}/{SOURCE_PREFIX}"
    return icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=prefix,
        region="us-west-2",
        anonymous=True,
    )


def _s3_pencil_storage():
    """Authenticated S3 storage for writing pencil Icechunk store to source.coop."""
    import icechunk

    access_key, secret_key, session_token = _get_s3_credentials()
    prefix = f"{SOURCE_COOP_BASE_PREFIX}/{PENCIL_PREFIX}"
    return icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=prefix,
        region="us-west-2",
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=session_token,
    )


# ─── Open source store ────────────────────────────────────────────────────────


def open_source_store():
    """Open slab Icechunk store from source.coop (anonymous read)."""
    import icechunk
    import xarray as xr

    storage = _s3_source_storage()
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds


# ─── Rechunk: lat-band iteration (writes Icechunk) ────────────────────────────


def rechunk_dataset(chunk_lat, chunk_lon, dry_run=False):
    """Rechunk slab → pencil-tile by lat-band iteration, writing to Icechunk.

    Strategy: iterate lat-bands of `chunk_lat` rows. For each band load all
    init_dates × all members × all leads × chunk_lat rows × all lons from the
    source slab store, then write lon-tiles within that band to the pencil store.
    One Icechunk commit per lat-band.
    """
    import dask.array as da
    import icechunk
    import xarray as xr

    src_label = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{SOURCE_PREFIX}"
    tgt_label = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{PENCIL_PREFIX}"

    print(f"\n{'='*60}")
    print(f"Rechunk ECMWF EA TP: slab → pencil-tile (Icechunk)")
    print(f"{'='*60}")
    print(f"\n  Source: {src_label}")

    ds_src = open_source_store()
    print(f"  Shape:  {dict(ds_src.sizes)}")
    src_chunks = ds_src["tp"].encoding.get("chunks", "unknown")
    print(f"  tp slab chunks = {src_chunks}")

    n_d   = ds_src.sizes["init_date"]
    n_m   = ds_src.sizes["member"]
    n_s   = ds_src.sizes["lead_time"]
    n_lat = ds_src.sizes["lat"]
    n_lon = ds_src.sizes["lon"]

    pencil_chunks = {
        "init_date": 1,
        "member": n_m,
        "lead_time": n_s,
        "lat": chunk_lat,
        "lon": chunk_lon,
    }

    n_lat_chunks = -(-n_lat // chunk_lat)
    n_lon_chunks = -(-n_lon // chunk_lon)
    chunk_bytes  = 1 * n_m * n_s * chunk_lat * chunk_lon * 4
    total_chunks = n_d * n_lat_chunks * n_lon_chunks
    total_gb     = n_d * n_m * n_s * n_lat * n_lon * 4 / (1024 ** 3)

    print(f"\n  Target: {tgt_label}")
    print(f"  Pencil chunks: (init_date=1, member={n_m}, lead_time={n_s}, lat={chunk_lat}, lon={chunk_lon})")
    print(f"  Chunk size: {chunk_bytes / (1024**2):.2f} MB")
    print(f"  Spatial tiles: {n_lat_chunks} lat × {n_lon_chunks} lon = {n_lat_chunks * n_lon_chunks} tiles/date")
    print(f"  Total chunks: {total_chunks} ({n_d} dates × {n_lat_chunks} lat × {n_lon_chunks} lon)")
    print(f"  Total data: {total_gb:.2f} GB")
    print(f"  Commits: ~{n_lat_chunks} (one per lat-band)")

    if dry_run:
        print("\n  (dry run — no rechunking)")
        ds_src.close()
        return

    # Create target Icechunk store
    print(f"\n  Creating target store...")
    target_storage = _s3_pencil_storage()
    try:
        target_repo = icechunk.Repository.create(
            target_storage, config=icechunk.RepositoryConfig.default()
        )
        print("  Created new repository")
    except Exception:
        target_repo = icechunk.Repository.open(
            target_storage, config=icechunk.RepositoryConfig.default()
        )
        print("  Opened existing repository")

    # Write pencil-chunked template (metadata only, GLAD pattern)
    target_session = target_repo.writable_session("main")

    chunk_tuple = (1, n_m, n_s, chunk_lat, chunk_lon)
    template = xr.Dataset(
        {
            "tp": (
                ds_src["tp"].dims,
                da.zeros(ds_src["tp"].shape, chunks=chunk_tuple, dtype=ds_src["tp"].dtype),
                ds_src["tp"].attrs,
            )
        },
        coords=ds_src.coords,
        attrs=ds_src.attrs,
    )

    encoding = {
        "tp": {"chunks": chunk_tuple}
    }

    template.to_zarr(
        target_session.store,
        compute=False,
        mode="w",
        encoding=encoding,
        consolidated=False,
    )
    target_session.commit(
        f"initialize ecmwf_ea_tp pencil template (1, {n_m}, {n_s}, {chunk_lat}, {chunk_lon})"
    )
    print("  Template written")
    ds_src.close()

    # Fill pencil store by lat-band
    print(f"\n  Filling pencil store (lat-band iteration)...")
    print(f"  Strategy: {n_lat_chunks} lat-bands, each with {n_lon_chunks} lon-tiles")
    start = time.time()

    ds_src = open_source_store()

    bands_done = 0
    tiles_done = 0
    for lat_start in range(0, n_lat, chunk_lat):
        lat_end = min(lat_start + chunk_lat, n_lat)

        # Load full lat-band (all init_dates, members, leads, all lons) from source
        band = ds_src.isel(lat=slice(lat_start, lat_end)).load()

        # One session for all lon-tiles in this band
        session2 = target_repo.writable_session("main")

        for lon_start in range(0, n_lon, chunk_lon):
            lon_end = min(lon_start + chunk_lon, n_lon)

            tile = band.isel(lon=slice(lon_start, lon_end))

            tile.to_zarr(
                session2.store,
                region={
                    "init_date": slice(0, n_d),
                    "member":    slice(0, n_m),
                    "lead_time": slice(0, n_s),
                    "lat":       slice(lat_start, lat_end),
                    "lon":       slice(lon_start, lon_end),
                },
                consolidated=False,
            )
            tiles_done += 1

        session2.commit(f"fill lat-band {lat_start}-{lat_end} ({n_lon_chunks} lon-tiles)")
        bands_done += 1

        elapsed = time.time() - start
        pct = 100 * bands_done / n_lat_chunks
        print(f"    [{pct:5.1f}%] band {bands_done}/{n_lat_chunks} "
              f"(lat {lat_start}-{lat_end}), {tiles_done} tiles total ({elapsed:.0f}s)")
        del band

    ds_src.close()
    elapsed = time.time() - start
    print(f"\n  Rechunk complete: {tiles_done} tiles in {bands_done} commits, "
          f"{elapsed:.0f}s ({elapsed/60:.1f} min)")


# ─── Rechunk: Coiled Dask P2P (writes plain Zarr) ────────────────────────────


def rechunk_coiled(chunk_lat, chunk_lon, n_workers, dry_run=False):
    """Rechunk slab → pencil-tile via Coiled Dask P2P, writing plain Zarr.

    Reads the slab Icechunk store from source.coop (anonymous), rechunks via
    Dask P2P shuffle on a Coiled cluster, writes plain Zarr back to source.coop.
    """
    import pickle

    import coiled
    import dask
    import distributed
    import icechunk
    import xarray as xr

    print(f"\n{'='*60}")
    print(f"Rechunk (Coiled P2P): ECMWF EA TP slab → pencil-tile")
    print(f"{'='*60}")

    # P2P rechunk configuration
    dask.config.set({
        "array.rechunk.method": "p2p",
        "optimization.fuse.active": False,
    })
    print("  Dask config: P2P rechunk enabled, fusion disabled")

    src_prefix = f"{SOURCE_COOP_BASE_PREFIX}/{SOURCE_PREFIX}"
    print(f"\n  Source: s3://{SOURCE_COOP_BUCKET}/{src_prefix}")

    storage = _s3_source_storage()
    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default()
    )
    session = repo.readonly_session("main")

    # Verify pickle serialization (required for Dask workers)
    try:
        pickle.dumps(session.store)
        print("  IcechunkStore is pickle-serializable")
    except Exception as e:
        print(f"  ERROR: IcechunkStore not serializable: {e}")
        return

    # Open with explicit source chunk sizes matching stored layout
    ds = xr.open_zarr(session.store, consolidated=False)
    source_chunks = {
        d: c for d, c in zip(ds["tp"].dims, ds["tp"].encoding["chunks"])
    }
    ds.close()
    ds = xr.open_zarr(session.store, consolidated=False, chunks=source_chunks)

    print(f"  Shape: {dict(ds.sizes)}")
    print(f"  Source chunks: {source_chunks}")

    n_d   = ds.sizes["init_date"]
    n_m   = ds.sizes["member"]
    n_s   = ds.sizes["lead_time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    size_gb = n_d * n_m * n_s * n_lat * n_lon * 4 / (1024 ** 3)

    pencil_chunks = {
        "init_date": 1,
        "member": n_m,
        "lead_time": n_s,
        "lat": chunk_lat,
        "lon": chunk_lon,
    }
    chunk_bytes  = 1 * n_m * n_s * chunk_lat * chunk_lon * 4
    n_lat_chunks = -(-n_lat // chunk_lat)
    n_lon_chunks = -(-n_lon // chunk_lon)
    n_target_chunks = n_d * n_lat_chunks * n_lon_chunks

    target_path = (
        f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{PENCIL_ZARR_PREFIX}"
    )
    print(f"\n  Target: {target_path}")
    print(f"  Pencil chunks: (init_date=1, member={n_m}, lead_time={n_s}, lat={chunk_lat}, lon={chunk_lon})")
    print(f"  Chunk size: {chunk_bytes / (1024**2):.2f} MB")
    print(f"  Total target chunks: {n_target_chunks} "
          f"({n_d} dates × {n_lat_chunks} lat × {n_lon_chunks} lon)")
    print(f"  Total data: {size_gb:.1f} GB")
    per_worker_gb = size_gb / n_workers
    print(f"  Workers: {n_workers} × n2-highmem-4 (32 GB each)")
    print(f"  Per-worker data: {per_worker_gb:.1f} GB")

    if dry_run:
        print("\n  (dry run — no rechunking)")
        ds.close()
        return

    # Quick credential check before launching cluster
    access_key, secret_key, session_token = _get_s3_credentials()
    if not access_key or not secret_key:
        print("ERROR: S3 credentials required for writing to source.coop")
        return

    # Launch fixed-size Coiled cluster (P2P requires static cluster)
    print(f"\n  Launching Coiled cluster...")
    overall_start = time.time()

    cluster = coiled.Cluster(
        name=f"ecmwf-ea-tp-pencil-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-highmem-4",
        region="us-west1",
        workspace=os.getenv("COILED_WORKSPACE"),
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=n_workers, timeout=600)
    print(f"  Cluster ready: {n_workers} workers")
    print(f"  Dashboard: {client.dashboard_link}")

    # Rechunk with Dask P2P
    print(f"\n  Starting P2P rechunk + write...")
    ds_rechunked = ds.chunk(pencil_chunks)

    # Re-read .env NOW (right before write) to get freshest token,
    # minimising time between token capture and S3 writes.
    load_dotenv(override=True)
    access_key, secret_key, session_token = _get_s3_credentials()
    print(f"  S3 credentials refreshed (key=...{access_key[-4:]})")

    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "token": session_token,
        "client_kwargs": {"region_name": "us-west-2"},
    }

    encoding = {
        "tp": {"chunks": (1, n_m, n_s, chunk_lat, chunk_lon)}
    }

    ds_rechunked.to_zarr(
        target_path,
        storage_options=storage_options,
        encoding=encoding,
        mode="w",
        consolidated=True,
    )

    print("  Write complete!")
    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"RECHUNK COMPLETE: ecmwf_ea_tp")
    print(f"  Target: {target_path}")
    print(f"  Chunks: (1, {n_m}, {n_s}, {chunk_lat}, {chunk_lon})")
    print(f"  Total chunks: {n_target_chunks}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print(f"{'='*60}")


# ─── Verify ───────────────────────────────────────────────────────────────────


def verify_pencil(coiled_mode=False):
    """Verify the pencil store — check chunks and spot-check a pixel ensemble."""
    import xarray as xr

    if coiled_mode:
        # Plain Zarr on source.coop
        target_path = (
            f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{PENCIL_ZARR_PREFIX}"
        )
        print(f"\n  Verifying (plain Zarr): {target_path}")
        ds = xr.open_zarr(target_path, storage_options={"anon": True}, consolidated=True)
    else:
        import icechunk

        prefix = f"{SOURCE_COOP_BASE_PREFIX}/{PENCIL_PREFIX}"
        print(f"\n  Verifying (Icechunk): s3://{SOURCE_COOP_BUCKET}/{prefix}")
        storage = icechunk.s3_storage(
            bucket=SOURCE_COOP_BUCKET,
            prefix=prefix,
            region="us-west-2",
            anonymous=True,
        )
        repo = icechunk.Repository.open(
            storage, config=icechunk.RepositoryConfig.default()
        )
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)

    print(f"  Dimensions: {dict(ds.sizes)}")
    chunks = ds["tp"].encoding.get("chunks", "unknown")
    print(f"  tp pencil chunks = {chunks}")

    if "init_date" in ds.coords:
        print(f"  init_date: {ds.init_date.values[0]} → {ds.init_date.values[-1]} "
              f"({len(ds.init_date)} dates)")

    # Spot check: load full ensemble at one pixel for one date (should be 1 chunk read)
    ts = ds["tp"].isel(init_date=0, lat=0, lon=0).values
    valid = np.count_nonzero(~np.isnan(ts))
    print(f"  tp ensemble at (date=0, lat=0, lon=0): shape={ts.shape}, "
          f"{valid}/{ts.size} valid ({100*valid/ts.size:.1f}%)")

    ds.close()
    print("  Verification passed.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk ECMWF EA TP Icechunk store from slab to pencil-tile layout"
    )
    parser.add_argument("--chunk-lat", type=int, default=16,
                        help="Pencil tile lat size (default: 16)")
    parser.add_argument("--chunk-lon", type=int, default=16,
                        help="Pencil tile lon size (default: 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sizes only, don't rechunk")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pencil store")
    parser.add_argument("--coiled", action="store_true",
                        help="Use Coiled Dask P2P rechunk (fast, writes plain Zarr to source.coop)")
    parser.add_argument("--n-workers", type=int, default=15,
                        help="Number of Coiled workers (default: 15)")
    args = parser.parse_args()

    if args.verify:
        verify_pencil(coiled_mode=args.coiled)
    elif args.coiled:
        rechunk_coiled(args.chunk_lat, args.chunk_lon, args.n_workers, args.dry_run)
    else:
        rechunk_dataset(args.chunk_lat, args.chunk_lon, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
