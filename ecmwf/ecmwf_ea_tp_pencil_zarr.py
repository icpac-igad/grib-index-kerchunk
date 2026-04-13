#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "dask[distributed]",
#     "fsspec",
#     "s3fs",
#     "icechunk",
#     "zarr<3",
# ]
# ///
"""
ECMWF EA TP — Rechunk Icechunk store to Pencil-Tile Zarr
=========================================================

Reads the materialized Icechunk store at:
  s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ea_tp_icechunk

Rechunks tp from the current "slab" layout to a "pencil-tile" layout and writes
plain Zarr v2 to a local directory or to source.coop S3.

Chunk layout comparison
-----------------------
  Source slab   : (1,  1, 53, 157, 145)  ~4.6 MB  1 date × 1 member × all leads × full EA
                  → optimal: one spatial map per date/member (51 reads per date for ensemble)

  Pencil-tile   : (1, 51, 53,  16,  16)  ~2.7 MB  1 date × all members × all leads × 16×16 tile
                  → optimal: full ensemble distribution at a pixel for one date (1 read)
                    ceil(157/16)=10 lat tiles × ceil(145/16)=10 lon tiles = 100 chunks/date

With the pencil-tile layout:
  Reading one pixel's ensemble + forecast:  1 HTTP GET (covers 16×16 neighbourhood)
  Reading a full spatial map (all members): 100 HTTP GETs (one per tile, parallel)
  Compared to slab:                         51 GETs for ensemble, 1 GET for map

Subcommands
-----------
  size-check  Evaluate actual Icechunk dimensions and compute chunk statistics
              (anonymous, no credentials needed)
  local       Rechunk and write to a local Zarr directory
  remote      Rechunk and write to source.coop S3 (requires .env credentials)
  verify      Inspect the written pencil store and time a single-pixel ensemble read

Usage:
  # Evaluate dimensions from the live Icechunk store:
  uv run ecmwf_ea_tp_pencil_zarr.py size-check

  # Test rechunking locally on a date subset:
  uv run ecmwf_ea_tp_pencil_zarr.py local --output-dir ./ecmwf_ea_tp_pencil \\
      --date-slice 0:10

  # Full rechunk locally (slow without Coiled):
  uv run ecmwf_ea_tp_pencil_zarr.py local --output-dir ./ecmwf_ea_tp_pencil

  # Full rechunk with Coiled:
  uv run ecmwf_ea_tp_pencil_zarr.py local --output-dir ./ecmwf_ea_tp_pencil \\
      --use-coiled --n-workers 50

  # Write to source.coop S3 (requires .env with STS credentials):
  uv run ecmwf_ea_tp_pencil_zarr.py remote

  # Inspect the written store:
  uv run ecmwf_ea_tp_pencil_zarr.py verify --output-dir ./ecmwf_ea_tp_pencil

Author: ICPAC GIK Team
"""

import argparse
import logging
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ecmwf_ea_tp_pencil_zarr.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

# Icechunk source (anonymous read)
IC_BUCKET = "us-west-2.opendata.source.coop"
IC_PREFIX = "e4drr-project/forecasts/ecmwf_ea_tp_icechunk"
IC_REGION = "us-west-2"

# Pencil-tile target chunk: 1 date × all members × all leads × 16×16 spatial tile
# 1 × 51 × 53 × 16 × 16 × 4 B = 691,968 × 4 = 2,767,872 B ≈ 2.64 MB
PENCIL_LAT_TILE = 16
PENCIL_LON_TILE = 16

# Plain Zarr output on source.coop (requires write credentials)
OUT_BUCKET = "us-west-2.opendata.source.coop"
OUT_PREFIX = "e4drr-project/forecasts/ecmwf_ea_tp_pencil"
OUT_REGION = "us-west-2"


# ─── Credential helpers ───────────────────────────────────────────────────────


def load_s3_credentials():
    """Load source.coop write credentials from ecmwf/.env (shell export format)."""
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise RuntimeError(
            "AWS credentials not found. Create ecmwf/.env with:\n"
            '  export AWS_ACCESS_KEY_ID="ASIA..."\n'
            '  export AWS_SECRET_ACCESS_KEY="..."\n'
            '  export AWS_SESSION_TOKEN="..."\n'
            '  export AWS_DEFAULT_REGION="us-west-2"'
        )
    return {"key": access_key, "secret": secret_key, "token": session_token}


# ─── Open source Icechunk store ───────────────────────────────────────────────


def open_icechunk_store(date_slice=None):
    """Open the source.coop Icechunk store anonymously.

    Args:
        date_slice: optional Python slice for init_date (e.g. slice(0, 10)).
                    Applied after opening to limit the dataset for local tests.

    Returns:
        xarray.Dataset (lazy, dask-backed)
    """
    import icechunk
    import xarray as xr

    logger.info(f"Opening Icechunk store: s3://{IC_BUCKET}/{IC_PREFIX}/")
    storage = icechunk.s3_storage(
        bucket=IC_BUCKET,
        prefix=IC_PREFIX,
        region=IC_REGION,
        anonymous=True,
    )
    repo = icechunk.Repository.open(
        storage,
        config=icechunk.RepositoryConfig.default(),
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    if date_slice is not None:
        ds = ds.isel(init_date=date_slice)
        logger.info(f"  Sliced to {ds.sizes['init_date']} dates for testing")

    return ds


# ─── Chunk size evaluation ────────────────────────────────────────────────────


def _chunk_stats(dim_size, tile_size):
    """Return (n_tiles, edge_size) for one dimension."""
    n_full = dim_size // tile_size
    edge = dim_size % tile_size
    n_tiles = n_full + (1 if edge else 0)
    edge_size = edge if edge else tile_size
    return n_tiles, edge_size


def evaluate_chunk_sizes(ds, lat_tile=PENCIL_LAT_TILE, lon_tile=PENCIL_LON_TILE):
    """Print a full size evaluation for the proposed pencil-tile chunk shape.

    Computes:
      - Exact chunk shape (capped at dimension boundaries)
      - Per-chunk uncompressed size in MB
      - Number of tiles in lat and lon
      - Total chunk count
      - Total uncompressed data volume
      - Comparison against the source slab layout
    """
    n_d   = ds.sizes["init_date"]
    n_m   = ds.sizes["member"]
    n_s   = ds.sizes["lead_time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]

    # Proposed pencil-tile shape
    chunk_shape = (1, n_m, n_s, lat_tile, lon_tile)
    elements    = int(np.prod(chunk_shape))
    chunk_mb    = elements * 4 / (1024 ** 2)

    # Tile counts
    lat_tiles, lat_edge = _chunk_stats(n_lat, lat_tile)
    lon_tiles, lon_edge = _chunk_stats(n_lon, lon_tile)
    chunks_per_date = lat_tiles * lon_tiles
    total_chunks    = n_d * chunks_per_date

    # Edge chunks (partial tiles at boundaries)
    edge_chunk_lat  = (1, n_m, n_s, lat_edge, lon_tile)
    edge_chunk_lon  = (1, n_m, n_s, lat_tile, lon_edge)
    edge_chunk_both = (1, n_m, n_s, lat_edge, lon_edge)
    edge_mb_lat     = int(np.prod(edge_chunk_lat))  * 4 / (1024 ** 2)
    edge_mb_lon     = int(np.prod(edge_chunk_lon))  * 4 / (1024 ** 2)
    edge_mb_both    = int(np.prod(edge_chunk_both)) * 4 / (1024 ** 2)

    # Total data
    total_elements = n_d * n_m * n_s * n_lat * n_lon
    total_gib      = total_elements * 4 / (1024 ** 3)

    # Source slab comparison
    slab_shape   = (1, 1, n_s, n_lat, n_lon)
    slab_mb      = int(np.prod(slab_shape)) * 4 / (1024 ** 2)
    slab_reads_for_ensemble = n_m   # one slab per member to get ensemble at one date
    pencil_reads_for_ensemble = 1   # one tile covers all members + all leads

    logger.info("")
    logger.info("=" * 65)
    logger.info("CHUNK SIZE EVALUATION — Pencil-Tile vs Slab")
    logger.info("=" * 65)
    logger.info("")
    logger.info("Icechunk store dimensions (from live store):")
    logger.info(f"  init_date : {n_d}")
    logger.info(f"  member    : {n_m}")
    logger.info(f"  lead_time : {n_s}")
    logger.info(f"  lat       : {n_lat}")
    logger.info(f"  lon       : {n_lon}")
    logger.info(f"  Total uncompressed: {total_gib:.1f} GiB")
    logger.info("")
    logger.info(f"Proposed pencil-tile chunk: {chunk_shape}")
    logger.info(f"  = 1 date × {n_m} members × {n_s} leads × {lat_tile} lat × {lon_tile} lon")
    logger.info(f"  Elements per chunk    : {elements:,}")
    logger.info(f"  Size (uncompressed)   : {chunk_mb:.2f} MB  ✓" if abs(chunk_mb - 2.64) < 0.5 else
                f"  Size (uncompressed)   : {chunk_mb:.2f} MB")
    logger.info("")
    logger.info("Spatial tiling:")
    logger.info(f"  lat tiles : ceil({n_lat}/{lat_tile}) = {lat_tiles}  "
                f"(last tile edge: {lat_edge} rows,  {edge_mb_lat:.2f} MB)")
    logger.info(f"  lon tiles : ceil({n_lon}/{lon_tile}) = {lon_tiles}  "
                f"(last tile edge: {lon_edge} cols,  {edge_mb_lon:.2f} MB)")
    logger.info(f"  corner tile            : {edge_mb_both:.2f} MB")
    logger.info(f"  Chunks per date        : {lat_tiles} × {lon_tiles} = {chunks_per_date}")
    logger.info(f"  Total chunks           : {n_d} dates × {chunks_per_date} = {total_chunks:,}")
    logger.info("")
    logger.info("Access pattern comparison:")
    logger.info(f"  Ensemble at one pixel, one date:")
    logger.info(f"    Slab   : {slab_reads_for_ensemble} GETs  ({slab_mb:.1f} MB each, {slab_reads_for_ensemble * slab_mb:.0f} MB total)")
    logger.info(f"    Pencil : {pencil_reads_for_ensemble} GET   ({chunk_mb:.2f} MB, covers 16×16 neighbourhood)")
    logger.info(f"  Full spatial map (all pixels), one date, one member:")
    logger.info(f"    Slab   : 1 GET   ({slab_mb:.1f} MB)")
    logger.info(f"    Pencil : {chunks_per_date} GETs  ({chunk_mb:.2f} MB each, {chunks_per_date * chunk_mb:.0f} MB total, parallelisable)")
    logger.info("")
    logger.info("Zarr dask chunk kwargs to use:")
    logger.info(f"  dict(init_date=1, member={n_m}, lead_time={n_s}, lat={lat_tile}, lon={lon_tile})")
    logger.info("=" * 65)

    return {
        "chunk_shape": chunk_shape,
        "chunk_mb": chunk_mb,
        "lat_tiles": lat_tiles,
        "lon_tiles": lon_tiles,
        "chunks_per_date": chunks_per_date,
        "total_chunks": total_chunks,
        "total_gib": total_gib,
        "dask_chunks": {
            "init_date": 1,
            "member": n_m,
            "lead_time": n_s,
            "lat": lat_tile,
            "lon": lon_tile,
        },
    }


# ─── Rechunk and write ────────────────────────────────────────────────────────


def _write_zarr(ds_pencil, store_target, zarr_chunk_shape):
    """Write rechunked dataset to a zarr store target (path string or s3fs.S3Map)."""
    import zarr

    ds_pencil.to_zarr(
        store_target,
        mode="w",
        encoding={
            "tp": {
                "chunks": zarr_chunk_shape,
                "compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
                "dtype": "float32",
                "fill_value": float("nan"),
            },
        },
        consolidated=True,
    )


def rechunk_to_local(args):
    """Rechunk the Icechunk store to pencil-tile layout, writing plain Zarr locally."""
    out_dir = Path(args.output_dir)
    logger.info("=" * 65)
    logger.info(f"RECHUNK → local: {out_dir}")
    logger.info("=" * 65)
    start = time.time()

    date_slice = _parse_date_slice(args.date_slice)
    ds = open_icechunk_store(date_slice=date_slice)
    stats = evaluate_chunk_sizes(ds)
    dask_chunks   = stats["dask_chunks"]
    zarr_chunks   = stats["chunk_shape"]

    ds_pencil = ds.chunk(dask_chunks)
    logger.info(f"  Rechunked (lazy): {dict(ds_pencil['tp'].chunks)}")

    if args.use_coiled:
        _start_coiled(args.n_workers)

    logger.info(f"  Writing to: {out_dir}/")
    _write_zarr(ds_pencil, str(out_dir), zarr_chunks)

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s")
    _quick_verify_local(out_dir)


def rechunk_to_remote(args):
    """Rechunk the Icechunk store to pencil-tile layout, writing plain Zarr to source.coop."""
    import s3fs

    logger.info("=" * 65)
    logger.info(f"RECHUNK → remote: s3://{OUT_BUCKET}/{OUT_PREFIX}/")
    logger.info("=" * 65)
    start = time.time()

    creds = load_s3_credentials()

    date_slice = _parse_date_slice(getattr(args, "date_slice", None))
    ds = open_icechunk_store(date_slice=date_slice)
    stats = evaluate_chunk_sizes(ds)
    dask_chunks = stats["dask_chunks"]
    zarr_chunks = stats["chunk_shape"]

    ds_pencil = ds.chunk(dask_chunks)

    # Write credentials for the output path; source is read anonymously via icechunk
    fs_write = s3fs.S3FileSystem(
        key=creds["key"],
        secret=creds["secret"],
        token=creds["token"],
        client_kwargs={"region_name": OUT_REGION},
    )
    os.environ.pop("AWS_NO_SIGN_REQUEST", None)

    out_path = f"s3://{OUT_BUCKET}/{OUT_PREFIX}"
    store = s3fs.S3Map(root=out_path, s3=fs_write, check=False)

    if args.use_coiled:
        _start_coiled(args.n_workers)

    logger.info(f"  Writing to: {out_path}/")
    _write_zarr(ds_pencil, store, zarr_chunks)

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s")


def verify_store(args):
    """Inspect the pencil Zarr store and time a single-pixel ensemble read."""
    import xarray as xr
    import zarr

    out_dir = Path(args.output_dir)
    logger.info("=" * 65)
    logger.info(f"VERIFY: {out_dir}")
    logger.info("=" * 65)

    if not out_dir.exists():
        logger.error(f"Store does not exist: {out_dir}")
        return

    ds = xr.open_zarr(str(out_dir), consolidated=True)
    logger.info(f"\n{ds}")

    logger.info("\nDimensions:")
    for dim, size in ds.sizes.items():
        if dim in ds.coords:
            vals = ds[dim].values
            logger.info(f"  {dim}: {size}  [{vals[0]} .. {vals[-1]}]")
        else:
            logger.info(f"  {dim}: {size}")

    tp_enc = ds["tp"].encoding
    chunk_shape = tp_enc.get("chunks", ())
    logger.info(f"\ntp encoding:")
    logger.info(f"  chunks      : {chunk_shape}")
    logger.info(f"  dtype       : {tp_enc.get('dtype')}")
    logger.info(f"  compressor  : {tp_enc.get('compressor')}")
    if chunk_shape:
        chunk_mb = np.prod(chunk_shape) * 4 / (1024 ** 2)
        logger.info(f"  chunk size  : {chunk_mb:.2f} MB (uncompressed)")

    zstore = zarr.open(str(out_dir))
    tp_arr = zstore["tp"]
    logger.info(f"\nOn-disk:")
    logger.info(f"  compressed  : {tp_arr.nbytes_stored / 1024**3:.3f} GiB")
    logger.info(f"  uncompressed: {tp_arr.nbytes / 1024**3:.3f} GiB")
    ratio = tp_arr.nbytes / tp_arr.nbytes_stored if tp_arr.nbytes_stored else 0
    logger.info(f"  ratio       : {ratio:.1f}×")

    # Time a single-pixel ensemble + forecast read (should be 1 tile GET)
    mid_lat = ds.sizes["lat"] // 2
    mid_lon = ds.sizes["lon"] // 2
    logger.info(f"\nSingle-pixel ensemble read (lat={mid_lat}, lon={mid_lon}, date_idx=0):")
    logger.info("  — with pencil-tile chunks this should be exactly 1 chunk read")
    t0 = time.time()
    sample = ds["tp"].isel(init_date=0, lat=mid_lat, lon=mid_lon).values
    elapsed_read = time.time() - t0
    logger.info(f"  Shape  : {sample.shape}  (member × lead_time)")
    logger.info(f"  Non-NaN: {np.sum(~np.isnan(sample))}")
    logger.info(f"  Time   : {elapsed_read:.3f}s")

    # Confirm tile boundaries align
    if chunk_shape:
        lat_tile = chunk_shape[3]
        lon_tile = chunk_shape[4]
        lat_tile_idx = mid_lat // lat_tile
        lon_tile_idx = mid_lon // lon_tile
        logger.info(f"\nTile containing that pixel:")
        logger.info(f"  lat tile {lat_tile_idx}: rows [{lat_tile_idx * lat_tile} .. "
                    f"{min((lat_tile_idx + 1) * lat_tile, ds.sizes['lat']) - 1}]")
        logger.info(f"  lon tile {lon_tile_idx}: cols [{lon_tile_idx * lon_tile} .. "
                    f"{min((lon_tile_idx + 1) * lon_tile, ds.sizes['lon']) - 1}]")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _parse_date_slice(s):
    """Parse '0:10' → slice(0, 10), None → None."""
    if s is None or s == "":
        return None
    parts = s.split(":")
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop  = int(parts[1]) if parts[1] else None
        return slice(start, stop)
    return None


def _start_coiled(n_workers):
    import coiled
    import distributed

    logger.info(f"  Starting Coiled cluster ({n_workers} workers)...")
    cluster = coiled.Cluster(
        name=f"ecmwf-pencil-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="e2-standard-4",
        package_sync=True,
        region="europe-west1",
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(10, n_workers), timeout=600)
    logger.info(f"  Cluster ready: {client.dashboard_link}")


def _quick_verify_local(out_dir):
    import xarray as xr

    ds = xr.open_zarr(str(out_dir), consolidated=True)
    logger.info(f"\nWritten store: {dict(ds.sizes)}")
    logger.info(f"  tp chunks: {ds['tp'].encoding.get('chunks')}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk ECMWF EA tp Icechunk → pencil-tile Zarr (1, 51, 53, 16, 16)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # size-check
    sub.add_parser(
        "size-check",
        help="Evaluate chunk sizes from the live Icechunk store (anonymous, no credentials)",
    )

    # local
    p_local = sub.add_parser("local", help="Write pencil Zarr to a local directory")
    p_local.add_argument("--output-dir", default="./ecmwf_ea_tp_pencil",
                         help="Output directory (default: ./ecmwf_ea_tp_pencil)")
    p_local.add_argument("--date-slice", default=None, metavar="START:STOP",
                         help="Limit dates for testing, e.g. '0:10' for first 10 dates")
    p_local.add_argument("--use-coiled", action="store_true",
                         help="Use Coiled cluster for distributed rechunking")
    p_local.add_argument("--n-workers", type=int, default=50)

    # remote
    p_remote = sub.add_parser("remote", help="Write pencil Zarr to source.coop S3")
    p_remote.add_argument("--date-slice", default=None, metavar="START:STOP")
    p_remote.add_argument("--use-coiled", action="store_true")
    p_remote.add_argument("--n-workers", type=int, default=50)

    # verify
    p_verify = sub.add_parser("verify", help="Inspect the written pencil Zarr store")
    p_verify.add_argument("--output-dir", default="./ecmwf_ea_tp_pencil")

    args = parser.parse_args()

    if args.cmd == "size-check":
        ds = open_icechunk_store()
        evaluate_chunk_sizes(ds)
    elif args.cmd == "local":
        rechunk_to_local(args)
    elif args.cmd == "remote":
        rechunk_to_remote(args)
    elif args.cmd == "verify":
        verify_store(args)


if __name__ == "__main__":
    main()
