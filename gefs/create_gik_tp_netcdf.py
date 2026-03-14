#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kerchunk",
#     "xarray",
#     "numpy",
#     "netcdf4",
#     "pandas",
#     "fsspec",
#     "s3fs",
#     "cfgrib",
#     "eccodes",
#     "requests",
#     "dask",
#     "h5py",
#     "pyarrow",
#     "gcsfs",
#     "distributed",
#     "gribberish",
# ]
# ///
"""
Create GIK-based GEFS TP NetCDF for comparison with Herbie output.

Uses gribberish to decode GRIB chunks streamed via byte-range reads,
following the same pattern as ecmwf/validate_gik_vs_herbie_2024.py.

Runs the 3-stage GIK pipeline for ensemble members, then streams TP data
directly from S3 using the [url, offset, length] references in the zarr
store. No zarr/xarray opening needed — pure byte-range streaming.

Usage:
    uv run create_gik_tp_netcdf.py
    uv run create_gik_tp_netcdf.py --date 20250106 --max-members 3
"""

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd
import fsspec

warnings.filterwarnings('ignore')
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Add parent gefs folder to path
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── Configuration ─────────────────────────────────────────────────────
TEMPLATE_PARQUET = SCRIPT_DIR / 'gefs-deflated-store-template-20241112.parquet'
TEMPLATE_TAR_GZ = SCRIPT_DIR / 'gik-fmrc-gefs-20241112.tar.gz'
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz"
OUTPUT_DIR = SCRIPT_DIR / 'gik_tp_gefs_output'

# GEFS 0.25° grid (same as ECMWF)
GEFS_GRID_SHAPE = (721, 1440)
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# East Africa / ICPAC region
LAT_MIN, LAT_MAX = -12, 15
LON_MIN, LON_MAX = 25, 52

# Precompute ICPAC indices (GEFS uses 0-360 longitude)
_lat_mask = (GEFS_LATS >= LAT_MIN) & (GEFS_LATS <= LAT_MAX)
_lon_mask = (GEFS_LONS >= LON_MIN) & (GEFS_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
ICPAC_LATS = GEFS_LATS[LAT_INDICES[0]:LAT_INDICES[-1] + 1]
ICPAC_LONS = GEFS_LONS[LON_INDICES[0]:LON_INDICES[-1] + 1]

# Target forecast steps (match Herbie script — skip T+0h, no APCP at init)
TARGET_STEPS_HOURS = [3, 6, 12, 24, 48, 72, 120, 168, 240]

# Try gribberish
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False


# ── GRIB byte-range streaming ────────────────────────────────────────

def fetch_grib_bytes(ref: list, fs) -> bytes:
    """Fetch GRIB bytes from S3 using [url, offset, length] reference."""
    url, offset, length = ref[0], ref[1], ref[2]
    with fs.open(url, "rb") as f:
        f.seek(offset)
        return f.read(length)


def decode_grib_bytes(grib_bytes: bytes) -> np.ndarray:
    """Decode GRIB bytes to numpy array using gribberish or cfgrib fallback."""
    if GRIBBERISH_AVAILABLE:
        try:
            flat = gribberish.parse_grib_array(grib_bytes, 0)
            return flat.reshape(GEFS_GRID_SHAPE)
        except Exception:
            pass
    # cfgrib fallback
    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name
    try:
        ds = xr.open_dataset(tmp_path, engine="cfgrib")
        arr = ds[list(ds.data_vars)[0]].values.copy()
        ds.close()
    finally:
        os.unlink(tmp_path)
    return arr


def subset_icpac(data: np.ndarray) -> np.ndarray:
    """Subset a (721, 1440) array to ICPAC region."""
    return data[LAT_INDICES[0]:LAT_INDICES[-1] + 1,
                LON_INDICES[0]:LON_INDICES[-1] + 1]


# ── Template download ────────────────────────────────────────────────

def download_template(url: str, local_path: Path):
    """Download template tar.gz from HuggingFace if not present."""
    if local_path.exists():
        print(f"  Template exists: {local_path} ({local_path.stat().st_size/1024/1024:.1f} MB)")
        return
    import requests
    print(f"  Downloading template from HuggingFace...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded: {local_path} ({local_path.stat().st_size/1024/1024:.1f} MB)")


# ── GIK pipeline ─────────────────────────────────────────────────────

def run_gik_pipeline(date_str: str, member: str, deflated_store: dict,
                     mapping_manager) -> dict:
    """Run 3-stage GIK pipeline for one member. Returns zarr store dict."""
    from gefs_util import (
        generate_axes, calculate_time_dimensions,
        cs_create_mapped_index_local, prepare_zarr_store,
        process_unique_groups
    )

    print(f"\n  [{member}] Running GIK pipeline...")
    t0 = time.time()

    # Stage 2: Create mapped index from .idx files
    axes = generate_axes(date_str)
    print(f"    Stage 2: Creating mapped index ({len(axes[0])} timesteps)...")
    gefs_kind = cs_create_mapped_index_local(
        axes, date_str, member,
        tar_gz_path=str(TEMPLATE_TAR_GZ),
        mapping_manager=mapping_manager
    )
    print(f"    Stage 2 done: {len(gefs_kind)} index entries")

    # Stage 3: Build zarr store
    print(f"    Stage 3: Building zarr store...")
    time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
    zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)
    updated_zstore = process_unique_groups(
        zstore, chunk_index, time_dims, time_coords,
        times, valid_times, steps
    )

    elapsed = time.time() - t0
    tp_chunks = [k for k in updated_zstore
                 if k.startswith('tp/accum/surface/tp/') and isinstance(updated_zstore[k], list)]
    print(f"    Done in {elapsed:.1f}s: {len(updated_zstore)} zarr refs, {len(tp_chunks)} TP data chunks")
    return updated_zstore


def stream_tp_from_zstore(zstore: dict, target_step_hours: list) -> np.ndarray:
    """Stream TP data from zarr store refs using gribberish.

    Finds TP chunk references [url, offset, length], fetches bytes from S3,
    decodes with gribberish, and subsets to ICPAC region.

    Returns: (n_steps, lat, lon) array for target steps.
    """
    fs = fsspec.filesystem("s3", anon=True)

    # Find TP data chunk keys — format: tp/accum/surface/tp/{step_idx}.0.0
    # The time dim is squeezed (only 1 time), so chunks are 3D: (step, lat, lon)
    tp_chunk_refs = {}
    for key, value in zstore.items():
        if key.startswith('tp/accum/surface/tp/') and isinstance(value, list) and len(value) >= 3:
            parts = key.split('/')[-1]  # e.g., "1.0.0"
            indices = parts.split('.')
            if len(indices) >= 1:
                step_idx = int(indices[0])
                tp_chunk_refs[step_idx] = value

    print(f"    Found {len(tp_chunk_refs)} TP chunk refs (step indices: {sorted(tp_chunk_refs.keys())[:10]}...)")

    # Get step values from zarr metadata to map step_idx → hours
    step_zattrs = zstore.get('tp/accum/surface/step/.zattrs', '')
    step_zarray = zstore.get('tp/accum/surface/step/.zarray', '')

    # Read step coordinate values
    import base64
    step_data_key = 'tp/accum/surface/step/0'
    step_hours_all = None
    if step_data_key in zstore:
        val = zstore[step_data_key]
        if isinstance(val, str) and val.startswith('base64:'):
            data = base64.b64decode(val[7:])
            step_hours_all = np.frombuffer(data, dtype='<f8')
            print(f"    Step hours from zarr: {step_hours_all[:10]}...")
        elif isinstance(val, list):
            # byte-range ref — need to fetch
            try:
                raw = fetch_grib_bytes(val, fs)
                step_hours_all = np.frombuffer(raw, dtype='<f8')
            except Exception:
                pass

    if step_hours_all is None:
        # Fallback: assume 3-hourly steps 0-240
        step_hours_all = np.arange(0, 243, 3, dtype=float)
        print(f"    Using fallback step hours: 0-240 at 3h intervals")

    # Map target step hours to step indices
    n_lats, n_lons = len(ICPAC_LATS), len(ICPAC_LONS)
    result = np.full((len(target_step_hours), n_lats, n_lons), np.nan, dtype=np.float32)

    def _fetch_one(out_idx, step_idx, ref):
        try:
            raw = fetch_grib_bytes(ref, fs)
            arr = decode_grib_bytes(raw)
            return out_idx, subset_icpac(arr).astype(np.float32), None
        except Exception as e:
            return out_idx, None, str(e)

    tasks = []
    for out_idx, target_h in enumerate(target_step_hours):
        # Find the step index that corresponds to this hour
        matches = np.where(np.abs(step_hours_all - target_h) < 0.5)[0]
        if len(matches) > 0:
            step_idx = int(matches[0])
            if step_idx in tp_chunk_refs:
                tasks.append((out_idx, step_idx, tp_chunk_refs[step_idx]))
            else:
                print(f"    No chunk ref for step_idx={step_idx} (T+{target_h}h)")
        else:
            print(f"    T+{target_h}h not found in step coordinates")

    print(f"    Streaming {len(tasks)} TP chunks from S3...")
    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        futs = {pool.submit(_fetch_one, oi, si, r): oi for oi, si, r in tasks}
        done = 0
        for fut in as_completed(futs):
            out_idx, arr, err = fut.result()
            if arr is not None:
                result[out_idx] = arr
                done += 1
            else:
                print(f"    Chunk decode failed: {err}")

    print(f"    Streamed {done}/{len(tasks)} chunks successfully")
    return result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create GIK GEFS TP NetCDF (gribberish streaming)")
    parser.add_argument("--date", type=str, default="20250106")
    parser.add_argument("--run", type=str, default="00")
    parser.add_argument("--max-members", type=int, default=3)
    args = parser.parse_args()

    from gefs_util import (
        build_gefs_deflated_store_from_template,
        LocalTarGzMappingManager
    )

    print("=" * 70)
    print("GIK GEFS TP NetCDF Creator (gribberish streaming)")
    print("=" * 70)
    print(f"Date: {args.date}, Run: {args.run}Z, Members: {args.max_members}")
    print(f"Gribberish available: {GRIBBERISH_AVAILABLE}")

    # Download template if needed
    print("\n[1] Checking templates...")
    download_template(TEMPLATE_URL, TEMPLATE_TAR_GZ)

    if not TEMPLATE_PARQUET.exists():
        print(f"  ERROR: Deflated store template not found: {TEMPLATE_PARQUET}")
        sys.exit(1)

    # Stage 1: Load deflated store from template
    print("\n[2] Loading deflated store template (Stage 1)...")
    deflated_store = build_gefs_deflated_store_from_template(str(TEMPLATE_PARQUET))
    print(f"  Loaded {len(deflated_store['refs'])} zarr refs")

    # Initialize mapping manager
    print("\n[3] Initializing mapping manager...")
    mapping_manager = LocalTarGzMappingManager(str(TEMPLATE_TAR_GZ))
    available = mapping_manager.list_ensemble_members()
    print(f"  Available members: {len(available)}")

    # Process ensemble members
    members = [f'gep{i:02d}' for i in range(1, args.max_members + 1)]
    print(f"\n[4] Processing {len(members)} ensemble members...")

    model_date = datetime.strptime(args.date, "%Y%m%d")
    run_hour = int(args.run)
    base_time = model_date + timedelta(hours=run_hour)

    member_tp_data = []

    for member in members:
        try:
            # Build zarr store with byte-range refs
            zstore = run_gik_pipeline(args.date, member, deflated_store, mapping_manager)

            # Stream TP via gribberish
            print(f"    Streaming TP data via gribberish...")
            tp_data = stream_tp_from_zstore(zstore, TARGET_STEPS_HOURS)
            print(f"    TP shape: {tp_data.shape}, range: {np.nanmin(tp_data):.4f} to {np.nanmax(tp_data):.4f}")

            member_tp_data.append(tp_data)

        except Exception as e:
            print(f"    ERROR for {member}: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    mapping_manager.cleanup()

    if not member_tp_data:
        print("\nERROR: No TP data extracted from any member")
        sys.exit(1)

    # Compute ensemble stats
    print(f"\n[5] Computing ensemble statistics ({len(member_tp_data)} members)...")
    stacked = np.stack(member_tp_data, axis=0)  # (n_members, n_steps, lat, lon)
    mean_arr = np.nanmean(stacked, axis=0)
    std_arr = np.nanstd(stacked, axis=0)
    print(f"  Stacked shape: {stacked.shape}")
    print(f"  Mean range: {np.nanmin(mean_arr):.4f} to {np.nanmax(mean_arr):.4f}")

    # Build output dataset
    valid_times = [base_time + timedelta(hours=h) for h in TARGET_STEPS_HOURS]

    ds_out = xr.Dataset({
        "tp_ensemble_mean": xr.DataArray(
            mean_arr[np.newaxis].astype(np.float32),
            dims=["time", "valid_time", "latitude", "longitude"],
            attrs={"long_name": "APCP ensemble mean", "units": "kg m-2"},
        ),
        "tp_ensemble_standard_deviation": xr.DataArray(
            std_arr[np.newaxis].astype(np.float32),
            dims=["time", "valid_time", "latitude", "longitude"],
            attrs={"long_name": "APCP ensemble standard deviation", "units": "kg m-2"},
        ),
    }, coords={
        "time": [base_time],
        "valid_time": valid_times,
        "latitude": ICPAC_LATS,
        "longitude": ICPAC_LONS,
    })
    ds_out.attrs["title"] = "GEFS Ensemble APCP (GIK method)"
    ds_out.attrs["institution"] = "ICPAC"
    ds_out.attrs["source"] = "GIK -> NOAA GEFS S3 (gribberish)"
    ds_out.attrs["model_date"] = model_date.strftime("%Y-%m-%d")
    ds_out.attrs["model_run"] = f"{run_hour:02d}Z"
    ds_out.attrs["n_ensemble_members"] = len(member_tp_data)
    ds_out.attrs["forecast_hours"] = str(TARGET_STEPS_HOURS)
    ds_out.attrs["history"] = f"Created {datetime.now().isoformat()} via create_gik_tp_netcdf.py"

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"GEFS_{args.date}_{run_hour:02d}Z_gik_tp.nc"
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_out.data_vars}
    ds_out.to_netcdf(out_file, encoding=encoding)
    ds_out.close()

    size_kb = out_file.stat().st_size / 1024
    print(f"\n[6] Saved: {out_file} ({size_kb:.1f} KB)")
    print(f"    {len(member_tp_data)} members, {len(TARGET_STEPS_HOURS)} steps")
    print("Done!")


if __name__ == "__main__":
    main()
