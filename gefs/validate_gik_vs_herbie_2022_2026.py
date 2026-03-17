#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kerchunk",
#     "numpy",
#     "pandas",
#     "xarray",
#     "fsspec",
#     "s3fs",
#     "pyarrow",
#     "netcdf4",
#     "gribberish",
#     "cfgrib",
#     "eccodes",
#     "herbie-data",
#     "matplotlib",
#     "cartopy",
#     "scipy",
#     "requests",
#     "dask",
#     "h5py",
#     "gcsfs",
#     "distributed",
# ]
# ///
"""
Validate GIK parquet references against Herbie for random dates across 2022–2026.

For each date:
  1) Runs the 3-stage GIK pipeline on-the-fly (no pre-built GCS parquets)
  2) Streams APCP from S3 via gribberish byte-range reads (ensemble mean & std)
  3) Fetches APCP via Herbie (same members → ensemble mean & std)
  4) Compares at T+48h: correlation, RMSE, MAE, relative diff
  5) Saves per-date NetCDFs (cached), comparison plots, scatter plot, JSON stats

Output directory: validation_gik_vs_herbie_gefs_2022_2026/

Usage:
    uv run validate_gik_vs_herbie_2022_2026.py                        # all ~50 dates
    uv run validate_gik_vs_herbie_2022_2026.py --dry-run               # print dates only
    uv run validate_gik_vs_herbie_2022_2026.py --max-members 5         # quick test
    uv run validate_gik_vs_herbie_2022_2026.py --dates 20220315,20240801
    uv run validate_gik_vs_herbie_2022_2026.py --skip-herbie           # use cached Herbie NCs
    uv run validate_gik_vs_herbie_2022_2026.py --skip-gik              # use cached GIK NCs
    uv run validate_gik_vs_herbie_2022_2026.py --year-start 2023 --year-end 2024
"""

import argparse
import base64
import calendar
import gc
import json
import logging
import os
import random
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import fsspec

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# Add gefs folder to path for gefs_util imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── Templates ────────────────────────────────────────────────────────
TEMPLATE_PARQUET = SCRIPT_DIR / "gefs-deflated-store-template-20241112.parquet"
TEMPLATE_TAR_GZ = SCRIPT_DIR / "gik-fmrc-gefs-20241112.tar.gz"
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz"

# ── GEFS 0.25° grid (0-360 longitude) ───────────────────────────────
GEFS_GRID_SHAPE = (721, 1440)
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# ── East Africa / ICPAC region ──────────────────────────────────────
LAT_MIN, LAT_MAX = -12, 15
LON_MIN, LON_MAX = 25, 52

# Precompute ICPAC indices (GEFS uses 0-360 longitude)
_lat_mask = (GEFS_LATS >= LAT_MIN) & (GEFS_LATS <= LAT_MAX)
_lon_mask = (GEFS_LONS >= LON_MIN) & (GEFS_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
ICPAC_LATS = GEFS_LATS[LAT_INDICES[0] : LAT_INDICES[-1] + 1]
ICPAC_LONS = GEFS_LONS[LON_INDICES[0] : LON_INDICES[-1] + 1]

# ── Forecast configuration ──────────────────────────────────────────
# Skip T+0h — no APCP at init
TARGET_STEPS = [3, 6, 12, 24, 48, 72, 120, 168, 240]
STEP_INDEX = 4  # T+48h for comparison metrics

# ── Date range defaults ─────────────────────────────────────────────
YEAR_START = 2022
YEAR_END = 2026  # through Feb 2026

# ── Output ──────────────────────────────────────────────────────────
OUTPUT_DIR = SCRIPT_DIR / "validation_gik_vs_herbie_gefs_2022_2026"

# East Africa extent for map plots
EA_EXTENT = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

# ── Gribberish ──────────────────────────────────────────────────────
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False


# ── Date selection ───────────────────────────────────────────────────

def generate_random_dates(year_start: int = YEAR_START,
                          year_end: int = YEAR_END,
                          seed: int = 42) -> List[date]:
    """Pick one random date per month from Jan year_start through Feb year_end."""
    rng = random.Random(seed)
    dates = []
    for year in range(year_start, year_end + 1):
        end_month = 2 if year == year_end else 12
        for month in range(1, end_month + 1):
            last_day = calendar.monthrange(year, month)[1]
            day = rng.randint(1, last_day)
            dates.append(date(year, month, day))
    return dates


# ── Template download ────────────────────────────────────────────────

def download_template(url: str, local_path: Path, logger=None):
    """Download template tar.gz from HuggingFace if not present."""
    log = logger.info if logger else print
    if local_path.exists():
        log(f"  Template exists: {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    import requests
    log(f"  Downloading template from HuggingFace...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    log(f"  Downloaded: {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ── GRIB byte-range streaming ───────────────────────────────────────

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
    return data[LAT_INDICES[0] : LAT_INDICES[-1] + 1,
                LON_INDICES[0] : LON_INDICES[-1] + 1]


# ── GIK pipeline (3-stage, on-the-fly) ──────────────────────────────

def run_gik_pipeline(date_str: str, member: str, deflated_store: dict,
                     mapping_manager, logger=None) -> Optional[dict]:
    """Run 3-stage GIK pipeline for one member. Returns zarr store dict or None."""
    from gefs_util import (
        generate_axes, calculate_time_dimensions,
        cs_create_mapped_index_local, prepare_zarr_store,
        process_unique_groups,
    )

    log = logger.debug if logger else lambda x: None
    t0 = time.time()

    try:
        axes = generate_axes(date_str)
        gefs_kind = cs_create_mapped_index_local(
            axes, date_str, member,
            tar_gz_path=str(TEMPLATE_TAR_GZ),
            mapping_manager=mapping_manager,
        )
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
        zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)
        updated_zstore = process_unique_groups(
            zstore, chunk_index, time_dims, time_coords,
            times, valid_times, steps,
        )
        elapsed = time.time() - t0
        log(f"    [{member}] Pipeline done in {elapsed:.1f}s: {len(updated_zstore)} zarr refs")
        return updated_zstore
    except Exception as e:
        log(f"    [{member}] Pipeline FAILED: {e}")
        return None


def stream_tp_from_zstore(zstore: dict, target_step_hours: List[int]) -> np.ndarray:
    """Stream TP data from zarr store refs using gribberish.

    Returns: (n_steps, lat, lon) array for target steps.
    """
    fs = fsspec.filesystem("s3", anon=True)

    # Find TP data chunk keys — format: tp/accum/surface/tp/{step_idx}.0.0
    tp_chunk_refs = {}
    for key, value in zstore.items():
        if key.startswith("tp/accum/surface/tp/") and isinstance(value, list) and len(value) >= 3:
            parts = key.split("/")[-1]  # e.g., "1.0.0"
            indices = parts.split(".")
            if len(indices) >= 1:
                step_idx = int(indices[0])
                tp_chunk_refs[step_idx] = value

    # Get step values from zarr metadata to map step_idx → hours
    step_data_key = "tp/accum/surface/step/0"
    step_hours_all = None
    if step_data_key in zstore:
        val = zstore[step_data_key]
        if isinstance(val, str) and val.startswith("base64:"):
            data = base64.b64decode(val[7:])
            step_hours_all = np.frombuffer(data, dtype="<f8")
        elif isinstance(val, list):
            try:
                raw = fetch_grib_bytes(val, fs)
                step_hours_all = np.frombuffer(raw, dtype="<f8")
            except Exception:
                pass

    if step_hours_all is None:
        # Fallback: assume 3-hourly steps 0-240
        step_hours_all = np.arange(0, 243, 3, dtype=float)

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
        matches = np.where(np.abs(step_hours_all - target_h) < 0.5)[0]
        if len(matches) > 0:
            step_idx = int(matches[0])
            if step_idx in tp_chunk_refs:
                tasks.append((out_idx, step_idx, tp_chunk_refs[step_idx]))

    if not tasks:
        return result

    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        futs = {pool.submit(_fetch_one, oi, si, r): oi for oi, si, r in tasks}
        for fut in as_completed(futs):
            out_idx, arr, err = fut.result()
            if arr is not None:
                result[out_idx] = arr

    return result


# ── GIK ensemble streaming ──────────────────────────────────────────

def stream_gik_tp(date_str: str, step_hours: List[int],
                  deflated_store: dict, mapping_manager,
                  max_members: int = 30,
                  logger=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Stream TP ensemble from GIK (3-stage pipeline + gribberish).

    Returns (mean, std, n_members) arrays of shape (n_steps, lat, lon).
    """
    log = logger.info if logger else print

    members = [f"gep{i:02d}" for i in range(1, max_members + 1)]
    n_steps = len(step_hours)
    n_lats, n_lons = len(ICPAC_LATS), len(ICPAC_LONS)

    member_data = []
    for i, member in enumerate(members, 1):
        zstore = run_gik_pipeline(date_str, member, deflated_store, mapping_manager, logger)
        if zstore is None:
            log(f"    GIK: {member} pipeline failed, skipping")
            continue

        tp = stream_tp_from_zstore(zstore, step_hours)
        valid_count = np.count_nonzero(~np.isnan(tp))
        if valid_count == 0:
            log(f"    GIK: {member} no valid TP data, skipping")
            continue

        member_data.append(tp)
        if i % 5 == 0:
            log(f"    GIK: {i}/{len(members)} members done")

    if not member_data:
        return None, None, 0

    stacked = np.stack(member_data, axis=0)  # (n_members, n_steps, lat, lon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)

    gc.collect()
    return mean, std, len(member_data)


# ── Herbie fetching ─────────────────────────────────────────────────

def subset_ea(ds: xr.Dataset) -> xr.Dataset:
    """Subset an xarray Dataset to the East Africa / ICPAC region."""
    lat_name = "latitude"
    lon_name = "longitude"

    # Handle longitude: GEFS uses 0-360
    lons = ds[lon_name].values
    if lons.max() > 180:
        ds = ds.assign_coords(
            {lon_name: ((ds[lon_name] + 180) % 360) - 180}
        )
        ds = ds.sortby(lon_name)

    # GEFS latitude may be ascending or descending
    lats = ds[lat_name].values
    if lats[0] > lats[-1]:
        ds = ds.sel(**{lat_name: slice(LAT_MAX, LAT_MIN),
                       lon_name: slice(LON_MIN, LON_MAX)})
    else:
        ds = ds.sel(**{lat_name: slice(LAT_MIN, LAT_MAX),
                       lon_name: slice(LON_MIN, LON_MAX)})
    return ds


def fetch_herbie_tp(date_str: str, step_hours: List[int],
                    max_members: int = 30,
                    logger=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Fetch APCP ensemble via Herbie. Returns (mean, std, n_members)."""
    from herbie import Herbie

    log = logger.info if logger else print
    log_warn = logger.warning if logger else print

    step_arrays = {}  # step -> (members, lat, lon)
    ea_lats = None
    ea_lons = None

    for fxx in step_hours:
        log(f"    Herbie: step T+{fxx}h ...")
        member_data = []

        for mem_num in range(1, max_members + 1):
            try:
                H = Herbie(
                    f"{date_str} 00:00",
                    model="gefs",
                    product="atmos.25",
                    member=mem_num,
                    fxx=fxx,
                )
                ds = H.xarray(":APCP:surface:", verbose=False)
                if isinstance(ds, list):
                    ds = ds[0]

                # Find the precipitation variable
                tp_var = None
                for v in ds.data_vars:
                    if "apcp" in v.lower() or "tp" in v.lower():
                        tp_var = v
                        break
                    sn = ds[v].attrs.get("GRIB_shortName", "")
                    if sn in ("tp", "apcp"):
                        tp_var = v
                        break
                if tp_var is None:
                    tp_var = list(ds.data_vars)[0]

                ds_sub = subset_ea(ds)
                if ea_lats is None:
                    ea_lats = ds_sub.latitude.values
                    ea_lons = ds_sub.longitude.values

                data = ds_sub[tp_var].values
                while data.ndim > 2:
                    data = data[0]

                member_data.append(data.astype(np.float32))
                ds.close()
            except Exception as e:
                if mem_num <= 3:
                    log_warn(f"      member {mem_num} FAILED: {e}")
                continue

        if member_data:
            arr = np.stack(member_data, axis=0)
            step_arrays[fxx] = arr
            log(f"      OK — {arr.shape[0]} members")
        else:
            log_warn(f"      FAILED — no members for T+{fxx}h")

    if not step_arrays:
        return None, None, 0

    ordered = [s for s in step_hours if s in step_arrays]
    n_members = step_arrays[ordered[0]].shape[0]
    n_lats = len(ea_lats)
    n_lons = len(ea_lons)

    mean = np.full((len(ordered), n_lats, n_lons), np.nan, dtype=np.float32)
    std = np.full((len(ordered), n_lats, n_lons), np.nan, dtype=np.float32)
    for i, fxx in enumerate(ordered):
        mean[i] = np.nanmean(step_arrays[fxx], axis=0)
        std[i] = np.nanstd(step_arrays[fxx], axis=0)

    return mean, std, n_members


# ── Comparison metrics ──────────────────────────────────────────────

def compute_metrics(gik: np.ndarray, herbie: np.ndarray, label: str,
                    step_idx: int = STEP_INDEX) -> dict:
    """Compare two (n_steps, lat, lon) arrays at a given step index."""
    from scipy import stats as sp_stats

    if step_idx >= gik.shape[0] or step_idx >= herbie.shape[0]:
        return {f"{label}_error": f"step_idx {step_idx} out of range"}

    g = gik[step_idx]
    h = herbie[step_idx]

    if g.shape != h.shape:
        return {f"{label}_error": f"shape mismatch {g.shape} vs {h.shape}"}

    diff = g - h
    valid = ~(np.isnan(g) | np.isnan(h))
    gv, hv = g[valid], h[valid]

    if len(gv) == 0:
        return {f"{label}_error": "no valid pixels"}

    if np.std(gv) > 0 and np.std(hv) > 0:
        r, p = sp_stats.pearsonr(gv, hv)
    else:
        r, p = float("nan"), float("nan")

    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    mae = float(np.nanmean(np.abs(diff)))
    max_abs = float(np.nanmax(np.abs(diff)))
    rel_pct = mae / max(np.nanmean(np.abs(gv)), 1e-12) * 100

    return {
        f"{label}_corr": float(r),
        f"{label}_corr_p": float(p),
        f"{label}_rmse": rmse,
        f"{label}_mae": mae,
        f"{label}_max_abs_diff": max_abs,
        f"{label}_rel_diff_pct": float(rel_pct),
        f"{label}_gik_range": [float(np.nanmin(g)), float(np.nanmax(g))],
        f"{label}_herbie_range": [float(np.nanmin(h)), float(np.nanmax(h))],
    }


# ── NetCDF save helper ──────────────────────────────────────────────

def save_tp_netcdf(path: Path, mean: np.ndarray, std: np.ndarray,
                   step_hours: List[int], date_str: str, n_members: int,
                   source: str):
    model_date = datetime.strptime(date_str, "%Y%m%d")
    base_time = model_date
    valid_times = [base_time + timedelta(hours=h) for h in step_hours]

    ds = xr.Dataset(
        {
            "tp_ensemble_mean": xr.DataArray(
                mean[np.newaxis].astype(np.float32),
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "APCP ensemble mean", "units": "kg m-2"},
            ),
            "tp_ensemble_standard_deviation": xr.DataArray(
                std[np.newaxis].astype(np.float32),
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "APCP ensemble standard deviation", "units": "kg m-2"},
            ),
        },
        coords={
            "time": [base_time],
            "valid_time": valid_times,
            "latitude": ICPAC_LATS,
            "longitude": ICPAC_LONS,
        },
    )
    ds.attrs["source"] = source
    ds.attrs["model_date"] = date_str
    ds.attrs["n_ensemble_members"] = n_members
    ds.attrs["forecast_hours"] = str(step_hours)

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    ds.close()


# ── Plotting ─────────────────────────────────────────────────────────

def plot_comparison(date_str: str, gik_mean: np.ndarray, herbie_mean: np.ndarray,
                    gik_std: np.ndarray, herbie_std: np.ndarray,
                    metrics: dict, output_dir: Path, step_idx: int = STEP_INDEX):
    """Create GIK | Herbie | Difference maps for mean & spread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05],
                           hspace=0.25, wspace=0.15)

    rows = [
        (gik_mean[step_idx], herbie_mean[step_idx],
         "Ensemble Mean (kg/m2)", "YlGnBu"),
        (gik_std[step_idx], herbie_std[step_idx],
         "Ensemble Spread (kg/m2)", "Oranges"),
    ]

    for row_idx, (g, h, title, cmap) in enumerate(rows):
        diff = g - h
        vmax = max(np.nanmax(np.abs(g)), np.nanmax(np.abs(h)), 1e-6)
        diff_vmax = max(np.nanmax(np.abs(diff)), 1e-6)

        panels = [
            (g, f"GIK — {title}", cmap, 0, vmax),
            (h, f"Herbie — {title}", cmap, 0, vmax),
            (diff, "Difference (GIK - Herbie)", "RdBu_r", -diff_vmax, diff_vmax),
        ]

        for col_idx, (data, ptitle, cm, vmin, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
            im = ax.pcolormesh(ICPAC_LONS, ICPAC_LATS, data, cmap=cm,
                               vmin=vmin, vmax=vmx,
                               transform=ccrs.PlateCarree(), shading="auto")
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
            ax.add_feature(cfeature.LAKES, alpha=0.3)
            ax.set_extent(EA_EXTENT, crs=ccrs.PlateCarree())
            ax.set_title(ptitle, fontsize=10, fontweight="bold")

            if col_idx == 2:
                cbar_ax = fig.add_subplot(gs[row_idx, 3])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label("kg/m2", fontsize=9)
                cbar.ax.tick_params(labelsize=8)

    mc = metrics.get("mean_corr", float("nan"))
    mr = metrics.get("mean_rmse", float("nan"))
    fh = TARGET_STEPS[step_idx]
    fig.suptitle(
        f"GIK vs Herbie GEFS — {date_str} 00Z — T+{fh}h\n"
        f"Mean r={mc:.8f}  RMSE={mr:.2e}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out = output_dir / f"compare_{date_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def plot_scatter_all(all_gik_means: List[np.ndarray],
                     all_herbie_means: List[np.ndarray],
                     all_dates: List[str],
                     output_dir: Path,
                     step_idx: int = STEP_INDEX):
    """Scatter plot of all grid points across all dates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    all_g = np.concatenate([g[step_idx].flatten() for g in all_gik_means])
    all_h = np.concatenate([h[step_idx].flatten() for h in all_herbie_means])
    valid = ~(np.isnan(all_g) | np.isnan(all_h))
    all_g, all_h = all_g[valid], all_h[valid]

    if len(all_g) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_h, all_g, s=1, alpha=0.3, c="steelblue")
    lim = max(np.max(all_g), np.max(all_h)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="1:1")
    r, _ = sp_stats.pearsonr(all_g, all_h)
    n_dates = len(all_dates)
    fh = TARGET_STEPS[step_idx]
    ax.set_xlabel("Herbie APCP (kg/m2)")
    ax.set_ylabel("GIK APCP (kg/m2)")
    ax.set_title(
        f"GIK vs Herbie GEFS — {n_dates} dates at T+{fh}h\n"
        f"r = {r:.8f}  N = {len(all_g):,}",
        fontweight="bold",
    )
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    out = output_dir / "scatter_all_dates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# ── Logging ──────────────────────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_gik_herbie_gefs")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(out_dir / "validation_log.txt", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate GIK vs Herbie GEFS APCP for random dates across 2022-2026")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected dates and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap ensemble members (default: 30)")
    parser.add_argument("--dates", type=str, default=None,
                        help="Override: comma-separated YYYYMMDD dates")
    parser.add_argument("--skip-herbie", action="store_true",
                        help="Skip Herbie fetch (use existing NC files)")
    parser.add_argument("--skip-gik", action="store_true",
                        help="Skip GIK pipeline (use existing NC files)")
    parser.add_argument("--year-start", type=int, default=YEAR_START,
                        help="Start year (default: 2022)")
    parser.add_argument("--year-end", type=int, default=YEAR_END,
                        help="End year (default: 2026, through Feb)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # Date selection
    if args.dates:
        dates = [datetime.strptime(d.strip(), "%Y%m%d").date()
                 for d in args.dates.split(",")]
    else:
        dates = generate_random_dates(
            year_start=args.year_start,
            year_end=args.year_end,
            seed=args.seed,
        )

    max_members = args.max_members or 30

    if args.dry_run:
        print(f"Random dates (seed={args.seed}, {args.year_start}-{args.year_end}):")
        for d in dates:
            print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%B %Y')})")
        print(f"\nTotal: {len(dates)} dates, {max_members} members each")
        return

    logger = setup_logging(out_dir)
    logger.info("=" * 70)
    logger.info("GIK vs Herbie GEFS Validation — 2022-2026")
    logger.info("=" * 70)
    logger.info(f"Dates: {len(dates)}  |  Steps: {TARGET_STEPS}")
    logger.info(f"Max members: {max_members}")
    logger.info(f"Gribberish available: {GRIBBERISH_AVAILABLE}")
    logger.info(f"Output: {out_dir}")
    logger.info("")

    # ── Initialize GIK templates (once) ──
    deflated_store = None
    mapping_manager = None

    if not args.skip_gik:
        from gefs_util import (
            build_gefs_deflated_store_from_template,
            LocalTarGzMappingManager,
        )

        logger.info("[INIT] Checking templates...")
        download_template(TEMPLATE_URL, TEMPLATE_TAR_GZ, logger)

        if not TEMPLATE_PARQUET.exists():
            logger.error(f"Deflated store template not found: {TEMPLATE_PARQUET}")
            sys.exit(1)

        logger.info("[INIT] Loading deflated store template (Stage 1)...")
        deflated_store = build_gefs_deflated_store_from_template(str(TEMPLATE_PARQUET))
        logger.info(f"  Loaded {len(deflated_store['refs'])} zarr refs")

        logger.info("[INIT] Initializing mapping manager...")
        mapping_manager = LocalTarGzMappingManager(str(TEMPLATE_TAR_GZ))
        logger.info(f"  Available members: {len(mapping_manager.list_ensemble_members())}")
        logger.info("")

    overall_t0 = time.time()
    all_results = []
    all_gik_means = []
    all_herbie_means = []
    all_dates_ok = []

    for idx, d in enumerate(dates, 1):
        ds = d.strftime("%Y%m%d")
        logger.info("=" * 70)
        logger.info(f"[{idx}/{len(dates)}]  {d.strftime('%B %Y')}  —  {ds}")
        logger.info("=" * 70)

        result = {"date": ds}
        gik_nc = out_dir / f"gik_{ds}_tp.nc"
        herbie_nc = out_dir / f"herbie_{ds}_tp.nc"

        # ── GIK streaming ──
        gik_mean = gik_std = None
        gik_n = 0
        if args.skip_gik and gik_nc.exists():
            logger.info("  GIK: loading cached NC")
            with xr.open_dataset(gik_nc) as ds_g:
                gik_mean = ds_g["tp_ensemble_mean"].isel(time=0).values
                gik_std = ds_g["tp_ensemble_standard_deviation"].isel(time=0).values
                gik_n = int(ds_g.attrs.get("n_ensemble_members", 0))
        elif not args.skip_gik:
            t0 = time.time()
            gik_mean, gik_std, gik_n = stream_gik_tp(
                ds, TARGET_STEPS,
                deflated_store=deflated_store,
                mapping_manager=mapping_manager,
                max_members=max_members,
                logger=logger,
            )
            elapsed = time.time() - t0
            logger.info(f"  GIK done: {gik_n} members in {elapsed:.1f}s")

            if gik_mean is not None:
                save_tp_netcdf(gik_nc, gik_mean, gik_std, TARGET_STEPS,
                               ds, gik_n, "GIK 3-stage pipeline + gribberish")
                logger.info(f"  GIK saved: {gik_nc}")
            else:
                logger.warning(f"  GIK: no data for {ds}")
        else:
            logger.info("  GIK: skipped (no cached NC)")

        # ── Herbie fetching ──
        herbie_mean = herbie_std = None
        herbie_n = 0
        if args.skip_herbie and herbie_nc.exists():
            logger.info("  Herbie: loading cached NC")
            with xr.open_dataset(herbie_nc) as ds_h:
                herbie_mean = ds_h["tp_ensemble_mean"].isel(time=0).values
                herbie_std = ds_h["tp_ensemble_standard_deviation"].isel(time=0).values
                herbie_n = int(ds_h.attrs.get("n_ensemble_members", 0))
        elif not args.skip_herbie:
            t0 = time.time()
            herbie_mean, herbie_std, herbie_n = fetch_herbie_tp(
                ds, TARGET_STEPS,
                max_members=max_members,
                logger=logger,
            )
            elapsed = time.time() - t0
            logger.info(f"  Herbie done: {herbie_n} members in {elapsed:.1f}s")

            if herbie_mean is not None:
                save_tp_netcdf(herbie_nc, herbie_mean, herbie_std, TARGET_STEPS,
                               ds, herbie_n, "Herbie GEFS atmos.25")
                logger.info(f"  Herbie saved: {herbie_nc}")
            else:
                logger.warning(f"  Herbie: no data for {ds}")
        else:
            logger.info("  Herbie: skipped (no cached NC)")

        # ── Compare ──
        if gik_mean is not None and herbie_mean is not None:
            # Handle potential shape mismatch in valid_time dimension
            min_steps = min(gik_mean.shape[0], herbie_mean.shape[0])
            effective_step_idx = min(STEP_INDEX, min_steps - 1)

            for label, g, h in [("mean", gik_mean, herbie_mean),
                                ("spread", gik_std, herbie_std)]:
                m = compute_metrics(g, h, label, step_idx=effective_step_idx)
                result.update(m)
                corr = m.get(f"{label}_corr", float("nan"))
                rmse = m.get(f"{label}_rmse", float("nan"))
                logger.info(f"  {label}: r={corr:.8f}  RMSE={rmse:.2e}")

            # Plot
            try:
                plot_comparison(ds, gik_mean, herbie_mean, gik_std, herbie_std,
                                result, out_dir, step_idx=effective_step_idx)
                logger.info(f"  Plot saved: {out_dir / f'compare_{ds}.png'}")
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

            all_gik_means.append(gik_mean)
            all_herbie_means.append(herbie_mean)
            all_dates_ok.append(ds)
        else:
            result["error"] = "missing data"
            logger.warning(f"  Skipping comparison — missing data")

        all_results.append(result)
        result["gik_members"] = gik_n
        result["herbie_members"] = herbie_n

    # ── Cleanup mapping manager ──
    if mapping_manager is not None:
        mapping_manager.cleanup()

    # ── Scatter plot ──
    if all_gik_means:
        try:
            plot_scatter_all(all_gik_means, all_herbie_means, all_dates_ok, out_dir)
            logger.info(f"Scatter plot saved: {out_dir / 'scatter_all_dates.png'}")
        except Exception as e:
            logger.warning(f"Scatter plot failed: {e}")

    # ── Save stats JSON ──
    stats_file = out_dir / "validation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Stats saved: {stats_file}")

    # ── Summary table ──
    total_time = time.time() - overall_t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Date':<12} {'Mean r':<16} {'Mean RMSE':<14} {'Spread r':<16} {'Spread RMSE':<14}")
    logger.info("-" * 72)
    for r in all_results:
        if "error" in r:
            logger.info(f"{r['date']:<12} ERROR: {r.get('error', '')}")
            continue
        mc = r.get("mean_corr", float("nan"))
        mr = r.get("mean_rmse", float("nan"))
        sc = r.get("spread_corr", float("nan"))
        sr = r.get("spread_rmse", float("nan"))
        logger.info(f"{r['date']:<12} {mc:<16.8f} {mr:<14.2e} {sc:<16.8f} {sr:<14.2e}")
    logger.info("-" * 72)
    n_ok = sum(1 for r in all_results if "error" not in r)
    logger.info(f"Successful: {n_ok}/{len(dates)} dates")
    logger.info(f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"Full log: {out_dir / 'validation_log.txt'}")


if __name__ == "__main__":
    main()
