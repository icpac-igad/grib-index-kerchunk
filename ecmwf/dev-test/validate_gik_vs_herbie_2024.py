#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "pyarrow",
#     "netcdf4",
#     "gribberish",
#     "cfgrib",
#     "eccodes",
#     "herbie-data",
#     "matplotlib",
#     "cartopy",
#     "scipy",
# ]
# ///
"""
Validate GIK parquet references against Herbie for 10 random 2024 dates.

Reads pre-built parquet files from the 2024 nested GCS path:
    gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/{MM}/{YYYYMMDD}/00z/

For each date:
  1) Streams tp from GIK parquets (all 51 members → ensemble mean & std)
  2) Fetches tp via Herbie (same members → ensemble mean & std)
  3) Compares grid-point values with correlation, RMSE, MAE

Output directory: validation_gik_vs_herbie_2024/

Usage:
    uv run validate_gik_vs_herbie_2024.py
    uv run validate_gik_vs_herbie_2024.py --dry-run
    uv run validate_gik_vs_herbie_2024.py --max-members 5
    uv run validate_gik_vs_herbie_2024.py --dates 20240315,20240406
"""

import argparse
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

# ── GCS credentials ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
GCS_SA_FILE = str(SCRIPT_DIR / "coiled-data.json")
GCS_BUCKET = "gik-ecmwf-aws-tf"
GCS_PREFIX = "run_par_ecmwf/2024"

# ── Forecast configuration ───────────────────────────────────────────
TARGET_STEPS = [36, 39, 42, 45, 48, 51, 54, 57, 60]

# ECMWF grid -- from grids.py, the single source of truth. Do not retype these
# axes; a hand-written 0-start longitude is what displaced every Icechunk store
# by 180 deg (HANDOVER_LONGITUDE_FIX.md).
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grids import field_shape as _field_shape, latitudes as _lats, longitudes as _lons  # noqa: E402

ECMWF_GRID_SHAPE = _field_shape("0p25")
ECMWF_LATS = _lats("0p25")
ECMWF_LONS = _lons("0p25")

# ICPAC region
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Precompute ICPAC indices
_lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
_lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
ICPAC_LATS = ECMWF_LATS[LAT_INDICES[0] : LAT_INDICES[-1] + 1]
ICPAC_LONS = ECMWF_LONS[LON_INDICES[0] : LON_INDICES[-1] + 1]

# Output
OUTPUT_DIR = SCRIPT_DIR / "validation_gik_vs_herbie_2024"

# Try gribberish
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False


# ── Date selection ────────────────────────────────────────────────────

def generate_10_dates(seed: int = 42) -> list[date]:
    """Pick one random date per month for March–December 2024 (10 dates)."""
    rng = random.Random(seed)
    dates = []
    for month in range(3, 13):
        last_day = calendar.monthrange(2024, month)[1]
        day = rng.randint(1, last_day)
        dates.append(date(2024, month, day))
    return dates


# ── GCS helpers ───────────────────────────────────────────────────────

def get_gcs_fs():
    import gcsfs
    return gcsfs.GCSFileSystem(token=GCS_SA_FILE)


def list_member_parquets(date_str: str, max_members: int = None) -> list[str]:
    """List parquet files for a date on GCS (2024 nested path)."""
    month = date_str[4:6]
    gcs_path = f"{GCS_BUCKET}/{GCS_PREFIX}/{month}/{date_str}/00z"
    fs = get_gcs_fs()
    files = sorted(fs.glob(f"{gcs_path}/*.parquet"))
    if max_members and len(files) > max_members:
        files = files[:max_members]
    return [f"gs://{f}" for f in files]


def member_key_from_filename(path: str) -> str:
    """Extract member key from parquet filename.

    '2024040600z-control.parquet' -> 'control'
    '2024040600z-ens_01.parquet'  -> 'ens01'
    """
    stem = os.path.basename(path).replace(".parquet", "")
    # Everything after the first '-'
    parts = stem.split("-", 1)
    raw = parts[1] if len(parts) > 1 else parts[0]
    return raw.replace("_", "")


# ── Parquet reading & GRIB decoding ──────────────────────────────────

def read_parquet_refs(parquet_path: str) -> Dict:
    """Read parquet file into a {key: value} zstore dict."""
    if parquet_path.startswith("gs://"):
        fs = get_gcs_fs()
        df = pd.read_parquet(parquet_path, filesystem=fs)
    else:
        df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in df.iterrows():
        key = row["key"]
        value = row["value"]
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
                if decoded.startswith("[") or decoded.startswith("{"):
                    value = json.loads(decoded)
                else:
                    value = decoded
            except Exception:
                pass
        elif isinstance(value, str):
            if value.startswith("[") or value.startswith("{"):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
        zstore[key] = value
    return zstore


def fetch_grib_bytes(ref: list, fs) -> bytes:
    url, offset, length = ref[0], ref[1], ref[2]
    if not url.endswith(".grib2"):
        url = url + ".grib2"
    with fs.open(url, "rb") as f:
        f.seek(offset)
        return f.read(length)


def decode_grib_bytes(grib_bytes: bytes) -> np.ndarray:
    if GRIBBERISH_AVAILABLE:
        try:
            flat = gribberish.parse_grib_array(grib_bytes, 0)
            return flat.reshape(ECMWF_GRID_SHAPE)
        except Exception:
            pass
    # Fallback cfgrib
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
    return data[LAT_INDICES[0] : LAT_INDICES[-1] + 1,
                LON_INDICES[0] : LON_INDICES[-1] + 1]


# ── GIK streaming ────────────────────────────────────────────────────

def _fetch_one_step(step, ref, fs):
    try:
        raw = fetch_grib_bytes(ref, fs)
        arr = decode_grib_bytes(raw)
        return step, subset_icpac(arr).astype(np.float32), None
    except Exception as e:
        return step, None, str(e)


def stream_tp_one_member(parquet_path: str, member_key: str,
                         step_hours: list[int]) -> Optional[np.ndarray]:
    """Stream tp for one member. Returns (n_steps, lat, lon) or None."""
    zstore = read_parquet_refs(parquet_path)
    fs = fsspec.filesystem("s3", anon=True)

    chunks = []
    for step in step_hours:
        for pat in [
            f"step_{step:03d}/tp/sfc/{member_key}/0.0.0",
            f"step_{step:03d}/tp/sfc/0.0.0",
            f"step_{step:03d}/tp/surface/{member_key}/0.0.0",
        ]:
            if pat in zstore:
                ref = zstore[pat]
                if isinstance(ref, list) and len(ref) >= 3:
                    chunks.append((step, ref))
                    break

    if not chunks:
        return None

    n_lats, n_lons = len(ICPAC_LATS), len(ICPAC_LONS)
    data = np.full((len(step_hours), n_lats, n_lons), np.nan, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
        futs = {pool.submit(_fetch_one_step, s, r, fs): s for s, r in chunks}
        for fut in as_completed(futs):
            step, arr, err = fut.result()
            if arr is not None:
                data[step_hours.index(step)] = arr

    return data


def stream_gik_tp(date_str: str, step_hours: list[int],
                  max_members: int = None,
                  parallel_members: int = 8,
                  logger=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Stream tp ensemble from GIK parquets.

    Returns (mean, std, n_members) arrays of shape (n_steps, lat, lon).
    """
    log = logger.info if logger else print

    parquets = list_member_parquets(date_str, max_members)
    n_members = len(parquets)
    log(f"  GIK: found {n_members} parquets for {date_str}")

    if not parquets:
        return None, None, 0

    n_steps = len(step_hours)
    n_lats, n_lons = len(ICPAC_LATS), len(ICPAC_LONS)
    all_data = np.full((n_members, n_steps, n_lats, n_lons), np.nan, dtype=np.float32)

    def _worker(idx_path):
        idx, path = idx_path
        mk = member_key_from_filename(path)
        return idx, stream_tp_one_member(path, mk, step_hours)

    done = 0
    with ThreadPoolExecutor(max_workers=parallel_members) as pool:
        futs = {pool.submit(_worker, (i, p)): i for i, p in enumerate(parquets)}
        for fut in as_completed(futs):
            idx, data = fut.result()
            if data is not None:
                all_data[idx] = data
            done += 1
            if done % 10 == 0:
                log(f"    GIK: {done}/{n_members} members done")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(all_data, axis=0)
        std = np.nanstd(all_data, axis=0)

    gc.collect()
    return mean, std, n_members


# ── Herbie fetching ───────────────────────────────────────────────────

def herbie_subset_icpac(ds: xr.Dataset) -> xr.Dataset:
    lons = ds["longitude"].values
    if lons.max() > 180:
        ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180)
        ds = ds.sortby("longitude")
    return ds.sel(latitude=slice(LAT_MAX, LAT_MIN),
                  longitude=slice(LON_MIN, LON_MAX))


def fetch_herbie_tp(date_str: str, step_hours: list[int],
                    max_members: int = None,
                    logger=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Fetch tp ensemble via Herbie. Returns (mean, std, n_members)."""
    from herbie import Herbie

    log = logger.info if logger else print
    log_warn = logger.warning if logger else print

    step_data = {}
    icpac_lats = None
    icpac_lons = None

    for fxx in step_hours:
        log(f"  Herbie: step T+{fxx}h ...")
        try:
            H = Herbie(f"{date_str} 00:00", model="ifs", product="enfo", fxx=fxx)
            ds_list = H.xarray(":tp:", verbose=False)

            if isinstance(ds_list, xr.Dataset):
                ds_list = [ds_list]

            ens_ds = None
            for ds in ds_list if isinstance(ds_list, list) else [ds_list]:
                if "number" in ds.dims:
                    ens_ds = ds
                    break
            if ens_ds is None:
                ens_ds = ds_list[0] if isinstance(ds_list, list) else ds_list

            tp_var = None
            for v in ens_ds.data_vars:
                if "tp" in v.lower() or ens_ds[v].attrs.get("shortName", "") == "tp":
                    tp_var = v
                    break
            if tp_var is None:
                tp_var = list(ens_ds.data_vars)[0]

            sub = herbie_subset_icpac(ens_ds)
            if icpac_lats is None:
                icpac_lats = sub.latitude.values
                icpac_lons = sub.longitude.values

            arr = sub[tp_var].values
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            if max_members and arr.shape[0] > max_members:
                arr = arr[:max_members]

            step_data[fxx] = arr.astype(np.float32)
            log(f"    OK — {arr.shape[0]} members")
            ens_ds.close()
        except Exception as e:
            log_warn(f"    FAILED at T+{fxx}h: {e}")

    if not step_data:
        return None, None, 0

    ordered = [s for s in step_hours if s in step_data]
    n_members = step_data[ordered[0]].shape[0]
    n_lats = len(icpac_lats)
    n_lons = len(icpac_lons)

    mean = np.full((len(ordered), n_lats, n_lons), np.nan, dtype=np.float32)
    std = np.full((len(ordered), n_lats, n_lons), np.nan, dtype=np.float32)
    for i, fxx in enumerate(ordered):
        mean[i] = np.nanmean(step_data[fxx], axis=0)
        std[i] = np.nanstd(step_data[fxx], axis=0)

    return mean, std, n_members


# ── Comparison metrics ────────────────────────────────────────────────

def compute_metrics(gik: np.ndarray, herbie: np.ndarray, label: str) -> dict:
    """Compare two (n_steps, lat, lon) arrays at STEP_INDEX=4 (T+48h)."""
    from scipy import stats as sp_stats

    step_idx = 4  # T+48h
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


# ── NetCDF save helper ───────────────────────────────────────────────

def save_tp_netcdf(path: Path, mean: np.ndarray, std: np.ndarray,
                   step_hours: list[int], date_str: str, n_members: int,
                   source: str):
    model_date = datetime.strptime(date_str, "%Y%m%d")
    base_time = model_date
    valid_times = [base_time + timedelta(hours=h) for h in step_hours]

    ds = xr.Dataset(
        {
            "tp_ensemble_mean": xr.DataArray(
                mean[np.newaxis], dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "tp ensemble mean", "units": "m"},
            ),
            "tp_ensemble_standard_deviation": xr.DataArray(
                std[np.newaxis], dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "tp ensemble std", "units": "m"},
            ),
        },
        coords={"time": [base_time], "valid_time": valid_times,
                "latitude": ICPAC_LATS, "longitude": ICPAC_LONS},
    )
    ds.attrs["source"] = source
    ds.attrs["model_date"] = date_str
    ds.attrs["n_ensemble_members"] = n_members
    ds.attrs["forecast_hours"] = str(step_hours)

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    ds.close()


# ── Plotting ──────────────────────────────────────────────────────────

def plot_comparison(date_str: str, gik_mean: np.ndarray, herbie_mean: np.ndarray,
                    gik_std: np.ndarray, herbie_std: np.ndarray,
                    metrics: dict, output_dir: Path):
    """Create GIK | Herbie | Difference maps for mean & spread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    step_idx = 4
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05],
                           hspace=0.25, wspace=0.15)

    rows = [
        (gik_mean[step_idx] * 1000, herbie_mean[step_idx] * 1000,
         "Ensemble Mean (mm)", "YlGnBu"),
        (gik_std[step_idx] * 1000, herbie_std[step_idx] * 1000,
         "Ensemble Spread (mm)", "Oranges"),
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
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.set_title(ptitle, fontsize=10, fontweight="bold")

            if col_idx == 2:
                cbar_ax = fig.add_subplot(gs[row_idx, 3])
                fig.colorbar(im, cax=cbar_ax).set_label("mm", fontsize=9)

    mc = metrics.get("mean_corr", float("nan"))
    mr = metrics.get("mean_rmse", float("nan"))
    fig.suptitle(
        f"GIK vs Herbie — {date_str} 00Z — T+48h\n"
        f"Mean r={mc:.8f}  RMSE={mr:.2e}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out = output_dir / f"compare_{date_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def plot_scatter_all(all_gik_means, all_herbie_means, output_dir: Path):
    """Scatter plot of all grid points across all dates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    all_g = np.concatenate([g[4].flatten() for g in all_gik_means])
    all_h = np.concatenate([h[4].flatten() for h in all_herbie_means])
    valid = ~(np.isnan(all_g) | np.isnan(all_h))
    all_g, all_h = all_g[valid], all_h[valid]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_h * 1000, all_g * 1000, s=1, alpha=0.3, c="steelblue")
    lim = max(np.max(all_g), np.max(all_h)) * 1000 * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="1:1")
    r, _ = sp_stats.pearsonr(all_g, all_h)
    ax.set_xlabel("Herbie tp (mm)")
    ax.set_ylabel("GIK tp (mm)")
    ax.set_title(f"GIK vs Herbie — All 10 dates at T+48h\nr = {r:.8f}  N = {len(all_g):,}",
                 fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    out = output_dir / "scatter_all_dates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# ── Logging ───────────────────────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_gik_herbie")
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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate GIK vs Herbie TP for 10 random 2024 dates")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected dates and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap ensemble members (default: all 51)")
    parser.add_argument("--parallel-members", type=int, default=8)
    parser.add_argument("--dates", type=str, default=None,
                        help="Override: comma-separated YYYYMMDD dates")
    parser.add_argument("--skip-herbie", action="store_true",
                        help="Skip Herbie fetch (use existing NC files)")
    parser.add_argument("--skip-gik", action="store_true",
                        help="Skip GIK streaming (use existing NC files)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.dates:
        dates = [datetime.strptime(d.strip(), "%Y%m%d").date()
                 for d in args.dates.split(",")]
    else:
        dates = generate_10_dates(seed=args.seed)

    if args.dry_run:
        print(f"10 random dates (seed={args.seed}):")
        for d in dates:
            print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%B %Y')})")
        return

    logger = setup_logging(out_dir)
    logger.info("=" * 70)
    logger.info("GIK vs Herbie Validation — 2024 parquets")
    logger.info("=" * 70)
    logger.info(f"Dates: {len(dates)}  |  Steps: {TARGET_STEPS}")
    logger.info(f"Max members: {args.max_members or 'all'}")
    logger.info(f"Output: {out_dir}")
    logger.info("")

    overall_t0 = time.time()
    all_results = []
    all_gik_means = []
    all_herbie_means = []

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
        else:
            t0 = time.time()
            gik_mean, gik_std, gik_n = stream_gik_tp(
                ds, TARGET_STEPS,
                max_members=args.max_members,
                parallel_members=args.parallel_members,
                logger=logger,
            )
            elapsed = time.time() - t0
            logger.info(f"  GIK done: {gik_n} members in {elapsed:.1f}s")

            if gik_mean is not None:
                save_tp_netcdf(gik_nc, gik_mean, gik_std, TARGET_STEPS,
                               ds, gik_n, "GIK parquet streaming")
                logger.info(f"  GIK saved: {gik_nc}")

        # ── Herbie fetching ──
        herbie_mean = herbie_std = None
        herbie_n = 0
        if args.skip_herbie and herbie_nc.exists():
            logger.info("  Herbie: loading cached NC")
            with xr.open_dataset(herbie_nc) as ds_h:
                herbie_mean = ds_h["tp_ensemble_mean"].isel(time=0).values
                herbie_std = ds_h["tp_ensemble_standard_deviation"].isel(time=0).values
                herbie_n = int(ds_h.attrs.get("n_ensemble_members", 0))
        else:
            t0 = time.time()
            herbie_mean, herbie_std, herbie_n = fetch_herbie_tp(
                ds, TARGET_STEPS,
                max_members=args.max_members,
                logger=logger,
            )
            elapsed = time.time() - t0
            logger.info(f"  Herbie done: {herbie_n} members in {elapsed:.1f}s")

            if herbie_mean is not None:
                save_tp_netcdf(herbie_nc, herbie_mean, herbie_std, TARGET_STEPS,
                               ds, herbie_n, "Herbie ECMWF IFS enfo")
                logger.info(f"  Herbie saved: {herbie_nc}")

        # ── Compare ──
        if gik_mean is not None and herbie_mean is not None:
            for label, g, h in [("mean", gik_mean, herbie_mean),
                                ("spread", gik_std, herbie_std)]:
                m = compute_metrics(g, h, label)
                result.update(m)
                corr = m.get(f"{label}_corr", float("nan"))
                rmse = m.get(f"{label}_rmse", float("nan"))
                logger.info(f"  {label}: r={corr:.8f}  RMSE={rmse:.2e}")

            # Plot
            try:
                plot_comparison(ds, gik_mean, herbie_mean, gik_std, herbie_std,
                                result, out_dir)
                logger.info(f"  Plot saved: {out_dir / f'compare_{ds}.png'}")
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

            all_gik_means.append(gik_mean)
            all_herbie_means.append(herbie_mean)
        else:
            result["error"] = "missing data"
            logger.warning(f"  Skipping comparison — missing data")

        all_results.append(result)

    # ── Scatter plot ──
    if all_gik_means:
        try:
            plot_scatter_all(all_gik_means, all_herbie_means, out_dir)
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
            logger.info(f"{r['date']:<12} ERROR")
            continue
        mc = r.get("mean_corr", float("nan"))
        mr = r.get("mean_rmse", float("nan"))
        sc = r.get("spread_corr", float("nan"))
        sr = r.get("spread_rmse", float("nan"))
        logger.info(f"{r['date']:<12} {mc:<16.8f} {mr:<14.2e} {sc:<16.8f} {sr:<14.2e}")
    logger.info("-" * 72)
    logger.info(f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"Full log: {out_dir / 'validation_log.txt'}")


if __name__ == "__main__":
    main()
