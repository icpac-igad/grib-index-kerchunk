#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
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
#     "huggingface_hub",
# ]
# ///
"""
Validate GIK parquet references from HuggingFace against Herbie for GEFS APCP.

This script consumes the published `E4DRR/gik-gefs-par` HuggingFace dataset
(monthly aggregates) instead of running the GIK 3-stage pipeline locally.
It proves the public dataset works end-to-end:

  HF monthly_agg   →  filter pushdown by (date, member)
                  →  reconstruct kerchunk zarr store
                  →  stream APCP byte-ranges from NOAA s3://noaa-gefs-pds/
                  →  ensemble mean & std
                  →  compare against Herbie GEFS atmos.25 fetch.

Default behaviour: pick one random date and run the comparison.

Usage:
    uv run validate_hf_gik_vs_herbie.py                  # random date, all 30 members
    uv run validate_hf_gik_vs_herbie.py --date 20240615  # specific date
    uv run validate_hf_gik_vs_herbie.py --max-members 5  # quick sanity check
    uv run validate_hf_gik_vs_herbie.py --skip-herbie    # use cached Herbie NC
    uv run validate_hf_gik_vs_herbie.py --seed 1         # different random date
"""

import argparse
import ast
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

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# ── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
HF_REPO = os.environ.get("HF_REPO", "E4DRR/gik-gefs-par")
HF_AGG_PREFIX = "run_par_gefs_agg/monthly_agg"
HF_AVAILABLE_DATES = ("20200925", "20251231")  # NOAA archive bounds for 00z

# ── GEFS 0.25° grid (0-360 longitude) ────────────────────────────────
GEFS_GRID_SHAPE = (721, 1440)
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# ── East Africa / ICPAC region ───────────────────────────────────────
LAT_MIN, LAT_MAX = -12, 15
LON_MIN, LON_MAX = 25, 52
_lat_mask = (GEFS_LATS >= LAT_MIN) & (GEFS_LATS <= LAT_MAX)
_lon_mask = (GEFS_LONS >= LON_MIN) & (GEFS_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
ICPAC_LATS = GEFS_LATS[LAT_INDICES[0] : LAT_INDICES[-1] + 1]
ICPAC_LONS = GEFS_LONS[LON_INDICES[0] : LON_INDICES[-1] + 1]
EA_EXTENT = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

# ── Forecast steps (skip T+0h — APCP is undefined at init) ───────────
TARGET_STEPS = [3, 6, 12, 24, 48, 72, 120, 168, 240]
STEP_INDEX = 4  # T+48h for the comparison metrics + plot

OUTPUT_DIR = SCRIPT_DIR / "validation_hf_gik_vs_herbie"

# ── Optional decoder ─────────────────────────────────────────────────
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False


# ── Date selection ───────────────────────────────────────────────────

def random_valid_date(seed: int) -> date:
    """Pick a random date within the published HF coverage (00z 2020-09-25 onwards)."""
    rng = random.Random(seed)
    start = datetime.strptime(HF_AVAILABLE_DATES[0], "%Y%m%d").date()
    end = datetime.strptime(HF_AVAILABLE_DATES[1], "%Y%m%d").date()
    days = (end - start).days
    return start + timedelta(days=rng.randint(0, days))


# ── HF aggregate helpers ─────────────────────────────────────────────

def hf_aggregate_url(date_str: str) -> str:
    """Return hf:// URL of the monthly aggregate covering this date."""
    yyyy, mm = date_str[:4], date_str[4:6]
    return (f"hf://datasets/{HF_REPO}/{HF_AGG_PREFIX}/"
            f"{yyyy}/{mm}_00z.parquet")


def fetch_member_zstores(date_str: str, max_members: int,
                         logger=None) -> Tuple[Dict[str, dict], dict]:
    """Read the monthly aggregate, filter to date+all-members, group rows by
    member and rebuild a {key: value} kerchunk store dict per member.

    Returns ({member_id: zstore_dict}, timing_info).
    """
    log = logger.info if logger else print

    url = hf_aggregate_url(date_str)
    log(f"  HF read: {url}")
    log(f"  Filter:  date={date_str}, member in (gep01..gep{max_members:02d})")

    members_wanted = [f"gep{i:02d}" for i in range(1, max_members + 1)]

    t0 = time.time()
    df = pd.read_parquet(
        url,
        filters=[
            ("date", "=", date_str),
            ("member", "in", members_wanted),
        ],
    )
    elapsed = time.time() - t0
    log(f"  Read {len(df):,} rows for {df['member'].nunique()} members in "
        f"{elapsed:.1f}s")

    timing = {"hf_read_s": round(elapsed, 2), "rows": len(df)}

    # GIK parquets store the entire zstore as one dict literal in the row
    # where key="refs" (a 1.5 MB bytes blob). Parse with ast.literal_eval so
    # nested lists (kerchunk byte-range refs) come back as Python lists.
    zstores: Dict[str, dict] = {}
    for member, sub in df.groupby("member"):
        refs_rows = sub[sub["key"] == "refs"]
        if len(refs_rows) != 1:
            log(f"    {member}: expected 1 'refs' row, got {len(refs_rows)} — skipping")
            continue
        blob = refs_rows.iloc[0]["value"]
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8")
        try:
            zs = ast.literal_eval(blob)
        except (SyntaxError, ValueError) as e:
            log(f"    {member}: literal_eval failed ({e}) — skipping")
            continue
        if not isinstance(zs, dict):
            log(f"    {member}: refs blob did not parse as dict — skipping")
            continue
        zstores[member] = zs

    if zstores:
        sample_keys = len(next(iter(zstores.values())))
        log(f"  Reconstructed {len(zstores)} member zstores "
            f"({sample_keys} zarr keys each)")
    return zstores, timing


# ── GRIB byte-range streaming (same as validate_gik_vs_herbie_2022_2026) ──

def fetch_grib_bytes(ref: list, fs) -> bytes:
    url, offset, length = ref[0], ref[1], ref[2]
    with fs.open(url, "rb") as f:
        f.seek(offset)
        return f.read(length)


def decode_grib_bytes(grib_bytes: bytes) -> np.ndarray:
    if GRIBBERISH_AVAILABLE:
        try:
            flat = gribberish.parse_grib_array(grib_bytes, 0)
            return flat.reshape(GEFS_GRID_SHAPE)
        except Exception:
            pass
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


def stream_tp_from_zstore(zstore: dict,
                          target_step_hours: List[int]) -> Tuple[np.ndarray, float]:
    """Stream APCP timesteps from a zarr store dict using gribberish.

    Returns ((n_steps, lat, lon) ICPAC-region array, streaming_seconds).
    """
    t_start = time.time()
    fs = fsspec.filesystem("s3", anon=True)

    tp_chunk_refs = {}
    for key, value in zstore.items():
        if key.startswith("tp/accum/surface/tp/") and isinstance(value, list) \
                and len(value) >= 3:
            tail = key.split("/")[-1]  # e.g. "1.0.0"
            try:
                step_idx = int(tail.split(".")[0])
            except ValueError:
                continue
            tp_chunk_refs[step_idx] = value

    # step→hours map from zarr coord
    step_data_key = "tp/accum/surface/step/0"
    step_hours_all = None
    if step_data_key in zstore:
        val = zstore[step_data_key]
        if isinstance(val, str) and val.startswith("base64:"):
            step_hours_all = np.frombuffer(base64.b64decode(val[7:]), dtype="<f8")
        elif isinstance(val, list):
            try:
                step_hours_all = np.frombuffer(fetch_grib_bytes(val, fs),
                                                dtype="<f8")
            except Exception:
                pass
    if step_hours_all is None:
        step_hours_all = np.arange(0, 243, 3, dtype=float)

    n_lats, n_lons = len(ICPAC_LATS), len(ICPAC_LONS)
    result = np.full((len(target_step_hours), n_lats, n_lons),
                     np.nan, dtype=np.float32)

    def _fetch_one(out_idx, ref):
        try:
            arr = decode_grib_bytes(fetch_grib_bytes(ref, fs))
            return out_idx, subset_icpac(arr).astype(np.float32)
        except Exception:
            return out_idx, None

    tasks = []
    for out_idx, target_h in enumerate(target_step_hours):
        matches = np.where(np.abs(step_hours_all - target_h) < 0.5)[0]
        if len(matches) > 0:
            step_idx = int(matches[0])
            if step_idx in tp_chunk_refs:
                tasks.append((out_idx, tp_chunk_refs[step_idx]))

    if not tasks:
        return result, time.time() - t_start

    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        futs = {pool.submit(_fetch_one, oi, r): oi for oi, r in tasks}
        for fut in as_completed(futs):
            out_idx, arr = fut.result()
            if arr is not None:
                result[out_idx] = arr

    return result, time.time() - t_start


def stream_gik_tp_from_hf(date_str: str, step_hours: List[int],
                          max_members: int = 30,
                          logger=None) -> Tuple[Optional[np.ndarray],
                                                 Optional[np.ndarray],
                                                 int, dict]:
    """Build ensemble mean & std by streaming APCP from HF aggregate refs."""
    log = logger.info if logger else print

    zstores, hf_timing = fetch_member_zstores(date_str, max_members, logger)
    if not zstores:
        return None, None, 0, hf_timing

    member_data = []
    stream_times = []
    for i, (member, zs) in enumerate(sorted(zstores.items()), 1):
        tp, secs = stream_tp_from_zstore(zs, step_hours)
        stream_times.append(secs)
        if np.count_nonzero(~np.isnan(tp)) == 0:
            log(f"    GIK: {member} no valid TP data, skipping")
            continue
        member_data.append(tp)
        if i % 5 == 0:
            log(f"    GIK: {i}/{len(zstores)} members streamed")

    if not member_data:
        return None, None, 0, hf_timing

    stacked = np.stack(member_data, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)

    timing = dict(hf_timing)
    timing["stream_total_s"] = round(sum(stream_times), 1)
    timing["stream_per_member_s"] = round(np.mean(stream_times), 2) if stream_times else 0
    timing["n_members"] = len(member_data)

    log(f"  GIK from HF: {len(member_data)} members, "
        f"stream={timing['stream_total_s']}s "
        f"(per member: {timing['stream_per_member_s']}s)")

    gc.collect()
    return mean, std, len(member_data), timing


# ── Herbie ───────────────────────────────────────────────────────────

def subset_ea(ds: xr.Dataset) -> xr.Dataset:
    lons = ds["longitude"].values
    if lons.max() > 180:
        ds = ds.assign_coords(
            {"longitude": ((ds["longitude"] + 180) % 360) - 180})
        ds = ds.sortby("longitude")
    lats = ds["latitude"].values
    if lats[0] > lats[-1]:
        return ds.sel(latitude=slice(LAT_MAX, LAT_MIN),
                      longitude=slice(LON_MIN, LON_MAX))
    return ds.sel(latitude=slice(LAT_MIN, LAT_MAX),
                  longitude=slice(LON_MIN, LON_MAX))


def fetch_herbie_tp(date_str: str, step_hours: List[int],
                    max_members: int = 30,
                    logger=None) -> Tuple[Optional[np.ndarray],
                                           Optional[np.ndarray],
                                           int, dict]:
    from herbie import Herbie

    log = logger.info if logger else print
    log_warn = logger.warning if logger else print

    step_arrays: Dict[int, np.ndarray] = {}
    ea_lats = ea_lons = None
    step_times = []
    member_times = []

    for fxx in step_hours:
        log(f"  Herbie: T+{fxx}h ...")
        t_step = time.time()
        member_data = []
        for mem_num in range(1, max_members + 1):
            try:
                t_mem = time.time()
                H = Herbie(f"{date_str} 00:00", model="gefs",
                           product="atmos.25", member=mem_num, fxx=fxx)
                ds = H.xarray(":APCP:surface:", verbose=False)
                member_times.append(time.time() - t_mem)
                if isinstance(ds, list):
                    ds = ds[0]
                tp_var = next((v for v in ds.data_vars
                                if "apcp" in v.lower() or "tp" in v.lower()),
                              list(ds.data_vars)[0])
                ds_sub = subset_ea(ds)
                if ea_lats is None:
                    ea_lats = ds_sub.latitude.values
                    ea_lons = ds_sub.longitude.values
                arr = ds_sub[tp_var].values
                while arr.ndim > 2:
                    arr = arr[0]
                member_data.append(arr.astype(np.float32))
                ds.close()
            except Exception as e:
                if mem_num <= 3:
                    log_warn(f"    member {mem_num} FAILED: {e}")
        step_times.append(time.time() - t_step)
        if member_data:
            step_arrays[fxx] = np.stack(member_data, axis=0)
            log(f"    {step_arrays[fxx].shape[0]} members ({step_times[-1]:.1f}s)")
        else:
            log_warn(f"    NO members for T+{fxx}h")

    if not step_arrays:
        return None, None, 0, {}

    ordered = [s for s in step_hours if s in step_arrays]
    n_lats = len(ea_lats); n_lons = len(ea_lons)
    n_members = step_arrays[ordered[0]].shape[0]
    mean = np.full((len(ordered), n_lats, n_lons), np.nan, dtype=np.float32)
    std = np.full((len(ordered), n_lats, n_lons), np.nan, dtype=np.float32)
    for i, fxx in enumerate(ordered):
        mean[i] = np.nanmean(step_arrays[fxx], axis=0)
        std[i] = np.nanstd(step_arrays[fxx], axis=0)

    timing = {
        "total_s": round(sum(step_times), 1),
        "per_step_avg_s": round(np.mean(step_times), 2) if step_times else 0,
        "per_member_avg_s": round(np.mean(member_times), 2) if member_times else 0,
        "n_fetches": len(member_times),
    }
    log(f"  Herbie: total={timing['total_s']}s, "
        f"per_member={timing['per_member_avg_s']}s ({timing['n_fetches']} fetches)")
    return mean, std, n_members, timing


# ── Metrics + I/O + plots (mirrors validate_gik_vs_herbie_2022_2026.py) ──

def compute_metrics(gik: np.ndarray, herbie: np.ndarray, label: str,
                    step_idx: int = STEP_INDEX) -> dict:
    from scipy import stats as sp_stats
    if step_idx >= gik.shape[0] or step_idx >= herbie.shape[0]:
        return {f"{label}_error": f"step_idx {step_idx} out of range"}
    g = gik[step_idx]; h = herbie[step_idx]
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
    return {
        f"{label}_corr": float(r),
        f"{label}_corr_p": float(p),
        f"{label}_rmse": rmse,
        f"{label}_mae": mae,
        f"{label}_max_abs_diff": float(np.nanmax(np.abs(diff))),
        f"{label}_rel_diff_pct": float(mae / max(np.nanmean(np.abs(gv)), 1e-12) * 100),
        f"{label}_gik_range": [float(np.nanmin(g)), float(np.nanmax(g))],
        f"{label}_herbie_range": [float(np.nanmin(h)), float(np.nanmax(h))],
    }


def save_tp_netcdf(path: Path, mean: np.ndarray, std: np.ndarray,
                   step_hours: List[int], date_str: str, n_members: int,
                   source: str):
    base_time = datetime.strptime(date_str, "%Y%m%d")
    valid_times = [base_time + timedelta(hours=h) for h in step_hours]
    ds = xr.Dataset(
        {
            "tp_ensemble_mean": xr.DataArray(
                mean[np.newaxis].astype(np.float32),
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "APCP ensemble mean", "units": "kg m-2"}),
            "tp_ensemble_standard_deviation": xr.DataArray(
                std[np.newaxis].astype(np.float32),
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "APCP ensemble std", "units": "kg m-2"}),
        },
        coords={
            "time": [base_time], "valid_time": valid_times,
            "latitude": ICPAC_LATS, "longitude": ICPAC_LONS,
        },
    )
    ds.attrs.update(source=source, model_date=date_str,
                    n_ensemble_members=n_members,
                    forecast_hours=str(step_hours))
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    ds.close()


def plot_comparison(date_str: str, gik_mean: np.ndarray, herbie_mean: np.ndarray,
                    gik_std: np.ndarray, herbie_std: np.ndarray,
                    metrics: dict, output_dir: Path,
                    step_idx: int = STEP_INDEX) -> Path:
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
    for r_idx, (g, h, title, cmap) in enumerate(rows):
        diff = g - h
        vmax = max(np.nanmax(np.abs(g)), np.nanmax(np.abs(h)), 1e-6)
        diff_vmax = max(np.nanmax(np.abs(diff)), 1e-6)
        panels = [
            (g, f"GIK (HF) — {title}", cmap, 0, vmax),
            (h, f"Herbie — {title}", cmap, 0, vmax),
            (diff, "Difference (GIK-HF − Herbie)", "RdBu_r", -diff_vmax, diff_vmax),
        ]
        for c_idx, (data, ptitle, cm, vmin, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[r_idx, c_idx], projection=ccrs.PlateCarree())
            im = ax.pcolormesh(ICPAC_LONS, ICPAC_LATS, data, cmap=cm,
                               vmin=vmin, vmax=vmx,
                               transform=ccrs.PlateCarree(), shading="auto")
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
            ax.add_feature(cfeature.LAKES, alpha=0.3)
            ax.set_extent(EA_EXTENT, crs=ccrs.PlateCarree())
            ax.set_title(ptitle, fontsize=10, fontweight="bold")
            if c_idx == 2:
                cax = fig.add_subplot(gs[r_idx, 3])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label("kg/m2", fontsize=9)
                cb.ax.tick_params(labelsize=8)

    mc = metrics.get("mean_corr", float("nan"))
    mr = metrics.get("mean_rmse", float("nan"))
    fh = TARGET_STEPS[step_idx]
    fig.suptitle(
        f"GIK (HF aggregate) vs Herbie GEFS — {date_str} 00Z — T+{fh}h\n"
        f"Source: huggingface.co/datasets/{HF_REPO}\n"
        f"Mean r={mc:.6f}  RMSE={mr:.2e}",
        fontsize=12, fontweight="bold", y=0.99,
    )
    out = output_dir / f"compare_hf_{date_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# ── Logging + main ──────────────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("validate_hf_gik_vs_herbie")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    fh = logging.FileHandler(out_dir / "validation_log.txt", mode="a")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


def main():
    parser = argparse.ArgumentParser(
        description="Validate HF GEFS aggregate refs vs Herbie for one date.")
    parser.add_argument("--date", type=str, default=None,
                        help="Date YYYYMMDD (default: random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random date (default: time-based)")
    parser.add_argument("--max-members", type=int, default=30,
                        help="Cap ensemble members (default: 30)")
    parser.add_argument("--skip-herbie", action="store_true",
                        help="Skip Herbie fetch (use cached NC)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    logger = setup_logging(out_dir)

    if args.date:
        d = datetime.strptime(args.date, "%Y%m%d").date()
    else:
        seed = args.seed if args.seed is not None else int(time.time())
        d = random_valid_date(seed)
        logger.info(f"Random seed: {seed}")

    date_str = d.strftime("%Y%m%d")

    logger.info("=" * 70)
    logger.info(f"GIK (HF) vs Herbie — {date_str} 00Z  ({d.strftime('%B %d, %Y')})")
    logger.info(f"HF source: {HF_REPO}")
    logger.info(f"Members:   up to {args.max_members}")
    logger.info(f"Steps:     {TARGET_STEPS}")
    logger.info(f"Output:    {out_dir}")
    logger.info("=" * 70)

    result = {"date": date_str, "hf_repo": HF_REPO}
    overall_t0 = time.time()

    # ── GIK from HF aggregate ──
    t0 = time.time()
    gik_mean, gik_std, gik_n, gik_timing = stream_gik_tp_from_hf(
        date_str, TARGET_STEPS,
        max_members=args.max_members, logger=logger,
    )
    elapsed = time.time() - t0
    logger.info(f"GIK done: {gik_n} members in {elapsed:.1f}s")
    result["gik_total_s"] = round(elapsed, 1)
    result.update({f"gik_{k}": v for k, v in gik_timing.items()})

    if gik_mean is not None:
        gik_nc = out_dir / f"gik_hf_{date_str}_tp.nc"
        save_tp_netcdf(gik_nc, gik_mean, gik_std, TARGET_STEPS,
                       date_str, gik_n,
                       f"GIK from HF aggregate ({HF_REPO}) + gribberish")
        logger.info(f"GIK saved: {gik_nc}")
    else:
        logger.error("GIK from HF: no data — aborting")
        sys.exit(1)

    # ── Herbie ──
    herbie_nc = out_dir / f"herbie_{date_str}_tp.nc"
    if args.skip_herbie and herbie_nc.exists():
        logger.info("Herbie: loading cached NC")
        with xr.open_dataset(herbie_nc) as ds_h:
            herbie_mean = ds_h["tp_ensemble_mean"].isel(time=0).values
            herbie_std = ds_h["tp_ensemble_standard_deviation"].isel(time=0).values
            herbie_n = int(ds_h.attrs.get("n_ensemble_members", 0))
    else:
        t0 = time.time()
        herbie_mean, herbie_std, herbie_n, h_timing = fetch_herbie_tp(
            date_str, TARGET_STEPS,
            max_members=args.max_members, logger=logger,
        )
        elapsed = time.time() - t0
        logger.info(f"Herbie done: {herbie_n} members in {elapsed:.1f}s")
        result["herbie_total_s"] = round(elapsed, 1)
        result.update({f"herbie_{k}": v for k, v in (h_timing or {}).items()})
        if herbie_mean is not None:
            save_tp_netcdf(herbie_nc, herbie_mean, herbie_std, TARGET_STEPS,
                           date_str, herbie_n, "Herbie GEFS atmos.25")
            logger.info(f"Herbie saved: {herbie_nc}")
        else:
            logger.error("Herbie: no data — aborting")
            sys.exit(1)

    result["gik_members"] = gik_n
    result["herbie_members"] = herbie_n

    # ── Compare ──
    min_steps = min(gik_mean.shape[0], herbie_mean.shape[0])
    eff_idx = min(STEP_INDEX, min_steps - 1)
    for label, g, h in [("mean", gik_mean, herbie_mean),
                        ("spread", gik_std, herbie_std)]:
        m = compute_metrics(g, h, label, step_idx=eff_idx)
        result.update(m)
        corr = m.get(f"{label}_corr", float("nan"))
        rmse = m.get(f"{label}_rmse", float("nan"))
        logger.info(f"  {label}: r={corr:.8f}  RMSE={rmse:.2e}")

    try:
        plot_path = plot_comparison(
            date_str, gik_mean, herbie_mean, gik_std, herbie_std,
            result, out_dir, step_idx=eff_idx,
        )
        logger.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

    stats_file = out_dir / f"stats_{date_str}.json"
    with open(stats_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Stats saved: {stats_file}")

    total_time = time.time() - overall_t0
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"DONE in {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"  Mean   r={result.get('mean_corr', float('nan')):.8f}  "
                f"RMSE={result.get('mean_rmse', float('nan')):.2e}")
    logger.info(f"  Spread r={result.get('spread_corr', float('nan')):.8f}  "
                f"RMSE={result.get('spread_rmse', float('nan')):.2e}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
