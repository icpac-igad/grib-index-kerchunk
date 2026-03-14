#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "herbie-data",
#     "xarray",
#     "numpy",
#     "netcdf4",
#     "cfgrib",
#     "eccodes",
# ]
# ///
"""
Fetch GEFS ensemble total-precipitation via Herbie for GIK comparison.

Downloads TP for selected dates and forecast hours, computes ensemble
mean & std over the East Africa / ICPAC region, and saves one NetCDF
per date into herbie_tp_gefs_output/.

The output layout mirrors the GIK pipeline's parquet-to-zarr streaming
output so that comparison is straightforward.

Usage:
    uv run fetch_tp_herbie_gefs.py                        # default 5 dates
    uv run fetch_tp_herbie_gefs.py --date 20250106        # single date
    uv run fetch_tp_herbie_gefs.py --max-members 5        # quick test
    uv run fetch_tp_herbie_gefs.py --dry-run              # print dates only
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
OUTPUT_DIR = Path("herbie_tp_gefs_output")
LOG_FILE = OUTPUT_DIR / "herbie_gefs_log.txt"

# Forecast steps: GEFS uses 3-hour intervals, 0-240h (81 timesteps)
# Use a representative subset for validation
TARGET_STEPS = [0, 3, 6, 12, 24, 48, 72, 120, 168, 240]

# East Africa / ICPAC region
LAT_MIN, LAT_MAX = -12, 15
LON_MIN, LON_MAX = 25, 52

# Default dates for validation (spanning different seasons)
DEFAULT_DATES = ["20250106", "20250301", "20250601", "20250901", "20241112"]


# ── Helpers ───────────────────────────────────────────────────────────

def setup_logging(out_dir: Path, log_path: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("herbie_gefs_tp")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


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


# ── Core fetch routine ────────────────────────────────────────────────

def fetch_tp_for_date(
    date_str: str,
    run: str = "00",
    steps: List[int] = TARGET_STEPS,
    max_members: Optional[int] = None,
    logger: logging.Logger = None,
    output_dir: Path = OUTPUT_DIR,
) -> Optional[Path]:
    """
    Download GEFS TP ensemble data for one date via Herbie, compute
    ensemble mean & std over East Africa, and save as NetCDF.

    Returns the output path, or None on failure.
    """
    from herbie import Herbie

    model_date = datetime.strptime(date_str, "%Y%m%d")
    run_hour = int(run)
    log = logger.info if logger else print
    log_err = logger.error if logger else print
    log_warn = logger.warning if logger else print

    log(f"Fetching GEFS TP for {date_str} {run}Z  steps={steps}")

    # GEFS has 30 ensemble members (gep01-gep30)
    # Herbie uses member=1..30 for GEFS
    n_members = max_members or 30

    step_arrays = {}  # step -> (members, lat, lon)
    ea_lats = None
    ea_lons = None

    for fxx in steps:
        log(f"  Step T+{fxx:03d}h ...")
        member_data = []

        for mem_num in range(1, n_members + 1):
            try:
                H = Herbie(
                    f"{date_str} {run}:00",
                    model="gefs",
                    product="atmos.25",
                    member=mem_num,
                    fxx=fxx,
                )

                # GEFS TP variable: APCP (accumulated precipitation)
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
                # Squeeze extra dims if present
                while data.ndim > 2:
                    data = data[0]

                member_data.append(data.astype(np.float32))
                ds.close()

            except Exception as e:
                if mem_num <= 3:
                    log_warn(f"    member {mem_num} FAILED: {e}")
                continue

        if member_data:
            arr = np.stack(member_data, axis=0)  # (members, lat, lon)
            step_arrays[fxx] = arr
            log(f"    OK - {arr.shape[0]} members, shape {arr.shape[1:]}")
        else:
            log_warn(f"    FAILED - no members retrieved for T+{fxx}h")

    if not step_arrays:
        log_err(f"No data retrieved for {date_str}")
        return None

    # Build ensemble mean & std: (n_steps, lat, lon)
    ordered_steps = [s for s in steps if s in step_arrays]
    n_valid = len(ordered_steps)
    n_lats = len(ea_lats)
    n_lons = len(ea_lons)

    mean_arr = np.full((n_valid, n_lats, n_lons), np.nan, dtype=np.float32)
    std_arr = np.full((n_valid, n_lats, n_lons), np.nan, dtype=np.float32)

    for i, fxx in enumerate(ordered_steps):
        data = step_arrays[fxx]
        mean_arr[i] = np.nanmean(data, axis=0)
        std_arr[i] = np.nanstd(data, axis=0)

    # Create xarray Dataset
    base_time = model_date + timedelta(hours=run_hour)
    valid_times = [base_time + timedelta(hours=h) for h in ordered_steps]
    n_members_actual = step_arrays[ordered_steps[0]].shape[0]

    ds_out = xr.Dataset(
        {
            "tp_ensemble_mean": xr.DataArray(
                mean_arr[np.newaxis, :, :, :],
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "APCP ensemble mean", "units": "kg m-2"},
            ),
            "tp_ensemble_standard_deviation": xr.DataArray(
                std_arr[np.newaxis, :, :, :],
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "APCP ensemble standard deviation", "units": "kg m-2"},
            ),
        },
        coords={
            "time": [base_time],
            "valid_time": valid_times,
            "latitude": ea_lats,
            "longitude": ea_lons,
        },
    )
    ds_out.attrs["title"] = "GEFS Ensemble APCP (Herbie)"
    ds_out.attrs["institution"] = "ICPAC"
    ds_out.attrs["source"] = "Herbie -> NOAA GEFS"
    ds_out.attrs["model_date"] = model_date.strftime("%Y-%m-%d")
    ds_out.attrs["model_run"] = f"{run_hour:02d}Z"
    ds_out.attrs["forecast_hours"] = str(ordered_steps)
    ds_out.attrs["n_ensemble_members"] = n_members_actual
    ds_out.attrs["history"] = f"Created {datetime.now().isoformat()} via fetch_tp_herbie_gefs.py"

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"GEFS_{date_str}_{run_hour:02d}Z_herbie_tp.nc"

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_out.data_vars}
    ds_out.to_netcdf(out_file, encoding=encoding)
    ds_out.close()

    size_kb = out_file.stat().st_size / 1024
    log(f"  Saved: {out_file} ({size_kb:.1f} KB, {n_members_actual} members, {n_valid} steps)")
    return out_file


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch GEFS ensemble TP via Herbie for GIK comparison"
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Single date to fetch (YYYYMMDD)")
    parser.add_argument("--dates", type=str, default=None,
                        help="Comma-separated dates (YYYYMMDD,...)")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour (default: 00)")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap ensemble members (default: 30)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dates only, don't fetch")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    logger = setup_logging(out_dir, out_dir / "herbie_gefs_log.txt")

    # Determine dates
    if args.date:
        dates = [args.date]
    elif args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
    else:
        dates = DEFAULT_DATES

    logger.info(f"Dates to process: {dates}")
    logger.info(f"Run: {args.run}Z, Max members: {args.max_members or 30}")

    if args.dry_run:
        for d in dates:
            logger.info(f"  [dry-run] {d}")
        return

    # Fetch
    t0 = time.time()
    results = []
    for date_str in dates:
        result = fetch_tp_for_date(
            date_str,
            run=args.run,
            max_members=args.max_members,
            logger=logger,
            output_dir=out_dir,
        )
        results.append((date_str, result))

    elapsed = time.time() - t0
    ok = sum(1 for _, r in results if r is not None)
    logger.info(f"\nDone: {ok}/{len(dates)} dates in {elapsed:.1f}s")
    for d, r in results:
        status = f"OK -> {r}" if r else "FAILED"
        logger.info(f"  {d}: {status}")


if __name__ == "__main__":
    main()
