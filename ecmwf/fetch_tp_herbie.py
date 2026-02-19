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
Fetch ECMWF IFS ensemble total-precipitation via Herbie for GIK comparison.

Downloads the same dates / forecast hours used by run_random_date_test.py
(seed=42, Jan 2024 → Jan 2026) and saves one NetCDF per date into
herbie_tp_output/ with variables:
    tp_ensemble_mean, tp_ensemble_standard_deviation

The output layout mirrors the GIK pipeline's random_run_test/ files so
that comparison is straightforward.

Usage:
    uv run fetch_tp_herbie.py                       # all 25 random dates
    uv run fetch_tp_herbie.py --date 20240301       # single date
    uv run fetch_tp_herbie.py --max-members 5       # quick test
    uv run fetch_tp_herbie.py --dry-run              # print dates only
"""

import argparse
import calendar
import logging
import random
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
OUTPUT_DIR = Path("herbie_tp_output")
LOG_FILE = OUTPUT_DIR / "herbie_log.txt"

# Same forecast steps as the GIK streaming pipeline
TARGET_STEPS = [36, 39, 42, 45, 48, 51, 54, 57, 60]

# ICPAC region (must match stream_cgan_variables.py)
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Date range (same as run_random_date_test.py)
START_YEAR, START_MONTH = 2024, 1
END_YEAR, END_MONTH = 2026, 1


# ── Helpers ───────────────────────────────────────────────────────────

def generate_random_dates(seed: int = 42) -> list[date]:
    """Same date generator as run_random_date_test.py."""
    rng = random.Random(seed)
    dates = []
    year, month = START_YEAR, START_MONTH
    while (year, month) <= (END_YEAR, END_MONTH):
        last_day = calendar.monthrange(year, month)[1]
        day = rng.randint(1, last_day)
        dates.append(date(year, month, day))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def setup_logging(out_dir: Path, log_path: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("herbie_tp")
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


def subset_icpac(ds: xr.Dataset) -> xr.Dataset:
    """Subset an xarray Dataset to the ICPAC region."""
    # Herbie returns latitude in descending order (90 → -90)
    lat_name = "latitude"
    lon_name = "longitude"

    # Handle longitude: ECMWF open-data uses 0-360 for some products
    lons = ds[lon_name].values
    if lons.max() > 180:
        # Shift to -180..180
        ds = ds.assign_coords(
            {lon_name: ((ds[lon_name] + 180) % 360) - 180}
        )
        ds = ds.sortby(lon_name)

    ds = ds.sel(
        **{
            lat_name: slice(LAT_MAX, LAT_MIN),
            lon_name: slice(LON_MIN, LON_MAX),
        }
    )
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
    Download TP ensemble data for one date via Herbie, compute ensemble
    mean & std over the ICPAC region, and save as NetCDF.

    Returns the output path, or None on failure.
    """
    from herbie import Herbie

    model_date = datetime.strptime(date_str, "%Y%m%d")
    run_hour = int(run)
    log = logger.info if logger else print
    log_err = logger.error if logger else print
    log_warn = logger.warning if logger else print

    log(f"Fetching TP for {date_str} {run}Z  steps={steps}")

    # Collect per-step ensemble arrays
    n_steps = len(steps)
    step_arrays = {}  # step -> (members, lat, lon)
    icpac_lats = None
    icpac_lons = None

    for fxx in steps:
        log(f"  Step T+{fxx}h ...")
        try:
            H = Herbie(
                f"{date_str} {run}:00",
                model="ifs",
                product="enfo",
                fxx=fxx,
            )

            # Download tp GRIB and open as xarray
            # Herbie may return a list of datasets (perturbed + control)
            ds_list = H.xarray(":tp:", verbose=False)

            # Handle single Dataset vs list
            if isinstance(ds_list, xr.Dataset):
                ds_list = [ds_list]

            # Find the dataset that has the 'number' dimension (ensemble members)
            ens_ds = None
            for ds in ds_list if isinstance(ds_list, list) else [ds_list]:
                if "number" in ds.dims:
                    ens_ds = ds
                    break

            if ens_ds is None:
                # Fall back: use the first dataset
                ens_ds = ds_list[0] if isinstance(ds_list, list) else ds_list

            # Identify the tp variable name (could be 'tp' or 'paramId_XXX')
            tp_var = None
            for v in ens_ds.data_vars:
                if "tp" in v.lower() or ens_ds[v].attrs.get("shortName", "") == "tp":
                    tp_var = v
                    break
            if tp_var is None:
                tp_var = list(ens_ds.data_vars)[0]

            # Subset to ICPAC
            ens_sub = subset_icpac(ens_ds)

            if icpac_lats is None:
                icpac_lats = ens_sub.latitude.values
                icpac_lons = ens_sub.longitude.values

            data = ens_sub[tp_var].values  # (number, lat, lon) or (lat, lon)

            if data.ndim == 2:
                # Single member / control — expand
                data = data[np.newaxis, :, :]

            if max_members and data.shape[0] > max_members:
                data = data[:max_members]

            step_arrays[fxx] = data.astype(np.float32)
            log(f"    OK — {data.shape[0]} members, shape {data.shape[1:]}")

            ens_ds.close()
            for d in (ds_list if isinstance(ds_list, list) else []):
                d.close()

        except Exception as e:
            log_warn(f"    FAILED at T+{fxx}h: {e}")
            continue

    if not step_arrays:
        log_err(f"No data retrieved for {date_str}")
        return None

    # Build ensemble mean & std arrays: (n_steps, lat, lon)
    ordered_steps = [s for s in steps if s in step_arrays]
    n_valid = len(ordered_steps)
    n_lats = len(icpac_lats)
    n_lons = len(icpac_lons)

    mean_arr = np.full((n_valid, n_lats, n_lons), np.nan, dtype=np.float32)
    std_arr = np.full((n_valid, n_lats, n_lons), np.nan, dtype=np.float32)

    for i, fxx in enumerate(ordered_steps):
        data = step_arrays[fxx]
        mean_arr[i] = np.nanmean(data, axis=0)
        std_arr[i] = np.nanstd(data, axis=0)

    # Create xarray Dataset matching GIK output layout
    base_time = model_date + timedelta(hours=run_hour)
    valid_times = [base_time + timedelta(hours=h) for h in ordered_steps]

    n_members = step_arrays[ordered_steps[0]].shape[0]

    ds_out = xr.Dataset(
        {
            "tp_ensemble_mean": xr.DataArray(
                mean_arr[np.newaxis, :, :, :],
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "tp ensemble mean", "units": "varies"},
            ),
            "tp_ensemble_standard_deviation": xr.DataArray(
                std_arr[np.newaxis, :, :, :],
                dims=["time", "valid_time", "latitude", "longitude"],
                attrs={"long_name": "tp ensemble standard deviation", "units": "varies"},
            ),
        },
        coords={
            "time": [base_time],
            "valid_time": valid_times,
            "latitude": icpac_lats,
            "longitude": icpac_lons,
        },
    )
    ds_out.attrs["title"] = "ECMWF IFS Ensemble TP (Herbie)"
    ds_out.attrs["institution"] = "ICPAC"
    ds_out.attrs["source"] = "Herbie → ECMWF IFS enfo"
    ds_out.attrs["model_date"] = model_date.strftime("%Y-%m-%d")
    ds_out.attrs["model_run"] = f"{run_hour:02d}Z"
    ds_out.attrs["forecast_hours"] = str(ordered_steps)
    ds_out.attrs["n_ensemble_members"] = n_members
    ds_out.attrs["history"] = f"Created {datetime.now().isoformat()} via fetch_tp_herbie.py"

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"IFS_{date_str}_{run_hour:02d}Z_herbie_tp.nc"

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_out.data_vars}
    ds_out.to_netcdf(out_file, encoding=encoding)
    ds_out.close()

    size_kb = out_file.stat().st_size / 1024
    log(f"  Saved: {out_file} ({size_kb:.1f} KB, {n_members} members, {n_valid} steps)")
    return out_file


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch ECMWF IFS ensemble TP via Herbie for GIK comparison"
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Single date to fetch (YYYYMMDD)")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour (default: 00)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for date selection (default: 42)")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap ensemble members (default: all)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory (default: herbie_tp_output)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dates and exit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_file = output_dir / "herbie_log.txt"

    # Build date list
    if args.date:
        dates = [datetime.strptime(args.date, "%Y%m%d").date()]
    else:
        dates = generate_random_dates(seed=args.seed)

    if args.dry_run:
        print(f"Dates ({len(dates)}):")
        for d in dates:
            print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%B %Y')})")
        return

    logger = setup_logging(output_dir, log_file)
    logger.info("=" * 70)
    logger.info("Herbie ECMWF IFS Ensemble TP Fetch")
    logger.info("=" * 70)
    logger.info(f"Dates: {len(dates)}  |  Run: {args.run}Z")
    logger.info(f"Steps: {TARGET_STEPS}")
    logger.info(f"Max members: {args.max_members or 'all'}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    overall_start = time.time()
    results = {}

    for idx, d in enumerate(dates, 1):
        ds = d.strftime("%Y%m%d")
        logger.info("=" * 70)
        logger.info(f"[{idx}/{len(dates)}]  {d.strftime('%B %Y')}  —  {ds}")
        logger.info("=" * 70)

        t0 = time.time()
        out = fetch_tp_for_date(
            ds, run=args.run, max_members=args.max_members, logger=logger,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0
        ok = out is not None
        results[ds] = ok
        status = "OK" if ok else "FAIL"
        logger.info(f"  [{status}] {ds} in {elapsed:.1f}s")

    # Summary
    total_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Date':<12} {'Month':<16} {'Status':<10}")
    logger.info("-" * 38)
    passed = 0
    for d in dates:
        ds = d.strftime("%Y%m%d")
        s = "OK" if results[ds] else "FAIL"
        logger.info(f"{ds:<12} {d.strftime('%b %Y'):<16} {s:<10}")
        passed += results[ds]
    logger.info("-" * 38)
    logger.info(f"Passed: {passed}/{len(dates)}")
    logger.info(f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")

    nc_files = sorted(output_dir.glob("*.nc"))
    if nc_files:
        logger.info(f"\nNetCDF files in {output_dir}/:")
        for f in nc_files:
            logger.info(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    logger.info(f"\nFull log: {log_file}")


if __name__ == "__main__":
    main()
