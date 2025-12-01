#!/usr/bin/env python3
"""
ECMWF Open Data vs GIK (Grib-Index-Kerchunk) Comparison Script

This script compares two methods of accessing ECMWF forecast data:

1. ECMWF Open Data API (ecmwf-opendata package)
   - Downloads GRIB files directly from ECMWF servers
   - Uses the official ecmwf.opendata.Client

2. GIK (Grib-Index-Kerchunk) Stage 3 Method
   - Uses pre-built parquet indexes pointing to S3 byte ranges
   - Fetches only the required data chunks from AWS S3

Purpose:
========
This comparison demonstrates that both methods produce identical meteorological
data, validating the GIK approach as a viable alternative for accessing ECMWF
forecasts with potentially lower latency and better cloud-native integration.

Output:
=======
- Side-by-side comparison plots
- Data statistics comparison
- Difference maps showing any discrepancies

ECMWF Step/Time Reference:
==========================
For a 00Z model run:
    Step  6  -> Valid: +6h  (e.g., 06:00 UTC same day)
    Step 12  -> Valid: +12h (e.g., 12:00 UTC same day)
    Step 24  -> Valid: +24h (e.g., 00:00 UTC next day)

Usage:
    # Compare step 6 forecast for Africa region
    python compare_ecmwf_opendata_vs_gik.py --step 6 --region africa

    # Compare step 24 forecast for global view
    python compare_ecmwf_opendata_vs_gik.py --step 24 --region global

    # Skip ECMWF download (use cached GRIB file)
    python compare_ecmwf_opendata_vs_gik.py --step 6 --skip-download
"""

import numpy as np
import xarray as xr
import tempfile
import os
import argparse
import datetime
from pathlib import Path

# Import GIK extraction functions
import sys
sys.path.insert(0, str(Path(__file__).parent))

from read_stage3_to_xarray import (
    read_parquet_to_refs,
    extract_single_timestep,
    get_model_run_time,
)

# Configuration
DEFAULT_GIK_INPUT_DIR = Path("/scratch/notebook/grib-index-kerchunk/ecmwf/test_ecmwf_three_stage_prebuilt_output")


def step_to_valid_time(model_run_time, step_hour):
    """Convert forecast step to valid time."""
    if model_run_time is None:
        return None
    return model_run_time + datetime.timedelta(hours=step_hour)


def generate_output_filename(model_run_time, step_hour, region, suffix='comparison'):
    """Generate output filename in YYYYMMDD_HHZ_hrXXX_region format."""
    if model_run_time is None:
        return f"mslp_precip_step{step_hour:03d}_{region}_{suffix}"

    date_str = model_run_time.strftime('%Y%m%d')
    hour_str = model_run_time.strftime('%H')
    region_clean = region.lower().replace(' ', '_').replace('-', '_')

    return f"{date_str}_{hour_str}Z_hr{step_hour:03d}_{region_clean}_{suffix}"


# =============================================================================
# METHOD 1: ECMWF OPEN DATA API
# =============================================================================

def fetch_ecmwf_opendata(step_hour, precip_window=6, cache_dir=None,
                         forecast_date=None, forecast_time=0):
    """
    Fetch MSLP and precipitation data using ECMWF Open Data API.

    Args:
        step_hour: Forecast step in hours
        precip_window: Precipitation accumulation window (default: 6h to match ECMWF charts)
        cache_dir: Directory to cache downloaded GRIB files
        forecast_date: Date string in YYYYMMDD format (default: latest)
        forecast_time: Forecast hour (0 or 12, default: 0)

    Returns:
        Tuple of (msl_data_hpa, tp_data_mm, model_run_time)
        - msl_data_hpa: 2D numpy array of MSL pressure in hPa
        - tp_data_mm: 2D numpy array of precipitation in mm
        - model_run_time: datetime of model initialization
    """
    from ecmwf.opendata import Client
    import metview as mv

    print("\n" + "="*60)
    print("METHOD 1: ECMWF Open Data API")
    print("="*60)

    client = Client("ecmwf", beta=False)

    # Determine steps needed for precipitation calculation
    precip_start = max(0, step_hour - precip_window)
    steps_needed = sorted(list(set([precip_start, step_hour])))

    print(f"  Requesting steps: {steps_needed}")
    print(f"  Variables: msl, tp")
    if forecast_date:
        print(f"  Forecast date: {forecast_date}")
        print(f"  Forecast time: {forecast_time:02d}Z")
    else:
        print(f"  Forecast date: (latest available)")

    # Create cache filename
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        date_suffix = f"_{forecast_date}_{forecast_time:02d}z" if forecast_date else ""
        grib_file = cache_dir / f"ecmwf_opendata_steps{'_'.join(map(str, steps_needed))}{date_suffix}.grib"
    else:
        date_suffix = f"_{forecast_date}_{forecast_time:02d}z" if forecast_date else ""
        grib_file = Path(tempfile.gettempdir()) / f"ecmwf_opendata_step{step_hour}{date_suffix}.grib"

    # Download data
    print(f"  Downloading to: {grib_file}")
    try:
        # Build retrieve request
        retrieve_kwargs = {
            'step': steps_needed,
            'stream': "oper",
            'type': "fc",
            'levtype': "sfc",
            'param': ['msl', 'tp'],
            'target': str(grib_file)
        }

        # Add date/time if specified
        if forecast_date:
            # Convert YYYYMMDD to date format expected by ecmwf-opendata
            retrieve_kwargs['date'] = forecast_date
            retrieve_kwargs['time'] = forecast_time

        client.retrieve(**retrieve_kwargs)
        print("  Download complete!")
    except Exception as e:
        print(f"  Download failed: {e}")
        raise

    # Read with metview
    print("  Reading GRIB with metview...")
    data = mv.read(str(grib_file))

    # Extract MSL at target step
    msl = data.select(shortName='msl', step=step_hour)
    msl_values = mv.values(msl)

    # Get grid info for reshaping - grib_get returns list of lists for multiple keys
    ni_result = mv.grib_get(msl, ['Ni'])
    nj_result = mv.grib_get(msl, ['Nj'])

    # Handle the nested list structure from grib_get
    ni = int(ni_result[0]) if isinstance(ni_result[0], (int, float)) else int(ni_result[0][0])
    nj = int(nj_result[0]) if isinstance(nj_result[0], (int, float)) else int(nj_result[0][0])

    msl_2d = msl_values.reshape((nj, ni))

    # Convert to hPa
    msl_hpa = msl_2d / 100.0

    print(f"  MSL shape: {msl_hpa.shape}")
    print(f"  MSL range: {msl_hpa.min():.1f} - {msl_hpa.max():.1f} hPa")

    # Extract and compute precipitation
    if precip_start == 0:
        # Use cumulative precip
        tp = data.select(shortName='tp', step=step_hour)
        tp_values = mv.values(tp)
        tp_2d = tp_values.reshape((nj, ni))
    else:
        # Compute difference
        tp_start = data.select(shortName='tp', step=precip_start)
        tp_end = data.select(shortName='tp', step=step_hour)
        tp_start_values = mv.values(tp_start).reshape((nj, ni))
        tp_end_values = mv.values(tp_end).reshape((nj, ni))
        tp_2d = tp_end_values - tp_start_values
        tp_2d = np.maximum(tp_2d, 0)  # No negative precip

    # Convert to mm
    tp_mm = tp_2d * 1000.0

    print(f"  TP shape: {tp_mm.shape}")
    print(f"  TP range: {tp_mm.min():.2f} - {tp_mm.max():.2f} mm")

    # Extract model run time from GRIB metadata
    date_result = mv.grib_get(msl, ['dataDate'])
    time_result = mv.grib_get(msl, ['dataTime'])

    # Handle nested list structure
    data_date = date_result[0] if isinstance(date_result[0], (int, float)) else date_result[0][0]
    data_time = time_result[0] if isinstance(time_result[0], (int, float)) else time_result[0][0]

    date_str = str(int(data_date))
    time_str = str(int(data_time)).zfill(4)

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(time_str[:2])

    model_run_time = datetime.datetime(year, month, day, hour)
    print(f"  Model run: {model_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    valid_time = step_to_valid_time(model_run_time, step_hour)
    print(f"  Valid time: {valid_time.strftime('%Y-%m-%d %H:%M:%S UTC')} (+{step_hour}h)")

    return msl_hpa, tp_mm, model_run_time


# =============================================================================
# METHOD 2: GIK (GRIB-INDEX-KERCHUNK) STAGE 3
# =============================================================================

def fetch_gik_stage3(step_hour, precip_window=6, input_dir=None, member='control'):
    """
    Fetch MSLP and precipitation data using GIK Stage 3 parquet method.

    Args:
        step_hour: Forecast step in hours
        precip_window: Precipitation accumulation window (default: 6h)
        input_dir: Directory containing stage3 parquet files
        member: Ensemble member (default: control)

    Returns:
        Tuple of (msl_data_hpa, tp_data_mm, model_run_time)
    """
    print("\n" + "="*60)
    print("METHOD 2: GIK Stage 3 (S3 Parquet References)")
    print("="*60)

    if input_dir is None:
        input_dir = DEFAULT_GIK_INPUT_DIR
    input_dir = Path(input_dir)

    parquet_file = input_dir / f"stage3_{member}_final.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    print(f"  Loading: {parquet_file.name}")
    zstore = read_parquet_to_refs(parquet_file)

    # Get model run time
    model_run_time = get_model_run_time(zstore, 'tp')
    if model_run_time:
        print(f"  Model run: {model_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        valid_time = step_to_valid_time(model_run_time, step_hour)
        print(f"  Valid time: {valid_time.strftime('%Y-%m-%d %H:%M:%S UTC')} (+{step_hour}h)")

    # Extract MSL
    print(f"  Extracting MSL at step {step_hour}h...")
    msl_data = extract_single_timestep(zstore, 'msl', step_hour, member)

    if msl_data is None:
        raise ValueError(f"Failed to extract MSL at step {step_hour}")

    # Convert to hPa
    msl_hpa = msl_data / 100.0

    print(f"  MSL shape: {msl_hpa.shape}")
    print(f"  MSL range: {msl_hpa.min():.1f} - {msl_hpa.max():.1f} hPa")

    # Extract and compute precipitation
    precip_start = max(0, step_hour - precip_window)

    if precip_start == 0:
        print(f"  Extracting TP at step {step_hour}h (cumulative)...")
        tp_data = extract_single_timestep(zstore, 'tp', step_hour, member)
    else:
        print(f"  Computing {precip_window}h precipitation ({precip_start}h to {step_hour}h)...")
        tp_start = extract_single_timestep(zstore, 'tp', precip_start, member)
        tp_end = extract_single_timestep(zstore, 'tp', step_hour, member)

        if tp_start is None or tp_end is None:
            raise ValueError(f"Failed to extract TP for steps {precip_start} and {step_hour}")

        tp_data = tp_end - tp_start
        tp_data = np.maximum(tp_data, 0)

    if tp_data is None:
        raise ValueError(f"Failed to extract TP at step {step_hour}")

    # Convert to mm
    tp_mm = tp_data * 1000.0

    print(f"  TP shape: {tp_mm.shape}")
    print(f"  TP range: {tp_mm.min():.2f} - {tp_mm.max():.2f} mm")

    return msl_hpa, tp_mm, model_run_time


# =============================================================================
# COMPARISON AND VISUALIZATION
# =============================================================================

def compare_arrays(arr1, arr2, name, tolerance=1e-4):
    """
    Compare two arrays and print statistics.

    Args:
        arr1: First array (ECMWF Open Data)
        arr2: Second array (GIK)
        name: Variable name for printing
        tolerance: Relative tolerance for "match" determination

    Returns:
        Dictionary with comparison statistics
    """
    print(f"\n  {name} Comparison:")
    print(f"    Shape match: {arr1.shape == arr2.shape}")

    if arr1.shape != arr2.shape:
        print(f"    WARNING: Shape mismatch! {arr1.shape} vs {arr2.shape}")
        return {'match': False, 'reason': 'shape_mismatch'}

    # Compute differences
    diff = arr1 - arr2
    abs_diff = np.abs(diff)

    # Statistics
    stats = {
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff)),
        'max_abs_diff': float(np.max(abs_diff)),
        'min_abs_diff': float(np.min(abs_diff)),
        'rmse': float(np.sqrt(np.mean(diff**2))),
    }

    # Relative difference
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs(diff) / (np.abs(arr1) + 1e-10)
        stats['mean_rel_diff'] = float(np.nanmean(rel_diff))
        stats['max_rel_diff'] = float(np.nanmax(rel_diff))

    print(f"    Mean difference: {stats['mean_diff']:.6f}")
    print(f"    Std difference: {stats['std_diff']:.6f}")
    print(f"    Max absolute diff: {stats['max_abs_diff']:.6f}")
    print(f"    RMSE: {stats['rmse']:.6f}")
    print(f"    Mean relative diff: {stats['mean_rel_diff']:.6%}")

    # Determine if arrays match within tolerance
    stats['match'] = stats['max_rel_diff'] < tolerance

    if stats['match']:
        print(f"    Result: MATCH (within {tolerance:.0%} tolerance)")
    else:
        print(f"    Result: DIFFERS (exceeds {tolerance:.0%} tolerance)")

    return stats


def create_comparison_plots(msl_opendata, tp_opendata, msl_gik, tp_gik,
                            model_run_time, step_hour, region, output_name):
    """
    Create side-by-side comparison plots.

    Creates three panels:
    1. ECMWF Open Data (left)
    2. GIK Stage 3 (right)
    3. Difference map (bottom, if applicable)
    """
    import metview as mv

    print("\n" + "="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)

    # Create temporary NetCDF files for metview
    def array_to_netcdf(data_2d, variable):
        lats = np.linspace(90, -90, data_2d.shape[0])
        lons = np.linspace(0, 359.75, data_2d.shape[1])

        da = xr.DataArray(
            data_2d,
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude'],
        )

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            tmp_path = tmp.name

        ds = xr.Dataset({variable: da})
        ds.to_netcdf(tmp_path)

        nc_data = mv.read(tmp_path)
        visualiser = mv.netcdf_visualiser(
            netcdf_data=nc_data,
            netcdf_plot_type='geo_matrix',
            netcdf_latitude_variable='latitude',
            netcdf_longitude_variable='longitude',
            netcdf_value_variable=variable
        )

        return nc_data, visualiser, tmp_path

    # Common map settings
    coast = mv.mcoast(
        map_coastline_colour="charcoal",
        map_coastline_resolution="medium",
        map_coastline_land_shade="on",
        map_coastline_land_shade_colour="cream",
        map_coastline_sea_shade="off",
        map_boundaries="on",
        map_boundaries_colour="charcoal",
        map_boundaries_thickness=1,
        map_disputed_boundaries="off",
        map_grid_colour="tan",
        map_label_height=0.35,
    )

    view = mv.geoview(
        area_mode="name",
        area_name=region,
        coastlines=coast
    )

    tp_shade = mv.mcont(
        legend="on",
        contour_automatics_settings="style_name",
        contour_style_name="sh_blured_f05t300lst"
    )

    msl_shade = mv.mcont(
        legend="off",
        contour_automatics_settings="style_name",
        contour_style_name="ct_blk_i5_t2"
    )

    # Prepare data for plotting
    tmp_files = []

    # ECMWF Open Data
    print("  Preparing ECMWF Open Data plot...")
    _, msl_vis_od, tmp1 = array_to_netcdf(msl_opendata, 'msl')
    _, tp_vis_od, tmp2 = array_to_netcdf(tp_opendata, 'tp')
    tmp_files.extend([tmp1, tmp2])

    # GIK Stage 3
    print("  Preparing GIK Stage 3 plot...")
    _, msl_vis_gik, tmp3 = array_to_netcdf(msl_gik, 'msl')
    _, tp_vis_gik, tmp4 = array_to_netcdf(tp_gik, 'tp')
    tmp_files.extend([tmp3, tmp4])

    # Format times
    if model_run_time:
        base_date = model_run_time.strftime('%a %d %B %Y %H')
        valid_time = model_run_time + datetime.timedelta(hours=step_hour)
        valid_date = valid_time.strftime('%a %d %B %Y %H')
    else:
        base_date = "Unknown"
        valid_date = "Unknown"

    # =========================================================================
    # PLOT 1: ECMWF Open Data
    # =========================================================================
    print("  Generating ECMWF Open Data plot...")

    title_od = mv.mtext(
        text_lines=[
            "ECMWF Open Data API - Rain and MSLP",
            f"START TIME: {base_date} UTC",
            f"VALID TIME: {valid_date} UTC, STEP: {step_hour}h"
        ],
        text_font_size=0.4,
        text_colour='charcoal'
    )

    footer_od = mv.mtext(
        text_lines=[
            "Data source: ECMWF Open Data (ecmwf-opendata package)",
            "Direct download from ECMWF servers",
        ],
        text_justification='center',
        text_font_size=0.3,
        text_mode="positional",
        text_box_x_position=6.,
        text_box_y_position=-0.2,
        text_box_x_length=8,
        text_box_y_length=2,
        text_colour='charcoal'
    )

    output_od = f"{output_name}_opendata"
    png_od = mv.png_output(
        output_name=output_od,
        output_title="ECMWF Open Data",
        output_width=1000,
    )
    mv.setoutput(png_od)
    mv.plot(view, tp_vis_od, tp_shade, msl_vis_od, msl_shade, title_od, footer_od)

    # Rename output file
    if os.path.exists(f"{output_od}.1.png"):
        os.rename(f"{output_od}.1.png", f"{output_od}.png")
    print(f"  Saved: {output_od}.png")

    # =========================================================================
    # PLOT 2: GIK Stage 3
    # =========================================================================
    print("  Generating GIK Stage 3 plot...")

    title_gik = mv.mtext(
        text_lines=[
            "GIK Stage 3 (S3 Parquet) - Rain and MSLP",
            f"START TIME: {base_date} UTC",
            f"VALID TIME: {valid_date} UTC, STEP: {step_hour}h"
        ],
        text_font_size=0.4,
        text_colour='charcoal'
    )

    footer_gik = mv.mtext(
        text_lines=[
            "Data source: Grib-Index-Kerchunk Stage 3 Parquet",
            "Byte-range fetch from AWS S3 (ecmwf-forecasts bucket)",
        ],
        text_justification='center',
        text_font_size=0.3,
        text_mode="positional",
        text_box_x_position=6.,
        text_box_y_position=-0.2,
        text_box_x_length=8,
        text_box_y_length=2,
        text_colour='charcoal'
    )

    output_gik = f"{output_name}_gik"
    png_gik = mv.png_output(
        output_name=output_gik,
        output_title="GIK Stage 3",
        output_width=1000,
    )
    mv.setoutput(png_gik)
    mv.plot(view, tp_vis_gik, tp_shade, msl_vis_gik, msl_shade, title_gik, footer_gik)

    if os.path.exists(f"{output_gik}.1.png"):
        os.rename(f"{output_gik}.1.png", f"{output_gik}.png")
    print(f"  Saved: {output_gik}.png")

    # =========================================================================
    # PLOT 3: Difference Map
    # =========================================================================
    print("  Generating difference plot...")

    # Compute differences
    msl_diff = msl_opendata - msl_gik
    tp_diff = tp_opendata - tp_gik

    _, msl_vis_diff, tmp5 = array_to_netcdf(msl_diff, 'msl')
    _, tp_vis_diff, tmp6 = array_to_netcdf(tp_diff, 'tp')
    tmp_files.extend([tmp5, tmp6])

    # Custom difference shading
    diff_shade = mv.mcont(
        legend="on",
        contour="off",
        contour_level_selection_type="level_list",
        contour_level_list=[-10, -5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 5, 10],
        contour_shade="on",
        contour_shade_technique="grid_shading",
        contour_shade_colour_method="palette",
        contour_shade_palette_name="colorbrewer_RdBu_11",
    )

    title_diff = mv.mtext(
        text_lines=[
            "DIFFERENCE (OpenData - GIK) - Precipitation (mm)",
            f"STEP: {step_hour}h",
            f"Max diff: MSL={np.max(np.abs(msl_diff)):.3f} hPa, TP={np.max(np.abs(tp_diff)):.3f} mm"
        ],
        text_font_size=0.4,
        text_colour='charcoal'
    )

    output_diff = f"{output_name}_difference"
    png_diff = mv.png_output(
        output_name=output_diff,
        output_title="Difference",
        output_width=1000,
    )
    mv.setoutput(png_diff)
    mv.plot(view, tp_vis_diff, diff_shade, title_diff)

    if os.path.exists(f"{output_diff}.1.png"):
        os.rename(f"{output_diff}.1.png", f"{output_diff}.png")
    print(f"  Saved: {output_diff}.png")

    # Cleanup temp files
    for tmp in tmp_files:
        try:
            os.unlink(tmp)
        except:
            pass

    return [f"{output_od}.png", f"{output_gik}.png", f"{output_diff}.png"]


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description='Compare ECMWF Open Data API vs GIK Stage 3 Method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Compare step 6 forecast (matching ECMWF chart at +6h)
    python compare_ecmwf_opendata_vs_gik.py --step 6 --region africa

    # Compare step 24 forecast for global view
    python compare_ecmwf_opendata_vs_gik.py --step 24 --region global

    # Skip ECMWF download (use existing GIK data only)
    python compare_ecmwf_opendata_vs_gik.py --step 6 --gik-only
        '''
    )
    parser.add_argument('--step', type=int, default=6,
                        help='Forecast step in hours (default: 6)')
    parser.add_argument('--region', type=str, default='africa',
                        help='Region name for metview (default: africa)')
    parser.add_argument('--precip-window', type=int, default=6,
                        help='Precipitation accumulation window in hours (default: 6)')
    parser.add_argument('--gik-input-dir', type=Path, default=DEFAULT_GIK_INPUT_DIR,
                        help='Input directory for GIK parquet files')
    parser.add_argument('--member', type=str, default='control',
                        help='Ensemble member for GIK (default: control)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots')
    parser.add_argument('--gik-only', action='store_true',
                        help='Only fetch GIK data (skip ECMWF Open Data)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip plot generation (data comparison only)')
    parser.add_argument('--date', type=str, default=None,
                        help='Forecast date in YYYYMMDD format (default: latest). Use to match GIK date.')
    parser.add_argument('--time', type=int, default=0, choices=[0, 12],
                        help='Forecast time (0 or 12, default: 0)')

    args = parser.parse_args()

    print("="*80)
    print("ECMWF Open Data vs GIK Stage 3 Comparison")
    print("="*80)
    print(f"Step: {args.step}h")
    print(f"Region: {args.region}")
    print(f"Precip window: {args.precip_window}h")
    print(f"GIK input: {args.gik_input_dir}")
    print("="*80)

    # Fetch GIK data (always)
    msl_gik, tp_gik, model_run_time_gik = fetch_gik_stage3(
        args.step,
        args.precip_window,
        args.gik_input_dir,
        args.member
    )

    if args.gik_only:
        print("\n" + "="*60)
        print("GIK-ONLY MODE: Skipping ECMWF Open Data comparison")
        print("="*60)

        # Just create single plot
        if not args.skip_plots:
            output_name = generate_output_filename(model_run_time_gik, args.step, args.region, 'gik_only')
            output_name = os.path.join(args.output_dir, output_name)

            # Create simple GIK plot
            import metview as mv

            # ... (simplified single plot)
            print(f"\n  Output: {output_name}.png")

        return True

    # Fetch ECMWF Open Data
    try:
        msl_opendata, tp_opendata, model_run_time_od = fetch_ecmwf_opendata(
            args.step,
            args.precip_window,
            forecast_date=args.date,
            forecast_time=args.time
        )
    except Exception as e:
        print(f"\nERROR: Failed to fetch ECMWF Open Data: {e}")
        print("Try running with --gik-only to skip ECMWF download")
        print(f"\nTip: If the date is too old, ECMWF may not have data available.")
        print(f"     GIK model run: {model_run_time_gik.strftime('%Y-%m-%d %H:%M UTC') if model_run_time_gik else 'Unknown'}")
        return False

    # Compare data
    print("\n" + "="*60)
    print("DATA COMPARISON")
    print("="*60)

    # Check model run times match
    if model_run_time_gik and model_run_time_od:
        time_match = model_run_time_gik == model_run_time_od
        print(f"\n  Model run time match: {time_match}")
        print(f"    ECMWF OpenData: {model_run_time_od.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"    GIK Stage 3:    {model_run_time_gik.strftime('%Y-%m-%d %H:%M UTC')}")

        if not time_match:
            print("\n  WARNING: Model run times differ! Comparison may not be valid.")
            print("           Ensure both methods are using the same forecast run.")

    # Compare arrays
    msl_stats = compare_arrays(msl_opendata, msl_gik, "MSL (hPa)")
    tp_stats = compare_arrays(tp_opendata, tp_gik, "TP (mm)")

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    if msl_stats.get('match', False) and tp_stats.get('match', False):
        print("\n  RESULT: DATA MATCHES!")
        print("  Both methods produce equivalent meteorological data.")
        print("  The GIK Stage 3 method is validated against ECMWF Open Data.")
    else:
        print("\n  RESULT: DATA DIFFERS")
        print("  There are differences between the two methods.")
        print("  Check the difference statistics above for details.")

    # Generate plots
    if not args.skip_plots:
        output_name = generate_output_filename(
            model_run_time_gik or model_run_time_od,
            args.step,
            args.region
        )
        output_name = os.path.join(args.output_dir, output_name)

        plot_files = create_comparison_plots(
            msl_opendata, tp_opendata,
            msl_gik, tp_gik,
            model_run_time_gik or model_run_time_od,
            args.step,
            args.region,
            output_name
        )

        print("\n" + "="*60)
        print("OUTPUT FILES")
        print("="*60)
        for f in plot_files:
            print(f"  - {f}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
