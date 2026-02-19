#!/usr/bin/env python3
"""
Convert GEFS ensemble zarr files to cGAN-ready NetCDF format (Version 2).

This script reads individual member zarr files created by run_single_gefs_to_zarr_gribberish.py,
computes ensemble statistics (mean and std), and outputs NetCDF files in the format expected by cGAN.

The cGAN model expects:
- 8 fields in order: cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp
- 4 channels per field: [mean_t1, std_t1, mean_t2, std_t2] for two consecutive timesteps
- Total: 32 channels input
- Dimensions: (time, valid_time, latitude, longitude) with ensemble collapsed to mean/std

Usage:
    python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00

    # With custom forecast hours (pairs of consecutive 6-hour steps)
    python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \
        --forecast_hours 30,36,42,48,54
"""

import os
import sys
import argparse
import warnings
import pickle
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import xarray as xr

warnings.filterwarnings('ignore')

# Variable name mapping: gefs_name -> cgan_name
VARIABLE_MAPPING = {
    'cape': 'cape',     # Convective Available Potential Energy
    'sp': 'pres',       # Surface Pressure -> Pressure
    'mslet': 'msl',     # Mean Sea Level Pressure
    'pwat': 'pwat',     # Precipitable Water
    't2m': 'tmp',       # 2m Temperature -> Temperature
    'u10': 'ugrd',      # 10m U-wind component
    'v10': 'vgrd',      # 10m V-wind component
    'tp': 'apcp',       # Total Precipitation -> Accumulated Precipitation
}

# Reverse mapping for lookup
CGAN_TO_GEFS = {v: k for k, v in VARIABLE_MAPPING.items()}

# cGAN expected field order (8 fields)
CGAN_FIELD_ORDER = ['cape', 'pres', 'pwat', 'tmp', 'ugrd', 'vgrd', 'msl', 'apcp']

# Normalization types for each field
NORMALIZATION_TYPES = {
    'cape': 'max_scale',      # data / max
    'pres': 'standardize',    # (data - mean) / std
    'pwat': 'max_scale',      # data / max
    'tmp': 'standardize',     # (data - mean) / std
    'ugrd': 'symmetric',      # data / max(|min|, max)
    'vgrd': 'symmetric',      # data / max(|min|, max)
    'msl': 'standardize',     # (data - mean) / std
    'apcp': 'log_transform',  # log10(1 + data)
}


def find_member_zarrs(input_dir: Path) -> dict:
    """Find all member zarr files in the input directory."""
    members = {}
    pattern = re.compile(r'gep(\d+).*\.zarr$')

    for item in input_dir.iterdir():
        if item.is_dir() and item.suffix == '.zarr':
            match = pattern.match(item.name)
            if match:
                member_num = int(match.group(1))
                members[member_num] = item

    return dict(sorted(members.items()))


def load_and_combine_members(member_paths: dict, variables: list = None) -> xr.Dataset:
    """Load all member zarr files and combine along 'member' dimension."""
    datasets = []
    member_ids = []

    for member_num, zarr_path in member_paths.items():
        print(f"  Loading member {member_num:02d}: {zarr_path.name}")
        ds = xr.open_zarr(zarr_path)

        if variables:
            available_vars = [v for v in variables if v in ds.data_vars]
            if not available_vars:
                continue
            ds = ds[available_vars]

        datasets.append(ds)
        member_ids.append(member_num)

    if not datasets:
        raise ValueError("No valid datasets found")

    print(f"  Combining {len(datasets)} members...")
    combined = xr.concat(datasets, dim='member')
    combined['member'] = member_ids

    return combined


def convert_step_to_hours(ds: xr.Dataset) -> np.ndarray:
    """Convert step coordinate from timedelta64 to hours."""
    step_values = ds.step.values
    if np.issubdtype(step_values.dtype, np.timedelta64):
        hours = step_values.astype('timedelta64[h]').astype(float)
    else:
        hours = step_values / 3.6e12
    return hours


def get_step_indices(ds: xr.Dataset, target_hours: list) -> list:
    """Get indices for target forecast hours."""
    step_hours = convert_step_to_hours(ds)
    indices = []
    for hour in target_hours:
        idx = np.argmin(np.abs(step_hours - hour))
        if np.abs(step_hours[idx] - hour) < 0.5:
            indices.append(idx)
        else:
            print(f"    Warning: No step found near hour {hour}")
    return indices


def compute_ensemble_stats(data: np.ndarray) -> tuple:
    """
    Compute ensemble mean and std from member data.

    Args:
        data: Array with shape (member, ...) where member is first axis

    Returns:
        (mean, std) arrays with member axis collapsed
    """
    ens_mean = np.nanmean(data, axis=0)
    ens_std = np.nanstd(data, axis=0)
    return ens_mean, ens_std


def normalize_field(data: np.ndarray, field_name: str, norm_stats: dict = None) -> np.ndarray:
    """
    Apply field-specific normalization.

    Args:
        data: Input data array
        field_name: cGAN field name (cape, pres, etc.)
        norm_stats: Optional dict with 'mean', 'std', 'min', 'max' per field

    Returns:
        Normalized data array
    """
    norm_type = NORMALIZATION_TYPES.get(field_name, 'none')

    if norm_type == 'log_transform':
        # Precipitation: log10(1 + data)
        return np.log10(1 + np.maximum(data, 0))

    elif norm_type == 'standardize' and norm_stats:
        # (data - mean) / std
        if field_name in norm_stats:
            mean = norm_stats[field_name].get('mean', 0)
            std = norm_stats[field_name].get('std', 1)
            return (data - mean) / std

    elif norm_type == 'max_scale' and norm_stats:
        # data / max
        if field_name in norm_stats:
            max_val = norm_stats[field_name].get('max', 1)
            return data / max_val

    elif norm_type == 'symmetric' and norm_stats:
        # data / max(|min|, max)
        if field_name in norm_stats:
            min_val = norm_stats[field_name].get('min', -1)
            max_val = norm_stats[field_name].get('max', 1)
            scale = max(abs(min_val), abs(max_val))
            return data / scale

    # No normalization or missing stats - return as is
    return data


def load_normalization_stats(norm_file: str) -> dict:
    """Load normalization statistics from pickle file."""
    if norm_file and os.path.exists(norm_file):
        print(f"  Loading normalization stats from: {norm_file}")
        with open(norm_file, 'rb') as f:
            return pickle.load(f)
    return None


def create_4channel_structure(ds: xr.Dataset, var_name: str,
                              hour_pairs: list, norm_stats: dict = None) -> dict:
    """
    Create 4-channel structure for a field: [mean_t1, std_t1, mean_t2, std_t2].

    Args:
        ds: Combined dataset with dimensions (member, step, latitude, longitude)
        var_name: GEFS variable name (e.g., 't2m')
        hour_pairs: List of (hour1, hour2) tuples for consecutive timesteps
        norm_stats: Optional normalization statistics

    Returns:
        Dict with data arrays for each hour pair
    """
    cgan_name = VARIABLE_MAPPING.get(var_name, var_name)
    step_hours = convert_step_to_hours(ds)

    results = {}

    for pair_idx, (h1, h2) in enumerate(hour_pairs):
        # Find indices for the two timesteps
        idx1 = np.argmin(np.abs(step_hours - h1))
        idx2 = np.argmin(np.abs(step_hours - h2))

        # Get data for both timesteps: shape (member, lat, lon)
        data_t1 = ds[var_name].isel(step=idx1).values  # (member, lat, lon)
        data_t2 = ds[var_name].isel(step=idx2).values

        # Compute ensemble statistics
        mean_t1, std_t1 = compute_ensemble_stats(data_t1)  # (lat, lon)
        mean_t2, std_t2 = compute_ensemble_stats(data_t2)

        # Apply normalization
        mean_t1 = normalize_field(mean_t1, cgan_name, norm_stats)
        std_t1 = normalize_field(std_t1, cgan_name, norm_stats)
        mean_t2 = normalize_field(mean_t2, cgan_name, norm_stats)
        std_t2 = normalize_field(std_t2, cgan_name, norm_stats)

        # Stack as 4 channels: (4, lat, lon)
        four_channels = np.stack([mean_t1, std_t1, mean_t2, std_t2], axis=0)

        results[pair_idx] = {
            'data': four_channels,
            'hours': (h1, h2),
            'cgan_name': cgan_name,
        }

    return results


def process_for_cgan(ds: xr.Dataset, forecast_hours: list,
                     norm_stats: dict = None) -> dict:
    """
    Process combined ensemble dataset for cGAN input format.

    Creates output with:
    - 8 fields in order: cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp
    - 4 channels per field: [mean_t1, std_t1, mean_t2, std_t2]
    - Hour pairs from forecast_hours (e.g., 30-36, 36-42, 42-48, 48-54)

    Returns:
        Dict with processed data for each valid time
    """
    # Create hour pairs from forecast hours
    hour_pairs = [(forecast_hours[i], forecast_hours[i+1])
                  for i in range(len(forecast_hours) - 1)]

    print(f"  Creating {len(hour_pairs)} valid time pairs: {hour_pairs}")

    # Map available GEFS vars to cGAN names
    available_gefs_vars = list(ds.data_vars)
    gefs_to_available = {}
    for gefs_var in available_gefs_vars:
        cgan_name = VARIABLE_MAPPING.get(gefs_var, gefs_var)
        gefs_to_available[cgan_name] = gefs_var

    print(f"  Available fields: {list(gefs_to_available.keys())}")

    # Process each field in cGAN order
    output_data = {i: {} for i in range(len(hour_pairs))}

    for cgan_field in CGAN_FIELD_ORDER:
        if cgan_field not in gefs_to_available:
            print(f"    Warning: {cgan_field} not available, skipping")
            continue

        gefs_var = gefs_to_available[cgan_field]
        print(f"  Processing {gefs_var} -> {cgan_field}")

        # Create 4-channel structure for this field
        field_data = create_4channel_structure(ds, gefs_var, hour_pairs, norm_stats)

        for pair_idx, data_dict in field_data.items():
            output_data[pair_idx][cgan_field] = data_dict['data']

    return output_data, hour_pairs


def save_cgan_netcdf(output_data: dict, hour_pairs: list,
                     lats: np.ndarray, lons: np.ndarray,
                     init_time: datetime, output_dir: Path, year: int):
    """
    Save processed data as NetCDF files for cGAN.

    Two output formats:
    1. Per-field files: {cgan_field}_{year}.nc with dims (time, valid_time, channel, lat, lon)
    2. Combined file: all_fields_{year}.nc with dims (time, valid_time, lat, lon, channel)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Get field list from first valid time
    first_pair_data = output_data[0]
    fields_available = list(first_pair_data.keys())

    n_valid_times = len(hour_pairs)
    n_lat = len(lats)
    n_lon = len(lons)

    # === Save per-field files ===
    for field_name in fields_available:
        # Shape: (1, n_valid_times, 4, lat, lon)
        field_array = np.zeros((1, n_valid_times, 4, n_lat, n_lon), dtype=np.float32)

        for vt_idx in range(n_valid_times):
            field_array[0, vt_idx, :, :, :] = output_data[vt_idx][field_name]

        # Create dataset
        ds_out = xr.Dataset(
            {
                field_name: xr.DataArray(
                    data=field_array,
                    dims=['time', 'valid_time', 'channel', 'latitude', 'longitude'],
                    attrs={
                        'long_name': f'{field_name} ensemble statistics',
                        'channels': 'mean_t1, std_t1, mean_t2, std_t2',
                        'hour_pairs': str(hour_pairs),
                    }
                )
            },
            coords={
                'time': [init_time],
                'valid_time': np.arange(n_valid_times),
                'channel': ['mean_t1', 'std_t1', 'mean_t2', 'std_t2'],
                'latitude': lats,
                'longitude': lons,
            }
        )

        ds_out.attrs.update({
            'title': f'GEFS {field_name} for cGAN inference',
            'description': 'Ensemble mean and std, 4 channels per valid time pair',
            'source': 'GEFS (Global Ensemble Forecast System)',
            'created_by': 'zarr_to_cgan_v2.py',
        })

        output_file = output_dir / f"{field_name}_{year}.nc"
        print(f"  Saving: {output_file}")
        ds_out.to_netcdf(output_file)
        saved_files.append(output_file)

    # === Save combined file with all 32 channels ===
    # Shape: (1, n_valid_times, lat, lon, 32)
    n_fields = len(fields_available)
    combined_array = np.zeros((1, n_valid_times, n_lat, n_lon, n_fields * 4), dtype=np.float32)

    channel_names = []
    for field_name in fields_available:
        for ch in ['mean_t1', 'std_t1', 'mean_t2', 'std_t2']:
            channel_names.append(f'{field_name}_{ch}')

    for vt_idx in range(n_valid_times):
        ch_offset = 0
        for field_name in fields_available:
            # field data shape: (4, lat, lon) -> need (lat, lon, 4)
            field_data = output_data[vt_idx][field_name]  # (4, lat, lon)
            field_data_transposed = np.moveaxis(field_data, 0, -1)  # (lat, lon, 4)
            combined_array[0, vt_idx, :, :, ch_offset:ch_offset+4] = field_data_transposed
            ch_offset += 4

    ds_combined = xr.Dataset(
        {
            'forecast_input': xr.DataArray(
                data=combined_array,
                dims=['time', 'valid_time', 'latitude', 'longitude', 'channel'],
                attrs={
                    'long_name': 'Combined forecast input for cGAN',
                    'n_fields': n_fields,
                    'channels_per_field': 4,
                    'field_order': str(fields_available),
                    'channel_names': str(channel_names),
                }
            )
        },
        coords={
            'time': [init_time],
            'valid_time': np.arange(n_valid_times),
            'latitude': lats,
            'longitude': lons,
            'channel': np.arange(n_fields * 4),
        }
    )

    ds_combined.attrs.update({
        'title': 'GEFS combined input for cGAN inference',
        'description': f'{n_fields} fields x 4 channels = {n_fields*4} total channels',
        'hour_pairs': str(hour_pairs),
        'source': 'GEFS (Global Ensemble Forecast System)',
        'created_by': 'zarr_to_cgan_v2.py',
    })

    combined_file = output_dir / f"forecast_input_{year}.nc"
    print(f"  Saving combined: {combined_file}")
    ds_combined.to_netcdf(combined_file)
    saved_files.append(combined_file)

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert GEFS ensemble zarr to cGAN format with ensemble statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (forecast hours 30, 36, 42, 48, 54 -> 4 valid time pairs)
    python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00

    # Custom forecast hours
    python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \\
        --forecast_hours 30,36,42,48,54

    # With normalization stats file
    python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \\
        --norm_file /path/to/FCSTNorm2018.pkl

Output Structure:
    Per-field files: {field}_{year}.nc
        Dimensions: (time=1, valid_time=N, channel=4, latitude, longitude)
        Channels: [mean_t1, std_t1, mean_t2, std_t2]

    Combined file: forecast_input_{year}.nc
        Dimensions: (time=1, valid_time=N, latitude, longitude, channel=32)
        32 channels = 8 fields x 4 channels each
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing member zarr files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for cGAN NetCDF files')
    parser.add_argument('--forecast_hours', type=str, default='30,36,42,48,54',
                       help='Comma-separated forecast hours (default: 30,36,42,48,54)')
    parser.add_argument('--norm_file', type=str, default=None,
                       help='Path to FCSTNorm2018.pkl normalization file (optional)')
    parser.add_argument('--date', type=str, default=None,
                       help='Initialization date YYYYMMDD (default: inferred from input_dir)')
    parser.add_argument('--run', type=str, default='00',
                       help='Model run hour (default: 00)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Parse forecast hours
    forecast_hours = [int(h.strip()) for h in args.forecast_hours.split(',')]

    # Infer date from directory name if not provided
    if args.date:
        date_str = args.date
    else:
        match = re.match(r'(\d{8})_(\d{2})', input_dir.name)
        if match:
            date_str = match.group(1)
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            print(f"Warning: Could not infer date, using {date_str}")

    year = int(date_str[:4])
    init_time = datetime.strptime(f"{date_str}{args.run}", '%Y%m%d%H')

    print("=" * 70)
    print("GEFS Zarr to cGAN Converter (v2 - Ensemble Statistics)")
    print("=" * 70)
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Init time:        {init_time}")
    print(f"  Forecast hours:   {forecast_hours}")
    print(f"  Hour pairs:       {[(forecast_hours[i], forecast_hours[i+1]) for i in range(len(forecast_hours)-1)]}")
    print()

    # Step 1: Load normalization stats if provided
    norm_stats = load_normalization_stats(args.norm_file)

    # Step 2: Find and load member zarr files
    print("1. Finding member zarr files...")
    member_paths = find_member_zarrs(input_dir)
    print(f"   Found {len(member_paths)} members")

    if not member_paths:
        print("Error: No member zarr files found!")
        sys.exit(1)

    print("\n2. Loading and combining members...")
    combined_ds = load_and_combine_members(member_paths)
    print(f"   Combined shape: {dict(combined_ds.sizes)}")
    print(f"   Variables: {list(combined_ds.data_vars)}")

    # Step 3: Process for cGAN format
    print("\n3. Processing for cGAN format (computing ensemble mean/std)...")
    output_data, hour_pairs = process_for_cgan(combined_ds, forecast_hours, norm_stats)

    # Step 4: Save to NetCDF
    print("\n4. Saving NetCDF files...")
    lats = combined_ds.latitude.values
    lons = combined_ds.longitude.values
    saved_files = save_cgan_netcdf(output_data, hour_pairs, lats, lons,
                                    init_time, output_dir, year)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Output files saved to: {output_dir}")
    for f in saved_files:
        print(f"    - {f.name}")

    # Verification
    print("\n5. Verification...")
    combined_file = output_dir / f"forecast_input_{year}.nc"
    if combined_file.exists():
        ds_check = xr.open_dataset(combined_file)
        print(f"  Combined file: {combined_file.name}")
        print(f"    Dimensions: {dict(ds_check.sizes)}")
        data = ds_check['forecast_input'].values
        print(f"    Shape: {data.shape}")
        print(f"    Value range: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
        print(f"    NaN count: {np.isnan(data).sum()} / {data.size}")
        ds_check.close()

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
