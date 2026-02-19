#!/usr/bin/env python
"""
Script to convert GEFS zarr files to NetCDF format for forecast input.

This script reads zarr files from the GEFS forecast data and converts them
to NetCDF format, which can then be used as input for the forecast_gfs script.

Usage:
    python convert_zarr_to_netcdf.py --config ../config/forecast_gfs.yaml

Or with custom parameters:
    python convert_zarr_to_netcdf.py --input_folder /path/to/zarr --output_folder /path/to/netcdf \
                                     --dates 2024-04-20 --year 2024
"""

import os
import sys
import argparse
import yaml
import xarray as xr
import numpy as np
from datetime import datetime

sys.path.insert(1, "../")
from data.data_gefs import all_fcst_fields


def convert_zarr_to_netcdf(input_folder, output_folder, dates, year, fields=None):
    """
    Convert zarr files to NetCDF format.

    Parameters:
        input_folder (str): Path to folder containing zarr files
        output_folder (str): Path to output folder for NetCDF files
        dates (list): List of dates to process (format: YYYY-MM-DD)
        year (int): Year of the data
        fields (list): List of fields to convert (default: all_fcst_fields)
    """
    if fields is None:
        fields = all_fcst_fields

    # Create output folder if it doesn't exist
    output_folder_year = os.path.join(output_folder, str(year))
    os.makedirs(output_folder_year, exist_ok=True)

    print(f"Converting zarr files to NetCDF...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder_year}")
    print(f"Fields to convert: {fields}")
    print(f"Dates: {dates}")

    for field in fields:
        print(f"\nProcessing field: {field}")

        # Construct input zarr path
        input_file = f"{field}_{year}.zarr"
        input_folder_year = os.path.join(input_folder, str(year))
        zarr_path = os.path.join(input_folder_year, input_file)

        if not os.path.exists(zarr_path):
            print(f"  Warning: Zarr file not found at {zarr_path}, skipping...")
            continue

        try:
            # Open the zarr file
            print(f"  Opening: {zarr_path}")
            ds = xr.open_zarr(zarr_path)

            # Filter by dates if specified
            if dates:
                dates_array = np.array(dates, dtype='datetime64[ns]')
                ds = ds.sel(time=dates_array)

            # Construct output NetCDF path
            output_file = f"{field}_{year}.nc"
            output_path = os.path.join(output_folder_year, output_file)

            # Save as NetCDF
            print(f"  Saving to: {output_path}")
            ds.to_netcdf(output_path)
            ds.close()

            print(f"  ✓ Successfully converted {field}")

        except Exception as e:
            print(f"  ✗ Error converting {field}: {str(e)}")
            continue

    print(f"\n✓ Conversion complete! NetCDF files saved to: {output_folder_year}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GEFS zarr files to NetCDF format'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to forecast_gfs.yaml configuration file'
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        help='Input folder containing zarr files (overrides config)'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='Output folder for NetCDF files (overrides config)'
    )
    parser.add_argument(
        '--dates',
        nargs='+',
        help='List of dates to process (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Year of the data'
    )
    parser.add_argument(
        '--fields',
        nargs='+',
        help='List of fields to convert (default: all forecast fields)'
    )

    args = parser.parse_args()

    # Load configuration from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        input_folder = args.input_folder or config['INPUT']['folder']
        dates = args.dates or config['INPUT']['dates']

        # Infer year from dates if not provided
        if args.year:
            year = args.year
        else:
            year = datetime.strptime(dates[0], "%Y-%m-%d").year

        # Set output folder (default to input_folder/netcdf)
        output_folder = args.output_folder or os.path.join(
            os.path.dirname(input_folder.rstrip('/')),
            'netcdf'
        )
    else:
        # Use command-line arguments only
        if not all([args.input_folder, args.dates, args.year]):
            parser.error(
                "Either --config or all of (--input_folder, --dates, --year) "
                "must be provided"
            )

        input_folder = args.input_folder
        dates = args.dates
        year = args.year
        output_folder = args.output_folder or os.path.join(
            os.path.dirname(input_folder.rstrip('/')),
            'netcdf'
        )

    # Convert zarr to NetCDF
    convert_zarr_to_netcdf(
        input_folder=input_folder,
        output_folder=output_folder,
        dates=dates,
        year=year,
        fields=args.fields
    )

    # Print instructions for updating config
    print("\n" + "="*70)
    print("To use these NetCDF files with forecast_gfs.py:")
    print("="*70)
    print(f"\n1. Update config/forecast_gfs.yaml INPUT folder to:")
    print(f"   folder: \"{output_folder}\"")
    print(f"\n2. Or create a new config file with the NetCDF path")
    print("="*70)


if __name__ == "__main__":
    main()
