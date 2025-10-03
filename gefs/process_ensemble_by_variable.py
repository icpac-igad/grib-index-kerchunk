#!/usr/bin/env python3
"""
GEFS Ensemble Processing by Variable
Memory-efficient processing: load one variable from all ensemble members,
concatenate along member dimension, save as NetCDF, then move to next variable.

This script implements the variable-by-variable strategy discussed in gefs-gik-part2.md
for memory-efficient ensemble processing with statistics computation.

Usage:
    python process_ensemble_by_variable.py <zarr_dir> [options]

Example:
    # Process all variables
    python process_ensemble_by_variable.py zarr_stores/20250909_18/

    # Process specific variables only
    python process_ensemble_by_variable.py zarr_stores/20250909_18/ --variables t2m,tp,cape

    # Skip statistics computation
    python process_ensemble_by_variable.py zarr_stores/20250909_18/ --no-stats
"""

import os
import sys
import argparse
import logging
import warnings
import time
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import xarray as xr
import pandas as pd
from glob import glob

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Custom formatter with timestamps
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                 datefmt='%H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(console_handler)

    return logger

def get_memory_usage() -> Tuple[float, float]:
    """Get current memory usage in GB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    used_gb = memory_info.rss / (1024**3)

    virtual_memory = psutil.virtual_memory()
    available_gb = virtual_memory.available / (1024**3)

    return used_gb, available_gb

def get_file_size(file_path: str) -> Tuple[int, str]:
    """Get file size in bytes and human readable format."""
    if not os.path.exists(file_path):
        return 0, "0B"

    size_bytes = os.path.getsize(file_path)

    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            size_str = f"{size_bytes:.1f}{unit}"
            break
        size_bytes /= 1024
    else:
        size_str = f"{size_bytes:.1f}PB"

    return os.path.getsize(file_path), size_str

def discover_ensemble_files(zarr_dir: str, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """Discover zarr files and extract ensemble member names."""
    logger.info(f"🔍 Scanning directory: {zarr_dir}")

    # Find all zarr files
    zarr_pattern = os.path.join(zarr_dir, "*.zarr")
    zarr_files = glob(zarr_pattern)

    if not zarr_files:
        raise FileNotFoundError(f"No zarr files found in {zarr_dir}")

    # Extract ensemble member names
    ensemble_members = []
    for zarr_file in zarr_files:
        filename = os.path.basename(zarr_file)
        # Extract member name (assuming format: gep01_combined_east_africa.zarr)
        if '_combined_' in filename:
            member_name = filename.split('_combined_')[0]
        else:
            # Fallback: assume first part before underscore
            member_name = filename.split('_')[0] if '_' in filename else filename.replace('.zarr', '')
        ensemble_members.append(member_name)

    # Sort for consistent ordering
    sorted_pairs = sorted(zip(ensemble_members, zarr_files))
    ensemble_members, zarr_files = zip(*sorted_pairs)

    logger.info(f"📋 Found {len(zarr_files)} zarr files:")
    for i, (member, zarr_file) in enumerate(zip(ensemble_members, zarr_files), 1):
        logger.info(f"   {i:2d}. {member}: {os.path.basename(zarr_file)}")

    return list(ensemble_members), list(zarr_files)

def discover_common_variables(zarr_files: List[str], logger: logging.Logger) -> List[str]:
    """Discover variables that are common across all zarr files."""
    logger.info("🔍 Discovering common variables across all files...")

    all_variables = None

    for i, zarr_file in enumerate(zarr_files):
        try:
            with xr.open_dataset(zarr_file, engine='zarr') as ds:
                file_variables = set(ds.data_vars.keys())

                if all_variables is None:
                    all_variables = file_variables
                else:
                    all_variables = all_variables.intersection(file_variables)

                logger.debug(f"   File {i+1}: {len(file_variables)} variables")

        except Exception as e:
            logger.warning(f"   ⚠️ Could not read {os.path.basename(zarr_file)}: {e}")
            continue

    if all_variables is None:
        raise ValueError("No readable zarr files found")

    common_variables = sorted(list(all_variables))
    logger.info(f"📋 Found {len(common_variables)} common variables: {common_variables}")

    return common_variables

def process_variable_ensemble(variable_name: str, ensemble_members: List[str],
                            zarr_files: List[str], output_dir: str,
                            logger: logging.Logger) -> Dict[str, any]:
    """Process a single variable across all ensemble members."""
    logger.info(f"🔄 Processing variable: {variable_name}")

    # Track memory and timing
    start_time = time.time()
    start_memory, available_memory = get_memory_usage()
    logger.info(f"   💾 Memory at start: {start_memory:.2f}GB used, {available_memory:.2f}GB available")

    # Results tracking
    results = {
        'variable': variable_name,
        'num_members': len(ensemble_members),
        'success': False,
        'file_size': '0B',
        'processing_time': 0.0,
        'peak_memory': start_memory
    }

    try:
        # Load variable data from all ensemble members
        logger.info(f"   📖 Loading {variable_name} from {len(zarr_files)} zarr files...")

        datasets = []
        for i, (member, zarr_file) in enumerate(zip(ensemble_members, zarr_files)):
            logger.debug(f"      Loading {member}: {os.path.basename(zarr_file)}")

            # Open dataset and extract only the target variable
            ds = xr.open_dataset(zarr_file, engine='zarr')

            if variable_name not in ds.data_vars:
                logger.warning(f"      ⚠️ Variable {variable_name} not found in {member}")
                ds.close()
                continue

            # Extract only the target variable and essential coordinates
            var_ds = ds[[variable_name]].copy()
            ds.close()

            # Add member coordinate
            var_ds = var_ds.expand_dims('member')
            var_ds['member'] = [member]

            datasets.append(var_ds)

            # Monitor memory usage
            current_memory, _ = get_memory_usage()
            results['peak_memory'] = max(results['peak_memory'], current_memory)

            # Cleanup
            gc.collect()

        if not datasets:
            raise ValueError(f"No valid datasets found for variable {variable_name}")

        logger.info(f"   🔗 Concatenating {len(datasets)} datasets along member dimension...")

        # Concatenate along member dimension
        ensemble_ds = xr.concat(datasets, dim='member')

        # Clean up individual datasets
        for ds in datasets:
            ds.close()
        del datasets
        gc.collect()

        # Update memory tracking
        current_memory, _ = get_memory_usage()
        results['peak_memory'] = max(results['peak_memory'], current_memory)

        logger.info(f"   📊 Ensemble dataset shape: {dict(ensemble_ds.sizes)}")

        # Add ensemble metadata
        ensemble_ds.attrs.update({
            'title': f'GEFS Ensemble - {variable_name.upper()}',
            'description': f'Ensemble forecast data for {variable_name} from {len(ensemble_members)} members',
            'ensemble_size': len(ensemble_members),
            'ensemble_members': ','.join(ensemble_members),
            'created_by': 'process_ensemble_by_variable.py',
            'creation_time': pd.Timestamp.now().isoformat()
        })

        # Save ensemble NetCDF
        output_file = os.path.join(output_dir, f"ensemble_{variable_name}.nc")
        logger.info(f"   💾 Saving ensemble NetCDF: {os.path.basename(output_file)}")

        # Use compression for NetCDF
        encoding = {
            variable_name: {
                'zlib': True,
                'complevel': 6,
                'shuffle': True
            }
        }

        ensemble_ds.to_netcdf(output_file, encoding=encoding)

        # Get file size
        file_size_bytes, file_size_str = get_file_size(output_file)
        results['file_size'] = file_size_str

        # Cleanup
        ensemble_ds.close()

        # Final memory and timing
        end_time = time.time()
        end_memory, _ = get_memory_usage()
        results['processing_time'] = end_time - start_time

        logger.info(f"   ✅ Success: {file_size_str} in {results['processing_time']:.1f}s")
        logger.info(f"   💾 Peak memory: {results['peak_memory']:.2f}GB")

        results['success'] = True
        return results

    except Exception as e:
        logger.error(f"   ❌ Failed to process {variable_name}: {str(e)}")
        results['error'] = str(e)
        return results

def compute_statistics_for_variable(variable_name: str, variable_file: str,
                                   output_dir: str, logger: logging.Logger) -> Tuple[bool, Dict]:
    """Compute ensemble statistics (mean and std) for a variable."""
    logger.info(f"📊 Computing statistics for {variable_name}")

    stats = {
        'variable': variable_name,
        'mean_file_size': '0B',
        'std_file_size': '0B',
        'success': False
    }

    try:
        # Open the variable ensemble file with simple loading
        logger.info(f"   Loading NetCDF file: {os.path.basename(variable_file)}")
        # Use decode_cf=False to avoid dtype attribute conflicts in coordinates
        ds_original = xr.open_dataset(variable_file, decode_cf=False)

        # Extract only the main data variable and essential coordinates
        main_var = variable_name
        if main_var not in ds_original.data_vars:
            logger.error(f"   Variable {main_var} not found in dataset")
            return False, stats

        # Create a clean dataset with only essential data
        essential_coords = ['member', 'latitude', 'longitude', 'valid_times']
        clean_coords = {k: v for k, v in ds_original.coords.items() if k in essential_coords}

        # Create new clean dataset
        ds = xr.Dataset(
            data_vars={main_var: ds_original[main_var]},
            coords=clean_coords
        )

        logger.info(f"   Clean dataset shape: {dict(ds.sizes)}")

        # Compute statistics along member dimension
        logger.info(f"   Computing mean and std for {ds.sizes['member']} members...")
        mean_da = ds[main_var].mean(dim='member', keep_attrs=True)
        std_da = ds[main_var].std(dim='member', keep_attrs=True)

        # Create new datasets for mean and std
        mean_ds = xr.Dataset(
            data_vars={main_var: mean_da},
            coords={k: v for k, v in clean_coords.items() if k != 'member'}
        )

        std_ds = xr.Dataset(
            data_vars={main_var: std_da},
            coords={k: v for k, v in clean_coords.items() if k != 'member'}
        )

        # Update attributes
        mean_ds.attrs.update({
            'title': f'GEFS Ensemble Mean - {variable_name.upper()}',
            'description': f'Ensemble mean of {variable_name} computed from {ds.sizes["member"]} members',
            'statistic': 'mean',
            'source_file': os.path.basename(variable_file),
            'created_by': 'process_ensemble_by_variable.py'
        })

        std_ds.attrs.update({
            'title': f'GEFS Ensemble Standard Deviation - {variable_name.upper()}',
            'description': f'Ensemble standard deviation of {variable_name} computed from {ds.sizes["member"]} members',
            'statistic': 'standard_deviation',
            'source_file': os.path.basename(variable_file),
            'created_by': 'process_ensemble_by_variable.py'
        })

        # Update variable attributes
        if variable_name in mean_ds.data_vars:
            mean_ds[variable_name].attrs['long_name'] = f"{mean_ds[variable_name].attrs.get('long_name', variable_name)} - Ensemble Mean"
            std_ds[variable_name].attrs['long_name'] = f"{std_ds[variable_name].attrs.get('long_name', variable_name)} - Ensemble Std Dev"

        # Save statistics
        mean_file = os.path.join(output_dir, f"ensemble_mean_{variable_name}.nc")
        std_file = os.path.join(output_dir, f"ensemble_std_{variable_name}.nc")

        encoding = {variable_name: {'zlib': True, 'complevel': 6}}

        logger.info(f"   Saving mean: {os.path.basename(mean_file)}")
        mean_ds.to_netcdf(mean_file, encoding=encoding)

        logger.info(f"   Saving std: {os.path.basename(std_file)}")
        std_ds.to_netcdf(std_file, encoding=encoding)

        # Get file sizes
        _, stats['mean_file_size'] = get_file_size(mean_file)
        _, stats['std_file_size'] = get_file_size(std_file)

        # Cleanup
        mean_ds.close()
        std_ds.close()
        ds.close()
        ds_original.close()

        stats['success'] = True
        return True, stats

    except Exception as e:
        logger.error(f"   ❌ Statistics computation failed: {str(e)}")
        stats['error'] = str(e)
        return False, stats

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Process GEFS ensemble members variable by variable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all variables with statistics
  python process_ensemble_by_variable.py zarr_stores/20250909_18/

  # Process specific variables only
  python process_ensemble_by_variable.py zarr_stores/20250909_18/ --variables t2m,tp,cape

  # Skip statistics computation (faster)
  python process_ensemble_by_variable.py zarr_stores/20250909_18/ --no-stats

  # Custom output directory
  python process_ensemble_by_variable.py zarr_stores/20250909_18/ --output processed_variables/
        """
    )

    parser.add_argument('zarr_dir', help='Directory containing ensemble zarr files')
    parser.add_argument('--variables', type=str,
                       help='Comma-separated list of variables to process (default: all)')
    parser.add_argument('--output', default=None,
                       help='Output directory (default: <zarr_dir>/processed_variables)')
    parser.add_argument('--no-stats', action='store_true',
                       help='Skip statistics computation (only create ensemble files)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Check input directory
    if not os.path.exists(args.zarr_dir):
        logger.error(f"❌ Directory not found: {args.zarr_dir}")
        sys.exit(1)

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.zarr_dir, "processed_variables")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("🚀 GEFS Ensemble Processing by Variable")
    logger.info(f"📂 Source directory: {args.zarr_dir}")
    logger.info(f"📂 Output directory: {output_dir}")

    # Show initial memory
    used_memory, available_memory = get_memory_usage()
    logger.info(f"🧠 Available memory: {available_memory:.1f}GB")

    try:
        # Discover ensemble files
        ensemble_members, zarr_files = discover_ensemble_files(args.zarr_dir, logger)

        # Discover common variables
        available_variables = discover_common_variables(zarr_files, logger)

        # Filter variables if requested
        if args.variables:
            requested_vars = [v.strip() for v in args.variables.split(',')]
            variables_to_process = [v for v in requested_vars if v in available_variables]

            if not variables_to_process:
                logger.error(f"❌ None of the requested variables found. Available: {available_variables}")
                sys.exit(1)

            missing_vars = [v for v in requested_vars if v not in available_variables]
            if missing_vars:
                logger.warning(f"⚠️ Requested variables not found: {missing_vars}")

            logger.info(f"📋 Processing {len(variables_to_process)} requested variables: {variables_to_process}")
        else:
            variables_to_process = available_variables
            logger.info(f"📋 Processing all {len(variables_to_process)} variables")

        # Process each variable
        total_start_time = time.time()
        results_summary = []

        for i, variable_name in enumerate(variables_to_process, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Variable {i}/{len(variables_to_process)}: {variable_name}")
            logger.info(f"{'='*60}")

            # Process ensemble for this variable
            results = process_variable_ensemble(
                variable_name, ensemble_members, zarr_files, output_dir, logger
            )
            results_summary.append(results)

            # Compute statistics if requested and processing was successful
            if not args.no_stats and results['success']:
                variable_file = os.path.join(output_dir, f"ensemble_{variable_name}.nc")
                if os.path.exists(variable_file):
                    stats_success, stats_results = compute_statistics_for_variable(
                        variable_name, variable_file, output_dir, logger
                    )
                    results.update(stats_results)

            # Memory cleanup
            gc.collect()

        # Final summary
        total_time = time.time() - total_start_time
        logger.info(f"\n{'='*60}")
        logger.info("📊 PROCESSING SUMMARY")
        logger.info(f"{'='*60}")

        successful_vars = [r for r in results_summary if r['success']]
        failed_vars = [r for r in results_summary if not r['success']]

        logger.info(f"✅ Successfully processed: {len(successful_vars)}/{len(variables_to_process)} variables")
        logger.info(f"❌ Failed: {len(failed_vars)} variables")
        logger.info(f"⏱️ Total processing time: {total_time:.1f} seconds")

        # Detailed results
        if successful_vars:
            logger.info("\n📋 Successful variables:")
            for result in successful_vars:
                logger.info(f"   {result['variable']}: {result['file_size']} "
                           f"({result['processing_time']:.1f}s, peak: {result['peak_memory']:.2f}GB)")

        if failed_vars:
            logger.info("\n❌ Failed variables:")
            for result in failed_vars:
                error_msg = result.get('error', 'Unknown error')
                logger.info(f"   {result['variable']}: {error_msg}")

        logger.info(f"\n📁 Output directory: {output_dir}")
        logger.info("🎉 Processing completed!")

    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()