#!/usr/bin/env python3
"""
Single GEFS Ensemble Member Zarr Processing V2 (Enhanced)
This script processes a single GEFS ensemble parquet file and converts it to a single
unified zarr store with optional regional subsetting and improved variable handling.

Key improvements over v1:
- Single unified zarr output instead of multiple files
- East Africa regional subsetting capability  
- Better variable mapping and validation
- Optimized chunking strategy
- Improved compression and encoding
- Enhanced error handling and logging
"""

import os
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import fsspec
import xarray as xr
import numpy as np
from dotenv import load_dotenv


# Suppress xarray future warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='xarray')

# Variable mapping from requested names to available GEFS variables
VARIABLE_MAPPING = {
    # Available variables (exact matches)
    'cape': 'cape',           # Convective Available Potential Energy
    't2m': 't2m',            # 2m Temperature
    'u10': 'u10',            # 10m U Wind Component
    'v10': 'v10',            # 10m V Wind Component
    'tp': 'tp',              # Total Precipitation
    'gust': 'gust',          # Surface Wind Gust
    'sp': 'sp',              # Surface Pressure
    'r2': 'r2',              # 2m Relative Humidity
    
    # Alternative names/aliases
    'apcp': 'tp',            # Accumulated Precipitation -> Total Precipitation
    'pres': 'sp',            # Pressure -> Surface Pressure
    'tmp': 't2m',            # Temperature -> 2m Temperature
    'ugrd': 'u10',           # U Grid Wind -> 10m U Wind
    'vgrd': 'v10',           # V Grid Wind -> 10m V Wind
    
    # Radiation variables
    'sdswrf': 'sdswrf',      # Downward Shortwave Radiation
    'suswrf': 'suswrf',      # Upward Shortwave Radiation
    
    # Note: pwat (Precipitable Water) and msl (Mean Sea Level Pressure) 
    # may not be available in this dataset
}

# Regional bounds for subsetting
REGIONS = {
    'east_africa': {
        'lon_min': 21, 'lon_max': 53,
        'lat_min': -12, 'lat_max': 23,
        'description': 'East Africa region covering Kenya, Tanzania, Ethiopia, etc.'
    },
    'global': {
        'lon_min': 0, 'lon_max': 360,
        'lat_min': -90, 'lat_max': 90,
        'description': 'Global coverage'
    }
}


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)


def read_parquet_fixed(parquet_path: str, logger) -> Dict[str, Any]:
    """Read parquet files with proper handling - enhanced version."""
    import ast
    
    logger.info(f"📖 Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"📊 Parquet file loaded: {len(df)} rows")
    
    if 'refs' in df['key'].values and len(df) <= 2:
        # Old format - single refs row
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
        logger.info(f"✅ Extracted {len(zstore)} entries from old format")
    else:
        # New format - each key-value pair is a row
        zstore = {}
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']
            
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except:
                        pass
            
            zstore[key] = value
        
        logger.info(f"✅ Loaded {len(zstore)} entries from new format")
    
    if 'version' in zstore:
        del zstore['version']
    
    return zstore


def discover_variables(zstore: Dict[str, Any], logger) -> Dict[str, Dict[str, Any]]:
    """Discover all available variables in the zarr store."""
    logger.info("🔍 Discovering available variables...")
    
    try:
        # Create reference filesystem
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")
        
        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        logger.info(f"✅ Successfully opened datatree with {len(dt.groups)} groups")
        
        # Find all variable groups (those with actual data variables)
        variable_groups = {}
        for group_path in dt.groups:
            if group_path in ['/', '']:
                continue
            try:
                group = dt[group_path]
                if hasattr(group, 'ds') and group.ds.data_vars:
                    for var_name in group.ds.data_vars:
                        variable_groups[group_path] = {
                            'variable': var_name,
                            'shape': group.ds[var_name].shape,
                            'dims': group.ds[var_name].dims,
                            'dataset': group.ds
                        }
                        break  # Take the first data variable from each group
            except Exception as e:
                logger.debug(f"Skipping group {group_path}: {e}")
                continue
        
        logger.info(f"📊 Found {len(variable_groups)} variable groups:")
        for group_path, info in variable_groups.items():
            logger.info(f"   {group_path}: {info['variable']} {info['shape']} {info['dims']}")
        
        return variable_groups
        
    except Exception as e:
        logger.error(f"❌ Error discovering variables: {e}")
        import traceback
        traceback.print_exc()
        return {}


def validate_requested_variables(requested_vars: List[str], available_groups: Dict[str, Dict], logger) -> Tuple[List[str], List[str]]:
    """Validate which requested variables are available."""
    logger.info("🔍 Validating requested variables...")
    
    # Get available variable names
    available_vars = set()
    for group_info in available_groups.values():
        available_vars.add(group_info['variable'])
    
    found_vars = []
    missing_vars = []
    
    for req_var in requested_vars:
        mapped_var = VARIABLE_MAPPING.get(req_var, req_var)
        if mapped_var in available_vars:
            found_vars.append(mapped_var)
            logger.info(f"✅ {req_var} -> {mapped_var} (available)")
        else:
            missing_vars.append(req_var)
            logger.warning(f"❌ {req_var} -> {mapped_var} (not available)")
    
    return found_vars, missing_vars


def subset_dataset(ds: xr.Dataset, region: str, logger) -> xr.Dataset:
    """Subset dataset to specified region."""
    if region == 'global':
        logger.info("🌍 Using global extent (no subsetting)")
        return ds
    
    if region not in REGIONS:
        logger.warning(f"⚠️ Unknown region '{region}', using global extent")
        return ds
    
    bounds = REGIONS[region]
    logger.info(f"🌍 Subsetting to {bounds['description']}")
    logger.info(f"   Longitude: {bounds['lon_min']}° to {bounds['lon_max']}°")
    logger.info(f"   Latitude: {bounds['lat_min']}° to {bounds['lat_max']}°")
    
    # Create masks for subsetting
    lat_mask = (ds.latitude >= bounds['lat_min']) & (ds.latitude <= bounds['lat_max'])
    lon_mask = (ds.longitude >= bounds['lon_min']) & (ds.longitude <= bounds['lon_max'])
    
    # Apply subsetting
    ds_subset = ds.sel(latitude=ds.latitude[lat_mask], longitude=ds.longitude[lon_mask])
    
    # Log subset statistics
    orig_points = ds.latitude.size * ds.longitude.size
    subset_points = ds_subset.latitude.size * ds_subset.longitude.size
    reduction = 100 * (1 - subset_points / orig_points)
    
    logger.info(f"📊 Subset statistics:")
    logger.info(f"   Original grid: {ds.latitude.size} x {ds.longitude.size} = {orig_points:,} points")
    logger.info(f"   Subset grid: {ds_subset.latitude.size} x {ds_subset.longitude.size} = {subset_points:,} points")
    logger.info(f"   Size reduction: {reduction:.1f}%")
    
    return ds_subset


def create_combined_dataset(zstore: Dict[str, Any], variable_groups: Dict[str, Dict], 
                           target_vars: List[str], region: str, logger) -> Optional[xr.Dataset]:
    """Create a combined dataset with all requested variables."""
    logger.info(f"🔧 Creating combined dataset with {len(target_vars)} variables...")
    
    try:
        # Create reference filesystem
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")
        
        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        
        # Find groups for target variables
        var_to_group = {}
        for group_path, info in variable_groups.items():
            var_name = info['variable']
            if var_name in target_vars:
                var_to_group[var_name] = group_path
        
        # Load and combine datasets
        combined_datasets = {}
        for var_name in target_vars:
            if var_name in var_to_group:
                group_path = var_to_group[var_name]
                logger.info(f"   Loading {var_name} from {group_path}...")
                
                # Get the dataset for this group
                group_ds = dt[group_path].ds
                
                # Apply regional subsetting
                group_ds_subset = subset_dataset(group_ds, region, logger)
                
                # Store with variable name as key
                combined_datasets[var_name] = group_ds_subset[var_name]
            else:
                logger.warning(f"   Skipping {var_name} - group not found")
        
        if not combined_datasets:
            logger.error("❌ No variables could be loaded")
            return None
        
        # Create combined dataset
        logger.info("🔗 Combining datasets...")
        
        # Get coordinate system from first dataset
        first_var = list(combined_datasets.keys())[0]
        first_group_path = var_to_group[first_var]
        coords_ds = dt[first_group_path].ds
        
        # Apply regional subsetting to coordinates
        coords_subset = subset_dataset(coords_ds, region, logger)
        
        # Create new dataset with all variables
        combined_ds = xr.Dataset(
            data_vars=combined_datasets,
            coords=coords_subset.coords
        )
        
        # Add global attributes
        combined_ds.attrs.update({
            'title': 'GEFS Ensemble Weather Data',
            'institution': 'NOAA/NCEP',
            'source': 'Global Ensemble Forecast System (GEFS)',
            'processing_version': 'v2.0',
            'region': REGIONS[region]['description'],
            'variables': ', '.join(target_vars),
            'created_by': 'run_single_gefs_to_zarr_v2.py'
        })
        
        # Add variable attributes (encoding will be handled in save function)
        for var_name in combined_ds.data_vars:
            # Add CF-compliant attributes based on variable
            if var_name == 't2m':
                combined_ds[var_name].attrs.update({
                    'long_name': '2 metre temperature',
                    'units': 'K',
                    'standard_name': 'air_temperature'
                })
            elif var_name == 'tp':
                combined_ds[var_name].attrs.update({
                    'long_name': 'Total precipitation',
                    'units': 'm',
                    'standard_name': 'precipitation_amount'
                })
            elif var_name in ['u10', 'ugrd']:
                combined_ds[var_name].attrs.update({
                    'long_name': '10 metre U wind component',
                    'units': 'm s**-1',
                    'standard_name': 'eastward_wind'
                })
            elif var_name in ['v10', 'vgrd']:
                combined_ds[var_name].attrs.update({
                    'long_name': '10 metre V wind component',
                    'units': 'm s**-1',
                    'standard_name': 'northward_wind'
                })
            elif var_name == 'cape':
                combined_ds[var_name].attrs.update({
                    'long_name': 'Convective available potential energy',
                    'units': 'J kg**-1'
                })
            elif var_name in ['sp', 'pres']:
                combined_ds[var_name].attrs.update({
                    'long_name': 'Surface pressure',
                    'units': 'Pa',
                    'standard_name': 'surface_air_pressure'
                })
        
        logger.info(f"✅ Combined dataset created with shape: {combined_ds.dims}")
        
        return combined_ds
        
    except Exception as e:
        logger.error(f"❌ Error creating combined dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def optimize_chunking(ds: xr.Dataset, region: str) -> Dict[str, int]:
    """Optimize chunking strategy based on dataset size and region."""
    if region == 'east_africa':
        # Optimized for smaller regional dataset
        chunks = {
            'time': min(10, ds.sizes.get('time', 81)),
            'latitude': min(64, ds.sizes.get('latitude', 141)),
            'longitude': min(70, ds.sizes.get('longitude', 129))
        }
    else:
        # Optimized for global dataset
        chunks = {
            'time': min(10, ds.sizes.get('time', 81)),
            'latitude': min(128, ds.sizes.get('latitude', 721)),
            'longitude': min(256, ds.sizes.get('longitude', 1440))
        }
    
    # Remove dimensions that don't exist in the dataset
    chunks = {k: v for k, v in chunks.items() if k in ds.dims}
    
    return chunks


def save_combined_zarr(ds: xr.Dataset, output_path: str, member_name: str, 
                      region: str, logger) -> bool:
    """Save the combined dataset as a single zarr store."""
    logger.info(f"💾 Saving combined zarr store to: {output_path}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Optimize chunking
        chunks = optimize_chunking(ds, region)
        logger.info(f"📦 Using chunks: {chunks}")
        
        # Create encoding for all variables
        import numcodecs
        encoding = {}
        for var_name in ds.data_vars:
            encoding[var_name] = {
                'chunks': tuple(chunks.get(dim, ds.sizes[dim]) for dim in ds[var_name].dims),
                'compressor': numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
            }
        
        # Add coordinate encoding
        for coord_name in ds.coords:
            if coord_name in ds.data_vars:
                continue  # Skip data variables that are also coordinates
            if ds[coord_name].dtype != object:  # Skip object dtypes
                encoding[coord_name] = {
                    'compressor': numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
                }
        
        logger.info("💾 Writing to zarr (this may take a few minutes)...")
        ds.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)
        
        # Calculate and report file size
        import subprocess
        result = subprocess.run(['du', '-sh', output_path], capture_output=True, text=True)
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            logger.info(f"✅ Zarr store saved: {size_str}")
        else:
            logger.info("✅ Zarr store saved successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving zarr store: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zarr_store(zarr_path: str, logger) -> bool:
    """Test reading the saved zarr store."""
    logger.info(f"🔍 Testing saved zarr store: {os.path.basename(zarr_path)}")
    
    try:
        ds = xr.open_dataset(zarr_path, engine='zarr')
        logger.info(f"📊 Variables: {list(ds.data_vars)}")
        logger.info(f"📐 Coordinates: {list(ds.coords)}")
        logger.info(f"🎯 Dataset shape: {dict(ds.sizes)}")
        
        # Test data access
        if ds.data_vars:
            first_var = list(ds.data_vars)[0]
            sample_data = ds[first_var].isel({dim: 0 for dim in ds[first_var].dims if dim != 'latitude' and dim != 'longitude'})
            if 'latitude' in sample_data.dims and 'longitude' in sample_data.dims:
                sample_data = sample_data.isel(latitude=slice(0, 5), longitude=slice(0, 5))
            sample_values = sample_data.compute()
            logger.info(f"✅ Sample data access successful: {sample_values.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing zarr store: {e}")
        return False


def main(date_str: str, run_str: str, member: str, 
         variables: Optional[List[str]] = None,
         region: str = 'global',
         output_dir: Optional[str] = None,
         parquet_dir: Optional[str] = None,
         verbose: bool = False):
    """
    Process a single GEFS ensemble member from parquet to unified zarr.
    
    Args:
        date_str: Date string in YYYYMMDD format
        run_str: Run string (e.g., '00', '06', '12', '18')
        member: Ensemble member (e.g., 'gep01')
        variables: List of variables to extract (default: all available)
        region: Region to extract ('global' or 'east_africa')
        output_dir: Output directory (default: ./zarr_stores)
        parquet_dir: Directory containing GEFS parquet files
        verbose: Enable verbose logging
    """
    logger = setup_logging(verbose)
    
    logger.info(f"🚀 Processing GEFS ensemble member: {member}")
    logger.info(f"📅 Date: {date_str}, Run: {run_str}")
    logger.info(f"🌍 Region: {region}")
    logger.info(f"🔧 Version: 2.0 (Enhanced)")
    
    # Set up directories
    if parquet_dir is None:
        parquet_dir = f"{date_str}_{run_str}"
    
    if output_dir is None:
        output_dir = "./zarr_stores"
    
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        logger.error(f"❌ Parquet directory {parquet_path} does not exist.")
        logger.info(f"💡 Run the GEFS download script first to create parquet files.")
        return False
    
    # Find parquet file for this member
    parquet_file = parquet_path / f"{member}.par"
    if not parquet_file.exists():
        logger.error(f"❌ Parquet file {parquet_file} not found.")
        return False
    
    logger.info(f"📂 Using parquet file: {parquet_file}")
    
    # Step 1: Read parquet file
    try:
        zstore = read_parquet_fixed(str(parquet_file), logger)
        logger.info(f"✅ Successfully loaded zarr store with {len(zstore)} entries")
    except Exception as e:
        logger.error(f"❌ Failed to read parquet file: {e}")
        return False
    
    # Step 2: Discover available variables
    variable_groups = discover_variables(zstore, logger)
    if not variable_groups:
        logger.error("❌ No variables found in zarr store")
        return False
    
    # Step 3: Validate requested variables
    if variables is None:
        # Use all available variables
        target_vars = [info['variable'] for info in variable_groups.values()]
        logger.info(f"📋 Using all available variables: {target_vars}")
    else:
        target_vars, missing_vars = validate_requested_variables(variables, variable_groups, logger)
        if missing_vars:
            logger.warning(f"⚠️ Missing variables will be skipped: {missing_vars}")
        if not target_vars:
            logger.error("❌ No requested variables are available")
            return False
    
    # Step 4: Create combined dataset
    combined_ds = create_combined_dataset(zstore, variable_groups, target_vars, region, logger)
    if combined_ds is None:
        logger.error("❌ Failed to create combined dataset")
        return False
    
    # Step 5: Save to zarr
    region_suffix = f"_{region}" if region != 'global' else ""
    zarr_filename = f"{member}_combined{region_suffix}.zarr"
    zarr_path = os.path.join(output_dir, f"{date_str}_{run_str}", zarr_filename)
    
    success = save_combined_zarr(combined_ds, zarr_path, member, region, logger)
    if not success:
        logger.error("❌ Failed to save zarr store")
        return False
    
    # Step 6: Test the saved zarr store
    if test_zarr_store(zarr_path, logger):
        logger.info(f"✅ Processing completed successfully!")
        logger.info(f"📁 Zarr store saved to: {zarr_path}")
        return True
    else:
        logger.warning("⚠️ Zarr store saved but testing failed")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process single GEFS ensemble member to unified zarr (v2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all variables for global coverage
  %(prog)s 20250709 00 gep01
  
  # Process specific variables for East Africa region
  %(prog)s 20250709 00 gep01 --variables t2m,tp,u10,v10 --region east_africa
  
  # Process all available variables for East Africa
  %(prog)s 20250709 00 gep01 --region east_africa
  
Available regions: global, east_africa
Variable aliases: pres->sp, tmp->t2m, ugrd->u10, vgrd->v10, apcp->tp
        """
    )
    
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format")
    parser.add_argument("run_str", type=str, help="Run string (e.g., '00', '06', '12', '18')")
    parser.add_argument("member", type=str, help="Ensemble member (e.g., 'gep01')")
    
    parser.add_argument("--variables", type=str, 
                       help="Comma-separated list of variables to extract (default: all available)")
    parser.add_argument("--region", type=str, default='east_africa', choices=['global', 'east_africa'],
                       help="Region to extract (default: global)")
    parser.add_argument("--output-dir", type=str, default="./zarr_stores",
                       help="Output directory (default: ./zarr_stores)")
    parser.add_argument("--parquet-dir", type=str,
                       help="Directory containing GEFS parquet files (default: {date_str}_{run_str})")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Parse variables if provided
    variables = None
    if args.variables:
        variables = [v.strip() for v in args.variables.split(',')]
    
    success = main(
        args.date_str, args.run_str, args.member,
        variables=variables,
        region=args.region,
        output_dir=args.output_dir,
        parquet_dir=args.parquet_dir,
        verbose=args.verbose
    )
    
    if not success:
        exit(1)