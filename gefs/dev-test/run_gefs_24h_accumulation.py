#!/usr/bin/env python3
"""
GEFS 24-hour Rainfall Accumulation Processing V2
This script processes GEFS ensemble data to create 24-hour rainfall accumulation
plots with threshold exceedance probabilities.

MEMORY OPTIMIZATION:
- Extracts coordinates FIRST to determine East Africa region indices
- Only loads regional subset (not full global array) using spatial slicing
- Skips S3 chunks outside the region of interest
- Reduces memory usage from ~400MB to ~15MB per ensemble member
- Prevents memory crashes when processing 30+ ensemble members

Additional plotting functions moved from run_day_gefs_ensemble_full.py:
- Ensemble comparison plots
- Multiple timestep plotting
- Probability maps and summaries
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import fsspec
import geopandas as gp
import matplotlib.patches as mpatches
import time
import tempfile

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Configuration
PARQUET_DIR = Path("20250918_00")  # Directory containing parquet files
BOUNDARY_JSON = "ea_ghcf_simple.geojson"  # GeoJSON file for boundaries

# Extract date and run hour from directory name
dir_name = PARQUET_DIR.name
if "_" in dir_name:
    date_str, run_hour_str = dir_name.split("_")
    MODEL_DATE = datetime.strptime(date_str, "%Y%m%d")
    MODEL_RUN_HOUR = int(run_hour_str)
else:
    # Fallback if format is different
    MODEL_DATE = datetime.now()
    MODEL_RUN_HOUR = 0

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Coverage area for the plot
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# 24-hour rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# GEFS timestep is 3 hours, so 24 hours = 8 timesteps
TIMESTEPS_PER_DAY = 8


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling - from original script."""
    import ast

    df = pd.read_parquet(parquet_path)

    if 'refs' in df['key'].values and len(df) <= 2:
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
        print(f"✅ Extracted {len(zstore)} entries from old format")
    else:
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

        print(f"✅ Loaded {len(zstore)} entries from new format")

    if 'version' in zstore:
        del zstore['version']

    return zstore


def validate_zarr_metadata(zstore, member_name):
    """
    Validate zarr metadata without xarray (works with Zarr v3).
    Based on test_ens13_validation_issue.py for ECMWF.
    """
    variables = set()
    for key in zstore.keys():
        if '/.zarray' in key:
            var_path = key.replace('/.zarray', '')
            variables.add(var_path)

    if not variables:
        raise ValueError(f"No valid zarr variables found in {member_name}")

    # Check if tp variable exists
    tp_found = any('tp' in var for var in variables)
    if not tp_found:
        raise ValueError(f"tp variable not found in {member_name}")

    print(f"✅ Validated {len(variables)} zarr variables in {member_name}")
    return True


def fetch_s3_byte_range_obstore(url, offset, length):
    """
    Fetch a byte range from S3 using obstore (fast Rust-based implementation).
    Based on ECMWF's aifs-etl.py implementation.
    """
    try:
        import obstore as obs
        from obstore.store import from_url

        # Parse S3 URL
        if url.startswith('s3://'):
            url_parts = url[5:].split('/', 1)
            bucket = url_parts[0]
            key = url_parts[1] if len(url_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 URL: {url}")

        # NOAA GEFS buckets are in us-east-1
        bucket_regions = {'noaa-gefs-pds': 'us-east-1'}
        region = bucket_regions.get(bucket, 'us-east-1')

        # Create S3 store (anonymous access)
        store = from_url(f"s3://{bucket}", region=region, skip_signature=True)

        # Fetch byte range
        result = obs.get_range(store, key, start=offset, end=offset + length)
        return bytes(result)

    except ImportError:
        # Fallback to fsspec if obstore not available
        return fetch_s3_byte_range_fsspec(url, offset, length)
    except Exception as e:
        # If obstore fails, fallback to fsspec
        return fetch_s3_byte_range_fsspec(url, offset, length)


def fetch_s3_byte_range_fsspec(url, offset, length, max_retries=3, retry_delay=2):
    """
    Fallback: Fetch a byte range from S3 using fsspec with retry logic.
    Based on ECMWF's aifs-etl.py implementation.
    """
    for attempt in range(max_retries):
        try:
            if url.startswith('s3://'):
                s3_path = url
            else:
                s3_path = f"s3://{url}"

            fs_s3 = fsspec.filesystem('s3', anon=True)

            with fs_s3.open(s3_path, 'rb') as f:
                f.seek(offset)
                data = f.read(length)

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"❌ Error fetching from S3 after {max_retries} attempts: {e}")
                return None

    return None


def decode_chunk_reference(chunk_ref):
    """
    Decode a chunk reference to determine its type and data.
    Based on ECMWF's aifs-etl.py implementation.
    """
    if isinstance(chunk_ref, str):
        if chunk_ref.startswith('base64:'):
            # Base64 encoded data
            import base64
            base64_str = chunk_ref[7:]
            try:
                decoded = base64.b64decode(base64_str)
                return 'base64', decoded
            except Exception as e:
                return 'unknown', chunk_ref
        else:
            return 'unknown', chunk_ref

    elif isinstance(chunk_ref, list) and len(chunk_ref) >= 3:
        url, offset, length = chunk_ref[0], chunk_ref[1], chunk_ref[2]
        if isinstance(url, str) and 's3://' in url:
            return 's3', (url, offset, length)

    return 'unknown', chunk_ref


def extract_variable_with_obstore(zstore, variable_path, use_obstore=True, spatial_slice=None):
    """
    Extract a variable directly from zarr references using obstore.
    This completely bypasses xarray and works with Zarr v3.
    Based on ECMWF's aifs-etl.py extract_variable_hybrid() function.

    Args:
        zstore: Zarr references dictionary from parquet
        variable_path: Path to variable (e.g., 'tp/accum/surface/tp')
        use_obstore: Use obstore for S3 (faster) vs fsspec
        spatial_slice: Optional dict with 'lat_start', 'lat_end', 'lon_start', 'lon_end' indices
                      to only extract a spatial subset (saves memory)

    Returns:
        numpy.ndarray: The variable data
    """
    # Step 1: Get metadata
    zarray_key = f"{variable_path}/.zarray"
    if zarray_key not in zstore:
        raise ValueError(f"Variable not found: {variable_path}")

    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]

    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'])
    chunks = tuple(metadata['chunks'])
    compressor = metadata.get('compressor', None)

    if spatial_slice:
        print(f"   📊 Extracting {variable_path}: full_shape={shape}, "
              f"subset=[{spatial_slice['lat_start']}:{spatial_slice['lat_end']}, "
              f"{spatial_slice['lon_start']}:{spatial_slice['lon_end']}], dtype={dtype}")
    else:
        print(f"   📊 Extracting {variable_path}: shape={shape}, dtype={dtype}")

    # Step 2: Collect all data chunks (optionally filtered by spatial region)
    chunks_data = {}

    for key in sorted(zstore.keys()):
        if key.startswith(variable_path + "/") and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
            # Parse chunk indices to check if we need this chunk
            chunk_idx_str = key.replace(variable_path + "/", "")
            try:
                chunk_indices = tuple(map(int, chunk_idx_str.split('.')))
            except:
                continue

            # If spatial slicing is enabled, skip chunks outside the region
            if spatial_slice and len(chunk_indices) >= 3:
                time_idx, lat_chunk_idx, lon_chunk_idx = chunk_indices[0], chunk_indices[1], chunk_indices[2]

                # Calculate which grid cells this chunk covers
                lat_chunk_start = lat_chunk_idx * chunks[1]
                lat_chunk_end = min(lat_chunk_start + chunks[1], shape[1])
                lon_chunk_start = lon_chunk_idx * chunks[2]
                lon_chunk_end = min(lon_chunk_start + chunks[2], shape[2])

                # Check if chunk overlaps with our region of interest
                if (lat_chunk_end <= spatial_slice['lat_start'] or
                    lat_chunk_start >= spatial_slice['lat_end'] or
                    lon_chunk_end <= spatial_slice['lon_start'] or
                    lon_chunk_start >= spatial_slice['lon_end']):
                    # Skip this chunk - it's outside our region
                    continue

            chunk_ref = zstore[key]
            ref_type, ref_data = decode_chunk_reference(chunk_ref)

            if ref_type == 'base64':
                # Decompress if needed
                data = ref_data
                if compressor is not None:
                    try:
                        import numcodecs
                        codec = numcodecs.get_codec(compressor)
                        data = codec.decode(data)
                    except:
                        try:
                            import blosc
                            data = blosc.decompress(data)
                        except:
                            pass
                chunks_data[key] = data

            elif ref_type == 's3':
                url, offset, length = ref_data

                # Fetch from S3
                if use_obstore:
                    data = fetch_s3_byte_range_obstore(url, offset, length)
                else:
                    data = fetch_s3_byte_range_fsspec(url, offset, length)

                if data is None:
                    print(f"⚠️ Skipping chunk {key}")
                    continue

                # Check if data is in GRIB2 format (GEFS-specific)
                if data[:4] == b'GRIB':
                    # Decode GRIB2 message using cfgrib
                    try:
                        # Write to temporary file for cfgrib
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
                            tmp.write(data)
                            tmp_path = tmp.name

                        # Open with cfgrib
                        ds = xr.open_dataset(tmp_path, engine='cfgrib')

                        # Extract the data array
                        var_names = list(ds.data_vars)
                        if var_names:
                            var_data = ds[var_names[0]].values
                            # Store the decoded numpy array (shape: 721, 1440 for GEFS)
                            chunks_data[key] = var_data
                        else:
                            print(f"⚠️ No variables found in GRIB2 chunk {key}")

                        # Clean up
                        os.unlink(tmp_path)
                        ds.close()

                    except ImportError:
                        print(f"⚠️ cfgrib not available - cannot decode GRIB2 data")
                        print(f"   Install with: pip install cfgrib")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error decoding GRIB2 chunk {key}: {e}")
                        continue

                # Not GRIB2 - handle as compressed zarr chunk
                elif compressor is not None:
                    try:
                        import numcodecs
                        codec = numcodecs.get_codec(compressor)
                        data = codec.decode(data)
                        chunks_data[key] = data
                    except Exception as e:
                        # Try alternative decompression methods
                        try:
                            import blosc
                            data = blosc.decompress(data)
                            chunks_data[key] = data
                        except:
                            try:
                                import zlib
                                data = zlib.decompress(data)
                                chunks_data[key] = data
                            except:
                                print(f"⚠️ Could not decompress chunk {key}: {e}")
                                print(f"   Compressor: {compressor}")
                                print(f"   Data size: {len(data)} bytes")
                                continue
                else:
                    # No compression, store raw data
                    chunks_data[key] = data

    if not chunks_data:
        raise ValueError(f"No data chunks found for {variable_path}")

    # Step 3: Reconstruct array
    # Determine output shape based on spatial slicing
    if spatial_slice:
        output_shape = (shape[0],
                       spatial_slice['lat_end'] - spatial_slice['lat_start'],
                       spatial_slice['lon_end'] - spatial_slice['lon_start'])
    else:
        output_shape = shape

    if len(chunks_data) == 1:
        # Single chunk - simple case
        chunk_data = list(chunks_data.values())[0]

        if isinstance(chunk_data, np.ndarray):
            array = chunk_data
        else:
            array = np.frombuffer(chunk_data, dtype=dtype)

        if array.size == np.prod(shape):
            array = array.reshape(shape)

        # Apply spatial slicing if requested
        if spatial_slice:
            array = array[:, spatial_slice['lat_start']:spatial_slice['lat_end'],
                         spatial_slice['lon_start']:spatial_slice['lon_end']]

        return array
    else:
        # Multiple chunks - need to reassemble
        # Determine actual dtype from first chunk (GRIB2 may differ from metadata)
        first_chunk = list(chunks_data.values())[0]
        if isinstance(first_chunk, np.ndarray):
            actual_dtype = first_chunk.dtype
        else:
            actual_dtype = dtype

        array = np.zeros(output_shape, dtype=actual_dtype)

        for chunk_key, chunk_data in chunks_data.items():
            # Parse chunk indices from key
            chunk_idx_str = chunk_key.replace(variable_path + "/", "")
            chunk_indices = tuple(map(int, chunk_idx_str.split('.')))

            # Convert bytes to array
            if isinstance(chunk_data, np.ndarray):
                chunk_array = chunk_data
            else:
                chunk_array = np.frombuffer(chunk_data, dtype=actual_dtype)

            # Handle GRIB2 data: comes as 2D (721, 1440) but metadata expects 3D (81, 721, 1440)
            # The chunk indices tell us which time position to fill
            if chunk_array.ndim == 2 and len(shape) == 3:
                # GRIB2 case: 2D data goes into 3D array
                time_idx = chunk_indices[0] if len(chunk_indices) > 0 else 0

                if spatial_slice:
                    # Only extract the regional subset
                    chunk_array_subset = chunk_array[spatial_slice['lat_start']:spatial_slice['lat_end'],
                                                     spatial_slice['lon_start']:spatial_slice['lon_end']]
                    array[time_idx, :, :] = chunk_array_subset
                else:
                    array[time_idx, :, :] = chunk_array
            else:
                # Standard zarr chunk reassembly
                # Calculate slice for this chunk in the FULL array
                slices = []
                for i, idx in enumerate(chunk_indices):
                    start = idx * chunks[i]
                    end = min(start + chunks[i], shape[i])
                    slices.append(slice(start, end))

                # If spatial slicing, adjust the slices for lat/lon dimensions
                if spatial_slice and len(slices) >= 3:
                    # Calculate intersection with our region
                    lat_slice_start = max(0, slices[1].start - spatial_slice['lat_start'])
                    lat_slice_end = min(output_shape[1], slices[1].stop - spatial_slice['lat_start'])
                    lon_slice_start = max(0, slices[2].start - spatial_slice['lon_start'])
                    lon_slice_end = min(output_shape[2], slices[2].stop - spatial_slice['lon_start'])

                    # Extract the relevant portion of the chunk
                    chunk_lat_start = max(0, spatial_slice['lat_start'] - slices[1].start)
                    chunk_lat_end = chunk_lat_start + (lat_slice_end - lat_slice_start)
                    chunk_lon_start = max(0, spatial_slice['lon_start'] - slices[2].start)
                    chunk_lon_end = chunk_lon_start + (lon_slice_end - lon_slice_start)

                    # Reshape chunk
                    chunk_shape = tuple(s.stop - s.start for s in slices)
                    if chunk_array.size == np.prod(chunk_shape):
                        chunk_array = chunk_array.reshape(chunk_shape)
                        chunk_array = chunk_array[:, chunk_lat_start:chunk_lat_end, chunk_lon_start:chunk_lon_end]

                    # Insert into output array
                    array[slices[0], lat_slice_start:lat_slice_end, lon_slice_start:lon_slice_end] = chunk_array
                else:
                    # No spatial slicing - standard reassembly
                    chunk_shape = tuple(s.stop - s.start for s in slices)
                    if chunk_array.size == np.prod(chunk_shape):
                        chunk_array = chunk_array.reshape(chunk_shape)
                    array[tuple(slices)] = chunk_array

        return array


def stream_single_member_precipitation(parquet_file, variable='tp', use_obstore=True):
    """
    Stream precipitation data for a single ensemble member using obstore method.
    This bypasses xarray and works with Zarr v3.
    MEMORY OPTIMIZED: Extracts only the East Africa region to avoid loading full global arrays.

    Args:
        parquet_file: Path to parquet file
        variable: Variable name (default: 'tp')
        use_obstore: Use obstore for faster S3 access (default: True)

    Returns:
        tuple: (regional_numpy, regional_xarray_coords) or (None, None) on error
    """
    member = parquet_file.stem
    member_start_time = time.time()
    print(f"\n📊 Processing {member}...")

    try:
        # Read zarr store from parquet
        zstore = read_parquet_fixed(parquet_file)

        # Validate metadata (no xarray - works with Zarr v3)
        validate_zarr_metadata(zstore, member)

        # Extract data using obstore method (bypasses xarray)
        if variable == 'tp':
            variable_path = 'tp/accum/surface/tp'
            coord_lat_path = 'tp/accum/surface/latitude'
            coord_lon_path = 'tp/accum/surface/longitude'
        else:
            print(f"⚠️ Variable {variable} not supported")
            return None, None

        # STEP 1: Extract coordinate arrays FIRST (small arrays, fast)
        # This allows us to calculate spatial indices before loading the big data array
        try:
            print(f"   📍 Extracting coordinates...")
            lat_array = extract_variable_with_obstore(zstore, coord_lat_path, use_obstore=False)
            lon_array = extract_variable_with_obstore(zstore, coord_lon_path, use_obstore=False)

            # Find indices for East Africa regional extraction
            lat_mask = (lat_array >= LAT_MIN) & (lat_array <= LAT_MAX)
            lon_mask = (lon_array >= LON_MIN) & (lon_array <= LON_MAX)

            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]

            if len(lat_indices) == 0 or len(lon_indices) == 0:
                print(f"⚠️ No data in region {LAT_MIN}-{LAT_MAX}°N, {LON_MIN}-{LON_MAX}°E")
                print(f"   Loading full global array as fallback...")
                spatial_slice = None
                regional_numpy = extract_variable_with_obstore(zstore, variable_path, use_obstore=use_obstore)
                regional_lats = lat_array
                regional_lons = lon_array
            else:
                # Create spatial slice specification
                spatial_slice = {
                    'lat_start': lat_indices[0],
                    'lat_end': lat_indices[-1] + 1,
                    'lon_start': lon_indices[0],
                    'lon_end': lon_indices[-1] + 1
                }

                print(f"   📍 Region indices: lat[{spatial_slice['lat_start']}:{spatial_slice['lat_end']}], "
                      f"lon[{spatial_slice['lon_start']}:{spatial_slice['lon_end']}]")

                # STEP 2: Extract ONLY the regional data (memory efficient!)
                regional_numpy = extract_variable_with_obstore(zstore, variable_path,
                                                               use_obstore=use_obstore,
                                                               spatial_slice=spatial_slice)

                # Extract regional coordinates
                regional_lats = lat_array[spatial_slice['lat_start']:spatial_slice['lat_end']]
                regional_lons = lon_array[spatial_slice['lon_start']:spatial_slice['lon_end']]

        except Exception as e:
            print(f"⚠️ Could not extract coordinates: {e}")
            print(f"   Loading full global array as fallback...")
            regional_numpy = extract_variable_with_obstore(zstore, variable_path, use_obstore=use_obstore)
            # Create dummy coordinates
            regional_lats = np.arange(regional_numpy.shape[1])
            regional_lons = np.arange(regional_numpy.shape[2])

        # Create a simple xarray-like object for coordinate storage
        # (for compatibility with downstream plotting functions)
        class CoordContainer:
            def __init__(self, lats, lons):
                self.latitude = type('obj', (object,), {'values': lats})()
                self.longitude = type('obj', (object,), {'values': lons})()

        regional_xarray_coords = CoordContainer(regional_lats, regional_lons)

        member_end_time = time.time()
        print(f"✅ {member} data shape: {regional_numpy.shape} | Time: {(member_end_time - member_start_time):.1f}s")

        return regional_numpy, regional_xarray_coords

    except Exception as e:
        print(f"❌ Error streaming {member}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def accumulate_24h_rainfall(precip_data):
    """
    Accumulate precipitation data into 24-hour periods.
    
    Parameters:
    -----------
    precip_data : xarray.DataArray or numpy.ndarray
        Precipitation data with time as first dimension
    
    Returns:
    --------
    accumulated : numpy.ndarray
        24-hour accumulated precipitation
    """
    if isinstance(precip_data, xr.DataArray):
        data = precip_data.values
    else:
        data = precip_data
    
    # Get time dimension
    n_timesteps = data.shape[0]
    
    # Calculate number of complete 24-hour periods
    n_days = n_timesteps // TIMESTEPS_PER_DAY
    
    # Create array to store 24-hour accumulations
    daily_shape = (n_days,) + data.shape[1:]
    daily_accumulations = np.zeros(daily_shape)
    
    for day in range(n_days):
        start_idx = day * TIMESTEPS_PER_DAY
        end_idx = (day + 1) * TIMESTEPS_PER_DAY
        
        # Sum precipitation over 24-hour period
        daily_accumulations[day] = np.sum(data[start_idx:end_idx], axis=0)
    
    return daily_accumulations


def calculate_exceedance_probabilities(ensemble_24h_data, thresholds):
    """
    Calculate probability of exceeding thresholds for ensemble data.
    
    Parameters:
    -----------
    ensemble_24h_data : dict
        Dictionary with member names as keys and 24h accumulated data as values
    thresholds : list
        List of threshold values
    
    Returns:
    --------
    probabilities : dict
        Dictionary with structure {day: {threshold: probability_array}}
    """
    # Get dimensions from first member
    first_member = list(ensemble_24h_data.values())[0]
    n_days = first_member.shape[0]
    n_members = len(ensemble_24h_data)
    
    # Initialize probabilities dictionary
    probabilities = {}
    
    for day in range(n_days):
        probabilities[day] = {}
        
        # Stack all member data for this day
        day_data = []
        for member_data in ensemble_24h_data.values():
            if member_data is not None:
                day_data.append(member_data[day])
        
        if day_data:
            day_stack = np.stack(day_data, axis=0)
            
            # Calculate probabilities for each threshold
            for threshold in thresholds:
                exceedance_count = np.sum(day_stack >= threshold, axis=0)
                probability = (exceedance_count / len(day_data)) * 100
                probabilities[day][threshold] = probability
    
    return probabilities, n_members


def load_geojson_boundaries(json_file):
    """Load GeoJSON boundaries from file."""
    try:
        with open(json_file, 'r') as f:
            geom = json.load(f)
        
        # Create GeoDataFrame
        gdf = gp.GeoDataFrame.from_features(geom)
        return gdf
    except Exception as e:
        print(f"Warning: Could not load boundary file: {e}")
        return None


def create_24h_probability_plot(probabilities, lons, lats, n_members, n_days, output_dir=None):
    """
    Create multi-panel plot showing 24h rainfall exceedance probabilities.
    
    The plot has rows for each 24-hour period and columns for each threshold.
    """
    # Create figure
    fig, axes = plt.subplots(n_days, len(THRESHOLDS_24H), 
                            figsize=(4*len(THRESHOLDS_24H), 4*n_days),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Handle single row or column
    if n_days == 1:
        axes = axes.reshape(1, -1)
    elif len(THRESHOLDS_24H) == 1:
        axes = axes.reshape(-1, 1)
    
    # Common color levels
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933', 
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']
    
    # Load boundaries
    gdf = load_geojson_boundaries(BOUNDARY_JSON)
    
    # Calculate base datetime (model run time)
    from datetime import timedelta
    base_datetime = MODEL_DATE + timedelta(hours=MODEL_RUN_HOUR)
    
    # Plot each panel
    for day in range(n_days):
        for t_idx, threshold in enumerate(THRESHOLDS_24H):
            ax = axes[day, t_idx]
            
            # Get probability data
            prob_data = probabilities[day][threshold]
            
            # Create contour plot
            cf = ax.contourf(lons, lats, prob_data, levels=levels, colors=colors,
                           transform=ccrs.PlateCarree(), extend='neither')
            
            # Add 50% contour line
            cs = ax.contour(lons, lats, prob_data, levels=[50], 
                          colors='black', linewidths=1, alpha=0.7,
                          transform=ccrs.PlateCarree())
            
            # Add boundaries from GeoJSON
            if gdf is not None:
                ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(), 
                                facecolor="none", edgecolor="black", linewidth=0.8)
            
            # Add coastlines and features (as backup)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', alpha=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            
            # Set extent
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
            
            # Calculate actual forecast dates
            start_datetime = base_datetime + timedelta(hours=day*24)
            end_datetime = base_datetime + timedelta(hours=(day+1)*24)
            
            # Convert to East Africa Time
            start_eat = start_datetime + timedelta(hours=EAT_OFFSET)
            end_eat = end_datetime + timedelta(hours=EAT_OFFSET)
            
            # Format title based on column
            max_prob = np.nanmax(prob_data)
            
            if t_idx == 0:  # First column - show full date information
                if day == 0:  # First row - show threshold header
                    title = f'>{threshold}mm\n'
                else:
                    title = ''
                
                # Add date range in EAT
                title += f'{start_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'to {end_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'Max: {max_prob:.0f}%'
                
                ax.set_title(title, fontsize=9, pad=10)
            else:
                # Other columns - just show threshold and max probability
                if day == 0:
                    ax.set_title(f'>{threshold}mm\nMax: {max_prob:.0f}%', fontsize=10)
                else:
                    ax.set_title(f'Max: {max_prob:.0f}%', fontsize=10)
            
            # Add gridlines to first column
            if t_idx == 0:
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}
    
    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)
    
    # Overall title with model run information
    model_run_str = f'{MODEL_DATE.strftime("%Y-%m-%d")} {MODEL_RUN_HOUR:02d}:00 UTC'
    model_run_eat = f'{MODEL_DATE.strftime("%Y-%m-%d")} {(MODEL_RUN_HOUR + EAT_OFFSET) % 24:02d}:00 EAT'
    
    fig.suptitle(f'GEFS 24-Hour Rainfall Exceedance Probabilities\n'
                 f'Model Run: {model_run_str} ({model_run_eat})\n'
                 f'Based on {n_members} ensemble members | Coverage: {LAT_MIN}°-{LAT_MAX}°N, {LON_MIN}°-{LON_MAX}°E',
                 fontsize=14, y=0.98)
    
    # Save figure with date and run info
    date_str = MODEL_DATE.strftime('%Y%m%d')
    run_str = f'{MODEL_RUN_HOUR:02d}'
    
    if output_dir:
        output_file = output_dir / f'probability_24h_accumulation_{date_str}_{run_str}z_all_thresholds.png'
    else:
        output_file = f'probability_24h_accumulation_{date_str}_{run_str}z_all_thresholds.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ 24-hour accumulation plot saved: {output_file}")
    plt.close()
    
    return str(output_file)


# ============================================================================
# ADDITIONAL PLOTTING FUNCTIONS MOVED FROM run_day_gefs_ensemble_full.py
# ============================================================================

def create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, variable='tp', output_dir=None, timestep_index=4):
    """Create comprehensive ensemble comparison plot using correct TP plotting approach."""
    print(f"\n🎨 Creating ensemble comparison plot for {variable}...")
    
    num_members = len(ensemble_numpy)
    if num_members == 0:
        print("❌ No ensemble data to plot")
        return None
    
    # Sort members
    members = sorted(ensemble_numpy.keys())
    
    # Create figure - 6 rows x 5 columns for 30 members
    fig, axes = plt.subplots(6, 5, figsize=(25, 30), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Use configured timestep
    timestep_idx = timestep_index
    
    # Calculate actual times for title
    forecast_hour = timestep_idx * 3
    utc_hour = (MODEL_RUN_HOUR + forecast_hour) % 24
    nairobi_hour = (utc_hour + 3) % 24
    
    print(f"📊 Plotting data for:")
    print(f"   - Timestep {timestep_idx} (T+{forecast_hour}h)")
    print(f"   - {utc_hour:02d}:00 UTC")
    print(f"   - {nairobi_hour:02d}:00 Nairobi time")
    
    # Get coordinates from first member
    first_member = members[0]
    first_xr = ensemble_xarray[first_member]
    lons = first_xr.longitude.values
    lats = first_xr.latitude.values
    
    # Check if we're plotting accumulated or instantaneous precipitation
    if forecast_hour <= 12:
        plot_title_suffix = f"T+{forecast_hour}h ({forecast_hour}-hour accumulation)"
    else:
        plot_title_suffix = f"T+{forecast_hour}h"
    
    # Plot each member
    all_data = []
    
    for idx, member in enumerate(members):
        ax = axes[idx]
        member_numpy = ensemble_numpy[member]
        
        if timestep_idx < member_numpy.shape[0]:
            # Get data for timestep
            data = member_numpy[timestep_idx]
            
            # Data is already in mm (kg/m²), no conversion needed
            plot_data = data
            
            # Store for ensemble statistics
            all_data.append(plot_data)
            
            # Calculate adaptive colorbar range
            finite_data = plot_data[np.isfinite(plot_data)]
            if len(finite_data) > 0:
                data_max = np.max(finite_data)
                # Adjust colorbar scaling for shorter forecasts
                if forecast_hour <= 6:
                    vmax = np.ceil(data_max * 1.2 / 5) * 5  # Round to nearest 5mm
                    vmax = max(5, vmax)  # Minimum 5mm for 6h
                elif forecast_hour <= 12:
                    vmax = np.ceil(data_max * 1.2 / 10) * 10  # Round to nearest 10mm
                    vmax = max(10, vmax)  # Minimum 10mm for 12h
                else:
                    vmax = np.ceil(data_max * 1.2 / 10) * 10
                    vmax = max(10, vmax)
            else:
                vmax = 10 if forecast_hour <= 6 else 20
            
            # Create plot
            levels = np.linspace(0, vmax, 11)
            im = ax.contourf(lons, lats, plot_data,
                           levels=levels, cmap='Blues',
                           transform=ccrs.PlateCarree(),
                           extend='max')
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            
            # Set extent
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
            
            # Title
            ax.set_title(f'{member}\nMax: {data_max:.1f}mm', fontsize=10)
            
        else:
            ax.text(0.5, 0.5, f'{member}\nNo data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{member}')
    
    # Hide unused subplots
    for idx in range(len(members), len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Total Precipitation (mm)', rotation=270, labelpad=20)
    
    # UPDATED TITLE with time information
    date_str = MODEL_DATE.strftime('%Y%m%d')
    plt.suptitle(f'GEFS Ensemble Total Precipitation Forecast - East Africa\n'
                f'{date_str} {MODEL_RUN_HOUR:02d}Z Run - {plot_title_suffix}\n'
                f'Valid: {utc_hour:02d}:00 UTC ({nairobi_hour:02d}:00 Nairobi)', 
                fontsize=16, y=0.98)
    
    # Save with timestep in filename
    if output_dir:
        output_file = output_dir / f'ensemble_all_members_comparison_T{forecast_hour:03d}.png'
    else:
        output_file = f'gefs_ensemble_{date_str}_{MODEL_RUN_HOUR:02d}z_all_members_comparison_T{forecast_hour:03d}.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Ensemble comparison plot saved: {output_file}")
    plt.close()
    
    return str(output_file)


def plot_multiple_timesteps(ensemble_numpy, ensemble_xarray, timesteps_to_plot, output_dir=None):
    """
    Plot ensemble data for multiple timesteps
    
    Parameters:
    -----------
    timesteps_to_plot : list of tuples
        Each tuple contains (timestep_index, description)
        Example: [(2, "6h"), (4, "12h"), (8, "24h")]
    """
    for timestep_idx, description in timesteps_to_plot:
        print(f"\n{'='*60}")
        print(f"Creating plot for {description} forecast (timestep {timestep_idx})")
        print(f"{'='*60}")
        
        # Create plot
        create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, 'tp', output_dir, timestep_idx)


def calculate_ensemble_probabilities(ensemble_numpy, thresholds=[5, 10, 15, 20, 25], timestep_idx=4):
    """
    Calculate empirical probabilities for precipitation exceeding thresholds.
    
    Parameters:
    -----------
    ensemble_numpy : dict
        Dictionary with member names as keys and numpy arrays as values
    thresholds : list
        List of precipitation thresholds in mm
    timestep_idx : int
        Time step index to analyze (default: 4 = 12h forecast)
    
    Returns:
    --------
    probabilities : dict
        Dictionary with thresholds as keys and probability arrays as values
    """
    print(f"\n📊 Calculating ensemble probabilities for {len(thresholds)} thresholds...")
    
    # Get ensemble data for the specified timestep
    members = sorted(ensemble_numpy.keys())
    n_members = len(members)
    
    if n_members == 0:
        print("❌ No ensemble data available")
        return None
    
    # Get data shape from first member
    first_data = ensemble_numpy[members[0]][timestep_idx]
    data_shape = first_data.shape
    
    # Stack all member data for the timestep
    ensemble_stack = np.zeros((n_members, *data_shape))
    
    for i, member in enumerate(members):
        member_data = ensemble_numpy[member]
        if timestep_idx < member_data.shape[0]:
            ensemble_stack[i] = member_data[timestep_idx]
        else:
            ensemble_stack[i] = np.nan
    
    # Calculate probabilities for each threshold
    probabilities = {}
    
    for threshold in thresholds:
        # Count how many members exceed threshold at each grid point
        exceedance_count = np.sum(ensemble_stack >= threshold, axis=0)
        
        # Convert to probability (percentage)
        probability = (exceedance_count / n_members) * 100
        
        probabilities[threshold] = probability
        
        # Print statistics
        max_prob = np.nanmax(probability)
        area_above_50 = np.sum(probability >= 50)
        print(f"   Threshold {threshold}mm: Max probability = {max_prob:.1f}%, "
              f"Grid points with P>=50% = {area_above_50}")
    
    return probabilities, n_members


def plot_probability_map(probability, threshold, lons, lats, timestep_idx, n_members, output_dir=None):
    """
    Plot probability map for a single threshold with Nairobi location.
    
    Parameters:
    -----------
    probability : numpy array
        Probability values (0-100%)
    threshold : float
        Precipitation threshold in mm
    lons, lats : numpy arrays
        Longitude and latitude coordinates
    """
    # Nairobi coordinates
    NAIROBI_LAT = -1.2921
    NAIROBI_LON = 36.8219
    
    # Calculate forecast hour and times
    forecast_hour = timestep_idx * 3
    utc_hour = (MODEL_RUN_HOUR + forecast_hour) % 24
    nairobi_hour = (utc_hour + 3) % 24
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Define probability levels and colors
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933', 
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']
    
    # Create contour plot
    cf = ax.contourf(lons, lats, probability, levels=levels, colors=colors,
                     transform=ccrs.PlateCarree(), extend='neither')
    
    # Add contour lines at key probability levels
    cs = ax.contour(lons, lats, probability, levels=[25, 50, 75], 
                    colors='black', linewidths=0.5, alpha=0.5,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d%%')
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.1, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')
    
    # Add rivers for context
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5, color='blue')
    
    # Set extent to East Africa
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    
    # Add Nairobi marker
    ax.plot(NAIROBI_LON, NAIROBI_LAT, marker='*', markersize=20, 
            color='red', markeredgecolor='darkred', markeredgewidth=1,
            transform=ccrs.PlateCarree(), zorder=10)
    
    # Add Nairobi label with background
    ax.text(NAIROBI_LON + 0.3, NAIROBI_LAT + 0.3, 'Nairobi', 
            transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='darkred', alpha=0.8),
            zorder=11)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label(f'Probability of exceeding {threshold}mm (%)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # Title with comprehensive information
    date_str = MODEL_DATE.strftime('%Y%m%d')
    plt.title(f'GEFS Ensemble Probability: Precipitation > {threshold}mm\n'
              f'{date_str} {MODEL_RUN_HOUR:02d}Z Run - T+{forecast_hour}h '
              f'({utc_hour:02d}:00 UTC / {nairobi_hour:02d}:00 EAT)\n'
              f'Based on {n_members} ensemble members',
              fontsize=14, pad=20)
    
    # Add text box with key information
    textstr = f'Max Probability: {np.nanmax(probability):.1f}%\n'
    textstr += f'Area with P≥50%: {np.sum(probability >= 50)} grid points\n'
    textstr += f'Area with P≥75%: {np.sum(probability >= 75)} grid points'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save figure
    if output_dir:
        output_file = output_dir / f'probability_exceeding_{threshold:02d}mm_T{forecast_hour:03d}.png'
    else:
        output_file = f'gefs_probability_{date_str}_{MODEL_RUN_HOUR:02d}z_{threshold:02d}mm_T{forecast_hour:03d}.png'
    
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Probability map saved: {output_file}")
    plt.close()
    
    return str(output_file)


def create_probability_summary_plot(probabilities, thresholds, lons, lats, timestep_idx, n_members, output_dir=None):
    """
    Create a multi-panel plot showing probabilities for all thresholds.
    """
    # Calculate times
    forecast_hour = timestep_idx * 3
    utc_hour = (MODEL_RUN_HOUR + forecast_hour) % 24
    nairobi_hour = (utc_hour + 3) % 24
    
    # Nairobi coordinates
    NAIROBI_LAT = -1.2921
    NAIROBI_LON = 36.8219
    
    # Create figure with subplots
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Common color levels for all subplots
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933', 
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']
    
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        probability = probabilities[threshold]
        
        # Create contour plot
        cf = ax.contourf(lons, lats, probability, levels=levels, colors=colors,
                         transform=ccrs.PlateCarree(), extend='neither')
        
        # Add 50% contour line
        cs = ax.contour(lons, lats, probability, levels=[50], 
                       colors='black', linewidths=1, alpha=0.7,
                       transform=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='navy')
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='gray')
        ax.add_feature(cfeature.LAND, alpha=0.1)
        
        # Set extent
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
        
        # Add Nairobi marker
        ax.plot(NAIROBI_LON, NAIROBI_LAT, marker='*', markersize=10, 
                color='red', markeredgecolor='darkred', markeredgewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=10)
        
        # Add title for each subplot
        max_prob = np.nanmax(probability)
        ax.set_title(f'P(TP > {threshold}mm) - Max: {max_prob:.0f}%', fontsize=12)
        
        # Add gridlines for first subplot only
        if i == 0:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
    
    # Hide the 6th subplot if we have 5 thresholds
    if n_thresholds < 6:
        axes[5].set_visible(False)
    
    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)
    
    # Overall title
    date_str = MODEL_DATE.strftime('%Y%m%d')
    fig.suptitle(f'GEFS Ensemble Exceedance Probabilities - East Africa\n'
                 f'{date_str} {MODEL_RUN_HOUR:02d}Z Run - T+{forecast_hour}h '
                 f'({utc_hour:02d}:00 UTC / {nairobi_hour:02d}:00 EAT)\n'
                 f'{n_members} ensemble members',
                 fontsize=16, y=0.98)
    
    # Add legend for Nairobi
    nairobi_marker = mpatches.Patch(color='red', label='★ Nairobi')
    plt.figlegend(handles=[nairobi_marker], loc='lower right', 
                 bbox_to_anchor=(0.9, 0.05), fontsize=12)
    
    # Save figure
    if output_dir:
        output_file = output_dir / f'probability_summary_all_thresholds_T{forecast_hour:03d}.png'
    else:
        output_file = f'gefs_probability_summary_{date_str}_{MODEL_RUN_HOUR:02d}z_T{forecast_hour:03d}.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Probability summary plot saved: {output_file}")
    plt.close()
    
    return str(output_file)


def process_ensemble_probabilities(ensemble_numpy, ensemble_xarray, timestep_idx=4, 
                                  thresholds=[5, 10, 15, 20, 25], output_dir=None):
    """
    Main function to process and plot ensemble probabilities.
    
    Parameters:
    -----------
    ensemble_numpy : dict
        Dictionary of ensemble member data
    ensemble_xarray : dict
        Dictionary of xarray datasets
    timestep_idx : int
        Timestep index to analyze (default: 4 = 12h)
    thresholds : list
        List of precipitation thresholds in mm
    output_dir : Path
        Output directory for plots
    """
    print(f"\n{'='*80}")
    print(f"Processing Ensemble Exceedance Probabilities")
    print(f"{'='*80}")
    
    if len(ensemble_numpy) == 0:
        print("❌ No ensemble data available")
        return None
    
    # Get coordinates from first member
    first_member = sorted(ensemble_xarray.keys())[0]
    first_xr = ensemble_xarray[first_member]
    lons = first_xr.longitude.values
    lats = first_xr.latitude.values
    
    # Calculate probabilities
    probabilities, n_members = calculate_ensemble_probabilities(
        ensemble_numpy, thresholds, timestep_idx
    )
    
    if probabilities is None:
        return None
    
    # Create individual plots for each threshold
    print(f"\n📊 Creating individual probability maps...")
    plot_files = []
    
    for threshold in thresholds:
        plot_file = plot_probability_map(
            probabilities[threshold], threshold, lons, lats, timestep_idx, n_members, output_dir
        )
        plot_files.append(plot_file)
    
    # Create summary plot
    print(f"\n📊 Creating summary probability plot...")
    summary_file = create_probability_summary_plot(
        probabilities, thresholds, lons, lats, timestep_idx, n_members, output_dir
    )
    
    print(f"\n✅ Probability processing complete!")
    print(f"   - Individual plots: {len(plot_files)}")
    print(f"   - Summary plot: {summary_file}")
    
    return probabilities, plot_files, summary_file


# ============================================================================
# TESTING ROUTINES - SWITCHED OFF AS REQUESTED
# ============================================================================

def run_additional_plotting_tests(ensemble_numpy, ensemble_xarray, output_dir=None):
    """
    Test routine for additional plotting functions moved from run_day_gefs_ensemble_full.py.
    This routine is switched off by default as requested.
    
    To enable testing, call this function from main() with enable_testing=True
    """
    print("\n" + "="*80)
    print("TESTING ADDITIONAL PLOTTING ROUTINES (SWITCHED OFF)")
    print("="*80)
    print("This testing routine is currently disabled.")
    print("To enable, modify the main() function and set enable_testing=True")
    
    # Uncomment the code below to enable testing
    """
    # Test ensemble comparison plot
    if len(ensemble_numpy) > 0:
        print("\n📊 Testing ensemble comparison plot...")
        create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, 'tp', output_dir, timestep_index=4)
    
    # Test multiple timesteps
    timesteps_to_plot = [
        (2, "6h"),   # 06:00 UTC = 09:00 Nairobi
        (4, "12h"),  # 12:00 UTC = 15:00 Nairobi  
        (6, "18h"),  # 18:00 UTC = 21:00 Nairobi
    ]
    print("\n📊 Testing multiple timesteps...")
    plot_multiple_timesteps(ensemble_numpy, ensemble_xarray, timesteps_to_plot, output_dir)
    
    # Test ensemble probabilities
    if len(ensemble_numpy) > 0:
        print("\n📊 Testing ensemble probabilities...")
        thresholds = [5, 10, 15, 20, 25]
        timestep_idx = 4
        
        probabilities, prob_files, summary_file = process_ensemble_probabilities(
            ensemble_numpy, ensemble_xarray, 
            timestep_idx=timestep_idx,
            thresholds=thresholds,
            output_dir=output_dir
        )
    
    print("\n✅ Testing routines completed!")
    """
    
    return None


def main():
    """Main processing function."""
    print("="*80)
    print("GEFS 24-Hour Rainfall Accumulation Processing V2")
    print("MEMORY OPTIMIZED: Regional extraction only")
    print("="*80)

    # TESTING CONTROL - Set to True to enable additional plotting tests
    enable_testing = False  # SWITCHED OFF as requested

    # Display model run information
    print(f"\n📅 Model Run Information:")
    print(f"   Date: {MODEL_DATE.strftime('%Y-%m-%d')}")
    print(f"   Run Hour: {MODEL_RUN_HOUR:02d}:00 UTC ({(MODEL_RUN_HOUR + EAT_OFFSET) % 24:02d}:00 EAT)")
    print(f"   Forecast starts: {MODEL_DATE.strftime('%Y-%m-%d')} {MODEL_RUN_HOUR:02d}:00 UTC")
    print(f"\n🌍 Spatial Domain: East Africa")
    print(f"   Latitude: {LAT_MIN}° to {LAT_MAX}°N")
    print(f"   Longitude: {LON_MIN}° to {LON_MAX}°E")
    print(f"   (Only regional data loaded - optimized for memory efficiency)")
    
    start_time = time.time()
    
    # Check if parquet directory exists
    if not PARQUET_DIR.exists():
        print(f"❌ Error: Directory {PARQUET_DIR} not found!")
        return False
    
    # Get list of parquet files
    parquet_files = sorted(PARQUET_DIR.glob("gep*.par"))
    print(f"📁 Found {len(parquet_files)} ensemble member files")
    
    # Use the same directory as input PAR files for output
    output_dir = PARQUET_DIR
    print(f"📝 Output will be saved in: {output_dir}")
    
    # Load ensemble data
    print("\n🌧️ Loading ensemble precipitation data...")
    load_start_time = time.time()
    ensemble_numpy = {}
    ensemble_xarray = {}
    
    for pf in parquet_files:  # Process all members
        numpy_data, xarray_data = stream_single_member_precipitation(pf)
        if numpy_data is not None:
            member_name = pf.stem
            ensemble_numpy[member_name] = numpy_data
            ensemble_xarray[member_name] = xarray_data
    
    load_end_time = time.time()
    print(f"\n✅ Successfully loaded {len(ensemble_numpy)} members")
    print(f"⏱️  Loading time: {(load_end_time - load_start_time):.1f} seconds")
    
    if len(ensemble_numpy) == 0:
        print("❌ No data loaded successfully!")
        return False
    
    # Get coordinates from first member
    first_member_xr = list(ensemble_xarray.values())[0]
    lons = first_member_xr.longitude.values
    lats = first_member_xr.latitude.values
    
    # Process accumulations
    print("\n📊 Processing 24-hour accumulations...")
    accum_start_time = time.time()
    ensemble_24h = {}
    
    for member, data in ensemble_numpy.items():
        # GEFS provides 3-hourly precipitation amounts (already incremental, not cumulative)
        # Timestep 0 is initial condition (NaN), actual forecast starts from timestep 1
        # We need to sum 8 consecutive 3-hour timesteps to get 24-hour totals
        n_timesteps = data.shape[0]
        
        # Skip timestep 0 (initial condition) and work with forecast timesteps 1-80
        forecast_timesteps = n_timesteps - 1  # 80 timesteps available for forecast
        n_days = forecast_timesteps // TIMESTEPS_PER_DAY  # 10 complete days
        
        # Extract forecast data (skip timestep 0 which is NaN)
        forecast_data = data[1:]  # timesteps 1-80 contain 3-hourly precipitation
        
        # Sum 8 consecutive forecast timesteps to get 24-hour accumulations
        daily_accum = np.zeros((n_days,) + data.shape[1:])
        
        for day in range(n_days):
            start_idx = day * TIMESTEPS_PER_DAY
            end_idx = (day + 1) * TIMESTEPS_PER_DAY
            
            if end_idx <= forecast_timesteps:
                # Sum the 8 timesteps (each 3-hour period) to get 24-hour total
                daily_accum[day] = np.sum(forecast_data[start_idx:end_idx], axis=0)
        
        ensemble_24h[member] = daily_accum
        print(f"  {member}: {n_days} days processed")
    
    accum_end_time = time.time()
    print(f"⏱️  Accumulation processing time: {(accum_end_time - accum_start_time):.1f} seconds")
    
    # Calculate exceedance probabilities
    print("\n📈 Calculating exceedance probabilities...")
    prob_start_time = time.time()
    probabilities, n_members = calculate_exceedance_probabilities(ensemble_24h, THRESHOLDS_24H)
    
    # Get number of days
    n_days = len(probabilities)
    prob_end_time = time.time()
    print(f"  Days: {n_days}")
    print(f"  Thresholds: {THRESHOLDS_24H} mm")
    print(f"⏱️  Probability calculation time: {(prob_end_time - prob_start_time):.1f} seconds")
    
    # Create visualization
    print("\n🎨 Creating 24-hour accumulation plots...")
    plot_start_time = time.time()
    plot_file = create_24h_probability_plot(probabilities, lons, lats, n_members, n_days, output_dir)
    plot_end_time = time.time()
    print(f"⏱️  Plotting time: {(plot_end_time - plot_start_time):.1f} seconds")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    for day in range(n_days):
        print(f"\n  Day {day+1} (Hours {day*24}-{(day+1)*24}):")
        for threshold in THRESHOLDS_24H:
            max_prob = np.nanmax(probabilities[day][threshold])
            area_50 = np.sum(probabilities[day][threshold] >= 50)
            print(f"    >{threshold:3d}mm: Max probability = {max_prob:5.1f}%, "
                  f"Grid points with P≥50% = {area_50:4d}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ Processing completed successfully!")
    print("="*80)
    
    # Detailed timing breakdown
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"   📊 Data Loading:           {(load_end_time - load_start_time):.1f} seconds ({(load_end_time - load_start_time)/total_time*100:.1f}%)")
    print(f"   🔄 24h Accumulation:       {(accum_end_time - accum_start_time):.1f} seconds ({(accum_end_time - accum_start_time)/total_time*100:.1f}%)")
    print(f"   📈 Probability Calculation: {(prob_end_time - prob_start_time):.1f} seconds ({(prob_end_time - prob_start_time)/total_time*100:.1f}%)")
    print(f"   🎨 Plot Generation:        {(plot_end_time - plot_start_time):.1f} seconds ({(plot_end_time - plot_start_time)/total_time*100:.1f}%)")
    print(f"   ⏱️  TOTAL TIME:             {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\n📁 Output directory: {output_dir}")
    print("="*80)
    
    # TESTING ROUTINES - SWITCHED OFF AS REQUESTED
    if enable_testing:
        print("\n🌡️ Running additional plotting tests...")
        run_additional_plotting_tests(ensemble_numpy, ensemble_xarray, output_dir)
    else:
        print("\n📝 Additional plotting tests are DISABLED.")
        print("     To enable, set enable_testing=True in main() function.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
