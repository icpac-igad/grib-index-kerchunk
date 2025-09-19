#!/usr/bin/env python3
"""
ECMWF Ensemble Parquet Creator - Efficient Version
Follows the efficient pattern from test_run_ecmwf_step1_scangrib.py:
- Scan each GRIB file once to extract all ensemble members
- Process all members together using fixed_ensemble_grib_tree
- Create one comprehensive parquet file with all ensemble data
- Then extract individual member parquet files if needed
"""

import fsspec
import pandas as pd
import numpy as np
import xarray as xr
import json
import copy
import os
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from kerchunk.grib2 import scan_grib
from kerchunk._grib_idx import strip_datavar_chunks

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for ECMWF data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'


def log_message(message: str, level: str = "INFO"):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def log_checkpoint(message: str, start_time: float = None):
    """Log a checkpoint with timestamp and elapsed time."""
    current_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if start_time is not None:
        elapsed = current_time - start_time
        print(f"[{timestamp}] {message} (Elapsed: {elapsed:.2f}s)")
    else:
        print(f"[{timestamp}] {message}")

    return current_time


# Import the efficient functions from existing files
try:
    from eutils import fixed_ensemble_grib_tree, ecmwf_filter_scan_grib
    log_message("Successfully imported efficient functions from eutils.py")
except ImportError:
    try:
        from test_run_ecmwf_step1_scangrib import fixed_ensemble_grib_tree, ecmwf_filter_scan_grib
        log_message("Successfully imported efficient functions from test file")
    except ImportError:
        log_message("Error: Could not import required functions. Please ensure eutils.py or test_run_ecmwf_step1_scangrib.py is available", "ERROR")
        exit(1)


def process_ecmwf_files_efficiently(
    ecmwf_files: List[str],
    date_str: str,
    run: str,
    output_dir: Path
) -> Dict:
    """
    Process ECMWF files efficiently by scanning each file once and extracting all ensemble members.

    Parameters:
    - ecmwf_files: List of ECMWF GRIB file URLs
    - date_str: Date string
    - run: Run hour
    - output_dir: Output directory path

    Returns:
    - Dictionary with processing results
    """
    print("="*80)
    print("ECMWF Efficient Ensemble Processing")
    print("="*80)

    log_message(f"Processing {len(ecmwf_files)} ECMWF files for date {date_str}")

    # Step 1: Process files and collect groups (scan each file once)
    file_processing_start = log_checkpoint("Starting file processing")
    all_groups = []

    for i, eurl in enumerate(ecmwf_files, 1):
        try:
            file_start = log_checkpoint(f"Processing file {i}/{len(ecmwf_files)}: {eurl.split('/')[-1]}")

            # This scans the GRIB file once and extracts all ensemble members
            groups, idx_mapping = ecmwf_filter_scan_grib(eurl)
            all_groups.extend(groups)

            log_checkpoint(f"File {i} completed, found {len(groups)} groups", file_start)

        except Exception as e:
            log_message(f"Error processing {eurl}: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    log_checkpoint(f"File processing completed. Total groups: {len(all_groups)}", file_processing_start)

    if not all_groups:
        raise ValueError("No valid groups were found")

    # Step 2: Build ensemble tree with all members together
    ensemble_start = log_checkpoint("Starting fixed_ensemble_grib_tree processing")
    remote_options = {"anon": True}  # Anonymous S3 access

    ensemble_tree = fixed_ensemble_grib_tree(
        all_groups,
        remote_options=remote_options,
        debug_output=True
    )

    log_checkpoint(f"Ensemble tree built with {len(ensemble_tree['refs'])} references", ensemble_start)

    # Step 3: Create deflated store for parquet
    deflate_start = log_checkpoint("Creating deflated store for parquet")
    deflated_ensemble_tree = copy.deepcopy(ensemble_tree)
    strip_datavar_chunks(deflated_ensemble_tree)
    log_checkpoint("Deflated store created", deflate_start)

    # Step 4: Save comprehensive parquet file
    parquet_start = log_checkpoint("Saving comprehensive ensemble parquet file")
    comprehensive_parquet = output_dir / f"ecmwf_{date_str}_{run}z_ensemble_all.parquet"
    create_parquet_file_fixed(deflated_ensemble_tree['refs'], str(comprehensive_parquet))
    log_checkpoint(f"Comprehensive parquet saved: {comprehensive_parquet}", parquet_start)

    # Step 5: Validate with datatree
    validation_start = log_checkpoint("Validating with xarray datatree")
    try:
        fs = fsspec.filesystem("reference", fo=ensemble_tree, remote_protocol='s3', remote_options={'anon': True})
        mapper = fs.get_mapper("")

        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        log_checkpoint("DataTree validation successful", validation_start)

        # Log available variables
        variables = list(dt.keys())
        log_message(f"Available variables: {len(variables)}")
        log_message(f"Variables: {variables[:10]}{'...' if len(variables) > 10 else ''}")

    except Exception as e:
        log_message(f"DataTree validation failed: {e}", "WARNING")

    return {
        'ensemble_tree': ensemble_tree,
        'deflated_tree': deflated_ensemble_tree,
        'comprehensive_parquet': str(comprehensive_parquet),
        'total_groups': len(all_groups),
        'total_refs': len(ensemble_tree['refs'])
    }


def extract_individual_member_parquets(
    ensemble_tree: Dict,
    output_dir: Path,
    target_members: List[int] = None
) -> Dict:
    """
    Extract individual ensemble member parquet files from the comprehensive ensemble tree.

    Parameters:
    - ensemble_tree: Complete ensemble tree dictionary
    - output_dir: Output directory
    - target_members: List of ensemble members to extract (default: control + first 5)

    Returns:
    - Dictionary with extraction results
    """
    if target_members is None:
        target_members = [-1, 1, 2, 3, 4, 5]  # Control + first 5 members

    log_message(f"Extracting individual parquet files for {len(target_members)} members")

    extraction_results = {}

    for member in target_members:
        member_name = "control" if member == -1 else f"ens{member:02d}"

        try:
            extract_start = log_checkpoint(f"Extracting {member_name}")

            # Extract member-specific references from the ensemble tree
            member_refs = extract_member_refs(ensemble_tree['refs'], member)

            if not member_refs:
                log_message(f"No references found for {member_name}", "WARNING")
                continue

            # Create member-specific parquet file
            member_parquet = output_dir / f"{member_name}.par"
            create_parquet_file_fixed(member_refs, str(member_parquet))

            # Validate member file
            try:
                fs = fsspec.filesystem("reference", fo={'refs': member_refs, 'version': 1},
                                     remote_protocol='s3', remote_options={'anon': True})
                mapper = fs.get_mapper("")
                dt_member = xr.open_datatree(mapper, engine="zarr", consolidated=False)

                extraction_results[member_name] = {
                    'success': True,
                    'parquet_file': str(member_parquet),
                    'refs_count': len(member_refs),
                    'variables': list(dt_member.keys())
                }

                log_checkpoint(f"{member_name} extraction completed successfully", extract_start)

            except Exception as e:
                log_message(f"Validation failed for {member_name}: {e}", "WARNING")
                extraction_results[member_name] = {
                    'success': False,
                    'parquet_file': str(member_parquet),
                    'refs_count': len(member_refs),
                    'error': str(e)
                }

        except Exception as e:
            log_message(f"Error extracting {member_name}: {e}", "ERROR")
            extraction_results[member_name] = {
                'success': False,
                'error': str(e)
            }

    return extraction_results


def extract_member_refs(all_refs: Dict, target_member: int) -> Dict:
    """
    Extract references for a specific ensemble member from the complete ensemble tree.

    Parameters:
    - all_refs: Complete ensemble references dictionary
    - target_member: Ensemble member number (-1 for control, 1-50 for perturbed)

    Returns:
    - Dictionary with member-specific references
    """
    member_refs = {}

    # Map target_member to the actual index in the number dimension
    # Control member (-1) maps to index 0, perturbed members (1-50) map to indices 1-50
    member_index = 0 if target_member == -1 else target_member

    for key, value in all_refs.items():
        if key.endswith('/.zgroup'):
            # Copy group definitions as-is
            member_refs[key] = value

        elif key.endswith('/.zattrs'):
            # Process attributes, removing number dimension where needed
            attrs = json.loads(value) if isinstance(value, str) else value

            # Check if this is a variable with number dimension
            if '_ARRAY_DIMENSIONS' in attrs and 'number' in attrs['_ARRAY_DIMENSIONS']:
                # Remove number from dimensions for individual member
                new_dims = [d for d in attrs['_ARRAY_DIMENSIONS'] if d != 'number']
                attrs['_ARRAY_DIMENSIONS'] = new_dims

            member_refs[key] = json.dumps(attrs) if not isinstance(value, str) else json.dumps(attrs)

        elif key.endswith('/.zarray'):
            # Process array metadata, adjusting for ensemble member extraction
            zarray = json.loads(value) if isinstance(value, str) else value

            # Check if this array has the number dimension
            # The number dimension is typically the 3rd dimension (index 2) in ECMWF data
            # Structure is often: [time, step, number, latitude, longitude]
            if 'shape' in zarray and len(zarray['shape']) >= 3:
                # Check if this looks like it has ensemble dimension
                # (51 members = control + 50 perturbed)
                has_ensemble = any(dim == 51 for dim in zarray['shape'])

                if has_ensemble:
                    # Find which dimension is the ensemble dimension
                    ensemble_dim_idx = None
                    for idx, dim in enumerate(zarray['shape']):
                        if dim == 51:
                            ensemble_dim_idx = idx
                            break

                    if ensemble_dim_idx is not None:
                        # Remove the ensemble dimension from shape
                        new_shape = list(zarray['shape'])
                        del new_shape[ensemble_dim_idx]
                        zarray['shape'] = new_shape

                        # Adjust chunks if present
                        if 'chunks' in zarray:
                            new_chunks = list(zarray['chunks'])
                            del new_chunks[ensemble_dim_idx]
                            zarray['chunks'] = new_chunks

            member_refs[key] = json.dumps(zarray) if not isinstance(value, str) else json.dumps(zarray)

        elif key == 'number/.zarray':
            # Skip the number coordinate array for individual members
            continue

        elif key == 'number/0':
            # Skip the number coordinate data for individual members
            continue

        else:
            # Process data chunks
            # Data chunks for variables with ensemble dimension need special handling
            if isinstance(value, (list, tuple)) and len(value) == 3:
                # This is a chunk reference [url, offset, size]
                # Check if this chunk belongs to our target member

                # Parse the key to determine which chunk this is
                # Format is typically: variable_name/chunk_indices
                parts = key.split('/')
                if len(parts) == 2 and '.' in parts[1]:
                    # Parse chunk indices
                    indices = parts[1].split('.')

                    # If we have a 5D variable (time, step, number, lat, lon)
                    # the 3rd index (index 2) would be the ensemble member
                    if len(indices) >= 3:
                        try:
                            ensemble_chunk_idx = int(indices[2])
                            # Only include this chunk if it matches our target member
                            if ensemble_chunk_idx == member_index:
                                # Adjust the key to remove the ensemble dimension
                                new_indices = indices[:2] + indices[3:]  # Skip the ensemble index
                                new_key = f"{parts[0]}/{'.'.join(new_indices)}"
                                member_refs[new_key] = value
                        except (ValueError, IndexError):
                            # If parsing fails, check variable name for coordinate vars
                            if parts[0] not in ['number', 'valid_time']:
                                member_refs[key] = value
                    else:
                        # No ensemble dimension in this chunk
                        if parts[0] != 'number':
                            member_refs[key] = value
                else:
                    # Not a chunk reference or special format, copy if not number-related
                    if 'number' not in key:
                        member_refs[key] = value
            else:
                # Other types of values, copy if not number-related
                if 'number' not in key:
                    member_refs[key] = value

    # Update root attributes to indicate this is a single member
    if '.zattrs' in member_refs:
        root_attrs = json.loads(member_refs['.zattrs'])
        root_attrs['extracted_ensemble_member'] = target_member
        member_refs['.zattrs'] = json.dumps(root_attrs)

    return member_refs


def create_parquet_file_fixed(zstore: dict, output_parquet_file: str):
    """Save zarr store dictionary as parquet file with proper encoding."""
    data = []

    for key, value in zstore.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        elif isinstance(value, (int, float, np.integer, np.floating)):
            encoded_value = str(value).encode('utf-8')
        else:
            encoded_value = str(value).encode('utf-8')

        data.append((key, encoded_value))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_parquet_file)
    log_message(f"Saved parquet file: {output_parquet_file} ({len(df)} rows)")


def save_processing_metadata(results: Dict, extraction_results: Dict, output_dir: Path, date_str: str, run: str):
    """Save processing metadata and results."""
    metadata = {
        'date_str': date_str,
        'run': run,
        'processing_approach': 'efficient_single_scan',
        'total_groups_processed': results['total_groups'],
        'total_refs_in_tree': results['total_refs'],
        'comprehensive_parquet': results['comprehensive_parquet'],
        'individual_members': {}
    }

    # Add individual member results
    for member_name, member_result in extraction_results.items():
        metadata['individual_members'][member_name] = {
            'success': member_result.get('success', False),
            'parquet_file': member_result.get('parquet_file', ''),
            'refs_count': member_result.get('refs_count', 0),
            'variables_count': len(member_result.get('variables', []))
        }

    metadata_file = output_dir / "processing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    log_message(f"Processing metadata saved to {metadata_file}")


def main():
    """Main function for efficient ECMWF ensemble processing."""
    print("="*80)
    print("ECMWF Efficient Ensemble Parquet Creator")
    print("="*80)

    start_time = time.time()

    # Configuration
    date_str = '20250628'
    run = '18'
    target_members = [-1, 1, 2, 3, 4, 5]  + list(range(6, 51))

    # Create output directory
    output_dir = Path(f"ecmwf_{date_str}_{run}_efficient")
    output_dir.mkdir(exist_ok=True)
    log_message(f"Output directory: {output_dir}")

    # Define ECMWF files - just the first few timesteps for efficiency
    ecmwf_files = [
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-0h-enfo-ef.grib2",
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-3h-enfo-ef.grib2"
    ]

    # Check file availability
    fs = fsspec.filesystem("s3", anon=True)
    available_files = []
    for f in ecmwf_files:
        if fs.exists(f):
            available_files.append(f)
        else:
            log_message(f"File not found: {f}", "WARNING")

    if not available_files:
        log_message("No valid ECMWF files found", "ERROR")
        return False

    log_message(f"Processing {len(available_files)} available ECMWF files")
    log_message(f"Target ensemble members: {target_members}")

    try:
        # Step 1: Efficient processing of all files (scan once per file)
        results = process_ecmwf_files_efficiently(available_files, date_str, run, output_dir)

        # Step 2: Extract individual member parquet files
        log_message("\n" + "="*50)
        log_message("Extracting Individual Member Parquet Files")
        log_message("="*50)

        extraction_results = extract_individual_member_parquets(
            results['ensemble_tree'], output_dir, target_members
        )

        # Step 3: Save metadata
        save_processing_metadata(results, extraction_results, output_dir, date_str, run)

        # Summary
        total_time = time.time() - start_time
        successful_members = sum(1 for r in extraction_results.values() if r.get('success', False))

        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)

        print(f"\n📊 Efficiency Comparison:")
        print(f"   Old approach: {len(target_members)} × {len(available_files)} = {len(target_members) * len(available_files)} file scans")
        print(f"   New approach: {len(available_files)} file scans (one per file)")
        print(f"   Efficiency gain: {(len(target_members) * len(available_files)) / len(available_files):.1f}x faster")

        print(f"\n📈 Results:")
        print(f"   Total processing time: {total_time/60:.1f} minutes")
        print(f"   Files processed: {len(available_files)}")
        print(f"   Total groups extracted: {results['total_groups']}")
        print(f"   Comprehensive parquet: {results['comprehensive_parquet']}")
        print(f"   Individual member files: {successful_members}/{len(target_members)} successful")

        print(f"\n📁 Output files:")
        print(f"   Directory: {output_dir}")
        parquet_files = list(output_dir.glob("*.par*"))
        for pf in parquet_files:
            print(f"   - {pf.name}")

        if successful_members == len(target_members):
            print(f"\n✅ All ensemble members processed successfully!")
            return True
        else:
            print(f"\n⚠️ {len(target_members) - successful_members} members failed processing")
            return False

    except Exception as e:
        log_message(f"Processing failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 Efficient ECMWF ensemble processing completed successfully!")
    else:
        print("\n❌ Efficient ECMWF ensemble processing failed")
        exit(1)
