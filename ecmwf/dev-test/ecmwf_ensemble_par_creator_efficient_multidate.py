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
import shutil
import zipfile
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


# Import the efficient functions from utils module
try:
    from utils_ecmwf_step1_scangrib import fixed_ensemble_grib_tree, ecmwf_filter_scan_grib
    log_message("Successfully imported efficient functions from utils_ecmwf_step1_scangrib")
except ImportError as e:
    log_message(f"Error: Could not import required functions from utils_ecmwf_step1_scangrib.py: {e}", "ERROR")
    log_message("Please ensure utils_ecmwf_step1_scangrib.py is in the same directory", "ERROR")
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
    # Comprehensive file in 'comprehensive' subdirectory
    comprehensive_dir = output_dir / "comprehensive"
    comprehensive_dir.mkdir(exist_ok=True)
    comprehensive_parquet = comprehensive_dir / f"ecmwf_{date_str}_{run}z_ensemble_all.parquet"
    create_parquet_file_fixed(deflated_ensemble_tree['refs'], str(comprehensive_parquet))
    log_checkpoint(f"Comprehensive parquet saved: {comprehensive_parquet}", parquet_start)

    # Step 5: Validate zarr metadata
    # Using simple zarr metadata validation instead of xarray to avoid version compatibility issues
    validation_start = log_checkpoint("Validating zarr metadata")
    try:
        # Find all zarr variables in the ensemble tree
        variables = set()
        for key in ensemble_tree['refs'].keys():
            if '/.zarray' in key:
                var_path = key.replace('/.zarray', '')
                variables.add(var_path)

        if not variables:
            raise ValueError("No valid zarr variables found in ensemble tree")

        log_checkpoint("Zarr metadata validation successful", validation_start)
        log_message(f"Validated {len(variables)} zarr variables")
        log_message(f"Sample variables: {sorted(list(variables))[:10]}{'...' if len(variables) > 10 else ''}")

    except Exception as e:
        log_message(f"Zarr validation failed: {e}", "WARNING")

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
        member_name = "control" if member == -1 else f"ens_{member:02d}"  # Changed to ens_01 format

        try:
            extract_start = log_checkpoint(f"Extracting {member_name}")

            # Extract member-specific references from the ensemble tree
            member_refs = extract_member_refs(ensemble_tree['refs'], member)

            if not member_refs:
                log_message(f"No references found for {member_name}", "WARNING")
                continue

            # Create member-specific parquet file
            # Member-specific subdirectory
            member_dir = output_dir / "members" / member_name
            member_dir.mkdir(parents=True, exist_ok=True)
            member_parquet = member_dir / f"{member_name}.parquet"
            create_parquet_file_fixed(member_refs, str(member_parquet))

            # Validate member file using simple zarr metadata check
            # This avoids fsspec/xarray version compatibility issues
            try:
                # Find all zarr variables
                variables = set()
                for key in member_refs.keys():
                    if '/.zarray' in key:
                        var_path = key.replace('/.zarray', '')
                        variables.add(var_path)

                if not variables:
                    raise ValueError("No valid zarr variables found in parquet")

                log_message(f"Validated {len(variables)} zarr variables in {member_name}")

                extraction_results[member_name] = {
                    'success': True,
                    'parquet_file': str(member_parquet),
                    'refs_count': len(member_refs),
                    'variables_count': len(variables)
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

    Fixed version that properly handles hierarchical zarr structure and chunk references.

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

    # First, identify all variables that have ensemble dimension
    variables_with_ensemble = set()
    hierarchical_paths = set()

    for key in all_refs:
        if '/.zattrs' in key:
            # Get the variable path (everything before /.zattrs)
            var_path = key.replace('/.zattrs', '')
            attrs = json.loads(all_refs[key]) if isinstance(all_refs[key], str) else all_refs[key]

            # Check if this has number dimension
            if '_ARRAY_DIMENSIONS' in attrs and 'number' in attrs['_ARRAY_DIMENSIONS']:
                variables_with_ensemble.add(var_path)
                # Track hierarchical structure
                parts = var_path.split('/')
                for i in range(1, len(parts) + 1):
                    hierarchical_paths.add('/'.join(parts[:i]))

    # Process all references
    for key, value in all_refs.items():
        # Skip pure number coordinate references
        if key.startswith('number/') and not key.endswith('/.zattrs') and not key.endswith('/.zarray'):
            continue

        # Determine the variable this key belongs to
        if '/' in key:
            # Find the variable path by removing the last component
            key_parts = key.split('/')

            # Check if this is a chunk reference (contains dots in last part)
            is_chunk = '.' in key_parts[-1] and not key_parts[-1].startswith('.')

            if is_chunk:
                # This is a data chunk
                var_path = '/'.join(key_parts[:-1])
                chunk_indices = key_parts[-1]

                # Check if this variable has ensemble dimension
                if var_path in variables_with_ensemble:
                    # Parse chunk indices
                    indices = chunk_indices.split('.')

                    # The ensemble dimension is typically the 3rd one (index 2)
                    # Structure: time.step.number.lat.lon or similar
                    if len(indices) > 2:
                        try:
                            chunk_member_idx = int(indices[2])

                            # Only include if it matches our target member
                            if chunk_member_idx == member_index:
                                # Create new key without ensemble index
                                new_indices = indices[:2] + indices[3:]
                                new_key = f"{var_path}/{'.'.join(new_indices)}"
                                member_refs[new_key] = value
                        except (ValueError, IndexError):
                            # Can't parse member index, skip this chunk
                            pass
                else:
                    # Variable doesn't have ensemble dimension, copy as-is
                    member_refs[key] = value

            elif key.endswith('/.zattrs'):
                # Handle attributes
                var_path = key.replace('/.zattrs', '')
                attrs = json.loads(value) if isinstance(value, str) else value

                # Remove number dimension if present
                if var_path in variables_with_ensemble:
                    if '_ARRAY_DIMENSIONS' in attrs:
                        attrs['_ARRAY_DIMENSIONS'] = [d for d in attrs['_ARRAY_DIMENSIONS'] if d != 'number']

                # Skip number coordinate attributes
                if var_path == 'number':
                    continue

                member_refs[key] = json.dumps(attrs)

            elif key.endswith('/.zarray'):
                # Handle array metadata
                var_path = key.replace('/.zarray', '')

                # Skip number coordinate array
                if var_path == 'number':
                    continue

                zarray = json.loads(value) if isinstance(value, str) else value

                # Modify shape and chunks if this variable has ensemble dimension
                if var_path in variables_with_ensemble:
                    if 'shape' in zarray:
                        # Find and remove dimension with size 51
                        new_shape = []
                        ensemble_dim_idx = None

                        for idx, dim in enumerate(zarray['shape']):
                            if dim == 51:
                                ensemble_dim_idx = idx
                            else:
                                new_shape.append(dim)

                        zarray['shape'] = new_shape

                        # Update chunks if present
                        if ensemble_dim_idx is not None and 'chunks' in zarray:
                            new_chunks = []
                            for idx, chunk in enumerate(zarray['chunks']):
                                if idx != ensemble_dim_idx:
                                    new_chunks.append(chunk)
                            zarray['chunks'] = new_chunks

                member_refs[key] = json.dumps(zarray)

            elif key.endswith('/.zgroup'):
                # Copy group definitions
                # Skip if it's the number group
                var_path = key.replace('/.zgroup', '')
                if var_path != 'number':
                    member_refs[key] = value
            else:
                # Other metadata or coordinate values
                # Skip if it's number-related
                if not key.startswith('number'):
                    member_refs[key] = value
        else:
            # Root level keys
            member_refs[key] = value

    # Add extraction metadata
    if '.zattrs' in member_refs:
        root_attrs = json.loads(member_refs['.zattrs'])
        root_attrs['extracted_ensemble_member'] = target_member
        root_attrs['extraction_method'] = 'fixed_hierarchical'
        member_refs['.zattrs'] = json.dumps(root_attrs)

    log_message(f"Extracted {len(member_refs)} references for member {target_member} (from {len(all_refs)} total)")

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


def create_zip_archive(source_dir: Path, zip_path: Path) -> bool:
    """
    Create a zip archive of the source directory.

    Parameters:
    - source_dir: Path to the directory to compress
    - zip_path: Path for the output zip file

    Returns:
    - True if successful, False otherwise
    """
    try:
        log_message(f"Creating zip archive: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zipf.write(file_path, arcname)

        # Get zip file size
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        log_message(f"Zip archive created: {zip_path} ({zip_size_mb:.2f} MB)")
        return True

    except Exception as e:
        log_message(f"Error creating zip archive: {e}", "ERROR")
        return False


def process_single_date(date_str: str, run: str, target_members: List[int], base_output_dir: Path = None) -> Tuple[bool, Optional[Path]]:
    """
    Process ECMWF ensemble data for a single date.

    Parameters:
    - date_str: Date string (e.g., '20251125')
    - run: Run hour (e.g., '00')
    - target_members: List of ensemble members to extract
    - base_output_dir: Base output directory (optional)

    Returns:
    - Tuple of (success, output_directory_path)
    """
    print("\n" + "="*80)
    print(f"Processing ECMWF Ensemble for {date_str} {run}z")
    print("="*80)

    start_time = time.time()

    # Create output directory with date naming
    if base_output_dir is None:
        output_dir = Path(f"ecmwf_{date_str}_{run}z_efficient")
    else:
        output_dir = base_output_dir / f"ecmwf_{date_str}_{run}z"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_message(f"Output directory: {output_dir}")

    # Define ECMWF files - all forecast timesteps
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
        log_message(f"No valid ECMWF files found for {date_str}", "ERROR")
        return False, None

    log_message(f"Processing {len(available_files)} available ECMWF files")
    log_message(f"Target ensemble members: {len(target_members)} members")

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

        print(f"\n📈 Results for {date_str}:")
        print(f"   Total processing time: {total_time/60:.1f} minutes")
        print(f"   Files processed: {len(available_files)}")
        print(f"   Total groups extracted: {results['total_groups']}")
        print(f"   Individual member files: {successful_members}/{len(target_members)} successful")

        if successful_members == len(target_members):
            log_message(f"All ensemble members processed successfully for {date_str}!")
            return True, output_dir
        else:
            log_message(f"{len(target_members) - successful_members} members failed processing for {date_str}", "WARNING")
            return True, output_dir  # Still return success if we got some data

    except Exception as e:
        log_message(f"Processing failed for {date_str}: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Main function for efficient ECMWF ensemble processing with multi-date support."""
    print("="*80)
    print("ECMWF Efficient Ensemble Parquet Creator - Multi-Date")
    print("="*80)

    overall_start_time = time.time()

    # Configuration - multiple dates
    dates = ['20251126', '20251127']
    run = '00'
    target_members = [-1, 1, 2, 3, 4, 5] + list(range(6, 51))

    log_message(f"Processing {len(dates)} dates: {dates}")
    log_message(f"Run: {run}z")
    log_message(f"Target members: {len(target_members)} ensemble members")

    # Track results for each date
    processing_results = {}
    successful_dates = []
    failed_dates = []
    zip_files_created = []

    for date_str in dates:
        print("\n" + "="*80)
        print(f"PROCESSING DATE: {date_str} {run}z")
        print("="*80)

        # Process single date
        success, output_dir = process_single_date(date_str, run, target_members)

        if success and output_dir is not None:
            # Create zip archive for this date
            zip_path = Path(f"ecmwf_{date_str}_{run}z_efficient.zip")
            zip_success = create_zip_archive(output_dir, zip_path)

            processing_results[date_str] = {
                'success': True,
                'output_dir': str(output_dir),
                'zip_file': str(zip_path) if zip_success else None,
                'zip_success': zip_success
            }
            successful_dates.append(date_str)

            if zip_success:
                zip_files_created.append(zip_path)
        else:
            processing_results[date_str] = {
                'success': False,
                'output_dir': None,
                'zip_file': None,
                'zip_success': False
            }
            failed_dates.append(date_str)

    # Overall Summary
    overall_time = time.time() - overall_start_time

    print("\n" + "="*80)
    print("MULTI-DATE PROCESSING SUMMARY")
    print("="*80)

    print(f"\n📅 Dates Processed: {len(dates)}")
    print(f"   Successful: {len(successful_dates)}")
    print(f"   Failed: {len(failed_dates)}")

    print(f"\n⏱️ Total Processing Time: {overall_time/60:.1f} minutes")

    if successful_dates:
        print(f"\n✅ Successfully processed dates:")
        for date_str in successful_dates:
            result = processing_results[date_str]
            zip_status = "zipped" if result['zip_success'] else "zip failed"
            print(f"   - {date_str}: {result['output_dir']} ({zip_status})")

    if failed_dates:
        print(f"\n❌ Failed dates:")
        for date_str in failed_dates:
            print(f"   - {date_str}")

    if zip_files_created:
        print(f"\n📦 Zip files created:")
        for zip_path in zip_files_created:
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"   - {zip_path} ({zip_size_mb:.2f} MB)")

    # Save overall processing summary
    summary = {
        'dates_processed': dates,
        'run': run,
        'successful_dates': successful_dates,
        'failed_dates': failed_dates,
        'total_processing_time_minutes': overall_time / 60,
        'results': processing_results
    }

    summary_file = Path("ecmwf_multidate_processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    log_message(f"Processing summary saved to {summary_file}")

    return len(failed_dates) == 0


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 Multi-date ECMWF ensemble processing completed successfully!")
    else:
        print("\n❌ Some dates failed processing - check logs for details")
        exit(1)
