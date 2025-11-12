#!/usr/bin/env python3
"""
ECMWF Ensemble Parquet Creator - Version 2
Downloads parquet files from Google Cloud Storage and processes them into ensemble member-wise parquet files.
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
import glob
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

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

def discover_gcs_parquet_files(gcs_path: str, date_str: str, fs) -> List[str]:
    """
    Discover available parquet files in GCS using glob pattern.

    Parameters:
    - gcs_path: Base GCS path
    - date_str: Date string for the files
    - fs: GCS filesystem object

    Returns:
    - List of available file paths in GCS
    """
    try:
        # Remove gs:// prefix for gcsfs
        bucket_path = gcs_path.replace("gs://", "").rstrip('/')

        # Try multiple possible file patterns
        patterns = [
            f"{bucket_path}/e_sg_mdt_{date_str}_*h.parquet",
            f"{bucket_path}/e_sg_mdt_{date_str}_*.parquet",
            f"{bucket_path}/*{date_str}*.parquet",
            f"{bucket_path}/*.parquet"
        ]

        available_files = []

        for pattern in patterns:
            try:
                log_message(f"Searching with pattern: {pattern}")
                files = fs.glob(pattern)
                if files:
                    log_message(f"Found {len(files)} files with pattern: {pattern}")
                    available_files.extend([f"gs://{f}" for f in files])
                    break  # Use first successful pattern
            except Exception as e:
                log_message(f"Error with pattern {pattern}: {e}", "WARNING")
                continue

        if not available_files:
            # Fallback: list directory contents
            try:
                log_message(f"Fallback: listing directory contents: {bucket_path}")
                dir_contents = fs.ls(bucket_path)
                parquet_files = [f"gs://{f}" for f in dir_contents if f.endswith('.parquet')]
                available_files.extend(parquet_files)
                log_message(f"Found {len(parquet_files)} parquet files in directory")
            except Exception as e:
                log_message(f"Error listing directory: {e}", "ERROR")

        return available_files

    except Exception as e:
        log_message(f"Error discovering files: {e}", "ERROR")
        return []

def download_gcs_parquet_files(gcs_path: str, local_dir: Path, date_str: str) -> List[Path]:
    """
    Download parquet files from Google Cloud Storage using service account credentials.

    Parameters:
    - gcs_path: Base GCS path (e.g., "gs://gik-ecmwf-aws-tf/fmrc/scan_grib20240529/")
    - local_dir: Local directory to download files to
    - date_str: Date string for the files

    Returns:
    - List of downloaded file paths
    """
    log_message(f"Setting up GCS filesystem for download with service account authentication")

    # Get service account credentials path from environment
    service_account_path = os.getenv('GCS_SERVICE_ACCOUNT_PATH')
    if not service_account_path:
        log_message("Error: GCS_SERVICE_ACCOUNT_PATH not found in environment variables", "ERROR")
        log_message("Please set GCS_SERVICE_ACCOUNT_PATH in your .env file", "ERROR")
        return []

    if not os.path.exists(service_account_path):
        log_message(f"Error: Service account file not found at {service_account_path}", "ERROR")
        return []

    # Create GCS filesystem with service account credentials
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token=service_account_path)
        log_message(f"Successfully authenticated with service account: {Path(service_account_path).name}")
    except ImportError:
        log_message("Error: gcsfs library not available. Please install gcsfs.", "ERROR")
        return []
    except Exception as e:
        log_message(f"Error setting up GCS authentication: {e}", "ERROR")
        return []

    # Create local download directory
    download_dir = local_dir / "downloaded_parquet"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Discover available files using glob patterns
    log_message("Discovering available parquet files in GCS...")
    available_gcs_files = discover_gcs_parquet_files(gcs_path, date_str, fs)

    if not available_gcs_files:
        log_message("No parquet files found in GCS bucket", "ERROR")
        return []

    log_message(f"Found {len(available_gcs_files)} parquet files in GCS")

    # Download discovered files
    downloaded_files = []

    for gcs_file_path in available_gcs_files:
        try:
            filename = Path(gcs_file_path).name
            local_file_path = download_dir / filename

            log_message(f"Downloading {filename}")
            fs.get(gcs_file_path, str(local_file_path))
            downloaded_files.append(local_file_path)

        except Exception as e:
            log_message(f"Error downloading {filename}: {e}", "ERROR")
            continue

    log_message(f"Downloaded {len(downloaded_files)} parquet files to {download_dir}")
    return downloaded_files

def inspect_parquet_structure(parquet_file: Path) -> Dict:
    """
    Inspect the structure of a parquet file to understand its format.

    Parameters:
    - parquet_file: Path to the parquet file

    Returns:
    - Dictionary with file structure information
    """
    try:
        df = pd.read_parquet(parquet_file)

        structure_info = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'sample_data': {}
        }

        # Get sample data for each column (first few rows)
        for col in df.columns:
            sample_values = df[col].head(3).tolist()
            structure_info['sample_data'][col] = sample_values

        return structure_info

    except Exception as e:
        return {'error': str(e)}

def load_parquet_to_zarr_store(parquet_file: Path) -> Dict:
    """
    Load a parquet file and convert it back to zarr store format.
    Handles different parquet file structures.

    Parameters:
    - parquet_file: Path to the parquet file

    Returns:
    - Dictionary representing zarr store
    """
    try:
        # First inspect the file structure
        structure = inspect_parquet_structure(parquet_file)
        log_message(f"Parquet file structure for {parquet_file.name}:")
        log_message(f"  Columns: {structure.get('columns', 'unknown')}")
        log_message(f"  Shape: {structure.get('shape', 'unknown')}")

        df = pd.read_parquet(parquet_file)
        zstore = {}

        # Handle different parquet file structures
        if 'key' in df.columns and 'value' in df.columns:
            # Standard zarr store format
            log_message(f"Processing as standard zarr store format")
            for _, row in df.iterrows():
                key = row['key']
                value = row['value']

                # Decode bytes if necessary
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8')
                    except UnicodeDecodeError:
                        # Keep as bytes if can't decode
                        pass

                zstore[key] = value

        elif len(df.columns) == 1:
            # Single column format - might be a JSON or serialized format
            col_name = df.columns[0]
            log_message(f"Processing as single column format: {col_name}")

            for idx, row in df.iterrows():
                key = f"data_{idx}"
                value = row[col_name]
                zstore[key] = value

        else:
            # Multi-column format - treat each column as a separate key
            log_message(f"Processing as multi-column format with {len(df.columns)} columns")

            for col in df.columns:
                for idx, value in enumerate(df[col]):
                    key = f"{col}_{idx}"
                    zstore[key] = value

        log_message(f"Loaded {len(zstore)} entries from parquet file")
        return zstore

    except Exception as e:
        log_message(f"Error loading parquet file {parquet_file}: {e}", "ERROR")
        return {}

def extract_ensemble_members_from_store(zstore: Dict) -> Dict[str, Dict]:
    """
    Extract individual ensemble members from a combined zarr store.
    Handles different data structures including direct parquet data.

    Parameters:
    - zstore: Combined zarr store dictionary

    Returns:
    - Dictionary mapping member names to their zarr stores
    """
    ensemble_stores = {}
    member_keys = defaultdict(dict)

    log_message(f"Analyzing zarr store with {len(zstore)} keys for ensemble extraction")

    # Debug: show some sample keys
    sample_keys = list(zstore.keys())[:10]
    log_message(f"Sample keys: {sample_keys}")

    # Strategy 1: Look for ensemble member indicators in the key path
    for key, value in zstore.items():
        if '/number/' in key or '_number_' in key or '/ens' in key:
            # Extract ensemble member information
            parts = key.split('/')

            # Find ensemble member number in the path
            member_num = None
            for i, part in enumerate(parts):
                if part == 'number' and i + 1 < len(parts):
                    member_num = parts[i + 1]
                    break
                elif part.startswith('number_'):
                    member_num = part.split('_')[1]
                    break
                elif part.startswith('ens') and part[3:].isdigit():
                    member_num = part[3:]
                    break

            if member_num is not None:
                try:
                    member_num = int(member_num)
                    member_name = f"ens{member_num:02d}" if member_num != -1 else "control"
                    # Create member-specific key by removing ensemble dimension
                    member_key = key.replace(f'/number/{member_num}', '').replace(f'_number_{member_num}', '')
                    member_keys[member_name][member_key] = value
                except ValueError:
                    continue

    # Strategy 2: Look for .zattrs with ensemble information
    if not member_keys:
        log_message("No ensemble structure found in paths, checking .zattrs files")
        for key, value in zstore.items():
            if key.endswith('.zattrs'):
                try:
                    if isinstance(value, str):
                        attrs = json.loads(value)
                    else:
                        attrs = value

                    if isinstance(attrs, dict) and ('number' in attrs or 'ens_number' in attrs):
                        member_num = attrs.get('number', attrs.get('ens_number'))
                        if member_num is not None:
                            member_name = f"ens{member_num:02d}" if member_num != -1 else "control"

                            # Extract all related keys for this member
                            base_path = key.replace('.zattrs', '')
                            for k, v in zstore.items():
                                if k.startswith(base_path):
                                    member_keys[member_name][k] = v
                except (json.JSONDecodeError, TypeError) as e:
                    log_message(f"Error parsing .zattrs in {key}: {e}", "WARNING")
                    continue

    # Strategy 3: If this appears to be direct columnar data, try to identify ensemble structure from data patterns
    if not member_keys and len(zstore) > 0:
        log_message("Trying to identify ensemble structure from data patterns")

        # Check if we have data that might represent ensemble information
        for key, value in zstore.items():
            if isinstance(value, (list, str)) and 'number' in str(value).lower():
                try:
                    # Try to parse if this contains ensemble member information
                    if isinstance(value, str) and value.isdigit():
                        member_num = int(value)
                        member_name = f"ens{member_num:02d}" if member_num != -1 else "control"
                        member_keys[member_name][key] = value
                    elif isinstance(value, list):
                        # Handle array of ensemble numbers
                        for i, item in enumerate(value):
                            if isinstance(item, (int, str)) and str(item).isdigit():
                                member_num = int(item)
                                member_name = f"ens{member_num:02d}" if member_num != -1 else "control"
                                member_keys[member_name][f"{key}_{i}"] = item
                except (ValueError, TypeError):
                    continue

    # Strategy 4: If still no ensemble members found, create a single "combined" entry
    if not member_keys:
        log_message("No ensemble structure detected, treating as combined data")
        member_keys["combined_data"] = zstore

    log_message(f"Extracted {len(member_keys)} ensemble groups: {list(member_keys.keys())}")
    return dict(member_keys)

def process_parquet_to_ensemble_members(parquet_files: List[Path], output_dir: Path, date_str: str) -> Dict:
    """
    Process downloaded parquet files and extract ensemble members.

    Parameters:
    - parquet_files: List of downloaded parquet file paths
    - output_dir: Output directory for ensemble member files
    - date_str: Date string for organization

    Returns:
    - Dictionary with processing results
    """
    log_message(f"Processing {len(parquet_files)} parquet files for ensemble extraction")

    # Create output directory for ensemble members
    ensemble_dir = output_dir / f"ensemble_members_{date_str}"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'total_files_processed': 0,
        'successful_extractions': 0,
        'ensemble_members_found': set(),
        'member_files': {},
        'errors': []
    }

    # Process each parquet file
    for i, parquet_file in enumerate(parquet_files):
        try:
            log_message(f"Processing {parquet_file.name}")

            # Load parquet to zarr store
            zstore = load_parquet_to_zarr_store(parquet_file)
            if not zstore:
                continue

            # For the first file, save detailed diagnostic information
            if i == 0:
                diagnostic_file = output_dir / f"diagnostic_first_file_{parquet_file.name}.json"
                diagnostic_info = {
                    'filename': parquet_file.name,
                    'zstore_keys_sample': list(zstore.keys())[:20] if zstore else [],
                    'zstore_size': len(zstore),
                    'sample_values': {}
                }

                # Add sample values for first few keys
                for j, (key, value) in enumerate(zstore.items()):
                    if j >= 5:  # Only first 5 keys
                        break
                    diagnostic_info['sample_values'][key] = {
                        'type': str(type(value)),
                        'value_preview': str(value)[:200] if value else 'None'
                    }

                with open(diagnostic_file, 'w') as f:
                    json.dump(diagnostic_info, f, indent=2, default=str)
                log_message(f"Saved diagnostic info to {diagnostic_file}")

            # Extract ensemble members
            ensemble_members = extract_ensemble_members_from_store(zstore)

            if not ensemble_members:
                log_message(f"No ensemble members found in {parquet_file.name}", "WARNING")
                continue

            # Save each ensemble member as separate parquet
            hour_match = parquet_file.stem.split('_')[-1]  # Extract hour from filename

            for member_name, member_store in ensemble_members.items():
                if member_store:  # Only process if store has content
                    member_filename = f"{member_name}_{hour_match}.par"
                    member_filepath = ensemble_dir / member_filename

                    # Save as parquet
                    create_parquet_file_simple(member_store, str(member_filepath))

                    # Track results
                    if member_name not in results['member_files']:
                        results['member_files'][member_name] = []
                    results['member_files'][member_name].append(str(member_filepath))
                    results['ensemble_members_found'].add(member_name)

            results['successful_extractions'] += 1

        except Exception as e:
            error_msg = f"Error processing {parquet_file.name}: {e}"
            log_message(error_msg, "ERROR")
            results['errors'].append(error_msg)

        results['total_files_processed'] += 1

    # Convert set to list for JSON serialization
    results['ensemble_members_found'] = list(results['ensemble_members_found'])

    log_message(f"Extraction complete: {results['successful_extractions']}/{results['total_files_processed']} files processed")
    log_message(f"Found ensemble members: {results['ensemble_members_found']}")

    return results

def create_parquet_file_simple(zstore: dict, output_parquet_file: str):
    """Save zarr store dictionary as parquet file."""
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

    # Only log for successful saves
    if len(df) > 0:
        log_message(f"Saved ensemble member parquet: {Path(output_parquet_file).name} ({len(df)} rows)")

def validate_ensemble_member_files(member_files: Dict[str, List[str]]) -> Dict:
    """
    Validate the extracted ensemble member files.

    Parameters:
    - member_files: Dictionary mapping member names to file lists

    Returns:
    - Validation results
    """
    validation_results = {
        'total_members': len(member_files),
        'total_files': sum(len(files) for files in member_files.values()),
        'validation_details': {},
        'all_valid': True
    }

    for member_name, file_list in member_files.items():
        member_validation = {
            'files_count': len(file_list),
            'valid_files': 0,
            'errors': []
        }

        for file_path in file_list:
            try:
                # Basic validation: can we read the parquet?
                df = pd.read_parquet(file_path)
                if len(df) > 0:
                    member_validation['valid_files'] += 1
            except Exception as e:
                member_validation['errors'].append(f"{Path(file_path).name}: {e}")
                validation_results['all_valid'] = False

        validation_results['validation_details'][member_name] = member_validation

    return validation_results

def main():
    """Main function for ECMWF ensemble processing from GCS parquet files."""
    print("="*80)
    print("ECMWF Ensemble Parquet Creator - Version 2")
    print("Downloads from Google Cloud Storage and extracts ensemble members")
    print("="*80)

    start_time = time.time()

    # Configuration from environment variables with defaults
    date_str = os.getenv('ECMWF_DATE', '20240529')
    gcs_bucket = os.getenv('GCS_BUCKET', 'gik-ecmwf-aws-tf')
    gcs_folder = os.getenv('GCS_FOLDER', 'fmrc')

    gcs_base_path = f"gs://{gcs_bucket}/{gcs_folder}/scan_grib{date_str}/"

    log_message(f"Configuration loaded:")
    log_message(f"  Date: {date_str}")
    log_message(f"  GCS Bucket: {gcs_bucket}")
    log_message(f"  GCS Folder: {gcs_folder}")
    log_message(f"  Full Path: {gcs_base_path}")

    # Create output directory
    output_dir = Path(f"ecmwf_gcs_{date_str}_v2")
    output_dir.mkdir(exist_ok=True)
    log_message(f"Output directory: {output_dir}")

    # Step 1: Download parquet files from GCS
    log_message("Step 1: Downloading parquet files from Google Cloud Storage")
    downloaded_files = download_gcs_parquet_files(gcs_base_path, output_dir, date_str)

    if not downloaded_files:
        log_message("No files downloaded. Exiting.", "ERROR")
        return False

    # Step 2: Process parquet files to extract ensemble members
    log_message("Step 2: Processing parquet files to extract ensemble members")
    processing_results = process_parquet_to_ensemble_members(downloaded_files, output_dir, date_str)

    # Step 3: Validate extracted files
    log_message("Step 3: Validating extracted ensemble member files")
    validation_results = validate_ensemble_member_files(processing_results['member_files'])

    # Save processing metadata
    metadata = {
        'date_str': date_str,
        'gcs_base_path': gcs_base_path,
        'processing_approach': 'gcs_download_and_ensemble_extraction_v2',
        'download_summary': {
            'files_downloaded': len(downloaded_files),
            'download_directory': str(output_dir / "downloaded_parquet")
        },
        'processing_results': processing_results,
        'validation_results': validation_results,
        'timestamp': datetime.now().isoformat()
    }

    metadata_file = output_dir / "processing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)

    print(f"\n📊 Processing Details:")
    print(f"   Date: {date_str}")
    print(f"   GCS Path: {gcs_base_path}")
    print(f"   Files Downloaded: {len(downloaded_files)}")
    print(f"   Files Processed: {processing_results['successful_extractions']}/{processing_results['total_files_processed']}")

    print(f"\n📈 Ensemble Results:")
    print(f"   Ensemble Members Found: {len(processing_results['ensemble_members_found'])}")
    print(f"   Members: {', '.join(processing_results['ensemble_members_found'])}")
    print(f"   Total Member Files Created: {validation_results['total_files']}")

    print(f"\n📁 Output Structure:")
    print(f"   Main Directory: {output_dir}")
    print(f"   Downloaded Files: {output_dir / 'downloaded_parquet'}")
    print(f"   Ensemble Members: {output_dir / f'ensemble_members_{date_str}'}")

    print(f"\n⏱️  Performance:")
    print(f"   Total Processing Time: {total_time/60:.1f} minutes")

    # Validation summary
    if validation_results['all_valid']:
        print(f"\n✅ All extracted files validated successfully!")
    else:
        print(f"\n⚠️ Some validation issues found:")
        for member, details in validation_results['validation_details'].items():
            if details['errors']:
                print(f"   {member}: {len(details['errors'])} errors")

    success = (processing_results['successful_extractions'] > 0 and
              len(processing_results['ensemble_members_found']) > 0)

    if success:
        print(f"\n🎉 Processing completed successfully!")
        print(f"📝 See {metadata_file} for detailed results")
        return True
    else:
        print(f"\n❌ Processing encountered errors")
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 ECMWF GCS ensemble processing completed!")
    else:
        print("\n❌ ECMWF GCS ensemble processing failed")
        exit(1)