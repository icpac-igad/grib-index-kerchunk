#!/usr/bin/env python3
"""
ECMWF GCS Uploader
Uploads processed ECMWF ensemble parquet files to Google Cloud Storage
with proper FMRC-compatible folder structure.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import gcsfs
from tqdm import tqdm

def log_message(message: str, level: str = "INFO"):
    """Simple logging function with timestamps."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

class ECMWFGCSUploader:
    def __init__(self, service_account_path: str, bucket_name: str = "gik-fmrc"):
        """
        Initialize the GCS uploader.

        Args:
            service_account_path: Path to GCS service account JSON file
            bucket_name: Target GCS bucket name
        """
        self.service_account_path = service_account_path
        self.bucket_name = bucket_name
        self.fs = None
        self._initialize_gcs_client()

    def _initialize_gcs_client(self):
        """Initialize GCS filesystem client with service account."""
        try:
            if not os.path.exists(self.service_account_path):
                raise FileNotFoundError(f"Service account file not found: {self.service_account_path}")

            self.fs = gcsfs.GCSFileSystem(token=self.service_account_path)
            log_message(f"GCS client initialized with service account: {Path(self.service_account_path).name}")

            # Test connection
            try:
                self.fs.ls(self.bucket_name)
                log_message(f"Successfully connected to bucket: {self.bucket_name}")
            except Exception as e:
                log_message(f"Warning: Could not list bucket contents: {e}", "WARNING")

        except Exception as e:
            log_message(f"Failed to initialize GCS client: {e}", "ERROR")
            raise

    def scan_local_files(self, local_base_path: str, date_str: str) -> Dict[str, Dict]:
        """
        Scan local directory structure to identify all par files.

        Args:
            local_base_path: Base path to ecmwf_gcs_YYYYMMDD_v2 directory
            date_str: Date string (e.g., '20240529')

        Returns:
            Dictionary mapping local file paths to upload metadata
        """
        local_path = Path(local_base_path) / f"ecmwf_gcs_{date_str}_v2" / "proce_ens"

        if not local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")

        log_message(f"Scanning local files in: {local_path}")

        upload_plan = {}
        total_files = 0

        # Scan all hour directories
        for hour_dir in sorted(local_path.glob("*h")):
            if not hour_dir.is_dir():
                continue

            hour_str = hour_dir.name.replace('h', '')
            try:
                hour_num = int(hour_str)
                hour_padded = f"{hour_num:03d}"
            except ValueError:
                log_message(f"Skipping invalid hour directory: {hour_dir.name}", "WARNING")
                continue

            # Scan all par files in this hour directory
            for par_file in hour_dir.glob("*.par"):
                member_name = par_file.stem

                # Determine ensemble folder name and filename
                if member_name == "control":
                    gcs_folder = "ens_control"
                    filename_member = "control"
                elif member_name.startswith("ens") and member_name[3:].isdigit():
                    member_num = int(member_name[3:])
                    gcs_folder = f"ens_{member_num:02d}"
                    filename_member = f"ens{member_num:02d}"
                else:
                    log_message(f"Skipping unknown member: {member_name}", "WARNING")
                    continue

                # Generate GCS path and filename
                gcs_filename = f"ecmwf-{date_str}00-{filename_member}-rt{hour_padded}.par"
                gcs_path = f"v2ecmwf_fmrc/{gcs_folder}/{gcs_filename}"

                upload_plan[str(par_file)] = {
                    'local_path': str(par_file),
                    'gcs_path': gcs_path,
                    'gcs_folder': gcs_folder,
                    'member': member_name,
                    'hour': hour_str,
                    'file_size': par_file.stat().st_size if par_file.exists() else 0
                }
                total_files += 1

        log_message(f"Found {total_files} files to upload across {len(set(f['gcs_folder'] for f in upload_plan.values()))} ensemble members")

        return upload_plan

    def check_existing_files(self, upload_plan: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Check which files already exist in GCS to avoid re-uploading.

        Args:
            upload_plan: Upload plan dictionary

        Returns:
            Filtered upload plan excluding existing files
        """
        log_message("Checking for existing files in GCS...")

        existing_files = set()
        filtered_plan = {}

        try:
            # Get list of all files in the target directory
            target_path = f"{self.bucket_name}/v2ecmwf_fmrc"
            all_gcs_files = self.fs.glob(f"{target_path}/**/*.par")
            existing_files = {f.replace(f"{self.bucket_name}/", "") for f in all_gcs_files}

            log_message(f"Found {len(existing_files)} existing files in GCS")

        except Exception as e:
            log_message(f"Could not check existing files: {e}", "WARNING")
            log_message("Proceeding with all files (may overwrite existing)", "WARNING")

        # Filter out existing files
        skipped = 0
        for local_path, metadata in upload_plan.items():
            if metadata['gcs_path'] in existing_files:
                log_message(f"Skipping existing file: {metadata['gcs_path']}")
                skipped += 1
            else:
                filtered_plan[local_path] = metadata

        log_message(f"Skipped {skipped} existing files, {len(filtered_plan)} files to upload")
        return filtered_plan

    def upload_single_file(self, local_path: str, gcs_path: str) -> Tuple[bool, str, str]:
        """
        Upload a single file to GCS.

        Args:
            local_path: Local file path
            gcs_path: Target GCS path

        Returns:
            Tuple of (success, local_path, error_message)
        """
        try:
            full_gcs_path = f"{self.bucket_name}/{gcs_path}"

            # Ensure local file exists
            if not os.path.exists(local_path):
                return False, local_path, f"Local file not found: {local_path}"

            # Get file size for validation
            local_size = os.path.getsize(local_path)

            # Upload file
            self.fs.put(local_path, full_gcs_path)

            # Verify upload by checking file exists and size
            try:
                gcs_info = self.fs.info(full_gcs_path)
                gcs_size = gcs_info.get('size', 0)

                if gcs_size != local_size:
                    return False, local_path, f"Size mismatch: local={local_size}, gcs={gcs_size}"

            except Exception as e:
                return False, local_path, f"Upload verification failed: {e}"

            return True, local_path, ""

        except Exception as e:
            return False, local_path, str(e)

    def upload_files_parallel(self, upload_plan: Dict[str, Dict], max_workers: int = 10) -> Dict:
        """
        Upload files in parallel with progress tracking.

        Args:
            upload_plan: Dictionary of files to upload
            max_workers: Maximum number of parallel workers

        Returns:
            Results dictionary with statistics
        """
        log_message(f"Starting parallel upload with {max_workers} workers...")

        results = {
            'total_files': len(upload_plan),
            'successful_uploads': 0,
            'failed_uploads': 0,
            'errors': [],
            'start_time': time.time()
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_metadata = {}
            for local_path, metadata in upload_plan.items():
                future = executor.submit(
                    self.upload_single_file,
                    metadata['local_path'],
                    metadata['gcs_path']
                )
                future_to_metadata[future] = metadata

            # Process completed uploads with progress bar
            with tqdm(total=len(upload_plan), desc="Uploading files") as pbar:
                for future in as_completed(future_to_metadata):
                    success, local_path, error_msg = future.result()
                    metadata = future_to_metadata[future]

                    if success:
                        results['successful_uploads'] += 1
                        pbar.set_postfix({
                            'Success': results['successful_uploads'],
                            'Failed': results['failed_uploads']
                        })
                    else:
                        results['failed_uploads'] += 1
                        error_info = {
                            'local_path': local_path,
                            'gcs_path': metadata['gcs_path'],
                            'error': error_msg
                        }
                        results['errors'].append(error_info)
                        log_message(f"Upload failed: {Path(local_path).name} - {error_msg}", "ERROR")

                    pbar.update(1)

        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']

        return results

    def save_upload_report(self, results: Dict, upload_plan: Dict, output_file: str):
        """Save detailed upload report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'bucket': self.bucket_name,
            'service_account': Path(self.service_account_path).name,
            'upload_summary': {
                'total_files': results['total_files'],
                'successful_uploads': results['successful_uploads'],
                'failed_uploads': results['failed_uploads'],
                'success_rate': results['successful_uploads'] / results['total_files'] * 100 if results['total_files'] > 0 else 0,
                'total_time_seconds': results['total_time'],
                'total_time_minutes': results['total_time'] / 60
            },
            'ensemble_summary': {},
            'errors': results['errors']
        }

        # Generate per-ensemble statistics
        ensemble_stats = {}
        for metadata in upload_plan.values():
            folder = metadata['gcs_folder']
            if folder not in ensemble_stats:
                ensemble_stats[folder] = {'total': 0, 'size_bytes': 0}
            ensemble_stats[folder]['total'] += 1
            ensemble_stats[folder]['size_bytes'] += metadata['file_size']

        report['ensemble_summary'] = ensemble_stats

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        log_message(f"Upload report saved to: {output_file}")

def main():
    """Main function for ECMWF GCS upload process."""
    print("="*80)
    print("ECMWF GCS Uploader")
    print("Uploading ensemble parquet files to Google Cloud Storage")
    print("="*80)

    # Configuration
    SERVICE_ACCOUNT_PATH = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
    BUCKET_NAME = "gik-fmrc"
    DATE_STR = "20240529"  # Can be made configurable
    LOCAL_BASE_PATH = "/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/grib-index-kerchunk/ecmwf"
    MAX_WORKERS = 8  # Adjust based on network capacity

    try:
        # Initialize uploader
        uploader = ECMWFGCSUploader(SERVICE_ACCOUNT_PATH, BUCKET_NAME)

        # Scan local files
        log_message("Step 1: Scanning local files...")
        upload_plan = uploader.scan_local_files(LOCAL_BASE_PATH, DATE_STR)

        if not upload_plan:
            log_message("No files found to upload. Exiting.", "ERROR")
            return False

        # Check for existing files
        log_message("Step 2: Checking for existing files in GCS...")
        filtered_plan = uploader.check_existing_files(upload_plan)

        if not filtered_plan:
            log_message("All files already exist in GCS. Nothing to upload.", "INFO")
            return True

        # Upload files
        log_message("Step 3: Uploading files to GCS...")
        results = uploader.upload_files_parallel(filtered_plan, MAX_WORKERS)

        # Generate report
        report_file = f"ecmwf_gcs_upload_report_{DATE_STR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        uploader.save_upload_report(results, filtered_plan, report_file)

        # Print summary
        print("\n" + "="*80)
        print("UPLOAD SUMMARY")
        print("="*80)
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful uploads: {results['successful_uploads']}")
        print(f"Failed uploads: {results['failed_uploads']}")
        print(f"Success rate: {results['successful_uploads']/results['total_files']*100:.1f}%")
        print(f"Total time: {results['total_time']/60:.1f} minutes")
        print(f"Average speed: {results['successful_uploads']/(results['total_time']/60):.1f} files/minute")

        if results['errors']:
            print(f"\n❌ {len(results['errors'])} errors occurred:")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"   {Path(error['local_path']).name}: {error['error']}")
            if len(results['errors']) > 5:
                print(f"   ... and {len(results['errors']) - 5} more errors (see report file)")

        success = results['failed_uploads'] == 0
        if success:
            print(f"\n🎉 All files uploaded successfully!")
        else:
            print(f"\n⚠️ Upload completed with {results['failed_uploads']} failures")

        print(f"📝 Detailed report: {report_file}")
        return success

    except Exception as e:
        log_message(f"Upload process failed: {e}", "ERROR")
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 ECMWF GCS upload completed successfully!")
    else:
        print("\n❌ ECMWF GCS upload failed")
        exit(1)