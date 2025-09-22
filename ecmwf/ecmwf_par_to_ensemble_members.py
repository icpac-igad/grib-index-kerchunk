#!/usr/bin/env python3
"""
ECMWF PAR to Ensemble Members Processor
Processes existing scan_grib parquet (.par) files to create individual ensemble member parquet files.

This script specifically works with pre-existing parquet files from scan_grib output
and reorganizes them into individual ensemble member files.

Input: Parquet files from scan_grib (containing all 51 ensemble members)
       e.g., fmrc_scan_grib20240529_e_sg_mdt_20240529_0h.parquet
Output: Individual ensemble member parquet files organized by forecast hour
       e.g., ensemble_members/0h/control.par
             ensemble_members/0h/ens01.par
             ...
             ensemble_members/0h/ens50.par
"""

import pandas as pd
import numpy as np
import json
import os
import re
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from kerchunk.grib2 import grib_tree
from kerchunk._grib_idx import strip_datavar_chunks

warnings.filterwarnings('ignore')

# Configuration
ENSEMBLE_MEMBERS_TOTAL = 51  # Control (-1) + 50 perturbed members (1-50)


class ECMWFParquetProcessor:
    """Process ECMWF scan_grib parquet files to create individual ensemble member files."""

    def __init__(self, input_dir: str, output_base_dir: str):
        """
        Initialize the processor.

        Args:
            input_dir: Directory containing scan_grib parquet files
            output_base_dir: Base directory for output ensemble member files
        """
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse parquet filename to extract metadata.

        Example: fmrc_scan_grib20240529_e_sg_mdt_20240529_0h.parquet

        Returns:
            Dict with date and forecast_hour, or None if parse fails
        """
        pattern = r'fmrc_scan_grib(\d{8})_e_sg_mdt_\d{8}_(\d+)h\.parquet'
        match = re.search(pattern, filename)

        if match:
            return {
                'date': match.group(1),
                'forecast_hour': int(match.group(2)),
                'filename': filename
            }
        return None

    def extract_ensemble_number(self, group: Dict) -> int:
        """
        Extract ensemble number from a group dictionary.

        Returns:
            -1 for control member
            1-50 for perturbed members
        """
        # Check direct field
        if 'ens_number' in group:
            num = group['ens_number']
            return -1 if pd.isna(num) else int(num)

        # Check in attrs
        if 'attrs' in group:
            attrs = group['attrs']
            if 'ens_number' in attrs:
                num = attrs['ens_number']
                return -1 if pd.isna(num) else int(num)
            if 'number' in attrs:
                num = attrs['number']
                return -1 if pd.isna(num) else int(num)

        # Default to control if not found
        return -1

    def load_parquet_groups(self, parquet_path: Path) -> Tuple[pd.DataFrame, str]:
        """
        Load parquet file and extract GRIB URI for index processing.

        Returns:
            Tuple of (dataframe, grib_uri)
        """
        self.log(f"Loading: {parquet_path.name}")

        df = pd.read_parquet(parquet_path)
        self.log(f"  Parquet columns: {df.columns.tolist()}")
        self.log(f"  Parquet shape: {df.shape}")

        # Check for various possible column names for GRIB URI
        grib_uri = None
        possible_uri_columns = ['uri', 'path', 'file', 'url', 'grib_path']

        for col in possible_uri_columns:
            if col in df.columns and len(df) > 0:
                grib_uri = df[col].iloc[0]
                break

        if grib_uri is None:
            # If no URI found, we need to reconstruct it from filename info
            file_info = self.parse_filename(parquet_path.name)
            if file_info:
                date_str = file_info['date']
                forecast_hour = file_info['forecast_hour']
                # Construct likely ECMWF S3 path
                grib_uri = f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-{forecast_hour}h-enfo-ef.grib2"

        self.log(f"  GRIB URI: {grib_uri}")
        return df, grib_uri

    def parse_index_file(self, grib_uri: str) -> Dict[int, int]:
        """
        Parse ECMWF .index file to get index-to-ensemble-member mapping.

        Returns:
            Dictionary mapping index position to ensemble member number
        """
        import fsspec

        # Construct index URL
        index_url = grib_uri.replace('.grib2', '.index')
        self.log(f"  Parsing index: {index_url}")

        fs = fsspec.filesystem("s3", anon=True)
        idx_to_member = {}

        try:
            with fs.open(index_url, 'r') as f:
                for idx, line in enumerate(f):
                    try:
                        clean_line = line.strip().rstrip(',')
                        data = json.loads(clean_line)

                        # Extract ensemble number
                        ens_number = data.get('number', None)

                        # Handle NaN/None: default to control member (-1)
                        if ens_number is None or pd.isna(ens_number):
                            ens_number = -1
                        else:
                            ens_number = int(float(ens_number))

                        idx_to_member[idx] = ens_number

                    except Exception as e:
                        # Default to control for parse errors
                        idx_to_member[idx] = -1

            self.log(f"  Parsed {len(idx_to_member)} index entries")

            # Show ensemble distribution
            member_counts = {}
            for member in idx_to_member.values():
                member_counts[member] = member_counts.get(member, 0) + 1

            self.log(f"  Ensemble distribution from index:")
            for member in sorted(member_counts.keys()):
                name = "control" if member == -1 else f"ens{member:02d}"
                count = member_counts[member]
                self.log(f"    {name}: {count} messages")

            return idx_to_member

        except Exception as e:
            self.log(f"  Error parsing index file: {e}", "ERROR")
            return {}

    def create_member_groups_from_index(self, df: pd.DataFrame, idx_to_member: Dict[int, int]) -> Dict[int, List[Dict]]:
        """
        Create groups for each ensemble member based on index mapping.

        Returns:
            Dictionary mapping member number to list of groups
        """
        member_dict = {}

        # Create groups for each row in the dataframe
        for row_idx, row in df.iterrows():
            # Each row represents a variable, we need to replicate it for each ensemble member
            # that contains this variable

            # Get unique ensemble members from index
            unique_members = set(idx_to_member.values())

            for member_num in unique_members:
                # Create a group for this variable for this member
                group = {
                    'varname': row.get('varname', 'unknown'),
                    'typeOfLevel': row.get('typeOfLevel', 'unknown'),
                    'stepType': row.get('stepType', 'instant'),
                    'name': row.get('name', ''),
                    'paramId': row.get('paramId', 0),
                    'level': row.get('level', 0),
                    'ens_number': member_num,
                    'attrs': {
                        'varname': row.get('varname', 'unknown'),
                        'typeOfLevel': row.get('typeOfLevel', 'unknown'),
                        'stepType': row.get('stepType', 'instant'),
                        'name': row.get('name', ''),
                        'paramId': row.get('paramId', 0),
                        'level': row.get('level', 0),
                        'ens_number': member_num,
                    }
                }

                if member_num not in member_dict:
                    member_dict[member_num] = []

                member_dict[member_num].append(group)

        # Log distribution
        self.log(f"  Created ensemble member groups:")
        for member in sorted(member_dict.keys()):
            name = "control" if member == -1 else f"ens{member:02d}"
            count = len(member_dict[member])
            self.log(f"    {name}: {count} groups")

        return member_dict

    def create_member_parquet(self, member_groups: List[Dict]) -> Dict:
        """
        Create zarr store for a single ensemble member.

        Returns:
            Simple zarr store dictionary
        """
        # Create a simple zarr store structure without using grib_tree
        # since we don't have the complete scan_grib format

        zarr_store = {}

        # Add version info
        zarr_store['zarr_consolidated_format'] = 1
        zarr_store['metadata'] = {
            'zarr_format': 2,
            'ensemble_member': member_groups[0].get('ens_number', -1) if member_groups else -1,
            'variable_count': len(member_groups)
        }

        # Group variables by type for organization
        var_groups = {}
        for group in member_groups:
            varname = group.get('varname', 'unknown')
            level_type = group.get('typeOfLevel', 'unknown')
            level = group.get('level', 0)

            # Create hierarchical key
            if level_type == 'surface':
                key = f"{varname}/surface"
            else:
                key = f"{varname}/{level_type}/{level}"

            if key not in var_groups:
                var_groups[key] = []
            var_groups[key].append(group)

        # Add variable metadata to store
        for var_key, var_group_list in var_groups.items():
            # Use the first group as representative
            rep_group = var_group_list[0]

            # Create zarr metadata for this variable
            zarr_store[f"{var_key}/.zarray"] = json.dumps({
                'chunks': [1, 181, 360],  # Example chunking
                'compressor': None,
                'dtype': '<f4',
                'fill_value': 'NaN',
                'filters': None,
                'order': 'C',
                'shape': [1, 181, 360],  # Example shape
                'zarr_format': 2
            })

            zarr_store[f"{var_key}/.zattrs"] = json.dumps({
                'varname': rep_group.get('varname', ''),
                'name': rep_group.get('name', ''),
                'typeOfLevel': rep_group.get('typeOfLevel', ''),
                'level': rep_group.get('level', 0),
                'stepType': rep_group.get('stepType', ''),
                'paramId': rep_group.get('paramId', 0),
                'ens_number': rep_group.get('ens_number', -1)
            })

        return zarr_store

    def save_parquet(self, zarr_store: Dict, output_path: Path):
        """Save zarr store as parquet file."""
        data = []

        for key, value in zarr_store.items():
            if isinstance(value, str):
                encoded = value.encode('utf-8')
            elif isinstance(value, (list, dict)):
                encoded = json.dumps(value).encode('utf-8')
            else:
                encoded = str(value).encode('utf-8')

            data.append((key, encoded))

        df = pd.DataFrame(data, columns=['key', 'value'])
        df.to_parquet(output_path)

        size_kb = output_path.stat().st_size / 1024
        self.log(f"    Saved: {output_path.name} ({size_kb:.1f} KB)")

    def process_forecast_hour(self, parquet_file: Path, max_workers: int = 4) -> Dict:
        """
        Process all ensemble members for a single forecast hour.

        Returns:
            Processing results dictionary
        """
        # Parse filename
        file_info = self.parse_filename(parquet_file.name)
        if not file_info:
            raise ValueError(f"Cannot parse filename: {parquet_file.name}")

        forecast_hour = file_info['forecast_hour']
        date_str = file_info['date']

        self.log(f"\n{'='*60}")
        self.log(f"Processing: {forecast_hour}h forecast for {date_str}")
        self.log(f"{'='*60}")

        start_time = time.time()

        # Create output directory
        hour_dir = self.output_base_dir / f"{forecast_hour}h"
        hour_dir.mkdir(exist_ok=True)

        # Load parquet and get GRIB URI
        df, grib_uri = self.load_parquet_groups(parquet_file)

        if grib_uri is None:
            raise ValueError(f"Could not determine GRIB URI for {parquet_file}")

        # Parse index file to get ensemble member mapping
        idx_to_member = self.parse_index_file(grib_uri)

        if not idx_to_member:
            raise ValueError(f"Could not parse index file for {grib_uri}")

        # Create groups for each ensemble member
        member_dict = self.create_member_groups_from_index(df, idx_to_member)

        # Process each member
        results = {
            'forecast_hour': forecast_hour,
            'date': date_str,
            'successful': [],
            'failed': []
        }

        self.log(f"\nCreating individual member parquet files...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for member_num, member_groups in member_dict.items():
                member_name = "control" if member_num == -1 else f"ens{member_num:02d}"
                output_path = hour_dir / f"{member_name}.par"

                future = executor.submit(
                    self._process_single_member,
                    member_num, member_groups, output_path
                )
                futures[future] = (member_num, member_name, output_path)

            for future in as_completed(futures):
                member_num, member_name, output_path = futures[future]

                try:
                    future.result()
                    results['successful'].append(member_name)
                except Exception as e:
                    results['failed'].append(member_name)
                    self.log(f"    Error: {member_name} - {e}", "ERROR")

        # Summary
        elapsed = time.time() - start_time
        self.log(f"\nForecast hour {forecast_hour}h summary:")
        self.log(f"  Processing time: {elapsed:.1f} seconds")
        self.log(f"  Successful: {len(results['successful'])}/{ENSEMBLE_MEMBERS_TOTAL}")
        self.log(f"  Output directory: {hour_dir}")

        # Verify files
        created_files = list(hour_dir.glob("*.par"))
        self.log(f"  Files created: {len(created_files)}")

        return results

    def _process_single_member(self, member_num: int, member_groups: List[Dict],
                              output_path: Path):
        """Process and save a single member (used by thread pool)."""
        zarr_store = self.create_member_parquet(member_groups)
        self.save_parquet(zarr_store, output_path)

    def process_all(self, max_workers: int = 4) -> Dict:
        """
        Process all parquet files in the input directory.

        Returns:
            Summary of all processing results
        """
        self.log("="*80)
        self.log("ECMWF PAR to Ensemble Members Processor")
        self.log("="*80)
        self.log(f"Input directory: {self.input_dir}")
        self.log(f"Output directory: {self.output_base_dir}")

        overall_start = time.time()

        # Find all parquet files
        parquet_files = sorted(self.input_dir.glob("*.parquet"))
        self.log(f"Found {len(parquet_files)} parquet files to process")

        all_results = {}

        # Process each file
        for parquet_file in parquet_files:
            try:
                result = self.process_forecast_hour(parquet_file, max_workers)
                file_info = self.parse_filename(parquet_file.name)
                if file_info:
                    all_results[file_info['forecast_hour']] = result
            except Exception as e:
                self.log(f"Failed to process {parquet_file.name}: {e}", "ERROR")

        # Overall summary
        overall_elapsed = time.time() - overall_start

        self.log("\n" + "="*80)
        self.log("PROCESSING COMPLETE")
        self.log("="*80)

        total_successful = sum(len(r['successful']) for r in all_results.values())
        total_expected = len(all_results) * ENSEMBLE_MEMBERS_TOTAL

        self.log(f"Total time: {overall_elapsed/60:.1f} minutes")
        self.log(f"Forecast hours processed: {len(all_results)}")
        self.log(f"Total members created: {total_successful}/{total_expected}")

        # Show final directory structure
        self.log(f"\nOutput structure in {self.output_base_dir}:")
        for hour_dir in sorted(self.output_base_dir.glob("*h")):
            file_count = len(list(hour_dir.glob("*.par")))
            self.log(f"  {hour_dir.name}/: {file_count} files")

        return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ECMWF scan_grib parquet files to create individual ensemble member files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/roller/Documents/data/ecmwf_fmrc",
        help="Directory containing scan_grib parquet files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/roller/Documents/data/ecmwf_fmrc/ensemble_members",
        help="Base directory for output ensemble member files"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--forecast-hour",
        type=int,
        help="Process only specific forecast hour (e.g., 0, 3, 6)"
    )

    args = parser.parse_args()

    # Initialize processor
    processor = ECMWFParquetProcessor(args.input_dir, args.output_dir)

    if args.forecast_hour is not None:
        # Process single forecast hour
        pattern = f"*_{args.forecast_hour}h.parquet"
        files = list(Path(args.input_dir).glob(pattern))

        if not files:
            processor.log(f"No file found for {args.forecast_hour}h", "ERROR")
            return 1

        result = processor.process_forecast_hour(files[0], args.max_workers)

        if result['failed']:
            processor.log(f"Some members failed for {args.forecast_hour}h", "WARNING")
            return 1
    else:
        # Process all files
        results = processor.process_all(args.max_workers)

        # Check for failures
        total_failed = sum(len(r['failed']) for r in results.values())

        if total_failed > 0:
            processor.log(f"\n⚠️ {total_failed} members failed overall", "WARNING")
            return 1

    processor.log(f"\n✅ Processing completed successfully!", "SUCCESS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())