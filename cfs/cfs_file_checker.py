#!/usr/bin/env python3
"""
CFS File Checker - Check for NOAA Climate Forecast System (CFS) flux files and index files

This script checks for CFS flux files (.grb2) and their corresponding index files (.idx)
in the NOAA S3 bucket without requiring credentials. Based on CFS_Data_Structure.md documentation.

File Pattern: flxf{YYYYMMDDHH}.{MEMBER}.{INIT_DATE}.grb2[.idx]
Location: s3://noaa-cfs-pds/cfs.{date}/{run}/6hrly_grib_01/

Usage:
    python cfs_file_checker.py --date 20250701 --run 00 --member 01
    python cfs_file_checker.py --date 20250701 --run 00 --member 01 --forecast-hours 24
    python cfs_file_checker.py --date 20250701 --run 00 --member 01 --list-all
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import fsspec
import pandas as pd


class CFSFileChecker:
    """Check for CFS flux files and index files in NOAA S3 bucket"""
    
    def __init__(self):
        """Initialize CFS file checker with S3 filesystem"""
        self.fs = fsspec.filesystem('s3', anon=True)  # Anonymous access to public bucket
        self.base_bucket = "noaa-cfs-pds"
        
    def generate_cfs_file_path(self, date: str, run: str, member: str, 
                              forecast_datetime: datetime, init_date: str) -> Tuple[str, str]:
        """
        Generate CFS flux file paths (.grb2 and .idx)
        
        Args:
            date: Init date (YYYYMMDD)
            run: Model run (00, 06, 12, 18)
            member: Ensemble member (01, 02, 03, 04)
            forecast_datetime: Target forecast time
            init_date: Model initialization date (YYYYMMDD + run)
            
        Returns:
            Tuple of (grb2_path, idx_path)
        """
        forecast_str = forecast_datetime.strftime("%Y%m%d%H")
        
        # Path structure: s3://noaa-cfs-pds/cfs.{date}/{run}/6hrly_grib_{member}/
        base_path = f"s3://{self.base_bucket}/cfs.{date}/{run}/6hrly_grib_{member}"
        
        # File pattern: flxf{YYYYMMDDHH}.{MEMBER}.{INIT_DATE}.grb2
        filename_base = f"flxf{forecast_str}.{member}.{init_date}"
        grb2_file = f"{base_path}/{filename_base}.grb2"
        idx_file = f"{base_path}/{filename_base}.grb2.idx"
        
        return grb2_file, idx_file
    
    def check_file_exists(self, file_path: str) -> Dict[str, Any]:
        """
        Check if a file exists and get its information
        
        Args:
            file_path: Full S3 path to file
            
        Returns:
            Dict with existence status and file info
        """
        try:
            # Remove s3:// prefix for fsspec
            path = file_path.replace('s3://', '')
            
            if self.fs.exists(path):
                info = self.fs.info(path)
                return {
                    'exists': True,
                    'size': info.get('size', 0),
                    'size_mb': round(info.get('size', 0) / (1024 * 1024), 2),
                    'last_modified': info.get('LastModified', 'Unknown'),
                    'path': file_path
                }
            else:
                return {
                    'exists': False,
                    'size': 0,
                    'size_mb': 0,
                    'last_modified': None,
                    'path': file_path
                }
        except Exception as e:
            return {
                'exists': False,
                'error': str(e),
                'path': file_path
            }
    
    def generate_forecast_times(self, init_datetime: datetime, max_hours: int = 240) -> List[datetime]:
        """
        Generate forecast times for CFS (6-hourly intervals)
        
        Args:
            init_datetime: Initialization datetime
            max_hours: Maximum forecast hours to generate
            
        Returns:
            List of forecast datetime objects
        """
        forecast_times = []
        current_time = init_datetime
        
        # CFS uses 6-hourly intervals
        for hours in range(0, max_hours + 1, 6):
            forecast_times.append(current_time + timedelta(hours=hours))
            
        return forecast_times
    
    def check_cfs_files(self, date: str, run: str, member: str, 
                       max_forecast_hours: int = 240) -> pd.DataFrame:
        """
        Check for CFS files for a given date, run, and member
        
        Args:
            date: Init date (YYYYMMDD)
            run: Model run (00, 06, 12, 18)
            member: Ensemble member (01, 02, 03, 04)
            max_forecast_hours: Maximum forecast hours to check
            
        Returns:
            DataFrame with file check results
        """
        print(f"Checking CFS files for: {date} {run}Z member {member}")
        print(f"Forecast hours: 0 to {max_forecast_hours} (6-hourly intervals)")
        print("-" * 70)
        
        # Generate initialization datetime
        init_datetime = datetime.strptime(f"{date}{run}", "%Y%m%d%H")
        init_date_str = f"{date}{run}"
        
        # Generate forecast times
        forecast_times = self.generate_forecast_times(init_datetime, max_forecast_hours)
        
        results = []
        
        for i, forecast_time in enumerate(forecast_times):
            # Generate file paths
            grb2_path, idx_path = self.generate_cfs_file_path(
                date, run, member, forecast_time, init_date_str
            )
            
            # Check both files
            grb2_info = self.check_file_exists(grb2_path)
            idx_info = self.check_file_exists(idx_path)
            
            forecast_hour = i * 6  # 6-hourly intervals
            
            results.append({
                'forecast_hour': forecast_hour,
                'forecast_time': forecast_time.strftime("%Y-%m-%d %H:%M"),
                'grb2_exists': grb2_info['exists'],
                'grb2_size_mb': grb2_info['size_mb'],
                'idx_exists': idx_info['exists'],
                'idx_size_mb': idx_info['size_mb'],
                'both_exist': grb2_info['exists'] and idx_info['exists'],
                'grb2_path': grb2_path,
                'idx_path': idx_path
            })
            
            # Print progress every 10 files
            if (i + 1) % 10 == 0:
                grb2_count = sum(1 for r in results if r['grb2_exists'])
                idx_count = sum(1 for r in results if r['idx_exists'])
                print(f"Checked {i+1:3d} forecast hours - "
                      f"GRB2: {grb2_count:3d} found, IDX: {idx_count:3d} found")
        
        return pd.DataFrame(results)
    
    def list_available_files(self, date: str, run: str, member: str, 
                           pattern: str = "flxf*.grb2.idx") -> List[str]:
        """
        List all available files matching pattern in the CFS directory
        
        Args:
            date: Init date (YYYYMMDD)
            run: Model run (00, 06, 12, 18)
            member: Ensemble member (01, 02, 03, 04)
            pattern: File pattern to match
            
        Returns:
            List of available file paths
        """
        directory = f"{self.base_bucket}/cfs.{date}/{run}/6hrly_grib_{member}/"
        
        try:
            files = self.fs.glob(f"{directory}{pattern}")
            return [f"s3://{f}" for f in files]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def summary_report(self, df: pd.DataFrame) -> None:
        """Print summary report of file check results"""
        total_files = len(df)
        grb2_found = df['grb2_exists'].sum()
        idx_found = df['idx_exists'].sum()
        both_found = df['both_exist'].sum()
        
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        print(f"Total forecast hours checked: {total_files}")
        print(f"GRB2 files found: {grb2_found:4d} ({100*grb2_found/total_files:.1f}%)")
        print(f"IDX files found:  {idx_found:4d} ({100*idx_found/total_files:.1f}%)")
        print(f"Complete pairs:   {both_found:4d} ({100*both_found/total_files:.1f}%)")
        
        if grb2_found > 0:
            total_grb2_size = df[df['grb2_exists']]['grb2_size_mb'].sum()
            avg_grb2_size = df[df['grb2_exists']]['grb2_size_mb'].mean()
            print(f"\nGRB2 files total size: {total_grb2_size:.1f} MB")
            print(f"Average GRB2 size: {avg_grb2_size:.1f} MB")
        
        if idx_found > 0:
            total_idx_size = df[df['idx_exists']]['idx_size_mb'].sum()
            avg_idx_size = df[df['idx_exists']]['idx_size_mb'].mean()
            print(f"IDX files total size: {total_idx_size:.1f} MB")
            print(f"Average IDX size: {avg_idx_size:.2f} MB")
        
        # Show missing files
        missing_grb2 = df[~df['grb2_exists']]
        missing_idx = df[~df['idx_exists']]
        
        if len(missing_grb2) > 0:
            print(f"\nMissing GRB2 files (first 5):")
            for _, row in missing_grb2.head().iterrows():
                print(f"  Hour {row['forecast_hour']:3d}: {row['grb2_path']}")
        
        if len(missing_idx) > 0:
            print(f"\nMissing IDX files (first 5):")
            for _, row in missing_idx.head().iterrows():
                print(f"  Hour {row['forecast_hour']:3d}: {row['idx_path']}")


def main():
    """Main function to handle command line arguments and run checks"""
    parser = argparse.ArgumentParser(
        description="Check for CFS flux files and index files in NOAA S3 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check first 24 hours for member 01
  python cfs_file_checker.py --date 20250701 --run 00 --member 01 --forecast-hours 24
  
  # Check default range (240 hours = 10 days)
  python cfs_file_checker.py --date 20250701 --run 00 --member 01
  
  # List all available idx files
  python cfs_file_checker.py --date 20250701 --run 00 --member 01 --list-all
  
  # Check longer forecast range (1 month)
  python cfs_file_checker.py --date 20250701 --run 00 --member 01 --forecast-hours 720
        """
    )
    
    parser.add_argument('--date', required=True, 
                       help='Initialization date (YYYYMMDD)')
    parser.add_argument('--run', required=True, choices=['00', '06', '12', '18'],
                       help='Model run hour (00, 06, 12, 18)')
    parser.add_argument('--member', required=True, choices=['01', '02', '03', '04'],
                       help='Ensemble member (01, 02, 03, 04)')
    parser.add_argument('--forecast-hours', type=int, default=240,
                       help='Maximum forecast hours to check (default: 240)')
    parser.add_argument('--list-all', action='store_true',
                       help='List all available files in the directory')
    parser.add_argument('--output', type=str,
                       help='Save results to CSV file')
    parser.add_argument('--idx-only', action='store_true',
                       help='Only check for .idx files')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYYMMDD format.")
        sys.exit(1)
    
    # Initialize checker
    checker = CFSFileChecker()
    
    print(f"CFS File Checker")
    print(f"Checking NOAA S3 bucket: s3://noaa-cfs-pds/")
    print(f"Target: cfs.{args.date}/{args.run}/6hrly_grib_{args.member}/")
    print()
    
    if args.list_all:
        print("Listing all available index files...")
        idx_files = checker.list_available_files(args.date, args.run, args.member, "flxf*.grb2.idx")
        grb2_files = checker.list_available_files(args.date, args.run, args.member, "flxf*.grb2")
        
        print(f"\nFound {len(idx_files)} .idx files")
        print(f"Found {len(grb2_files)} .grb2 files")
        
        if idx_files:
            print("\nAvailable .idx files (first 10):")
            for i, file_path in enumerate(idx_files[:10]):
                filename = file_path.split('/')[-1]
                print(f"  {i+1:3d}. {filename}")
            
            if len(idx_files) > 10:
                print(f"  ... and {len(idx_files) - 10} more files")
    else:
        # Check specific files
        results_df = checker.check_cfs_files(
            args.date, args.run, args.member, args.forecast_hours
        )
        
        # Display summary
        checker.summary_report(results_df)
        
        # Save to file if requested
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        
        # Show sample of found files
        found_files = results_df[results_df['both_exist']]
        if len(found_files) > 0:
            print(f"\nSample of complete file pairs found:")
            for _, row in found_files.head(3).iterrows():
                print(f"  Hour {row['forecast_hour']:3d}: {row['forecast_time']}")
                print(f"    GRB2: {row['grb2_size_mb']:.1f} MB")
                print(f"    IDX:  {row['idx_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()