#!/usr/bin/env python3
"""
GEFS Ensemble Zarr Stream Processing
This script processes GEFS ensemble parquet files and converts them to zarr stores 
using a coiled dask cluster, following the pattern from GFS implementation.

Works with local parquet files created by run_day_gefs_ensemble_full.py
"""

from asyncio import run
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import coiled
from dask.distributed import Client

from utils import StorageType
from utils import process_and_upload_datatree


def main(date_str: str, run_str: str, parquet_dir: str = None, ensemble_member: str = None, storage_type: str = "local"):
    """
    Process GEFS ensemble data from parquet to zarr using coiled dask cluster.
    
    Args:
        date_str: Date string in YYYYMMDD format
        run_str: Run string (e.g., '00', '06', '12', '18')
        parquet_dir: Directory containing GEFS parquet files (defaults to {date_str}_{run_str})
        ensemble_member: Specific ensemble member (e.g., 'gep01'), if None processes all
        storage_type: "local" or "gcs"
    """
    # Load environment variables from the specified file
    load_dotenv(dotenv_path='./env_gik')
    year = date_str[0:4]
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcp_service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    project_id = os.getenv("PROJECT_ID")
    
    storage_choice = StorageType.GCS if storage_type.lower() == "gcs" else StorageType.LOCAL
    
    # East Africa bounding box (matching GEFS extent)
    latlon_bounds = {
        "lat_min": -15.0,
        "lat_max": 25.0,
        "lon_min": 15.0,
        "lon_max": 55.0
    }

    if not gcs_bucket_name or not gcp_service_account_json or not project_id:
        raise ValueError("GCS_BUCKET_NAME, GCP_SERVICE_ACCOUNT_JSON, or GCP_PROJECT_ID not set in the environment.")

    # Set up parquet directory
    if parquet_dir is None:
        parquet_dir = f"{date_str}_{run_str}"
    
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        raise ValueError(f"Parquet directory {parquet_path} does not exist. Run run_day_gefs_ensemble_full.py first.")

    # Setup cluster
    cluster_name = f"gefs-par-zarr-stream-{date_str}"
    if ensemble_member:
        cluster_name += f"-{ensemble_member}"
    
    cluster = coiled.Cluster(
        n_workers=2,
        name=cluster_name,
        software="gik-zarr2",
        scheduler_vm_types=["n2-standard-4"],
        worker_vm_types="n2-standard-4",
        region="us-east1",
        arm=False,
        compute_purchase_option="spot",
        tags={"workload": "gefs-upload"},
    )
    client = Client(cluster)

    try:
        # Get list of parquet files
        if ensemble_member:
            parquet_files = list(parquet_path.glob(f"{ensemble_member}.par"))
            if not parquet_files:
                raise ValueError(f"No parquet file found for ensemble member {ensemble_member}")
        else:
            parquet_files = sorted(parquet_path.glob("gep*.par"))
            if not parquet_files:
                raise ValueError(f"No GEFS parquet files found in {parquet_path}")
        
        print(f"🔍 Found {len(parquet_files)} parquet files to process")
        
        results = []
        
        for parquet_file in parquet_files:
            member = parquet_file.stem  # Extract member name from filename
            print(f"\n🚀 Processing ensemble member: {member}")
            
            # Define GCS path for this ensemble member
            gcs_path = f"cgan_gefs_var/{date_str}{run_str}/{member}"
            local_save_path = f"./zarr_stores/{date_str}{run_str}/{member}"

            # Process and upload
            try:
                member_results = process_and_upload_datatree(
                    parquet_path=str(parquet_file),
                    gcs_bucket=gcs_bucket_name,
                    gcs_path=gcs_path,
                    client=client,
                    credentials_path=gcp_service_account_json,
                    date_str=date_str,
                    run_str=run_str,
                    project_id=project_id,
                    storage_type=storage_choice,
                    local_save_path=local_save_path,
                    latlon_bounds=latlon_bounds
                )
                results.extend(member_results)
                print(f"✅ Successfully processed {member}")
            except Exception as e:
                print(f"❌ Failed to process {member}: {e}")
                continue

        # Print results
        print(f"\n📊 Processing Summary:")
        print(f"   Total members processed: {len(results)}")
        for result in results:
            print(f"   {result}")

    finally:
        # Close the cluster and client
        client.close()
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream GEFS ensemble parquet files into zarr in GCS.")
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format.")
    parser.add_argument("run_str", type=str, help="Run string (e.g., '00', '06', '12', '18').")
    parser.add_argument("--parquet_dir", type=str, help="Directory containing GEFS parquet files (defaults to {date_str}_{run_str})")
    parser.add_argument("--member", type=str, help="Specific ensemble member (e.g., 'gep01'). If not specified, processes all members.")
    parser.add_argument("--storage", type=str, default="local", choices=["local", "gcs"], 
                       help="Storage type: 'local' or 'gcs' (default: local)")
    
    args = parser.parse_args()
    main(args.date_str, args.run_str, args.parquet_dir, args.member, args.storage)