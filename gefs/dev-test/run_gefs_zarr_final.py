#!/usr/bin/env python3
"""
GEFS Ensemble Zarr Processing - Final Working Version
This script processes GEFS ensemble parquet files and converts them to zarr stores 
using a coiled dask cluster, uploading parquet files to workers first.
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import coiled
from dask.distributed import Client
import pandas as pd
import json
import fsspec
import xarray as xr


def read_parquet_fixed(parquet_content):
    """Read parquet files from bytes content."""
    import ast
    import io
    
    # Convert bytes to DataFrame
    df = pd.read_parquet(io.BytesIO(parquet_content))
    print(f"📊 Parquet file loaded: {len(df)} rows")
    
    if 'refs' in df['key'].values and len(df) <= 2:
        # Old format - single refs row
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
        print(f"✅ Extracted {len(zstore)} entries from old format")
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
        
        print(f"✅ Loaded {len(zstore)} entries from new format")
    
    if 'version' in zstore:
        del zstore['version']
    
    return zstore


def save_zarr_locally_simple(zstore, member_name, local_save_path):
    """Save zarr store locally using simple approach."""
    print(f"💾 Saving {member_name} zarr store locally...")
    
    try:
        # Create reference filesystem
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")
        
        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        
        # Save precipitation data locally
        if '/tp/accum/surface' in dt.groups:
            tp_data = dt['/tp/accum/surface'].ds
            output_path = f"{local_save_path}/{member_name}_tp.zarr"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tp_data.to_zarr(output_path, mode='w')
            print(f"✅ {member_name} zarr store saved to: {output_path}")
            return output_path
        else:
            print(f"⚠️ No precipitation data found for {member_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error saving {member_name} locally: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_member_with_content(parquet_content, member_name, local_save_path):
    """Process a single member on a dask worker with parquet content."""
    print(f"🚀 Worker processing: {member_name}")
    
    # Read parquet file from content
    zstore = read_parquet_fixed(parquet_content)
    
    # Save locally
    result = save_zarr_locally_simple(zstore, member_name, local_save_path)
    
    return f"{member_name}: {result if result else 'Failed'}"


def main(date_str: str, run_str: str, parquet_dir: str = None, ensemble_member: str = None):
    """
    Process GEFS ensemble data from parquet to zarr using coiled dask cluster.
    """
    # Set up parquet directory
    if parquet_dir is None:
        parquet_dir = f"{date_str}_{run_str}"
    
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        raise ValueError(f"Parquet directory {parquet_path} does not exist. Run run_day_gefs_ensemble_full.py first.")

    # Setup cluster
    cluster_name = f"gefs-final-zarr-{date_str}"
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
        tags={"workload": "gefs-final-upload"},
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
        
        # Submit tasks to workers
        futures = []
        for parquet_file in parquet_files:
            member = parquet_file.stem  # Extract member name from filename
            
            # Read parquet content locally
            print(f"📖 Reading {member} parquet file locally...")
            with open(parquet_file, 'rb') as f:
                parquet_content = f.read()
            
            # Define local save path for this ensemble member
            local_save_path = f"./zarr_stores/{date_str}{run_str}"
            
            # Submit to worker
            future = client.submit(
                process_member_with_content,
                parquet_content,
                member,
                local_save_path
            )
            futures.append(future)
        
        # Collect results
        print(f"\n⏳ Waiting for workers to complete...")
        results = client.gather(futures)
        
        # Print results
        print(f"\n📊 Processing Summary:")
        successful = 0
        for result in results:
            print(f"   {result}")
            if "Failed" not in result:
                successful += 1
        
        print(f"   Total members processed successfully: {successful}/{len(results)}")
        
        return successful > 0

    finally:
        # Close the cluster and client
        client.close()
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final GEFS ensemble zarr processing with dask.")
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format.")
    parser.add_argument("run_str", type=str, help="Run string (e.g., '00', '06', '12', '18').")
    parser.add_argument("--parquet_dir", type=str, help="Directory containing GEFS parquet files (defaults to {date_str}_{run_str})")
    parser.add_argument("--member", type=str, help="Specific ensemble member (e.g., 'gep01'). If not specified, processes all members.")
    
    args = parser.parse_args()
    success = main(args.date_str, args.run_str, args.parquet_dir, args.member)
    
    if not success:
        exit(1)