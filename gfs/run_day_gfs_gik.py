
import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from utils import (
    generate_axes,
    filter_build_grib_tree,
    KerchunkZarrDictStorageManager,
    calculate_time_dimensions,
    cs_create_mapped_index,
    prepare_zarr_store,
    process_unique_groups,
    create_parquet_df,
    nonclusterworker_upload_to_gcs,
)

def main(date_str: str, run_str: str):
    # Load environment variables
    load_dotenv(dotenv_path='./env_gik')
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcp_service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    if not gcs_bucket_name or not gcp_service_account_json:
        raise ValueError("GCS_BUCKET_NAME or GCP_SERVICE_ACCOUNT_JSON not set in the environment.")

    # Generate axes for processing
    axes = generate_axes(date_str,"60min")

    # Define GFS file paths
    gfs_files = [
        f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/{run_str}/atmos/gfs.t{run_str}z.pgrb2.0p25.f000",
        f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/{run_str}/atmos/gfs.t{run_str}z.pgrb2.0p25.f001",
    ]

    # Forecast variable mappings
    forecast_dict = {
        "Convective available potential energy": "CAPE:surface",
        "Convective precipitation (water)": "ACPCP:surface",
        "Medium cloud cover": "MCDC:middle cloud layer",
        "Surface pressure": "PRES:surface",
        "Surface upward short-wave radiation flux": "USWRF:surface",
        "Surface downward short-wave radiation flux": "DSWRF:surface",
        "2 metre temperature": "TMP:2 m above ground",
        "Cloud water": "CWAT",
        "Precipitable water": "PWAT",
        "Ice water mixing ratio": "ICMR:200 mb",
        "Cloud mixing ratio": "CLMR:200 mb",
        "Rain mixing ratio": "RWMR:200 mb",
        "Total Precipitation": "APCP:surface",
        "U component of wind": "UGRD:200 mb",
        "V component of wind": "VGRD:200 mb",
    }

    # Build GRIB tree
    _, deflated_gfs_grib_tree_store = filter_build_grib_tree(gfs_files, forecast_dict)

    # Calculate time dimensions
    time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)

    # Create mapped index
    gfs_kind = cs_create_mapped_index(
        axes, gcs_bucket_name, date_str, gcp_service_account_json=gcp_service_account_json
    )

    # Prepare Zarr store
    zstore, chunk_index = prepare_zarr_store(deflated_gfs_grib_tree_store, gfs_kind)

    # Process unique groups
    updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)

    # Create Parquet DataFrame
    zdf = create_parquet_df(updated_zstore, date_str, run_str)

    # Save to Parquet file
    output_parquet_file = f"gfs_{date_str}_{run_str}.par"
    zdf.to_parquet(output_parquet_file)
    year=date_str[0:4]
    nonclusterworker_upload_to_gcs(gcs_bucket_name,output_parquet_file,f'gik_day_parqs/{year}/{output_parquet_file}',gcp_service_account_json)
    print(f"Parquet file saved: {output_parquet_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GFS data and save as a Parquet file.")
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format.")
    parser.add_argument("run_str", type=str, help="Run string (e.g., '00', '06', '12', '18').")

    args = parser.parse_args()
    main(args.date_str, args.run_str)

