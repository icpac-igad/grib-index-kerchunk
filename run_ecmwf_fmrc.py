import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from utils import (
    generate_axes,
    cs_create_mapped_index,
    create_parquet_df,
    nonclusterworker_upload_to_gcs,
)

def main(date_str: str, ecmwf_hr: str):
    # Load environment variables
    load_dotenv(dotenv_path='./env_gik')
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcp_service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    if not gcs_bucket_name or not gcp_service_account_json:
        raise ValueError("GCS_BUCKET_NAME or GCP_SERVICE_ACCOUNT_JSON not set in the environment.")

    # Generate axes for processing
    axes = generate_axes(date_str)

    # Define ECMWF file paths
    ecmwf_files = [
        f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/20230529-{ecmwf_hr}h-enfo-ef.grib2",
    ]

    # Create mapped index
    gfs_kind = cs_create_mapped_index(
        axes, gcs_bucket_name, date_str, gcp_service_account_json=gcp_service_account_json
    )

    # Create Parquet DataFrame
    zdf = create_parquet_df(gfs_kind, date_str, ecmwf_hr)

    # Save to Parquet file
    output_parquet_file = f"ecmwf_{date_str}_{ecmwf_hr}.par"
    zdf.to_parquet(output_parquet_file)
    nonclusterworker_upload_to_gcs(gcs_bucket_name, output_parquet_file, f'gik_day_parqs/{date_str}/{output_parquet_file}', gcp_service_account_json)
    print(f"Parquet file saved: {output_parquet_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ECMWF data and save as a Parquet file.")
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format.")
    parser.add_argument("ecmwf_hr", type=str, help="ECMWF hour (e.g., '024', '048', '072').")

    args = parser.parse_args()
    main(args.date_str, args.ecmwf_hr)
