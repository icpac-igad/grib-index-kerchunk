import os
import argparse
from dotenv import load_dotenv
import coiled
from dask.distributed import Client

from utils import download_parquet_from_gcs
from utils import process_and_upload_datatree


def main(date_str: str, run_str: str):
    # Load environment variables from the specified file
    load_dotenv(dotenv_path='./env_gik')
    year=date_str[0:4]
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcp_service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    project_id = os.getenv("GCP_PROJECT_ID")

    if not gcs_bucket_name or not gcp_service_account_json or not project_id:
        raise ValueError("GCS_BUCKET_NAME, GCP_SERVICE_ACCOUNT_JSON, or GCP_PROJECT_ID not set in the environment.")

    # Setup cluster
    cluster = coiled.Cluster(
        n_workers=2,
        name=f"gfs-par-zarr-make-{date_str}",
        software="gik-coiled-v6",
        scheduler_vm_types=["n2-standard-4"],
        worker_vm_types="n2-standard-4",
        region="us-east1",
        arm=False,
        compute_purchase_option="spot",
        tags={"workload": "gfs-upload"},
    )
    client = Client(cluster)

    try:
        # Define paths and configurations
        # Download and save Parquet file locally
        parquet_path = download_parquet_from_gcs(
        gcs_bucket_name=gcs_bucket_name,
        year=year,
        date_str=date_str,
        run_str=run_str,
        service_account_json=gcp_service_account_json)

        #parquet_path = f"gfs_{date_str}.par"
        gcs_path = f"cgan_gfs_var/{date_str}"

        # Process and upload
        results = process_and_upload_datatree(
            parquet_path=parquet_path,
            gcs_bucket=gcs_bucket_name,
            gcs_path=gcs_path,
            client=client,
            credentials_path=gcp_service_account_json,
            date_str=date_str,
            project_id=project_id,
        )

        # Print results
        for result in results:
            print(result)

    finally:
        # Close the cluster and client
        client.close()
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream gik parquet file into zarr in GCS.")
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format.")
    parser.add_argument("run_str", type=str, help="Run string (e.g., '00', '06', '12', '18').")
    args = parser.parse_args()
    main(args.date_str, args.run_str)

