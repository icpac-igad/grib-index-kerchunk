import logging
import json
import os
import re
import pandas as pd
import fsspec
from google.cloud import storage
from google.oauth2 import service_account
from kerchunk.grib2 import scan_grib
from kerchunk._grib_idx import _extract_single_group


logger = logging.getLogger("utils-logs")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": self.format_message(record.getMessage()),
            "function": record.funcName,
            "line": record.lineno,
        }
        try:
            return json.dumps(log_data)
        except (TypeError, ValueError):
            return f"{{\"timestamp\": \"{self.formatTime(record, self.datefmt)}\", \"level\": \"{record.levelname}\", \"function\": \"{record.funcName}\", \"line\": {record.lineno}, \"message\": \"{self.format_message(record.getMessage())}\"}}"

    def format_message(self, message):
        return message.replace('\n', ' ')


def setup_logging(log_level: int = logging.INFO, log_file: str = "logfile.log"):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


def nonclusterworker_upload_to_gcs(bucket_name, source_file_name, destination_blob_name, dask_worker_credentials_path):
    try:
        print(f"Using credentials file at: {dask_worker_credentials_path}")
        if not os.path.exists(dask_worker_credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {dask_worker_credentials_path}")
        storage_client = storage.Client.from_service_account_json(dask_worker_credentials_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload file to GCS: {str(e)}")
        raise


def ecmwf_s3_url_maker(date_str):
    fs_s3 = fsspec.filesystem("s3", anon=True)
    s3url_glob = fs_s3.glob(f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/*")
    s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "index"]
    pattern = re.compile(r"\d{14}-\d+h-enfo-ef\.grib2$")
    fmt_s3og = sorted(["s3://" + f for f in s3url_only_grib if pattern.search(f)])
    print(f"Generated {len(fmt_s3og)} URLs for date {date_str}")
    return fmt_s3og


def zip_folder(folder_path, output_zip_path):
    import shutil
    output_zip_path = os.path.splitext(output_zip_path)[0]
    shutil.make_archive(output_zip_path, 'zip', folder_path)
    print(f"Folder '{folder_path}' has been compressed to '{output_zip_path}.zip'")


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' has been created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def _map_grib_file_by_group(fname: str, total_groups: int):
    import time
    import warnings

    start_time = time.time()
    logger.info(f"Starting to process {total_groups} groups from file: {fname}")

    processed_groups = 0
    successful_groups = 0
    failed_groups = 0

    def process_groups():
        nonlocal processed_groups, successful_groups, failed_groups
        for i, group in enumerate(scan_grib(fname), start=1):
            try:
                result = _extract_single_group(group, i)
                processed_groups += 1
                if result is not None:
                    successful_groups += 1
                else:
                    failed_groups += 1
                yield result
            except Exception as e:
                failed_groups += 1
                processed_groups += 1
                logger.info(f"Skipping processing of group {i}: {str(e)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.concat(filter(None, process_groups())).set_index("idx")

    logger.info(f"Completed processing {fname}")
    return df


def s3_parse_ecmwf_grib_idx(fs: fsspec.AbstractFileSystem, basename: str, suffix: str = "index") -> pd.DataFrame:
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"
    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)
    baseinfo = fs.info(basename)

    with fs.open(fname, "r") as f:
        splits = []
        for idx, line in enumerate(f):
            data = json.loads(line.strip().rstrip(','))
            splits.append([
                int(idx),
                int(data.get("_offset", 0)),
                int(data.get("_length", 0)),
                data.get("date", "Unknown Date"),
                data,
                int(data.get("number", -1))
            ])
    result = pd.DataFrame(splits, columns=["idx", "offset", "length", "date", "attr", "ens_number"])
    result["idx_uri"] = fname
    result["grib_uri"] = basename
    result["indexed_at"] = pd.Timestamp.now()

    if "s3" in fs.protocol:
        result["grib_etag"] = baseinfo.get("ETag")
        result["grib_updated_at"] = pd.to_datetime(baseinfo.get("LastModified")).tz_localize(None)
        idxinfo = fs.info(fname)
        result["idx_etag"] = idxinfo.get("ETag")
        result["idx_updated_at"] = pd.to_datetime(idxinfo.get("LastModified")).tz_localize(None)

    print(f'Completed index files and found {len(result.index)} entries')
    return result.set_index("idx")


def s3_ecmwf_scan_grib_storing(fs, basename, date_str, suffix, ecmwf_hr, gcs_bucket_name, gcp_service_account_json):
    idx_file_index = s3_parse_ecmwf_grib_idx(fs, basename, suffix)
    total_groups = len(idx_file_index.index)
    logger.info(total_groups)
    dd = _map_grib_file_by_group(basename, total_groups)
    logger.info(len(dd.index))
    output_parquet_file = f'e_sg_mdt_{date_str}_{ecmwf_hr}.parquet'
    dd.to_parquet(output_parquet_file, engine='pyarrow')

    destination_blob_name = f'fmrc/scan_grib{date_str}/{output_parquet_file}'
    nonclusterworker_upload_to_gcs(
        bucket_name=gcs_bucket_name,
        source_file_name=output_parquet_file,
        destination_blob_name=destination_blob_name,
        dask_worker_credentials_path=gcp_service_account_json
    )

