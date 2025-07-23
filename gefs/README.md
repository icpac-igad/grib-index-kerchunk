## To run the full ens member run and emprical probablity test 

```
coiled notebook start --name dask-thresholds --vm-type n2-standard-2 --software  gik-zarr2 --workspace=geosfm
```
upload the coiled the data service account, and the files 

1. gefs_utils.py 
2. run_day_gefs_ensemble_full.py 
3. run_gefs_24h_accumulation.py 
4. ea_ghcf_simple.geojson

# GEFS Processing Architecture: Crucial Distinction

## The Two-Step Process

### Step 1: One-Time Expensive Preprocessing (run_gefs_preprocessing.py)
- **Purpose**: Create parquet mapping files that describe the GRIB data structure
- **When**: Run ONCE per ensemble member to create the mapping templates
- **What it creates**: Parquet files in GCS at `gs://bucket/gefs/{member}/gefs-time-{date}-{member}-rt{hour}.parquet`
- **Why expensive**: Scans actual GRIB files to build index mappings
- **Reusability**: These parquet files can be reused across different dates!

### Step 2: Fast Daily Processing (run_day_gefs_ensemble_full.py)
- **Purpose**: Process new date's GEFS data using existing parquet structure + new GRIB index
- **When**: Run daily for each new forecast date
- **How it works**:
  1. Uses existing parquet mapping files from GCS (created in Step 1)
  2. Reads GRIB index (.idx) files directly from new date's S3 data
  3. Combines existing mapping structure with new index data
  4. Avoids expensive GRIB scanning by reusing parquet mappings



