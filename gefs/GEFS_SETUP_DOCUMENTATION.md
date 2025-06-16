# GEFS Setup Documentation

## Overview

This documentation explains the two-phase approach for processing Global Ensemble Forecast System (GEFS) data, mirroring the efficient GFS processing workflow.

## Comparison: GFS vs GEFS Workflow

### GFS Workflow (Original)

**Phase 1: Pre-processing (One-time per date)**
```python
# GFS pre-processing creates mapping files in GCS
# Location: gfs/utils.py
process_gfs_time_idx_data(s3url, bucket_name)
↓
build_idx_grib_mapping(fs, basename=s3url)
↓ 
Save: gs://bucket/time_idx/{year}/{date}/gfs-time-{date}-rt{forecast_hour:03}.parquet
```

**Phase 2: Main Processing (Fast ~2 minutes)**
```python
# Main GFS processing reads pre-built mappings
# Location: run_day_gfs_gik.py + gfs/utils.py
cs_create_mapped_index() → reads from GCS mappings
↓
Fast processing using pre-built indices
```

### GEFS Workflow (New Implementation)

**Phase 1: Pre-processing (One-time per date/member)**
```python
# GEFS pre-processing creates mapping files in GCS
# Location: gefs_index_preprocessing.py
process_gefs_time_idx_data(s3url, bucket_name)
↓
build_idx_grib_mapping(fs, basename=s3url)
↓
Save: gs://bucket/time_idx/gefs/{year}/{date}/{member}/gefs-time-{date}-{member}-rt{forecast_hour:03}.parquet
```

**Phase 2: Main Processing (Fast ~2 minutes per member)**
```python
# Main GEFS processing reads pre-built mappings
# Location: run_day_gefs_gik.py + gefs_util.py  
cs_create_mapped_index() → reads from GCS mappings
↓
Fast processing using pre-built indices
```

## File Structure

### Created Files

1. **`gefs_index_preprocessing.py`** - Pre-processing module
   - Creates index mappings for GEFS ensemble members
   - Uploads mappings to GCS bucket
   - One-time setup per date/member

2. **`run_day_gefs_gik.py`** - Main GEFS processing script
   - Uses pre-built mappings for fast processing
   - Processes single ensemble member
   - Generates final parquet files

3. **`gefs_util.py`** - GEFS utility functions
   - Adapted from GFS utilities
   - Reads pre-built GCS mappings
   - Handles GEFS-specific data structures

### GCS Storage Structure

```
gs://bucket/
├── time_idx/
│   ├── gfs/                    # GFS mappings
│   │   └── {year}/{date}/
│   │       └── gfs-time-{date}-rt{hour:03}.parquet
│   └── gefs/                   # GEFS mappings  
│       └── {year}/{date}/{member}/
│           └── gefs-time-{date}-{member}-rt{hour:03}.parquet
└── gik_day_parqs/             # Final processed data
    └── {year}/
        ├── gfs_{date}_{run}.par        # GFS final data
        └── gefs_{member}_{date}.par    # GEFS final data
```

## Usage Instructions

### Step 1: Pre-processing (One-time setup)

Run this **once per date** to create index mappings for all GEFS ensemble members:

```python
from gefs_index_preprocessing import main_create_daily_gefs_mappings

# Create mappings for all 30 ensemble members
date_str = "20241112"
bucket_name = "gik-gefs-aws-tf"
results = main_create_daily_gefs_mappings(date_str, bucket_name)
```

Or for a single member:
```python
from gefs_index_preprocessing import main_create_single_member_mappings

results = main_create_single_member_mappings(date_str, "gep01", bucket_name)
```

### Step 2: Main Processing (Fast execution)

After pre-processing is complete, run the main processing:

```python
# For each ensemble member
python run_day_gefs_gik.py
# or
exec(open('run_day_gefs_gik.py').read())
```

## Key Differences: GFS vs GEFS

| Aspect | GFS | GEFS |
|--------|-----|------|
| **Forecast Period** | 5 days | 10 days |
| **Time Intervals** | 1 hour | 3 hours |
| **Ensemble Members** | Single deterministic | 30 members (gep01-gep30) |
| **S3 Bucket** | `noaa-gfs-bdp-pds` | `noaa-gefs-pds` |
| **File Pattern** | `gfs.t00z.pgrb2.0p25.f{hour:03}` | `{member}.t00z.pgrb2s.0p25.f{hour:03}` |
| **GCS Storage** | `time_idx/{year}/{date}/` | `time_idx/gefs/{year}/{date}/{member}/` |

## Performance Benefits

### Without Pre-processing
- **GFS**: ~30+ minutes for full processing
- **GEFS**: ~30+ minutes × 30 members = 15+ hours

### With Pre-processing
- **GFS**: ~2 minutes for full processing
- **GEFS**: ~2 minutes per member = ~1 hour for all 30 members

**Speed improvement**: ~15x faster with pre-processing!

## Technical Details

### GEFS Data Structure
- **Forecast Hours**: 000, 003, 006, ..., 240 (3-hour intervals)
- **Ensemble Members**: gep01, gep02, ..., gep30
- **Variables**: 13 meteorological variables optimized for ensemble forecasting
- **Spatial Resolution**: 0.25° × 0.25° global grid

### Memory Management
- Async processing with semaphores for concurrency control
- Chunked reading for large parquet files
- Automatic cleanup of temporary files

### Error Handling
- Comprehensive logging for debugging
- Graceful handling of missing files
- Verification functions for GCS uploads

## Workflow Commands

### 1. Initial Setup (One-time)
```bash
# Upload credentials to Coiled environment
# Install dependencies: gcsfs, fsspec, kerchunk, etc.
```

### 2. Daily Pre-processing (Once per day)
```python
# Run in Coiled notebook or cluster
from gefs_index_preprocessing import main_create_daily_gefs_mappings
results = main_create_daily_gefs_mappings("20241112", "gik-gefs-aws-tf")
```

### 3. Main Processing (Per ensemble member)
```python
# Modify date_str and ensemble_member in run_day_gefs_gik.py
date_str = '20241112'
ensemble_member = 'gep01'  # Change for each member
exec(open('run_day_gefs_gik.py').read())
```

### 4. Verification
```python
from gefs_index_preprocessing import verify_gefs_mappings_in_gcs
files = verify_gefs_mappings_in_gcs("bucket", "20241112", "gep01", "credentials.json")
```

## Best Practices

1. **Run pre-processing during off-peak hours** - It's computationally intensive
2. **Process ensemble members in parallel** - Use multiple Coiled clusters
3. **Monitor GCS storage costs** - Clean up old mapping files periodically
4. **Verify mappings before main processing** - Use verification functions
5. **Set up alerts for missing data** - Monitor for failed pre-processing jobs

## Troubleshooting

### Common Issues

1. **Missing mapping files**
   ```
   Error: Expected mapping path not found in GCS
   ```
   **Solution**: Run pre-processing for the date/member first

2. **Credentials errors**
   ```
   Error: Failed to upload to GCS
   ```
   **Solution**: Ensure `coiled-data-key.json` is uploaded to workers

3. **Memory issues**
   ```
   Error: Out of memory during processing
   ```
   **Solution**: Reduce batch_size or max_concurrent parameters

4. **Incomplete ensemble**
   ```
   Warning: Only X of 30 members processed
   ```
   **Solution**: Check for missing GEFS data on NOAA servers

## Next Steps

1. **Automation**: Set up scheduled pre-processing jobs
2. **Monitoring**: Implement alerts for failed processing
3. **Optimization**: Fine-tune concurrency parameters
4. **Validation**: Compare GEFS outputs with operational forecasts
5. **Extension**: Add support for additional variables or higher resolution data

This setup provides the foundation for efficient GEFS ensemble processing, enabling rapid analysis of ensemble forecasts for weather prediction and research applications.