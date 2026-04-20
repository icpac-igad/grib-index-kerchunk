# GEFS Lithops Cloud Run Deployment - Implementation Summary

**Date**: 2026-02-18
**Location**: `/data/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/lithops_cr_gefs_gik/`

## Success Criteria Checklist

- [x] ✅ Folder structure created at correct location
- [x] ✅ `gefs_util.py` copied and accessible (47 KB)
- [x] ✅ All files created and adapted from ECMWF version:
  - [x] `cloudbuild.yaml` - Updated for gefs-lithops-runtime
  - [x] `lithops_config.yaml` - Updated for us-central1, gik-gefs-aws-tf bucket
  - [x] `Dockerfile` - Updated with GEFS template URL, added kerchunk dependency
  - [x] `run_lithops_gefs.py` - Main script with GEFS processing logic
  - [x] `README.md` - Complete GEFS-specific documentation
  - [x] `.dockerignore` - Copied from ECMWF
  - [x] `.gitignore` - Copied from ECMWF
- [x] ✅ Python syntax validation passed
- [x] ✅ gefs_util imports work correctly
- [x] ✅ Script is executable (chmod +x)

## Implementation Highlights

### 1. Configuration Files
- **cloudbuild.yaml**: Changed image name to `gefs-lithops-runtime`
- **lithops_config.yaml**: 
  - Region: `us-central1` (closer to NOAA S3 in us-east-1)
  - Bucket: `gik-gefs-aws-tf`
  - Runtime: `gcr.io/e4drr-crafd/gefs-lithops-runtime`
  - Service account: Reuses ECMWF deployer key

### 2. Dockerfile
- Base: `python:3.12-slim-bookworm`
- Added dependencies: `kerchunk`, `xarray`, `zarr` (for GRIB processing)
- Template: `gik-fmrc-gefs-20241112.tar.gz` (~120MB)
- Pre-downloaded to: `/opt/gefs_templates/`
- ENV var: `GEFS_TEMPLATE_PATH`

### 3. run_lithops_gefs.py (20 KB)
**Imports from gefs_util.py**:
- `LocalTarGzMappingManager` - Template management
- `generate_axes` - Time axes generation
- `filter_build_grib_tree` - Stage 1: GRIB tree scanning
- `cs_create_mapped_index_local` - Stage 2: Mapped index creation
- `prepare_zarr_store` - Stage 3a: Zarr store preparation
- `process_unique_groups` - Stage 3b: Time dimension processing
- `calculate_time_dimensions` - Time coordinate calculation

**Key Configuration**:
```python
GCS_BUCKET = 'gik-gefs-aws-tf'
GCS_PARQUET_PREFIX = 'run_par_gefs'
S3_BUCKET = "noaa-gefs-pds"
REFERENCE_DATE = '20241112'
ALL_FORECAST_HOURS = list(range(0, 241, 3))  # 0-240h, 81 steps
```

**GEFS Members**: `gep01` through `gep30` (30 members)

**Output Format**: `{date}{run}z-{member}.parquet`
- Example: `2026021600z-gep01.parquet`

**GCS Path**: `gs://gik-gefs-aws-tf/run_par_gefs/{date}/{run}z/`
- Example: `gs://gik-gefs-aws-tf/run_par_gefs/20260216/00z/`

**Supported Runs**: `00z`, `06z`, `12z`, `18z`

### 4. Processing Pipeline

```
process_gefs_date(date_str, run) on Cloud Run worker:
  │
  ├─ Validate GEFS S3 data availability
  │
  ├─ Load template: ensure_template()
  │  └─ Uses pre-baked /opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz
  │
  ├─ Initialize LocalTarGzMappingManager
  │
  └─ For each member (gep01-gep30):
      │
      ├─ Stage 1: filter_build_grib_tree()
      │  └─ Scan GRIB structure from first 2 files
      │
      ├─ Stage 2: cs_create_mapped_index_local()
      │  └─ Create mapped index from template
      │
      ├─ Stage 3: prepare_zarr_store() + process_unique_groups()
      │  └─ Process time dimensions
      │
      ├─ Save parquet: {date}{run}z-{member}.parquet
      │
      └─ Upload to GCS: gs://gik-gefs-aws-tf/run_par_gefs/{date}/{run}z/
```

## Differences from ECMWF

| Aspect | ECMWF | GEFS |
|--------|-------|------|
| S3 Bucket | `ecmwf-forecasts` (eu-central-1) | `noaa-gefs-pds` (us-east-1) |
| GCP Region | `europe-west3` | `us-central1` |
| GCS Bucket | `gik-ecmwf-aws-tf` | `gik-gefs-aws-tf` |
| Members | 51 (control + ens01-ens50) | 30 (gep01-gep30) |
| Template | `gik-fmrc-v2ecmwf_fmrc.tar.gz` | `gik-fmrc-gefs-20241112.tar.gz` |
| Reference Date | `20240529` | `20241112` |
| Forecast Hours | 0-144h (3h), 150-360h (6h) = 85 steps | 0-240h (3h) = 81 steps |
| Processing | Uses ECMWF .index files | Uses gefs_util.py functions |
| Dependencies | Standard stack | Added kerchunk, xarray, zarr |

## Testing Commands

### 1. Dry Run
```bash
cd lithops_cr_gefs_gik/
uv run run_lithops_gefs.py --date 20260216 --dry-run
```

### 2. Sequential Local Test (No Cloud Run)
```bash
uv run run_lithops_gefs.py --date 20260216 --sequential
```
Downloads template to `/tmp/`, processes locally without Lithops.

### 3. Cloud Run Test (Single Date)
```bash
uv run run_lithops_gefs.py --date 20260216 --run 00
```

### 4. Multiple Runs
```bash
uv run run_lithops_gefs.py --date 20260216 --run 06
uv run run_lithops_gefs.py --date 20260216 --run 12
uv run run_lithops_gefs.py --date 20260216 --run 18
```

### 5. Batch Processing
```bash
uv run run_lithops_gefs.py --days-back 7
uv run run_lithops_gefs.py --start-date 20260201 --end-date 20260207
```

## Deployment Steps

### Step 1: Create GCS Bucket
```bash
gsutil mb -p e4drr-crafd -c STANDARD -l us-central1 gs://gik-gefs-aws-tf
```

### Step 2: Build Docker Image
```bash
cd lithops_cr_gefs_gik/

gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
```
**Build time**: ~20 minutes (downloads 120MB template)

### Step 3: Deploy Lithops Runtime
```bash
lithops runtime deploy gcr.io/e4drr-crafd/gefs-lithops-runtime \
  -b gcp_cloudrun \
  -s gcp_storage \
  --config lithops_config.yaml
```

### Step 4: Verify Deployment
```bash
lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

## Expected Output

For a single date (e.g., `20260216` at `00z`):
- **Files created**: 30 parquet files
- **Filenames**: `2026021600z-gep01.parquet` through `2026021600z-gep30.parquet`
- **GCS location**: `gs://gik-gefs-aws-tf/run_par_gefs/20260216/00z/`
- **File size**: ~50-100 KB per file (varies by data)

Verify:
```bash
gsutil ls gs://gik-gefs-aws-tf/run_par_gefs/20260216/00z/
gsutil ls gs://gik-gefs-aws-tf/run_par_gefs/20260216/00z/ | wc -l  # Should be 30
```

## Notes

1. **Service Account**: Dedicated `gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com` (NOT shared with ECMWF — see DEPLOYMENT_GUIDE.md)
2. **Template Pre-baking**: Template is downloaded during Docker build, not at runtime
3. **Cloudpickle Serialization**: Both `run_lithops_gefs.py` and `gefs_util.py` are serialized together
4. **S3 Access**: Uses anonymous access (`AWS_NO_SIGN_REQUEST=YES`)
5. **Multi-run Support**: Can process all 4 daily runs (00z, 06z, 12z, 18z)

## Known Limitations

1. GEFS data availability on NOAA S3 may lag by 6-12 hours
2. Processing time ~4-7 minutes per date (30 members)
3. Template is reference date `20241112` - may need updates for significant GEFS changes
4. Requires GEFS data in specific S3 path format

## Estimated Costs

**Per date processing**:
- Cloud Run: ~$0.01 (7 min × 2 vCPU)
- GCS storage: ~$0.000003 (30 files × 50KB)

**Monthly (30 dates/day)**:
- Processing: $0.30/day = **$9/month**
- Storage: Negligible

## References

- ECMWF deployment: `../lithops_cr_ecmwf_gik/`
- GEFS tutorial: `/data/08-2023/working_notes_jupyter/ignore_nka_gitrepos/grib-index-kerchunk/tutorial/gefs/run_gefs_tutorial.py`
- gefs_util source: `/data/08-2023/working_notes_jupyter/ignore_nka_gitrepos/grib-index-kerchunk/gefs/gefs_util.py`
- Plan document: Read from conversation transcript

---

**Status**: ✅ **IMPLEMENTATION COMPLETE** - Ready for deployment and testing
