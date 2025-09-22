# GEFS East Africa Ensemble Processing Pipeline

## Overview

Complete pipeline for processing GEFS (Global Ensemble Forecast System) ensemble data for the East Africa region using the **grib-index-kerchunk method** for efficient GRIB data access without full file scanning.

## 🔑 CRITICAL: The Two-Step Architecture

### Understanding the Grib-Index-Kerchunk Method

This pipeline uses a **two-step process** that separates expensive one-time preprocessing from fast daily processing:

#### **Step 0: One-Time Expensive Preprocessing** (ONLY ONCE PER ENSEMBLE MEMBER)
```bash
# Run ONCE to create reusable parquet mapping templates in GCS
python run_gefs_preprocessing.py  # Wrapper script for all members
# OR for individual member:
python gefs_index_preprocessing_fixed.py --date 20241112 --run 00 --member gep01 --bucket gik-gefs-aws-tf
```

**Purpose**: Create parquet mapping files that describe the GRIB data structure
- **When**: Run ONCE per ensemble member to create mapping templates
- **Creates**: Parquet files in GCS at `gs://bucket/gefs/{member}/gefs-time-{date}-{member}-rt{hour}.parquet`
- **Why expensive**: Scans actual GRIB files to build complete index mappings
- **⚡ Key benefit**: These parquet files can be reused across different dates!

#### **Daily Processing Workflow** (RUN FOR EACH NEW FORECAST)

```bash
# Step 1: Fast parquet creation using existing GCS mappings
python run_day_gefs_ensemble_full.py  # Uses mappings from Step 0

# Step 3: Convert all 30 ensemble members to zarr
for i in $(seq -f "%02g" 1 30); do
    python run_single_gefs_to_zarr.py 20250709 00 gep$i \
        --region east_africa \
        --variables t2m,tp,u10,v10,cape,sp,mslet,pwat
done

# Step 4: Concatenate ensemble and compute statistics
python process_ensemble_by_variable.py zarr_stores/20250709_00/

# Step 4: Create plots (optional)
python run_gefs_24h_accumulation.py
python plot_ensemble_east_africa.py 
```

### How the Fast Daily Processing Works

1. **Uses existing parquet mappings** from GCS (created in Step 0)
2. **Reads only GRIB index** (.idx) files from new date's S3 data
3. **Combines** existing mapping structure with new index data
4. **Avoids expensive GRIB scanning** by reusing parquet mappings
5. **Result**: 10-100x faster processing for daily runs!

## References

- [NOAA GEFS on AWS](https://registry.opendata.aws/noaa-gefs/)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- Complete technical details: `docs/GEFS_Complete_Documentation.md`

## Acknowledgements

This work was funded in part by:

1. Hazard modeling, impact estimation, climate storylines for event catalogue
   on drought and flood disasters in the Eastern Africa (E4DRR) project.
   https://icpac-igad.github.io/e4drr/ United Nations | Complex Risk Analytics
   Fund (CRAF'd) 
2. The Strengthening Early Warning Systems for Anticipatory Action (SEWAA)
   Project. https://cgan.icpac.net/

