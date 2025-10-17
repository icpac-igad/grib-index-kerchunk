# GEFS/CFS Kerchunk-GRIB Index Processing Methodology

## Overview

This document details the kerchunk/GRIB indexing approach used for processing GEFS (Global Ensemble Forecast System) data and provides a comprehensive plan for adapting it to CFS (Climate Forecast System) data.

## GEFS Processing Architecture

### 1. Core Components

#### 1.1 GRIB Tree Building (`filter_build_grib_tree`)
- **Purpose**: Scan GRIB files and build hierarchical tree structure
- **Process**:
  1. Use `scan_grib()` to read GRIB messages from S3 files
  2. Filter messages based on variable dictionary
  3. Build tree structure with `grib_tree()`
  4. Create deflated copy with `strip_datavar_chunks()`

#### 1.2 Mapped Index Creation (`cs_create_mapped_index`)
- **Purpose**: Create efficient index mapping for ensemble members
- **Key Features**:
  - Asynchronous processing with semaphore control
  - Batch processing (default: 5 files per batch)
  - Reference date support for reusing mapping templates
  - Memory-efficient chunked reading

#### 1.3 Zarr Store Preparation
- **Process Flow**:
  1. Generate temporal axes (3-hour intervals for 10-day forecast)
  2. Calculate time dimensions and coordinates
  3. Create mapped index combining fresh GRIB indices with reference mappings
  4. Process unique variable groups
  5. Store as parquet files for efficient access

### 2. Key Data Structures

#### 2.1 GEFS File Pattern
```
s3://noaa-gefs-pds/gefs.{date}/{run}/atmos/pgrb2sp25/{member}.t{run}z.pgrb2s.0p25.f{forecast_hour:03d}
```
- Date format: YYYYMMDD
- Run: 00, 06, 12, 18
- Members: gep01 to gep30
- Forecast hours: 000 to 240 (3-hour intervals)

#### 2.2 Variable Processing
Variables are filtered and processed with specific level types:
- Surface variables: pres, dswrf, cape, uswrf, apcp, gust
- Height above ground: tmp, rh, ugrd, vgrd (2m, 10m)
- Atmospheric: pwat, tcdc
- Cloud ceiling: hgt

### 3. Optimization Techniques

#### 3.1 Reference Mapping Reuse
- Pre-built parquet mappings stored in GCS
- Structure: `gs://{bucket}/time_idx/gefs/{year}/{date}/{member}/gefs-time-{date}-{member}-rt{hour:03d}.parquet`
- Combines fresh GRIB index (binary positions) with template structure

#### 3.2 Parallel Processing
- Asynchronous file processing with ThreadPoolExecutor
- Semaphore-controlled concurrency
- Batch processing to manage memory

#### 3.3 Memory Management
- Chunked parquet reading
- Deflated zarr stores
- Streaming data access pattern

## CFS Adaptation Plan

### 1. CFS Data Structure Analysis

#### 1.1 File Organization
```
s3://noaa-cfs-pds/cfs.{date}/{run}/6hrly_grib_01/
```
- 6-hourly data intervals (vs 3-hourly for GEFS)
- Two main file types:
  - **Flux files**: `flxf{YYYYMMDD}{HH}.01.{init_date}.grb2` - Contains surface fluxes, precipitation, radiation
  - **Pressure files**: `pgbf{YYYYMMDD}{HH}.01.{init_date}.grb2` - Contains atmospheric variables
- Each GRIB2 file has corresponding `.idx` file
- Forecast hours: up to 5154 hours (215 days)

#### 1.2 Target Variables
- **pr**: Precipitation rate (prate.01.*.daily.grb2) - 44.3 MB
- **tasmin**: Minimum daily temperature (tmin.01.*.daily.grb2) - 37.4 MB
- **tasmax**: Maximum daily temperature (tmax.01.*.daily.grb2) - 36.2 MB
- **tas**: Mean daily temperature (tmp2m.01.*.daily.grb2) - 91.7 MB
- **rsds**: Surface downwelling shortwave radiation (dswsfc.01.*.daily.grb2) - 36.1 MB
- **rsus**: Surface upwelling shortwave radiation (uswsfc.01.*.daily.grb2) - 24.4 MB
- **rlds**: Surface downwelling longwave radiation (dlwsfc.01.*.daily.grb2) - 42.2 MB
- **rlus**: Surface upwelling longwave radiation (ulwsfc.01.*.daily.grb2) - 31.7 MB
- **sfcWind**: Surface wind velocity (10m) (wnd10m.01.*.daily.grb2) - 138.2 MB

### 2. Implementation Strategy

#### Phase 1: Core Adaptation
1. **Create CFS-specific utilities** (`cfs_util.py`):
   - Modify `generate_axes()` for 6-hourly intervals
   - Adapt file URL patterns for CFS structure
   - Update variable dictionaries for CFS parameters

2. **Index Preprocessing**:
   - Create `cfs_index_preprocessing.py` based on GEFS version
   - Build mapping parquet files for CFS variables
   - Store in structured GCS directory

3. **Variable Mapping**:
   
   **IMPORTANT: CFS variables are split across two file types:**
   
   **From flux files (flxf*.grb2):**
   ```python
   cfs_flux_dict = {
       "Precipitation rate": "PRATE:surface",
       "Temperature 2m": "TMP:2 m above ground",
       "Downward SW radiation": "DSWRF:surface", 
       "Upward SW radiation": "USWRF:surface",
       "Downward LW radiation": "DLWRF:surface",
       "Upward LW radiation": "ULWRF:surface",
       "10m U wind": "UGRD:10 m above ground",
       "10m V wind": "VGRD:10 m above ground"
   }
   ```
   
   **From pressure files (pgbf*.grb2):**
   ```python
   cfs_pgb_dict = {
       # Need to check for TMIN/TMAX availability
       # These may require post-processing from 6-hourly data
   }
   ```
   
   **Note**: TMIN/TMAX (daily min/max temperature) are not directly available in the 6-hourly files and may need to be computed from the 6-hourly TMP data.

#### Phase 2: Processing Pipeline
1. **Batch Processing Script**:
   - Adapt `run_cfs_preprocessing.py` for CFS ensemble members
   - Handle 6-hourly time steps
   - Process flux files for most variables
   - Compute daily min/max from 6-hourly temperature data

2. **Streaming Implementation**:
   - Create `run_day_cfs_ensemble.py` for full ensemble processing
   - Implement dual-file processing (flux + pressure files)
   - Stream 8 variables directly from flux files
   - Post-process temperature for daily min/max values

#### Phase 3: Optimization
1. **Reference Mapping Strategy**:
   - Build initial reference mappings for common CFS dates
   - Implement mapping reuse pattern
   - Store in GCS with appropriate directory structure

2. **Memory Optimization**:
   - Chunk size tuning for larger CFS files
   - Parallel processing configuration
   - Batch size adjustment based on variable sizes

### 3. Key Differences to Handle

1. **Temporal Resolution**:
   - CFS: 6-hourly vs GEFS: 3-hourly
   - Adjust timestep calculations throughout

2. **File Structure**:
   - CFS splits variables across flux (flxf) and pressure (pgbf) files
   - Must scan both file types for complete variable set
   - Variables found in flux files: PRATE, TMP (2m), DSWRF, USWRF, DLWRF, ULWRF, UGRD/VGRD (10m)

3. **Variable Processing**:
   - TMIN/TMAX not directly available - compute from 6-hourly TMP data
   - Wind components (UGRD/VGRD) need to be combined for sfcWind magnitude
   - All target variables except TMIN/TMAX are in flux files

4. **Forecast Length**:
   - CFS extends to 215+ days (5154 hours)
   - Much longer than GEFS 10-day forecasts
   - Requires different axis generation strategy

### 4. Testing Strategy

1. **Unit Tests**:
   - Test individual CFS functions
   - Validate variable mappings
   - Check time axis generation

2. **Integration Tests**:
   - Process single CFS member
   - Validate zarr store creation
   - Test streaming access

3. **Performance Tests**:
   - Benchmark processing times
   - Monitor memory usage
   - Optimize batch sizes

## Code Reuse Opportunities

### Direct Reuse (85%):
- Zarr store creation logic
- Parquet file handling
- Async processing framework
- Memory management patterns

### Minor Modifications (15%):
- URL pattern generation for dual file types (flux + pressure)
- Time axis calculations for 6-hourly intervals
- Variable dictionaries for CFS parameters
- File naming conventions for flxf/pgbf patterns
- Dual-file scanning in `filter_build_grib_tree`

## Next Steps

1. **Implement `cfs_util.py`** with CFS-specific adaptations:
   - Dual file type URL generation (flux + pressure)
   - 6-hourly time axis generation
   - CFS variable dictionaries for flux files
   - Modified scan functions for flxf file patterns

2. **Create CFS preprocessing pipeline**:
   - Build initial reference mappings from flux files
   - Handle 8 variables directly available in flux files
   - Implement TMIN/TMAX computation from 6-hourly TMP

3. **Test and validate**:
   - Process single CFS date with all variables
   - Verify kerchunk zarr store creation
   - Test streaming access patterns

4. **Scale and optimize**:
   - Implement full temporal range processing
   - Optimize for longer forecast periods (215+ days)
   - Performance tuning for dual-file access patterns

## Implementation Priority

**High Priority**: Focus on the 8 variables available in flux files (flxf*.grb2):
- PRATE, TMP (2m), DSWRF, USWRF, DLWRF, ULWRF, UGRD/VGRD (10m)

**Medium Priority**: Implement TMIN/TMAX computation from 6-hourly temperature data

**Low Priority**: Explore additional variables from pressure files if needed