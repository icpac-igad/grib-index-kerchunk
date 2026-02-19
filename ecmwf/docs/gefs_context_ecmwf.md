# GEFS Context Applied to ECMWF Processing

## Overview
This document describes how the GEFS (Global Ensemble Forecast System) processing methodology can be adapted for ECMWF ensemble data processing. The key difference is that ECMWF stores all ensemble members in a single GRIB file per timestep, while GEFS has separate files for each member.

## Key Differences Between GEFS and ECMWF

### 1. File Organization
- **GEFS**: One file per ensemble member per timestep
  - Example: `gep01.t00z.pgrb2s.0p25.f000`, `gep02.t00z.pgrb2s.0p25.f000`
- **ECMWF**: All ensemble members in one file per timestep
  - Example: `20250628180000-0h-enfo-ef.grib2` contains all 51 members (control + 50 perturbed)

### 2. Ensemble Member Identification
- **GEFS**: Member identified by filename (gep01-gep30)
- **ECMWF**: Member identified by metadata within GRIB messages (`ens_number`: -1 for control, 1-50 for perturbed)

### 3. Data Structure
- **GEFS**: Requires aggregating multiple files to build ensemble dimension
- **ECMWF**: Requires extracting and organizing ensemble members from single file

## GEFS Processing Pipeline Components

### 1. Scan and Build GRIB Tree (`gefs_util.py`)
```python
# GEFS approach - scan multiple files
gefs_grib_tree_store = grib_tree([group for f in gefs_files for group in scan_grib(f)])

# Key functions:
- build_gefs_grib_tree()
- filter_build_grib_tree()  # with variable filtering
- strip_datavar_chunks()     # deflate the store
```

### 2. Time Dimension Handling
```python
# GEFS uses 3-hour intervals, 10-day forecast
- generate_axes()           # Create time axes
- calculate_time_dimensions()  # Build time coords
```

### 3. Mapped Index Creation
```python
# GEFS uses pre-built mappings stored in GCS
- create_gefs_mapped_index()    # Uses local parquet mappings
- cs_create_mapped_index()      # Uses GCS stored mappings
```

### 4. Zarr Store Processing
```python
# Process unique variable groups
- prepare_zarr_store()
- process_unique_groups()
- create_parquet_file()  # Save as .par file
```

## ECMWF Adaptation Strategy

### 1. Ensemble Member Extraction (New Component)
ECMWF requires extracting ensemble member information from GRIB metadata:
```python
# From test_run_ecmwf_step1_scangrib.py
- ecmwf_filter_scan_grib()  # Scan and add ensemble info
- ecmwf_idx_df_create_with_keys()  # Map indices to ensemble members
```

### 2. Modified GRIB Tree Building
Instead of aggregating multiple files, organize ensemble dimension within tree:
```python
# ECMWF approach - organize ensemble dimension
- fixed_ensemble_grib_tree()  # Build tree with ensemble support
- organize_ensemble_tree()    # Add ensemble dimensions to attributes
```

### 3. Parquet File Creation Per Member
Following GEFS pattern, create individual parquet files for each ensemble member:
```python
# Process each ensemble member separately
for member in range(-1, 51):  # -1 for control, 1-50 for perturbed
    member_store = extract_member_from_tree(ensemble_tree, member)
    create_parquet_file(member_store, f"ecmwf_ens{member:02d}.par")
```

## Proposed ECMWF Processing Flow

### Step 1: Initial Scan and Index Creation
1. Scan ECMWF GRIB file with index parsing
2. Create ensemble member mapping from index
3. Build initial GRIB tree with ensemble metadata

### Step 2: Ensemble Member Separation
1. Extract groups for each ensemble member
2. Create member-specific zarr stores
3. Apply time dimension processing per member

### Step 3: Parquet File Generation
1. For each ensemble member:
   - Create deflated zarr store
   - Process unique variable groups
   - Generate parquet file with proper structure
2. Store parquet files in organized directory structure

### Step 4: Metadata and Validation
1. Generate metadata file with ensemble configuration
2. Validate parquet files can be opened as datatree
3. Ensure variable accessibility for downstream processing

## Implementation Notes

### Memory Management
- Process ensemble members in batches (similar to GEFS)
- Use async processing for parallel member extraction
- Implement chunking for large files

### File Naming Convention
```
ecmwf_{date}_{run}/
  ├── control.par       # Control member (ens_number = -1)
  ├── ens01.par        # Perturbed member 1
  ├── ens02.par        # Perturbed member 2
  ...
  └── ens50.par        # Perturbed member 50
```

### Variable Filtering
Apply same variable selection as GEFS:
- Surface variables: pressure, temperature, wind, precipitation
- Level variables: geopotential height, humidity
- Special handling for accumulated variables

### Time Handling
- ECMWF uses different forecast intervals (varies by product)
- Adapt time axis generation for ECMWF forecast structure
- Maintain compatibility with FMRC (Forecast Model Run Collection) approach

## Benefits of GEFS-Style Processing for ECMWF

1. **Consistency**: Same processing pipeline for both GEFS and ECMWF
2. **Modularity**: Each ensemble member as separate parquet file
3. **Scalability**: Process members in parallel
4. **Compatibility**: Works with existing downstream tools
5. **Flexibility**: Easy to select specific members for analysis

## Next Steps

1. Implement `ecmwf_ensemble_par_creator.py` following GEFS pattern
2. Test with sample ECMWF data for validation
3. Optimize for ECMWF-specific characteristics
4. Create utility functions for ensemble member selection
5. Document API for downstream usage