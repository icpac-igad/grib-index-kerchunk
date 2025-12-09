# Session Changes Documentation - December 8, 2025

## Overview

This session focused on:
1. Diagnosing memory issues with batch GEFS processing
2. Creating scripts to convert GEFS zarr files to cGAN-ready NetCDF format
3. Fixing precipitation (`tp`) variable discovery in the processing pipeline

---

## Files Created

### 1. `zarr_to_cgan.py` (Version 1)
**Purpose**: Initial converter from zarr to cGAN NetCDF format

**Features**:
- Reads individual member zarr files
- Combines 30 ensemble members along a new 'member' dimension
- Filters to specific forecast hours (30-54, every 6h)
- Renames variables per cGAN mapping (t2m→tmp, sp→pres, etc.)
- Outputs one NetCDF per variable

**Output Format**:
```
Shape: (time=1, member=30, valid_time=5, latitude=141, longitude=129)
```

**Issue Found**: This version kept all 30 members separate, but cGAN actually needs **ensemble statistics** (mean and std), not individual members.

---

### 2. `zarr_to_cgan_v2.py` (Version 2 - Recommended)
**Purpose**: Correct converter that computes ensemble statistics as required by cGAN

**Key Transformations**:
1. **Ensemble Statistics**: Computes mean and std across 30 members (members are collapsed)
2. **4-Channel Structure** per field:
   - `mean_t1`: Ensemble mean at timestep 1
   - `std_t1`: Ensemble std at timestep 1
   - `mean_t2`: Ensemble mean at timestep 2
   - `std_t2`: Ensemble std at timestep 2
3. **Hour Pairs**: Consecutive 6-hour steps (30-36, 36-42, 42-48, 48-54) → 4 valid_time indices
4. **Field Order**: cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp (8 fields total)
5. **Normalization** (optional): Can apply field-specific normalization with `--norm_file`

**Output Files**:
| File | Dimensions | Description |
|------|------------|-------------|
| `{field}_2025.nc` | (time=1, valid_time=4, channel=4, lat=141, lon=129) | Per-field file with 4 channels |
| `forecast_input_2025.nc` | (time=1, valid_time=4, lat=141, lon=129, channel=28) | Combined file with 7×4=28 channels |

**Variable Mapping**:
```python
VARIABLE_MAPPING = {
    'cape': 'cape',     # Convective Available Potential Energy
    'sp': 'pres',       # Surface Pressure -> Pressure
    'mslet': 'msl',     # Mean Sea Level Pressure
    'pwat': 'pwat',     # Precipitable Water
    't2m': 'tmp',       # 2m Temperature -> Temperature
    'u10': 'ugrd',      # 10m U-wind component
    'v10': 'vgrd',      # 10m V-wind component
    'tp': 'apcp',       # Total Precipitation -> Accumulated Precipitation
}
```

**Usage**:
```bash
# Basic usage
python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00

# With custom forecast hours
python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \
    --forecast_hours 30,36,42,48,54

# With normalization stats file
python zarr_to_cgan_v2.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \
    --norm_file /path/to/FCSTNorm2018.pkl
```

---

## Files Modified

### 1. `run_single_gefs_to_zarr_gribberish.py`

**Changes Made**:

#### a) Fixed `tp` (precipitation) variable discovery

**Problem**: The `tp` variable has a different path structure in parquet files:
- Other variables: `var/type/level/0.0.0` (e.g., `t2m/instant/heightAboveGround/0.0.0`)
- Precipitation: `tp/accum/surface/tp/X.0.0` (starts from index 1, not 0)

**Solution**: Added special handling in `discover_variables()` function:
```python
# Special handling for tp (precipitation) which has different structure:
# tp/accum/surface/tp/X.0.0 (starts from 1, not 0)
if 'tp' not in variables:
    tp_prefix = 'tp/accum/surface/tp'
    if f'{tp_prefix}/1.0.0' in zstore:
        variables['tp'] = {
            'chunks': [],
            'path_prefix': tp_prefix
        }
```

#### b) Fixed step alignment for variables with different timesteps

**Problem**: `tp` has 80 chunks (steps 1-80) while other variables have 81 chunks (steps 0-80). This caused xarray alignment errors.

**Solution**: Modified `create_zarr_dataset()` to:
1. Accept a dict of steps per variable instead of a single list
2. Find common steps across all variables
3. Align data to common steps, padding with NaN where needed

```python
def create_zarr_dataset(variables_data, steps_dict, lats, lons, member_name, region):
    # Handle backward compatibility
    if isinstance(steps_dict, list):
        steps_dict = {var_name: steps_dict for var_name in variables_data.keys()}

    # Find common steps across all variables
    all_step_sets = [set(steps_dict[var]) for var in variables_data.keys()]
    common_steps = sorted(set.intersection(*all_step_sets))

    # Align each variable to common steps
    for var_name, data_3d in variables_data.items():
        if set(var_steps) != set(common_steps):
            # Align with NaN padding for missing steps
            aligned_data = np.full((len(common_steps), ...), np.nan)
            ...
```

---

## Memory Issues Diagnosed

### Problem
Running batch processing with `parallel -j 2` was getting killed (OOM):
```bash
for i in $(seq -f "%02g" 1 30); do
    python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep$i ...
done
```

### Root Causes Found
1. **Only 8GB RAM** with **no swap space**
2. **Multiple stopped/zombie Python processes** holding ~1.5GB of memory
3. Each processing job uses ~1-2GB peak memory
4. Running 2 jobs in parallel = 2-4GB, leaving no headroom

### Solutions Provided
1. **Kill zombie processes**:
   ```bash
   kill -9 <stopped_process_pids>
   ```

2. **Run sequentially** (recommended for 8GB RAM):
   ```bash
   for i in $(seq -f "%02g" 1 30); do
       python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep$i \
           --region east_africa --variables t2m,tp,u10,v10,cape,sp,mslet,pwat
   done
   ```

3. **Use `--ungroup` with parallel** for real-time output:
   ```bash
   seq -f "%02g" 1 30 | parallel -j 1 --ungroup \
       'python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep{} ...'
   ```

---

## cGAN Input Format Requirements

Based on analysis of `FORECAST_WORKFLOW.md` and cGAN tutorial files:

### Expected Input Structure
- **8 fields** in order: cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp
- **4 channels per field**: [mean_t1, std_t1, mean_t2, std_t2]
- **Total**: 32 channels
- **Dimensions**: (time, valid_time, latitude, longitude, channel)

### Ensemble Processing
- 30 GEFS members are **collapsed** into ensemble mean and std
- Individual members are NOT passed to the model
- Statistics capture ensemble uncertainty

### Normalization (per field type)
| Field Type | Method | Example Fields |
|------------|--------|----------------|
| Log transform | `log10(1 + data)` | apcp (precipitation) |
| Standardize | `(data - mean) / std` | msl, pres, tmp |
| Max scale | `data / max` | cape, pwat |
| Symmetric | `data / max(\|min\|, max)` | ugrd, vgrd |

---

## Output Directory Structure

After running the updated workflow:

```
zarr_stores/20250918_00/
├── gep01_gribberish_east_africa.zarr  # Now includes tp
├── gep02_gribberish_east_africa.zarr
├── ...
└── gep30_gribberish_east_africa.zarr

cgan_input/20250918_00/
├── cape_2025.nc           # Per-field files (4 channels each)
├── pres_2025.nc
├── pwat_2025.nc
├── tmp_2025.nc
├── ugrd_2025.nc
├── vgrd_2025.nc
├── msl_2025.nc
├── apcp_2025.nc           # Precipitation (if tp available)
└── forecast_input_2025.nc # Combined file (28-32 channels)
```

---

## Next Steps Required

1. **Re-process all 30 members** with the updated `run_single_gefs_to_zarr_gribberish.py` to include `tp`:
   ```bash
   for i in $(seq -f "%02g" 1 30); do
       python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep$i \
           --region east_africa \
           --variables t2m,tp,u10,v10,cape,sp,mslet,pwat
   done
   ```

2. **Run zarr_to_cgan_v2.py** to create cGAN input files:
   ```bash
   python zarr_to_cgan_v2.py \
       --input_dir zarr_stores/20250918_00 \
       --output_dir cgan_input/20250918_00 \
       --forecast_hours 30,36,42,48,54
   ```

3. **Verify output** has all 8 fields including `apcp` (precipitation)

---

## Files Summary

| File | Status | Description |
|------|--------|-------------|
| `zarr_to_cgan.py` | Created | V1 converter (keeps members separate) |
| `zarr_to_cgan_v2.py` | Created | V2 converter (computes ensemble stats) - **Recommended** |
| `run_single_gefs_to_zarr_gribberish.py` | Modified | Fixed tp discovery and step alignment |
| `SESSION_CHANGES_20251208.md` | Created | This documentation file |
