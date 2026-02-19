# cGAN Forecast Workflow: Zarr to NetCDF Conversion

This document describes the complete workflow for running GEFS-based cGAN forecasts, including the crucial conversion from Zarr to NetCDF format.

## Overview of Changes

The forecast system has been configured to work with NetCDF files converted from Zarr format. This workflow involves:

1. **Pre-processing**: Converting Zarr files to NetCDF using `zarrto_nc.py`
2. **Configuration**: Updating paths to point to correct data locations
3. **Forecasting**: Running the cGAN model to generate downscaled precipitation forecasts

---

## Configuration Changes Made

### 1. `config/data_paths.yaml`

**Purpose**: Point to correct normalization constants location

```yaml
ICPAC_CLOUD:
    GENERAL:
        CONSTANTS_PATH_GEFS: '/home/nkalladath_icpac_net/data/CONSTANTS/'
        # Changed from: '/home/nshruti_icpac_net/CONSTANTS/'
```

**Why**: The `FCSTNorm2018.pkl` file contains critical normalization statistics for forecast fields:
- `min`, `max`: For fields like wind components (ugrd, vgrd)
- `mean`, `std`: For fields like pressure (msl, pres) and temperature (tmp)
- These are used to normalize input data to O(1) scale before feeding to the GAN

**Location**: `/home/nkalladath_icpac_net/data/CONSTANTS/FCSTNorm2018.pkl`

### 2. `config/forecast_gfs.yaml`

**Purpose**: Configure model, input data, and output locations

```yaml
MODEL:
    folder: "/home/nkalladath_icpac_net/data/logfile_gefs_v3/"
    # Changed from: "/home/nshruti_icpac_net/logfile_gefs_v3"
    checkpoint: 345600

INPUT:
    folder: "/home/nkalladath_icpac_net/data/netcdf/"
    # Changed from: "/home/nshruti_icpac_net/GEFS/"
    dates: ["2024-04-20"]
    start_hour: 30
    end_hour: 54

OUTPUT:
    folder: "/home/nkalladath_icpac_net/data/cGAN_gefs/predictions/"
    # Changed from: "/home/nshruti_icpac_net/cGAN_gefs/predictions/"
    ensemble_members: 50
```

**Key Change**: `INPUT.folder` now points to `/data/netcdf/` which contains NetCDF files converted from Zarr.

### 3. `scripts/forecast_gfs.py`

**Purpose**: Modified to read NetCDF instead of Zarr files

```python
# Lines 210-215 (approximately)
# Changed from:
# input_file = f"{field}_{d.year}.zarr"
# nc_file = xr.open_zarr(nc_in_path)

# Changed to:
input_file = f"{field}_{d.year}.nc"
nc_in_path = os.path.join(input_folder_year, input_file)
nc_file = xr.open_dataset(nc_in_path)  # Use open_dataset for NetCDF
```

**Additional change at line 56**: Added error exit if YAML parsing fails
```python
sys.exit(1)  # Exit on YAML error
```

### 4. `data/tfrecords_generator.py`

**Purpose**: Prevent automatic directory creation

```python
# Lines 28-30 (approximately)
# Commented out:
#if not os.path.exists(records_folder):
#    os.makedirs(records_folder)
```

**Why**: Gives user control over tfrecord generation; prevents unexpected directory creation.

---

## Complete Workflow

### Step 1: Convert Zarr to NetCDF

The `zarrto_nc.py` script is **crucial** for this workflow. It converts Zarr files to NetCDF format.

**Script location**: `/home/nkalladath_icpac_net/cGAN_tutorial/example_notebooks/zarrto_nc.py`

**Usage Option 1 - Using config file** (Recommended):
```bash
cd /home/nkalladath_icpac_net/cGAN_tutorial/example_notebooks
python zarrto_nc.py --config ../config/forecast_gfs.yaml
```

This will:
- Read input folder from `forecast_gfs.yaml`
- Read dates from `forecast_gfs.yaml`
- Automatically create output in `netcdf/` subdirectory
- Convert all fields defined in `data.data_gefs.all_fcst_fields`:
  - `cape`: Convective Available Potential Energy
  - `pres`: Pressure
  - `pwat`: Precipitable Water
  - `tmp`: Temperature
  - `ugrd`, `vgrd`: Wind components (U and V)
  - `msl`: Mean Sea Level pressure
  - `apcp`: Accumulated Precipitation

**Usage Option 2 - Command line arguments**:
```bash
python zarrto_nc.py \
    --input_folder /home/nshruti_icpac_net/zarr/ \
    --output_folder /home/nkalladath_icpac_net/data/netcdf/ \
    --dates 2024-04-20 \
    --year 2024
```

**Usage Option 3 - Convert specific fields only**:
```bash
python zarrto_nc.py --config ../config/forecast_gfs.yaml \
    --fields apcp tmp pres
```

**What the script does**:
1. Opens Zarr files from: `{input_folder}/{year}/{field}_{year}.zarr`
2. Filters by specified dates
3. Converts to NetCDF format
4. Saves to: `{output_folder}/{year}/{field}_{year}.nc`
5. Prints instructions for updating config

**Example output structure**:
```
/home/nkalladath_icpac_net/data/netcdf/
└── 2024/
    ├── cape_2024.nc
    ├── pres_2024.nc
    ├── pwat_2024.nc
    ├── tmp_2024.nc
    ├── ugrd_2024.nc
    ├── vgrd_2024.nc
    ├── msl_2024.nc
    └── apcp_2024.nc
```

### Step 2: Update Configuration

After conversion, ensure `config/forecast_gfs.yaml` points to the NetCDF folder:

```yaml
INPUT:
    folder: "/home/nkalladath_icpac_net/data/netcdf/"
```

### Step 3: Run Forecast

```bash
cd /home/nkalladath_icpac_net/cGAN_tutorial/example_notebooks

# Set TensorFlow to use legacy Keras (required for TF 2.16+)
export TF_USE_LEGACY_KERAS=1

# Run the forecast
python -c "
import sys
sys.path.insert(1,'../scripts/')
import forecast_gfs
forecast_gfs.make_fcst()
"
```

Or import and run interactively:
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
sys.path.insert(1,"../scripts/")
import forecast_gfs

# Forecast runs automatically on import
# Or explicitly call:
forecast_gfs.make_fcst()
```

### Step 4: View Output

Forecasts are saved as NetCDF files:
```
/home/nkalladath_icpac_net/data/cGAN_gefs/predictions/test/2024/
└── GAN_20240420.nc
```

**Output file structure**:
- Dimensions: `(time, member, valid_time, latitude, longitude)`
- Variable: `precipitation` (mm/h)
- 50 ensemble members (configurable)
- Forecast hours: 30-54 (every 6 hours)

---

## Technical Details

### Why Convert Zarr to NetCDF?

**Zarr advantages**:
- Chunked, compressed storage (efficient for large datasets)
- Cloud-optimized (parallel reads)
- Native to modern Python data stack

**NetCDF advantages**:
- Widely supported format
- Better compatibility with legacy tools
- Easier to inspect with standard tools (ncdump, ncview)

**In this workflow**: The original GEFS data is stored in Zarr format, but the forecast script was adapted to work with NetCDF for compatibility.

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Original Zarr Data                                         │
│  /home/nshruti_icpac_net/zarr/{year}/{field}_{year}.zarr   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ zarrto_nc.py (conversion)
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  Converted NetCDF Data                                      │
│  /home/nkalladath_icpac_net/data/netcdf/{year}/             │
│  {field}_{year}.nc                                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ forecast_gfs.py (reads with xr.open_dataset)
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  Normalization                                              │
│  - Load FCSTNorm2018.pkl                                    │
│  - Apply field-specific normalization                       │
│  - Compute ensemble mean and std                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  cGAN Generator                                             │
│  - Input: Normalized forecast fields (4 channels per field)│
│  - Constants: Orography + Land-Sea Mask                     │
│  - Noise: Random noise for ensemble generation              │
│  - Output: High-resolution precipitation (0.1° grid)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  Denormalization & Output                                   │
│  - Apply inverse log transform: 10^x - 1                    │
│  - Cap at 100 mm/h                                          │
│  - Save as NetCDF                                           │
│  /home/nkalladath_icpac_net/data/cGAN_gefs/predictions/    │
└─────────────────────────────────────────────────────────────┘
```

### Field Normalization Methods

Different meteorological fields use different normalization strategies:

1. **Precipitation (`apcp`)**:
   - Transform: `log10(1 + x)` (no normalization constants needed)
   - Reason: Log transform handles wide range of values

2. **Pressure/Temperature (`msl`, `pres`, `tmp`)**:
   - Transform: `(x - mean) / std`
   - Reason: Values bounded away from zero, use standard scaling

3. **Non-negative fields (`cape`, `pwat`)**:
   - Transform: `x / max`
   - Reason: Simple scaling to [0, 1]

4. **Wind components (`ugrd`, `vgrd`)**:
   - Transform: `x / max(|min|, max)`
   - Reason: Preserve sign, symmetric scaling

### Input Data Structure

Each field NetCDF file contains:
- **time dimension**: Forecast initialization dates
- **step dimension**: Forecast lead times (e.g., 6h, 12h, 18h...)
- **latitude/longitude**: Spatial coordinates
- **ensemble members**: GEFS has 30-31 ensemble members

The forecast script:
- Selects specific date and two consecutive time steps
- Computes ensemble mean and standard deviation
- Stacks these as 4 channels per field
- Total input: 4 × 8 fields = 32 channels

---

## Troubleshooting

### Error: `AssertionError` at `forecast_gfs.py:46`

**Cause**: `fcst_norm is None` (normalization file not loaded)

**Solution**: Check `CONSTANTS_PATH_GEFS` in `config/data_paths.yaml` points to directory containing `FCSTNorm2018.pkl`

### Error: File not found when opening NetCDF

**Cause**: NetCDF files don't exist or wrong path

**Solution**:
1. Run `zarrto_nc.py` to create NetCDF files
2. Verify `INPUT.folder` in `forecast_gfs.yaml` points to correct location
3. Check files exist: `ls /home/nkalladath_icpac_net/data/netcdf/2024/`

### Error: `KeyError` when selecting time/step

**Cause**: Date or time step not in NetCDF file

**Solution**:
- Check available dates: `ncdump -v time file.nc`
- Ensure `zarrto_nc.py` was run with correct dates
- Verify `start_hour` and `end_hour` in config are available in data

---

## Summary of All Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `config/data_paths.yaml` | Path update | Point to correct CONSTANTS directory for FCSTNorm2018.pkl |
| `config/forecast_gfs.yaml` | Path updates | Update MODEL, INPUT, OUTPUT folders to user directories |
| `scripts/forecast_gfs.py` | Code modification | Switch from `xr.open_zarr()` to `xr.open_dataset()` |
| `scripts/forecast_gfs.py` | Error handling | Add `sys.exit(1)` on YAML parse error |
| `data/tfrecords_generator.py` | Code comment | Disable automatic directory creation |
| `example_notebooks/zarrto_nc.py` | New file | **Critical conversion script** from Zarr to NetCDF |

---

## Git Remotes Configuration

The repository has been configured with two remotes:

- **origin**: `git@github.com:nishadhka/cGAN_tutorial.git` (your fork)
- **upstream**: `https://github.com/snath-xoc/cGAN_tutorial.git` (original repo)

This allows you to:
- Push changes to your fork: `git push origin main`
- Pull updates from upstream: `git pull upstream main`
