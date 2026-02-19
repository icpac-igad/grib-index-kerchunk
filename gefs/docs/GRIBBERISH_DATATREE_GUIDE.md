# GEFS Ensemble Processing with Gribberish and Zarr DataTree

This guide documents the new gribberish-based GEFS processing pipeline, which provides ~80x faster GRIB decoding compared to the traditional cfgrib approach.

## Table of Contents

1. [Overview](#overview)
2. [New Files Created](#new-files-created)
3. [Gribberish vs cfgrib Comparison](#gribberish-vs-cfgrib-comparison)
4. [DataTree Structure](#datatree-structure)
5. [Usage Examples](#usage-examples)
6. [Opening Zarr Files](#opening-zarr-files)
7. [Working with the DataTree](#working-with-the-datatree)

---

## Overview

The gribberish-based pipeline provides a significantly faster method for processing GEFS ensemble data by:

- Using **gribberish** (Rust-based GRIB decoder) instead of cfgrib/eccodes
- Reading byte offset/length from parquet reference files
- Fetching raw GRIB bytes directly from S3
- Decoding in-memory without temporary files
- Storing results in a zarr-backed xarray DataTree

---

## New Files Created

### 1. `plot_gefs_gribberish.py`

**Purpose**: Simple plotting script using gribberish for direct GRIB decoding.

**Features**:
- Reads parquet files to get byte references
- Uses gribberish to decode GRIB data directly from S3
- Creates ensemble comparison and probability plots
- No zarr/datatree storage (in-memory only)

**Usage**:
```bash
python plot_gefs_gribberish.py --parquet_dir 20250918_00 --timestep 4 --members 5
```

### 2. `gefs_gribberish_datatree.py`

**Purpose**: Full ensemble processing pipeline with zarr DataTree storage.

**Features**:
- Builds xarray DataTree with all ensemble members
- Stores data in zarr format for efficient access
- Calculates empirical exceedance probabilities
- Creates multiple plot types (mean, probability panels, member comparison)

**Usage**:
```bash
# Process 5 members, timesteps 0-8
python gefs_gribberish_datatree.py --parquet_dir 20250918_00 --members 5 --timesteps 0-8

# Full processing with custom thresholds
python gefs_gribberish_datatree.py --parquet_dir 20250918_00 --timesteps 0-24 --thresholds "1,5,10,25,50"
```

---

## Gribberish vs cfgrib Comparison

### Performance Benchmarks

| Metric | cfgrib | gribberish | Speedup |
|--------|--------|------------|---------|
| Decode time per chunk | ~2000ms | ~25ms | **~80x** |
| Memory usage | Higher (temp files) | Lower (in-memory) | ~2x better |
| Dependencies | eccodes (C library) | Pure Rust | Simpler |

### How Each Method Works

#### cfgrib (Traditional Method)

```python
import tempfile
import xarray as xr

# 1. Fetch GRIB bytes from S3
grib_bytes = fetch_from_s3(url, offset, length)

# 2. Write to temporary file (SLOW - disk I/O)
with tempfile.NamedTemporaryFile(suffix='.grib2') as tmp:
    tmp.write(grib_bytes)
    tmp_path = tmp.name

    # 3. Open with cfgrib/eccodes (SLOW - file parsing)
    ds = xr.open_dataset(tmp_path, engine='cfgrib')
    data = ds[var_name].values

# Total: ~2000ms per chunk
```

**Bottlenecks**:
- Disk I/O for temporary file
- eccodes library initialization
- Full GRIB parsing even for single message

#### gribberish (New Method)

```python
import gribberish

# 1. Fetch GRIB bytes from S3
grib_bytes = fetch_from_s3(url, offset, length)

# 2. Decode directly in memory (FAST - Rust implementation)
flat_array = gribberish.parse_grib_array(grib_bytes, 0)
data = flat_array.reshape((721, 1440))

# Total: ~25ms per chunk
```

**Advantages**:
- No disk I/O
- Direct byte buffer decoding
- Rust implementation (fast, memory-safe)
- Single message extraction

### Code Example: Hybrid Approach

The scripts support falling back to cfgrib if gribberish fails:

```python
def decode_grib_hybrid(grib_bytes, grid_shape=(721, 1440)):
    """Decode with gribberish, fallback to cfgrib on failure."""
    try:
        # Fast path: gribberish
        flat_array = gribberish.parse_grib_array(grib_bytes, 0)
        return flat_array.reshape(grid_shape), 'gribberish'
    except Exception:
        # Fallback: cfgrib
        return decode_with_cfgrib(grib_bytes), 'cfgrib'
```

---

## DataTree Structure

The zarr DataTree organizes ensemble data hierarchically:

```
ensemble_datatree_20250918_00z.zarr/
│
├── /                           # Root node with metadata
│   └── attrs:
│       ├── title: "GEFS Ensemble Forecast Data"
│       ├── institution: "NOAA/NCEP"
│       ├── decoder: "gribberish"
│       ├── model_date: "2025-09-18"
│       ├── run_hour: 0
│       ├── n_members: 30
│       └── region: "East Africa"
│
├── /gep01/                     # Ensemble member 1
│   └── tp                      # Total precipitation
│       ├── dims: (time, latitude, longitude)
│       ├── coords: time, latitude, longitude, forecast_hour
│       └── attrs: long_name, units, standard_name
│
├── /gep02/                     # Ensemble member 2
│   └── tp
│
├── ... (gep03 through gep30)
│
├── /ensemble_stats/            # Pre-computed statistics
│   ├── mean                    # Ensemble mean
│   ├── std                     # Ensemble standard deviation
│   ├── max                     # Ensemble maximum
│   └── min                     # Ensemble minimum
│
└── /probabilities/             # Exceedance probabilities
    ├── prob_gt_5mm             # P(precip > 5mm)
    ├── prob_gt_10mm            # P(precip > 10mm)
    ├── prob_gt_15mm            # P(precip > 15mm)
    ├── prob_gt_20mm            # P(precip > 20mm)
    └── prob_gt_25mm            # P(precip > 25mm)
```

### Dimensions and Coordinates

| Coordinate | Description | Example Values |
|------------|-------------|----------------|
| `time` | Valid forecast times | datetime64 array |
| `latitude` | Latitude (degrees north) | -12 to 23 (East Africa) |
| `longitude` | Longitude (degrees east) | 21 to 53 (East Africa) |
| `forecast_hour` | Hours from model run | 0, 3, 6, ..., 72 |

---

## Usage Examples

### Basic Processing

```bash
# Process 5 members for first 24 timesteps (3 days)
python gefs_gribberish_datatree.py \
    --parquet_dir 20250918_00 \
    --members 5 \
    --timesteps 0-24 \
    --plot_timestep 4

# Process all 30 members for 10-day forecast
python gefs_gribberish_datatree.py \
    --parquet_dir 20250918_00 \
    --timesteps 0-80

# Custom probability thresholds
python gefs_gribberish_datatree.py \
    --parquet_dir 20250918_00 \
    --thresholds "1,5,10,25,50,100"

# Skip zarr saving (plots only)
python gefs_gribberish_datatree.py \
    --parquet_dir 20250918_00 \
    --no_zarr
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet_dir` | `20250918_00` | Directory containing parquet files |
| `--timesteps` | `0-24` | Timesteps to load (e.g., "0-24" or "0,4,8,12") |
| `--members` | all | Maximum number of ensemble members |
| `--plot_timestep` | `4` | Timestep index for plotting (4 = T+12h) |
| `--thresholds` | `5,10,15,20,25` | Precipitation thresholds in mm |
| `--no_zarr` | False | Skip saving to zarr |

---

## Opening Zarr Files

### Using xarray DataTree

```python
import xarray as xr
from xarray import open_datatree

# Open the zarr store
dt = open_datatree("20250918_00/ensemble_datatree_20250918_00z.zarr", engine="zarr")

# View structure
print(dt)

# Access root attributes
print(f"Model date: {dt.attrs['model_date']}")
print(f"Members: {dt.attrs['n_members']}")
```

### Accessing Ensemble Members

```python
# Get data for a specific member
gep01_tp = dt['gep01']['tp']
print(f"Shape: {gep01_tp.shape}")
print(f"Max precip: {float(gep01_tp.max()):.2f} mm")

# Get data for specific timestep
timestep_4 = dt['gep01']['tp'].isel(time=4)
print(f"Forecast hour: {int(timestep_4.forecast_hour)} h")

# Subset to specific region
nairobi_region = dt['gep01']['tp'].sel(
    latitude=slice(0, -3),
    longitude=slice(36, 38)
)
```

### Accessing Ensemble Statistics

```python
# Get pre-computed ensemble mean
ensemble_mean = dt['ensemble_stats']['mean']

# Get standard deviation
ensemble_std = dt['ensemble_stats']['std']

# Plot ensemble spread
import matplotlib.pyplot as plt

timestep = 4
mean_data = ensemble_mean.isel(time=timestep).values
std_data = ensemble_std.isel(time=timestep).values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(mean_data, cmap='Blues')
axes[0].set_title('Ensemble Mean')
axes[1].imshow(std_data, cmap='Reds')
axes[1].set_title('Ensemble Std Dev')
plt.show()
```

### Accessing Exceedance Probabilities

```python
# Get probability of exceeding 10mm
prob_10mm = dt['probabilities']['prob_gt_10mm']

# Find areas with >50% probability
high_risk = prob_10mm.isel(time=4).values > 50
print(f"Grid points with P>50%: {high_risk.sum()}")

# Plot probability map
import cartopy.crs as ccrs

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

prob_data = prob_10mm.isel(time=4)
prob_data.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='YlOrRd',
    vmin=0, vmax=100
)
ax.coastlines()
plt.title('Probability of Precipitation > 10mm')
plt.show()
```

### Stacking All Members for Custom Analysis

```python
import numpy as np

# Get list of member names
members = [name for name in dt.children.keys() if name.startswith('gep')]
print(f"Members: {len(members)}")

# Stack all member data into single array
# Shape: (n_members, n_times, n_lats, n_lons)
all_data = np.stack([dt[m]['tp'].values for m in members], axis=0)
print(f"Stacked shape: {all_data.shape}")

# Calculate custom percentiles
percentile_90 = np.nanpercentile(all_data, 90, axis=0)
percentile_10 = np.nanpercentile(all_data, 10, axis=0)

# Calculate spread (90th - 10th percentile)
spread = percentile_90 - percentile_10
```

### Lazy Loading with Dask

```python
# Open with dask for lazy loading (memory efficient)
dt = open_datatree(
    "20250918_00/ensemble_datatree_20250918_00z.zarr",
    engine="zarr",
    chunks={'time': 1, 'latitude': 100, 'longitude': 100}
)

# Operations are lazy until .compute() is called
mean_precip = dt['gep01']['tp'].mean(dim='time')
result = mean_precip.compute()  # Triggers actual computation
```

---

## Working with the DataTree

### Computing New Probabilities

```python
def compute_exceedance_probability(dt, threshold):
    """Compute exceedance probability for a custom threshold."""
    members = [name for name in dt.children.keys() if name.startswith('gep')]
    n_members = len(members)

    # Stack member data
    all_data = np.stack([dt[m]['tp'].values for m in members], axis=0)

    # Count exceedances
    exceedance_count = np.sum(all_data >= threshold, axis=0)
    probability = (exceedance_count / n_members) * 100

    return probability

# Compute probability for 50mm threshold
prob_50mm = compute_exceedance_probability(dt, 50)
```

### Creating Time Series at a Point

```python
# Nairobi coordinates
nairobi_lat, nairobi_lon = -1.29, 36.82

# Extract time series for all members at Nairobi
time_series = {}
for member in members:
    ts = dt[member]['tp'].sel(
        latitude=nairobi_lat,
        longitude=nairobi_lon,
        method='nearest'
    ).values
    time_series[member] = ts

# Plot spaghetti diagram
plt.figure(figsize=(12, 6))
for member, ts in time_series.items():
    plt.plot(ts, alpha=0.3, color='blue')

# Add ensemble mean
mean_ts = np.mean(list(time_series.values()), axis=0)
plt.plot(mean_ts, color='red', linewidth=2, label='Ensemble Mean')

plt.xlabel('Forecast Hour')
plt.ylabel('Precipitation (mm)')
plt.title('GEFS Ensemble Forecast for Nairobi')
plt.legend()
plt.show()
```

### Exporting to NetCDF

```python
# Export single member to NetCDF
dt['gep01'].to_netcdf('gep01_precipitation.nc')

# Export ensemble statistics
dt['ensemble_stats'].to_netcdf('ensemble_stats.nc')

# Export probabilities
dt['probabilities'].to_netcdf('probabilities.nc')
```

---

## Dependencies

```bash
# Required
pip install gribberish numpy pandas xarray zarr fsspec s3fs matplotlib cartopy

# Optional (for fallback)
pip install cfgrib eccodes
```

### Version Requirements

- Python >= 3.9
- zarr >= 3.0.0 (tested with 3.1.5)
- xarray >= 2024.1.0
- gribberish >= 0.1.0

---

## Troubleshooting

### Common Issues

1. **gribberish not found**
   ```bash
   pip install gribberish
   ```

2. **Zarr version compatibility**
   ```bash
   pip install zarr>=3.0.0
   ```

3. **S3 access errors**
   ```python
   # Ensure anonymous access is configured
   import os
   os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
   ```

4. **Memory issues with large ensembles**
   - Use `--timesteps` to limit temporal range
   - Use `--members` to limit ensemble size
   - Process in batches

---

## Output Files

After running `gefs_gribberish_datatree.py`, the following files are created in the parquet directory:

| File | Description |
|------|-------------|
| `ensemble_datatree_YYYYMMDD_HHz.zarr/` | Zarr DataTree store |
| `ensemble_mean_TXXX.png` | Ensemble mean precipitation plot |
| `probability_panels_TXXX.png` | Multi-panel probability plot |
| `member_comparison_TXXX.png` | All members side-by-side |

---

## References

- [gribberish GitHub](https://github.com/mpiannucci/gribberish)
- [xarray DataTree Documentation](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- [GEFS Data on AWS](https://registry.opendata.aws/noaa-gefs/)
