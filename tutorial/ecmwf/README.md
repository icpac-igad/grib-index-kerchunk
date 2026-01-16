# ECMWF GIK Tutorial - Deep Dive

## Quick Start

```bash
cd tutorial/ecmwf
python run_ecmwf_data_streaming_20260106.py
```

**Runtime**: ~29 minutes for 51 ensemble members

**Output**: `output_plots/ecmwf_24h_probability_20260106_00z_all_thresholds.png`

## Performance Benchmarks

### Data Streaming Performance

| Metric | Value |
|--------|-------|
| **Total Time** | 28.8 minutes |
| **Members** | 51 (control + ens01-ens50) |
| **Timesteps/Member** | 85 |
| **Time/Member** | ~34 seconds |
| **Parallel Connections** | 8 |

### Per-Member Breakdown

| Operation | Time | Notes |
|-----------|------|-------|
| S3 Fetch (parallel) | ~29s | 8 concurrent connections |
| Gribberish Decode | ~4s | 85 timesteps @ 30ms each |
| **Total** | ~34s | Per ensemble member |

### Speed Comparison

| Configuration | Time/Member | Total (51 members) |
|--------------|-------------|-------------------|
| Sequential fetch + cfgrib | ~10 min | ~8.5 hours |
| Sequential fetch + gribberish | ~3 min | ~2.5 hours |
| **Parallel fetch + gribberish** | **~34s** | **~29 min** |

## ECMWF Data Characteristics

| Property | Value |
|----------|-------|
| **Source** | `s3://ecmwf-forecasts/` |
| **Ensemble Members** | 51 (control + 50 perturbed) |
| **Grid Resolution** | 0.25 degree (~25 km) |
| **Grid Size** | 721 x 1440 (lat x lon) |
| **Forecast Hours** | 85 total |
| **Time Resolution** | 3h (0-144h), 6h (150-360h) |
| **Longitude Range** | -180 to 179.75 |
| **Index Format** | JSON (`.index` files) |

### Forecast Hour Schedule

```
Hours 0-144:   3-hourly intervals (49 timesteps)
  0, 3, 6, 9, 12, 15, ..., 141, 144

Hours 150-360: 6-hourly intervals (36 timesteps)
  150, 156, 162, 168, ..., 354, 360

Total: 85 forecast timesteps per member
```

## Available Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_ecmwf_tutorial.py` | Create parquet reference files | ~5-30 min |
| `run_ecmwf_data_streaming_v2.py` | Stream data (v2, 4 members demo) | ~4 min |
| `run_ecmwf_data_streaming_20260106.py` | Stream all 51 members | ~29 min |

## Data Streaming Script Details

### `run_ecmwf_data_streaming_20260106.py`

This is the main production script for streaming ECMWF ensemble data.

#### Key Features

1. **Parallel S3 Fetching**: 8 concurrent connections per member
2. **Gribberish Decoding**: Rust-based GRIB decoder (~30ms/chunk)
3. **Disk-based Storage**: Temporary zarr store prevents memory issues
4. **Automatic Cleanup**: Removes temp files after processing

#### Configuration

```python
# Directory containing parquet files
PARQUET_DIR = Path("ecmwf_three_stage_20260106_00z")

# East Africa coverage
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# Rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# Parallel fetching
MAX_PARALLEL_FETCHES = 8
```

#### Output

```
output_plots/ecmwf_24h_probability_20260106_00z_all_thresholds.png
```

Multi-panel plot showing:
- 10 rows: Forecast days (24h periods ending at T+24h, T+48h, ..., T+240h)
- 6 columns: Thresholds (>5mm, >25mm, >50mm, >75mm, >100mm, >125mm)
- Colors: Exceedance probability 0-100%
- Boundaries: East Africa countries from `ea_ghcf_simple.geojson`

## How It Works

### 1. Load Parquet References

```python
# Read parquet file containing zarr references
df = pd.read_parquet('ecmwf_three_stage_20260106_00z/stage3_control_final.parquet')

# Convert to zstore dictionary
zstore = {}
for _, row in df.iterrows():
    key = row['key']
    value = row['value']
    # Parse JSON values
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    zstore[key] = value
```

### 2. Discover Precipitation Chunks

```python
# Find all precipitation data references
# Pattern: step_{hours}/tp/sfc/{member}/0.0.0
chunks = discover_precipitation_chunks_ecmwf(zstore, 'control')
# Returns: [(0, 'step_0/tp/sfc/control/0.0.0', [url, offset, length]), ...]
```

### 3. Parallel S3 Fetching

```python
from concurrent.futures import ThreadPoolExecutor

# Fetch all 85 timesteps in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch_grib_bytes, ref, fs): idx
               for idx, ref in enumerate(chunk_refs)}

    for future in as_completed(futures):
        idx = futures[future]
        grib_bytes = future.result()
        grib_data[idx] = grib_bytes
```

### 4. Gribberish Decoding

```python
import gribberish

# Decode GRIB bytes to numpy array (~30ms)
flat_array = gribberish.parse_grib_array(grib_bytes, 0)
array_2d = flat_array.reshape((721, 1440))

# Extract East Africa region
ea_data = array_2d[lat_slice, lon_slice]
```

### 5. Calculate 24h Accumulation

ECMWF precipitation is cumulative from model initialization:

```python
# Convert cumulative to 24h accumulation
# ECMWF tp is in meters, convert to mm
for end_hour in timesteps:
    start_hour = end_hour - 24
    if start_hour in timesteps:
        precip_24h = (data[end_hour] - data[start_hour]) * 1000.0  # m to mm
```

### 6. Exceedance Probability

```python
# For each grid point, count members exceeding threshold
for member in range(51):
    for threshold in [5, 25, 50, 75, 100, 125]:
        exceedance_count[threshold] += (precip_24h >= threshold)

# Convert to probability
probability = exceedance_count / n_members * 100
```

## Parquet File Structure

### Input Files

```
ecmwf_three_stage_20260106_00z/
├── stage3_control_final.parquet
├── stage3_ens_01_final.parquet
├── stage3_ens_02_final.parquet
├── ...
└── stage3_ens_50_final.parquet
```

### Parquet Contents

Each parquet file contains zarr references:

| Key | Value |
|-----|-------|
| `.zattrs` | JSON metadata |
| `.zgroup` | Group metadata |
| `step_0/tp/sfc/control/0.0.0` | `[url, offset, length]` |
| `step_3/tp/sfc/control/0.0.0` | `[url, offset, length]` |
| ... | ... |

The `[url, offset, length]` tuple points to the exact bytes on S3.

## Memory Management

### Disk-based Zarr Storage

```python
# Create temporary zarr store on disk
store = zarr.open_group('temp_zarr_cache/ecmwf_20260106_00z_*', mode='w')

# Shape: (51 members, 85 timesteps, 141 lats, 129 lons)
store.create_dataset(
    'precipitation',
    shape=(51, 85, 141, 129),
    chunks=(1, 85, 141, 129),  # One member per chunk
    dtype=np.float32,
    compressors=[BloscCodec(cname='lz4', clevel=3)]
)
```

### Automatic Cleanup

```python
# Remove temp storage after processing
cleanup_temp_zarr_store(temp_zarr_path)
```

## Troubleshooting

### Slow S3 Fetching

ECMWF GRIB files are large (~3GB), and data offsets can be at the end of the file.

**Solution**: Use parallel fetching (8 connections default)

```python
MAX_PARALLEL_FETCHES = 8  # Adjust based on bandwidth
```

### Member Name Mismatch

Parquet keys use `ens01` format (with leading zero), not `ens1`.

```python
# Correct format
member_name = f'ens{member_num:02d}'  # ens01, ens02, ..., ens50
```

### Memory Issues

If running out of memory:

1. Reduce `MAX_PARALLEL_FETCHES` (trades speed for memory)
2. Process fewer members at a time
3. Ensure disk-based zarr storage is enabled

### Missing Dependencies

```bash
# Core
pip install pandas numpy zarr fsspec s3fs

# Fast decoding (highly recommended)
pip install gribberish

# Plotting
pip install matplotlib cartopy geopandas
```

## Example Output

### Console Output

```
======================================================================
ECMWF Data Streaming - 20260106 Full Ensemble
======================================================================
Gribberish Available: True
Plotting Available: True
Parquet Directory: ecmwf_three_stage_20260106_00z
East Africa Region: 141 x 129 grid points
======================================================================

Found 51 parquet files
Member names: ['control', 'ens01', 'ens02', ...]

[Step 1] Extracting model information...
  Model date: 2026-01-06 00Z
  Timesteps per member: 85

[Step 2] Creating temporary disk storage...
  Storage shape: (51, 85, 141, 129)

[Step 3] Streaming Precipitation Data
======================================================================
  [1] Streaming control...
    Found 85 timesteps
    Fetching with 8 parallel connections...
    Fetched 85/85 in 27.6s (0.32s avg)
    Completed: gribberish=85, cfgrib=0, failed=0, time=31.8s

  [2] Streaming ens01...
    ...

Successfully loaded 51 ensemble members

[Step 4] Calculating Exceedance Probabilities
======================================================================
  24h Periods: 77
  Thresholds: [5, 25, 50, 75, 100, 125] mm
  Valid Members: 51

[Step 5] Creating Visualization
======================================================================
  Plot saved: output_plots/ecmwf_24h_probability_20260106_00z_all_thresholds.png

======================================================================
PROCESSING COMPLETE!
======================================================================

Summary:
  Model Date: 2026-01-06 00Z
  Ensemble Members: 51
  Total Time: 1726.3 seconds (28.8 minutes)
```

## Three-Stage Pipeline (for reference)

If you need to create new parquet files for a different date:

```bash
# Run the tutorial script with Stage 1
python run_ecmwf_tutorial.py --date 20260107 --run-stage1

# Or use the production script
cd ../../ecmwf
python ecmwf_three_stage_multidate.py --use-local-template --dates 20260107
```

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [ECMWF on AWS](https://registry.opendata.aws/ecmwf-forecasts/)
- [gribberish](https://github.com/mpiannucci/gribberish)
- [Kerchunk](https://fsspec.github.io/kerchunk/)
- [Zarr](https://zarr.readthedocs.io/)

## Acknowledgements

This work was funded by:
1. **E4DRR Project** - UN Complex Risk Analytics Fund (CRAF'd)
   https://icpac-igad.github.io/e4drr/
2. **SEWAA Project** - Strengthening Early Warning Systems for Anticipatory Action
   https://cgan.icpac.net/
