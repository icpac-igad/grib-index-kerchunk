# Grib-Index-Kerchunk (GIK) Tutorials

## Quick Start

Stream ensemble weather forecast data directly from cloud storage without downloading full GRIB files.

| Tutorial | Members | Runtime | Output |
|----------|---------|---------|--------|
| **[GEFS](./gefs/)** | 30 | ~14 min | 24h rainfall probability maps |
| **[ECMWF](./ecmwf/)** | 51 | ~29 min | 24h rainfall probability maps |

## What is GIK?

The **Grib-Index-Kerchunk** method creates zarr-compatible parquet reference files that enable:
- **No full file downloads** - fetch only the bytes you need
- **80x faster decoding** - gribberish (Rust) vs cfgrib
- **Cloud-native streaming** - data stays on S3, you stream it

## Quickest Demo

```bash
# GEFS (30 members, ~14 minutes)
cd gefs
python run_gefs_data_streaming_v2.py

# ECMWF (51 members, ~29 minutes)
cd ecmwf
python run_ecmwf_data_streaming_20260106.py
```

Both scripts will:
1. Stream precipitation data from AWS S3
2. Calculate 24-hour exceedance probabilities
3. Generate multi-panel probability plots

## Performance Summary

### Data Streaming (with gribberish + parallel fetching)

| Model | Members | Timesteps | Time/Member | Total Time |
|-------|---------|-----------|-------------|------------|
| GEFS | 30 | 80 | ~28s | ~14 min |
| ECMWF | 51 | 85 | ~34s | ~29 min |

### Speed Breakdown (ECMWF example)

| Operation | Sequential | Parallel (8x) |
|-----------|------------|---------------|
| S3 Fetch | 2.5s/step | 0.34s/step avg |
| Decode | 30ms/step | 30ms/step |
| **Per Member** | ~180s | ~34s |

## Prerequisites

```bash
# Core packages
pip install kerchunk zarr xarray pandas numpy fsspec s3fs

# Fast decoding (highly recommended)
pip install gribberish

# Plotting
pip install matplotlib cartopy geopandas
```

## Tutorial Details

### GEFS Tutorial ([gefs/README.md](./gefs/README.md))
- **Source**: NOAA GEFS on AWS (`s3://noaa-gefs-pds/`)
- **Members**: 30 (gep01-gep30)
- **Resolution**: 0.25 degree
- **Variable**: Total precipitation (APCP)
- **Region**: East Africa (21-53E, 12S-23N)

### ECMWF Tutorial ([ecmwf/README.md](./ecmwf/README.md))
- **Source**: ECMWF on AWS (`s3://ecmwf-forecasts/`)
- **Members**: 51 (control + ens01-ens50)
- **Resolution**: 0.25 degree
- **Variable**: Total precipitation (tp)
- **Region**: East Africa (21-53E, 12S-23N)

## Three-Stage Pipeline

```
Stage 1: Scan GRIB files (ONE-TIME, ~30 min)
    ↓   Creates: ecmwf_{date}_{run}z_efficient.zip

Stage 2: Merge index + template (fast, ~5 min)
    ↓   Downloads fresh .index files from S3
    ↓   Merges with HuggingFace template

Stage 3: Create final parquet (fast, ~2 min)
    ↓   Output: stage3_{member}_final.parquet

Data Streaming: Stream & decode (parallel, ~30s/member)
        Uses gribberish for 80x faster decoding
```

## Pre-built Templates

Templates are downloaded automatically from Hugging Face:

| Dataset | Template | Size |
|---------|----------|------|
| GEFS | [gik-fmrc-gefs-20241112.tar.gz](https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/blob/main/gik-fmrc-gefs-20241112.tar.gz) | ~3 MB |
| ECMWF | [gik-fmrc-v2ecmwf_fmrc.tar.gz](https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/blob/main/gik-fmrc-v2ecmwf_fmrc.tar.gz) | ~120 MB |

## Output Examples

### GEFS Output
```
output_plots/gefs_24h_probability_20250106_00z_all_thresholds.png
```
- 9 forecast days x 6 thresholds (5, 25, 50, 75, 100, 125 mm)
- Based on 30 ensemble members

### ECMWF Output
```
output_plots/ecmwf_24h_probability_20260106_00z_all_thresholds.png
```
- 10 forecast days x 6 thresholds
- Based on 51 ensemble members

## Key Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `gefs/run_gefs_tutorial.py` | Create GEFS parquet files | ~5 min |
| `gefs/run_gefs_data_streaming_v2.py` | Stream GEFS data + plot | ~14 min |
| `ecmwf/run_ecmwf_tutorial.py` | Create ECMWF parquet files | ~5-30 min |
| `ecmwf/run_ecmwf_data_streaming_20260106.py` | Stream ECMWF data + plot | ~29 min |

## Memory Efficiency

Both data streaming scripts use **disk-based zarr storage** to avoid memory issues:
- Data is written to temporary zarr store on disk
- Only one member's data is held in memory at a time
- Automatic cleanup after processing

```python
# Storage shape example (ECMWF)
# (51 members, 85 timesteps, 141 lats, 129 lons)
temp_zarr_cache/ecmwf_20260106_00z_*/precipitation
```

## Directory Structure

```
tutorial/
├── README.md                    # This file
├── gefs/
│   ├── README.md               # GEFS deep dive
│   ├── run_gefs_tutorial.py    # Create parquet files
│   ├── run_gefs_data_streaming_v2.py  # Stream + plot
│   ├── ea_ghcf_simple.geojson  # East Africa boundaries
│   ├── output_parquet/         # Parquet reference files
│   └── output_plots/           # Generated plots
└── ecmwf/
    ├── README.md               # ECMWF deep dive
    ├── run_ecmwf_tutorial.py   # Create parquet files
    ├── run_ecmwf_data_streaming_20260106.py  # Stream + plot
    ├── ea_ghcf_simple.geojson  # East Africa boundaries
    ├── ecmwf_three_stage_20260106_00z/  # Parquet files
    └── output_plots/           # Generated plots
```

## Comparison: Traditional vs GIK

| Aspect | Traditional | GIK Method |
|--------|-------------|------------|
| Download | Full GRIB files (100s MB) | Index files only (KB) |
| Decode | cfgrib (~2s/chunk) | gribberish (~30ms/chunk) |
| Memory | Load entire files | Stream on demand |
| Storage | Local copies needed | Data stays on cloud |

## References

- [NOAA GEFS on AWS](https://registry.opendata.aws/noaa-gefs/)
- [ECMWF on AWS](https://registry.opendata.aws/ecmwf-forecasts/)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- [gribberish](https://github.com/mpiannucci/gribberish)

## Acknowledgements

This work was funded by:
1. **E4DRR Project** - UN Complex Risk Analytics Fund (CRAF'd)
   https://icpac-igad.github.io/e4drr/
2. **SEWAA Project** - Strengthening Early Warning Systems for Anticipatory Action
   https://cgan.icpac.net/
