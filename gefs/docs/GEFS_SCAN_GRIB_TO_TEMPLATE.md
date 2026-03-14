# GEFS Stage 1: scan_grib to Template Transformation

## Overview

This document describes the replacement of `kerchunk.grib2.scan_grib()` with a
pre-built parquet template for GEFS Stage 1 processing, achieving a **112x
speedup** (5.5s to 0.05s per invocation). This mirrors the optimization
previously applied to the ECMWF pipeline (commit `e0a41671`).

## Background

The GIK three-stage pipeline creates lightweight parquet reference files that
point into remote GRIB files on S3. Stage 1 builds the zarr metadata structure
(the "deflated store") which defines variable names, dimensions, chunk layouts,
and coordinate metadata. This structure is **identical across all 30 GEFS ensemble
members** for a given model configuration.

### Before: scan_grib approach

```
scan_grib(f000, f003)  →  grib_tree()  →  strip_datavar_chunks()  →  deflated store
         ~5.5s                ~0.5s              instant
```

- Called once per member (30 times per date), even though the result is identical
- Required downloading 2 GRIB files from S3 per call
- Required `kerchunk`, `cfgrib`, and `eccodes` dependencies
- Total for 30 members: **~165 seconds** of redundant computation

### After: template approach

```
read_parquet(template)  →  deflated store
        ~0.05s
```

- Called **once** per date, result reused for all 30 members
- No S3 downloads needed for Stage 1
- Total: **0.05 seconds**

## Implementation

### New files

| File | Description |
|------|-------------|
| `gefs-deflated-store-template-20241112.parquet` | Pre-built deflated store (26 KB, 915 zarr metadata entries) |

### Modified files

| File | Changes |
|------|---------|
| `gefs_util.py` | Added `build_gefs_deflated_store_from_template()`, `_load_deflated_parquet_from_tar()`, `_filter_deflated_refs()` |
| `run_day_gefs_ensemble_full.py` | Loads template once in `main()`, passes to all member processing; falls back to scan_grib |
| `tutorial/gefs/run_gefs_tutorial.py` | Updated to use template-based Stage 1 |

### Key function: `build_gefs_deflated_store_from_template()`

```python
def build_gefs_deflated_store_from_template(
    template_path: str,
    filter_vars: Optional[Dict[str, str]] = None
) -> dict:
```

**Parameters:**
- `template_path`: Path to the `.parquet` template file, or a `.tar.gz` archive
  containing one
- `filter_vars`: Optional dict to keep only specific variables (e.g.,
  `{"Total Precipitation": "APCP:surface"}`)

**Returns:** `{'version': 1, 'refs': {...}}` — same format as
`strip_datavar_chunks()` output, compatible with `prepare_zarr_store()`.

**Important:** Values in the refs dict are kept as raw strings (JSON for
`.zattrs`/`.zarray`, `base64:` for coordinate data). They must NOT be parsed
into Python dicts, because kerchunk's `store_coord_var` and `store_data_var`
expect JSON strings.

## Template Generation

The template was generated from reference date **2024-11-12** using:

```python
from kerchunk.grib2 import scan_grib
from kerchunk._grib_idx import build_idx_grib_mapping

# Scan 2 GRIB files to capture full variable/dimension schema
scanned = []
for fxx in [0, 3]:
    url = f"s3://noaa-gefs-pds/gefs.20241112/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f{fxx:03d}"
    scanned.extend(scan_grib(url, storage_options={"anon": True}))

# Build zarr tree and strip data chunks
store = grib_tree(scanned, remote_options={"anon": True})
strip_datavar_chunks(store)

# Serialize to parquet
zstore_dict_to_df(store["refs"]).to_parquet("gefs-deflated-store-template-20241112.parquet")
```

### Why only 2 files

The zarr structure (variable names, dimension names, chunk shapes, coordinate
schemas) is identical across all 81 timesteps (f000-f240) of a forecast run.
Scanning `f000` and `f003` captures the complete variable/dimension schema.
The actual byte-range references for all timesteps come from Stage 2.

### Template contents

The 915 entries include:
- `.zgroup` entries for the zarr hierarchy
- `.zattrs` entries with variable metadata (GRIB attributes, CF conventions)
- `.zarray` entries defining chunk shapes, dtypes, compression
- `base64:`-encoded coordinate data (latitude, longitude grids)
- Covers all **39 GEFS variable paths** (t2m, tp, u10, v10, cape, etc.)

## Variable Name Mapping

A key challenge: GEFS `.idx` files use GRIB abbreviations (TMP, APCP, UGRD)
while zarr paths use cfgrib short names (t2m, tp, u10). The `_filter_deflated_refs()`
function includes a static `GRIB_TO_CFGRIB` lookup table mapping ~35 common
variables:

```python
GRIB_TO_CFGRIB = {
    'tmp': 't2m', 'apcp': 'tp', 'ugrd': 'u10', 'vgrd': 'v10',
    'pres': 'sp', 'cape': 'cape', 'pwat': 'pwat', 'tcdc': 'tcc',
    'dpt': 'd2m', 'hgt': 'gh', 'gust': 'gust', ...
}
```

## Template Validity

The template remains valid as long as NOAA does not change:
- GEFS grid resolution (currently 0.25 deg, 721x1440)
- Variable set or naming
- Dimension layout or chunk structure

If the model configuration changes, regenerate the template by running the
scan_grib pipeline on a date with the new configuration.

## Comparison with ECMWF

| Aspect | ECMWF | GEFS |
|--------|-------|------|
| Template source | HuggingFace tar.gz | Local parquet (+ tar.gz support) |
| Template format | Key/value zarr metadata (2776 rows) | Key/value zarr metadata (915 rows) |
| Reference date | 2024-05-29 | 2024-11-12 |
| Original Stage 1 time | ~73 min (scan_grib) | ~5.5s (scan_grib) |
| Template Stage 1 time | ~5s | ~0.05s |
| Speedup | ~876x | ~112x |
| Fallback | No (template required) | Yes (falls back to scan_grib) |

## Fallback Behavior

Both `run_day_gefs_ensemble_full.py` and the tutorial script automatically
fall back to `scan_grib` if the template file is not found:

```python
if template_path.exists():
    deflated_store = build_gefs_deflated_store_from_template(str(template_path))
else:
    print("Template not found, using scan_grib fallback")
    deflated_store = None  # scan_grib runs per-member in process_ensemble_member()
```

## Performance Summary

| Scenario | Stage 1 Time | Notes |
|----------|-------------|-------|
| scan_grib per member | 30 x 5.5s = 165s | Original approach |
| Template load once | 1 x 0.05s = 0.05s | Template approach |
| **Speedup** | **3300x total** | For full 30-member ensemble |
