# GIK Pipeline Comparison: CFS, ECMWF, and GEFS

A technical comparison of the three GIK (Grib-Index-Kerchunk) pipeline implementations,
how each builds parquet reference files, and a reflection on eliminating `scan_grib`
across all products.

---

## 1. Overview of the Three Pipelines

| Aspect | CFS (`run_lithops_cfs.py`) | ECMWF (`run_lithops_ecmwf.py`) | GEFS (`run_gefs_tutorial.py`) |
|--------|---------------------------|--------------------------------|-------------------------------|
| **Data source** | `s3://noaa-cfs-pds/` | `s3://ecmwf-forecasts/` | `s3://noaa-gefs-pds/` |
| **Index format** | `.idx` (colon-separated text) | `.index` (JSON-lines) | `.idx` (colon-separated text) |
| **Stage 1 method** | Pre-built template (no `scan_grib`) | Pre-built template (no `scan_grib`) | `scan_grib` on 2 GRIB files |
| **Stage 1 time** | ~5 seconds | ~5 seconds | ~30 seconds |
| **Self-contained?** | Yes (no kerchunk on workers) | Yes (no kerchunk on workers) | No (imports `gefs_util.py`) |
| **Ensemble structure** | 124 members (31 dates x 4 runs) | 51 members (1 ctrl + 50 ens) | 30 members (gep01-gep30) |
| **Forecast range** | 6-monthly (5160h, 6-hourly) | 15-day (360h, 3h then 6h) | 10-day (240h, 3-hourly) |
| **Files per member** | ~861 GRIB files | 85 shared mega-files | 81 GRIB files |
| **Uses kerchunk?** | No | No | Yes |

---

## 2. The Three Stages Explained

Every GIK pipeline follows the same three-stage pattern to produce a parquet
reference file:

```
Stage 1: Build zarr metadata skeleton    (what variables/dimensions exist)
Stage 2: Parse index files for byte refs  (where each chunk lives in the GRIB)
Stage 3: Merge + save as parquet          (combine skeleton + refs → final file)
```

The fundamental insight is that **the zarr structure (Stage 1) is the same for
every forecast run** of the same model. Variable names, dimensions, chunk layouts,
and coordinate schemas do not change between runs. Only the **byte offsets** change
(Stage 2), because each run writes different-sized GRIB messages.

---

## 3. Stage 1: How Each Pipeline Gets the Zarr Skeleton

### CFS: Pre-Built Template Parquet (No `scan_grib`)

```
Template file:  cfs-zarr-template-20251101-00.parquet  (~1705 metadata keys, 68 variables)
Created by:     run_cfs_template_creation.py --zarr-template
One-time cost:  ~24 seconds (scan_grib on f000 + f006)
Daily cost:     ~5 seconds (read parquet from disk or tar.gz)
```

The CFS pipeline loads the zarr skeleton from a pre-built parquet file. This
parquet contains all `.zattrs`, `.zarray`, and `.zgroup` entries for every
variable group — but no data chunk references.

```python
# Stage 1 in run_lithops_cfs.py
def build_deflated_store_from_template(template_path, init_date, run):
    if template_path.endswith('.parquet'):
        zstore = _load_parquet_as_zstore(template_path)  # ~5 seconds
        return zstore
    # ... or extract from tar.gz archive
```

**Why no `scan_grib` is needed at runtime:** The zarr structure for CFS flux files
is identical across all init dates and runs. A one-time scan of the reference date
(2025-11-01 00z) captures the complete schema. The template parquet is reused for
all 124 ensemble members.

### ECMWF: Pre-Built Template Archive (No `scan_grib`)

```
Template file:  gik-fmrc-v2ecmwf_fmrc.tar.gz (HuggingFace)
Created from:   Reference date 20240529
One-time cost:  Original scan_grib run (~73 minutes)
Daily cost:     ~5 seconds (extract parquet from tar.gz)
```

The ECMWF pipeline uses the exact same approach as CFS. The zarr skeleton for
each of the 51 members was captured once from a reference date and archived
to HuggingFace.

```python
# Stage 1 in run_lithops_ecmwf.py
def build_deflated_stores_from_template(template_tar_path):
    with tarfile.open(template_tar_path, 'r:gz') as tar:
        for member_dir in all_members:  # ens_control, ens_01, ..., ens_50
            template_df = pd.read_parquet(io.BytesIO(parquet_bytes))
            zstore = {}
            for _, row in template_df.iterrows():
                zstore[row['key']] = parse_value(row['value'])
            deflated_stores[member] = zstore
    return deflated_stores  # 51 zarr skeletons, ~5 seconds total
```

### GEFS: Live `scan_grib` (Still Uses Kerchunk)

```
Template file:  gik-fmrc-gefs-20241112.tar.gz (HuggingFace) — mapping parquets only
Scan target:    f000 + f003 (2 GRIB files per member)
Runtime cost:   ~30 seconds per member
```

The GEFS pipeline **still runs `scan_grib` on every execution**. It scans the
first two GRIB files (`f000` and `f003`) to build the zarr tree:

```python
# Stage 1 in run_gefs_tutorial.py
def process_ensemble_member(member, target_date, target_run, ...):
    # Build GRIB URLs for structure scanning
    gefs_files = [
        f"s3://noaa-gefs-pds/gefs.{target_date}/{target_run}/.../f000",
        f"s3://noaa-gefs-pds/gefs.{target_date}/{target_run}/.../f003",
    ]

    # Stage 1: LIVE scan — requires kerchunk + cfgrib + eccodes
    _, deflated_store = filter_build_grib_tree(gefs_files, forecast_dict)
```

This calls `kerchunk.grib2.scan_grib()` which:
1. Downloads the GRIB file header
2. Iterates message-by-message using cfgrib/eccodes
3. Extracts variable names, dimensions, grid specs, compression info
4. Builds zarr-compatible references via `grib_tree()`
5. Strips data chunk refs via `strip_datavar_chunks()` → returns deflated store

**Why 2 files?** Scanning `f000` captures most variables, but some variables only
appear starting from `f003` (accumulated fields like precipitation). Together they
capture the complete variable/dimension schema.

---

## 4. Stage 2: Parsing Index Files for Byte References

This is where each pipeline reads the lightweight index files that NOAA/ECMWF
publish alongside every GRIB file. These index files are text (~1-100 KB) and
contain the byte offset of every GRIB message.

### CFS `.idx` Format (Colon-Separated Text)

```
1:0:d=2025110100:UFLX:surface:6 hour fcst:
2:46931:d=2025110100:VFLX:surface:6 hour fcst:
...
31:1713139:d=2025110100:PRATE:surface:6 hour fcst:
36.1:1811792:d=2025110100:UGRD:10 m above ground:6 hour fcst:
36.2:1811792:d=2025110100:VGRD:10 m above ground:6 hour fcst:
37:1930034:d=2025110100:TMP:2 m above ground:6 hour fcst:
```

**CFS-specific quirks:**
- Fractional record numbers (`36.1`, `36.2`) for vector components packed in
  one GRIB message (UGRD/VGRD share the same byte offset)
- Byte length must be calculated from consecutive offsets
  (`offset[i+1] - offset[i]`), not provided directly
- ~103 entries per file, covering all flux variables

**CFS parsing (`parse_cfs_idx`):**
```python
def parse_cfs_idx(idx_url):
    # Read text, split by lines
    for line in content.strip().split('\n'):
        parts = line.strip().rstrip(':').split(':')
        record = float(parts[0])   # float, not int! (handles 36.1, 36.2)
        byte_offset = int(parts[1])
        variable = parts[3]        # e.g., "PRATE"
        level = parts[4]           # e.g., "surface"

    # Calculate byte_length from consecutive unique offsets
    # Shared-offset entries (UGRD/VGRD) get the same length
```

### ECMWF `.index` Format (JSON-Lines)

```json
{"_offset": 3456, "_length": 12345, "param": "tp", "number": 1, "levtype": "sfc", "step": "0"}
{"_offset": 15801, "_length": 12200, "param": "2t", "number": 1, "levtype": "sfc", "step": "0"}
{"_offset": 28001, "_length": 12100, "param": "tp", "number": 2, "levtype": "sfc", "step": "0"}
```

**ECMWF-specific notes:**
- Both `_offset` and `_length` are provided directly (no calculation needed)
- All 51 members are interleaved in one file per timestep
- Must filter by `number` field to extract a specific member
- Uses short parameter names (`tp`, `2t`, `u`, `v`) not GRIB abbreviations

**ECMWF parsing (`parse_grib_index`):**
```python
def parse_grib_index(idx_url, member_filter=None):
    for line in f:
        entry_data = json.loads(line.strip())    # Direct JSON parsing
        member_num = int(entry_data['number'])   # Filter by member
        offset = entry_data['_offset']           # Provided directly
        length = entry_data['_length']           # Provided directly
```

### GEFS `.idx` Format (Colon-Separated Text)

```
1:0:d=20250106:TMP:2 m above ground:3 hour fcst:
2:456789:d=20250106:APCP:surface:0-3 hour acc fcst:
3:890123:d=20250106:UGRD:10 m above ground:3 hour fcst:
```

**GEFS-specific notes:**
- One file per member per timestep (no member filtering needed)
- Same colon-separated format as CFS but without fractional record numbers
- Byte length calculated from consecutive offsets (same as CFS)

**GEFS parsing (uses kerchunk):**
```python
# Uses kerchunk._grib_idx.parse_grib_idx()
# Then merges with pre-built mapping parquets via map_from_index()
gefs_kind = cs_create_mapped_index_local(
    axes, target_date, member,
    tar_gz_path=mapping_manager.tar_gz_path,
    mapping_manager=mapping_manager
)
```

The GEFS pipeline relies on kerchunk's `parse_grib_idx` and `map_from_index`
functions rather than parsing the `.idx` text directly. This is the key
difference — CFS and ECMWF parse index files independently of kerchunk.

---

## 5. Stage 3: Merge + Save Parquet

All three pipelines converge to the same output format: a parquet file with
columns `[key, value]` where each row is a zarr store entry.

**Metadata rows** (from Stage 1 template):
```
key: "prate/instant/surface/.zattrs"     value: '{"_ARRAY_DIMENSIONS": ["latitude", "longitude"], ...}'
key: "prate/instant/surface/prate/.zarray"  value: '{"chunks": [181, 360], "dtype": "<f4", ...}'
```

**Data reference rows** (from Stage 2 index parsing):
```
key: "step_0006/prate/surface/0.0.0"    value: '["s3://noaa-cfs-pds/.../flxf2025110106.01.2025110100.grb2", 1713139, 35895]'
key: "step_0012/tmp/2_m_above_ground/0.0.0"  value: '["s3://noaa-cfs-pds/.../flxf2025110112.01.2025110100.grb2", 1930034, 77620]'
```

The merge is a simple dict overlay:

```python
# Same logic in all three pipelines
merged = template_store.copy()         # Start with zarr metadata
for key, ref in index_refs.items():    # Overlay fresh byte-range refs
    merged[key] = ref

# Save as parquet
df = pd.DataFrame([(k, encode(v)) for k, v in merged.items()], columns=['key', 'value'])
df.to_parquet(output_file)
```

---

## 6. Architectural Comparison

### CFS Lithops (Template-Based, Self-Contained)

```
run_lithops_cfs.py  (~1200 lines, self-contained)
    │
    ├─ Stage 1: Load parquet template → 1705 metadata keys (~5s)
    │   - No scan_grib, no kerchunk dependency
    │   - Template from HuggingFace tar.gz or local parquet
    │
    ├─ Stage 2: Parse .idx files → ALL 103 variables per timestep
    │   - Custom parse_cfs_idx() handles fractional records (36.1, 36.2)
    │   - ThreadPoolExecutor for parallel I/O
    │   - Generates ~925 refs for 48h test (103 vars x 9 timesteps)
    │
    └─ Stage 3: Merge + save → single parquet per member
        - 2630 total keys (1705 metadata + 925 data refs)
```

### ECMWF Lithops (Template-Based, Self-Contained)

```
run_lithops_ecmwf.py  (~1075 lines, self-contained)
    │
    ├─ Stage 1: Extract member parquets from tar.gz → 51 zarr skeletons (~5s)
    │   - No scan_grib, no kerchunk dependency
    │   - Template per member from HuggingFace archive
    │
    ├─ Stage 2: Parse .index files → JSON-lines with member filtering
    │   - Direct json.loads() per line
    │   - Filter by member number field
    │   - 85 timesteps per member
    │
    └─ Stage 3: Merge + save → one parquet per member (51 total)
        - Upload to GCS
```

### GEFS Tutorial (Kerchunk-Based)

```
run_gefs_tutorial.py  (~424 lines, imports gefs_util.py ~1100 lines)
    │
    ├─ Stage 1: scan_grib on f000 + f003 → grib_tree → deflated store (~30s)
    │   - REQUIRES kerchunk, cfgrib, eccodes
    │   - Live GRIB scanning on every run
    │   - filter_build_grib_tree() from gefs_util.py
    │
    ├─ Stage 2: parse_grib_idx + map_from_index using template parquets
    │   - REQUIRES kerchunk._grib_idx functions
    │   - Merges with pre-built mapping parquets from tar.gz
    │   - cs_create_mapped_index_local() from gefs_util.py
    │
    └─ Stage 3: process_unique_groups → zarr store → parquet
        - prepare_zarr_store() + process_unique_groups() from gefs_util.py
        - More complex zarr store construction
```

---

## 7. Why CFS and ECMWF Don't Need `scan_grib`

The `scan_grib` function from kerchunk does two things:

1. **Discovers the zarr structure** — what variables exist, their dimensions,
   chunk shapes, coordinate metadata, compression codecs
2. **Extracts byte references** — the `[url, offset, length]` for each data chunk

The key insight is that **these two outputs serve different purposes and change
at different rates:**

| What | Changes between runs? | How to get it |
|------|----------------------|---------------|
| Zarr structure (variables, dimensions, chunks) | No | Template (one-time scan) |
| Byte references (offset, length per chunk) | Yes (every run) | Index file (`.idx` / `.index`) |

Since the zarr structure is stable, we only need to run `scan_grib` once to
capture it as a template. For every subsequent run, we load the template (Stage 1)
and get fresh byte references from the index files (Stage 2).

**CFS eliminated `scan_grib` by:**
1. Running `scan_grib` once on `flxf2025110100.01.2025110100.grb2` (f000) and
   `flxf2025110106.01.2025110100.grb2` (f006)
2. Building `grib_tree` with `remote_options={"anon": True}`
3. Stripping data chunks via `strip_datavar_chunks`
4. Saving the 1705 metadata keys as a parquet file
5. Loading this parquet at runtime instead of scanning

**ECMWF eliminated `scan_grib` by:**
1. Running `scan_grib` once on reference date `20240529`
2. Archiving the per-member zarr stores as parquets in a tar.gz
3. Uploading to HuggingFace
4. Loading from tar.gz at runtime (~5 seconds)

---

## 8. How GEFS Can Be Made Template-Based (No `scan_grib`)

The GEFS pipeline currently runs `scan_grib` on every execution because it was
built before the template-based approach was developed. Here is a concrete plan
to eliminate `scan_grib` from GEFS:

### Step 1: Create a Zarr Template (One-Time)

Run `scan_grib` on two reference-date GRIB files and save the deflated store:

```python
# One-time: create_gefs_zarr_template.py
from kerchunk.grib2 import scan_grib
from kerchunk._grib_idx import grib_tree, strip_datavar_chunks

# Scan reference files
ref_files = [
    "s3://noaa-gefs-pds/gefs.20241112/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f000",
    "s3://noaa-gefs-pds/gefs.20241112/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003",
]
all_groups = []
for f in ref_files:
    groups = scan_grib(f, storage_options={"anon": True})
    all_groups.extend(groups)

# Build tree and strip data refs
tree = grib_tree(all_groups, remote_options={"anon": True})
deflated = strip_datavar_chunks(tree)

# Save as template parquet
data = [(k, json.dumps(v) if isinstance(v, (list, dict)) else str(v))
        for k, v in deflated.items()]
df = pd.DataFrame(data, columns=['key', 'value'])
df.to_parquet('gefs-zarr-template-20241112-00.parquet')
```

### Step 2: Replace `filter_build_grib_tree` with Template Loading

In the GEFS pipeline, replace the `scan_grib` call:

```python
# BEFORE (current — requires kerchunk at runtime):
_, deflated_store = filter_build_grib_tree(gefs_files, forecast_dict)

# AFTER (template-based — no kerchunk needed):
template_df = pd.read_parquet('gefs-zarr-template-20241112-00.parquet')
deflated_store = {}
for _, row in template_df.iterrows():
    key = row['key']
    value = row['value']
    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
        value = json.loads(value)
    deflated_store[key] = value
```

### Step 3: Replace `parse_grib_idx` + `map_from_index` with Direct Parsing

The GEFS `.idx` format is simpler than CFS (no fractional records):

```
1:0:d=20250106:TMP:2 m above ground:3 hour fcst:
2:456789:d=20250106:APCP:surface:0-3 hour acc fcst:
```

A standalone parser (no kerchunk needed):

```python
def parse_gefs_idx(idx_url):
    """Parse GEFS .idx file directly — no kerchunk dependency."""
    fs = fsspec.filesystem("s3", anon=True)
    content = fs.cat(idx_url).decode('utf-8')

    raw_entries = []
    for line in content.strip().split('\n'):
        parts = line.strip().rstrip(':').split(':')
        if len(parts) < 6:
            continue
        raw_entries.append({
            'record': int(parts[0]),
            'byte_offset': int(parts[1]),
            'variable': parts[3],
            'level': parts[4],
        })

    # Calculate byte_length from consecutive offsets
    for i in range(len(raw_entries)):
        if i + 1 < len(raw_entries):
            raw_entries[i]['byte_length'] = raw_entries[i+1]['byte_offset'] - raw_entries[i]['byte_offset']
        else:
            raw_entries[i]['byte_length'] = -1  # Last entry: use file size

    return raw_entries
```

### Step 4: Remove Kerchunk Dependencies

With these changes, the GEFS pipeline would:
- **No longer import** `kerchunk`, `cfgrib`, or `eccodes`
- **No longer need** `gefs_util.py` baked into the Docker image
- **Become self-contained** like CFS and ECMWF Lithops scripts
- **Stage 1** drops from ~30 seconds to ~5 seconds

### What Stays the Same

- The pre-built **mapping parquets** in the HuggingFace tar.gz
  (`gik-fmrc-gefs-20241112.tar.gz`) would still be used for Stage 2's
  `map_from_index` step — OR could be replaced entirely by the direct
  `.idx` parsing approach (like CFS does)
- The final parquet format (Stage 3) is unchanged
- All downstream data streaming code works identically

---

## 9. Summary: Evolution from Kerchunk to Template-Based

```
                    GEFS (current)         ECMWF (Lithops)        CFS (Lithops)
                    ──────────────         ───────────────        ─────────────
Stage 1             scan_grib (live)       Template (pre-built)   Template (pre-built)
                    ~30s, needs kerchunk   ~5s, no kerchunk       ~5s, no kerchunk

Stage 2             kerchunk parse_grib_idx  Custom JSON parser   Custom .idx parser
                    + map_from_index       ~5-15 min              ~5-15 min
                    ~2-3 min

Stage 3             process_unique_groups  Dict merge + parquet   Dict merge + parquet
                    ~1 min                 ~1-2s                  ~1-2s

Dependencies        kerchunk, cfgrib,      pandas, fsspec,        pandas, fsspec,
                    eccodes, zarr,         s3fs, pyarrow          s3fs, pyarrow
                    xarray, gefs_util.py

Self-contained?     No                     Yes                    Yes
Docker image        Must include           Minimal Python +       Minimal Python +
                    gefs_util.py +         pip packages           pip packages
                    kerchunk + eccodes
```

The trajectory is clear: **template-based pipelines are simpler, faster, and
have fewer dependencies.** The GEFS pipeline is the last to be migrated but
follows the same pattern.

### Template Validity

Templates remain valid as long as the forecast model configuration does not
change. Specifically:

- **Grid resolution** (0.25 deg for all three)
- **Variable set** (which GRIB messages are included)
- **Dimension layout** (lat/lon grid, forecast hours)
- **Compression codecs** (JPEG2000, CCSDS, etc.)

If NOAA or ECMWF changes their model grid or variable list, a new template
must be regenerated by running `scan_grib` once on the updated GRIB files.
This is a rare event (model upgrades happen every few years).

---

## 10. File Reference

| File | Lines | Role |
|------|-------|------|
| `cfs/lithops-cr-gik-cfs/run_lithops_cfs.py` | ~1200 | CFS Lithops pipeline (template-based, self-contained) |
| `cfs/run_cfs_template_creation.py` | ~340 | One-time CFS template + mapping parquet creation |
| `ecmwf/lithops-cr-gik-ecmwf/run_lithops_ecmwf.py` | ~1075 | ECMWF Lithops pipeline (template-based, self-contained) |
| `tutorial/gefs/run_gefs_tutorial.py` | ~424 | GEFS tutorial (kerchunk-based, imports gefs_util.py) |
| `gefs/gefs_util.py` | ~1100 | GEFS utilities (scan_grib, parse_grib_idx, etc.) |
