# VirtualiZarr/Icechunk Migration Roadmap for GEFS and ECMWF Parsers

## Executive Summary

This document outlines the two-phase migration path to transform the current parquet-based Grib-Index-Kerchunk (GIK) workflow into full-fledged VirtualiZarr plugins with Icechunk persistence, following the HRRRparser architecture pattern.

---

## Current State Analysis

### Phase 1: Parquet Reference Creation (Current Implementation)

**GEFS** (`run_gefs_tutorial.py`):
```
Template TAR.GZ → Scan GRIB structure → Merge with templates → Parquet file
     ↓                    ↓                      ↓                 ↓
HuggingFace        filter_build_grib_tree  cs_create_mapped_index  DataFrame
```

**ECMWF** (`run_ecmwf_tutorial.py`):
```
Stage 1: GRIB scanning (optional) → deflated parquet
Stage 2: Fresh index + template metadata → merged references
Stage 3: Final zarr-compatible parquet files
```

**Output Format**: Parquet files containing `[key, value]` pairs where value is either:
- JSON metadata (`.zarray`, `.zattrs`, `.zgroup`)
- S3 reference triplets `[url, offset, length]` for chunk data

### Phase 2: Data Streaming (Current Implementation)

**GEFS** (`run_gefs_data_streaming_v2.py`):
```python
# Current: Manual pandas → fsspec → gribberish → zarr temp store
df = pd.read_parquet(parquet_path)
zstore = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}
grib_bytes = fetch_grib_bytes(zstore[chunk_key], fs)  # fsspec S3
array_2d = gribberish.parse_grib_array(grib_bytes, 0)  # Manual decode
zarr_store['precipitation'][idx] = array_2d  # Manual assembly
```

**ECMWF** (`run_ecmwf_data_streaming_20260106.py`):
- Same pattern with parallel S3 fetching via ThreadPoolExecutor
- Handles ECMWF-specific longitude conventions (-180 to 180)

---

## Target Architecture: VirtualiZarr Plugin Pattern

### Reference: HRRRparser Architecture

The HRRRparser (`/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/icpac-gik/hrrr-parser/`) provides the canonical pattern:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HRRRParser                                  │
├─────────────────────────────────────────────────────────────────────┤
│  __init__(steps: int)                                               │
│    └── register_codec(CODEC_ID, HRRRGribberishCodec)               │
│                                                                     │
│  __call__(url, registry) → ManifestStore                           │
│    ├── ObstoreReader: read GRIB file bytes                         │
│    ├── _scan_messages: extract byte offsets + metadata             │
│    ├── _create_variable_array → ManifestArray                      │
│    │     ├── ChunkManifest: {ChunkKey: ChunkEntry(path,offset,len)}│
│    │     └── create_v3_array_metadata(codecs=[HRRRGribberishCodec])│
│    └── ManifestStore(registry, ManifestGroup)                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    HRRRGribberishCodec                              │
├─────────────────────────────────────────────────────────────────────┤
│  class HRRRGribberishCodec(ArrayBytesCodec):                       │
│    └── async _decode_single(chunk_data, chunk_spec) → NDBuffer     │
│          ├── gribberish.parse_grib_message_metadata() for coords   │
│          └── gribberish.parse_grib_array() for data arrays         │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components Required

| Component | HRRRparser | GEFSparser (needed) | ECMWFparser (needed) |
|-----------|------------|---------------------|----------------------|
| Parser Protocol | `HRRRParser.__call__()` | `GEFSParser.__call__()` | `ECMWFParser.__call__()` |
| Codec Class | `HRRRGribberishCodec` | `GEFSGribberishCodec` | `ECMWFGribberishCodec` |
| Coordinate Handling | HRRR-specific (Lambert) | GEFS 0-360 longitude | ECMWF -180 to 180 |
| Ensemble Handling | Single file | 30 members (separate files) | 51 members (shared files) |

---

## Migration Phase 1: Parquet → ManifestStore/Icechunk

### 1.1 GEFSParser Implementation

**Goal**: Transform `run_gefs_tutorial.py` output to ManifestStore directly

```python
# Target: gefsparser/parser.py
from virtualizarr.manifests import ChunkEntry, ChunkManifest, ManifestArray, ManifestStore
from virtualizarr.manifests.utils import create_v3_array_metadata
from zarr.registry import register_codec

class GEFSParser:
    """VirtualiZarr parser for GEFS ensemble data."""

    def __init__(self, ensemble_members: list[str] = None, variables: dict = None):
        self.ensemble_members = ensemble_members or [f'gep{i:02d}' for i in range(1, 31)]
        self.variables = variables or {'tp': 'APCP:surface'}
        register_codec("gefs_gribberish", GEFSGribberishCodec)

    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        """
        Parse GEFS ensemble data to ManifestStore.

        url: Base URL pattern, e.g., "s3://noaa-gefs-pds/gefs.20250106/00/atmos/pgrb2sp25/"
        """
        # Step 1: Scan all ensemble member files
        member_refs = self._scan_ensemble_members(url, registry)

        # Step 2: Create ManifestArrays with ChunkManifest pointing to GRIB bytes
        arrays = {}
        for varname, var_refs in member_refs.items():
            manifest_array = self._create_manifest_array(varname, var_refs)
            arrays[varname] = manifest_array

        # Step 3: Add coordinate arrays
        arrays.update(self._create_coordinate_arrays(member_refs))

        # Step 4: Return ManifestStore
        return ManifestStore(registry=registry, group=ManifestGroup(arrays=arrays))

    def _create_manifest_array(self, varname, refs) -> ManifestArray:
        """Create ManifestArray with gribberish codec."""
        entries = {}
        for chunk_key, (path, offset, length) in refs.items():
            entries[ChunkKey(chunk_key)] = ChunkEntry.with_validation(
                path=path, offset=offset, length=length
            )

        codec = GEFSGribberishCodec(var=varname).to_dict()
        metadata = create_v3_array_metadata(
            shape=self._get_shape(refs),
            chunk_shape=self._get_chunk_shape(),
            data_type=np.dtype('float32'),
            codecs=[codec],
            dimension_names=['member', 'time', 'latitude', 'longitude']
        )

        return ManifestArray(metadata=metadata, chunkmanifest=ChunkManifest(entries=entries))
```

### 1.2 GEFSGribberishCodec Implementation

```python
# Target: gefsparser/codecs/gefs_gribberish.py
from dataclasses import dataclass
from zarr.abc.codec import ArrayBytesCodec
from gribberish import parse_grib_array, parse_grib_message_metadata

CODEC_ID = "gefs_gribberish"

@dataclass(frozen=True)
class GEFSGribberishCodec(ArrayBytesCodec):
    """Zarr v3 codec for GEFS GRIB data using gribberish."""

    var: str | None = None
    grid_shape: tuple = (721, 1440)  # GEFS 0.25° global

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.to_bytes()

        if self.var == "latitude":
            meta = parse_grib_message_metadata(chunk_bytes, 0)
            lat, _ = meta.latlng()
            data = lat
        elif self.var == "longitude":
            meta = parse_grib_message_metadata(chunk_bytes, 0)
            _, lng = meta.latlng()
            # GEFS uses 0-360, convert to -180 to 180 if needed
            data = np.where(lng > 180, lng - 360, lng)
        elif self.var == "time":
            meta = parse_grib_message_metadata(chunk_bytes, 0)
            data = np.datetime64(meta.reference_date, "s")
        else:
            # Data variable - use gribberish for fast decode
            flat_array = parse_grib_array(chunk_bytes, 0)
            data = flat_array.reshape(self.grid_shape)

        return data.astype(chunk_spec.dtype.to_native_dtype()).reshape(chunk_spec.shape)
```

### 1.3 Writing to Icechunk

```python
# Target: Create persistent Icechunk store from ManifestStore
import icechunk
from virtualizarr import open_virtual_dataset

# Configure Icechunk with virtual chunk container for S3 access
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        "s3://noaa-gefs-pds/",  # Base path for GEFS data
        icechunk.s3_store(
            region="us-east-1",
            anonymous=True  # NOAA open data
        )
    )
)

# Create repository
storage = icechunk.s3_storage(
    bucket="my-icechunk-bucket",
    prefix="gefs/20250106/",
    region="us-east-1"
)
repo = icechunk.Repository.create(storage, config=config)

# Parse GEFS data to virtual dataset
parser = GEFSParser(ensemble_members=[f'gep{i:02d}' for i in range(1, 31)])
registry = ObjectStoreRegistry.from_prefix("s3://noaa-gefs-pds/")
manifest_store = parser("s3://noaa-gefs-pds/gefs.20250106/00/atmos/pgrb2sp25/", registry)

# Convert to xarray and write to Icechunk
vds = xr.open_dataset(manifest_store, engine="zarr", zarr_format=3)
session = repo.writable_session("main")
vds.virtualizarr.to_icechunk(session.store)
session.commit("Added GEFS 20250106 00z virtual references")
```

---

## Migration Phase 2: Icechunk-Native Data Streaming with Dask

### 2.1 Reading from Icechunk (Replaces Manual Parquet Reading)

**Current** (`run_gefs_data_streaming_v2.py`):
```python
# Manual: pandas → dict → fsspec → gribberish
df = pd.read_parquet(parquet_path)
zstore = {}
for _, row in df.iterrows():
    zstore[key] = json.loads(row['value'])
grib_bytes = fetch_grib_bytes(zstore[chunk_key], fs)
array_2d = gribberish.parse_grib_array(grib_bytes, 0)
```

**Target** (Icechunk + xarray):
```python
import icechunk
import xarray as xr
import dask.array as da

# Open Icechunk repository with credentials for virtual chunks
credentials = icechunk.containers_credentials({
    "s3://noaa-gefs-pds/": icechunk.s3_credentials(anonymous=True)
})

repo = icechunk.Repository.open(
    icechunk.s3_storage(bucket="my-bucket", prefix="gefs/20250106/"),
    virtual_chunk_credentials=credentials
)

# Open as xarray Dataset - codec handles GRIB decoding automatically
session = repo.readonly_session()
ds = xr.open_zarr(session.store, zarr_format=3)

# Lazy loading - data not fetched until needed
precip = ds['tp']  # ManifestArray with GEFSGribberishCodec
print(precip.shape)  # (30, 80, 721, 1440) - members, time, lat, lon

# Regional subsetting - only fetches needed chunks
ea_precip = precip.sel(
    latitude=slice(23, -12),  # East Africa
    longitude=slice(21, 53)
)

# Load to memory (triggers codec decode)
ea_data = ea_precip.values  # GEFSGribberishCodec._decode_single() called
```

### 2.2 Dask Integration for Parallel Operations

```python
import dask
from dask.distributed import Client

# Start Dask cluster
client = Client(n_workers=4)

# Open Icechunk as Dask-backed xarray Dataset
ds = xr.open_zarr(session.store, zarr_format=3, chunks={'member': 1, 'time': 10})

# Lazy computation graph
precip_24h = ds['tp'].diff('time', n=8)  # 24h accumulation (8 x 3h steps)
exceedance_prob = (precip_24h > 50).mean('member') * 100

# Parallel computation - Dask handles scheduling
# Each worker calls GEFSGribberishCodec for its chunks
result = exceedance_prob.compute()

# Or persist intermediate results
daily_precip = precip_24h.persist()  # Computed in parallel, held in memory
```

### 2.3 Comparison: Current vs Target Architecture

| Aspect | Current (Parquet-based) | Target (Icechunk/VirtualiZarr) |
|--------|-------------------------|--------------------------------|
| **Reference Storage** | Parquet DataFrame | Icechunk ManifestStore |
| **Metadata Handling** | Manual JSON parsing | Zarr v3 ArrayMetadata |
| **Chunk Assembly** | Manual numpy loops | Zarr automatic |
| **GRIB Decoding** | Explicit gribberish calls | Codec invoked by Zarr |
| **Parallel Fetching** | ThreadPoolExecutor | Dask distributed |
| **Regional Subsetting** | Manual index calculation | xarray `.sel()` |
| **Memory Management** | Temporary zarr on disk | Lazy Dask arrays |
| **Code Lines** | ~800 lines | ~100 lines |

---

## Implementation Roadmap

### Phase 1: Core Parser Development (Weeks 1-3)

```
Week 1: GEFSParser skeleton
├── Implement Parser protocol (__call__ signature)
├── Port _scan_messages from HRRRparser pattern
└── Create GEFSGribberishCodec with GEFS-specific handling

Week 2: ECMWFParser skeleton
├── Handle ensemble member encoding (control + ens01-ens50)
├── Implement ECMWF coordinate conventions (-180 to 180)
└── Create ECMWFGribberishCodec

Week 3: Template integration
├── Port LocalTarGzMappingManager pattern for pre-built templates
├── Implement three-stage pipeline within parser
└── Unit tests for codec decoding
```

### Phase 2: Icechunk Integration (Weeks 4-5)

```
Week 4: ManifestStore → Icechunk
├── Configure VirtualChunkContainer for NOAA S3
├── Configure VirtualChunkContainer for ECMWF S3
├── Implement to_icechunk() workflow
└── Test round-trip: parse → write → read

Week 5: Credentials and persistence
├── Handle anonymous S3 access for NOAA
├── Handle anonymous S3 access for ECMWF
├── Implement versioning for daily forecast updates
└── Performance benchmarking
```

### Phase 3: Dask-Native Streaming (Weeks 6-7)

```
Week 6: Replace manual streaming
├── Port exceedance probability calculation to xarray
├── Implement Dask chunking strategy
├── Test parallel performance vs current ThreadPoolExecutor

Week 7: Production integration
├── Create run_gefs_icechunk.py example
├── Create run_ecmwf_icechunk.py example
├── Documentation and tutorials
└── Package as gefsparser/ecmwfparser Python packages
```

---

## Key Differences: GEFS vs ECMWF Parsers

| Aspect | GEFSParser | ECMWFParser |
|--------|-----------|-------------|
| **S3 Source** | `s3://noaa-gefs-pds/` | `s3://ecmwf-forecasts/` |
| **File Structure** | Separate file per member | All members in single file |
| **Longitude Convention** | 0 to 360 | -180 to 180 |
| **Index Format** | `.idx` (text) | `.index` (JSON) |
| **Ensemble Members** | 30 (gep01-gep30) | 51 (control + ens01-ens50) |
| **Precipitation Variable** | `APCP:surface` | `tp/sfc/{member}` |
| **S3 Latency** | Fast (~0.5s/fetch) | Slower (~2.5s/fetch) |

---

## External References

### VirtualiZarr
- Custom Parser Guide: https://virtualizarr.readthedocs.io/en/stable/custom_parsers.html
- Parser Protocol: `virtualizarr.parsers.typing.Parser`

### Icechunk
- Virtual Datasets: https://icechunk.io/en/latest/virtual/
- VirtualChunkContainer API
- Credentials handling for S3

### Prior Art
- **HRRRparser**: `/icpac-gik/hrrr-parser/` - Reference implementation
- **MeteoSwiss icon-ch-vzarr**: https://github.com/MeteoSwiss/icon-ch-vzarr - GRIB VirtualiZarr proof-of-concept

---

## Benefits of Migration

### For Developers
1. **Reduced Code**: ~800 lines → ~100 lines for data streaming
2. **Standardized APIs**: Uses community-maintained VirtualiZarr/Icechunk
3. **Type Safety**: Zarr v3 metadata validation
4. **Testability**: Codec can be unit tested in isolation

### For Users
1. **Simple Access**: `xr.open_zarr(icechunk_store)` - just works
2. **Lazy Loading**: Only fetch data you need
3. **Dask Integration**: Automatic parallel processing
4. **Versioning**: Icechunk tracks forecast updates

### For Operations
1. **Performance**: 5-10x faster via gribberish codec
2. **Memory Efficient**: No intermediate files needed
3. **Scalable**: Dask handles large ensemble processing
4. **Auditable**: Icechunk commit history tracks changes

---

## Conclusion

The migration from parquet-based GIK workflow to VirtualiZarr/Icechunk plugins requires:

1. **Parser Development**: Implement `GEFSParser` and `ECMWFParser` following the HRRRparser pattern
2. **Codec Implementation**: Create product-specific gribberish codecs handling coordinate conventions
3. **Icechunk Persistence**: Configure VirtualChunkContainer for S3 virtual references
4. **Data Streaming Refactor**: Replace manual loops with xarray/Dask native operations

This transforms operational necessity (GEFS/ECMWF data access) into reusable community infrastructure following the successful HRRRparser model.
