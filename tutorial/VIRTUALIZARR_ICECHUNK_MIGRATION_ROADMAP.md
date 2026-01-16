# VirtualiZarr Custom Parser Migration Roadmap for GEFS and ECMWF

## Executive Summary

This document outlines the migration path to transform the current parquet-based Grib-Index-Kerchunk (GIK) workflow into VirtualiZarr **custom parsers** with Icechunk persistence, following the HRRRparser architecture pattern and insights from VirtualiZarr GitHub Issue #312.

**Key Terminology**: VirtualiZarr uses "custom parser" (not "plugin"). A parser is a callable implementing the `virtualizarr.parsers.typing.Parser` protocol.

**Critical Innovation**: The GIK method avoids scanning petabytes of GRIB files by reading lightweight text index files (.idx/.index) containing byte offsets. This enables reference file creation with **<5% of original GRIB data read**.

---

## The Grib-Index-Kerchunk (GIK) Method: Core Innovation

### The Problem: Petabyte-Scale GRIB Archives

Traditional approaches to creating Zarr virtual references from GRIB files require scanning every message in every file:

```
Traditional scan_grib approach:
├── GEFS: 2,400 files per run (30 members × 80 hours) × ~200MB = 480 GB scanned
├── ECMWF: 85 files per run × ~4GB × 51 members = ~17 TB data accessed
└── Total: Prohibitively expensive at petabyte scale
```

### The Solution: Index-Based Reference Creation

The GIK method leverages a critical insight: **GRIB index files (.idx/.index) contain all byte offset information needed for ChunkEntry creation without reading the actual GRIB data.**

```
GIK Method (Index-Based):
┌────────────────────────────────────────────────────────────────────────────┐
│                     <5% GRIB DATA READ                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Step 0 (ONE-TIME): Scan 1-2 sample GRIB files                            │
│  ├── Extract metadata structure (variable names, levels, grid shape)      │
│  ├── Build template parquet mapping                                        │
│  └── Store in GCS: gs://bucket/{product}/{member}/*.parquet               │
│                                                                            │
│  Step 1 (DAILY): Read index files only (~KB each vs GB GRIB files)        │
│  ├── GEFS: Parse .idx text files → byte_offset, byte_length, uri          │
│  ├── ECMWF: Parse .index JSON files → _offset, _length, member            │
│  └── Combine with template structure → Updated parquet references         │
│                                                                            │
│  Result: Reference files with S3 [url, offset, length] triplets           │
│          created WITHOUT reading the actual GRIB binary data!             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Analogy: Video Streaming vs Download

| **Aspect**              | **Video Streaming (HTML5)**                        | **Weather Data (GIK Method)**                      |
|-------------------------|----------------------------------------------------|----------------------------------------------------|
| **Download Workflow**   | Full video download for playback                   | Full GRIB file download for analysis               |
| **Streaming Workflow**  | Stream segments on demand using index              | Stream slices on demand using Kerchunk metadata    |
| **Metadata Handling**   | Indexed file for frames, timecodes                 | Indexed metadata for variables, timestamps, region |
| **Efficiency**          | Lower bandwidth; no full downloads needed          | **<5% GRIB data read** vs 100% traditional         |
| **Scalability**         | Scales easily across devices                       | Scales horizontally using Dask cluster             |

### Performance Comparison

| Method | Per Timestep | GEFS 30 Members | ECMWF 51 Members | Data Downloaded |
|--------|--------------|-----------------|------------------|-----------------|
| **scan_grib** (traditional) | 45-90 sec | 60-120 min | 64-120 hours | 100% GRIB files |
| **GIK (Index-based)** | 0.5-1 sec | 8-10 min | 30-40 min | **<5% GRIB data** |
| **Speedup** | **~85x** | **~10x** | **~100x** | **~6000x less I/O** |

### Index File Formats

**GEFS (.idx text format)**:
```
1:0:d=2025010600:PRES:surface:anl:
2:73452:d=2025010600:PRMSL:mean sea level:anl:
3:196623:d=2025010600:UGRD:10 m above ground:anl:
```
→ Fields: message_number, byte_offset, date, variable, level, forecast_type

**ECMWF (.index JSON format)**:
```json
{"_offset": 0, "_length": 1045678, "param": "tp", "levtype": "sfc", "number": 1, "step": "6"}
{"_offset": 1045678, "_length": 1045678, "param": "tp", "levtype": "sfc", "number": 2, "step": "6"}
```
→ Fields: byte_offset, byte_length, variable, level, ensemble_member, step

---

## Parquet to Icechunk: Bridge Path for Existing GIK Output

### Existing Parquet Format (GIK Output)

The GIK operational workflow produces parquet files containing `[key, value]` pairs:

```python
# Current parquet structure from GIK
# key: Zarr path (e.g., "tp/instant/surface/tp/0.0.0")
# value: Either JSON metadata OR S3 reference triplet

Sample parquet contents:
┌─────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│ key                                     │ value                                                │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ .zgroup                                 │ {"zarr_format": 2}                                   │
│ tp/instant/surface/.zarray              │ {"shape": [85, 721, 1440], "chunks": [1, 721, 1440]} │
│ tp/instant/surface/.zattrs              │ {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"], ...}   │
│ tp/instant/surface/tp/0.0.0             │ ["s3://noaa-gefs-pds/gefs...grib2", 73452, 1045678]  │
│ tp/instant/surface/tp/1.0.0             │ ["s3://noaa-gefs-pds/gefs...grib2", 1119130, 1045678]│
└─────────────────────────────────────────┴──────────────────────────────────────────────────────┘
```

### Direct Conversion: Parquet → ChunkManifest → Icechunk

The S3 reference triplets `[url, offset, length]` map directly to VirtualiZarr's `ChunkEntry`:

```python
# parquet_to_icechunk.py - Convert existing GIK parquet to Icechunk

import pandas as pd
import json
from virtualizarr.manifests import ChunkEntry, ChunkManifest, ManifestArray, ManifestStore, ManifestGroup
from zarr.core.metadata import ArrayV3Metadata
import icechunk

def parquet_to_manifest_store(parquet_path: str) -> ManifestStore:
    """
    Convert existing GIK parquet file to VirtualiZarr ManifestStore.

    This provides a migration path for operational parquet files
    without re-running the GIK pipeline.
    """
    df = pd.read_parquet(parquet_path)

    # Separate metadata from chunk references
    metadata_entries = {}
    chunk_refs = {}

    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        # Check if it's a chunk reference (S3 triplet) or metadata (JSON)
        if key.endswith('.zarray') or key.endswith('.zattrs') or key.endswith('.zgroup'):
            metadata_entries[key] = json.loads(value) if isinstance(value, str) else value
        else:
            # Chunk reference: ["s3://bucket/path", offset, length]
            try:
                ref = json.loads(value) if isinstance(value, str) else value
                if isinstance(ref, list) and len(ref) == 3:
                    path, offset, length = ref
                    chunk_refs[key] = ChunkEntry.with_validation(
                        path=path, offset=offset, length=length
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    # Build ManifestArrays from chunk references
    arrays = {}
    for key, entry in chunk_refs.items():
        # Extract array path from chunk key (e.g., "tp/instant/surface/tp/0.0.0" → "tp/instant/surface")
        parts = key.rsplit('/', 2)
        if len(parts) >= 2:
            array_path = '/'.join(parts[:-1])
            chunk_key = parts[-1]

            if array_path not in arrays:
                arrays[array_path] = {'entries': {}, 'metadata': None}
            arrays[array_path]['entries'][chunk_key] = entry

    # Create ManifestArrays with gribberish codec
    manifest_arrays = {}
    for array_path, data in arrays.items():
        zarray_key = f"{array_path}/.zarray"
        if zarray_key in metadata_entries:
            zarray = metadata_entries[zarray_key]

            # Convert Zarr v2 metadata to v3 with gribberish codec
            metadata = ArrayV3Metadata(
                shape=tuple(zarray['shape']),
                chunk_shape=tuple(zarray['chunks']),
                data_type=np.dtype(zarray['dtype']),
                codecs=[GEFSGribberishCodec().to_dict()],  # Add gribberish codec
                dimension_names=metadata_entries.get(f"{array_path}/.zattrs", {}).get('_ARRAY_DIMENSIONS', [])
            )

            manifest_arrays[array_path] = ManifestArray(
                metadata=metadata,
                chunkmanifest=ChunkManifest(entries=data['entries'])
            )

    return ManifestStore(group=ManifestGroup(arrays=manifest_arrays))


def parquet_to_icechunk(parquet_path: str, icechunk_storage, s3_base_url: str):
    """
    Convert GIK parquet directly to Icechunk store.

    Parameters
    ----------
    parquet_path : str
        Path to existing GIK parquet file
    icechunk_storage : icechunk storage config
        Where to store the Icechunk repository
    s3_base_url : str
        Base URL for virtual chunk container (e.g., "s3://noaa-gefs-pds/")
    """
    # Configure Icechunk for virtual chunks
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            s3_base_url,
            icechunk.s3_store(region="us-east-1", anonymous=True)
        )
    )

    # Create repository
    repo = icechunk.Repository.create(icechunk_storage, config=config)

    # Convert parquet to ManifestStore
    manifest_store = parquet_to_manifest_store(parquet_path)

    # Write to Icechunk
    import xarray as xr
    vds = xr.open_dataset(manifest_store, engine="zarr", zarr_format=3)

    session = repo.writable_session("main")
    vds.virtualizarr.to_icechunk(session.store)
    session.commit(f"Imported from GIK parquet: {parquet_path}")

    return repo


# Example usage for operational parquet files
if __name__ == "__main__":
    # Convert GEFS parquet from GIK pipeline
    gefs_parquet = "gefs_20250106_00z_gep01.parquet"
    gefs_storage = icechunk.local_storage("./icechunk/gefs/20250106/")
    repo = parquet_to_icechunk(
        gefs_parquet,
        gefs_storage,
        s3_base_url="s3://noaa-gefs-pds/"
    )

    # Convert ECMWF parquet
    ecmwf_parquet = "ecmwf_20250106_00z_control.parquet"
    ecmwf_storage = icechunk.local_storage("./icechunk/ecmwf/20250106/")
    repo = parquet_to_icechunk(
        ecmwf_parquet,
        ecmwf_storage,
        s3_base_url="s3://ecmwf-forecasts/"
    )
```

### Migration Path Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GIK → VirtualiZarr/Icechunk Migration                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OPTION A: Preserve Existing Parquet (Immediate)                           │
│  ──────────────────────────────────────────────────                        │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐           │
│  │ GIK Parquet  │ → │ parquet_to_       │ → │ Icechunk Store  │           │
│  │ (existing)   │    │ manifest_store() │    │ (VirtualiZarr)  │           │
│  └──────────────┘    └──────────────────┘    └─────────────────┘           │
│                                                                             │
│  Benefits: No re-processing, immediate migration of operational data       │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────         │
│                                                                             │
│  OPTION B: Custom Parser with GIK Method (Future)                          │
│  ────────────────────────────────────────────────                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐           │
│  │ Index Files  │ → │ GEFSParser/      │ → │ ManifestStore   │ → Icechunk │
│  │ (.idx/.index)│    │ ECMWFParser      │    │ (direct)        │           │
│  └──────────────┘    └──────────────────┘    └─────────────────┘           │
│                                                                             │
│  Benefits: Native VirtualiZarr, no intermediate parquet, <5% GRIB read    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Background: VirtualiZarr Issue #312 Discussion

### Problem Statement
Kerchunk's GRIB reader forces inlining of derived coordinates that are not stored at the message level. VirtualiZarr does not support reading inlined refs. This necessitates a VirtualiZarr-specific approach without direct kerchunk dependency.

### Key Insights from @mpiannucci
- **gribberish maintenance**: Actively maintained across three production applications; suitable for VirtualiZarr integration
- **Recommended approach**: Start with cfgrib wrapper for maximum compliance, or leverage GRIB codec to generate coordinates dynamically
- **Critical insight**: Every GRIB message contains metadata sufficient for coordinate generation—no need for inline storage

### Alternative Approaches
- **earthkit-data**: Creates GribField objects for each message; can extract byte ranges for ChunkEntry construction
- **MeteoSwiss icon-ch-vzarr**: Proof-of-concept demonstrating basic GRIB VirtualiZarr parser functionality

### Coordinate Alignment Question
Can we assume coordinate alignment across all messages in a GRIB file?
- **If yes**: Use `open_virtual_datatree`
- **If no**: Implement `open_virtual_groups` method for problematic datasets

---

## VirtualiZarr Parser Protocol

### Definition (from docs)
> "A parser is a callable that accepts the URL pointing to a data source and an ObjectStoreRegistry" and returns a ManifestStore instance.

### Protocol Signature
```python
from virtualizarr.parsers.typing import Parser
from virtualizarr.manifests import ManifestStore
from icechunk import ObjectStoreRegistry

class GEFSParser:
    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        ...
```

### Implementation Steps (per VirtualiZarr docs)
1. **Validate format**: Extract file headers or magic bytes
2. **Extract metadata**: Array counts, shapes, chunk dimensions, codecs
3. **Process each array**:
   - Create `zarr.core.metadata.ArrayV3Metadata` objects
   - Extract byte ranges into `ChunkManifest`
   - Construct `ManifestArray` instances
4. **Group arrays**: Into `ManifestGroup` objects with optional group-level metadata
5. **Instantiate ManifestStore**: With the top-level group

### Data Model Requirements
- All arrays must have dimension names
- Arrays in the same group with common dimension name must have same length

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

## Target Architecture: VirtualiZarr Custom Parser Pattern

### Reference: HRRRparser Architecture

The HRRRparser (`/icpac-gik/hrrr-parser/`) provides the canonical pattern:

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

| Component | HRRRparser | GEFS Parser (needed) | ECMWF Parser (needed) |
|-----------|------------|----------------------|------------------------|
| Parser Protocol | `HRRRParser.__call__()` | `GEFSParser.__call__()` | `ECMWFParser.__call__()` |
| Codec Class | `HRRRGribberishCodec` | `GEFSGribberishCodec` | `ECMWFGribberishCodec` |
| Coordinate Handling | HRRR-specific (Lambert) | GEFS 0-360 longitude | ECMWF -180 to 180 |
| Ensemble Handling | Single file | 30 members (separate files) | 51 members (shared files) |
| Coordinate Generation | From GRIB metadata | From GRIB metadata | From GRIB metadata |

---

## Migration Phase 1: Parquet → ManifestStore/Icechunk

### 1.1 GEFS Custom Parser Implementation

**Goal**: Transform `run_gefs_tutorial.py` output to ManifestStore directly

```python
# Target: gefsparser/parser.py
from virtualizarr.manifests import ChunkEntry, ChunkManifest, ManifestArray, ManifestStore, ManifestGroup
from zarr.core.metadata import ArrayV3Metadata
from zarr.registry import register_codec
from icechunk import ObjectStoreRegistry

class GEFSParser:
    """VirtualiZarr custom parser for GEFS ensemble data.

    Implements the virtualizarr.parsers.typing.Parser protocol.
    """

    def __init__(self, ensemble_members: list[str] = None, variables: dict = None):
        self.ensemble_members = ensemble_members or [f'gep{i:02d}' for i in range(1, 31)]
        self.variables = variables or {'tp': 'APCP:surface'}
        register_codec("gefs_gribberish", GEFSGribberishCodec)

    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        """
        Parse GEFS ensemble data to ManifestStore.

        Parameters
        ----------
        url : str
            Base URL pattern, e.g., "s3://noaa-gefs-pds/gefs.20250106/00/atmos/pgrb2sp25/"
        registry : ObjectStoreRegistry
            Object store registry for accessing remote data

        Returns
        -------
        ManifestStore
            Virtual Zarr store with references to GRIB byte ranges
        """
        # Step 1: Scan all ensemble member files
        member_refs = self._scan_ensemble_members(url, registry)

        # Step 2: Create ManifestArrays with ChunkManifest pointing to GRIB bytes
        arrays = {}
        for varname, var_refs in member_refs.items():
            manifest_array = self._create_manifest_array(varname, var_refs)
            arrays[varname] = manifest_array

        # Step 3: Add coordinate arrays (generated from GRIB metadata, not inlined)
        arrays.update(self._create_coordinate_arrays(member_refs))

        # Step 4: Return ManifestStore
        return ManifestStore(registry=registry, group=ManifestGroup(arrays=arrays))

    def _create_manifest_array(self, varname: str, refs: dict) -> ManifestArray:
        """Create ManifestArray with gribberish codec.

        Coordinates are generated dynamically by the codec from GRIB message
        metadata, avoiding the kerchunk limitation of forced inline storage.
        """
        entries = {}
        for chunk_key, (path, offset, length) in refs.items():
            entries[ChunkKey(chunk_key)] = ChunkEntry.with_validation(
                path=path, offset=offset, length=length
            )

        codec = GEFSGribberishCodec(var=varname).to_dict()
        metadata = ArrayV3Metadata(
            shape=self._get_shape(refs),
            chunk_shape=self._get_chunk_shape(),
            data_type=np.dtype('float32'),
            codecs=[codec],
            dimension_names=['member', 'time', 'latitude', 'longitude']
        )

        return ManifestArray(metadata=metadata, chunkmanifest=ChunkManifest(entries=entries))
```

### 1.2 GEFS Gribberish Codec Implementation

```python
# Target: gefsparser/codecs/gefs_gribberish.py
from dataclasses import dataclass
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.array_spec import ArraySpec
from gribberish import parse_grib_array, parse_grib_message_metadata
import numpy as np

CODEC_ID = "gefs_gribberish"

@dataclass(frozen=True)
class GEFSGribberishCodec(ArrayBytesCodec):
    """Zarr v3 codec for GEFS GRIB data using gribberish.

    Key design choice: Coordinates are generated dynamically from GRIB message
    metadata rather than being inlined. This addresses the kerchunk limitation
    identified in VirtualiZarr Issue #312.
    """

    var: str | None = None
    grid_shape: tuple = (721, 1440)  # GEFS 0.25° global

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.to_bytes()

        if self.var == "latitude":
            # Generate latitude from GRIB message metadata
            meta = parse_grib_message_metadata(chunk_bytes, 0)
            lat, _ = meta.latlng()
            data = lat
        elif self.var == "longitude":
            # Generate longitude from GRIB message metadata
            meta = parse_grib_message_metadata(chunk_bytes, 0)
            _, lng = meta.latlng()
            # GEFS uses 0-360, convert to -180 to 180 if needed
            data = np.where(lng > 180, lng - 360, lng)
        elif self.var == "time":
            # Generate time from GRIB message metadata
            meta = parse_grib_message_metadata(chunk_bytes, 0)
            data = np.datetime64(meta.reference_date, "s")
        else:
            # Data variable - use gribberish for fast decode
            flat_array = parse_grib_array(chunk_bytes, 0)
            data = flat_array.reshape(self.grid_shape)

        return data.astype(chunk_spec.dtype.to_native_dtype()).reshape(chunk_spec.shape)

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        raise NotImplementedError("GRIB encoding not supported")
```

### 1.3 ECMWF Custom Parser Implementation

```python
# Target: ecmwfparser/parser.py
class ECMWFParser:
    """VirtualiZarr custom parser for ECMWF ensemble data.

    Key differences from GEFS:
    - All 51 ensemble members in single GRIB file
    - Longitude convention: -180 to 180
    - Uses JSON index format (.index) vs text (.idx)
    """

    def __init__(self, variables: dict = None):
        self.variables = variables or {'tp': 'tp/sfc'}
        # 51 members: control + ens01 to ens50
        self.ensemble_members = ['control'] + [f'ens{i:02d}' for i in range(1, 51)]
        register_codec("ecmwf_gribberish", ECMWFGribberishCodec)

    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        """Parse ECMWF ensemble data to ManifestStore."""
        # ECMWF stores all members in single file - different scanning approach
        member_refs = self._scan_ecmwf_file(url, registry)

        arrays = {}
        for varname, var_refs in member_refs.items():
            manifest_array = self._create_manifest_array(varname, var_refs)
            arrays[varname] = manifest_array

        arrays.update(self._create_coordinate_arrays(member_refs))

        return ManifestStore(registry=registry, group=ManifestGroup(arrays=arrays))
```

### 1.4 Writing to Icechunk

```python
# Target: Create persistent Icechunk store from ManifestStore
import icechunk
import xarray as xr

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

# Parse GEFS data to virtual dataset using custom parser
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

| Aspect | Current (Parquet-based) | Target (Custom Parser/Icechunk) |
|--------|-------------------------|----------------------------------|
| **Reference Storage** | Parquet DataFrame | Icechunk ManifestStore |
| **Metadata Handling** | Manual JSON parsing | Zarr v3 ArrayMetadata |
| **Chunk Assembly** | Manual numpy loops | Zarr automatic |
| **GRIB Decoding** | Explicit gribberish calls | Codec invoked by Zarr |
| **Coordinate Generation** | Inline storage (kerchunk) | Dynamic from GRIB metadata |
| **Parallel Fetching** | ThreadPoolExecutor | Dask distributed |
| **Regional Subsetting** | Manual index calculation | xarray `.sel()` |
| **Memory Management** | Temporary zarr on disk | Lazy Dask arrays |
| **Code Lines** | ~800 lines | ~100 lines |

---

## Implementation Roadmap

### Phase 0: Immediate Migration (Bridge Path)

```
Task 0: Parquet → Icechunk Bridge (IMMEDIATE WIN)
├── Implement parquet_to_manifest_store() converter
├── Map existing [url, offset, length] triplets to ChunkEntry
├── Configure VirtualChunkContainer for NOAA/ECMWF S3
├── Convert operational parquet files to Icechunk stores
├── Validate: compare xr.open_zarr(icechunk) vs current fsspec approach
└── Document migration path for existing GIK users

Benefits:
- No re-processing of historical data
- Immediate access via xr.open_zarr()
- Preserves <5% GRIB read efficiency from GIK method
```

### Phase 1: Core Custom Parser Development

```
Task 1: GEFS Custom Parser with GIK Method
├── Implement Parser protocol (__call__ signature)
├── Port index file parsing (parse_grib_idx) - reads .idx text files
├── Create GEFSGribberishCodec with GEFS-specific handling
├── Dynamic coordinate generation (not inlined)
└── Integrate template reuse from GIK (avoid repeated GRIB scanning)

Task 2: ECMWF Custom Parser with GIK Method
├── Handle ensemble member encoding (control + ens01-ens50)
├── Port JSON index parsing (.index files) - <5% data read
├── Implement ECMWF coordinate conventions (-180 to 180)
├── Create ECMWFGribberishCodec
└── Single-file multi-member scanning with index lookup

Task 3: Template integration
├── Port LocalTarGzMappingManager pattern for pre-built templates
├── Implement template reuse across dates (ONE-TIME scan, DAILY index read)
├── Unit tests for codec decoding
└── Performance validation: confirm <5% GRIB data read maintained
```

### Phase 2: Icechunk Integration

```
Task 4: ManifestStore → Icechunk
├── Configure VirtualChunkContainer for NOAA S3 (anonymous)
├── Configure VirtualChunkContainer for ECMWF S3 (anonymous)
├── Implement to_icechunk() workflow from both:
│   ├── Direct parser output (new workflow)
│   └── Converted parquet (bridge path)
└── Test round-trip: parse → write → read

Task 5: Credentials and persistence
├── Handle anonymous S3 access for NOAA
├── Handle anonymous S3 access for ECMWF
├── Implement versioning for daily forecast updates
└── Performance benchmarking vs parquet-based approach
```

### Phase 3: Dask-Native Streaming

```
Task 6: Replace manual streaming
├── Port exceedance probability calculation to xarray
├── Implement Dask chunking strategy
├── Test parallel performance vs current ThreadPoolExecutor
└── Validate East Africa regional subsetting efficiency

Task 7: Production integration
├── Create run_gefs_icechunk.py example
├── Create run_ecmwf_icechunk.py example
├── Create parquet_migration.py for existing operational data
├── Documentation and tutorials
└── Package as gefsparser/ecmwfparser Python packages
```

---

## Key Differences: GEFS vs ECMWF Custom Parsers

| Aspect | GEFS Parser | ECMWF Parser |
|--------|-------------|--------------|
| **S3 Source** | `s3://noaa-gefs-pds/` | `s3://ecmwf-forecasts/` |
| **File Structure** | Separate file per member | All members in single file |
| **Longitude Convention** | 0 to 360 | -180 to 180 |
| **Index Format** | `.idx` (text) | `.index` (JSON) |
| **Ensemble Members** | 30 (gep01-gep30) | 51 (control + ens01-ens50) |
| **Precipitation Variable** | `APCP:surface` | `tp/sfc/{member}` |
| **S3 Latency** | Fast (~0.5s/fetch) | Slower (~2.5s/fetch) |

---

## Alternative Approaches (from Issue #312)

### Option 1: earthkit-data for ChunkEntry Extraction
```python
from earthkit.data import from_source

# earthkit-data creates GribField for each message
ds = from_source("file", "gefs.grib2")
for field in ds:
    # Extract ChunkEntry information
    offset = field.offset
    length = field.length
    path = field.path
    chunk_entry = ChunkEntry.with_validation(path=path, offset=offset, length=length)
```

### Option 2: cfgrib Wrapper (recommended by @mpiannucci)
Starting with cfgrib wrapper provides maximum compliance with GRIB conventions, then optimize with gribberish codec for performance.

### Option 3: Coordinate Alignment Check
```python
def open_virtual_dataset_or_groups(url, registry):
    """Handle both aligned and misaligned coordinate cases."""
    try:
        # Try aligned case first
        return open_virtual_datatree(url, registry)
    except CoordinateAlignmentError:
        # Fall back to groups for problematic datasets
        return open_virtual_groups(url, registry)
```

---

## External References

### VirtualiZarr
- **Custom Parser Guide**: https://virtualizarr.readthedocs.io/en/stable/custom_parsers.html
- **Parser Protocol**: `virtualizarr.parsers.typing.Parser`
- **Issue #312 Discussion**: https://github.com/zarr-developers/VirtualiZarr/issues/312

### Icechunk
- **Virtual Datasets**: https://icechunk.io/en/latest/virtual/
- **VirtualChunkContainer API**
- **Credentials handling for S3**

### Prior Art
- **HRRRparser**: `/icpac-gik/hrrr-parser/` - Reference implementation
- **MeteoSwiss icon-ch-vzarr**: https://github.com/MeteoSwiss/icon-ch-vzarr - GRIB VirtualiZarr proof-of-concept

### GRIB Libraries
- **gribberish**: Rust GRIB decoder, actively maintained by @mpiannucci
- **earthkit-data**: ECMWF toolkit for GribField extraction
- **cfgrib**: Reference implementation for GRIB conventions

---

## Benefits of Migration

### For Developers
1. **Reduced Code**: ~800 lines → ~100 lines for data streaming
2. **Standardized APIs**: Uses community-maintained VirtualiZarr/Icechunk
3. **Type Safety**: Zarr v3 metadata validation
4. **Testability**: Codec can be unit tested in isolation
5. **No Inline Coordinates**: Dynamic generation from GRIB metadata

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

The migration from parquet-based GIK workflow to VirtualiZarr custom parsers requires:

1. **Custom Parser Development**: Implement `GEFSParser` and `ECMWFParser` following the `virtualizarr.parsers.typing.Parser` protocol
2. **Codec Implementation**: Create product-specific gribberish codecs with dynamic coordinate generation (addressing kerchunk inline limitation)
3. **Icechunk Persistence**: Configure VirtualChunkContainer for S3 virtual references
4. **Data Streaming Refactor**: Replace manual loops with xarray/Dask native operations

This transforms operational necessity (GEFS/ECMWF data access) into reusable community infrastructure following the successful HRRRparser model and insights from VirtualiZarr Issue #312.

---

## Changelog

### V2.1 (2026-01-15)
- **NEW SECTION**: Added "The Grib-Index-Kerchunk (GIK) Method: Core Innovation"
  - Emphasized <5% GRIB data read vs 100% traditional scanning
  - Added index file format documentation (.idx text, .index JSON)
  - Included performance comparison table (85x speedup)
  - Added video streaming analogy for intuitive understanding
- **NEW SECTION**: Added "Parquet to Icechunk: Bridge Path"
  - Added `parquet_to_manifest_store()` converter code
  - Added `parquet_to_icechunk()` migration function
  - Documented two-option migration path diagram
- **UPDATED ROADMAP**: Added Phase 0 (Immediate Migration)
  - Parquet → Icechunk bridge for existing operational data
  - No re-processing required for historical parquet files
- **UPDATED TASKS**: Integrated GIK method into custom parser tasks
  - Task 1/2 now include index file parsing (not GRIB scanning)
  - Task 3 includes template reuse across dates

### V2 (2026-01-15)
- Updated terminology: "plugin" → "custom parser" per VirtualiZarr conventions
- Added context from GitHub Issue #312 discussion
- Incorporated @mpiannucci recommendations for coordinate generation
- Added earthkit-data as alternative approach
- Clarified coordinate alignment question (open_virtual_datatree vs open_virtual_groups)
- Referenced MeteoSwiss icon-ch-vzarr proof-of-concept
- Removed time estimates from roadmap (focus on tasks, not timelines)

### V1 (Initial)
- Original migration roadmap with VirtualiZarr plugin terminology
- Basic parser and codec implementation examples
- Icechunk integration workflow
