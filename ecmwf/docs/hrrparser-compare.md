 ---
  Comparison: hrrr-parser Virtual Zarr vs Simple Byte-Range Approach

  Overview

  | Aspect        | hrrr-parser (Virtual Zarr)        | test_gribberish_netcdf.py (Simple) |
  |---------------|-----------------------------------|------------------------------------|
  | GRIB Decoder  | gribberish                        | cfgrib                             |
  | Data Access   | Lazy (on-demand)                  | Eager (materialize all)            |
  | Output Format | Virtual Zarr (ManifestStore)      | NetCDF file                        |
  | Memory Model  | Stream chunks as needed           | Load all into memory               |
  | Complexity    | High (custom codec, VirtualiZarr) | Low (sequential loop)              |

  ---
  Architecture Comparison

  1. hrrr-parser: Virtual Zarr with gribberish

  Flow:
  GRIB file → scan byte offsets → build ChunkManifest → ManifestStore → xarray (lazy)
                                         ↓
                                HRRRGribberishCodec (decodes on access)

  Key components:

  A. Message Scanning (parser.py:71-113)
  def _scan_messages(filepath: str, reader: ObstoreReader) -> dict[str, dict]:
      levels: dict[str, dict] = {}
      for offset, size, data in _split_file(reader):
          chunk_entry = ChunkEntry.with_validation(
              path=filepath, offset=offset, length=size,
          )
          dataset = parse_grib_dataset(data, encode_coords=False)
          # Organize by level coordinate and variable name
          levels[level_coord]["variables"][var_name] = VarInfo(...)
  - Scans GRIB file to find all messages
  - Creates ChunkEntry objects with byte offset/length
  - No data decoding during scan - only metadata extraction

  B. Custom Zarr Codec (codecs/hrrr_gribberish.py:42-120)
  @dataclass(frozen=True)
  class HRRRGribberishCodec(ArrayBytesCodec):
      var: str | None
      steps: int = 1

      async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec):
          chunk_bytes = chunk_data.to_bytes()

          if self.var == "latitude" or self.var == "longitude":
              message = parse_grib_message_metadata(chunk_bytes, 0)
              lat, lng = message.latlng()
              data = lat if self.var == "latitude" else lng
          elif self.var in LEVEL_COORDINATES:
              message = parse_grib_message_metadata(chunk_bytes, 0)
              data = np.float64(message.level_value)
          else:
              data = parse_grib_array(chunk_bytes, 0)  # Actual GRIB decompression
          return data
  - Implements Zarr v3 ArrayBytesCodec interface
  - Decodes GRIB bytes on-demand when data is accessed
  - Uses gribberish functions:
    - parse_grib_message_metadata() - extract coords/metadata
    - parse_grib_array() - decompress data array

  C. ManifestArray Creation (parser.py:168-221)
  def _create_variable_array(varname, varinfo, coord_values, steps):
      entries: dict[ChunkKey, ChunkEntry] = {}
      for idx, coord_value in enumerate(coord_values):
          for step_idx, step_value in enumerate(step_array):
              key = f"0.{step_idx}.{idx}.0.0"
              entries[ChunkKey(key)] = varinfo.chunk_entries[coord_value]

      chunk_manifest = ChunkManifest(entries=entries)
      return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
  - Maps logical chunk indices to physical byte ranges
  - No data stored - just references

  ---
  2. test_gribberish_netcdf.py: Simple Byte-Range Approach

  Flow:
  Parquet refs → fetch byte ranges → cfgrib decode → stack in memory → write NetCDF

  Key components:

  A. Parquet Reference Reading (lines 27-42)
  def read_parquet_refs(parquet_path):
      df = pd.read_parquet(parquet_path)
      zstore = {}
      for _, row in df.iterrows():
          key = row['key']
          value = row['value']  # Contains [url, offset, length]
          zstore[key] = value
      return zstore

  B. Byte-Range Fetching (lines 75-89)
  def fetch_grib_bytes(zstore, chunk_key, fs):
      ref = zstore[chunk_key]
      url, offset, length = ref[0], ref[1], ref[2]
      s3_path = f's3://{url}'

      with fs.open(s3_path, 'rb') as f:
          f.seek(offset)
          grib_bytes = f.read(length)
      return grib_bytes

  C. cfgrib Decoding (lines 58-72)
  def decode_with_cfgrib(grib_bytes):
      with tempfile.NamedTemporaryFile(suffix='.grib2') as tmp:
          tmp.write(grib_bytes)
          tmp_path = tmp.name

      ds = xr.open_dataset(tmp_path, engine='cfgrib')
      array_2d = ds[var_name].values
      return array_2d

  D. Sequential Processing (lines 119-134)
  timestep_data = []
  for step, chunk_key in tp_chunks:
      grib_bytes = fetch_grib_bytes(zstore, chunk_key, fs)
      array_2d = decode_with_cfgrib(grib_bytes)
      timestep_data.append(array_2d)

  data_3d = np.stack(timestep_data, axis=0)  # All in memory

  ---
  Key Differences

  | Feature             | hrrr-parser                  | Simple Approach        |
  |---------------------|------------------------------|------------------------|
  | Reference Storage   | VirtualiZarr ChunkManifest   | Parquet DataFrame      |
  | Decode Library      | gribberish (Rust-based)      | cfgrib (eccodes-based) |
  | When Data Loads     | On .values access            | In processing loop     |
  | Multi-variable      | All vars in one dataset      | Single variable (tp)   |
  | Coordinate Handling | Extracted from GRIB metadata | Hardcoded (linspace)   |
  | Dimensionality      | 5D (time, step, level, y, x) | 3D (step, lat, lon)    |
  | Output              | In-memory virtual store      | NetCDF file            |

  ---
  gribberish vs cfgrib Usage

  gribberish (hrrr-parser):
  from gribberish import parse_grib_dataset, parse_grib_message_metadata, parse_grib_array

  # Metadata extraction
  message = parse_grib_message_metadata(chunk_bytes, 0)
  lat, lng = message.latlng()
  level_value = message.level_value
  reference_date = message.reference_date

  # Data decoding
  data = parse_grib_array(chunk_bytes, 0)
  - Direct Python bindings to Rust library
  - Fast metadata extraction without full decode
  - Fine-grained control over what to extract

  cfgrib (simple approach):
  ds = xr.open_dataset(tmp_path, engine='cfgrib')
  array_2d = ds[var_name].values
  - Requires writing to temp file first
  - xarray integration built-in
  - Less control, simpler API

  ---
  Advantages/Trade-offs

  hrrr-parser Virtual Zarr:
  - ✅ Lazy loading - only fetch what you need
  - ✅ Memory efficient for large datasets
  - ✅ Multi-dimensional organization
  - ✅ Compatible with Zarr ecosystem (Icechunk, xarray)
  - ✅ Accurate coordinates from GRIB metadata
  - ❌ Complex implementation
  - ❌ Requires VirtualiZarr + custom codec
  - ❌ Read-only (no encoding support)

  Simple Byte-Range Approach:
  - ✅ Simple, understandable code
  - ✅ Works with existing tools (cfgrib)
  - ✅ Direct to NetCDF output
  - ❌ Loads all data into memory
  - ❌ Sequential processing (slow for many chunks)
  - ❌ Hardcoded coordinates (may be inaccurate)
  - ❌ Single variable focus

  ---
  Integration Points

  hrrr-parser requires:
  - gribberish>=0.23.0
  - virtualizarr>=2.1.2
  - obstore>=0.6.0
  - Python 3.11+

  Simple approach requires:
  - pandas
  - fsspec
  - xarray + cfgrib
  - eccodes (C library)

  The hrrr-parser represents a more sophisticated approach suited for production systems needing
  lazy access to large forecast datasets, while the simple approach is more appropriate for one-off
  data extraction tasks where simplicity matters more than efficiency.

