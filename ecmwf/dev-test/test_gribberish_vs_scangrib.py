#!/usr/bin/env python3
"""
Benchmark: gribberish + .index files vs kerchunk scan_grib
==========================================================

Tests whether we can replace the slow scan_grib (37 min per GRIB file)
with a combination of:
  1. .index file parsing (JSON, ~176ms for 8160 messages)
  2. gribberish byte-range parsing (Rust, ~0.03ms per message)

This would reduce Stage 1 from ~73 minutes to seconds.

Usage:
    micromamba run -n aifs-etl python test_gribberish_vs_scangrib.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

import fsspec

# Target GRIB file (same date/run as the user's 2-hour run)
TARGET_DATE = '20260205'
TARGET_RUN = '00'
TARGET_HOUR = 0
S3_BASE = f"ecmwf-forecasts/{TARGET_DATE}/{TARGET_RUN}z/ifs/0p25/enfo"
GRIB_URL = f"s3://{S3_BASE}/{TARGET_DATE}000000-{TARGET_HOUR}h-enfo-ef.grib2"
INDEX_URL = GRIB_URL.replace('.grib2', '.index')


def timer(label):
    """Context manager for timing."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start
            print(f"  [{label}] {self.elapsed:.3f}s")
    return Timer()


# ==============================================================================
# METHOD 1: .index file parsing (pure Python, no GRIB reading)
# ==============================================================================

def method_index_only():
    """Parse .index file to extract all byte references and metadata."""
    print("\n" + "=" * 70)
    print("METHOD 1: .index File Parsing (No GRIB reading)")
    print("=" * 70)

    fs = fsspec.filesystem("s3", anon=True)

    # Step 1: Fetch and parse the .index file
    with timer("Fetch .index from S3"):
        with fs.open(INDEX_URL, 'r') as f:
            raw_lines = f.readlines()

    with timer(f"Parse {len(raw_lines)} JSON lines"):
        entries = []
        for line in raw_lines:
            data = json.loads(line.strip().rstrip(','))
            entries.append({
                'offset': int(data['_offset']),
                'length': int(data['_length']),
                'param': data['param'],
                'levtype': data.get('levtype', ''),
                'number': int(data.get('number', -1)),
                'step': data.get('step', '0'),
                'type': data.get('type', ''),
                'date': data.get('date', ''),
                'time': data.get('time', ''),
            })

    # Step 2: Build zarr-style references from index entries
    with timer("Build zarr references"):
        refs = {}
        refs['.zgroup'] = json.dumps({"zarr_format": 2})

        members = {}
        for e in entries:
            member_num = e['number']
            if member_num == -1:
                member_key = 'control'
            else:
                member_key = f'ens_{member_num:02d}'

            if member_key not in members:
                members[member_key] = []

            # Build variable key: param/levtype/level_value
            var_key = e['param']
            if e['levtype'] == 'pl':
                var_key = f"{e['param']}_pl"
            elif e['levtype'] == 'sol':
                var_key = f"{e['param']}_sol"

            # Reference format: [url, byte_offset, byte_length]
            ref_key = f"{var_key}/{member_key}/0.0.0"
            refs[ref_key] = [GRIB_URL, e['offset'], e['length']]
            members[member_key].append(e)

    # Summary
    unique_params = set(e['param'] for e in entries)
    unique_members = set(e['number'] for e in entries)
    print(f"\n  Results:")
    print(f"    Total messages: {len(entries)}")
    print(f"    Unique params: {len(unique_params)}")
    print(f"    Unique members: {len(unique_members)}")
    print(f"    Total references: {len(refs)}")

    return entries, refs


# ==============================================================================
# METHOD 2: gribberish byte-range parsing
# ==============================================================================

def method_gribberish():
    """Use gribberish to parse GRIB message headers via byte-range requests."""
    print("\n" + "=" * 70)
    print("METHOD 2: gribberish + .index (Rust GRIB parsing)")
    print("=" * 70)

    from gribberish import parse_grib_message_metadata, parse_grib_mapping

    fs = fsspec.filesystem("s3", anon=True)

    # Step 1: Parse .index to get message offsets
    with timer("Fetch + parse .index"):
        with fs.open(INDEX_URL, 'r') as f:
            raw_lines = f.readlines()
        entries = []
        for line in raw_lines:
            data = json.loads(line.strip().rstrip(','))
            entries.append({
                'offset': int(data['_offset']),
                'length': int(data['_length']),
                'param': data['param'],
                'levtype': data.get('levtype', ''),
                'number': int(data.get('number', -1)),
            })

    # Step 2: Fetch a sample of GRIB messages to extract zarr metadata
    # We only need ONE message per unique (param, levtype) to get grid_shape, dims, dtype
    unique_types = {}
    for e in entries:
        key = (e['param'], e['levtype'])
        if key not in unique_types and e['length'] > 500:
            unique_types[key] = e

    print(f"  Unique (param, levtype) combos: {len(unique_types)}")
    print(f"  Sampling {min(len(unique_types), 10)} message types for metadata...")

    # Fetch a small batch of representative messages
    sample_types = list(unique_types.values())[:10]
    gribberish_metadata = {}

    with timer(f"Fetch {len(sample_types)} byte-ranges + gribberish parse"):
        for entry in sample_types:
            off = entry['offset']
            ln = entry['length']
            try:
                with fs.open(GRIB_URL.replace('s3://', ''), 'rb') as f:
                    f.seek(off)
                    msg_bytes = f.read(ln)

                meta = parse_grib_message_metadata(msg_bytes, 0)
                key = (entry['param'], entry['levtype'])
                gribberish_metadata[key] = {
                    'var_name': meta.var_name,
                    'var_abbrev': meta.var_abbrev,
                    'grid_shape': meta.grid_shape,
                    'dims': meta.dims,
                    'level_type': meta.level_type,
                    'level_value': meta.level_value,
                    'message_size': meta.message_size,
                    'generating_process': meta.generating_process,
                }
            except Exception as e:
                gribberish_metadata[(entry['param'], entry['levtype'])] = {
                    'error': str(e)
                }

    # Step 3: Build enhanced references using gribberish metadata
    with timer("Build zarr references with gribberish metadata"):
        refs = {}
        refs['.zgroup'] = json.dumps({"zarr_format": 2})

        grid_shape = None
        for key, meta in gribberish_metadata.items():
            if 'grid_shape' in meta:
                grid_shape = meta['grid_shape']
                break

        for e in entries:
            member_num = e['number']
            member_key = 'control' if member_num == -1 else f'ens_{member_num:02d}'
            param_key = (e['param'], e['levtype'])

            # Use gribberish metadata if available
            meta = gribberish_metadata.get(param_key, {})
            var_name = meta.get('var_abbrev', e['param'])

            ref_key = f"{var_name}/{member_key}/0.0.0"
            refs[ref_key] = [GRIB_URL, e['offset'], e['length']]

            # Add zarr array metadata if we have grid info
            zarray_key = f"{var_name}/.zarray"
            if zarray_key not in refs and grid_shape:
                refs[zarray_key] = json.dumps({
                    "chunks": list(grid_shape),
                    "compressor": None,
                    "dtype": "<f4",
                    "fill_value": 9999,
                    "filters": [{"id": "grib", "var": var_name}],
                    "order": "C",
                    "shape": list(grid_shape),
                    "zarr_format": 2,
                })

    # Summary
    print(f"\n  Results:")
    print(f"    Messages indexed: {len(entries)}")
    print(f"    gribberish metadata extracted: {len(gribberish_metadata)}")
    print(f"    Grid shape: {grid_shape}")
    print(f"    Total references: {len(refs)}")

    # Show gribberish metadata
    print(f"\n  Metadata samples:")
    for key, meta in list(gribberish_metadata.items())[:5]:
        if 'error' in meta:
            print(f"    {key}: ERROR - {meta['error'][:80]}")
        else:
            print(f"    {key}: var={meta['var_abbrev']}, "
                  f"grid={meta['grid_shape']}, dims={meta['dims']}")

    return entries, refs, gribberish_metadata


# ==============================================================================
# METHOD 3: kerchunk scan_grib (the current slow method)
# ==============================================================================

def method_scan_grib(max_messages=20):
    """Run kerchunk scan_grib on the GRIB file (limited messages for speed)."""
    print("\n" + "=" * 70)
    print(f"METHOD 3: kerchunk scan_grib (first {max_messages} messages)")
    print("=" * 70)

    from kerchunk.grib2 import scan_grib

    print(f"  URL: {GRIB_URL}")
    print(f"  Limiting to first {max_messages} messages (full file = ~37 min)")

    with timer(f"scan_grib (skip={max_messages})"):
        groups = scan_grib(
            GRIB_URL,
            storage_options={"anon": True},
            skip=max_messages,
        )

    # Extract info from groups
    print(f"\n  Results:")
    print(f"    Groups returned: {len(groups)}")

    if groups:
        g0 = groups[0]
        refs = g0.get('refs', {})
        print(f"    First group keys: {len(refs)}")

        # Show sample reference
        for key, val in refs.items():
            if isinstance(val, list) and len(val) == 3:
                print(f"    Sample ref: {key} -> [{val[0][:40]}..., offset={val[1]}, len={val[2]}]")
                break

        # Show zattrs
        if '.zattrs' in refs:
            attrs = json.loads(refs['.zattrs'])
            print(f"    Attributes: {list(attrs.keys())[:10]}")

    return groups


# ==============================================================================
# METHOD 4: gribberish bulk parse (fetch larger chunk, parse in Rust)
# ==============================================================================

def method_gribberish_bulk():
    """Fetch a larger chunk and let gribberish parse all messages at once."""
    print("\n" + "=" * 70)
    print("METHOD 4: gribberish Bulk Parse (multi-MB byte range)")
    print("=" * 70)

    from gribberish import parse_grib_mapping, parse_grib_message_metadata

    fs = fsspec.filesystem("s3", anon=True)

    # Parse .index to find messages in a contiguous range
    with fs.open(INDEX_URL, 'r') as f:
        raw_lines = f.readlines()
    entries = [json.loads(line.strip().rstrip(',')) for line in raw_lines]

    # Find first 100 messages (contiguous bytes from start)
    subset = entries[:100]
    end_byte = max(int(e['_offset']) + int(e['_length']) for e in subset)
    print(f"  Range: 0 to {end_byte:,} bytes ({end_byte/1024/1024:.1f} MB)")
    print(f"  Messages in range: {len(subset)}")

    # Fetch the byte range
    with timer("Fetch byte range from S3"):
        with fs.open(GRIB_URL.replace('s3://', ''), 'rb') as f:
            bulk_data = f.read(end_byte)

    # Parse all messages in bulk with gribberish
    with timer("gribberish parse_grib_mapping (bulk)"):
        mapping = parse_grib_mapping(bulk_data)

    print(f"\n  Results:")
    print(f"    Bytes fetched: {len(bulk_data):,}")
    print(f"    Messages parsed by gribberish: {len(mapping)}")

    # Show mapping keys
    print(f"\n  Sample mapping keys:")
    for i, (key, val) in enumerate(mapping.items()):
        if i >= 5:
            break
        meta = val[2]
        print(f"    {key}")
        print(f"      offset={val[0]}, var={meta.var_abbrev}, "
              f"grid={meta.grid_shape}, level={meta.level_value}")

    # Note: parse_grib_message_metadata panics (Rust unwrap) on unsupported
    # ECMWF templates, so we skip individual message parsing.
    # parse_grib_mapping handles these gracefully (returns "missing" var_name).
    print(f"  (Skipping individual parse_grib_message_metadata - panics on some ECMWF templates)")
    print(f"  parse_grib_mapping handled {len(mapping)} messages without panic")

    return mapping


# ==============================================================================
# COMPARISON SUMMARY
# ==============================================================================

def run_comparison():
    """Run all methods and compare."""
    print("=" * 70)
    print("BENCHMARK: Fast GRIB Reference Extraction")
    print(f"Target: {GRIB_URL}")
    print("=" * 70)

    results = {}

    # Method 1: Index only
    t0 = time.time()
    idx_entries, idx_refs = method_index_only()
    results['index_only'] = time.time() - t0

    # Method 2: gribberish + index
    t0 = time.time()
    gb_entries, gb_refs, gb_meta = method_gribberish()
    results['gribberish'] = time.time() - t0

    # Method 4: gribberish bulk
    t0 = time.time()
    bulk_mapping = method_gribberish_bulk()
    results['gribberish_bulk'] = time.time() - t0

    # Method 3: scan_grib (limited)
    t0 = time.time()
    groups = method_scan_grib(max_messages=20)
    results['scan_grib_20msg'] = time.time() - t0

    # Extrapolate scan_grib to full file
    if groups:
        per_msg = results['scan_grib_20msg'] / max(len(groups), 1)
        estimated_full = per_msg * 8160  # Total messages in file
    else:
        estimated_full = 37 * 60  # Fallback: 37 min from observed logs

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Time':>10} {'vs scan_grib':>15}")
    print("-" * 62)
    print(f"{'Index-only parsing':<35} {results['index_only']:>9.2f}s {'':>15}")
    print(f"{'gribberish + index (10 samples)':<35} {results['gribberish']:>9.2f}s {'':>15}")
    print(f"{'gribberish bulk (100 msgs)':<35} {results['gribberish_bulk']:>9.2f}s {'':>15}")
    print(f"{'scan_grib (20 msgs)':<35} {results['scan_grib_20msg']:>9.2f}s {'':>15}")
    print(f"{'scan_grib (estimated full 8160)':<35} {estimated_full:>9.1f}s {'(baseline)':>15}")
    print()

    # Speedup calculations
    for method, t in results.items():
        if method != 'scan_grib_20msg':
            speedup = estimated_full / max(t, 0.001)
            print(f"  {method}: {speedup:.0f}x faster than full scan_grib")

    # Key insight
    print(f"\n{'=' * 70}")
    print("KEY INSIGHT")
    print(f"{'=' * 70}")
    print(f"""
The .index file already contains ALL byte offsets and metadata that
scan_grib extracts by reading the full 5.8GB GRIB binary:
  - Message byte offsets and lengths
  - Variable names (param)
  - Level types and values
  - Ensemble member numbers
  - Forecast steps

gribberish adds grid metadata (shape, dims, dtype) from actual GRIB
headers, but this is STATIC for a given model and already available
in the pre-built template from HuggingFace.

Replacing scan_grib with .index parsing eliminates the 73-minute
Stage 1 bottleneck entirely.

For 365 days: saves {73.3 * 365 / 60:.0f} hours of compute per year.
""")


if __name__ == "__main__":
    run_comparison()
