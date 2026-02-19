#!/usr/bin/env python3
"""
Test gribberish with ECMWF parquet files.
Quick test to see if gribberish can decode ECMWF GRIB2 chunks.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    import gribberish
    print(f"  gribberish version: {gribberish.__version__ if hasattr(gribberish, '__version__') else 'unknown'}")
    print(f"  gribberish dir: {dir(gribberish)}")
except ImportError as e:
    print(f"  ERROR: {e}")
    exit(1)

try:
    import fsspec
    print("  fsspec: OK")
except ImportError as e:
    print(f"  fsspec ERROR: {e}")
    exit(1)

# Parquet file
PARQUET_FILE = Path("ecmwf_three_stage_20251126_00z/stage3_ens_01_final.parquet")

def read_parquet_refs(parquet_path):
    """Read parquet and extract zstore references."""
    print(f"\n1. Reading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   Rows: {len(df)}")

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            try:
                value = json.loads(value)
            except:
                pass
        zstore[key] = value

    print(f"   References: {len(zstore)}")
    return zstore


def find_tp_chunks(zstore):
    """Find TP variable chunks."""
    print("\n2. Finding TP chunks...")
    tp_chunks = []

    for key in zstore.keys():
        if '/tp/' in key and key.endswith('/0.0.0'):
            tp_chunks.append(key)

    print(f"   Found {len(tp_chunks)} TP chunks")
    if tp_chunks:
        print(f"   Example: {tp_chunks[0]}")

    return tp_chunks


def fetch_grib_chunk(zstore, chunk_key):
    """Fetch GRIB bytes from S3."""
    ref = zstore[chunk_key]

    if not isinstance(ref, list) or len(ref) < 3:
        print(f"   Unexpected ref format: {type(ref)}")
        return None

    url, offset, length = ref[0], ref[1], ref[2]

    # Fix URL if needed
    if not url.endswith('.grib2'):
        url = url + '.grib2'

    print(f"   URL: {url[:80]}...")
    print(f"   Offset: {offset}, Length: {length}")

    # Fetch from S3
    fs = fsspec.filesystem('s3', anon=True)
    s3_path = url if url.startswith('s3://') else f's3://{url}'

    with fs.open(s3_path, 'rb') as f:
        f.seek(offset)
        data = f.read(length)

    print(f"   Fetched: {len(data)} bytes")
    print(f"   Header: {data[:4]}")

    return data


def test_gribberish_decode(grib_bytes):
    """Test decoding with gribberish."""
    print("\n4. Testing gribberish decode...")

    # Check what functions are available
    print(f"   Available functions: {[x for x in dir(gribberish) if not x.startswith('_')]}")

    # Try different gribberish APIs
    try:
        # Method 1: parse_grib_array (if exists)
        if hasattr(gribberish, 'parse_grib_array'):
            print("   Trying parse_grib_array...")
            start = time.time()
            result = gribberish.parse_grib_array(grib_bytes, 0)
            elapsed = time.time() - start
            print(f"   SUCCESS! Shape: {result.shape}, Time: {elapsed*1000:.1f}ms")
            return result
    except Exception as e:
        print(f"   parse_grib_array failed: {e}")

    try:
        # Method 2: Message class (if exists)
        if hasattr(gribberish, 'Message'):
            print("   Trying Message class...")
            start = time.time()
            msg = gribberish.Message(grib_bytes)
            print(f"   Message created: {msg}")
            if hasattr(msg, 'data'):
                result = msg.data()
                elapsed = time.time() - start
                print(f"   SUCCESS! Shape: {result.shape}, Time: {elapsed*1000:.1f}ms")
                return result
    except Exception as e:
        print(f"   Message class failed: {e}")

    try:
        # Method 3: GribMessage (if exists)
        if hasattr(gribberish, 'GribMessage'):
            print("   Trying GribMessage class...")
            start = time.time()
            msg = gribberish.GribMessage(grib_bytes)
            if hasattr(msg, 'to_ndarray'):
                result = msg.to_ndarray()
            elif hasattr(msg, 'data'):
                result = msg.data()
            elif hasattr(msg, 'values'):
                result = msg.values
            else:
                print(f"   GribMessage attrs: {[x for x in dir(msg) if not x.startswith('_')]}")
                return None
            elapsed = time.time() - start
            print(f"   SUCCESS! Shape: {result.shape}, Time: {elapsed*1000:.1f}ms")
            return result
    except Exception as e:
        print(f"   GribMessage failed: {e}")

    try:
        # Method 4: parse_grib_messages (if exists)
        if hasattr(gribberish, 'parse_grib_messages'):
            print("   Trying parse_grib_messages...")
            start = time.time()
            messages = gribberish.parse_grib_messages(grib_bytes)
            print(f"   Messages: {messages}")
            if messages:
                msg = messages[0]
                print(f"   First message attrs: {[x for x in dir(msg) if not x.startswith('_')]}")
                if hasattr(msg, 'data'):
                    result = msg.data()
                    elapsed = time.time() - start
                    print(f"   SUCCESS! Shape: {result.shape}, Time: {elapsed*1000:.1f}ms")
                    return result
    except Exception as e:
        print(f"   parse_grib_messages failed: {e}")

    try:
        # Method 5: scan/read functions
        for func_name in ['scan', 'read', 'decode', 'load']:
            if hasattr(gribberish, func_name):
                print(f"   Trying {func_name}...")
                func = getattr(gribberish, func_name)
                result = func(grib_bytes)
                print(f"   Result: {result}")
    except Exception as e:
        print(f"   Other methods failed: {e}")

    print("   Could not find working decode method")
    return None


def test_cfgrib_comparison(grib_bytes):
    """Compare with cfgrib for reference."""
    print("\n5. Comparing with cfgrib...")
    try:
        import tempfile
        import xarray as xr
        import os

        start = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
            tmp.write(grib_bytes)
            tmp_path = tmp.name

        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        result = ds[var_name].values
        ds.close()
        os.unlink(tmp_path)

        elapsed = time.time() - start
        print(f"   cfgrib SUCCESS! Shape: {result.shape}, Time: {elapsed*1000:.1f}ms")
        return result
    except Exception as e:
        print(f"   cfgrib failed: {e}")
        return None


def main():
    print("=" * 60)
    print("GRIBBERISH ECMWF COMPATIBILITY TEST")
    print("=" * 60)

    # Read parquet
    zstore = read_parquet_refs(PARQUET_FILE)

    # Find TP chunks
    tp_chunks = find_tp_chunks(zstore)
    if not tp_chunks:
        print("ERROR: No TP chunks found!")
        return

    # Fetch first chunk
    print("\n3. Fetching first TP chunk from S3...")
    chunk_key = tp_chunks[0]
    grib_bytes = fetch_grib_chunk(zstore, chunk_key)

    if grib_bytes is None:
        print("ERROR: Could not fetch GRIB chunk!")
        return

    # Test gribberish
    gribberish_result = test_gribberish_decode(grib_bytes)

    # Compare with cfgrib
    cfgrib_result = test_cfgrib_comparison(grib_bytes)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if gribberish_result is not None:
        print("  gribberish: COMPATIBLE!")
        print(f"    Shape: {gribberish_result.shape}")
        print(f"    Min: {np.nanmin(gribberish_result):.6f}")
        print(f"    Max: {np.nanmax(gribberish_result):.6f}")
    else:
        print("  gribberish: NOT COMPATIBLE or API not found")

    if cfgrib_result is not None:
        print("  cfgrib: Works (baseline)")
        print(f"    Shape: {cfgrib_result.shape}")


if __name__ == "__main__":
    main()
