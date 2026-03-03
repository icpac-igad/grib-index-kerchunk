#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "fsspec",
#     "s3fs",
#     "pyarrow",
#     "gribberish",
#     "huggingface_hub",
# ]
# ///
"""
Process 3: Open GIK Parquets as Materialized xarray Dataset
=============================================================

Eagerly fetches all data from S3 using parallel byte-range reads and gribberish
decoding, producing an in-memory xarray Dataset. Unlike process2 (lazy/virtual),
this script downloads and decodes every timestep immediately.

Use this when you need the full dataset in memory for analysis, plotting, or
export to NetCDF.

The parquet files come from the E4DRR/gik-ecmwf-par HuggingFace dataset, which
contains pre-built reference files for multiple dates. To create your own
parquets for dates not in the catalog, use process1_make_virtual_manifest.py.

Usage:
    # Quick demo: 1 member
    uv run process3_open_materialized_dataset.py

    # Specific date, 3 members
    uv run process3_open_materialized_dataset.py --date 20250101 --members 3

    # Different variable
    uv run process3_open_materialized_dataset.py --var t2m --members 1

Author: ICPAC GIK Team
"""

import argparse
import json
import time

import fsspec
import gribberish
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

STEPS = list(range(0, 145, 3)) + [150, 156, 162, 168]  # 53 lead times
GRID = (721, 1440)


def parquet_to_zstore(parquet_path):
    """Read a GIK parquet and return a {zarr_key: value} dict."""
    df = pd.read_parquet(parquet_path)
    zstore = {}
    for _, row in df.iterrows():
        val = row["value"]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if isinstance(val, str) and len(val) > 0 and val[0] in ("[", "{"):
            val = json.loads(val)
        zstore[row["key"]] = val
    return zstore


def fetch_step(ref, s3):
    """Fetch GRIB bytes for one timestep from S3."""
    url, offset, length = ref[0], ref[1], ref[2]
    if not url.endswith(".grib2"):
        url += ".grib2"
    with s3.open(url, "rb") as f:
        f.seek(offset)
        return f.read(length)


def main():
    parser = argparse.ArgumentParser(
        description='Open ECMWF GIK parquets as materialized xarray (eagerly fetches all data)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run process3_open_materialized_dataset.py                       # 1 member
    uv run process3_open_materialized_dataset.py --date 20250101 --members 3
    uv run process3_open_materialized_dataset.py --var t2m --members 1
"""
    )
    parser.add_argument("--date", default="20250101", help="Date YYYYMMDD")
    parser.add_argument("--var", default="tp", help="Variable name (default: tp)")
    parser.add_argument("--members", type=int, default=1,
                        help="Number of members to load (default: 1, max: 51)")
    args = parser.parse_args()

    print("=" * 70)
    print("ECMWF Materialized Dataset (parallel S3 fetch + gribberish decode)")
    print("=" * 70)

    print(f"\nLoading catalog from HuggingFace...")
    catalog = pd.read_parquet(
        hf_hub_download("E4DRR/gik-ecmwf-par", "catalog.parquet",
                        repo_type="dataset")
    )

    subset = catalog[(catalog["date"] == args.date) & (catalog["run"] == "00z")]
    if subset.empty:
        dates = sorted(catalog["date"].unique())
        print(f"No data for {args.date} 00z. Available: {dates[0]}..{dates[-1]}")
        return

    n = min(args.members, len(subset))
    subset = subset.head(n)
    print(f"Date: {args.date} 00z — loading {n} member(s)")

    s3 = fsspec.filesystem("s3", anon=True)

    t0 = time.time()
    members_data = {}

    for _, row in subset.iterrows():
        mt0 = time.time()
        pq_path = hf_hub_download(
            "E4DRR/gik-ecmwf-par", row["hf_path"], repo_type="dataset"
        )
        zstore = parquet_to_zstore(pq_path)
        member_key = row["member"].replace("_", "")

        # Find variable references
        step_refs = []
        for i, h in enumerate(STEPS):
            for pattern in [
                f"step_{h:03d}/{args.var}/sfc/{member_key}/0.0.0",
                f"step_{h:03d}/{args.var}/surface/{member_key}/0.0.0",
            ]:
                if pattern in zstore and isinstance(zstore[pattern], list):
                    step_refs.append((i, zstore[pattern]))
                    break

        print(f"  {row['member']}: {len(step_refs)}/{len(STEPS)} {args.var} refs found, "
              f"fetching from S3...", end="", flush=True)

        data = np.full((len(STEPS), *GRID), np.nan, dtype=np.float32)
        with ThreadPoolExecutor(8) as pool:
            futs = {pool.submit(fetch_step, ref, s3): i for i, ref in step_refs}
            for fut in as_completed(futs):
                try:
                    arr = gribberish.parse_grib_array(fut.result(), 0).reshape(GRID)
                    data[futs[fut]] = arr
                except Exception:
                    pass

        members_data[row["member"]] = data
        print(f" done ({time.time()-mt0:.1f}s)")

    # Assemble xarray
    ds = xr.Dataset(
        {args.var: (["member", "step", "latitude", "longitude"],
                np.stack([members_data[m] for m in sorted(members_data)]))},
        coords={
            "member": sorted(members_data.keys()),
            "step": STEPS,
            "latitude": np.linspace(90, -90, 721),
            "longitude": np.linspace(-180, 179.75, 1440),
        },
    )

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(ds)
    print(f"\nTotal: {elapsed:.1f}s")
    valid = int((~np.isnan(ds[args.var].values)).sum())
    total = ds[args.var].values.size
    print(f"Valid cells: {valid:,}/{total:,} ({100*valid/total:.1f}%)")
    if valid > 0:
        vals = ds[args.var].values[~np.isnan(ds[args.var].values)]
        print(f"Range: [{vals.min():.6f}, {vals.max():.6f}]")


if __name__ == "__main__":
    main()
