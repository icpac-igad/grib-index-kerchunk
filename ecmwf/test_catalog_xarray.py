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
Quick test: catalog → parquet → xarray Dataset for one date (1 member only).

Usage:
    uv run test_catalog_xarray.py
    uv run test_catalog_xarray.py --date 20250101 --members 3
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

s3 = fsspec.filesystem("s3", anon=True)


def parquet_to_zstore(parquet_path):
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


def fetch_step(ref):
    url, offset, length = ref[0], ref[1], ref[2]
    if not url.endswith(".grib2"):
        url += ".grib2"
    with s3.open(url, "rb") as f:
        f.seek(offset)
        return f.read(length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20250101", help="Date YYYYMMDD")
    parser.add_argument("--members", type=int, default=1,
                        help="Number of members to load (default: 1, max: 51)")
    args = parser.parse_args()

    print(f"Loading catalog...")
    catalog = pd.read_parquet(
        hf_hub_download("E4DRR/gik-ecmwf-par", "catalog.parquet",
                        repo_type="dataset")
    )

    subset = catalog[(catalog["date"] == args.date) & (catalog["run"] == "00z")]
    if subset.empty:
        print(f"No data for {args.date} 00z. Available dates:")
        print(sorted(catalog["date"].unique())[:10], "...")
        return

    n = min(args.members, len(subset))
    subset = subset.head(n)
    print(f"Date: {args.date} 00z — loading {n} member(s)")

    t0 = time.time()
    members_data = {}

    for _, row in subset.iterrows():
        mt0 = time.time()
        pq_path = hf_hub_download(
            "E4DRR/gik-ecmwf-par", row["hf_path"], repo_type="dataset"
        )
        zstore = parquet_to_zstore(pq_path)
        member_key = row["member"].replace("_", "")

        # Find tp references
        step_refs = []
        for i, h in enumerate(STEPS):
            for pattern in [
                f"step_{h:03d}/tp/sfc/{member_key}/0.0.0",
                f"step_{h:03d}/tp/surface/{member_key}/0.0.0",
            ]:
                if pattern in zstore and isinstance(zstore[pattern], list):
                    step_refs.append((i, zstore[pattern]))
                    break

        print(f"  {row['member']}: {len(step_refs)}/{len(STEPS)} tp refs found, "
              f"fetching from S3...", end="", flush=True)

        data = np.full((len(STEPS), *GRID), np.nan, dtype=np.float32)
        with ThreadPoolExecutor(8) as pool:
            futs = {pool.submit(fetch_step, ref): i for i, ref in step_refs}
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
        {"tp": (["member", "step", "latitude", "longitude"],
                np.stack([members_data[m] for m in sorted(members_data)]))},
        coords={
            "member": sorted(members_data.keys()),
            "step": STEPS,
            "latitude": np.linspace(90, -90, 721),
            "longitude": np.linspace(-180, 179.75, 1440),
        },
    )

    elapsed = time.time() - t0
    print(f"\n{ds}")
    print(f"\nTotal: {elapsed:.1f}s")
    valid = int((~np.isnan(ds["tp"].values)).sum())
    total = ds["tp"].values.size
    print(f"Valid cells: {valid:,}/{total:,} ({100*valid/total:.1f}%)")


if __name__ == "__main__":
    main()
