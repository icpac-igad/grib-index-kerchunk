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
#     "dask",
#     "huggingface_hub",
# ]
# ///
"""
Process 2: Open GIK Parquets as Lazy xarray Dataset
====================================================

Builds an xarray Dataset backed by dask.delayed — NO data is fetched from S3
until you call .load() or .sel().compute(). This makes it possible to set up
a dataset covering all 51 members and 85 timesteps, then selectively load only
the slices you need.

Three phases:
  Phase 1 (fast):     Download parquets from HuggingFace, build reference lookup
  Phase 2 (instant):  Build xarray Dataset with dask arrays — NO S3 reads
  Phase 3 (on-demand): Call .load() or .sel() to fetch only needed bytes from S3

The parquet files come from the E4DRR/gik-ecmwf-par HuggingFace dataset, which
contains pre-built reference files for multiple dates. To create your own
parquets for dates not in the catalog, use process1_make_virtual_manifest.py.

Usage:
    # Quick demo: 1 member, lazy (no S3 reads)
    uv run process2_open_virtual_dataset.py

    # Specific date, 5 members
    uv run process2_open_virtual_dataset.py --date 20250101 --members 5

    # Load step 24 to verify data access
    uv run process2_open_virtual_dataset.py --load-step 24

    # All 51 members, load step 48
    uv run process2_open_virtual_dataset.py --members 51 --load-step 48

Author: ICPAC GIK Team
"""

import argparse
import json
import time

import dask
import dask.array as da
import fsspec
import gribberish
import numpy as np
import pandas as pd
import xarray as xr
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


def make_lazy_fetcher(ref, s3_fs):
    """Return a dask-delayed function that fetches + decodes one GRIB chunk."""

    def _fetch():
        url, offset, length = ref[0], ref[1], ref[2]
        if not url.endswith(".grib2"):
            url += ".grib2"
        with s3_fs.open(url, "rb") as f:
            f.seek(offset)
            grib_bytes = f.read(length)
        return gribberish.parse_grib_array(grib_bytes, 0).reshape(GRID).astype(
            np.float32
        )

    return _fetch


def build_lazy_dataset(zstores, member_names, var_name="tp"):
    """Build an xarray Dataset with dask arrays — no S3 reads until .load().

    Each chunk is a dask.delayed call that fetches + decodes one GRIB message.
    """
    s3_fs = fsspec.filesystem("s3", anon=True)
    n_steps = len(STEPS)
    nlat, nlon = GRID

    # Build dask arrays: one delayed task per (member, step) chunk
    member_arrays = []
    refs_found = 0

    for m_idx, (member_name, zstore) in enumerate(
        zip(member_names, zstores)
    ):
        member_key = member_name.replace("_", "")
        step_arrays = []

        for s_idx, h in enumerate(STEPS):
            ref = None
            for pattern in [
                f"step_{h:03d}/{var_name}/sfc/{member_key}/0.0.0",
                f"step_{h:03d}/{var_name}/surface/{member_key}/0.0.0",
            ]:
                if pattern in zstore and isinstance(zstore[pattern], list):
                    ref = zstore[pattern]
                    refs_found += 1
                    break

            if ref is not None:
                fetcher = make_lazy_fetcher(ref, s3_fs)
                delayed = da.from_delayed(
                    dask.delayed(fetcher)(),
                    shape=GRID,
                    dtype=np.float32,
                )
            else:
                delayed = da.full(GRID, np.nan, dtype=np.float32)

            step_arrays.append(delayed)

        # Stack steps: (n_steps, nlat, nlon)
        member_arr = da.stack(step_arrays, axis=0)
        member_arrays.append(member_arr)

    # Stack members: (n_members, n_steps, nlat, nlon)
    data = da.stack(member_arrays, axis=0)

    ds = xr.Dataset(
        {var_name: (["member", "step", "latitude", "longitude"], data)},
        coords={
            "member": member_names,
            "step": STEPS,
            "latitude": np.linspace(90, -90, nlat),
            "longitude": np.linspace(-180, 179.75, nlon),
        },
    )
    return ds, refs_found


def main():
    parser = argparse.ArgumentParser(
        description='Open ECMWF GIK parquets as lazy xarray (no data fetched until .load())',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run process2_open_virtual_dataset.py                         # 1 member, lazy
    uv run process2_open_virtual_dataset.py --date 20250101 --members 51  # all members
    uv run process2_open_virtual_dataset.py --load-step 24          # load step 24 only
"""
    )
    parser.add_argument("--date", default="20250101", help="Date YYYYMMDD")
    parser.add_argument("--var", default="tp", help="Variable name (default: tp)")
    parser.add_argument("--members", type=int, default=1,
                        help="Number of members (default: 1, max: 51)")
    parser.add_argument("--load-step", type=int, default=None,
                        help="If set, actually fetch this lead time (hours) to verify")
    args = parser.parse_args()

    print("=" * 70)
    print("ECMWF Lazy Virtual Dataset (dask.delayed, no S3 reads until .load())")
    print("=" * 70)

    print("\nPhase 1: Loading catalog + parquets from HuggingFace...")
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
    print(f"Date: {args.date} 00z — building lazy dataset for {n} member(s)\n")

    # Phase 1: Download parquets + parse references (fast, no S3 reads)
    t0 = time.time()
    zstores = []
    member_names = []
    for _, row in subset.iterrows():
        pq_path = hf_hub_download(
            "E4DRR/gik-ecmwf-par", row["hf_path"], repo_type="dataset"
        )
        zstores.append(parquet_to_zstore(pq_path))
        member_names.append(row["member"])
        print(f"  {row['member']}: parquet loaded", flush=True)

    phase1 = time.time() - t0

    # Phase 2: Build lazy xarray (instant, no S3 reads)
    t1 = time.time()
    ds, refs_found = build_lazy_dataset(zstores, member_names, var_name=args.var)
    phase2 = time.time() - t1

    print(f"\n{'='*70}")
    print(f"Lazy xarray Dataset (NO data fetched from S3 yet):")
    print(f"{'='*70}")
    print(ds)
    print(f"\nPhase 1 (parquet download + parse): {phase1:.1f}s")
    print(f"Phase 2 (build lazy dataset):        {phase2:.1f}s")
    print(f"References found: {refs_found} / {n * len(STEPS)}")
    print(f"Data in memory: 0 bytes (all dask.delayed)")
    print(f"\nTo materialize a slice:")
    print(f'  ds["{args.var}"].sel(step=24).load()   # ~2 MB per member from S3')

    # Phase 3: optionally load one step to verify
    if args.load_step is not None:
        print(f"\n{'='*70}")
        print(f"Phase 3: Loading step={args.load_step} from S3...")
        t2 = time.time()
        sliced = ds[args.var].sel(step=args.load_step).load()
        phase3 = time.time() - t2
        valid = int((~np.isnan(sliced.values)).sum())
        total = sliced.values.size
        print(f"  Shape: {sliced.shape}")
        print(f"  Valid: {valid:,}/{total:,} ({100*valid/total:.1f}%)")
        if valid > 0:
            vals = sliced.values[~np.isnan(sliced.values)]
            print(f"  Range: [{vals.min():.6f}, {vals.max():.6f}]")
        print(f"  Fetch time: {phase3:.1f}s")


if __name__ == "__main__":
    main()
