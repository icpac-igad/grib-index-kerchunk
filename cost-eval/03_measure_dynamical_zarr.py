#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "dynamical-catalog>=0.5.0",
#     "xarray",
#     "zarr",
#     "icechunk",
#     "boto3",
#     "botocore",
#     "pyyaml",
#     "pandas",
#     "numpy",
# ]
# ///
"""
03_measure_dynamical_zarr.py
============================

Measure the **storage volume of the dynamical.org ARCO Zarr stores** that
serve as the apples-to-apples alternative to the GIK parquet catalogs.

Two complementary numbers per product:

1. **Uncompressed xarray view** (``ds.nbytes``) — what an analyst materialises
   after decompression. Sum of ``var.nbytes`` across all data variables.

2. **On-disk compressed bytes** — the actual S3 storage footprint that
   dynamical.org pays for, summed across every object under the icechunk
   prefix (``chunks/``, ``manifests/``, ``snapshots/``, ``transactions/``,
   ``repo/``, ``overwritten/``). This is the number that drives the
   "duplicated-archive storage cost" line in the cost model.

The ratio between the two is the dataset's effective Zarr compression
ratio — generally 3-10× for typical climate variables with zstd/blosc.

Output schemas
--------------
``dynamical_vars_<product>.csv`` (one row per data variable)::

    product, name, dtype, shape, nbytes_uncompressed

``dynamical_store_<product>.csv`` (one row per top-level subdir)::

    product, subdir, n_objects, bytes_on_disk

``03_dynamical_summary.json``: headlines + comparison vs step-01 GRIB volume.

Usage
-----
    uv run cost-eval/03_measure_dynamical_zarr.py
    uv run cost-eval/03_measure_dynamical_zarr.py --skip-store-list
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import boto3
import dynamical_catalog
import pandas as pd
import yaml
from botocore import UNSIGNED
from botocore.config import Config


# ---------------------------------------------------------------------------
# Product → dynamical catalog id (matches the CLI examples in the catalog page)
# ---------------------------------------------------------------------------

DYNAMICAL_IDS = {
    "gefs":  "noaa-gefs-forecast-35-day",
    "ecmwf": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
}


# ---------------------------------------------------------------------------
# Part 1: xarray view (uncompressed)
# ---------------------------------------------------------------------------

def measure_xarray_view(dataset_id: str) -> dict:
    """Open the dataset via dynamical_catalog and capture per-var sizes."""
    ds = dynamical_catalog.open(dataset_id)
    per_var = []
    for name, da in ds.data_vars.items():
        per_var.append({
            "name": name,
            "dtype": str(da.dtype),
            "shape": list(da.shape),
            "nbytes_uncompressed": int(da.nbytes),
        })

    init_time = ds["init_time"].values if "init_time" in ds.coords else None
    members  = ds["ensemble_member"].size if "ensemble_member" in ds.dims else None
    summary = {
        "dataset_id": dataset_id,
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "n_data_vars": len(ds.data_vars),
        "nbytes_uncompressed_total": int(ds.nbytes),
        "tib_uncompressed_total": round(ds.nbytes / 1024**4, 3),
        "init_time_count": int(ds["init_time"].size) if "init_time" in ds.coords else None,
        "init_time_start": str(init_time[0])[:25] if init_time is not None else None,
        "init_time_end":   str(init_time[-1])[:25] if init_time is not None else None,
        "ensemble_members": int(members) if members else None,
    }
    return summary, per_var


# ---------------------------------------------------------------------------
# Part 2: on-disk size via paginated S3 LIST (anonymous, us-west-2)
# ---------------------------------------------------------------------------

def measure_on_disk(bucket: str, prefix: str, region: str) -> dict:
    """Sum object bytes under `prefix`, grouped by top-level subdirectory.

    Uses boto3 paginated ListObjectsV2 — anonymous (UNSIGNED) access.
    For a typical icechunk repo with millions of chunks this is the
    fastest way to get an authoritative size: each LIST call returns
    Size in the same response, so we only stream metadata.
    """
    client = boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version=UNSIGNED),
    )
    paginator = client.get_paginator("list_objects_v2")

    by_subdir_bytes = defaultdict(int)
    by_subdir_count = defaultdict(int)
    n_pages = 0
    started = time.time()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        n_pages += 1
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # subdir is the first path component after `prefix`
            rel = key[len(prefix):] if key.startswith(prefix) else key
            subdir = rel.split("/", 1)[0] if "/" in rel else "(root)"
            by_subdir_bytes[subdir] += obj["Size"]
            by_subdir_count[subdir] += 1
        if n_pages % 50 == 0:
            elapsed = time.time() - started
            total_so_far = sum(by_subdir_count.values())
            print(f"  ... {n_pages} pages, {total_so_far:,} objects, "
                  f"{sum(by_subdir_bytes.values())/2**40:.1f} TiB on-disk "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - started
    total_bytes = sum(by_subdir_bytes.values())
    total_count = sum(by_subdir_count.values())
    rows = [
        {"subdir": k, "n_objects": by_subdir_count[k], "bytes_on_disk": by_subdir_bytes[k]}
        for k in sorted(by_subdir_bytes)
    ]
    summary = {
        "bucket":           bucket,
        "prefix":           prefix,
        "region":           region,
        "n_pages":          n_pages,
        "n_objects":        total_count,
        "bytes_on_disk":    int(total_bytes),
        "gib_on_disk":      round(total_bytes / 1024**3, 3),
        "tib_on_disk":      round(total_bytes / 1024**4, 3),
        "list_seconds":     round(elapsed, 1),
        "by_subdir":        {k: int(by_subdir_bytes[k]) for k in sorted(by_subdir_bytes)},
        "n_objects_by_subdir": {k: by_subdir_count[k] for k in sorted(by_subdir_count)},
    }
    return summary, rows


# ---------------------------------------------------------------------------
# Cost-model bridge: compare to step 01 GRIB volume
# ---------------------------------------------------------------------------

def cross_reference(product: str, view_summary: dict, store_summary: dict | None,
                    grib_summary: dict | None) -> dict:
    """Build the dynamical-vs-GIK comparison block."""
    block: dict = {}
    if not grib_summary:
        return block
    primary = grib_summary.get("primary", {}).get(product) or {}
    recent  = grib_summary.get("recent_check", {}).get(product) or {}

    # GIK's source GRIB at recent file sizes, extrapolated to the same
    # number of init_times that dynamical actually holds.
    n_dynamical_inits = view_summary.get("init_time_count") or 0
    if recent.get("mean_bytes_per_date_all_runs") and n_dynamical_inits:
        mean_per_date = float(recent["mean_bytes_per_date_all_runs"])
        # Multiply by dynamical's init_time_count for an apples-to-apples
        # archive footprint (their archive window).
        grib_matched = int(mean_per_date * n_dynamical_inits)
        block["matched_grib_bytes_recent"] = grib_matched
        block["matched_grib_tib_recent"]   = round(grib_matched / 1024**4, 3)
        block["dynamical_uncompressed_tib"] = view_summary["tib_uncompressed_total"]
        block["ratio_uncompressed_over_grib"] = round(
            view_summary["nbytes_uncompressed_total"] / grib_matched, 2
        )
        if store_summary:
            on_disk = store_summary["bytes_on_disk"]
            block["dynamical_on_disk_tib"] = round(on_disk / 1024**4, 3)
            block["dynamical_compression_ratio"] = (
                round(view_summary["nbytes_uncompressed_total"] / on_disk, 2)
            )
            block["ratio_dynamical_on_disk_over_grib"] = round(on_disk / grib_matched, 2)
    return block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_grib_summary(results_dir: Path) -> dict | None:
    p = results_dir / "01_grib_sizes_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--products", nargs="+", default=["gefs", "ecmwf"],
                    choices=["gefs", "ecmwf"])
    ap.add_argument("--skip-store-list", action="store_true",
                    help="skip the on-disk S3 list (faster but no compression ratio)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(__file__).with_name(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    grib_summary = _load_grib_summary(results_dir)

    headline: dict[str, dict] = {}
    for product in args.products:
        dataset_id = DYNAMICAL_IDS[product]
        print(f"[{product}] opening {dataset_id} …")
        view_summary, per_var = measure_xarray_view(dataset_id)
        print(f"[{product}] {view_summary['n_data_vars']} variables, "
              f"{view_summary['tib_uncompressed_total']} TiB uncompressed "
              f"across {view_summary['init_time_count']} init_times")

        pd.DataFrame(per_var).to_csv(
            results_dir / f"dynamical_vars_{product}.csv", index=False
        )

        store_summary = None
        if not args.skip_store_list:
            ds_cfg = dynamical_catalog.load_catalog()[dataset_id]
            ic = ds_cfg["icechunk"]
            print(f"[{product}] listing s3://{ic['bucket']}/{ic['prefix']} "
                  f"(region {ic['region']}) — this is the slow step …")
            store_summary, by_sub = measure_on_disk(
                ic["bucket"], ic["prefix"], ic["region"]
            )
            print(f"[{product}] on-disk total = "
                  f"{store_summary['tib_on_disk']} TiB across "
                  f"{store_summary['n_objects']:,} objects "
                  f"in {store_summary['list_seconds']}s")
            pd.DataFrame(
                [{"product": product, **r} for r in by_sub]
            ).to_csv(results_dir / f"dynamical_store_{product}.csv", index=False)

        compare = cross_reference(product, view_summary, store_summary, grib_summary)
        headline[product] = {
            "xarray_view":   view_summary,
            "on_disk_store": store_summary,
            "vs_gik_grib":   compare,
        }
        print(f"[{product}] {json.dumps(headline[product], indent=2)}")

    summary_path = results_dir / "03_dynamical_summary.json"
    summary_path.write_text(json.dumps(headline, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
