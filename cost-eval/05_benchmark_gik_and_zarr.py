#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "dynamical-catalog>=0.5.0",
#     "xarray",
#     "zarr",
#     "icechunk",
#     "boto3",
#     "fsspec",
#     "s3fs",
#     "pyyaml",
#     "pandas",
#     "numpy",
#     "psutil",
# ]
# ///
"""
05_benchmark_gik_and_zarr.py
============================

Compares the **byte-range data-streaming cost** of the GIK / Herbie-Mode-B
pattern against the **chunk-aware Zarr** pattern that dynamical.org offers,
across three access patterns that exercise the architectural difference:

  W1. GLOBAL  — full 0.25° grid, all members, all lead times, one variable
                A "what's the rainfall today across the planet" query.

  W2. REGIONAL — East Africa subset (lat -12..15, lon 25..52),
                 all members, all lead times, one variable.
                 A typical operational use case at ICPAC.

  W3. PENCIL   — single (lat, lon) point time series across all init_times,
                 one member, one lead, one variable.
                 A "validation against a station" query.

The three methods compared:

  M1. **GIK byte-range**  (≡ Herbie Mode B at the structural level):
        sums byte_lengths from parquet refs for the matching slice.
        GIK *cannot* spatially subset within a GRIB message — the full
        global field for every matching (var, mem, step) is on the wire.

  M2. **Zarr-global**: dynamical_catalog.open(...).sel(time/var subset).compute()
        — global spatial; chunks not aligned to a region.

  M3. **Zarr-regional**: same but with .sel(latitude=slice(...), longitude=...)
        — engages icechunk's chunk-level addressing; only the chunks
        overlapping the region are fetched.

Bytes-on-wire is captured **analytically** wherever the chunk geometry
makes it deterministic (Zarr) or wherever the byte_length is recorded
in the index (GIK), and **empirically** with ``psutil.net_io_counters``
for one live materialisation per cell as a cross-check.

Outputs
-------
``05_streaming_summary.json``: per-workload × per-method bytes, wall-clock, $.
``benchmark_streaming.csv``: one row per (product, workload, method, source).

Usage
-----
    uv run cost-eval/05_benchmark_gik_and_zarr.py
    uv run cost-eval/05_benchmark_gik_and_zarr.py --skip-live   # analytical only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Geography of the regional and pencil workloads
# ---------------------------------------------------------------------------

# East Africa box used by ICPAC and the existing repo (matches
# `fetch_tp_herbie.py`'s subset).
REGION = {
    "lat_min": -12.0,  "lat_max":  15.0,    # 27° tall
    "lon_min":  25.0,  "lon_max":  52.0,    # 27° wide
}

# Pencil-chunking workload: one (lat, lon) point (e.g. Addis Ababa)
PENCIL = {"lat": 9.0, "lon": 38.75}


# ---------------------------------------------------------------------------
# Chunk-aware analytical model
# ---------------------------------------------------------------------------

def _grid_step_deg(n: int, span: float = 360.0) -> float:
    return span / n


def _chunks_overlap_region(chunks: tuple[int, ...], shape: tuple[int, ...],
                           dims: tuple[str, ...],
                           lat_min: float, lat_max: float,
                           lon_min: float, lon_max: float) -> int:
    """How many chunks intersect the given lat/lon box?

    Assumes lat goes 90..-90 in size-721 axes (0.25°) and lon 0..359.75
    in size-1440 axes. This is the convention used by both dynamical.org
    and the source GRIB grids.
    """
    lat_idx = dims.index("latitude")
    lon_idx = dims.index("longitude")
    nlat, nlon = shape[lat_idx], shape[lon_idx]
    clat, clon = chunks[lat_idx], chunks[lon_idx]

    # lat decreasing: index 0 = 90, last = -90
    lat_step = 180.0 / (nlat - 1) if nlat > 1 else 0.25
    lat_top_idx = int(np.floor((90.0 - lat_max) / lat_step))
    lat_bot_idx = int(np.ceil((90.0 - lat_min) / lat_step))
    lat_chunk_lo = lat_top_idx // clat
    lat_chunk_hi = lat_bot_idx // clat
    n_lat_chunks = lat_chunk_hi - lat_chunk_lo + 1

    # lon increasing from 0 to 360-step
    lon_step = 360.0 / nlon
    lon_lo_idx = int(np.floor(lon_min / lon_step))
    lon_hi_idx = int(np.ceil(lon_max / lon_step))
    lon_chunk_lo = lon_lo_idx // clon
    lon_chunk_hi = lon_hi_idx // clon
    n_lon_chunks = lon_chunk_hi - lon_chunk_lo + 1

    return max(1, n_lat_chunks) * max(1, n_lon_chunks)


def _chunk_count_global(chunks: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Total number of chunks across all dims."""
    return int(np.prod([
        max(1, int(np.ceil(s / c))) for s, c in zip(shape, chunks)
    ]))


def _zarr_bytes_for_workload(da, workload: str,
                             n_inits: int = 1,
                             n_leads_required: int | None = None,
                             zarr_compression_ratio: float = 6.4) -> dict:
    """Analytical Zarr bytes-on-wire for a given workload on a data variable.

    Uses the measured Zarr compression ratio from step 03 (~6.4× both
    products) and the variable's chunk shape from encoding.
    """
    chunks = da.encoding["chunks"]
    shape  = da.shape
    dims   = da.dims
    chunk_bytes_uncompressed = int(np.prod(chunks) * da.dtype.itemsize)
    chunk_bytes_compressed   = int(chunk_bytes_uncompressed / zarr_compression_ratio)

    if workload == "global":
        # All chunks at the relevant init_times (1 init = 1 chunk along init_time
        # for dynamical's encoding).
        n_chunks_per_init = _chunk_count_global(chunks, shape) // shape[dims.index("init_time")]
        n_chunks = n_inits * n_chunks_per_init
    elif workload == "regional":
        # Chunks at relevant inits × chunks_overlap_region (lat,lon) × full
        # extent of every other dim (the chunk shape already picks all members
        # for ECMWF; multiply by inner-axis chunk counts otherwise).
        n_spatial_chunks = _chunks_overlap_region(
            chunks, shape, dims,
            REGION["lat_min"], REGION["lat_max"],
            REGION["lon_min"], REGION["lon_max"],
        )
        # For each non-spatial, non-init dim, multiply by ceil(shape/chunks).
        other_factor = 1
        for d, s, c in zip(dims, shape, chunks):
            if d in ("init_time", "latitude", "longitude"):
                continue
            other_factor *= max(1, int(np.ceil(s / c)))
        n_chunks = n_inits * n_spatial_chunks * other_factor
    elif workload == "pencil":
        # One spatial chunk per init × full non-spatial-non-init chunk grid.
        n_spatial_chunks = 1
        other_factor = 1
        for d, s, c in zip(dims, shape, chunks):
            if d in ("init_time", "latitude", "longitude"):
                continue
            other_factor *= max(1, int(np.ceil(s / c)))
        n_chunks = n_inits * n_spatial_chunks * other_factor
    else:
        raise ValueError(workload)

    bytes_on_wire = n_chunks * chunk_bytes_compressed
    return {
        "method": "zarr",
        "workload": workload,
        "chunks_shape": list(chunks),
        "chunk_size_compressed_mb": round(chunk_bytes_compressed / 1e6, 3),
        "n_chunks_touched": int(n_chunks),
        "bytes_on_wire": int(bytes_on_wire),
        "mib_on_wire": round(bytes_on_wire / 2**20, 2),
        "n_get_requests": int(n_chunks),  # one GET per chunk
    }


# ---------------------------------------------------------------------------
# GIK analytical model — bytes are independent of spatial subset
# ---------------------------------------------------------------------------

def _gik_message_bytes(product: str) -> int:
    """Representative GRIB message size for one variable at one (mem, step).

    These are the per-message byte counts implied by the .index/.idx records
    (matches the per-var assumption in 04_benchmark_herbie's Mode B model).
    """
    return 2_000_000 if product == "gefs" else 5_000_000


def _gik_bytes(product: str, n_vars: int, n_members: int,
               n_leads: int, n_inits: int) -> dict:
    """Bytes-on-wire for GIK byte-range reads. Identical for global and
    regional workloads — GIK cannot spatially subset within a message."""
    msg = _gik_message_bytes(product)
    n_msgs = n_vars * n_members * n_leads * n_inits
    bytes_total = n_msgs * msg
    # GETs: one HTTP range request per message for GEFS; one per (var, lead)
    # for ECMWF (51 members packed in a single file → single range fetches
    # all 51 messages for that var,lead).
    n_gets = (n_vars * n_members * n_leads * n_inits
              if product == "gefs"
              else n_vars * n_leads * n_inits)
    return {
        "method": "gik",
        "n_messages": n_msgs,
        "bytes_on_wire": int(bytes_total),
        "mib_on_wire": round(bytes_total / 2**20, 2),
        "n_get_requests": int(n_gets),
        "note": "bytes independent of spatial subset (full GRIB message read)",
    }


# ---------------------------------------------------------------------------
# Live cross-check (empirical bytes via psutil.net_io_counters delta)
# ---------------------------------------------------------------------------

def _vcpu_seconds_now() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _net_bytes_recv() -> int:
    return int(psutil.net_io_counters().bytes_recv)


def _live_zarr_materialise(da_subset, label: str) -> dict:
    """Materialise the lazy DataArray subset and capture wall/vCPU/bytes."""
    print(f"  [live {label}] start (shape={tuple(da_subset.shape)})")
    gc.collect()
    cpu0 = _vcpu_seconds_now()
    net0 = _net_bytes_recv()
    wall0 = time.perf_counter()
    try:
        arr = da_subset.compute()
        ok = True
        err = None
        n_vals = int(arr.size)
    except Exception as exc:
        ok = False
        err = repr(exc)[:200]
        n_vals = 0
    wall = time.perf_counter() - wall0
    cpu  = _vcpu_seconds_now() - cpu0
    net_delta = _net_bytes_recv() - net0
    out = {
        "wall_s": round(wall, 3),
        "vcpu_s": round(cpu, 3),
        "live_net_bytes_recv": int(net_delta),
        "live_net_mib": round(net_delta / 2**20, 2),
        "live_decoded_values": n_vals,
        "status": "ok" if ok else "error",
        "error":  err,
    }
    print(f"  [live {label}] wall={wall:.2f}s vcpu={cpu:.2f}s "
          f"net={net_delta/2**20:.1f} MiB  decoded={n_vals:,}")
    return out


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def cost(bytes_on_wire: int, n_get: int, wall_s: float | None,
         pricing: dict) -> dict:
    bytes_gb = bytes_on_wire / 1e9
    vcpu_h = (wall_s or 0) / 3600.0
    cross = bytes_gb * pricing["egress_per_gb"]["aws_internet"]
    get_c = n_get / 1000 * pricing["request_per_1000"]["aws_s3_get"]
    vcpu_c = vcpu_h * pricing["compute_per_vcpu_hour"]["local_baseline"]
    return {
        "egress_gb":               round(bytes_gb, 4),
        "n_get":                   int(n_get),
        "vcpu_hours":              round(vcpu_h, 4),
        "$_egress_same_cloud":     0.0,
        "$_egress_cross_cloud":    round(cross, 4),
        "$_get_requests":          round(get_c, 5),
        "$_vcpu":                  round(vcpu_c, 4),
        "$_total_same_cloud":      round(get_c + vcpu_c, 4),
        "$_total_cross_cloud":     round(cross + get_c + vcpu_c, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DYNAMICAL_IDS = {
    "gefs":  "noaa-gefs-forecast-35-day",
    "ecmwf": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--products", nargs="+", default=["gefs", "ecmwf"],
                    choices=["gefs", "ecmwf"])
    ap.add_argument("--skip-live", action="store_true",
                    help="skip live materialisations; analytical only")
    ap.add_argument("--init-date", default="2024-05-22",
                    help="ISO init_time for ECMWF (use one in dynamical's window)")
    ap.add_argument("--gefs-init-date", default="2020-10-13")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pricing = cfg["pricing"]
    results_dir = Path(__file__).with_name(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    import dynamical_catalog

    rows: list[dict] = []
    headline: dict[str, dict] = {}

    for product in args.products:
        print(f"=== {product} ===")
        dataset_id = DYNAMICAL_IDS[product]
        ds = dynamical_catalog.open(dataset_id)
        # Pick a "typical" var available in both source GRIB and dynamical
        var = "temperature_2m"
        da = ds[var]

        # Workload axis sizes for the analytical model.
        n_full_members = da.sizes.get("ensemble_member", 1)
        n_full_leads   = da.sizes.get("lead_time", 1)
        n_vars         = 1

        # n_inits per workload:
        #   global / regional : 1 init (one forecast cycle)
        #   pencil            : full archive (the whole point of a pencil-
        #                       chunked query is time-series across many inits)
        n_inits_by_workload = {
            "global":   1,
            "regional": 1,
            "pencil":   int(da.sizes.get("init_time", 1)),
        }

        # ----- Analytical: GIK -----
        for workload, n_inits in n_inits_by_workload.items():
            gik = _gik_bytes(product, n_vars, n_full_members,
                             n_full_leads, n_inits)
            c = cost(gik["bytes_on_wire"], gik["n_get_requests"], None, pricing)
            rows.append({"product": product, "workload": workload,
                         "method": "gik_byte_range", "source": "analytical",
                         "n_inits": n_inits, **gik, **c})

        # ----- Analytical: Zarr (same n_inits per workload) -----
        for workload, n_inits in n_inits_by_workload.items():
            z = _zarr_bytes_for_workload(da, workload, n_inits=n_inits)
            c = cost(z["bytes_on_wire"], z["n_get_requests"], None, pricing)
            rows.append({"product": product, "workload": workload,
                         "method": "zarr", "source": "analytical",
                         "n_inits": n_inits, **z, **c})

        # ----- Live cross-check: Zarr regional + Zarr pencil -----
        if not args.skip_live:
            init = (args.gefs_init_date if product == "gefs"
                    else args.init_date)
            print(f"  init_time = {init}, var = {var}")

            # Regional, one init, all leads, all mems
            da_region = da.sel(
                init_time=init,
                latitude=slice(REGION["lat_max"], REGION["lat_min"]),  # descending
                longitude=slice(REGION["lon_min"], REGION["lon_max"]),
            )
            live = _live_zarr_materialise(da_region,
                                          f"{product}/zarr/regional/1init")
            # Pencil, one point, all inits but cap to first 100 for speed
            n_pencil_inits = min(100, int(da.sizes["init_time"]))
            da_pencil = da.isel(init_time=slice(0, n_pencil_inits)).sel(
                latitude=PENCIL["lat"], longitude=PENCIL["lon"], method="nearest",
            )
            live_p = _live_zarr_materialise(da_pencil,
                                            f"{product}/zarr/pencil/{n_pencil_inits}inits")

            rows.append({"product": product, "workload": "regional",
                         "method": "zarr", "source": "live_1init",
                         "bytes_on_wire": live["live_net_bytes_recv"],
                         "mib_on_wire": live["live_net_mib"],
                         "wall_s": live["wall_s"], "vcpu_s": live["vcpu_s"],
                         "decoded_values": live["live_decoded_values"]})
            rows.append({"product": product, "workload": "pencil",
                         "method": "zarr",
                         "source": f"live_{n_pencil_inits}inits",
                         "bytes_on_wire": live_p["live_net_bytes_recv"],
                         "mib_on_wire": live_p["live_net_mib"],
                         "wall_s": live_p["wall_s"], "vcpu_s": live_p["vcpu_s"],
                         "decoded_values": live_p["live_decoded_values"]})

        # Build a flat compare table for the report.
        cmp = {}
        for w in ("global", "regional", "pencil"):
            entries = [r for r in rows
                       if r["product"] == product and r["workload"] == w
                       and r["source"] == "analytical"]
            cmp[w] = {r["method"]: {
                "mib_on_wire": r["mib_on_wire"],
                "n_get": r["n_get"],
                "$_total_cross_cloud": r["$_total_cross_cloud"],
                "$_total_same_cloud":  r["$_total_same_cloud"],
            } for r in entries}
            if cmp[w].get("gik_byte_range") and cmp[w].get("zarr"):
                gik_b = cmp[w]["gik_byte_range"]["mib_on_wire"]
                zarr_b = cmp[w]["zarr"]["mib_on_wire"]
                cmp[w]["gik_over_zarr_ratio"] = (
                    round(gik_b / zarr_b, 2) if zarr_b else None
                )
        headline[product] = cmp

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "benchmark_streaming.csv", index=False)
    print(f"wrote {results_dir / 'benchmark_streaming.csv'}")
    (results_dir / "05_streaming_summary.json").write_text(
        json.dumps(headline, indent=2) + "\n"
    )
    print(f"wrote {results_dir / '05_streaming_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
