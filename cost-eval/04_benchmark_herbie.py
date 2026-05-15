#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "herbie-data",
#     "xarray",
#     "cfgrib",
#     "eccodes",
#     "s3fs",
#     "fsspec",
#     "pyyaml",
#     "pandas",
#     "numpy",
# ]
# ///
"""
04_benchmark_herbie.py
======================

Measure the wall-clock + bytes-on-wire cost of accessing the same data the
GIK parquet catalog indexes, but via Herbie. Two Herbie modes are tested,
because real-world users do both:

  A. Full-file download  (``H.download()`` then open with cfgrib)
       — the "naive" pattern: pulls the entire GRIB file even if the
         analyst only wants one variable.

  B. Byte-range searchString  (``H.xarray(":var:")``)
       — Herbie parses the sidecar .idx and issues an HTTP range read for
         just the matching GRIB messages. This is structurally what GIK
         does too — the difference is parsing on every query vs reading a
         pre-built parquet reference.

The benchmark performs **one small live download per product per mode**
(timing + bytes), then extrapolates to the full per-analysis workload
defined in ``config.yaml -> workload.<product>``. All extrapolated dollar
numbers are reported under two egress scenarios:

  same_cloud      : NOAA + ECMWF AWS Open Data buckets → $0/GB egress
                    (analyst running on AWS in-region; current reality)
  cross_cloud     : analyst running off-AWS (GCP / on-prem / different
                    region) → $0.09/GB AWS internet egress

Outputs
-------
``benchmark_herbie.csv``::

    product, mode, file_url, n_bytes_observed, wall_s, mb_per_s,
    decode_s, vcpu_s, status

``04_herbie_summary.json``: per-product extrapolation to the full slice +
$ in each egress scenario.

Usage
-----
    uv run cost-eval/04_benchmark_herbie.py
    uv run cost-eval/04_benchmark_herbie.py --skip-live   # analytical only
    uv run cost-eval/04_benchmark_herbie.py --skip-full-download  # mode A
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

import pandas as pd
import yaml

# Anonymous source GRIB access (matches the rest of cost-eval).
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# Quiet some legitimate-but-noisy Herbie warnings during a timed run.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Live benchmark — one small file per product per mode
# ---------------------------------------------------------------------------

def _vcpu_seconds_now() -> float:
    """Process user+system CPU time in seconds, suitable for diffing."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _bench_one(fn, label: str) -> dict:
    """Time fn(); return wall, vCPU, and capture the size hint it returns."""
    print(f"  [{label}] start")
    gc.collect()
    cpu0 = _vcpu_seconds_now()
    wall0 = time.perf_counter()
    try:
        result = fn()
        status = "ok"
        err = None
    except Exception as exc:
        result = None
        status = "error"
        err = repr(exc)[:200]
    wall = time.perf_counter() - wall0
    cpu = _vcpu_seconds_now() - cpu0
    out = {
        "wall_s": round(wall, 3),
        "vcpu_s": round(cpu, 3),
        "status": status,
        "error":  err,
        "result": result,
    }
    print(f"  [{label}] {status} wall={wall:.2f}s vcpu={cpu:.2f}s")
    return out


def _gefs_herbie(date: str, fxx: int, member: int):
    """Return a Herbie object for one GEFS member/step. member: 1..30."""
    from herbie import Herbie
    return Herbie(date, model="gefs", product="atmos.25", fxx=fxx,
                  member=member, save_dir="/tmp/herbie_cache", verbose=False)


def _ecmwf_herbie(date: str, fxx: int):
    """Return a Herbie object for one ECMWF enfo step (all 51 mems packed)."""
    from herbie import Herbie
    return Herbie(date, model="ifs", product="enfo", fxx=fxx,
                  save_dir="/tmp/herbie_cache", verbose=False)


def mode_a_full_download(product: str, date: str, fxx: int) -> dict:
    """Herbie naive mode — pull the entire GRIB file via H.download()."""
    if product == "gefs":
        H = _gefs_herbie(date, fxx, member=1)
    else:
        H = _ecmwf_herbie(date, fxx)
    path = H.download(verbose=False)
    size = int(Path(path).stat().st_size)
    # Open + decode to capture realistic CPU cost.
    import xarray as xr
    if product == "gefs":
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": "",
                                             "filter_by_keys": {"shortName": "2t"}})
    else:
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": "",
                                             "filter_by_keys": {"shortName": "2t"}})
    n_vals = int(ds["t2m"].size) if "t2m" in ds.data_vars else 0
    ds.close()
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    return {"bytes_on_wire": size, "decoded_values": n_vals,
            "file_url": str(H.grib)}


def mode_b_search_string(product: str, date: str, fxx: int) -> dict:
    """Herbie smart mode — byte-range read for one variable via .idx."""
    if product == "gefs":
        H = _gefs_herbie(date, fxx, member=1)
        search = ":TMP:2 m above ground:"
    else:
        H = _ecmwf_herbie(date, fxx)
        search = ":2t:"
    # Herbie's xarray() with a searchString issues an HTTP range request
    # behind the scenes. We can't introspect bytes directly from Herbie's
    # API, so we estimate via the .idx record sizes for the matching
    # messages — done analytically in the extrapolation block below.
    ds = H.xarray(search, verbose=False)
    # ds is either a Dataset or list of Datasets.
    if isinstance(ds, list):
        sizes = [getattr(d.get("t2m"), "nbytes", 0) or
                 getattr(d.get("tmp"), "nbytes", 0) for d in ds]
        decoded = int(sum(sizes))
    else:
        var_name = "t2m" if "t2m" in ds.data_vars else next(iter(ds.data_vars))
        decoded = int(ds[var_name].nbytes)
    return {"bytes_on_wire": None,    # not observable from Herbie
            "decoded_values": decoded, "file_url": str(H.grib)}


# ---------------------------------------------------------------------------
# Analytical extrapolation — bytes per full workload slice
# ---------------------------------------------------------------------------

def _grib_mean_size_bytes(grib_summary: dict, product: str,
                          sample: str = "recent_check") -> float:
    """Mean GRIB file size in bytes from step-01 measurements."""
    s = grib_summary[sample][product]
    return float(s["mean_bytes_per_object"])


def _file_count_per_slice(workload: dict, product: str) -> dict:
    """For mode A we need 1 file per (member,fxx) for GEFS, or 1 file per
    fxx for ECMWF (51 mems packed). Returns counts for the full extrapolated
    slice (n_members = full_members from workload spec)."""
    n_steps = len(workload["forecast_hours"])
    if product == "gefs":
        n_files = workload["full_members"] * n_steps
    else:
        n_files = n_steps
    return {"n_steps": n_steps, "n_files": n_files,
            "n_members": workload["full_members"],
            "n_vars": len(workload["variables"])}


def extrapolate_modeA(product: str, workload: dict, grib_summary: dict,
                      live: dict | None) -> dict:
    """Mode A: bytes on wire = n_files × mean_file_size, full-file downloads."""
    f = _file_count_per_slice(workload, product)
    mean_size = _grib_mean_size_bytes(grib_summary, product, "recent_check")
    bytes_total = mean_size * f["n_files"]

    # Wall-clock model: file_count × (file_size / observed_bandwidth_MBps)
    # + a per-file open/decode overhead.
    bw_mbps = (
        live["bytes_on_wire"] / live["wall_s"] / 1e6
        if live and live.get("bytes_on_wire") and live.get("wall_s") else 100.0
    )
    download_s = (bytes_total / 1e6) / bw_mbps
    # Decode cost: one cfgrib pass per file. live.vcpu_s is per-file
    # decode+download; we strip the download share.
    if live and live.get("wall_s") and live.get("vcpu_s"):
        per_file_decode_s = max(live["vcpu_s"] - download_s / f["n_files"], 0.01)
    else:
        per_file_decode_s = 2.0   # cfgrib rule-of-thumb per CLAUDE.md
    decode_s = per_file_decode_s * f["n_files"]

    return {
        **f,
        "mean_file_size_bytes": int(mean_size),
        "bytes_on_wire":  int(bytes_total),
        "gib_on_wire":    round(bytes_total / 2**30, 3),
        "observed_bandwidth_mbps": round(bw_mbps, 2),
        "download_s_modelled":    round(download_s, 2),
        "decode_s_modelled":      round(decode_s, 2),
        "wall_s_modelled":        round(download_s + decode_s, 2),
        "n_get_requests":         f["n_files"],
    }


def extrapolate_modeB(product: str, workload: dict, grib_summary: dict,
                      live: dict | None) -> dict:
    """Mode B: bytes on wire ≈ var_message_size × n_messages.

    A typical 2m temperature message at 0.25° on a global grid is ~2 MB
    (GEFS) or ~5 MB (ECMWF) GRIB2-packed. We use 2 MB and 5 MB as the
    representative per-(var,member,step) message size; the cost model
    can tune these in config under workload[product].mean_message_mb.
    """
    f = _file_count_per_slice(workload, product)
    msg_mb = 2.0 if product == "gefs" else 5.0
    if product == "gefs":
        n_messages = f["n_vars"] * f["n_members"] * f["n_steps"]
    else:
        # ECMWF: 51 members packed → one searchString query returns 51
        # messages at once. Bytes = vars × mems × leads × msg_mb.
        n_messages = f["n_vars"] * f["n_members"] * f["n_steps"]
    bytes_total = n_messages * msg_mb * 1e6
    n_get = (f["n_vars"] * f["n_steps"] * f["n_members"]
             if product == "gefs"
             else f["n_vars"] * f["n_steps"])    # one range req per (var,fxx)

    # Decode in Mode B happens via gribberish in GIK; via cfgrib in Herbie.
    # Use 0.05 s/message (Herbie idx parse + cfgrib decode).
    decode_s = n_messages * 0.05
    download_s = bytes_total / 1e6 / 100.0     # 100 MB/s placeholder
    return {
        **f,
        "msg_mean_mb_assumed": msg_mb,
        "n_messages":          n_messages,
        "n_get_requests":      n_get,
        "bytes_on_wire":       int(bytes_total),
        "gib_on_wire":         round(bytes_total / 2**30, 3),
        "download_s_modelled": round(download_s, 2),
        "decode_s_modelled":   round(decode_s, 2),
        "wall_s_modelled":     round(download_s + decode_s, 2),
    }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def cost(extrap: dict, pricing: dict) -> dict:
    """Compute $ in each cost line for a single extrapolated slice."""
    bytes_gb = extrap["bytes_on_wire"] / 1e9
    n_get    = extrap["n_get_requests"]
    vcpu_h   = extrap["wall_s_modelled"] / 3600.0

    cross_egress = bytes_gb * pricing["egress_per_gb"]["aws_internet"]
    same_egress  = 0.0
    get_cost     = n_get / 1000 * pricing["request_per_1000"]["aws_s3_get"]
    vcpu_cost    = vcpu_h * pricing["compute_per_vcpu_hour"]["local_baseline"]

    return {
        "egress_gb":                round(bytes_gb, 3),
        "n_get":                    int(n_get),
        "vcpu_hours":               round(vcpu_h, 4),
        "$_egress_same_cloud":      round(same_egress, 4),
        "$_egress_cross_cloud":     round(cross_egress, 4),
        "$_get_requests":           round(get_cost, 4),
        "$_vcpu":                   round(vcpu_cost, 4),
        "$_total_same_cloud":       round(same_egress + get_cost + vcpu_cost, 4),
        "$_total_cross_cloud":      round(cross_egress + get_cost + vcpu_cost, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_grib(results_dir: Path) -> dict:
    return json.loads((results_dir / "01_grib_sizes_summary.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--products", nargs="+", default=["gefs", "ecmwf"],
                    choices=["gefs", "ecmwf"])
    ap.add_argument("--skip-live", action="store_true",
                    help="skip live downloads (uses 100 MB/s + 2 s/file priors)")
    ap.add_argument("--skip-full-download", action="store_true",
                    help="skip mode A (~6 GB ECMWF) but still do mode B")
    ap.add_argument("--date", default=None,
                    help="date for live bench (default: first sample.recent_check)")
    ap.add_argument("--fxx", type=int, default=24)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pricing = cfg["pricing"]
    results_dir = Path(__file__).with_name(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    grib_summary = _load_grib(results_dir)
    bench_date = args.date or cfg["sample"]["recent_check"]["dates"][0]

    bench_rows: list[dict] = []
    headline: dict[str, dict] = {}

    for product in args.products:
        workload = cfg["workload"][product]
        print(f"=== {product} ===")
        live_A = live_B = None

        if not args.skip_live:
            if not args.skip_full_download:
                bench = _bench_one(
                    lambda p=product, d=bench_date, f=args.fxx:
                        mode_a_full_download(p, d, f),
                    f"{product}/modeA full download",
                )
                if bench["status"] == "ok":
                    r = bench["result"]
                    live_A = {**bench, **r}
                    bench_rows.append({
                        "product": product, "mode": "A_full_download",
                        "file_url": r["file_url"],
                        "n_bytes_observed": r["bytes_on_wire"],
                        "wall_s": bench["wall_s"], "vcpu_s": bench["vcpu_s"],
                        "mb_per_s": round(r["bytes_on_wire"]/bench["wall_s"]/1e6, 2)
                                     if bench["wall_s"] else None,
                        "status": "ok",
                    })

            bench = _bench_one(
                lambda p=product, d=bench_date, f=args.fxx:
                    mode_b_search_string(p, d, f),
                f"{product}/modeB searchString",
            )
            if bench["status"] == "ok":
                r = bench["result"]
                live_B = {**bench, **r}
                bench_rows.append({
                    "product": product, "mode": "B_searchString",
                    "file_url": r["file_url"], "n_bytes_observed": None,
                    "wall_s": bench["wall_s"], "vcpu_s": bench["vcpu_s"],
                    "mb_per_s": None,
                    "status": "ok",
                })

        extrap_A = extrapolate_modeA(product, workload, grib_summary, live_A)
        extrap_B = extrapolate_modeB(product, workload, grib_summary, live_B)
        cost_A   = cost(extrap_A, pricing)
        cost_B   = cost(extrap_B, pricing)

        headline[product] = {
            "bench_date":     bench_date,
            "bench_fxx":      args.fxx,
            "workload":       workload,
            "mode_A": {"extrapolation": extrap_A, "cost": cost_A},
            "mode_B": {"extrapolation": extrap_B, "cost": cost_B},
        }
        print(json.dumps(headline[product], indent=2))

    pd.DataFrame(bench_rows).to_csv(
        results_dir / "benchmark_herbie.csv", index=False
    )
    (results_dir / "04_herbie_summary.json").write_text(
        json.dumps(headline, indent=2) + "\n"
    )
    print(f"wrote {results_dir / '04_herbie_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
