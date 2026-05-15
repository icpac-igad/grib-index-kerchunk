#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "s3fs",
#     "fsspec",
#     "pyyaml",
#     "pandas",
#     "tqdm",
# ]
# ///
"""
01_measure_grib_sizes.py
========================

Anonymous S3 HEAD over every (date, member, step) GRIB object covered by the
sample window in ``config.yaml``. Writes one row per object to
``results/grib_sizes_{gefs,ecmwf}.csv`` plus a small summary JSON.

This is the *primary measurement* underlying every storage and egress number
the cost model later quotes. It deliberately does **not** download any
GRIB bytes — only the content-length is read.

Output schema (``grib_sizes_<product>.csv``)::

    product, date, run, member, step, url, size_bytes

For ECMWF the ``member`` column is the literal string ``"packed"`` because
all 51 ensemble members share one file per timestep; the cost model
amortises the per-file size across ``n_members_in_file`` from config.

Usage
-----
    uv run cost-eval/01_measure_grib_sizes.py
    uv run cost-eval/01_measure_grib_sizes.py --config cost-eval/config.yaml \\
        --products gefs ecmwf --concurrency 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import s3fs
import yaml
from tqdm import tqdm

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"


# ---------------------------------------------------------------------------
# URL enumeration
# ---------------------------------------------------------------------------

def _forecast_hours(spec: dict) -> list[int]:
    """Resolve a forecast-hour spec (either {start,stop,step} or {ranges:[...]})."""
    if "ranges" in spec:
        out: list[int] = []
        for r in spec["ranges"]:
            out.extend(range(r["start"], r["stop"] + 1, r["step"]))
        return out
    return list(range(spec["start"], spec["stop"] + 1, spec["step"]))


def _run_forecast_hours(product_cfg: dict, run: str) -> list[int]:
    """Pick the forecast-hour list for a given run via the run→preset map."""
    preset = product_cfg["runs"][run]
    return _forecast_hours(product_cfg["forecast_hours"][preset])


def gefs_urls(cfg: dict, dates: Iterable[str], runs: Iterable[str]) -> list[dict]:
    bucket = cfg["s3_bucket"]
    tmpl = cfg["path_template"]
    members = cfg["members"]
    rows = []
    for run in runs:
        hours = _run_forecast_hours(cfg, run)
        for date in dates:
            for member in members:
                for step in hours:
                    key = tmpl.format(date=date, run=run, member=member, step=step)
                    rows.append({
                        "product": "gefs",
                        "date": date,
                        "run": run,
                        "member": member,
                        "step": step,
                        "url": f"s3://{bucket}/{key}",
                    })
    return rows


def ecmwf_urls(cfg: dict, dates: Iterable[str], runs: Iterable[str]) -> list[dict]:
    bucket = cfg["s3_bucket"]
    tmpl = cfg["path_template"]
    rows = []
    for run in runs:
        hours = _run_forecast_hours(cfg, run)
        for date in dates:
            for step in hours:
                key = tmpl.format(date=date, run=run, step=step)
                rows.append({
                    "product": "ecmwf",
                    "date": date,
                    "run": run,
                    "member": "packed",
                    "step": step,
                    "url": f"s3://{bucket}/{key}",
                })
    return rows


# ---------------------------------------------------------------------------
# HEAD probe
# ---------------------------------------------------------------------------

def _head_size(fs: s3fs.S3FileSystem, url: str) -> int | None:
    """Return content-length in bytes, or None on 404 / error."""
    try:
        info = fs.info(url)
        return int(info.get("size") or info.get("Size") or 0) or None
    except FileNotFoundError:
        return None
    except Exception as exc:
        # Surface unexpected errors but don't abort the whole run.
        print(f"[warn] {url}: {exc}", file=sys.stderr)
        return None


def probe(rows: list[dict], concurrency: int) -> pd.DataFrame:
    fs = s3fs.S3FileSystem(anon=True, default_block_size=4 * 1024)
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_head_size, fs, r["url"]): r for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="HEAD"):
            r = futures[fut]
            r = dict(r)
            r["size_bytes"] = fut.result()
            out.append(r)
    df = pd.DataFrame(out)
    # Stable ordering for diffability of the CSV.
    df = df.sort_values(["product", "date", "run", "member", "step"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, archive_cfg: dict | None) -> dict:
    """Per-(product, sample) summary suitable for the JSON sidecar.

    A "date" here means a calendar day; the per-date total includes *all* runs
    that were probed (typically 00z, 06z, 12z, 18z), so the resulting figure
    is the daily archive footprint, not a per-cycle one.
    """
    present = df[df["size_bytes"].notna()].copy()
    missing = int(df["size_bytes"].isna().sum())
    runs_probed = sorted(df["run"].unique().tolist())
    n_dates = present["date"].nunique()
    total_bytes = int(present["size_bytes"].sum())
    per_date = present.groupby("date")["size_bytes"].sum() if n_dates else None
    per_run  = present.groupby("run")["size_bytes"].sum().to_dict()
    files_per_run = present.groupby("run").size().to_dict()

    summary = {
        "runs_probed":           runs_probed,
        "n_objects_probed":      int(len(df)),
        "n_objects_present":     int(len(present)),
        "n_objects_missing":     missing,
        "n_dates_sampled":       int(n_dates),
        "dates_sampled":         sorted(df["date"].unique().tolist()),
        "files_per_run":         {k: int(v) for k, v in files_per_run.items()},
        "bytes_per_run":         {k: int(v) for k, v in per_run.items()},
        "total_bytes_sampled":   total_bytes,
        "total_gib_sampled":     round(total_bytes / 2**30, 3),
        "mean_bytes_per_date_all_runs":   float(per_date.mean()) if n_dates else 0.0,
        "mean_bytes_per_object": (
            float(present["size_bytes"].mean()) if len(present) else 0.0
        ),
    }
    if n_dates and archive_cfg is not None:
        bytes_per_date = float(per_date.mean())
        n_full = archive_cfg["n_dates_published"]
        # "n_dates_published" is the calendar-date count in the published
        # archive; multiplying by the all-runs daily mean gives a full-archive
        # estimate that already includes every cycle.
        summary["extrapolated_full_archive_bytes"] = int(bytes_per_date * n_full)
        summary["extrapolated_full_archive_tib"] = round(
            bytes_per_date * n_full / 2**40, 3
        )
        summary["extrapolated_full_archive_pib"] = round(
            bytes_per_date * n_full / 2**50, 4
        )
        summary["projection_one_year_tib"] = round(
            bytes_per_date * 365 / 2**40, 3
        )
        summary["archive_window"] = f"{archive_cfg['start']}..{archive_cfg['end']}"
        summary["archive_n_dates"] = n_full
    return summary


def _probe_and_write(rows: list[dict], concurrency: int, csv_path: Path,
                     archive_cfg: dict | None, label: str) -> dict:
    print(f"[{label}] {len(rows)} objects to HEAD")
    df = probe(rows, concurrency)
    df.to_csv(csv_path, index=False)
    print(f"[{label}] wrote {csv_path}")
    summary = summarise(df, archive_cfg)
    print(f"[{label}] {json.dumps(summary, indent=2)}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--products", nargs="+", default=["gefs", "ecmwf"],
                    choices=["gefs", "ecmwf"])
    ap.add_argument("--concurrency", type=int, default=None,
                    help="override config.output.http_concurrency")
    ap.add_argument("--skip-recent-check", action="store_true",
                    help="omit the recent_check probe")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    concurrency = args.concurrency or cfg["output"]["http_concurrency"]
    results_dir = Path(__file__).with_name(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    headline: dict[str, dict] = {"primary": {}, "recent_check": {}}

    # ----- primary (per-product, HF-overlapping) -----
    for product in args.products:
        product_cfg = cfg["products"][product]
        archive_cfg = cfg["archive"][product]
        runs = list(product_cfg["runs"].keys())   # all 4 cycles
        dates = cfg["sample"]["primary"][product]["dates"]
        rows = (gefs_urls(product_cfg, dates, runs)
                if product == "gefs" else ecmwf_urls(product_cfg, dates, runs))
        csv_path = results_dir / f"grib_sizes_{product}.csv"
        headline["primary"][product] = _probe_and_write(
            rows, concurrency, csv_path, archive_cfg, f"{product}/primary"
        )

    # ----- recent_check (same dates for both products) -----
    if not args.skip_recent_check:
        recent_dates = cfg["sample"]["recent_check"]["dates"]
        for product in args.products:
            product_cfg = cfg["products"][product]
            archive_cfg = cfg["archive"][product]
            runs = list(product_cfg["runs"].keys())
            rows = (gefs_urls(product_cfg, recent_dates, runs)
                    if product == "gefs"
                    else ecmwf_urls(product_cfg, recent_dates, runs))
            csv_path = results_dir / f"grib_sizes_recent_{product}.csv"
            headline["recent_check"][product] = _probe_and_write(
                rows, concurrency, csv_path, archive_cfg,
                f"{product}/recent_check"
            )

    summary_path = results_dir / "01_grib_sizes_summary.json"
    summary_path.write_text(json.dumps(headline, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
