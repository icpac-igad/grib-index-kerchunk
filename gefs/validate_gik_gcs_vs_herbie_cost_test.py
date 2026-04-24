#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "google-cloud-storage",
#     "pyarrow",
#     "netcdf4",
#     "gribberish",
#     "cfgrib",
#     "eccodes",
#     "herbie-data",
#     "matplotlib",
#     "cartopy",
#     "scipy",
#     "requests",
# ]
# ///
"""
Validate pre-built GIK parquets (from GCS) against Herbie for the cost-test
random dates in cost_test_random_dates.json.

Instead of running the 3-stage GIK pipeline on-the-fly (as in
validate_gik_vs_herbie_2022_2026.py), this script:
  1) Loads dates from cost_test_random_dates.json (50 dates, 00z)
  2) For each date, downloads per-member parquet files from GCS
       gs://gik-gefs-aws-tf/run_par_gefs/{YYYY}/{MM}/{DATE}/00z/{DATE}00z-gep{NN}.parquet
     (Auth: SA key path from lithops-cr-gik-gefs/lithops_config.yaml,
      falling back to GOOGLE_APPLICATION_CREDENTIALS / ADC.)
  3) Reads each parquet -> zarr store dict -> streams APCP bytes from S3
     via gribberish (ensemble mean & std)
  4) Fetches APCP via Herbie for the same members
  5) Compares at T+48h: correlation, RMSE, MAE, relative diff
  6) Saves per-date NetCDFs (cached), comparison plots, scatter plot, JSON stats

Output directory: validation_gik_gcs_vs_herbie_cost_test/

Usage:
    uv run validate_gik_gcs_vs_herbie_cost_test.py
    uv run validate_gik_gcs_vs_herbie_cost_test.py --dry-run
    uv run validate_gik_gcs_vs_herbie_cost_test.py --max-members 5
    uv run validate_gik_gcs_vs_herbie_cost_test.py --dates 20220121,20240101
    uv run validate_gik_gcs_vs_herbie_cost_test.py --skip-herbie
    uv run validate_gik_gcs_vs_herbie_cost_test.py --skip-gik
"""

import argparse
import base64
import gc
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import fsspec

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse helpers from the on-the-fly validator
from validate_gik_vs_herbie_2022_2026 import (
    GEFS_GRID_SHAPE, GEFS_LATS, GEFS_LONS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    LAT_INDICES, LON_INDICES, ICPAC_LATS, ICPAC_LONS,
    TARGET_STEPS, STEP_INDEX, EA_EXTENT,
    GRIBBERISH_AVAILABLE,
    fetch_grib_bytes, decode_grib_bytes, subset_icpac,
    stream_tp_from_zstore,
    fetch_herbie_tp,
    compute_metrics, save_tp_netcdf,
    plot_comparison, plot_scatter_all,
    setup_logging,
)

# ── GCS / Lithops configuration ─────────────────────────────────────
LITHOPS_DIR = SCRIPT_DIR / "lithops-cr-gik-gefs"
LITHOPS_CONFIG = LITHOPS_DIR / "lithops_config.yaml"
GCS_BUCKET = "gik-gefs-aws-tf"
GCS_PARQUET_PREFIX = "run_par_gefs"

# ── Inputs / outputs ────────────────────────────────────────────────
DATES_JSON = SCRIPT_DIR / "cost_test_random_dates.json"
OUTPUT_DIR = SCRIPT_DIR / "validation_gik_gcs_vs_herbie_cost_test"
PARQUET_CACHE = OUTPUT_DIR / "parquet_cache"


# ── Config / credentials ────────────────────────────────────────────

def load_lithops_config(path: Path) -> dict:
    """Minimal YAML-free loader; only fields we need."""
    cfg: Dict[str, Optional[str]] = {
        "project_name": None,
        "credentials_path": None,
        "bucket": None,
    }
    if not path.exists():
        return cfg
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if ":" not in line or line.lstrip().startswith("-"):
            continue
        key, _, val = line.strip().partition(":")
        val = val.strip().strip('"').strip("'")
        if key == "project_name" and val:
            cfg["project_name"] = val
        elif key == "credentials_path" and val:
            cfg["credentials_path"] = val
        elif key == "bucket" and val:
            cfg["bucket"] = val
    return cfg


def resolve_credentials(logger, override: Optional[Path] = None
                        ) -> Tuple[Optional[str], Optional[str]]:
    """Return (credentials_path, project_name).

    Priority: --sa-key override > lithops_config.yaml path > coiled-data.json
    (in gefs/) > GOOGLE_APPLICATION_CREDENTIALS > ADC.
    The project in the SA key JSON wins over lithops_config.yaml's project.
    """
    cfg = load_lithops_config(LITHOPS_CONFIG)
    config_project = cfg.get("project_name")

    candidates: List[Path] = []
    if override is not None:
        candidates.append(override)
    cred = cfg.get("credentials_path")
    if cred:
        candidates.append(
            Path(cred) if os.path.isabs(cred) else (LITHOPS_DIR / cred).resolve()
        )
    candidates.append(SCRIPT_DIR / "coiled-data.json")

    for cand in candidates:
        if cand and cand.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cand)
            try:
                key_project = json.loads(cand.read_text()).get("project_id")
            except Exception:
                key_project = None
            project = key_project or config_project
            logger.info(f"  Using SA key: {cand}  (project={project})")
            return str(cand), project

    env_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_cred and Path(env_cred).exists():
        logger.info(f"  Using GOOGLE_APPLICATION_CREDENTIALS: {env_cred}")
        return env_cred, config_project
    logger.warning("  No SA key found — relying on Application Default Credentials")
    return None, config_project


def make_gcs_fs(project: Optional[str]):
    import gcsfs
    kwargs = {}
    if project:
        kwargs["project"] = project
    return gcsfs.GCSFileSystem(**kwargs)


# ── Date loading ─────────────────────────────────────────────────────

def load_cost_test_dates(path: Path) -> List[str]:
    obj = json.loads(path.read_text())
    return list(obj["dates"])


# ── Parquet -> zarr store dict ──────────────────────────────────────

def _parquet_to_zstore(parquet_path: Path) -> dict:
    """Convert a GIK per-member parquet into the dict expected by
    stream_tp_from_zstore.

    Supports both layouts:
      (a) Kerchunk LazyReferenceMapper: one row with key=='refs' whose value
          is a Python-repr'd dict blob of {zarr_path: value} — this is what
          the Lithops pipeline writes to GCS.
      (b) Legacy flat key/value: one row per zarr path.
    """
    import ast
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    keys = set(df["key"])
    if "refs" in keys and len(df) <= 3:
        blob = df.loc[df["key"] == "refs", "value"].iloc[0]
        if isinstance(blob, (bytes, bytearray)):
            blob = blob.decode("utf-8")
        return ast.literal_eval(blob)

    zstore: Dict[str, object] = {}
    for _, row in df.iterrows():
        key = row["key"]
        value = row["value"]
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                pass
        if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
            try:
                value = json.loads(value)
            except Exception:
                pass
        zstore[key] = value
    return zstore


# ── GCS download ─────────────────────────────────────────────────────

def gcs_prefix_for_date(date_str: str, run: str = "00") -> str:
    y, m = date_str[:4], date_str[4:6]
    return f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{y}/{m}/{date_str}/{run}z"


def download_member_parquets(fs, date_str: str, run: str,
                             members: List[str],
                             cache_dir: Path,
                             logger) -> Dict[str, Path]:
    """Download per-member parquets to cache_dir. Returns {member: local_path}."""
    prefix = gcs_prefix_for_date(date_str, run)
    try:
        remote_items = fs.ls(prefix, detail=False)
    except Exception as e:
        logger.error(f"  GCS ls failed for {prefix}: {e}")
        return {}

    remote_by_name: Dict[str, str] = {}
    for path in remote_items:
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".parquet"):
            remote_by_name[name] = path

    date_cache = cache_dir / date_str / f"{run}z"
    date_cache.mkdir(parents=True, exist_ok=True)

    found: Dict[str, Path] = {}
    missing: List[str] = []
    for member in members:
        fname = f"{date_str}{run}z-{member}.parquet"
        local = date_cache / fname
        remote = remote_by_name.get(fname)
        if remote is None:
            missing.append(member)
            continue
        if not local.exists() or local.stat().st_size == 0:
            try:
                fs.get(remote, str(local))
            except Exception as e:
                logger.warning(f"  download {fname} failed: {e}")
                missing.append(member)
                continue
        found[member] = local

    if missing:
        logger.warning(f"  missing {len(missing)}/{len(members)} parquets: {missing[:5]}"
                       f"{'...' if len(missing) > 5 else ''}")
    return found


# ── GIK ensemble streaming from GCS parquets ────────────────────────

def stream_gik_tp_from_gcs(date_str: str, run: str, step_hours: List[int],
                           fs, cache_dir: Path,
                           max_members: int = 30,
                           logger=None) -> Tuple[Optional[np.ndarray],
                                                 Optional[np.ndarray],
                                                 int, dict]:
    """Download parquets from GCS, build zstores, stream TP via gribberish."""
    log = logger.info if logger else print

    members = [f"gep{i:02d}" for i in range(1, max_members + 1)]
    n_lats, n_lons = len(ICPAC_LATS), len(ICPAC_LONS)

    t_dl = time.time()
    member_files = download_member_parquets(fs, date_str, run, members,
                                            cache_dir, logger)
    download_secs = time.time() - t_dl
    log(f"    GIK: {len(member_files)}/{len(members)} parquets ready in {download_secs:.1f}s")

    if not member_files:
        return None, None, 0, {"download_total": download_secs}

    all_parse = []
    all_stream = []
    member_data: List[np.ndarray] = []

    for i, member in enumerate(members, 1):
        pfile = member_files.get(member)
        if pfile is None:
            continue
        try:
            t0 = time.time()
            zstore = _parquet_to_zstore(pfile)
            parse_secs = time.time() - t0
            all_parse.append(parse_secs)

            tp, stream_secs = stream_tp_from_zstore(zstore, step_hours)
            all_stream.append(stream_secs)

            if np.count_nonzero(~np.isnan(tp)) == 0:
                log(f"    GIK: {member} no valid TP, skipping")
                continue
            member_data.append(tp)
            if i % 5 == 0:
                log(f"    GIK: {i}/{len(members)} members streamed")
        except Exception as e:
            log(f"    GIK: {member} FAILED: {e}")

    if not member_data:
        return None, None, 0, {"download_total": download_secs}

    stacked = np.stack(member_data, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)

    timings = {
        "download_total": round(download_secs, 1),
        "parquet_parse_total": round(sum(all_parse), 1),
        "parquet_parse_per_member": round(np.mean(all_parse), 2) if all_parse else 0,
        "stream_total": round(sum(all_stream), 1),
        "stream_per_member": round(np.mean(all_stream), 2) if all_stream else 0,
    }
    log(f"    GIK timing: dl={timings['download_total']}s "
        f"parse={timings['parquet_parse_total']}s "
        f"stream={timings['stream_total']}s "
        f"(per member parse={timings['parquet_parse_per_member']}s "
        f"stream={timings['stream_per_member']}s)")
    gc.collect()
    return mean, std, len(member_data), timings


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate GCS-hosted GIK parquets vs Herbie for cost-test dates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-members", type=int, default=30)
    parser.add_argument("--dates", type=str, default=None,
                        help="Override: comma-separated YYYYMMDD dates")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour (default 00)")
    parser.add_argument("--skip-herbie", action="store_true")
    parser.add_argument("--skip-gik", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dates-json", type=str, default=str(DATES_JSON))
    parser.add_argument("--sa-key", type=str, default=None,
                        help="Override SA key JSON path (else auto-discovered)")
    parser.add_argument("--first", type=int, default=None,
                        help="Take only the first N dates from the JSON file")
    parser.add_argument("--cleanup", action="store_true",
                        help="After each date's PNG is written, delete its "
                             "parquet cache and both NC files to save disk")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cache_dir = out_dir / "parquet_cache"

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        dates = load_cost_test_dates(Path(args.dates_json))
    if args.first is not None:
        dates = dates[: args.first]

    if args.dry_run:
        print(f"Dates from {args.dates_json}: {len(dates)}")
        for d in dates:
            print(f"  {d}")
        print(f"\nRun: {args.run}z  max_members={args.max_members}")
        print(f"GCS: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/YYYY/MM/DATE/{args.run}z/")
        return

    logger = setup_logging(out_dir)
    logger.info("=" * 70)
    logger.info("GIK (GCS parquets) vs Herbie — cost-test dates")
    logger.info("=" * 70)
    logger.info(f"Dates: {len(dates)}  |  Steps: {TARGET_STEPS}  |  Run: {args.run}z")
    logger.info(f"Max members: {args.max_members}")
    logger.info(f"Gribberish available: {GRIBBERISH_AVAILABLE}")
    logger.info(f"GCS source: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")
    logger.info(f"Parquet cache: {cache_dir}")
    logger.info(f"Output: {out_dir}")

    fs = None
    if not args.skip_gik:
        override = Path(args.sa_key) if args.sa_key else None
        cred_path, project = resolve_credentials(logger, override=override)
        try:
            fs = make_gcs_fs(project)
            fs.ls(f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}", detail=False)
            logger.info("  GCS auth OK")
        except Exception as e:
            logger.error(f"  GCS auth/list failed: {e}")
            logger.error("  Set credentials_path in lithops_config.yaml or "
                         "export GOOGLE_APPLICATION_CREDENTIALS.")
            sys.exit(2)

    overall_t0 = time.time()
    all_results = []
    all_gik_means = []
    all_herbie_means = []
    all_dates_ok = []

    for idx, ds in enumerate(dates, 1):
        logger.info("=" * 70)
        try:
            month_label = datetime.strptime(ds, "%Y%m%d").strftime("%B %Y")
        except Exception:
            month_label = ds
        logger.info(f"[{idx}/{len(dates)}]  {month_label}  —  {ds}")
        logger.info("=" * 70)

        result = {"date": ds, "run": f"{args.run}z"}
        gik_nc = out_dir / f"gik_{ds}_tp.nc"
        herbie_nc = out_dir / f"herbie_{ds}_tp.nc"

        # ── GIK streaming (GCS parquets) ──
        gik_mean = gik_std = None
        gik_n = 0
        if gik_nc.exists():
            logger.info("  GIK: loading cached NC")
            with xr.open_dataset(gik_nc) as ds_g:
                gik_mean = ds_g["tp_ensemble_mean"].isel(time=0).values
                gik_std = ds_g["tp_ensemble_standard_deviation"].isel(time=0).values
                gik_n = int(ds_g.attrs.get("n_ensemble_members", 0))
        elif not args.skip_gik:
            t0 = time.time()
            gik_mean, gik_std, gik_n, gik_timings = stream_gik_tp_from_gcs(
                ds, args.run, TARGET_STEPS,
                fs=fs, cache_dir=cache_dir,
                max_members=args.max_members,
                logger=logger,
            )
            elapsed = time.time() - t0
            logger.info(f"  GIK done: {gik_n} members in {elapsed:.1f}s")
            if gik_timings:
                for k, v in gik_timings.items():
                    result[f"gik_{k}_s"] = v
                result["gik_total_s"] = round(elapsed, 1)
            if gik_mean is not None:
                save_tp_netcdf(gik_nc, gik_mean, gik_std, TARGET_STEPS,
                               ds, gik_n, "GIK parquets from GCS + gribberish")
                logger.info(f"  GIK saved: {gik_nc}")
            else:
                logger.warning(f"  GIK: no data for {ds}")
        else:
            logger.info("  GIK: skipped (no cached NC)")

        # ── Herbie fetching ──
        herbie_mean = herbie_std = None
        herbie_n = 0
        if herbie_nc.exists():
            logger.info("  Herbie: loading cached NC")
            with xr.open_dataset(herbie_nc) as ds_h:
                herbie_mean = ds_h["tp_ensemble_mean"].isel(time=0).values
                herbie_std = ds_h["tp_ensemble_standard_deviation"].isel(time=0).values
                herbie_n = int(ds_h.attrs.get("n_ensemble_members", 0))
        elif not args.skip_herbie:
            t0 = time.time()
            herbie_mean, herbie_std, herbie_n, herbie_timings = fetch_herbie_tp(
                ds, TARGET_STEPS,
                max_members=args.max_members,
                logger=logger,
            )
            elapsed = time.time() - t0
            logger.info(f"  Herbie done: {herbie_n} members in {elapsed:.1f}s")
            if herbie_timings:
                result["herbie_total_s"] = round(herbie_timings["total"], 1)
                result["herbie_per_member_avg_s"] = round(herbie_timings["per_member_avg"], 2)
                result["herbie_n_fetches"] = herbie_timings["n_fetches"]
            if herbie_mean is not None:
                save_tp_netcdf(herbie_nc, herbie_mean, herbie_std, TARGET_STEPS,
                               ds, herbie_n, "Herbie GEFS atmos.25")
                logger.info(f"  Herbie saved: {herbie_nc}")
            else:
                logger.warning(f"  Herbie: no data for {ds}")
        else:
            logger.info("  Herbie: skipped (no cached NC)")

        # ── Compare ──
        if gik_mean is not None and herbie_mean is not None:
            min_steps = min(gik_mean.shape[0], herbie_mean.shape[0])
            effective_step_idx = min(STEP_INDEX, min_steps - 1)

            for label, g, h in [("mean", gik_mean, herbie_mean),
                                ("spread", gik_std, herbie_std)]:
                m = compute_metrics(g, h, label, step_idx=effective_step_idx)
                result.update(m)
                corr = m.get(f"{label}_corr", float("nan"))
                rmse = m.get(f"{label}_rmse", float("nan"))
                logger.info(f"  {label}: r={corr:.8f}  RMSE={rmse:.2e}")

            png_ok = False
            try:
                plot_comparison(ds, gik_mean, herbie_mean, gik_std, herbie_std,
                                result, out_dir, step_idx=effective_step_idx)
                logger.info(f"  Plot saved: {out_dir / f'compare_{ds}.png'}")
                png_ok = True
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

            all_gik_means.append(gik_mean)
            all_herbie_means.append(herbie_mean)
            all_dates_ok.append(ds)

            if args.cleanup and png_ok:
                import shutil
                removed = []
                for fp in (gik_nc, herbie_nc):
                    try:
                        if fp.exists():
                            fp.unlink()
                            removed.append(fp.name)
                    except Exception as e:
                        logger.warning(f"  cleanup: rm {fp} failed: {e}")
                date_cache = cache_dir / ds
                if date_cache.exists():
                    try:
                        shutil.rmtree(date_cache)
                        removed.append(f"parquet_cache/{ds}/")
                    except Exception as e:
                        logger.warning(f"  cleanup: rmtree {date_cache} failed: {e}")
                if removed:
                    logger.info(f"  cleanup: removed {', '.join(removed)}")
        else:
            result["error"] = "missing data"
            logger.warning("  Skipping comparison — missing data")

        result["gik_members"] = gik_n
        result["herbie_members"] = herbie_n
        all_results.append(result)

    if all_gik_means:
        try:
            plot_scatter_all(all_gik_means, all_herbie_means, all_dates_ok, out_dir)
            logger.info(f"Scatter plot saved: {out_dir / 'scatter_all_dates.png'}")
        except Exception as e:
            logger.warning(f"Scatter plot failed: {e}")

    stats_file = out_dir / "validation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Stats saved: {stats_file}")

    total_time = time.time() - overall_t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Date':<12} {'Mean r':<16} {'Mean RMSE':<14} "
                f"{'Spread r':<16} {'Spread RMSE':<14}")
    logger.info("-" * 72)
    for r in all_results:
        if "error" in r:
            logger.info(f"{r['date']:<12} ERROR: {r.get('error','')}")
            continue
        mc = r.get("mean_corr", float("nan"))
        mr = r.get("mean_rmse", float("nan"))
        sc = r.get("spread_corr", float("nan"))
        sr = r.get("spread_rmse", float("nan"))
        logger.info(f"{r['date']:<12} {mc:<16.8f} {mr:<14.2e} "
                    f"{sc:<16.8f} {sr:<14.2e}")
    logger.info("-" * 72)
    n_ok = sum(1 for r in all_results if "error" not in r)
    logger.info(f"Successful: {n_ok}/{len(dates)} dates")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
