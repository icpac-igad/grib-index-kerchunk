#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Random-date pipeline test: GIK → cGAN total-precipitation streaming.

Picks one random date per month from January 2024 to January 2026 (25 months),
runs the GIK three-stage pipeline (template fast-path) with GCS upload, then
streams only total precipitation (tp) via stream_cgan_variables.py and saves
the NetCDF output into random_run_test/.

All stdout/stderr is tee'd to random_run_test/run_log.txt.

Usage:
    uv run run_random_date_test.py
    uv run run_random_date_test.py --dry-run          # print dates only
    uv run run_random_date_test.py --max-members 3    # fast test with 3 members
"""

import argparse
import calendar
import logging
import os
import random
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "random_run_test"
LOG_FILE   = OUTPUT_DIR / "run_log.txt"

# Month range: Jan 2024 → Jan 2026 (inclusive)
START_YEAR, START_MONTH = 2024, 1
END_YEAR,   END_MONTH   = 2026, 1


# ── Helpers ───────────────────────────────────────────────────────────

def generate_random_dates(seed: int = 42) -> list[date]:
    """Return one random date per month from START to END (inclusive)."""
    rng = random.Random(seed)
    dates = []
    year, month = START_YEAR, START_MONTH
    while (year, month) <= (END_YEAR, END_MONTH):
        last_day = calendar.monthrange(year, month)[1]
        day = rng.randint(1, last_day)
        dates.append(date(year, month, day))
        # advance
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def setup_logging() -> logging.Logger:
    """Create logger that writes to both console and LOG_FILE."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("random_run_test")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def run_cmd(cmd: list[str], logger: logging.Logger, label: str) -> bool:
    """Run a subprocess, stream output to logger, return success bool."""
    logger.info(f"[{label}] $ {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(SCRIPT_DIR),
        )
        for line in proc.stdout:
            logger.info(f"[{label}] {line.rstrip()}")
        proc.wait()
        elapsed = time.time() - t0
        if proc.returncode == 0:
            logger.info(f"[{label}] completed OK in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"[{label}] FAILED (exit {proc.returncode}) after {elapsed:.1f}s")
            return False
    except Exception as exc:
        logger.error(f"[{label}] exception: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Random-date GIK + tp-only cGAN streaming test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the selected dates and exit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for date selection (default: 42)")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap ensemble members (passed to both scripts)")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour (default: 00)")
    parser.add_argument("--parallel-workers", type=int, default=8,
                        help="Parallel workers for GIK Stage 2 (default: 8)")
    parser.add_argument("--parallel-fetches", type=int, default=8,
                        help="Parallel S3 fetches for streaming (default: 8)")
    args = parser.parse_args()

    dates = generate_random_dates(seed=args.seed)

    if args.dry_run:
        print(f"Random dates ({len(dates)} months):")
        for d in dates:
            print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%B %Y')})")
        return

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Random-date GIK + TP streaming pipeline")
    logger.info("=" * 70)
    logger.info(f"Months: {len(dates)}  |  Run: {args.run}Z  |  Seed: {args.seed}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Log file:   {LOG_FILE}")
    if args.max_members:
        logger.info(f"Max members: {args.max_members}")
    logger.info("")

    overall_start = time.time()
    results = {}  # date_str -> {"gik": bool, "stream": bool}

    for idx, d in enumerate(dates, 1):
        date_str = d.strftime("%Y%m%d")
        month_label = d.strftime("%B %Y")
        logger.info("=" * 70)
        logger.info(f"[{idx}/{len(dates)}]  {month_label}  —  date={date_str}")
        logger.info("=" * 70)

        results[date_str] = {"gik": False, "stream": False}

        # ── Step A: GIK three-stage pipeline (template fast-path + GCS upload)
        gik_cmd = [
            "uv", "run", "run_ecmwf_tutorial.py",
            "--date", date_str,
            "--run", args.run,
            "--skip-grib-scan",
            "--upload-gcs",
            "--parallel-workers", str(args.parallel_workers),
        ]
        if args.max_members:
            gik_cmd += ["--max-members", str(args.max_members)]

        gik_ok = run_cmd(gik_cmd, logger, f"GIK {date_str}")
        results[date_str]["gik"] = gik_ok

        if not gik_ok:
            logger.warning(f"GIK failed for {date_str}, skipping streaming step")
            continue

        # ── Step B: Stream tp-only to NetCDF
        stream_cmd = [
            "uv", "run", "stream_cgan_variables.py",
            "--date", date_str,
            "--run", args.run,
            "--variables", "tp",
            "--output-dir", str(OUTPUT_DIR),
            "--parallel-fetches", str(args.parallel_fetches),
        ]
        if args.max_members:
            stream_cmd += ["--max-members", str(args.max_members)]

        stream_ok = run_cmd(stream_cmd, logger, f"STREAM {date_str}")
        results[date_str]["stream"] = stream_ok

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Date':<12} {'Month':<16} {'GIK':<10} {'Stream TP':<10}")
    logger.info("-" * 48)
    gik_pass = stream_pass = 0
    for d in dates:
        ds = d.strftime("%Y%m%d")
        r = results[ds]
        g = "OK" if r["gik"] else "FAIL"
        s = "OK" if r["stream"] else "FAIL"
        logger.info(f"{ds:<12} {d.strftime('%b %Y'):<16} {g:<10} {s:<10}")
        gik_pass += r["gik"]
        stream_pass += r["stream"]

    logger.info("-" * 48)
    logger.info(f"GIK:    {gik_pass}/{len(dates)} passed")
    logger.info(f"Stream: {stream_pass}/{len(dates)} passed")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # List NC files produced
    nc_files = sorted(OUTPUT_DIR.glob("*.nc"))
    if nc_files:
        logger.info(f"\nNetCDF files in {OUTPUT_DIR}/:")
        for f in nc_files:
            size_kb = f.stat().st_size / 1024
            logger.info(f"  {f.name}  ({size_kb:.1f} KB)")
    else:
        logger.warning("No NetCDF files produced.")

    logger.info(f"\nFull log: {LOG_FILE}")


if __name__ == "__main__":
    main()
