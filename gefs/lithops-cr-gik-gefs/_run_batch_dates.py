#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lithops",
#     "google-cloud-storage",
#     "google-cloud-pubsub",
#     "google-api-python-client",
#     "google-auth",
#     "httplib2",
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "requests",
#     "kerchunk",
#     "xarray",
#     "cfgrib",
#     "eccodes",
#     "h5py",
#     "dask",
#     "distributed",
#     "zarr",
# ]
# ///
"""Helper: run a list of dates through run_with_lithops. Called by run_backfill_single_run.sh."""
import os, sys, time
sys.path.insert(0, '.')
from run_lithops_gefs import run_with_lithops, print_summary

dates = sys.argv[1].split()
max_workers = int(sys.argv[2])
run = sys.argv[3] if len(sys.argv) > 3 else '00'

print(f"Dates to process ({len(dates)}): {', '.join(dates)}")
print(f"Run: {run}z  Max workers: {max_workers}")
print("=" * 70)

start = time.time()
results = run_with_lithops(dates, run=run, max_workers=max_workers)
print_summary(dates, results, time.time() - start)

# Lithops keeps background threads alive after print_summary; force exit so the
# bash chain can move on to the next batch.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
