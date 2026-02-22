#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "dask",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "pyarrow",
#     "icechunk",
#     "gribberish",
#     "coiled",
#     "distributed",
# ]
# ///
"""
ECMWF EA Total Precipitation — Materialized Icechunk Store
===========================================================

Creates a materialized Icechunk store for ECMWF IFS ensemble total precipitation
(tp) over the East Africa / ICPAC region from GIK parquet reference files hosted
on HuggingFace (E4DRR/gik-ecmwf-par).

Each parquet contains [url, byte_offset, byte_length] triplets pointing into
remote GRIB files on S3 (ecmwf-forecasts bucket).  Workers fetch the lightweight
parquets from HuggingFace, look up the tp byte-range references for each lead
time, stream the raw GRIB bytes from S3, decode with gribberish, subset to East
Africa, and return numpy arrays.  The coordinator writes results into an Icechunk
store on GCS and commits in batches.

Subcommands:

  init   — Create empty template store (GLAD pattern: compute=False)
  fill   — Populate with real data using Dask/Coiled workers
  verify — Inspect store contents

Usage:
    python ecmwf_ea_tp_icechunk.py init \\
        --start-date 20240301 --end-date 20251231

    python ecmwf_ea_tp_icechunk.py fill \\
        --start-date 20240301 --end-date 20251231 \\
        --n-workers 20 --commit-batch 20

    python ecmwf_ea_tp_icechunk.py verify

Author: ICPAC GIK Team
Date: 2026-02-21
"""

import json
import logging
import os
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ecmwf_ea_tp_icechunk.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_ACCOUNT_FILE = str(SCRIPT_DIR / "coiled-data.json")
GCS_BUCKET = "gik-ecmwf-aws-tf"
GCS_PREFIX = "ecmwf-ea-ic-store"

# HuggingFace parquet source
HF_REPO = "E4DRR/gik-ecmwf-par"
HF_BASE_URL = (
    "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/run_par_ecmwf"
)

# ECMWF forecast lead times (00z run): 0-144h @3h + 150-168h @6h
LEAD_TIME_HOURS_3H = list(range(0, 145, 3))   # 49 steps
LEAD_TIME_HOURS_6H = [150, 156, 162, 168]     # 4 steps
LEAD_TIME_HOURS = LEAD_TIME_HOURS_3H + LEAD_TIME_HOURS_6H  # 53 steps

# Ensemble members: 1 control + 50 ensemble
MEMBER_IDS = ["control"] + [f"ens_{i:02d}" for i in range(1, 51)]  # 51 members

# ECMWF global grid
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# East Africa / ICPAC bounding box
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Precompute EA spatial indices
_lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
_lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
EA_LATS = ECMWF_LATS[LAT_INDICES[0] : LAT_INDICES[-1] + 1]
EA_LONS = ECMWF_LONS[LON_INDICES[0] : LON_INDICES[-1] + 1]
N_LAT = len(EA_LATS)  # 157
N_LON = len(EA_LONS)  # 145
N_STEPS = len(LEAD_TIME_HOURS)  # 53
N_MEMBERS = len(MEMBER_IDS)  # 51

# Chunk shape: one init_date, one member, all lead times and spatial dims
CHUNK_SHAPE = (1, 1, N_STEPS, N_LAT, N_LON)


# ─── Date helpers ───────────────────────────────────────────────────────────


def build_date_list(start_date: str, end_date: str) -> List[str]:
    """Build list of YYYYMMDD date strings between start and end (inclusive)."""
    dates = pd.date_range(start_date, end_date, freq="D")
    return [d.strftime("%Y%m%d") for d in dates]


def member_key_from_id(member_id: str) -> str:
    """Convert member filename ID to parquet key format.

    'control'  -> 'control'
    'ens_01'   -> 'ens01'
    """
    return member_id.replace("_", "")


# ─── Phase 1: init ─────────────────────────────────────────────────────────


def init_store(args):
    """Create empty template Icechunk store.

    Follows the GLAD pattern: write structure only with compute=False.
    """
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating ECMWF EA tp Icechunk store")
    logger.info("=" * 60)
    start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    n_dates = len(dates)
    logger.info(f"  Dates: {n_dates} ({dates[0]} to {dates[-1]})")
    logger.info(f"  Members: {N_MEMBERS}")
    logger.info(f"  Lead times: {N_STEPS}")
    logger.info(f"  Spatial: {N_LAT} lat x {N_LON} lon")

    # Coordinate arrays
    init_date = pd.to_datetime(dates).values.astype("datetime64[ns]")
    lead_time = np.array(LEAD_TIME_HOURS, dtype=np.int32)
    member = np.array(MEMBER_IDS, dtype="U10")

    shape = (n_dates, N_MEMBERS, N_STEPS, N_LAT, N_LON)
    size_gb = np.prod(shape) * 4 / (1024**3)
    logger.info(f"  Shape: {shape}")
    logger.info(f"  Uncompressed size: {size_gb:.1f} GiB")
    logger.info(f"  Chunk shape: {CHUNK_SHAPE}")

    # Create lazy template (no memory allocation)
    template = xr.Dataset(
        {
            "tp": (
                ("init_date", "member", "lead_time", "lat", "lon"),
                da.zeros(shape, chunks=shape, dtype=np.float32),
                {
                    "long_name": "Total precipitation",
                    "units": "m",
                    "source": "ECMWF IFS ensemble (enfo)",
                },
            ),
        },
        coords={
            "init_date": ("init_date", init_date),
            "member": ("member", member),
            "lead_time": ("lead_time", lead_time, {"units": "hours"}),
            "lat": ("lat", EA_LATS, {"units": "degrees_north"}),
            "lon": ("lon", EA_LONS, {"units": "degrees_east"}),
        },
        attrs={
            "title": "ECMWF IFS Ensemble TP — East Africa Subset",
            "source": "GIK parquet references → S3 GRIB byte-range reads",
            "institution": "ICPAC",
            "region": "East Africa (ICPAC)",
            "lat_range": f"{LAT_MIN} to {LAT_MAX}",
            "lon_range": f"{LON_MIN} to {LON_MAX}",
        },
    )
    logger.info(f"  Template:\n{template}")

    # Set up Icechunk store
    if args.local:
        logger.info(f"  Using local storage: {args.local}")
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        logger.info(f"  Using GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}")
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

    # Write metadata only (compute=False)
    session = repo.writable_session("main")
    template.to_zarr(
        session.store,
        compute=False,
        mode="w",
        encoding={
            "tp": {
                "chunks": CHUNK_SHAPE,
                "fill_value": float("nan"),
            },
        },
        consolidated=False,
    )
    session.commit("initialize ECMWF EA tp template")
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("INIT COMPLETE")
    logger.info(f"  Shape: {shape}")
    logger.info(f"  Chunks: {CHUNK_SHAPE}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


# ─── Phase 2: fill ─────────────────────────────────────────────────────────


def fill_store(args):
    """Fill store with real data from HuggingFace parquets → S3 GRIB reads.

    Workers process one date at a time (all 51 members, 53 lead times).
    Results are written to Icechunk and committed in batches.
    """
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: Populating ECMWF EA tp store")
    logger.info("=" * 60)
    overall_start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    n_dates = len(dates)
    logger.info(f"  Dates: {n_dates} ({dates[0]} to {dates[-1]})")

    # Open target Icechunk store
    if args.local:
        target_storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        target_storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default(),
    )

    # Resume detection: find completed date indices from commit messages
    # Commit messages are per-date: "fill date 42 (20240411): 51/51 members"
    completed_indices = set()
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("fill date "):
                try:
                    # Parse "fill date 42 (20240411): 51/51 members"
                    idx_str = msg.split("fill date ")[1].split(" ")[0]
                    completed_indices.add(int(idx_str))
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    start_idx = max(completed_indices) + 1 if completed_indices else 0
    if start_idx > 0:
        logger.info(f"  Resuming from date index {start_idx} ({len(completed_indices)} dates done)")

    remaining = [(i, d) for i, d in enumerate(dates) if i >= start_idx]
    if not remaining:
        logger.info("  All dates already filled!")
        return
    logger.info(f"  Remaining: {len(remaining)} dates")

    # Launch Coiled cluster
    n_workers = args.n_workers
    cluster = coiled.Cluster(
        name=f"ecmwf-ea-tp-{int(time.time()) % 10000}",
        n_workers=[min(5, n_workers), n_workers],
        worker_vm_types="n2-standard-4",
        package_sync=True,
        region="europe-west1",
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(10, n_workers), timeout=600)
    logger.info(f"  Cluster ready: {client.dashboard_link}")

    # ── Worker function: one Dask task per (date, member) ──
    # Each task downloads 1 parquet from HF, fetches 53 GRIB chunks from S3
    # in parallel via ThreadPoolExecutor(8), returns ~4.6 MB.
    # This gives Dask full visibility: 50 dates × 51 members = 2,550 tasks
    # per batch, perfect load balancing across all workers.

    def read_member_tp_ea(date_str, member_id, lead_time_hours,
                          hf_base_url, grid_shape, lat_idx_start, lat_idx_end,
                          lon_idx_start, lon_idx_end):
        """Process one (date, member): fetch 53 tp steps from S3.

        Returns dict with member data array of shape (53, n_lat, n_lon) ~4.6 MB.
        """
        import json
        import os
        import tempfile
        import warnings
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import fsspec
        import numpy as np
        import pandas as pd

        warnings.filterwarnings("ignore")
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

        try:
            import gribberish
            has_gribberish = True
        except ImportError:
            has_gribberish = False

        n_steps = len(lead_time_hours)
        n_lat = lat_idx_end - lat_idx_start
        n_lon = lon_idx_end - lon_idx_start

        year = date_str[:4]
        month = date_str[4:6]

        # Download parquet from HuggingFace
        parquet_url = (
            f"{hf_base_url}/{year}/{month}/{date_str}/00z/"
            f"{date_str}00z-{member_id}.parquet"
        )
        df = pd.read_parquet(parquet_url)

        # Build zstore dict
        zstore = {}
        for _, row in df.iterrows():
            key = row["key"]
            value = row["value"]
            if isinstance(value, bytes):
                try:
                    decoded = value.decode("utf-8")
                    if decoded.startswith("[") or decoded.startswith("{"):
                        value = json.loads(decoded)
                    else:
                        value = decoded
                except Exception:
                    pass
            elif isinstance(value, str):
                if value.startswith("[") or value.startswith("{"):
                    try:
                        value = json.loads(value)
                    except Exception:
                        pass
            zstore[key] = value
        del df

        member_key = member_id.replace("_", "")

        # Collect (step_idx, ref) pairs for tp
        step_refs = []
        for s_idx, step_h in enumerate(lead_time_hours):
            for pattern in [
                f"step_{step_h:03d}/tp/sfc/{member_key}/0.0.0",
                f"step_{step_h:03d}/tp/sfc/0.0.0",
                f"step_{step_h:03d}/tp/surface/{member_key}/0.0.0",
            ]:
                if pattern in zstore:
                    val = zstore[pattern]
                    if isinstance(val, list) and len(val) >= 3:
                        step_refs.append((s_idx, val))
                        break
        del zstore

        member_data = np.full((n_steps, n_lat, n_lon), np.nan, dtype=np.float32)
        if not step_refs:
            return {"date_str": date_str, "member_id": member_id, "data": member_data}

        s3_fs = fsspec.filesystem("s3", anon=True)

        def _fetch_one_step(s_idx, ref):
            url, offset, length = ref[0], ref[1], ref[2]
            if not url.endswith(".grib2"):
                url = url + ".grib2"
            with s3_fs.open(url, "rb") as f:
                f.seek(offset)
                grib_bytes = f.read(length)

            arr = None
            if has_gribberish:
                try:
                    flat = gribberish.parse_grib_array(grib_bytes, 0)
                    arr = flat.reshape(grid_shape)
                except Exception:
                    pass
            if arr is None:
                import xarray as xr
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".grib2"
                ) as tmp:
                    tmp.write(grib_bytes)
                    tmp_path = tmp.name
                try:
                    ds = xr.open_dataset(tmp_path, engine="cfgrib")
                    arr = ds[list(ds.data_vars)[0]].values.copy()
                    ds.close()
                finally:
                    os.unlink(tmp_path)

            ea_arr = arr[lat_idx_start:lat_idx_end,
                         lon_idx_start:lon_idx_end].astype(np.float32)
            return s_idx, ea_arr

        # Fetch all 53 steps in parallel (8 threads — I/O bound S3 reads)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {
                pool.submit(_fetch_one_step, s_idx, ref): s_idx
                for s_idx, ref in step_refs
            }
            for fut in as_completed(futs):
                try:
                    s_idx, ea_arr = fut.result()
                    member_data[s_idx] = ea_arr
                except Exception:
                    pass

        return {"date_str": date_str, "member_id": member_id, "data": member_data}

    # Precompute EA spatial slice indices
    lat_idx_start = int(LAT_INDICES[0])
    lat_idx_end = int(LAT_INDICES[-1]) + 1
    lon_idx_start = int(LON_INDICES[0])
    lon_idx_end = int(LON_INDICES[-1]) + 1

    # Process in batches
    COMMIT_BATCH = args.commit_batch
    total_written = 0
    total_failed = 0
    failed_dates = []

    for batch_start in range(0, len(remaining), COMMIT_BATCH):
        batch = remaining[batch_start : batch_start + COMMIT_BATCH]
        batch_idx_min = batch[0][0]
        batch_idx_max = batch[-1][0]
        n_tasks = len(batch) * N_MEMBERS
        logger.info(
            f"  Batch: date indices {batch_idx_min}-{batch_idx_max} "
            f"({len(batch)} dates × {N_MEMBERS} members = {n_tasks} tasks, "
            f"{total_written}/{len(remaining)} dates done)"
        )

        # Submit one Dask task per (date, member)
        # key format: "d{date_idx}-m{member_idx}" for dashboard readability
        futures = {}
        for date_idx, date_str in batch:
            for m_idx, member_id in enumerate(MEMBER_IDS):
                future = client.submit(
                    read_member_tp_ea,
                    date_str,
                    member_id,
                    LEAD_TIME_HOURS,
                    HF_BASE_URL,
                    ECMWF_GRID_SHAPE,
                    lat_idx_start, lat_idx_end,
                    lon_idx_start, lon_idx_end,
                    key=f"d{date_idx}-m{m_idx:02d}",
                )
                futures[future] = (date_idx, date_str, m_idx)

        # Collect results and write+commit each date to Icechunk as soon as
        # all 51 members arrive.  Each date gets its own commit so that:
        #   - If the process is killed, all previously committed dates survive
        #   - Memory stays bounded to ~245 MB (one assembled date at a time)
        #   - Resume detection finds individual committed dates
        date_members = {}     # date_idx -> {m_idx: (53,157,145) array}
        date_expected = {}    # date_idx -> total members received (ok + fail)
        date_fail_count = {}  # date_idx -> number of failed members

        batch_ok = 0
        batch_fail = 0
        tasks_done = 0

        for future in distributed.as_completed(futures):
            date_idx, date_str, m_idx = futures[future]
            try:
                result = future.result()
                date_members.setdefault(date_idx, {})[m_idx] = result["data"]
                del result
            except Exception as e:
                date_fail_count[date_idx] = date_fail_count.get(date_idx, 0) + 1

            date_expected[date_idx] = date_expected.get(date_idx, 0) + 1
            tasks_done += 1

            if tasks_done % 100 == 0:
                logger.info(f"    Progress: {tasks_done}/{n_tasks} member-tasks done")

            # Check if this date is complete (all 51 members received)
            if date_expected[date_idx] == N_MEMBERS:
                members = date_members.pop(date_idx, {})
                n_ok = len(members)
                n_fail = date_fail_count.get(date_idx, 0)

                if n_ok == 0:
                    batch_fail += 1
                    total_failed += 1
                    failed_dates.append(date_str)
                    logger.error(
                        f"    Date {date_idx} ({date_str}) FAILED: 0/{N_MEMBERS} members"
                    )
                    continue

                # Assemble (51, 53, 157, 145), write, and commit immediately
                data = np.full(
                    (N_MEMBERS, N_STEPS, N_LAT, N_LON), np.nan, dtype=np.float32
                )
                for m_i, member_data in members.items():
                    data[m_i] = member_data
                del members

                try:
                    session = target_repo.writable_session("main")
                    ds_write = xr.Dataset({
                        "tp": (
                            ("init_date", "member", "lead_time", "lat", "lon"),
                            data[np.newaxis],  # (1, 51, 53, 157, 145)
                        ),
                    })
                    ds_write.to_zarr(
                        session.store,
                        region={"init_date": slice(date_idx, date_idx + 1)},
                        consolidated=False,
                    )
                    del data
                    session.commit(
                        f"fill date {date_idx} ({date_str}): "
                        f"{n_ok}/{N_MEMBERS} members"
                    )

                    batch_ok += 1
                    total_written += 1
                    logger.info(
                        f"    Committed date {date_idx} ({date_str}) "
                        f"[{n_ok}/{N_MEMBERS} members, {n_fail} failed] "
                        f"({total_written}/{len(remaining)} total)"
                    )
                except Exception as e:
                    batch_fail += 1
                    total_failed += 1
                    failed_dates.append(date_str)
                    logger.error(
                        f"    Date {date_idx} ({date_str}) WRITE/COMMIT FAILED: {e}"
                    )

    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  Dates written: {total_written}/{n_dates}")
    logger.info(f"  Failed: {total_failed} — {failed_dates[:20]}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)

    results = {
        "status": "success" if not failed_dates else "partial",
        "dates_written": total_written,
        "dates_total": n_dates,
        "failed_dates": failed_dates,
        "elapsed_min": elapsed / 60,
    }
    results_path = f"ecmwf_ea_tp_fill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results: {results_path}")


# ─── Phase 3: verify ───────────────────────────────────────────────────────


def verify_store(args):
    """Inspect and verify the Icechunk store."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Inspecting ECMWF EA tp store")
    logger.info("=" * 60)

    if args.local:
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )

    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default(),
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    for dim in ["init_date", "member", "lead_time", "lat", "lon"]:
        if dim in ds.dims:
            vals = ds[dim].values
            logger.info(f"  {dim}: {ds.sizes[dim]} [{vals[0]} .. {vals[-1]}]")

    for var in ds.data_vars:
        da = ds[var]
        logger.info(f"\nVariable '{var}': dtype={da.dtype}, shape={da.shape}")
        if hasattr(da, "encoding") and "chunks" in da.encoding:
            logger.info(f"  chunks: {da.encoding['chunks']}")

    # Spot-check: load one date+member
    if args.spot_check and "tp" in ds.data_vars:
        logger.info("\nSpot-check: loading first date, first member...")
        try:
            sample = ds["tp"].isel(init_date=0, member=0).load()
            n_valid = int((~np.isnan(sample.values)).sum())
            n_total = sample.values.size
            pct = 100 * n_valid / n_total if n_total else 0
            logger.info(f"  Valid (non-NaN): {n_valid}/{n_total} ({pct:.1f}%)")
            if n_valid > 0:
                vals = sample.values[~np.isnan(sample.values)]
                logger.info(f"  Min: {float(vals.min()):.6f}")
                logger.info(f"  Max: {float(vals.max()):.6f}")
                logger.info(f"  Mean: {float(vals.mean()):.6f}")
        except Exception as e:
            logger.error(f"  Spot-check failed: {e}")

    # Commit history
    try:
        commits = list(repo.ancestry(branch="main"))
        logger.info(f"\nCommit history ({len(commits)} commits):")
        for c in commits[:10]:
            logger.info(f"  {c.message}")
        if len(commits) > 10:
            logger.info(f"  ... and {len(commits) - 10} more")
    except Exception:
        pass

    logger.info("\nVerification complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ECMWF EA TP — Materialized Icechunk Store",
    )
    sub = parser.add_subparsers(dest="command")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    common.add_argument("--gcs-prefix", type=str, default=GCS_PREFIX)
    common.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    common.add_argument("--local", type=str, default=None,
                        help="Local filesystem path (overrides GCS)")

    # ── init ──
    p_init = sub.add_parser("init", parents=[common],
                            help="Create empty template store")
    p_init.add_argument("--start-date", type=str, default="20240301",
                        help="Start date YYYYMMDD")
    p_init.add_argument("--end-date", type=str, default="20251231",
                        help="End date YYYYMMDD")

    # ── fill ──
    p_fill = sub.add_parser("fill", parents=[common],
                            help="Fill store from HF parquets + S3 GRIB reads")
    p_fill.add_argument("--start-date", type=str, default="20240301")
    p_fill.add_argument("--end-date", type=str, default="20251231")
    p_fill.add_argument("--n-workers", type=int, default=20)
    p_fill.add_argument("--commit-batch", type=int, default=20,
                        help="Number of dates per Icechunk commit batch")

    # ── verify ──
    p_verify = sub.add_parser("verify", parents=[common],
                              help="Inspect store contents")
    p_verify.add_argument("--spot-check", action="store_true", default=True)
    p_verify.add_argument("--no-spot-check", action="store_false",
                          dest="spot_check")

    args = parser.parse_args()

    if args.command == "init":
        init_store(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
