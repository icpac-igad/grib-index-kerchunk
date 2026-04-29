# Coiled Dask Implementation Plan for ECMWF cGAN Streaming

**Date:** 2026-02-05
**Based on:** CMORPH parallel processing lessons from `deploy-itt/arco_fetch/CMORPH/`

---

## Current State Analysis

### Sequential Processing Performance
- **51 members** × **12 variables** × **9 timesteps** = **5,508 S3 fetches**
- Sequential time: ~4 hours (240 minutes)
- Bottleneck: Single-threaded S3 fetches from ECMWF data

### Target Performance
- **20-30 Coiled workers** processing in parallel
- Target time: **15-30 minutes** (12-16x speedup)
- Cost: ~$2-3 per run (Coiled pricing)

---

## Lessons from CMORPH Implementation

### Key Errors to Avoid

1. **ConflictError** - Concurrent commits to same Icechunk branch
   - Root cause: Multiple workers updating same branch simultaneously
   - Solution: **Batch-wise branches** (`batch_0`, `batch_1`, etc.)

2. **GCS Rate Limiting (429)** - Too many writes to same reference file
   - Root cause: 20 workers all updating same branch ref
   - Solution: Each batch has its own branch

3. **Chunk Shape Mismatch** - Different array shapes cannot concatenate
   - Solution: Validate dimensions before processing

### CMORPH Architecture Pattern
```
Worker 0 → batch_0 branch (writes independently)
Worker 1 → batch_1 branch (writes independently)
Worker 2 → batch_2 branch (writes independently)
...
Worker N → batch_N branch (writes independently)

Then: Sequential aggregation from all batch branches
```

---

## Proposed ECMWF cGAN Architecture

### Phase 1: Parallel Data Fetching (Coiled Workers)

```
┌──────────────────────────────────────────────────────────────────┐
│                     COILED CLUSTER (20 workers)                   │
├──────────────────────────────────────────────────────────────────┤
│  Worker 0: Members 0-2      → Icechunk batch_0 branch            │
│  Worker 1: Members 3-5      → Icechunk batch_1 branch            │
│  Worker 2: Members 6-8      → Icechunk batch_2 branch            │
│  ...                                                              │
│  Worker 16: Members 48-50   → Icechunk batch_16 branch           │
├──────────────────────────────────────────────────────────────────┤
│  Each worker:                                                     │
│    1. Reads parquet references                                    │
│    2. Fetches GRIB bytes from ECMWF S3                           │
│    3. Decodes with gribberish                                    │
│    4. Subsets to ICPAC region                                    │
│    5. Writes to unique Icechunk batch branch                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              INTERMEDIATE ICECHUNK STORE (GCS)                    │
├──────────────────────────────────────────────────────────────────┤
│  Branch: batch_0  →  {member_0, member_1, member_2} data         │
│  Branch: batch_1  →  {member_3, member_4, member_5} data         │
│  Branch: batch_2  →  {member_6, member_7, member_8} data         │
│  ...                                                              │
│                                                                   │
│  Each branch stores:                                              │
│    - Raw member data for all variables and timesteps              │
│    - Shape: (n_members_in_batch, n_vars, n_steps, lat, lon)      │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 2: Aggregation (Local Client)

```
┌──────────────────────────────────────────────────────────────────┐
│                   LOCAL AGGREGATION                               │
├──────────────────────────────────────────────────────────────────┤
│  1. Read all batch branches from Icechunk                         │
│  2. Concatenate member data across batches                        │
│  3. Compute ensemble statistics:                                  │
│     - ensemble_mean = nanmean(all_members, axis=0)               │
│     - ensemble_std = nanstd(all_members, axis=0)                 │
│  4. Write final NetCDF: IFS_YYYYMMDD_HHz_cgan.nc                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Icechunk Store Configuration

```python
# GCS Icechunk store for intermediate results
ICECHUNK_GCS_BUCKET = "cpc_awc"  # or dedicated bucket
ICECHUNK_GCS_PREFIX = "ecmwf_cgan_intermediate"

def create_icechunk_repo(gcs_bucket, gcs_prefix, service_account_file):
    """Create or open Icechunk repository for intermediate storage."""
    import icechunk

    config = icechunk.RepositoryConfig.default()

    # Setup virtual chunk container for ECMWF S3 source
    container = icechunk.VirtualChunkContainer(
        "s3://ecmwf-forecasts/",
        store=icechunk.s3_store(region="eu-west-2", anonymous=True),
    )
    config.set_virtual_chunk_container(container)

    # GCS storage for Icechunk metadata
    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account_file
    )

    try:
        return icechunk.Repository.open(storage, config=config)
    except:
        return icechunk.Repository.create(storage, config=config)
```

### 2. Worker Function

```python
def process_member_batch_worker(args):
    """
    Worker function to process a batch of ensemble members.

    Each worker:
    1. Fetches GRIB data for assigned members
    2. Decodes and subsets to ICPAC region
    3. Writes to unique Icechunk branch
    """
    (batch_id, member_indices, parquet_dir, variables, steps,
     creds_content, gcs_bucket, gcs_prefix) = args

    branch_name = f"batch_{batch_id}"

    # ... fetch and decode GRIB data for members ...

    # Write to Icechunk with unique branch
    repo = open_icechunk_repo(gcs_bucket, gcs_prefix, creds_content)

    try:
        repo.create_branch(branch_name, repo.lookup_branch("main"))
    except:
        repo.delete_branch(branch_name)
        repo.create_branch(branch_name, repo.lookup_branch("main"))

    session = repo.writable_session(branch_name)

    # Write member data as Zarr arrays
    ds.to_zarr(session.store, group=f"members")

    session.commit(f"Batch {batch_id}: members {member_indices}")

    return {'batch_id': batch_id, 'status': 'success', ...}
```

### 3. Coiled Cluster Setup

```python
def create_coiled_cluster(n_workers=20):
    """Create Coiled cluster for ECMWF data streaming."""
    import coiled
    from dask.distributed import Client

    cluster = coiled.Cluster(
        name=f"ecmwf-cgan-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-4",  # 4 vCPU, 16GB RAM
        package_sync=True,
        region="eu-west-1",  # Close to ECMWF S3
        workspace="e4drr",
        idle_timeout="15 minutes",
    )

    return Client(cluster), cluster
```

### 4. Main Orchestration

```python
def stream_cgan_variables_coiled(
    parquet_dir,
    date_str,
    run_hour,
    n_workers=20,
    members_per_batch=3,
    service_account_file="coiled-data-e4drr.json",
    gcs_bucket="cpc_awc",
    gcs_prefix="ecmwf_cgan_intermediate"
):
    """
    Stream ECMWF data for cGAN using Coiled parallel processing.
    """
    # Phase 1: Parallel fetching via Coiled
    client, cluster = create_coiled_cluster(n_workers)

    # Split 51 members into batches
    member_batches = create_member_batches(51, members_per_batch)

    # Submit batch tasks
    futures = client.map(process_member_batch_worker, task_args)

    # Collect results
    results = gather_results(futures)

    cluster.close()

    # Phase 2: Aggregate from Icechunk branches
    all_member_data = aggregate_from_icechunk(
        gcs_bucket, gcs_prefix, results['batch_branches']
    )

    # Compute ensemble statistics
    ensemble_mean = np.nanmean(all_member_data, axis=0)
    ensemble_std = np.nanstd(all_member_data, axis=0)

    # Save final NetCDF
    save_cgan_netcdf(ensemble_mean, ensemble_std, date_str, run_hour)
```

---

## Data Flow Summary

```
Step 1: Local client creates parquet references
        ↓
Step 2: Coiled cluster (20 workers) processes members in parallel
        - Worker 0: members 0-2 → batch_0 branch
        - Worker 1: members 3-5 → batch_1 branch
        - ...
        - Worker 16: members 48-50 → batch_16 branch
        ↓
Step 3: Workers fetch GRIB from ECMWF S3 (parallel)
        ↓
Step 4: Workers decode and subset to ICPAC region
        ↓
Step 5: Workers write to Icechunk batch branches (no conflicts!)
        ↓
Step 6: Local client reads all batch branches
        ↓
Step 7: Compute ensemble mean/std
        ↓
Step 8: Write final NetCDF: IFS_YYYYMMDD_HHz_cgan.nc
```

---

## Expected Performance

| Metric | Sequential | Coiled (20 workers) | Improvement |
|--------|------------|---------------------|-------------|
| Total Time | ~240 min | ~15-20 min | **12-16x** |
| S3 Fetches/sec | ~0.4 | ~8 | 20x |
| Cost per run | $0 | ~$2-3 | N/A |
| Members parallel | 1 | 17 batches | 17x |

---

## Files to Create/Modify

1. **`stream_cgan_variables_coiled.py`** - New Coiled-enabled script
2. **`icechunk_utils.py`** - Icechunk helper functions
3. **Update `ECMWF_cGAN_INFERENCE_WORKFLOW.md`** - Document parallel workflow

---

## Dependencies

```bash
# Additional packages for Coiled/Icechunk
pip install coiled dask[complete] distributed icechunk virtualizarr obstore
```

---

## Usage

```bash
# Run with Coiled (parallel, ~15-20 minutes)
python stream_cgan_variables_coiled.py \
  --parquet-dir ecmwf_three_stage_20260203_00z \
  --n-workers 20 \
  --service-account coiled-data-e4drr.json

# Compare with sequential (for testing)
python stream_cgan_variables.py \
  --parquet-dir ecmwf_three_stage_20260203_00z \
  --max-members 5
```

---

## Risk Mitigation

1. **Worker failures**: Implement retry logic for individual batches
2. **Icechunk conflicts**: Each batch has unique branch - no conflicts
3. **Memory pressure**: Limit members per batch to 3 (controllable)
4. **Network issues**: Workers close to ECMWF S3 (eu-west-1)

---

*Plan created: 2026-02-05*
*Based on: CMORPH processing lessons and ECMWF workflow documentation*
