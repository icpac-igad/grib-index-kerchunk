# GEFS Lithops Cloud Run — Deployment, Test, and Cost Evaluation

**Updated**: 2026-04-16
**Status**: Operational. Stage 2 threading deployed (commit a9bbf94). 50-date cost batch in progress.

---

## Deployment Summary

| Component | Status | Details |
|-----------|--------|---------|
| Service account | Created | `gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com` |
| Artifact Registry | In use | `gcr.io/e4drr-crafd/gefs-lithops-runtime` |
| Docker image | Built via Cloud Build | template tar.gz + deflated-store parquet baked in |
| Cloud Run service | Auto-deployed by Lithops | `lithops-worker-*` (us-east1) |
| Output bucket | Cross-project (sewaa-416306) | `gs://gik-gefs-aws-tf` |
| Staging bucket | `gs://lithops-us-east1-gefs-e4drr` | holds job inputs/results |

Worker config: **2 vCPU, 2048 MiB, concurrency=1, timeout=3600s**.

---

## Architecture: how dates and members flow

```
Local machine (orchestrator)
  └── resume_random_batch.sh          ← serial loop, one date at a time
        └── uv run run_lithops_gefs.py --date $D --run 00
              └── Lithops FunctionExecutor
                    └── fexec.map(process_gefs_date, [date])
                          └── 1 × Cloud Run worker (2 vCPU, 2 GiB)
                                ├── Stage 1: load template parquet (baked in image)
                                ├── Stage 2: ThreadPoolExecutor(2 threads)
                                │     ├── thread-1: gep01 .idx fetch → gefs_kind
                                │     ├── thread-2: gep02 .idx fetch → gefs_kind  (parallel)
                                │     ├── thread-1: gep03 .idx fetch → gefs_kind
                                │     └── ...  (30 members, 2 at a time)
                                ├── Stage 3: serial zarr → parquet per member
                                └── Upload 30 × parquet → gs://gik-gefs-aws-tf
```

Key points:
- **Dates are serial** in the batch loop (one Cloud Run activation per date).
- **Members are parallel** inside each activation (2 threads, matching 2 vCPU).
- `--max-workers N` controls how many *dates* Lithops can dispatch in parallel if
  multiple dates are passed at once — unused in the serial batch loop.

---

## Performance: before and after Stage 2 threading

### Before (commit b7b41d5 — serial member loop)

All 30 members processed one-by-one inside one Cloud Run worker.

| Date | Wall time (observed) | rc | Notes |
|------|---------------------|----|-------|
| 20221122 | 25m 39s | 0 | full run, pre-threading |
| 20221224 | 25m 44s | 0 | full run, pre-threading |
| 20230129 | 27m 41s | 0 | full run, pre-threading |
| 20230218 | 28m 58s | 0 | full run, pre-threading |

~35s/member × 30 members ≈ 17–20 min Cloud Run compute + ~5–8 min
orchestrator polling overhead = ~25–29 min total wall time.

### After (commit a9bbf94 — ThreadPoolExecutor(2) for Stage 2)

Stage 2 (S3 `.idx` fetch) runs 2 members in parallel. Each thread owns its own
`LocalTarGzMappingManager` (120 MB each) — thread-safe, peak memory ~480 MB
inside the 2 GiB limit. Stage 3 (zarr→parquet) remains serial.

Worker log from smoke test (date 20240307, 3 members) confirms interleaved execution:

```
Processed 10/81 timesteps for gep01
Processed 10/81 timesteps for gep02   ← gep01 + gep02 in parallel
Processed 20/81 timesteps for gep01
Processed 20/81 timesteps for gep02
...
Processed 10/81 timesteps for gep03   ← gep03 starts after a thread frees
```

Expected improvement: Stage 2 time halved (~10 min → ~5 min), Stage 3 unchanged
(~8 min serial). Target per-date wall: **~12–15 min** vs ~25 min before.
Results from the current batch (2026-04-16) will confirm.

---

## Exact commands used

### One-time setup
```bash
cd devops/lithops_cr_gefs_gik

# Activate lithops SA key (already in place)
# lithops_config.yaml points to service_account/gefs-lithops-deployer-key.json
```

### Smoke test (3 members, confirms parquets land in GCS)
```bash
uv run run_lithops_gefs.py --date 20240307 --run 00 --max-members 3 --yes
# Expected: 3 parquets in gs://gik-gefs-aws-tf/run_par_gefs/2024/03/20240307/00z/
# Wall time: ~2–3 min
```

### Full single-date test (30 members)
```bash
uv run run_lithops_gefs.py --date 20230602 --run 00 --yes
# Expected: 30 parquets in gs://gik-gefs-aws-tf/run_par_gefs/2023/06/20230602/00z/
# Wall time: ~12–15 min (with threading), ~25 min (without)
```

### 50-date random batch (cost estimation)
```bash
# Dates defined in cost_test_random_dates.json (seed=42, one per month Jan 2022–Feb 2026, 00z)

# Start batch under nohup (survives session drops):
nohup ./resume_random_batch.sh > nohup_random.out 2>&1 &

# The script:
# 1. Reads all 50 dates from cost_test_random_dates.json
# 2. For each date: checks GCS for existing 30 parquets (skip if found)
# 3. Runs: timeout 1800 uv run run_lithops_gefs.py --date $D --run 00 --max-workers 20 --yes
# 4. Logs SKIP/START/END with timestamps to batch_random_YYYYMMDDTHHMMSSZ.log

# Check progress:
grep -E "SKIP|START|END|BATCH" batch_random_20260416T070859Z.log

# Count done vs missing:
python3 -c "
import json, subprocess
dates = json.load(open('cost_test_random_dates.json'))['dates']
done = [d for d in dates if subprocess.run(
    ['gsutil', 'ls', f'gs://gik-gefs-aws-tf/run_par_gefs/{d[:4]}/{d[4:6]}/{d}/00z/'],
    capture_output=True).stdout.count(b'.parquet') >= 30]
print(f'{len(done)}/{len(dates)} complete')
"
```

### Verify GCS output
```bash
# Count parquets for a specific date
gsutil ls gs://gik-gefs-aws-tf/run_par_gefs/2022/11/20221122/00z/ | wc -l
# Expected: 30

# List all processed years/months
gsutil ls gs://gik-gefs-aws-tf/run_par_gefs/
```

### View Cloud Run worker logs
```bash
gcloud logging read \
  'resource.type=cloud_run_revision AND resource.labels.service_name=~"lithops-worker"' \
  --project=e4drr-crafd --limit=50 \
  --format='value(timestamp,textPayload)'
```

---

## 50-date random batch: results to date (2026-04-16)

Seed: 42. One date per month, Jan 2022 – Feb 2026, all 00z run.
Defined in `cost_test_random_dates.json`.

### 2022 dates (all 12 complete)

| Date | Parquets in GCS | How processed |
|------|-----------------|---------------|
| 20220121 | 30/30 | Earlier test run |
| 20220204 | 30/30 | Earlier test run |
| 20220301 | 30/30 | Earlier test run |
| 20220424 | 30/30 | Earlier test run |
| 20220509 | 30/30 | Earlier test run |
| 20220608 | 30/30 | Earlier test run |
| 20220708 | 30/30 | Earlier test run |
| 20220805 | 30/30 | Earlier test run |
| 20220924 | 30/30 | Earlier test run |
| 20221004 | 30/30 | Earlier test run |
| 20221122 | 30/30 | batch_random_20260414T103832Z.log, 25m39s, rc=0 |
| 20221224 | 30/30 | batch_random_20260414T103832Z.log, 25m44s, rc=0 |

### 2023 dates (all 12 complete)

| Date | Parquets in GCS | Wall time | Notes |
|------|-----------------|-----------|-------|
| 20230129 | 30/30 | 27m41s | rc=0 |
| 20230218 | 30/30 | 28m58s | rc=0 |
| 20230303 | 30/30 | rc=124 (timeout) | Worker completed; GCS-rescue verified output |
| 20230419 | 30/30 | 30m00s | rc=124 (timeout at 1800s); GCS-rescue |
| 20230514 | 30/30 | 30m00s | rc=124; GCS-rescue |
| 20230602 | 30/30 | 30m00s | rc=124; GCS-rescue. Also used for threading test |
| 20230701 | 30/30 | 30m00s | rc=124; GCS-rescue |
| 20230803 | 30/30 | 30m00s | rc=124; GCS-rescue |
| 20230907 | 30/30 | 30m00s | rc=124; GCS-rescue |
| 20231008 | 30/30 | 30m00s | rc=124; GCS-rescue |
| 20231117 | 30/30 | 30m00s | rc=124; GCS-rescue |
| 20231220 | 30/30 | 30m00s | rc=124; GCS-rescue |

Note on rc=124: `timeout 1800` killed the local orchestrator after 30 min (worker
still running). Lithops' GCS-rescue logic then detected the completed parquets
and marked the date as successful. Cloud Run worker finished its work; no data
was lost.

### 2024–2026 dates (23 remaining — batch running 2026-04-16 with threading)

Started 20240423 at 07:10 UTC. Expected ~12–15 min/date with threading.
Results will be appended here once complete.

### Batch logs

| Log file | Dates covered | Notes |
|----------|---------------|-------|
| `batch_random_20260414T090912Z.log` | (aborted — python not found) | |
| `batch_random_20260414T091145Z.log` | 2022 Jan–Mar | early run, no skip logic |
| `batch_random_20260414T100151Z.log` | 2022 Jan–Mar | prefix bug, redone |
| `batch_random_20260414T103832Z.log` | 2022 Nov – 2024 Mar | main pre-threading run |
| `batch_random_20260416T070859Z.log` | 2024 Apr – 2026 Feb | **current**, with threading |

---

## Cost evaluation

Cloud Run on-demand (us-east1) pricing for 2 vCPU / 2 GiB worker:

| Component | Rate |
|-----------|------|
| vCPU | $0.000024 / vCPU-s |
| Memory | $0.0000025 / GiB-s |

Per-date billable compute (observed ~20 min = 1200 s pre-threading):

| Component | Seconds × units | Cost |
|-----------|-----------------|------|
| vCPU | 1200 × 2 | $0.058 |
| Memory | 1200 × 2 GiB | $0.006 |
| **Per-date total** | | **~$0.064** |

With threading (~12 min = 720 s expected):

| Component | Seconds × units | Cost |
|-----------|-----------------|------|
| vCPU | 720 × 2 | $0.035 |
| Memory | 720 × 2 GiB | $0.004 |
| **Per-date total** | | **~$0.039** |

Scaling (using threaded estimate):

| Scope | Dates | Cost |
|-------|-------|------|
| One 00z run | 1 | ~$0.04 |
| Daily (4 runs) | 4 | ~$0.16 |
| Month (4 × 30) | 120 | ~$4.70 |
| Year backfill (4 × 365) | 1460 | ~$57 |
| Full 5-year hindcast (4 × 1825) | 7300 | ~$285 |

GCS storage: ~15 MiB/date × 7300 dates = ~107 GiB ≈ $2.20/mo at standard
us-east1 rates. Negligible vs compute.

---

## Code changes (commit a9bbf94, 2026-04-15)

File: `run_lithops_gefs.py`

```python
# New import
from concurrent.futures import ThreadPoolExecutor, as_completed

# New constant (env-overridable)
PARALLEL_WORKERS = int(os.environ.get('PARALLEL_WORKERS', '2'))

# New Stage 2 thread worker — each thread has its own MappingManager
def _stage2_worker(args: tuple) -> tuple:
    member, date_str, run, tar_gz_path = args
    mapping_manager = LocalTarGzMappingManager(tar_gz_path)
    axes = generate_axes(date_str)
    gefs_kind = cs_create_mapped_index_local(
        axes, date_str, member,
        tar_gz_path=tar_gz_path,
        mapping_manager=mapping_manager,
    )
    mapping_manager.cleanup()
    return (member, gefs_kind)

# Inside process_gefs_date() — Stage 2 now threaded:
n_workers = min(PARALLEL_WORKERS, len(members))
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    future_to_member = {
        executor.submit(_stage2_worker, args): args[0]
        for args in task_args
    }
    for future in as_completed(future_to_member):
        fut_result = future.result()   # renamed from 'result' to avoid shadowing
        ...

# Stage 3 remains serial (zarr→parquet, one member at a time)
```

Pattern mirrors `devops/lithops_cr_ecmwf_gik/run_lithops_ecmwf.py` which uses
`ThreadPoolExecutor(4)` for its Stage 2 on a 4 vCPU worker.

---

## Files referenced

| File | Purpose |
|------|---------|
| `run_lithops_gefs.py` | Main orchestration + worker function |
| `resume_random_batch.sh` | nohup batch wrapper with GCS skip logic |
| `cost_test_random_dates.json` | 50 reproducible random dates (seed=42) |
| `batch_random_20260414T103832Z.log` | Pre-threading batch log |
| `batch_random_20260416T070859Z.log` | Current batch log (with threading) |
| `lithops_config.yaml` | Lithops backend config (us-east1, gefs SA) |
| `Dockerfile` | Runtime image (template baked in) |
| `cloudbuild.yaml` | Cloud Build config for image rebuild |
