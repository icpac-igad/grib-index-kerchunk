# GEFS Backfill — Operations & Cost

Backfill historical GEFS parquet reference files (00z / 06z / 12z / 18z)
into your GCS bucket using the Cloud Run + lithops pipeline.

> **Date floor:** NOAA's GEFS realtime/reforecast archive on
> `s3://noaa-gefs-pds/` starts at **2020-09-25**. Earlier dates have no
> upstream data.

---

## Prerequisites

1. Cloud Run runtime image deployed (see [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md))
2. Service account key at `service_account/<sa-key>.json` (used by `lithops_config.yaml`)
3. `gsutil` and `uv` on `PATH`
4. Env vars set in the orchestrator shell:
   ```bash
   export GCS_BUCKET=<your-bucket-name>
   export GOOGLE_APPLICATION_CREDENTIALS=service_account/<sa-key>.json
   ```

---

## Quick start — full backfill 2020-09-25 → 2025-12-31

```bash
mkdir -p logs/backfill
nohup bash run_backfill_chain.sh > logs/backfill/chain.out 2>&1 &
echo "Chain PID: $!"
```

Already-completed dates (≥30 parquets at the run-hour prefix) are skipped
automatically via `gsutil ls`, so re-running is safe and resumes where it
left off.

### Single-year or partial range

Edit the `run_range` calls in `run_backfill_chain.sh`, or call
`run_backfill_single_run.sh` directly:

```bash
bash run_backfill_single_run.sh --run 00 --start 20240101 --end 20241231 --workers 20
```

---

## Scripts

| File | Purpose |
|------|---------|
| `_run_batch_dates.py` | Helper: dispatches a list of dates through `run_with_lithops`, force-exits so lithops' background threads don't hang the bash wrapper |
| `run_backfill_single_run.sh` | Scans GCS for missing dates in `[--start, --end]` for one run-hour; dispatches batches of `--workers` at a time |
| `run_backfill_chain.sh` | Sequential wrapper for multiple year-ranges (one nohup process drives the whole multi-year backfill) |

---

## Operational notes

### Cold-start HTTP 500 / 429 storms

With `min_workers=0`, the first burst of dispatches hits N cold-start
containers and ~5 of them 500/429-out. Pin one warm instance:

```bash
gcloud run services update <lithops-worker-service> \
  --region=<region> --min-instances=1
```

Adds ~$0.40/day idle cost; eliminates burst failures.

### `--workers` choice (concurrency)

| Setting       | Retry rate | Use when                                      |
|---------------|------------|-----------------------------------------------|
| `--workers 20` | ~23%      | Default — full speed, accept retry overhead   |
| `--workers 10` | ~3%       | Cleanup pass for stragglers, fewer 429 storms |

### `FAIL` in the run log ≠ data lost

A `FAIL ... Max retries exceeded` line means the lithops orchestrator gave
up after 3 retries. But the worker often still wrote its 30 parquets to
GCS via the rescue path. **Always verify with a GCS count, not the log
counts.**

```bash
uv run --with google-cloud-storage python3 -c "
import os
from google.cloud import storage
c = storage.Client()
b = c.bucket(os.environ['GCS_BUCKET'])
for Y in (2020, 2021, 2022, 2023, 2024, 2025):
    counts = {}
    for blob in b.list_blobs(prefix=f'run_par_gefs/{Y}/'):
        parts = blob.name.split('/')
        if len(parts) >= 5 and parts[4] == '00z' and blob.name.endswith('.parquet'):
            counts[parts[3]] = counts.get(parts[3], 0) + 1
    done = sum(1 for n in counts.values() if n >= 30)
    print(f'{Y}: {done} dates with ≥30 parquets at 00z')
"
```

### Lithops shutdown hang (already fixed)

`_run_batch_dates.py` calls `os._exit(0)` after `print_summary` to bypass
lithops' background-thread leak that otherwise blocks bash chains forever
between batches. Without this, every batch reaches `print_summary` and
then idles indefinitely.

### Zombie-PID wait (chain script)

`run_backfill_chain.sh` reads `/proc/$WAIT_PID/status` and checks for
`State: Z` instead of `kill -0` — necessary because `kill -0` returns
success on zombie processes too, which would hang the chain forever.

---

## Observed timings

| Phase                        | Wall clock |
|------------------------------|------------|
| One date (single worker)     | ~22 min    |
| Batch of 20 (`--workers 20`) | **~31 min** (with `os._exit` fix) |
| Batch of 10 (`--workers 10`) | ~31 min    |
| Full year (365 dates @ 00z)  | ~10–13 h   |
| 6-year backfill (1,924 dates)| ~54 h      |

---

## Cost

### Per-date unit cost (Cloud Run, request-based, 2 vCPU / 2 GiB / ~22 min)

Cloud Run pricing in us-east1:

| Resource | Rate                  |
|----------|-----------------------|
| vCPU     | $0.000024 / vCPU-s    |
| Memory   | $0.0000025 / GiB-s    |
| Idle CPU (`min-instances=1`) | $0.000018 / vCPU-s |
| Requests | $0.40 / million       |

Component breakdown for one successful date:

| Component                | Cost     |
|--------------------------|----------|
| 2 vCPU × 1,320 s         | $0.0634  |
| 2 GiB × 1,320 s          | $0.0066  |
| **Per-date (no retry)**  | **$0.0700** |

Blended including retry overhead:

| Profile         | Retry rate | **Per-date blended** |
|-----------------|------------|----------------------|
| `--workers 20`  | ~23%       | **$0.092**           |
| `--workers 10`  | ~3%        | **$0.075**           |

> Retry overhead = the original GCP worker keeps running for another
> ~6 min after lithops' 25-min `WAIT_TIMEOUT` fires + a fresh retry
> worker starts. Both are billed.

### Full backfill estimate (2020-09-25 → 2025-12-31, 00z only, `--workers 20`)

| Year | Dates | Cost   |
|------|-------|--------|
| 2020 | 98 (Sep 25 → Dec 31) | $9.02   |
| 2021 | 365   | $33.58 |
| 2022 | 365   | $33.58 |
| 2023 | 365   | $33.58 |
| 2024 | 366   | $33.67 |
| 2025 | 365   | $33.58 |
| **Subtotal compute** | **1,924** | **$177.01** |
| Min-instances=1 idle (54 h chain runtime) | | $7.10 |
| Cloud Build runtime image (one-time) | | $0.50 |
| GCS storage (~16 GB, first 12 months) | | $3.84 |
| Network egress (NOAA Open Data + intra-region GCS) | | $0 |
| **One-time backfill total** | | **~$188** |
| Ongoing GCS storage (year 2+) | | **$3.84/year** |

**Per-date all-in:** ~$0.098

### Multi-run-hour scaling

The numbers above are 00z only (one run-hour per date). To backfill all
four run-hours (00 / 06 / 12 / 18), multiply compute by ~4 and storage
by 4:

| Scope                  | Compute | Storage (year 1) |
|------------------------|---------|------------------|
| 00z only (recommended) | ~$177   | $3.84            |
| All 4 run-hours        | ~$708   | $15.36           |

### Cost-optimization levers

| Lever                                     | Savings | Trade-off                   |
|-------------------------------------------|---------|-----------------------------|
| Default `--workers 10` end-to-end         | ~18%    | +30% wall-clock             |
| Drop `min-instances=0` between batches    | ~$7     | risk of 429 storms          |
| Run within free tier window (360K vCPU-s/mo) | ~$10/yr | requires careful scheduling |
| Skip rebuilding the runtime image         | $0.50   | one-time, only first run    |

For a single-year backfill (365 dates) realistic budget = **$32–35** at
`--workers 20` or **$28–30** at `--workers 10`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Chain hangs after first batch summary line | lithops shutdown thread leak | already fixed via `os._exit(0)` in `_run_batch_dates.py` |
| HTTP 500 on dispatch, all 20 workers fail | Cloud Run cold-start race | `gcloud run services update <svc> --min-instances=1` |
| HTTP 429 "Too Many Requests" on dispatch | Burst exceeded autoscaler | drop to `--workers 10`, or set Cloud Run `--max-instances` ≥ workers |
| `FAIL ... Max retries exceeded` but parquets land later | Orchestrator timed out before worker finished | lithops will mark FAIL but worker writes via rescue path; verify with GCS count |
| Same date FAILs on multiple retries | Genuinely missing upstream NOAA data | check `s3://noaa-gefs-pds/` directly; pre-2020-09-25 dates have no archive |
