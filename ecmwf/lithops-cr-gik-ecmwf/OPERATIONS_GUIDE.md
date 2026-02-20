# ECMWF Lithops Cloud Run — Operations Guide

> Project: `e4drr-crafd` | Region: `europe-west3` (Frankfurt)
> Bucket: `gs://gik-ecmwf-aws-tf/run_par_ecmwf/`
> Runtime: `gcr.io/e4drr-crafd/ecmwf-lithops-runtime`

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [GCS Output Structure](#2-gcs-output-structure)
3. [Configuration Changes Made](#3-configuration-changes-made)
4. [Code Changes Made to run_lithops_ecmwf.py](#4-code-changes-made-to-run_lithops_ecmwfpy)
5. [Known Failure Modes and Fixes](#5-known-failure-modes-and-fixes)
6. [Log Checking Routines](#6-log-checking-routines)
7. [GCS Bucket Audit Routines](#7-gcs-bucket-audit-routines)
8. [Backfill Operations](#8-backfill-operations)
9. [Recovery Playbook](#9-recovery-playbook)
10. [Cost Reference](#10-cost-reference)

---

## 1. System Architecture

```
run_lithops_ecmwf.py   (local orchestrator, runs via `uv run`)
        |
        | cloudpickle-serializes process_ecmwf_date()
        | uploads job payload to GCS (lithops-europe-west3-3b6921 bucket)
        |
        v
[Lithops FunctionExecutor — gcp_cloudrun backend]
        |
        | HTTP POST to Cloud Run service URL (one request per date)
        | ThreadPoolExecutor — all invocations sent simultaneously
        |
  +-----+-----+-----+-----+
  |     |     |     |     |
 [CR1] [CR2] [CR3] ... [CRN]   (Cloud Run containers, max 35)
  |
  v  Each container runs process_ecmwf_date(date_str, run):
     Stage 1 — Load Zarr template from HuggingFace (baked into image)
     Stage 2 — Fetch ECMWF S3 .index files + merge with template (~8 min)
     Stage 3 — Build final parquet files from merged refs
     Upload  — Write *.parquet to GCS at YYYY/MM/YYYYMMDD/{run}z/
     Cleanup — rm -rf /tmp/ecmwf_{date}_{run}z
        |
        v
[Lithops monitor thread] polls GCS every ~1s for result pickles
        |
        v
[fexec.wait() returns] → collect f.status, f.result() per future
```

**Supported runs**: ECMWF ensemble (`enfo`) produces forecasts at all 4 daily run
times — **00Z, 06Z, 12Z, and 18Z** — and all four are supported by the current
pipeline. Note: the original code (pre-2026-02-18) hardcoded `{date}000000` in the
S3 index filename regardless of the actual run hour, so only 00Z worked correctly.
This was fixed in commit `db13b78` — S3 paths now embed the run hour correctly:
`{date}{run}0000-{hour}h-enfo-ef.index`. Verified: 20260216 18z produced 51 parquet
files in 5.7 minutes after the fix.

**Typical per-date processing time**: ~8–9 minutes (dominated by Stage 2 S3 fetching).

---

## 2. GCS Output Structure

### New path format (current — introduced 2026-02-18)

```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/
  └── YYYY/
      └── MM/
          └── YYYYMMDD/
              └── {run}z/
                  ├── member_000.parquet
                  ├── member_001.parquet
                  └── ... (51 members total, one .parquet per ensemble member)
```

**Example**:
```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/10/20241015/00z/member_000.parquet
```

### Old path format (pre-2026-02-18, no longer written)

```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/YYYYMMDD/{run}z/
```

A complete date has **51 parquet files** (one per ensemble member, indexed 000–050).

---

## 3. Configuration Changes Made

### `lithops_config.yaml` — `max_workers` raised from 20 → 35

**File**: `lithops_config.yaml`

```yaml
# BEFORE (original)
gcp_cloudrun:
    max_workers: 20        # Cloud Run maxScale = 20 instances

# AFTER (current)
gcp_cloudrun:
    runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime
    runtime_memory: 2048   # 2 GB RAM
    runtime_cpu: 2          # 2 vCPUs
    runtime_timeout: 3600   # 1 hour max per invocation
    max_workers: 35         # 31 dates/month max + 4 buffer
    min_workers: 0
    worker_processes: 1     # 1 process per container (pipeline is CPU-intensive)
```

**Why**: `max_workers` sets Cloud Run's `maxScale` at runtime **deploy time**.
With `max_workers: 20`, dispatching 31 dates simultaneously meant 11+ invocations
received HTTP 500 during the cold-start race (Cloud Run was scaling up but rejected
excess requests). Raising to 35 gives a buffer above the largest monthly batch (31 days).

**Important**: After changing `max_workers`, the runtime must be **redeployed**:
```bash
# Delete old runtime
lithops runtime delete gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Deploy new runtime (picks up new maxScale)
lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Verify maxScale
lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

---

## 4. Code Changes Made to `run_lithops_ecmwf.py`

### 4.1 GCS path restructure — `upload_to_gcs()` (~line 561)

**Problem**: Output was stored flat as `YYYYMMDD/{run}z/`, making it hard to browse
or partition by year/month.

```python
# BEFORE
gcs_path = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{date_str}/{run}z"

# AFTER
year = date_str[:4]
month = date_str[4:6]
gcs_path = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date_str}/{run}z"
```

### 4.2 Added `--yes / -y` flag — `main()` (~line 905)

**Problem**: Interactive confirmation prompt blocked batch scripts.

```python
# Added to argparse:
parser.add_argument('--yes', '-y', action='store_true',
                    help='Skip interactive confirmation (for batch/automated runs)')

# Condition changed from:
if len(dates) > 5:
# to:
if len(dates) > 5 and not args.yes:
```

### 4.3 Full retry logic in `run_with_lithops()` (~line 791)

**Problem**: Original code had no retry — a single HTTP 500 during invocation meant
that date was silently skipped.

**Solution**: Up to 3 attempts with a 30-second delay between retries.

```python
MAX_RETRIES  = 3
WAIT_TIMEOUT = 600    # 10 minutes — above the ~8.5 min processing time
RETRY_DELAY  = 30     # seconds between retries (Cloud Run warms up by then)

remaining = list(dates)

for attempt in range(1, MAX_RETRIES + 1):
    if not remaining:
        break

    if attempt > 1:
        time.sleep(RETRY_DELAY)   # let Cloud Run instances warm up

    futures = fexec.map(process_ecmwf_date, remaining, extra_args=(run,))

    try:
        fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)
    except TimeoutError:
        # Lithops raises TimeoutError unconditionally via signal handler;
        # throw_except=False does NOT suppress it. Catch explicitly.
        logger.warning(f"Attempt {attempt}: wait timeout — collecting partial results")

    # ... collect succeeded/failed per f.status ...
    remaining = still_failing
```

### 4.4 `TimeoutError` catch — `fexec.wait()` (~line 846)

**Problem**: `lithops/utils.py` raises `TimeoutError` unconditionally via a SIGALRM
signal handler when `timeout=` is exceeded. This is **not** suppressed by `throw_except=False`
(which only suppresses errors inside the remote function). Without the try/except the
orchestrator process crashed.

```python
# BEFORE (crashed the process after timeout)
fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)

# AFTER (correctly catches and continues)
try:
    fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)
except TimeoutError:
    logger.warning(
        f"Attempt {attempt}: wait timeout ({WAIT_TIMEOUT}s) — collecting "
        f"completed results, remaining will be retried"
    )
```

**Why `WAIT_TIMEOUT = 600` (10 min)**:
- Normal processing: ~8.5 minutes per date
- 10 minutes gives workers time to complete before timing out dates that
  merely had slow invocations
- Former value was 1800s (30 min) — caused 30-minute hangs when all invocations got 429'd

### 4.5 GCS verification rescue — after `f.status` loop (~line 882)

**Problem**: Lithops' monitor thread polls GCS every ~1 second to detect worker
completion by looking for result pickle files. If the **orchestrator's network is
interrupted** (e.g. a transient DNS failure resolving `storage.googleapis.com`),
the monitor thread crashes and `f.status` for completed workers remains stuck at
`'invoked'` or `'running'` — never updating to `'success'` — even though the Cloud
Run worker finished successfully and wrote all parquet files to the output GCS bucket.

This caused "0/31 succeeded" reports for months that were actually 100% complete in GCS.

**Solution**: After the `f.status` loop, scan all "failed" dates in GCS directly.
GCS is the authoritative ground truth.

```python
# GCS verification: rescue dates whose Cloud Run worker completed and wrote
# to GCS but whose f.status was not updated to 'success' because the Lithops
# monitor thread was interrupted (e.g. a transient DNS/network failure).
if still_failing:
    try:
        import gcsfs
        _fs = gcsfs.GCSFileSystem()
        gcs_rescued = []
        for date in still_failing:
            year, month = date[:4], date[4:6]
            gcs_prefix = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date}/{run}z"
            try:
                items = _fs.ls(gcs_prefix)
                if items:   # directory exists with parquet files
                    all_results[date] = {
                        'date': date, 'run': run, 'success': True,
                        'gcs_path': f"gs://{gcs_prefix}",
                        'message': 'GCS-verified (f.status stale — monitor interrupted)',
                    }
                    gcs_rescued.append(date)
                    logger.info(f"  GCS-OK {date}: found {len(items)} file(s)")
            except Exception:
                pass   # not in GCS — genuine failure
        if gcs_rescued:
            still_failing = [d for d in still_failing if d not in set(gcs_rescued)]
            print(f"  GCS verification rescued {len(gcs_rescued)} date(s)")
    except Exception as gcs_exc:
        logger.warning(f"  GCS verification skipped: {gcs_exc}")
```

**Log signature when this triggers**:
```
WARNING - Attempt 1: wait timeout (600s) — collecting completed results, remaining will be retried
INFO -   GCS-OK 20241001: found 51 file(s)
INFO -   GCS-OK 20241002: found 51 file(s)
...
  GCS verification rescued 23 date(s) (Lithops monitor was interrupted)
INFO - ExecutorID ... | JobID M001 - Starting function invocation ... Total: 8 activations
```

---

## 5. Known Failure Modes and Fixes

### 5.1 HTTP 500 — Cloud Run cold-start race

**Symptom**:
```
urllib.error.HTTPError: HTTP Error 500: Internal Server Error
exception calling callback for <Future ... state=finished raised HTTPError>
```

**Cause**: All N invocations arrive simultaneously at Cloud Run when it has 0 warm
instances. Cloud Run scales up but rejects excess requests with 500 until instances
are ready. With `maxScale: 35`, up to 35 simultaneous invocations are handled; any
beyond that get 500.

**Impact**: Affected futures get `f.status = 'error'`, never run.

**Fix applied**:
- `max_workers: 35` in `lithops_config.yaml` (runtime redeployed)
- Retry logic retries those dates 30 seconds later when instances are warm

**These stack traces in the log are normal noise** — printed by Python's
`concurrent.futures` ThreadPoolExecutor callback mechanism. They do not crash
the orchestrator.

---

### 5.2 HTTP 429 — Cloud Run rate limit

**Symptom**:
```
urllib.error.HTTPError: HTTP Error 429: Too Many Requests
```

**Cause**: Similar to 500 but triggered by Cloud Run's invocation rate limiter rather
than the concurrency limiter. Occurs on the very first wave (all-cold-start scenario).

**Fix**: Same as 5.1 — retry logic handles it. After the first wave warms the instances,
subsequent retries succeed immediately.

---

### 5.3 TimeoutError crash

**Symptom**: Process exits abruptly with a Python `TimeoutError` traceback after
exactly `WAIT_TIMEOUT` seconds.

**Cause**: `lithops/utils.py` installs a SIGALRM signal handler that calls
`raise TimeoutError(...)` unconditionally when the wait deadline passes.
`throw_except=False` on `fexec.wait()` does **not** suppress this.

**Fix applied**: Wrapped `fexec.wait()` in `try/except TimeoutError` (see §4.4).

---

### 5.4 `f.status` stale after DNS/network interruption

**Symptom**:
```
WARNING - Attempt 1: wait timeout (600s) — collecting completed results...
FAIL 20241001: invoke failed (status=running), will retry
...
Attempt 1 done: 0/31 succeeded, 31 queued for retry
```
But GCS already has all 31 dates.

**Cause**: Lithops' monitor thread (`StorageMonitor`) polls
`gs://lithops-europe-west3-3b6921/lithops.jobs/{executor_id}/` every ~1 second.
If DNS resolution for `storage.googleapis.com` fails transiently, the monitor
thread raises `google.api_core.exceptions.RetryError` and exits. After that,
no `f.status` updates happen regardless of what Cloud Run workers do.

**Key log signature**:
```
google.api_core.exceptions.RetryError: Timeout of 120.0s exceeded, last exception:
HTTPSConnectionPool(host='storage.googleapis.com'...): Failed to resolve
'storage.googleapis.com' ([Errno -3] Temporary failure in name resolution)
```

**Fix applied**: GCS verification rescue block (§4.5) — scans GCS after each attempt
and promotes completed dates to `success`.

---

### 5.5 Executor hangs indefinitely (no timeout set)

**Symptom**: `fexec.wait()` runs forever (observed hanging for >2 hours).

**Cause**: Pre-fix code had no `timeout=` parameter. If any invocations got HTTP 500
(future result pickle never written), `fexec.wait()` polled GCS indefinitely.

**Fix applied**: `timeout=WAIT_TIMEOUT` (600s) added to `fexec.wait()`.

---

### 5.6 Wrong S3 filename for non-00Z runs (fixed 2026-02-18, commit db13b78)

**Symptom** (pre-fix): Index validation returns 0 members for any 06Z, 12Z, or 18Z
date — dates silently skipped, no parquet files written, no crash.

**Root cause**: ECMWF S3 index filenames embed the run hour in the timestamp component:
```
00z:  s3://.../20260216/00z/ifs/0p25/enfo/20260216000000-0h-enfo-ef.index
06z:  s3://.../20260216/06z/ifs/0p25/enfo/20260216060000-0h-enfo-ef.index
12z:  s3://.../20260216/12z/ifs/0p25/enfo/20260216120000-0h-enfo-ef.index
18z:  s3://.../20260216/18z/ifs/0p25/enfo/20260216180000-0h-enfo-ef.index
```
The original code hardcoded `000000`, so a 18Z request silently fetched the **00Z
index file** (returning 0 members for that member/hour combination) rather than
the correct 18Z file. This was misdiagnosed as "no S3 data for 06Z/18Z".

**Fix applied** (`run_lithops_ecmwf.py` lines ~324 and ~602):
```python
# BEFORE — always fetched 00z file regardless of run
f"{date_str}000000-{hour}h-enfo-ef.index"

# AFTER — correctly embeds the run hour
f"{date_str}{run}0000-{hour}h-enfo-ef.index"
```

**Verification**: `20260216 18z` → 51 parquet files in 5.7 minutes (SUCCESS).

**Current status**: All four runs (00Z, 06Z, 12Z, 18Z) are fully supported. The
current backfill (`run_backfill_00z.sh`) is intentionally scoped to 00Z only for
the 2024-03 to 2025-12 historical archive. Run 12Z, 06Z, or 18Z separately as needed.

---

## 6. Log Checking Routines

### 6.1 Live progress during a running backfill

```bash
# Filter out stack trace noise, show only progress lines
grep -v "File \"\|raise \|urllib\|concurrent\|lithops/\|cloudrun\|invokers\|serverless\|thread\|_base\|result()\|    return \|           \^\|    response\|   ^^^" \
    logs/backfill_00z/run_2024-10_onwards.log | tail -30

# Or tail only the meaningful lines
tail -f logs/backfill_00z/run_2024-10_onwards.log | \
    grep -E "(Attempt|GCS-OK|GCS veri|FAIL |succeeded|rescued|Month:|Completed|OK  20)"
```

### 6.2 Per-month summary log

```bash
# Show all completed months and whether they passed or failed
cat logs/backfill_00z/summary.log | grep -E "^(OK|FAIL)"

# Count OK vs FAIL
grep -c "^OK"   logs/backfill_00z/summary.log
grep -c "^FAIL" logs/backfill_00z/summary.log
```

**Example healthy output**:
```
OK  2024-03  (31 dates)  20m12s
OK  2024-04  (30 dates)  19m47s
...
```

### 6.3 Per-month detailed log

Each month writes its own log under `logs/backfill_00z/backfill_00z_YYYY-MM.log`.

```bash
# Check a specific month's log
cat logs/backfill_00z/backfill_00z_2024-10.log | \
    grep -E "(GCS-OK|  OK|  FAIL|  GIVE UP|succeeded|rescued|Completed)"
```

### 6.4 Lithops executor log (detailed worker invocation info)

Lithops writes per-executor logs to `/tmp/lithops-roller/logs/`:

```bash
# List recent executor logs
ls -lt /tmp/lithops-roller/logs/ | head -10

# Pattern: {executor_id}-M000.log = first map() call, M001 = first retry, etc.
# Check M000 for initial dispatch outcome
grep -E "(Starting function|activations|Success|failed)" \
    /tmp/lithops-roller/logs/<executor_id>-M000.log | tail -20

# Check M001 for retry outcome
grep -E "(Starting function|activations|Success|failed)" \
    /tmp/lithops-roller/logs/<executor_id>-M001.log | tail -20
```

### 6.5 Check if the process is still running

```bash
ps aux | grep "run_lithops\|run_backfill" | grep -v grep
```

**Expected output when healthy**:
```
roller  430024  0.0  0.0  bash run_backfill_00z.sh --from 2024-10
roller  441803  0.0  0.1  uv run run_lithops_ecmwf.py ...
roller  441807  0.4  0.5  python3 run_lithops_ecmwf.py --start-date 20250201 ...
```

If only the `bash` PID remains (no `python3`), the backfill is between months
(cleaning up and starting next).

### 6.6 Detect if backfill is stuck

The orchestrator should emit a new log line at least every ~60 seconds while running.
If the log file hasn't grown in >15 minutes:

```bash
# Check last modification time of the log
stat logs/backfill_00z/run_2024-10_onwards.log | grep Modify

# Compare with current time
date
```

If stuck for >20 minutes with no new lines AND the python3 process is alive, it may
be waiting in `fexec.wait()` on the second retry. This is expected — the 10-min
timeout will fire and GCS rescue will run.

---

## 7. GCS Bucket Audit Routines

### 7.1 Full audit — all months (requires gcsfs in uv env)

```bash
PYTHONPATH=/home/roller/.cache/uv/environments-v2/run-lithops-ecmwf-08a3f6790ffc62de/lib/python3.12/site-packages \
python3 -c "
import gcsfs, calendar
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')

months = []
y, m = 2024, 3
while (y, m) <= (2025, 12):
    months.append((y, m))
    m += 1
    if m > 12: m = 1; y += 1

print('Month       Expected  InGCS  Status')
print('-' * 50)
total_e = total_f = 0
for y, m in months:
    expected = calendar.monthrange(y, m)[1]
    path = f'gik-ecmwf-aws-tf/run_par_ecmwf/{y:04d}/{m:02d}/'
    try:
        items = fs.ls(path)
        found = len(items)
        status = 'COMPLETE' if found == expected else f'PARTIAL ({found}/{expected})'
    except Exception:
        found = 0
        status = 'MISSING'
    total_e += expected; total_f += found
    print(f'  {y:04d}-{m:02d}  {expected:8d}  {found:5d}  {status}')
print('-' * 50)
print(f'  TOTAL    {total_e:8d}  {total_f:5d}  ({total_f*100//total_e}%)')
"
```

### 7.2 Check a specific month

```bash
PYTHONPATH=/home/roller/.cache/uv/environments-v2/run-lithops-ecmwf-08a3f6790ffc62de/lib/python3.12/site-packages \
python3 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')
items = fs.ls('gik-ecmwf-aws-tf/run_par_ecmwf/2024/10/')
print(f'{len(items)} date directories found:')
for x in sorted(items): print(' ', x.split('/')[-1])
"
```

### 7.3 Check a specific date for correct file count

A healthy date has exactly **51 parquet files**:

```bash
PYTHONPATH=/home/roller/.cache/uv/environments-v2/run-lithops-ecmwf-08a3f6790ffc62de/lib/python3.12/site-packages \
python3 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')
files = fs.ls('gik-ecmwf-aws-tf/run_par_ecmwf/2024/10/20241015/00z/')
print(f'{len(files)} files:')
for f in sorted(files): print(' ', f.split('/')[-1])
"
```

**Expected**: 51 files (member_000.parquet through member_050.parquet).

Note: Occasionally a date will have only 10 files if the ECMWF ensemble was
partially available at the time of processing (e.g. `20250122` showed 10 files).
These dates should be re-processed.

### 7.4 Find partial dates (fewer than 51 files)

```bash
PYTHONPATH=/home/roller/.cache/uv/environments-v2/run-lithops-ecmwf-08a3f6790ffc62de/lib/python3.12/site-packages \
python3 -c "
import gcsfs, calendar
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')

partial = []
months = []
y, m = 2024, 3
while (y, m) <= (2025, 12):
    months.append((y, m))
    m += 1
    if m > 12: m = 1; y += 1

for y, m in months:
    path = f'gik-ecmwf-aws-tf/run_par_ecmwf/{y:04d}/{m:02d}/'
    try:
        date_dirs = fs.ls(path)
    except Exception:
        continue
    for dd in date_dirs:
        date = dd.split('/')[-1]
        run_path = f'{dd}/00z/'
        try:
            files = fs.ls(run_path)
            if len(files) < 51:
                partial.append((date, len(files)))
        except Exception:
            partial.append((date, 0))

if partial:
    print('Partial dates (< 51 files):')
    for date, n in sorted(partial):
        print(f'  {date}: {n} files')
else:
    print('All dates have 51 files.')
"
```

### 7.5 Quick gsutil check (if correct gcloud auth is active)

```bash
# List years
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/

# List months in a year
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/

# List dates in a month
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/10/

# Count files in a date
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/10/20241015/00z/ | wc -l
```

**Note**: `gsutil` uses the active `gcloud` auth account. The `cloudrun-deployer`
service account does not have `storage.objects.list` permissions. Use the
`gcsfs` approach above (with the JSON key) or switch to a user account:
```bash
gcloud auth application-default login
```

---

## 8. Backfill Operations

### 8.1 Run the full 2024-03 → 2025-12 backfill

```bash
# Dry run first — always verify dates/months before committing
bash run_backfill_00z.sh --dry-run

# Full run
nohup bash run_backfill_00z.sh > logs/backfill_00z/run_full.log 2>&1 &
echo "PID: $!"
```

### 8.2 Resume from a specific month

```bash
# If the backfill stopped at 2025-03, resume from there
nohup bash run_backfill_00z.sh --from 2025-03 > logs/backfill_00z/run_resume_2025-03.log 2>&1 &
echo "PID: $!"
```

### 8.3 Process a specific month only

```bash
bash run_backfill_00z.sh --from 2024-10 --to 2024-10
```

### 8.4 Process a single date manually

```bash
uv run run_lithops_ecmwf.py \
    --start-date 20241015 \
    --end-date   20241015 \
    --run        00 \     # or 06, 12, 18 — all four runs are supported
    --max-workers 5 \
    --yes
```

### 8.5 Backfill script options

| Flag | Default | Description |
|------|---------|-------------|
| `--from YYYY-MM` | `2024-03` | Start month (inclusive) |
| `--to YYYY-MM` | `2025-12` | End month (inclusive) |
| `--workers N` | `35` | Max Cloud Run workers |
| `--dry-run` | off | Preview only, no processing |

### 8.6 Monitor a running backfill

```bash
# Watch summary as months complete (live)
watch -n 30 'grep -E "^(OK|FAIL)" logs/backfill_00z/summary.log | tail -10'

# Tail live log (filtered)
tail -f logs/backfill_00z/run_2024-10_onwards.log | \
    grep -E "(GCS-OK|FAIL |succeeded|rescued|Month:|Completed|^OK|^FAIL)"
```

---

## 9. Recovery Playbook

### Scenario A: Backfill reports FAIL for a month

1. Check the per-month log:
   ```bash
   cat logs/backfill_00z/backfill_00z_2024-10.log | \
       grep -E "(GIVE UP|FAIL|succeeded)"
   ```
2. Check GCS directly (§7.2) — the month may actually be complete despite
   the FAIL report (GCS verification should prevent this, but check).
3. If genuinely incomplete, re-run that month:
   ```bash
   bash run_backfill_00z.sh --from 2024-10 --to 2024-10
   ```

### Scenario B: Process appears stuck (log not growing for >20 min)

1. Check if `python3 run_lithops_ecmwf.py` is still running:
   ```bash
   ps aux | grep run_lithops | grep -v grep
   ```
2. If alive: it is waiting in `fexec.wait()` on the 10-min timeout. Wait.
3. If dead: check the last lines of the log for the error.
4. If the bash wrapper is still alive but python died, the month will be
   marked FAIL and the next month will attempt. Check summary.log.

### Scenario C: DNS failure causes "0/N succeeded" despite GCS being complete

1. The GCS rescue block should now handle this automatically.
2. If you still see it, verify GCS manually (§7.2).
3. If GCS has all dates, the bash script will still have recorded FAIL
   (because `run_lithops_ecmwf.py` exited with code 1). Re-run that month:
   ```bash
   bash run_backfill_00z.sh --from 2024-10 --to 2024-10
   ```
   The GCS verification will immediately rescue all 31 dates and report OK in <1 minute.

### Scenario D: Cloud Run runtime out of date / image stale

Rebuild and redeploy:
```bash
# Rebuild image via Cloud Build
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project=e4drr-crafd \
    --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

# Delete old runtime
lithops runtime delete gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Deploy new runtime
lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml
```

### Scenario E: Max retries exceeded for some dates

After all 3 attempts, the summary will show FAIL with "Max retries exceeded".

1. Identify the specific dates from the month log.
2. Check if those dates actually have data on ECMWF S3:
   ```bash
   # Spot-check index availability
   uv run python -c "
   from run_lithops_ecmwf import validate_index_availability
   ok, msgs, members = validate_index_availability('20241006', '00')
   print(f'Valid: {ok}, messages: {msgs}, members: {members}')
   "
   ```
3. If S3 data exists, retry the specific dates:
   ```bash
   uv run run_lithops_ecmwf.py \
       --start-date 20241006 --end-date 20241006 \
       --run 00 --max-workers 5 --yes
   ```
4. If S3 data is genuinely missing for that date, it cannot be processed.

---

## 10. Cost Reference

| Unit | Cost |
|------|------|
| Per date processed | ~$0.026 USD |
| Full backfill (671 dates, 2024-03 → 2025-12) | ~$17.45 USD |
| Per month (~30 dates) | ~$0.78 USD |
| Cloud Run: 2 vCPU × 2 GB × ~8.5 min | ~$0.026/date |

**Wall time per month**: ~19–21 minutes
(10 min `fexec.wait()` timeout + ~1 min GCS rescue + ~8.5 min retry for remaining dates)

**Full backfill wall time**: ~22 months × 20 min ≈ 7.3 hours (sequential)

**Laptop internet requirement**: The orchestrator (`run_lithops_ecmwf.py`) must
maintain internet connectivity throughout the entire run. The Lithops monitor thread
polls `storage.googleapis.com` every ~1 second. A DNS/network interruption will cause
`f.status` to stall (see §5.4). The GCS rescue block mitigates data loss but the
orchestrator still needs to be alive and connected to drive the retry loop.

**Recommendation for long unattended backfills**: Run from a GCP VM in `europe-west3`
with `tmux` or `screen` so:
- No laptop dependency
- Lowest network latency to GCS and Cloud Run
- Session survives SSH disconnection
