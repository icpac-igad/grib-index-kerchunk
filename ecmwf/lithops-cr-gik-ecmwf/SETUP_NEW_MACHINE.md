# Running the ECMWF Lithops Backfill from a New Machine

> This guide covers everything needed to run the ECMWF parquet generation
> pipeline from any machine — a laptop, a GCP VM, or a colleague's workstation.
> The Cloud Run workers and GCS bucket are shared infrastructure; only the
> **orchestrator** (the machine that drives Lithops) needs to be set up.

---

## Table of Contents

1. [Recommended Machine Setup](#1-recommended-machine-setup)
2. [Service Account and IAM Permissions](#2-service-account-and-iam-permissions)
3. [Credential Setup](#3-credential-setup)
4. [Code Setup](#4-code-setup)
5. [Lithops Configuration](#5-lithops-configuration)
6. [Verify the Setup](#6-verify-the-setup)
7. [Running Backfills](#7-running-backfills)
8. [Retry and Recovery](#8-retry-and-recovery)
9. [Long-Running Sessions — tmux on a GCP VM](#9-long-running-sessions--tmux-on-a-gcp-vm)
10. [What Lives in GCP vs What Lives Locally](#10-what-lives-in-gcp-vs-what-lives-locally)

---

## 1. Recommended Machine Setup

### Option A — GCP VM (strongly recommended for long backfills)

A VM in the same region (`europe-west3`) as the Cloud Run service eliminates
network latency for the Lithops monitor thread polling GCS. DNS failures that
plagued long laptop runs (6-hour cleanup hangs) do not occur on GCP infrastructure.

```bash
# Create a small VM in europe-west3 — orchestration is lightweight
gcloud compute instances create ecmwf-orchestrator \
    --project=e4drr-crafd \
    --zone=europe-west3-a \
    --machine-type=e2-small \          # 2 vCPU, 2 GB — sufficient for orchestrator
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --scopes=cloud-platform            # gives the VM GCP API access via its own SA
```

SSH in and use **tmux** so the session survives disconnection:
```bash
gcloud compute ssh ecmwf-orchestrator --zone=europe-west3-a
tmux new -s backfill
```

### Option B — Laptop / workstation

Works fine for short runs (single month, spot checks). For multi-hour backfills:
- Keep the machine awake and internet connected throughout
- A DNS hiccup causes Lithops monitor thread to hang (GCS rescue mitigates data
  loss, but the orchestrator process can freeze for hours — see OPERATIONS_GUIDE.md §5.4)
- Use `nohup ... &` and monitor with log tails rather than foreground runs

### Minimum requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Linux / macOS | Windows via WSL2 |
| Python | 3.10+ | Managed automatically by `uv` |
| Internet | Stable | Required for Lithops GCS polling |
| Disk | 500 MB | For uv cache and logs only |
| RAM | 512 MB | Orchestrator is lightweight |

---

## 2. Service Account and IAM Permissions

The pipeline uses a single GCP service account for all operations:

**Service account**: `ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com`

This account needs the following IAM roles:

| Role | Resource | Why |
|------|----------|-----|
| `roles/run.invoker` | Cloud Run service `ecmwf-lithops-runtime` | Lithops invokes workers via HTTP POST |
| `roles/storage.objectAdmin` | Bucket `gik-ecmwf-aws-tf` | Write parquet output, read results |
| `roles/storage.objectAdmin` | Bucket `lithops-europe-west3-3b6921` | Lithops job staging (function pickles, result pickles) |
| `roles/run.admin` | Project `e4drr-crafd` | Deploy/delete/list Lithops runtimes |
| `roles/artifactregistry.reader` | Registry `gcr.io/e4drr-crafd` | Pull runtime container image |
| `roles/cloudbuild.builds.editor` | Project `e4drr-crafd` | Rebuild runtime image (only if Dockerfile changes) |
| `roles/iam.serviceAccountUser` | Service account itself | Impersonate SA during Cloud Build |

### Verify current permissions

```bash
gcloud projects get-iam-policy e4drr-crafd \
    --flatten="bindings[].members" \
    --filter="bindings.members:ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com" \
    --format="table(bindings.role)"
```

### Grant missing permissions (if setting up fresh)

```bash
SA="ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com"
PROJECT="e4drr-crafd"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" --role="roles/run.invoker"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" --role="roles/artifactregistry.reader"
```

---

## 3. Credential Setup

### 3.1 Get the service account key

The key file (`ecmwf-lithops-deployer-key.json`) must be present at:
```
lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json
```

**If you already have the key** (copy from original machine):
```bash
# From original machine — scp to new machine
scp service_account/ecmwf-lithops-deployer-key.json \
    user@new-machine:/path/to/lithops_cr_ecmwf_gik/service_account/
```

**If you need to generate a new key**:
```bash
gcloud iam service-accounts keys create \
    service_account/ecmwf-lithops-deployer-key.json \
    --iam-account=ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
    --project=e4drr-crafd
```

> **Security**: Never commit the key JSON to git. The `.gitignore` already
> excludes `service_account/*.json`.

### 3.2 Activate for gcloud / gsutil (optional but useful for GCS checks)

```bash
gcloud auth activate-service-account \
    --key-file=service_account/ecmwf-lithops-deployer-key.json \
    --project=e4drr-crafd
```

### 3.3 Set Application Default Credentials (ADC) — required for gcsfs

The Python script calls `gcsfs.GCSFileSystem()` (no explicit token) in two places:

| Callsite | Runs on | Auth source |
|----------|---------|-------------|
| `upload_to_gcs()` | Cloud Run worker | Worker's own SA identity (automatic) |
| GCS verification rescue block | **Orchestrator** (your machine) | ADC ← needs setup |

`gcloud auth activate-service-account` (step 3.2) sets credentials for gcloud/gsutil
but **not** for Google client libraries (gcsfs, google-cloud-storage). You must also
set ADC so that gcsfs on the orchestrator can authenticate:

```bash
# Option A — set env var (simplest; add to ~/.bashrc to persist)
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service_account/ecmwf-lithops-deployer-key.json"

# Option B — populate ADC file via gcloud
gcloud auth application-default login \
    --impersonate-service-account=ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
```

Option A is recommended for automated / tmux runs — no browser needed.

> **On a GCP VM** created with `--scopes=cloud-platform` (see §1), the metadata service
> provides credentials automatically and neither option A nor B is needed.

### 3.4 ECMWF S3 access — no credentials needed

ECMWF open data on S3 (`s3://ecmwf-forecasts`) is **anonymous**. The pipeline
sets `AWS_NO_SIGN_REQUEST=YES` inside each Cloud Run worker automatically. No
AWS credentials are required on the orchestrator machine.

---

## 4. Code Setup

### 4.1 Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart shell
uv --version              # confirm: uv 0.5+
```

### 4.2 Clone the repository

```bash
git clone <repo-url> cno-e4drr
cd cno-e4drr/devops/lithops_cr_ecmwf_gik
```

### 4.3 Place the service account key

```bash
mkdir -p service_account
cp /path/to/ecmwf-lithops-deployer-key.json service_account/
```

### 4.4 Verify uv resolves dependencies

`run_lithops_ecmwf.py` uses PEP 723 inline script dependencies — `uv run`
installs them automatically on first run (cached thereafter):

```bash
uv run python -c "import lithops; print(lithops.__version__)"
# Should print: 3.6.3 (or similar)
```

Dependencies installed automatically by uv:
- `lithops` — orchestration framework
- `google-cloud-storage`, `google-api-python-client`, `google-auth` — GCP APIs
- `gcsfs` — GCS filesystem interface for result verification
- `pandas`, `numpy`, `pyarrow` — parquet generation
- `s3fs`, `fsspec` — ECMWF S3 access
- `requests` — template download fallback

---

## 5. Lithops Configuration

`lithops_config.yaml` (already in the repo — no changes needed on a new machine):

```yaml
lithops:
    backend: gcp_cloudrun
    storage: gcp_storage

gcp:
    project_name: e4drr-crafd
    region: europe-west3
    credentials_path: service_account/ecmwf-lithops-deployer-key.json  # ← key file

gcp_cloudrun:
    runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime   # already deployed
    runtime_memory: 2048
    runtime_cpu: 2
    runtime_timeout: 3600
    max_workers: 35       # Cloud Run maxScale — set at deploy time, not changeable per-run
    min_workers: 0
    worker_processes: 1

gcp_storage:
    bucket: gik-ecmwf-aws-tf
    region: europe-west3
```

**Important**: The `credentials_path` is relative to the directory where you run
the script from. Always run from inside `lithops_cr_ecmwf_gik/`.

### What you do NOT need to do on a new machine

- **No runtime redeploy** — the runtime image (`gcr.io/e4drr-crafd/ecmwf-lithops-runtime`)
  is already deployed in Cloud Run. Any machine with the right credentials can invoke it.
- **No Docker** — Docker is only needed if you rebuild the runtime image (rare).
- **No Lithops init** — Lithops reads `lithops_config.yaml` automatically.
- **No Terraform** — infrastructure is already provisioned.

### When you DO need to redeploy the runtime

Only if `Dockerfile` or `cloudbuild.yaml` changes (e.g. updating Python dependencies
baked into the image):

```bash
# Step 1: Rebuild image via Cloud Build
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project=e4drr-crafd \
    --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

# Step 2: Delete old runtime
uv run lithops runtime delete gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Step 3: Deploy new runtime (picks up new maxScale from lithops_config.yaml)
uv run lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Verify
uv run lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

---

## 6. Verify the Setup

### 6.1 Confirm credentials work

```bash
# GCS read access
PYTHONPATH=$(uv run python -c "import site; print(site.getsitepackages()[0])") \
python3 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')
items = fs.ls('gik-ecmwf-aws-tf/run_par_ecmwf/2024/03/')
print(f'GCS OK: {len(items)} dates found in 2024/03')
"
```

### 6.2 Confirm Lithops can reach Cloud Run

```bash
uv run python -c "
import lithops
fe = lithops.FunctionExecutor(config_file='lithops_config.yaml')
print('Lithops FunctionExecutor: OK')
print(f'Backend: {fe.backend}')
"
```

### 6.3 Dry run a single month

```bash
uv run run_lithops_ecmwf.py \
    --start-date 20240301 \
    --end-date   20240331 \
    --run 00 \
    --max-workers 35 \
    --yes \
    --dry-run
# Should print 31 dates without invoking anything
```

### 6.4 Test a single date (live — costs ~$0.026)

```bash
uv run run_lithops_ecmwf.py \
    --date 20240315 \
    --run 00 \
    --max-workers 5 \
    --yes
```

Check GCS for the result:
```bash
uv run python -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')
files = fs.ls('gik-ecmwf-aws-tf/run_par_ecmwf/2024/03/20240315/00z/')
print(f'{len(files)} parquet files (expect 51)')
"
```

---

## 7. Running Backfills

### Full historical backfill (2024-03 → 2025-12, 00Z only)

```bash
# From inside lithops_cr_ecmwf_gik/ with tmux active
nohup bash run_backfill_00z.sh \
    > logs/backfill_00z/run_full.log 2>&1 &
echo "PID: $!"
```

### Resume from a specific month

```bash
nohup bash run_backfill_00z.sh --from 2025-06 \
    > logs/backfill_00z/run_resume_2025-06.log 2>&1 &
```

### Specific month range

```bash
bash run_backfill_00z.sh --from 2024-10 --to 2024-12
```

### Partial month (e.g. Feb 1–18 2026)

The backfill script works in whole months. For partial months, call the script directly:

```bash
nohup uv run run_lithops_ecmwf.py \
    --start-date 20260201 \
    --end-date   20260218 \
    --run 00 \
    --max-workers 35 \
    --yes \
    > logs/backfill_00z/backfill_2026-02-partial.log 2>&1 &
```

### Monitor progress

```bash
# Summary (one line per completed month)
grep -E "^(OK|FAIL)" logs/backfill_00z/summary.log

# Live log (filtered — removes stack trace noise)
tail -f logs/backfill_00z/run_full.log | \
    grep -E "(GCS-OK|FAIL |succeeded|rescued|Month:|Completed in|^OK|^FAIL)"

# Check running processes
ps aux | grep "run_lithops\|run_backfill" | grep -v grep
```

---

## 8. Retry and Recovery

### Automatic retry (built-in)

The pipeline has **3 automatic retry attempts** per batch with 30s delay between:
1. Attempt 1 — all dates dispatched; HTTP 429/500 dates collected as failures
2. **GCS verification** — dates already in GCS are rescued (handles monitor interruptions)
3. Attempt 2 — only genuinely missing dates retried
4. Attempt 3 — final attempt for any still-missing dates

### If a month is marked FAIL in summary.log

```bash
# Re-run that month — GCS verification will rescue completed dates instantly
bash run_backfill_00z.sh --from 2025-11 --to 2025-11
```

### If the orchestrator process gets stuck (cleanup hang)

Symptom: log stopped growing but `python3 run_lithops_ecmwf.py` process is still alive.
Root cause: Lithops background thread blocked on stale socket after many retries.

```bash
# Find the stuck python PID
ps aux | grep "python3 run_lithops" | grep -v grep | awk '{print $2}'

# Kill it (SIGTERM first — Lithops catches it gracefully)
kill -TERM <PID>

# Data is safe — GCS already has all completed dates
# Re-run the month to get a clean OK in summary.log
bash run_backfill_00z.sh --from <YYYY-MM> --to <YYYY-MM>
```

### GCS audit — verify what's actually complete

```bash
uv run python -c "
import gcsfs, calendar
fs = gcsfs.GCSFileSystem(token='service_account/ecmwf-lithops-deployer-key.json')
months = []
y, m = 2024, 3
while (y, m) <= (2025, 12):
    months.append((y, m))
    m += 1
    if m > 12: m = 1; y += 1

print(f'{\"Month\":<12}{\"Expected\":>10}{\"InGCS\":>8}  Status')
print('-' * 45)
for y, m in months:
    exp = calendar.monthrange(y, m)[1]
    try:
        n = len(fs.ls(f'gik-ecmwf-aws-tf/run_par_ecmwf/{y:04d}/{m:02d}/'))
        st = 'COMPLETE' if n == exp else f'PARTIAL ({n}/{exp})'
    except:
        n, st = 0, 'MISSING'
    print(f'  {y:04d}-{m:02d}  {exp:>10}  {n:>6}  {st}')
"
```

### Poller pattern — queue a job after another finishes

When you want job B to start only after job A completes, use a `kill -0` poll loop
instead of `wait` (which only works for child processes of the same shell):

```bash
nohup bash -c '
cd /path/to/lithops_cr_ecmwf_gik
while kill -0 <JOB_A_PID> 2>/dev/null; do sleep 30; done
echo "[$(date)] Job A done, starting Job B..."
bash run_backfill_00z.sh --from 2026-01 --to 2026-01
' > logs/backfill_00z/poller.log 2>&1 &
```

---

## 9. Long-Running Sessions — tmux on a GCP VM

For unattended backfills (~3–7 hours), a GCP VM with tmux is strongly recommended.

### Create and connect

```bash
# Create VM (one-time)
gcloud compute instances create ecmwf-orchestrator \
    --project=e4drr-crafd \
    --zone=europe-west3-a \
    --machine-type=e2-small \
    --image-family=debian-12 \
    --image-project=debian-cloud

# SSH in
gcloud compute ssh ecmwf-orchestrator --zone=europe-west3-a
```

### Setup on the VM (one-time)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Clone repo
git clone <repo-url> ~/cno-e4drr
cd ~/cno-e4drr/devops/lithops_cr_ecmwf_gik

# Copy service account key (from local machine)
# On LOCAL machine:
gcloud compute scp service_account/ecmwf-lithops-deployer-key.json \
    ecmwf-orchestrator:~/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/ \
    --zone=europe-west3-a

# Install tmux
sudo apt-get install -y tmux
```

### Start a backfill in tmux

```bash
tmux new -s backfill
cd ~/cno-e4drr/devops/lithops_cr_ecmwf_gik

# Start backfill (runs inside tmux — survives SSH disconnect)
bash run_backfill_00z.sh --from 2024-10 2>&1 | tee logs/backfill_00z/run.log

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t backfill
```

### Stop the VM when done (saves cost)

```bash
gcloud compute instances stop ecmwf-orchestrator --zone=europe-west3-a
# Restart later:
gcloud compute instances start ecmwf-orchestrator --zone=europe-west3-a
```

> VM cost: `e2-small` ≈ $0.017/hr. A 7-hour backfill costs ~$0.12 in VM compute
> — negligible compared to the ~$17 Cloud Run cost.

---

## 10. What Lives in GCP vs What Lives Locally

| Component | Where | Notes |
|-----------|-------|-------|
| Cloud Run service `ecmwf-lithops-runtime` | GCP `europe-west3` | Already deployed — shared by all orchestrators |
| GCS output bucket `gik-ecmwf-aws-tf` | GCP `europe-west3` | Parquet files written here |
| Lithops staging bucket `lithops-europe-west3-3b6921` | GCP `europe-west3` | Job payloads & result pickles — auto-managed by Lithops |
| Container image `gcr.io/e4drr-crafd/ecmwf-lithops-runtime` | GCP Artifact Registry | Re-built only when Dockerfile changes |
| `run_lithops_ecmwf.py` | **Local** (orchestrator machine) | Serialized to workers via cloudpickle |
| `lithops_config.yaml` | **Local** | Points to key file — must be in same directory |
| `service_account/ecmwf-lithops-deployer-key.json` | **Local** | Never committed to git |
| `logs/backfill_00z/` | **Local** | Orchestrator logs only — not in GCP |
| ECMWF S3 source data | AWS `eu-central-1` | `s3://ecmwf-forecasts` — public, no credentials |
