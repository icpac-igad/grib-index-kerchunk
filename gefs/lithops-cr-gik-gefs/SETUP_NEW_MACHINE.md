# Running the GEFS Lithops Pipeline from a New Machine

> This guide covers everything needed to run the GEFS parquet generation
> pipeline from any machine — a laptop, a GCP VM, or a colleague's workstation.
> The Cloud Run workers and GCS bucket are shared infrastructure; only the
> **orchestrator** (the machine that drives Lithops) needs to be set up.
>
> The pipeline supports two GCP projects:
>
> | Project | Config file | Service Account | Bucket | Region |
> |---------|-------------|-----------------|--------|--------|
> | `e4drr-crafd` | `lithops_config.yaml` | `gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com` | `gik-gefs-aws-tf` | `us-east1` |
> | `sewaa-416306` | `lithops_config_sewaa.yaml` | `gefs-lithops-deployer@sewaa-416306.iam.gserviceaccount.com` | `gik-gefs-sewaa-tf` | `us-east1` |
>
> Steps below use shell variables `$PROJECT` and `$SA` so a single command set
> works for either project.

---

## Table of Contents

1. [Recommended Machine Setup](#1-recommended-machine-setup)
2. [Service Account and IAM Permissions](#2-service-account-and-iam-permissions)
3. [Credential Setup](#3-credential-setup)
4. [Code Setup](#4-code-setup)
5. [Lithops Configuration](#5-lithops-configuration)
6. [Verify the Setup](#6-verify-the-setup)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Retry and Recovery](#8-retry-and-recovery)
9. [Long-Running Sessions — tmux on a GCP VM](#9-long-running-sessions--tmux-on-a-gcp-vm)
10. [What Lives in GCP vs What Lives Locally](#10-what-lives-in-gcp-vs-what-lives-locally)

---

## 1. Recommended Machine Setup

### Option A — GCP VM (strongly recommended for long backfills)

A VM in the same region (`us-east1`) as the Cloud Run service eliminates
network latency for the Lithops monitor thread polling GCS.

```bash
# e4drr-crafd project
gcloud compute instances create gefs-orchestrator \
    --project=e4drr-crafd \
    --zone=us-east1-b \
    --machine-type=e2-small \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --scopes=cloud-platform

# --- OR --- sewaa-416306 project
gcloud compute instances create gefs-orchestrator \
    --project=sewaa-416306 \
    --zone=us-east1-b \
    --machine-type=e2-small \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --scopes=cloud-platform
```

SSH in and use **tmux** so the session survives disconnection:
```bash
gcloud compute ssh gefs-orchestrator --zone=us-east1-b
tmux new -s gefs
```

### Option B — Laptop / workstation

Works fine for short runs (single date, spot checks). For multi-day backfills,
use `nohup ... &` and monitor with log tails rather than foreground runs.

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

### e4drr-crafd project (one-time setup — dedicated SA, not shared with ECMWF)

```bash
PROJECT="e4drr-crafd"
SA="gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com"

# Preferred: terraform apply from devops/lithops_cr_gefs_gik/terraform/
# (creates SA, IAM bindings, and buckets in one shot — see DEPLOYMENT_GUIDE.md)
#
# Manual fallback (equivalent to the terraform module):
gcloud iam service-accounts create gefs-lithops-deployer \
    --display-name="GEFS Lithops Deployer" \
    --project=$PROJECT

for ROLE in roles/run.invoker roles/run.admin roles/storage.objectAdmin \
            roles/artifactregistry.reader roles/cloudbuild.builds.editor; do
    gcloud projects add-iam-policy-binding $PROJECT \
        --member="serviceAccount:$SA" --role="$ROLE"
done

gcloud iam service-accounts add-iam-policy-binding $SA \
    --member="serviceAccount:$SA" \
    --role="roles/iam.serviceAccountUser" \
    --project=$PROJECT
```

### sewaa-416306 project (one-time setup)

Create a dedicated service account for GEFS:

```bash
PROJECT="sewaa-416306"
SA="gefs-lithops-deployer@sewaa-416306.iam.gserviceaccount.com"

# Create the SA
gcloud iam service-accounts create gefs-lithops-deployer \
    --display-name="GEFS Lithops Deployer" \
    --project=$PROJECT

# Grant required roles
for ROLE in roles/run.invoker roles/run.admin roles/storage.objectAdmin \
            roles/artifactregistry.reader roles/cloudbuild.builds.editor; do
    gcloud projects add-iam-policy-binding $PROJECT \
        --member="serviceAccount:$SA" --role="$ROLE"
done

# Allow SA to impersonate itself during Cloud Build
gcloud iam service-accounts add-iam-policy-binding $SA \
    --member="serviceAccount:$SA" \
    --role="roles/iam.serviceAccountUser" \
    --project=$PROJECT
```

### Required IAM roles (both projects)

| Role | Why |
|------|-----|
| `roles/run.invoker` | Lithops invokes workers via HTTP POST |
| `roles/storage.objectAdmin` | Write parquet output, read/write Lithops staging |
| `roles/run.admin` | Deploy/delete/list Lithops runtimes |
| `roles/artifactregistry.reader` | Pull runtime container image |
| `roles/cloudbuild.builds.editor` | Rebuild runtime image when Dockerfile changes |
| `roles/iam.serviceAccountUser` | Impersonate SA during Cloud Build |

---

## 3. Credential Setup

### 3.1 Get the service account key

**e4drr-crafd** — generate a key for the dedicated GEFS SA:
```bash
mkdir -p service_account
gcloud iam service-accounts keys create \
    service_account/gefs-lithops-deployer-key.json \
    --iam-account=gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
    --project=e4drr-crafd
```
This is the path `lithops_config.yaml` expects (`credentials_path: service_account/gefs-lithops-deployer-key.json`).

**sewaa-416306** — generate a key for the new SA:
```bash
mkdir -p service_account
gcloud iam service-accounts keys create \
    service_account/sewaa-gefs-sa-key.json \
    --iam-account=gefs-lithops-deployer@sewaa-416306.iam.gserviceaccount.com \
    --project=sewaa-416306
```

> **Security**: Never commit key JSON to git. The `.gitignore` already
> excludes `service_account/*.json`.

### 3.2 Set Application Default Credentials (ADC)

Required for `gcsfs.GCSFileSystem()` calls in the orchestrator (GCS verification):

```bash
# e4drr-crafd
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service_account/gefs-lithops-deployer-key.json"

# sewaa-416306
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service_account/sewaa-gefs-sa-key.json"
```

Add to `~/.bashrc` to persist. On a GCP VM created with `--scopes=cloud-platform`,
credentials are provided automatically by the metadata service — no key file needed.

### 3.3 NOAA S3 access — no credentials needed

GEFS data on S3 (`s3://noaa-gefs-pds`) is **public**. The pipeline sets
`AWS_NO_SIGN_REQUEST=YES` inside each Cloud Run worker automatically.

---

## 4. Code Setup

### 4.1 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version  # confirm: uv 0.5+
```

### 4.2 Clone the repository

```bash
git clone <repo-url> cno-e4drr
cd cno-e4drr/devops/lithops_cr_gefs_gik
```

### 4.3 Place the service account key

```bash
mkdir -p service_account
# e4drr-crafd: copy gefs-lithops-deployer-key.json (generated in §3.1)
# sewaa-416306: copy sewaa-gefs-sa-key.json (generated in §3.1)
```

### 4.4 Verify uv resolves dependencies

`run_lithops_gefs.py` uses PEP 723 inline script dependencies — `uv run`
installs them automatically on first run (cached thereafter):

```bash
uv run python -c "import lithops; print(lithops.__version__)"
# Should print: 3.6.3 (or similar)
```

---

## 5. Lithops Configuration

Two config files are provided:

| File | Project | Region | Bucket |
|------|---------|--------|--------|
| `lithops_config.yaml` | `e4drr-crafd` | `us-east1` | `gik-gefs-aws-tf` |
| `lithops_config_sewaa.yaml` | `sewaa-416306` | `us-east1` | `gik-gefs-sewaa-tf` |

Both point `credentials_path` to `service_account/<key>.json` relative to the
directory where you run the script from. Always run from inside `lithops_cr_gefs_gik/`.

### Selecting a config at runtime

```bash
# e4drr-crafd (default — uses lithops_config.yaml automatically)
uv run run_lithops_gefs.py --date 20260206

# sewaa-416306
LITHOPS_CONFIG_FILE=lithops_config_sewaa.yaml uv run run_lithops_gefs.py --date 20260206
```

### GCS bucket setup for sewaa (one-time)

```bash
# Create the output bucket in us-east1 (co-located with NOAA GEFS S3 data)
gsutil mb -p sewaa-416306 -l us-east1 gs://gik-gefs-sewaa-tf

# Grant SA access
gsutil iam ch \
    serviceAccount:gefs-lithops-deployer@sewaa-416306.iam.gserviceaccount.com:roles/storage.objectAdmin \
    gs://gik-gefs-sewaa-tf
```

### When you need to (re)deploy the runtime

Only if `Dockerfile` or `cloudbuild*.yaml` changes:

```bash
# e4drr-crafd
gcloud builds submit --config=cloudbuild.yaml --project=e4drr-crafd

# sewaa-416306
gcloud builds submit --config=cloudbuild-sewaa.yaml --project=sewaa-416306

# Delete old runtime (choose appropriate config)
uv run lithops runtime delete gcr.io/e4drr-crafd/gefs-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Deploy new runtime
uv run lithops runtime deploy gcr.io/e4drr-crafd/gefs-lithops-runtime \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Verify
uv run lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

---

## 6. Verify the Setup

### 6.1 Confirm credentials work

```bash
# e4drr-crafd
uv run python -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='service_account/gefs-lithops-deployer-key.json')
items = fs.ls('gik-gefs-aws-tf/run_par_gefs/')
print(f'GCS OK: {len(items)} dates found')
"

# sewaa-416306
uv run python -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='service_account/sewaa-gefs-sa-key.json')
items = fs.ls('gik-gefs-sewaa-tf/run_par_gefs/')
print(f'GCS OK: {len(items)} dates found')
"
```

### 6.2 Confirm Lithops can reach Cloud Run

```bash
# e4drr-crafd
uv run python -c "
import lithops
fe = lithops.FunctionExecutor(config_file='lithops_config.yaml')
print(f'Backend: {fe.backend}  OK')
"

# sewaa-416306
uv run python -c "
import lithops
fe = lithops.FunctionExecutor(config_file='lithops_config_sewaa.yaml')
print(f'Backend: {fe.backend}  OK')
"
```

### 6.3 Dry run

```bash
# e4drr-crafd
uv run run_lithops_gefs.py --days-back 3 --dry-run

# sewaa-416306
LITHOPS_CONFIG_FILE=lithops_config_sewaa.yaml \
  uv run run_lithops_gefs.py --days-back 3 --dry-run
```

### 6.4 Test a single date (live)

```bash
uv run run_lithops_gefs.py --date 20260206 --max-workers 5
```

---

## 7. Running the Pipeline

### Single date

```bash
uv run run_lithops_gefs.py --date 20260206
```

### Last N days

```bash
uv run run_lithops_gefs.py --days-back 30 --max-workers 20
```

### Date range

```bash
uv run run_lithops_gefs.py \
    --start-date 20260101 \
    --end-date   20260228 \
    --max-workers 20
```

### With sewaa-416306

```bash
LITHOPS_CONFIG_FILE=lithops_config_sewaa.yaml \
  uv run run_lithops_gefs.py \
    --start-date 20260101 \
    --end-date   20260228 \
    --max-workers 20
```

### Background / long-running (nohup)

```bash
mkdir -p logs
nohup uv run run_lithops_gefs.py \
    --start-date 20260101 \
    --end-date   20260228 \
    --max-workers 20 \
    > logs/backfill_2026-01-02.log 2>&1 &
echo "PID: $!"
```

### Monitor progress

```bash
# Live log
tail -f logs/backfill_2026-01-02.log

# Summarise results
grep -E "^(OK|FAIL)" logs/backfill_2026-01-02.log

# Check processes
ps aux | grep "run_lithops_gefs" | grep -v grep
```

---

## 8. Retry and Recovery

### Identify missing dates in GCS

```bash
# e4drr-crafd
uv run python -c "
import gcsfs, sys
fs = gcsfs.GCSFileSystem(token='service_account/gefs-lithops-deployer-key.json')
bucket = 'gik-gefs-aws-tf'
prefix = 'run_par_gefs'
try:
    dates = [p.split('/')[-1] for p in fs.ls(f'{bucket}/{prefix}/')]
    print(f'{len(dates)} dates in GCS')
    for d in sorted(dates)[-10:]:
        n = len(fs.ls(f'{bucket}/{prefix}/{d}/00z/'))
        print(f'  {d}: {n} files')
except Exception as e:
    print(f'Error: {e}')
"
```

### Re-run failed dates

```bash
# Re-run a single date (GCS already has other completed dates — safe to retry)
uv run run_lithops_gefs.py --date 20260115
```

### If the orchestrator process hangs

```bash
# Find PID
ps aux | grep "run_lithops_gefs" | grep -v grep | awk '{print $2}'

# Kill gracefully
kill -TERM <PID>

# Re-run the affected date range — already-uploaded files won't be duplicated
uv run run_lithops_gefs.py --start-date 20260115 --end-date 20260120
```

---

## 9. Long-Running Sessions — tmux on a GCP VM

```bash
# Create VM (one-time) — us-east1 for co-location with NOAA S3
gcloud compute instances create gefs-orchestrator \
    --project=sewaa-416306 \
    --zone=us-east1-b \
    --machine-type=e2-small \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --scopes=cloud-platform

# SSH in
gcloud compute ssh gefs-orchestrator --zone=us-east1-b --project=sewaa-416306
```

### One-time VM setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Install tmux
sudo apt-get install -y tmux

# Clone repo
git clone <repo-url> ~/cno-e4drr
cd ~/cno-e4drr/devops/lithops_cr_gefs_gik

# Copy SA key from local machine (if not using --scopes=cloud-platform)
# On LOCAL machine:
gcloud compute scp service_account/sewaa-gefs-sa-key.json \
    gefs-orchestrator:~/cno-e4drr/devops/lithops_cr_gefs_gik/service_account/ \
    --zone=us-east1-b --project=sewaa-416306
```

### Start a run in tmux

```bash
tmux new -s gefs
cd ~/cno-e4drr/devops/lithops_cr_gefs_gik

LITHOPS_CONFIG_FILE=lithops_config_sewaa.yaml \
  uv run run_lithops_gefs.py \
    --start-date 20260101 \
    --end-date   20260228 \
    --max-workers 20 \
  2>&1 | tee logs/backfill.log

# Detach: Ctrl+B then D
# Reattach: tmux attach -t gefs
```

### Stop VM when done (saves cost)

```bash
gcloud compute instances stop gefs-orchestrator --zone=us-east1-b --project=sewaa-416306
# Restart:
gcloud compute instances start gefs-orchestrator --zone=us-east1-b --project=sewaa-416306
```

> VM cost: `e2-small` ≈ $0.017/hr — negligible compared to Cloud Run costs.

---

## 10. What Lives in GCP vs What Lives Locally

| Component | Where | Notes |
|-----------|-------|-------|
| Cloud Run service `gefs-lithops-runtime` | GCP `us-east1` | Shared by all orchestrators in the project |
| GCS output bucket `gik-gefs-aws-tf` | GCP `us-east1` (e4drr) | Parquet files |
| GCS output bucket `gik-gefs-sewaa-tf` | GCP `us-east1` (sewaa) | Parquet files |
| Lithops staging bucket | GCP `us-east1` | Job payloads & results — auto-managed by Lithops |
| Container image `gcr.io/<project>/gefs-lithops-runtime` | GCP Artifact Registry | Rebuilt only when Dockerfile changes |
| `run_lithops_gefs.py` | **Local** (orchestrator) | Serialized to workers via cloudpickle |
| `lithops_config*.yaml` | **Local** | Points to key file — must be in same directory |
| `service_account/*.json` | **Local** | Never committed to git |
| `logs/` | **Local** | Orchestrator logs only |
| GEFS S3 source data | AWS `us-east-1` | `s3://noaa-gefs-pds` — public, no credentials |
