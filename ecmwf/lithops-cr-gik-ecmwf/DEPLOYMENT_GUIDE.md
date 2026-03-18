# ECMWF Parquet Generation - Lithops Cloud Run Deployment Guide

**Date**: 2026-02-08
**Project**: e4drr-crafd
**Region**: us-central1
**Service**: ecmwf-parquet-worker
**Source Application**: /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/
**Orchestration**: Lithops Python Library

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Step 1: Create Deployer Service Account](#step-1-create-deployer-service-account)
5. [Step 2: Create Worker Script](#step-2-create-worker-script)
6. [Step 3: Build Docker Image](#step-3-build-docker-image)
7. [Step 4: Deploy to Cloud Run](#step-4-deploy-to-cloud-run)
8. [Step 5: Configure Lithops Client](#step-5-configure-lithops-client)
9. [Step 6: Execute Daily Tasks](#step-6-execute-daily-tasks)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)
12. [File Structure](#file-structure)

---

## Overview

This deployment uses **Lithops** to orchestrate Cloud Run workers for generating ECMWF parquet files. Each worker processes a single date, creating Stage 3 parquet files and uploading them to GCS.

### What is Lithops?

[Lithops](https://github.com/lithops-cloud/lithops) is a Python framework for serverless computing that abstracts different backends (AWS Lambda, Google Cloud Run, IBM Cloud Functions, etc.). It allows you to:

- Execute Python functions in parallel across cloud workers
- Map tasks over data ranges (e.g., date ranges)
- Automatically handle serialization, networking, and error handling
- Monitor execution and gather results

### Workflow

```
┌──────────────────┐
│  Local Machine   │
│  lithops.call()  │
└────────┬─────────┘
         │ HTTP POST with date payload
         ▼
┌────────────────────────────────────┐
│  Cloud Run: ecmwf-parquet-worker   │
│  - Receives date string            │
│  - Runs GIK three-stage pipeline   │
│  - Uploads parquets to GCS         │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  GCS Bucket: gik-ecmwf-aws-tf      │
│  gs://gik-ecmwf-aws-tf/            │
│    run_par_ecmwf/YYYYMMDD_00z/     │
│      ├─ stage3_control_final.parquet
│      ├─ stage3_ens01_final.parquet │
│      └─ ...                        │
└────────────────────────────────────┘
```

### Use Cases

1. **Batch Processing**: Generate parquets for a date range (e.g., last 30 days)
2. **Daily Automation**: Scheduled cron job to process yesterday's forecast
3. **Backfill**: Process historical dates in parallel
4. **On-Demand**: Process specific dates as needed

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Lithops Orchestration                       │
│                                                                 │
│  ┌──────────────┐        ┌──────────────┐                     │
│  │  Local Code  │ ─map─► │ Date Range   │                     │
│  │  lithops.py  │        │ [20260201,   │                     │
│  └──────────────┘        │  20260202,   │                     │
│                          │  20260203]   │                     │
│                          └──────┬───────┘                     │
└─────────────────────────────────┼─────────────────────────────┘
                                  │ Parallel HTTP invocations
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
         ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
         │ Cloud Run   │ │ Cloud Run   │ │ Cloud Run   │
         │ Instance 1  │ │ Instance 2  │ │ Instance 3  │
         │ (20260201)  │ │ (20260202)  │ │ (20260203)  │
         └─────┬───────┘ └─────┬───────┘ └─────┬───────┘
               │               │               │
               └───────────────┼───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  GCS Bucket          │
                    │  gik-ecmwf-aws-tf    │
                    │  run_par_ecmwf/      │
                    └──────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestrator | Lithops Python | Task distribution and monitoring |
| Worker Runtime | Cloud Run | Serverless container execution |
| Processing Code | GIK Pipeline | ECMWF GRIB → Parquet conversion |
| Data Source | ECMWF S3 | Public GRIB2 ensemble forecasts |
| Output Storage | GCS | Parquet file hosting |
| Build System | Cloud Build | Docker image creation |
| Image Registry | Artifact Registry | Docker image storage |

---

## Prerequisites

- `gcloud` CLI authenticated with project access
- `terraform` >= 1.5.0
- Python 3.10+ with `lithops` installed
- Access to e4drr-crafd GCP project
- GCS bucket `gik-ecmwf-aws-tf` with write permissions
- ECMWF source code at `/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/`

**Install Lithops:**

```bash
pip install lithops
# or
micromamba install -c conda-forge lithops
```

---

## Step 1: Create Deployer Service Account

This service account will have permissions for Cloud Build, Artifact Registry, Cloud Run, and GCS.

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/

# Switch to user account with admin permissions
gcloud config set account nkalladath@icpac.net

# Initialize and apply Terraform
terraform init
terraform apply -var="project_id=e4drr-crafd"

# Extract service account key
terraform output -raw service_account_key_private | base64 -d > ecmwf-lithops-deployer-key.json

# Activate the service account
gcloud auth activate-service-account --key-file=ecmwf-lithops-deployer-key.json
```

**Expected output:**
```
service_account_email = "ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com"
artifact_registry_repo = "us-central1-docker.pkg.dev/e4drr-crafd/ecmwf-lithops"
gcs_bucket_access = "gs://gik-ecmwf-aws-tf (objectAdmin)"
```

---

## Step 2: Create Worker Script

Create the Cloud Run worker script that will be invoked by Lithops.

**Location:** `/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/cloudrun_lithops_ecmwf_par.py`

This script:
- Receives HTTP POST requests with date payloads
- Runs the GIK three-stage pipeline for that date
- Uploads parquets to GCS
- Returns status JSON

**Key Features:**
- Flask web server for Lithops compatibility
- Accepts date in request body
- Uses template fast-path (`--skip-grib-scan`)
- Parallel Stage 2 processing (8 workers)
- Automatic GCS upload
- Structured JSON response for Lithops

See the script implementation in [Step 2: Create Worker Script](#step-2-create-worker-script-implementation).

---

## Step 3: Build Docker Image

### 3a. Review Dockerfile

The Dockerfile:
- Based on `python:3.12-slim-bookworm`
- Installs system dependencies (GDAL, eccodes, compilers)
- Copies the entire `cgan_ecmwf/` directory
- Installs Python dependencies
- Runs Flask server on port 8080

**Location:** `lithops_cr_ecmwf_gik/Dockerfile`

### 3b. Copy Dockerfile to Source Context

```bash
cp /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/Dockerfile \
   /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/Dockerfile.cloudrun
```

### 3c. Submit Build to Cloud Build

```bash
# Activate deployer service account
gcloud auth activate-service-account \
  --key-file=/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json

# Submit build
gcloud builds submit \
  /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/ \
  --config=/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
```

**Expected output:**
```
Creating temporary archive...
Uploading tarball to [gs://e4drr-crafd_cloudbuild/source/...]
...
STATUS: SUCCESS
```

**Build time:** ~5-8 minutes (larger dependencies than other services)

---

## Step 4: Deploy to Cloud Run

### Option A: Deploy with Terraform (recommended)

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/terraform/

# Initialize Terraform
terraform init

# Review the plan
terraform plan -var="project_id=e4drr-crafd"

# Apply
terraform apply -var="project_id=e4drr-crafd"
```

### Option B: Deploy with gcloud CLI

```bash
gcloud run deploy ecmwf-parquet-worker \
  --image=us-central1-docker.pkg.dev/e4drr-crafd/ecmwf-lithops/ecmwf-parquet-worker:latest \
  --region=us-central1 \
  --project=e4drr-crafd \
  --service-account=ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
  --set-env-vars="AWS_NO_SIGN_REQUEST=YES,GCS_BUCKET=gik-ecmwf-aws-tf,GCS_PARQUET_PREFIX=run_par_ecmwf" \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=4 \
  --timeout=3600 \
  --max-instances=20 \
  --min-instances=0 \
  --concurrency=1
```

**Expected output:**
```
Service [ecmwf-parquet-worker] revision [...] has been deployed and is serving 100 percent of traffic.
Service URL: https://ecmwf-parquet-worker-462481537368.us-central1.run.app
```

**Resource Justification:**
- **8Gi memory**: GIK pipeline processes 51 ensemble members with ~6,685 GRIB references each
- **4 vCPUs**: Parallel Stage 2 processing uses 8 workers (benefits from hyperthreading)
- **3600s timeout**: Full pipeline (~10.5 min) + GCS upload (~2 min) + buffer
- **Concurrency 1**: Each instance processes one date at a time
- **Max 20 instances**: Lithops can submit up to 20 parallel date tasks

---

## Step 5: Configure Lithops Client

Create a Lithops configuration file for Cloud Run backend.

**Location:** `~/.lithops/config` or project-local `.lithops_config`

```yaml
lithops:
    backend: gcp_cloudrun
    storage: gcp_storage
    log_level: INFO

gcp:
    project_name: e4drr-crafd
    region: us-central1
    credentials_path: /path/to/ecmwf-lithops-deployer-key.json

gcp_cloudrun:
    region: us-central1
    max_workers: 20
    runtime: ecmwf-parquet-worker
    runtime_memory: 8192
    runtime_timeout: 3600
    invoke_pool_threads: 20

gcp_storage:
    bucket: gik-ecmwf-aws-tf
    region: us-central1
```

**Alternative: In-code configuration:**

```python
import lithops

config = {
    'backend': 'gcp_cloudrun',
    'storage': 'gcp_storage',
    'gcp': {
        'project_name': 'e4drr-crafd',
        'region': 'us-central1',
        'credentials_path': '/path/to/ecmwf-lithops-deployer-key.json'
    },
    'gcp_cloudrun': {
        'region': 'us-central1',
        'max_workers': 20,
        'runtime': 'ecmwf-parquet-worker',
        'runtime_memory': 8192,
        'runtime_timeout': 3600
    }
}

fexec = lithops.FunctionExecutor(config=config)
```

---

## Step 6: Execute Daily Tasks

### Single Date Processing

```python
import lithops

def process_ecmwf_date(date_str):
    """Process a single ECMWF date via Cloud Run worker."""
    import requests

    worker_url = "https://ecmwf-parquet-worker-462481537368.us-central1.run.app/process"

    response = requests.post(
        worker_url,
        json={"date": date_str, "run": "00"},
        timeout=3600
    )

    return response.json()

# Execute with Lithops
fexec = lithops.FunctionExecutor()
future = fexec.call_async(process_ecmwf_date, "20260206")
result = fexec.get_result()

print(result)
# Output: {'success': True, 'date': '20260206', 'output_dir': 'ecmwf_three_stage_20260206_00z', ...}
```

### Batch Date Range Processing

```python
import lithops
from datetime import datetime, timedelta

def process_ecmwf_date(date_str):
    """Process a single ECMWF date via Cloud Run worker."""
    import requests

    worker_url = "https://ecmwf-parquet-worker-462481537368.us-central1.run.app/process"

    response = requests.post(
        worker_url,
        json={"date": date_str, "run": "00"},
        timeout=3600
    )

    return response.json()

# Generate date range (last 7 days)
end_date = datetime.now()
dates = [(end_date - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
print(f"Processing dates: {dates}")

# Execute in parallel with Lithops
fexec = lithops.FunctionExecutor()
futures = fexec.map(process_ecmwf_date, dates)
results = fexec.get_result()

# Print summary
for date, result in zip(dates, results):
    status = "✓" if result.get('success') else "✗"
    print(f"{status} {date}: {result.get('message', 'No message')}")
```

### Standalone Invocation Script

**Location:** `lithops_cr_ecmwf_gik/run_lithops_ecmwf.py`

```python
#!/usr/bin/env python3
"""
Lithops-based ECMWF Parquet Generation
Run GIK pipeline for date ranges using Cloud Run workers
"""

import argparse
import lithops
from datetime import datetime, timedelta

WORKER_URL = "https://ecmwf-parquet-worker-462481537368.us-central1.run.app/process"

def process_ecmwf_date(date_str):
    """Process a single ECMWF date via Cloud Run worker."""
    import requests

    response = requests.post(
        WORKER_URL,
        json={"date": date_str, "run": "00"},
        timeout=3600
    )

    return response.json()

def main():
    parser = argparse.ArgumentParser(description='Run ECMWF parquet generation via Lithops')
    parser.add_argument('--date', type=str, help='Single date to process (YYYYMMDD)')
    parser.add_argument('--start-date', type=str, help='Start date for range (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='End date for range (YYYYMMDD)')
    parser.add_argument('--days-back', type=int, help='Process last N days')
    parser.add_argument('--max-workers', type=int, default=10, help='Max parallel workers')
    args = parser.parse_args()

    # Determine date list
    dates = []

    if args.date:
        dates = [args.date]
    elif args.days_back:
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime("%Y%m%d") for i in range(args.days_back)]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y%m%d")
        end = datetime.strptime(args.end_date, "%Y%m%d")
        dates = [(start + timedelta(days=i)).strftime("%Y%m%d")
                 for i in range((end - start).days + 1)]
    else:
        print("Error: Specify --date, --days-back, or --start-date/--end-date")
        return

    print(f"Processing {len(dates)} dates: {dates[0]} to {dates[-1]}")

    # Execute with Lithops
    config = {'backend': 'gcp_cloudrun', 'gcp_cloudrun': {'max_workers': args.max_workers}}
    fexec = lithops.FunctionExecutor(config=config)

    futures = fexec.map(process_ecmwf_date, dates)
    results = fexec.get_result()

    # Print summary
    success_count = sum(1 for r in results if r.get('success'))
    print(f"\n{'='*70}")
    print(f"Completed: {success_count}/{len(dates)} successful")
    print(f"{'='*70}")

    for date, result in zip(dates, results):
        status = "✓" if result.get('success') else "✗"
        print(f"{status} {date}: {result.get('message', 'No message')}")

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Single date
python run_lithops_ecmwf.py --date 20260206

# Last 7 days
python run_lithops_ecmwf.py --days-back 7

# Date range
python run_lithops_ecmwf.py --start-date 20260201 --end-date 20260207

# Custom parallelism
python run_lithops_ecmwf.py --days-back 30 --max-workers 20
```

---

## Configuration Reference

### Environment Variables (Cloud Run)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_NO_SIGN_REQUEST` | `YES` | Anonymous S3 access for ECMWF data |
| `GCS_BUCKET` | `gik-ecmwf-aws-tf` | GCS bucket for parquet upload |
| `GCS_PARQUET_PREFIX` | `run_par_ecmwf` | Prefix path within bucket |
| `PARALLEL_WORKERS` | `8` | Stage 2 parallel workers |
| `PORT` | `8080` | Flask server port |

### Cloud Run Resources

| Setting | Value | Reason |
|---------|-------|--------|
| Memory | 8Gi | 51 members × ~6,685 references + Python overhead |
| CPU | 4 vCPUs | Parallel Stage 2 processing (8 workers benefit from 4 cores) |
| Timeout | 3600s (1 hour) | Full pipeline (~10.5 min) + GCS upload + buffer |
| Concurrency | 1 | Each instance processes one date at a time |
| Max instances | 20 | Parallel date processing via Lithops |
| Min instances | 0 | Scale to zero when idle |

### Lithops Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `backend` | `gcp_cloudrun` | Use Cloud Run for execution |
| `storage` | `gcp_storage` | Use GCS for intermediate storage |
| `max_workers` | 20 | Max parallel Cloud Run instances |
| `runtime_memory` | 8192 MB | Match Cloud Run memory |
| `runtime_timeout` | 3600 s | Match Cloud Run timeout |
| `invoke_pool_threads` | 20 | Parallel HTTP invocations |

---

## Troubleshooting

### Check Cloud Run Logs

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=ecmwf-parquet-worker" \
  --project=e4drr-crafd \
  --limit=50 \
  --format="table(timestamp,severity,textPayload)"
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Build fails: dependencies | Pip install errors for kerchunk, cfgrib | Ensure system dependencies installed (libeccodes-dev, gcc, g++) |
| Container OOM killed | Service restarts, incomplete processing | Increase `--memory` to 12Gi or 16Gi |
| Timeout before completion | 504 Gateway Timeout | Increase `--timeout` or reduce `max_members` |
| GCS upload permission denied | 403 Forbidden on bucket | Grant service account `roles/storage.objectAdmin` on `gik-ecmwf-aws-tf` |
| ECMWF S3 access denied | Boto3 credential errors | Ensure `AWS_NO_SIGN_REQUEST=YES` is set |
| Lithops connection timeout | HTTP timeout during invocation | Check Cloud Run service is `--allow-unauthenticated` |
| Multiple instances processing same date | Duplicate uploads | Set `--concurrency=1` to ensure one date per instance |
| Stage 1 template not found | Hugging Face download fails | Pre-download template or use `--run-stage1` (not recommended) |

### Verify IAM Bindings

```bash
# Check Cloud Run service IAM
gcloud run services get-iam-policy ecmwf-parquet-worker \
  --region=us-central1 \
  --project=e4drr-crafd

# Check GCS bucket IAM
gsutil iam get gs://gik-ecmwf-aws-tf

# Verify service account has bucket access
gcloud projects get-iam-policy e4drr-crafd \
  --flatten="bindings[].members" \
  --filter="bindings.members:ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com"
```

### Test Worker Directly (Bypass Lithops)

```bash
SERVICE_URL="https://ecmwf-parquet-worker-462481537368.us-central1.run.app"

# Test health endpoint
curl $SERVICE_URL/health

# Test processing endpoint
curl -X POST $SERVICE_URL/process \
  -H "Content-Type: application/json" \
  -d '{"date": "20260206", "run": "00"}' \
  --max-time 3600
```

### Monitor Lithops Execution

```python
import lithops

fexec = lithops.FunctionExecutor()
futures = fexec.map(process_ecmwf_date, dates)

# Monitor progress
while not fexec.wait(futures, timeout=10):
    print(f"Running: {fexec.get_result(throw_except=False)}")

results = fexec.get_result()
```

---

## File Structure

```
lithops_cr_ecmwf_gik/
├── DEPLOYMENT_GUIDE.md              # This document
├── Dockerfile                        # Cloud Run worker container
├── cloudbuild.yaml                   # Cloud Build configuration
├── .dockerignore                     # Docker build exclusions
├── .gitignore                        # Git exclusions
├── run_lithops_ecmwf.py              # Lithops orchestration script
├── test_worker.py                    # Worker endpoint testing
├── lithops_config.yaml               # Lithops configuration template
├── service_account/                  # Deployer SA (Terraform)
│   ├── main.tf                       # SA, APIs, IAM, Artifact Registry, GCS
│   ├── variables.tf                  # Input variables
│   ├── outputs.tf                    # Outputs with usage instructions
│   ├── terraform.tfvars.example      # Example variable values
│   ├── .gitignore                    # Ignores keys & state
│   └── ecmwf-lithops-deployer-key.json  # (generated, git-ignored)
└── terraform/                        # Cloud Run infra (Terraform)
    ├── main.tf                       # Provider & API enablement
    ├── cloud_run.tf                  # Cloud Run v2 service
    ├── variables.tf                  # Input variables
    ├── outputs.tf                    # Service URL & helper commands
    └── terraform.tfvars.example      # Example variable values

# Modified source files in cGAN repo:
cGAN_tutorial/example_notebooks/cgan_ecmwf/
├── cloudrun_lithops_ecmwf_par.py     # Cloud Run worker script (NEW)
└── Dockerfile.cloudrun               # Copied from devops/lithops_cr_ecmwf_gik/
```

---

## Summary

| Component | Value |
|-----------|-------|
| **Service Name** | ecmwf-parquet-worker |
| **Service URL** | https://ecmwf-parquet-worker-462481537368.us-central1.run.app |
| **Project** | e4drr-crafd |
| **Region** | us-central1 |
| **Deployer SA** | ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com |
| **Artifact Registry** | us-central1-docker.pkg.dev/e4drr-crafd/ecmwf-lithops |
| **Image** | ecmwf-parquet-worker:latest |
| **GCS Output** | gs://gik-ecmwf-aws-tf/run_par_ecmwf/ |
| **Memory** | 8Gi |
| **CPU** | 4 vCPUs |
| **Timeout** | 3600s (1 hour) |
| **Concurrency** | 1 (one date per instance) |
| **Max Instances** | 20 (for Lithops parallelism) |
| **Orchestration** | Lithops Python library |
| **Source** | /home/roller/.../cGAN_tutorial/example_notebooks/cgan_ecmwf/ |

---

## Quick Reference Commands

### Build & Deploy

```bash
# Build
gcloud builds submit /path/to/cgan_ecmwf/ \
  --config=lithops_cr_ecmwf_gik/cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

# Deploy
gcloud run deploy ecmwf-parquet-worker \
  --image=us-central1-docker.pkg.dev/e4drr-crafd/ecmwf-lithops/ecmwf-parquet-worker:latest \
  --region=us-central1 \
  --project=e4drr-crafd \
  --service-account=ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
  --memory=8Gi --cpu=4 --timeout=3600 --max-instances=20 --concurrency=1
```

### Execute with Lithops

```bash
# Single date
python run_lithops_ecmwf.py --date 20260206

# Last 7 days
python run_lithops_ecmwf.py --days-back 7
```

### Monitor

```bash
# View logs
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=ecmwf-parquet-worker" \
  --project=e4drr-crafd --limit=20

# Check GCS output
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/
```

---

**Next Steps:**
1. Create service account with Terraform (Step 1)
2. Create worker script `cloudrun_lithops_ecmwf_par.py` (Step 2)
3. Build Docker image (Step 3)
4. Deploy to Cloud Run (Step 4)
5. Test with Lithops (Steps 5-6)
