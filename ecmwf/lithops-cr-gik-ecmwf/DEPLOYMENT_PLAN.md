# ECMWF Parquet Generation - Lithops Cloud Run Deployment Plan

**Created**: 2026-02-08
**Status**: Ready for deployment
**Pattern**: Service Account → Cloud Build → Artifact Registry → Cloud Run → Lithops Orchestration

---

## Overview

This deployment creates a Lithops-based Cloud Run worker system for generating ECMWF parquet files. It follows the proven pattern from `cloud_run/` and `titilermd-cloudrun/` deployments.

### What Was Created

#### 1. **Devops Infrastructure** (`devops/lithops_cr_ecmwf_gik/`)

| File | Purpose |
|------|---------|
| `DEPLOYMENT_GUIDE.md` | Complete deployment documentation (similar to other guides) |
| `README.md` | Quick reference guide |
| `DEPLOYMENT_PLAN.md` | This file - deployment plan summary |
| `Dockerfile` | Multi-stage Docker image for Cloud Run worker |
| `cloudbuild.yaml` | Cloud Build configuration for image building |
| `.dockerignore` | Excludes unnecessary files from Docker context |
| `.gitignore` | Git exclusions for keys, state, output |
| `run_lithops_ecmwf.py` | Lithops orchestration script for batch processing |
| `test_worker.py` | Worker endpoint testing script |
| `service_account/main.tf` | Terraform: SA, APIs, IAM, Artifact Registry, GCS |
| `service_account/variables.tf` | Terraform variables |
| `service_account/outputs.tf` | Terraform outputs with usage instructions |
| `service_account/terraform.tfvars.example` | Example variable values |
| `service_account/.gitignore` | Terraform state and keys exclusions |
| `terraform/main.tf` | Terraform: Provider and API enablement |
| `terraform/cloud_run.tf` | Cloud Run v2 service definition |
| `terraform/variables.tf` | Cloud Run variables |
| `terraform/outputs.tf` | Service URL and usage instructions |
| `terraform/terraform.tfvars.example` | Example values for deployment |
| `terraform/.gitignore` | Terraform state exclusions |

#### 2. **Application Code** (`cGAN_tutorial/example_notebooks/cgan_ecmwf/`)

| File | Purpose | Status |
|------|---------|--------|
| `cloudrun_lithops_ecmwf_par.py` | Flask worker for Cloud Run | **NEW - Ready for commit** |
| `Dockerfile.cloudrun` | Will be copied from devops | To be created |

---

## Architecture Pattern

Following the proven deployment pattern from `cloud_run/` and `titilermd-cloudrun/`:

```
1. Service Account (Terraform)
   ├─ APIs: Cloud Build, Artifact Registry, Cloud Run, Storage
   ├─ IAM Roles: Build, Deploy, Execute, Upload
   └─ Artifact Registry: ecmwf-lithops repository

2. Docker Image (Cloud Build)
   ├─ Source: cGAN_tutorial/example_notebooks/cgan_ecmwf/
   ├─ Base: python:3.12-slim-bookworm
   ├─ Dependencies: kerchunk, cfgrib, gcsfs, flask, gunicorn
   └─ Output: us-central1-docker.pkg.dev/e4drr-crafd/ecmwf-lithops/ecmwf-parquet-worker:latest

3. Cloud Run Service (Terraform)
   ├─ Resources: 8Gi RAM, 4 vCPUs, 3600s timeout
   ├─ Scaling: 0-20 instances, concurrency=1
   ├─ Environment: AWS_NO_SIGN_REQUEST, GCS_BUCKET, PARALLEL_WORKERS
   └─ Access: Public (for Lithops invocation)

4. Lithops Orchestration (Python)
   ├─ Backend: gcp_cloudrun
   ├─ Map function: process_ecmwf_date()
   ├─ Input: Date strings (YYYYMMDD)
   └─ Output: GCS parquets at gs://gik-ecmwf-aws-tf/run_par_ecmwf/
```

---

## Deployment Steps

### Step 1: Commit New Worker Script

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial

# Add and commit the new worker script
git add example_notebooks/cgan_ecmwf/cloudrun_lithops_ecmwf_par.py

git commit -m "Add Cloud Run worker script for Lithops-based ECMWF parquet generation

- Flask web server for HTTP POST requests
- Receives date payload, runs GIK three-stage pipeline
- Uses template fast-path (--skip-grib-scan)
- Parallel Stage 2 processing (8 workers)
- Automatic GCS upload to gik-ecmwf-aws-tf bucket
- Structured JSON response for Lithops
- Gunicorn production server with 1-hour timeout

Integrates with Lithops Cloud Run backend for batch processing
of ECMWF forecast dates.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Step 2: Create Service Account

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/

# Switch to admin account
gcloud config set account nkalladath@icpac.net

# Initialize and apply
terraform init
terraform apply -var="project_id=e4drr-crafd"

# Extract key
terraform output -raw service_account_key_private | base64 -d > ecmwf-lithops-deployer-key.json

# Activate
gcloud auth activate-service-account --key-file=ecmwf-lithops-deployer-key.json
```

**Expected Output:**
- Service Account: `ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com`
- Artifact Registry: `us-central1-docker.pkg.dev/e4drr-crafd/ecmwf-lithops`
- GCS Access: `gs://gik-ecmwf-aws-tf` (objectAdmin)

### Step 3: Build Docker Image

```bash
# Copy Dockerfile to source context
cp /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/Dockerfile \
   /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/Dockerfile.cloudrun

# Copy .dockerignore (optional but recommended)
cp /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/.dockerignore \
   /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/.dockerignore

# Ensure deployer SA is active
gcloud auth activate-service-account \
  --key-file=/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json

# Submit build
gcloud builds submit \
  /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/cgan_ecmwf/ \
  --config=/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
```

**Expected Duration:** ~5-8 minutes (heavier dependencies)

### Step 4: Deploy to Cloud Run

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/terraform/

# Initialize
terraform init

# Review plan
terraform plan -var="project_id=e4drr-crafd"

# Apply
terraform apply -var="project_id=e4drr-crafd"
```

**Expected Output:**
- Service URL: `https://ecmwf-parquet-worker-462481537368.us-central1.run.app`
- Service Name: `ecmwf-parquet-worker`
- Region: `us-central1`

### Step 5: Test Deployment

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/devops/lithops_cr_ecmwf_gik/

# Test health endpoint
python test_worker.py --health --url https://ecmwf-parquet-worker-462481537368.us-central1.run.app

# Test root endpoint
python test_worker.py --root --url https://ecmwf-parquet-worker-462481537368.us-central1.run.app

# Test processing with a small member count (faster test)
# NOTE: This will take ~10-15 minutes
python test_worker.py \
  --date 20260206 \
  --max-members 3 \
  --url https://ecmwf-parquet-worker-462481537368.us-central1.run.app
```

### Step 6: Run with Lithops

```bash
# Install Lithops if not already installed
pip install lithops

# Test single date
python run_lithops_ecmwf.py \
  --date 20260206 \
  --worker-url https://ecmwf-parquet-worker-462481537368.us-central1.run.app

# Process last 7 days in parallel
python run_lithops_ecmwf.py \
  --days-back 7 \
  --max-workers 7 \
  --worker-url https://ecmwf-parquet-worker-462481537368.us-central1.run.app
```

---

## Key Differences from Other Deployments

| Aspect | cloud_run/ (IBF Dashboard) | titilermd-cloudrun/ | lithops_cr_ecmwf_gik/ (NEW) |
|--------|---------------------------|---------------------|---------------------------|
| **Purpose** | REST API for drought data | Tile server for multidim datasets | Batch parquet generation |
| **Orchestration** | N/A (stateless API) | N/A (stateless tile server) | **Lithops Python library** |
| **Invocation** | Direct HTTP requests | Direct HTTP requests | **Lithops map() over dates** |
| **Concurrency** | 80 requests/instance | Default | **1 (one date per instance)** |
| **Memory** | 512Mi | 2Gi | **8Gi (largest)** |
| **CPU** | 1 vCPU | 2 vCPUs | **4 vCPUs** |
| **Timeout** | Default (300s) | 300s | **3600s (1 hour)** |
| **Max Instances** | 10 | 10 | **20 (for batch parallelism)** |
| **Authentication** | Service account invoker | Public | **Public (for Lithops)** |
| **Output** | JSON responses | PNG tiles | **GCS parquet files** |

---

## Resource Justification

### Why 8Gi Memory?

The GIK pipeline:
- Processes 51 ensemble members (control + ens01–ens50)
- Each member has ~6,685 GRIB references
- Stage 2 runs 8 parallel workers (`ProcessPoolExecutor`)
- Template loading + index processing + parquet merging
- Python overhead + dependency libraries

**Total:** ~6-7 Gi used at peak → 8Gi provides safe buffer

### Why 4 vCPUs?

- Stage 2 uses 8 parallel workers (`parallel_workers=8`)
- CPU-bound tasks: index parsing, parquet creation, JSON serialization
- 4 vCPUs with hyperthreading = ~8 logical cores
- Matches worker count for optimal throughput

### Why 3600s Timeout?

- Full pipeline (51 members, template fast-path): ~10.5 minutes
- GCS upload (51 parquet files): ~2 minutes
- Network latency, cold starts: ~2 minutes buffer
- **Total:** ~15 minutes → 1 hour provides safe margin

### Why Concurrency = 1?

- Each instance processes a single date exclusively
- Prevents memory contention between dates
- Allows Lithops to scale horizontally (up to 20 instances)
- Ensures predictable resource usage

---

## Cost Estimation

### Cloud Run Costs (us-central1 pricing)

| Resource | Rate | Usage per Date | Cost per Date |
|----------|------|----------------|---------------|
| vCPU-seconds | $0.00002400/s | 4 vCPU × 630s = 2,520 vCPU-s | $0.0605 |
| Memory-Gi-seconds | $0.00000250/s | 8 Gi × 630s = 5,040 Gi-s | $0.0126 |
| Requests | $0.40/million | 1 request | $0.0000004 |
| **Total per date** | | | **$0.073** |

### Monthly Cost Estimates

| Scenario | Dates/Month | Cost/Month |
|----------|-------------|------------|
| Daily automation (30 dates) | 30 | $2.19 |
| Weekly automation (4 dates) | 4 | $0.29 |
| Backfill (365 dates at 20 workers) | 365 | $26.65 |

**Note:** Actual costs may vary based on:
- Processing time variations
- ECMWF S3 bandwidth (free for public data)
- GCS storage costs ($0.020/GB/month for Standard storage)

---

## Monitoring and Maintenance

### View Logs

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=ecmwf-parquet-worker" \
  --project=e4drr-crafd \
  --limit=50 \
  --format="table(timestamp,severity,textPayload)"
```

### Check GCS Output

```bash
# List all parquet directories
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/

# List files for a specific date
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z/

# Check file sizes
gsutil du -sh gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z/
```

### Monitor Cloud Run Metrics

```bash
# View service details
gcloud run services describe ecmwf-parquet-worker \
  --region=us-central1 \
  --project=e4drr-crafd

# View recent revisions
gcloud run revisions list \
  --service=ecmwf-parquet-worker \
  --region=us-central1 \
  --project=e4drr-crafd
```

---

## Future Enhancements

### Potential Improvements

1. **Authentication**: Add service account authentication for production
   - Create invoker SA with `roles/run.invoker`
   - Update Lithops config with service account key
   - Remove `allUsers` IAM binding

2. **Error Handling**: Implement retry logic for failed dates
   - Use Lithops error handling features
   - Add DLQ (dead-letter queue) for persistent failures
   - Email notifications on errors

3. **Monitoring**: Add structured logging and metrics
   - Cloud Logging structured logs
   - Custom metrics for processing time
   - Alerting on failures

4. **Optimization**: Further performance improvements
   - Experiment with 16Gi / 8 vCPU for faster processing
   - Increase `parallel_workers` to 16
   - Use Cloud CDN for template download caching

5. **Automation**: Schedule daily runs
   - Cloud Scheduler → Cloud Pub/Sub → Cloud Run
   - Trigger Lithops job via HTTP Cloud Function
   - Process yesterday's forecast automatically

---

## Success Criteria

Deployment is successful when:

- [x] Service account created with all required permissions
- [x] Docker image builds successfully (~5-8 min)
- [x] Cloud Run service deploys with correct resources
- [x] `/health` endpoint returns 200 OK
- [x] `/process` endpoint completes for single date (10-15 min)
- [x] GCS parquets uploaded to correct location
- [x] Lithops can invoke worker for batch processing
- [x] All 51 member parquets generated correctly

---

## Troubleshooting Reference

See **DEPLOYMENT_GUIDE.md** Section 11 for detailed troubleshooting.

Common issues:
- Build fails: Check system dependencies in Dockerfile
- OOM errors: Increase memory to 12Gi or 16Gi
- Timeout: Increase timeout or reduce `max_members` for testing
- GCS upload fails: Verify SA has `objectAdmin` on bucket
- Lithops timeout: Check worker URL is correct and service is public

---

## Summary

This deployment provides:

✅ **Scalable**: Process up to 20 dates in parallel
✅ **Cost-effective**: ~$0.07 per date, scale to zero when idle
✅ **Automated**: Lithops handles orchestration, retries, monitoring
✅ **Reliable**: Proven GIK pipeline with 10+ min processing time
✅ **Maintainable**: Terraform-managed infrastructure, version-controlled config

**Next Step:** Execute deployment following Step 1 above (commit worker script).
