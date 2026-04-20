# GEFS Lithops Cloud Run — Deployment Guide

**Date**: 2026-04-09
**Project**: e4drr-crafd (or sewaa-416306)
**Region**: us-east1 (co-located with NOAA GEFS S3 in AWS us-east-1)
**Runtime image**: `gcr.io/<PROJECT_ID>/gefs-lithops-runtime`
**Deployer SA (dedicated, NOT shared)**: `gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com`

> Modeled on `devops/crma-api-cr/DEPLOYMENT_GUIDE.md` and the working
> `lithops_cr_ecmwf_gik` pattern. The three caveats from those deployments
> (dedicated SA, activate before submitting, pass `--service-account` to
> Cloud Build) apply here too.

---

## Important: Do NOT reuse the ECMWF service account

GEFS gets its own dedicated SA (`gefs-lithops-deployer@e4drr-crafd`) so that:

- IAM blast radius for GEFS jobs is isolated from ECMWF.
- Key rotation / revocation for one pipeline doesn't break the other.
- Bucket-level grants are scoped to the GEFS SA only.

The dedicated SA is defined in `terraform/main.tf`.

## Important: The output bucket is cross-project

`gs://gik-gefs-aws-tf` does **not** live in `e4drr-crafd`. It lives in
**sewaa-416306** (project number `8979869085`), in `us-east1`. The same is
true of `gs://gik-ecmwf-aws-tf` — both GIK buckets were created in sewaa
and are accessed cross-project from runtime SAs that live in e4drr-crafd.

This works because:
- Lithops only performs object-level operations (`storage.objects.*`) on the
  data bucket. It calls `buckets.get` once at startup but never tries to list
  or create buckets at the project level.
- GCS allows bucket-level IAM grants to principals from other projects.
- Therefore the `project_name: e4drr-crafd` field in `lithops_config.yaml` is
  effectively *decorative* for storage — it only governs the Cloud Run
  backend (which really does live in e4drr-crafd).

**Consequence for terraform**: this module does NOT create the output bucket
(it would try to create it in the wrong project). It only creates a
cross-project IAM binding granting `gefs-lithops-deployer@e4drr-crafd`
`roles/storage.objectAdmin` on the pre-existing `gs://gik-gefs-aws-tf`.

### Step 1a — Cross-project IAM (manual fallback)

The terraform `google_storage_bucket_iam_member.gefs_output_sa` resource
requires the principal running `terraform apply` to have
`storage.buckets.setIamPolicy` on the bucket — i.e. perms in **sewaa-416306**.
If you're applying terraform from an e4drr-crafd-only credential, that step
will fail. In that case, run this manually with a sewaa-authorized account
(e.g. `nka-terraform-access@sewaa-416306` or any owner of the bucket):

```bash
gsutil iam ch \
  serviceAccount:gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com:objectAdmin \
  gs://gik-gefs-aws-tf
```

Verify:
```bash
gsutil iam get gs://gik-gefs-aws-tf | \
  grep -A1 gefs-lithops-deployer
```

---

## Pre-Deployment Checklist

### 1. Create the dedicated SA + buckets (Terraform — run once per project)

```bash
cd devops/lithops_cr_gefs_gik/terraform/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: pick e4drr-crafd OR sewaa-416306 block
terraform init
terraform apply
```

This creates:
- [x] Service account `gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com`
- [x] Project IAM roles: `run.invoker`, `run.admin`, `storage.objectAdmin`,
      `artifactregistry.reader`, `cloudbuild.builds.editor`
- [x] Self-impersonation (`iam.serviceAccountUser`) — required by Cloud Build
- [x] **Lithops staging bucket** (in this project) with SA grant — used by
      Lithops to stage function pickles + result pickles
- [x] **Cross-project IAM binding** on the pre-existing output bucket
      `gs://gik-gefs-aws-tf` (lives in sewaa-416306) — see Step 1a if this fails
- [x] Required APIs enabled (run, compute, iam, storage, artifactregistry, cloudbuild)

The output bucket itself is **not** created here. It already exists in
sewaa-416306 — see "The output bucket is cross-project" above.

### 2. Generate and store the SA key

```bash
mkdir -p ../service_account
gcloud iam service-accounts keys create \
  ../service_account/gefs-lithops-deployer-key.json \
  --iam-account=gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  --project=<PROJECT_ID>
chmod 600 ../service_account/gefs-lithops-deployer-key.json
```

Make sure `service_account/` is git-ignored (the existing `.gitignore` covers it).

### 3. Update `lithops_config.yaml` to point at the NEW key

Edit `lithops_config.yaml` (and `lithops_config_sewaa.yaml` if deploying to
sewaa) so `gcp.credentials_path` points at the GEFS key, **not** the ECMWF key:

```yaml
gcp:
    project_name: e4drr-crafd
    region: us-east1
    credentials_path: service_account/gefs-lithops-deployer-key.json
```

### 4. Activate the deployer SA

**CRITICAL** — same caveat as `crma-api-cr`. Without this, `gcloud builds
submit` runs as the default Cloud Build SA which lacks the GCR/Cloud Run
permissions and fails with `NOT_FOUND: Requested entity was not found`.

```bash
cd devops/lithops_cr_gefs_gik/
gcloud auth activate-service-account \
  --key-file=service_account/gefs-lithops-deployer-key.json

gcloud auth list
# Active account should be: gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com
```

### 5. Submit Cloud Build with `--service-account`

```bash
gcloud builds submit . \
  --config=cloudbuild.yaml \
  --project=<PROJECT_ID> \
  --service-account=projects/<PROJECT_ID>/serviceAccounts/gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com
```

For sewaa:
```bash
gcloud builds submit . \
  --config=cloudbuild-sewaa.yaml \
  --project=sewaa-416306 \
  --service-account=projects/sewaa-416306/serviceAccounts/gefs-lithops-deployer@sewaa-416306.iam.gserviceaccount.com
```

This pushes both `gcr.io/<PROJECT_ID>/gefs-lithops-runtime:$BUILD_ID` and `:latest`.

### 6. Deploy the Lithops Cloud Run runtime

```bash
uv run lithops runtime deploy gcr.io/<PROJECT_ID>/gefs-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml
```

Verify:
```bash
uv run lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

### 7. Smoke test with one date

```bash
uv run run_lithops_gefs.py --date 20260206 --max-workers 5
```

The latest commits already in this directory and ready to deploy:

- `b8020c1` — Make GEFS Lithops script self-contained (no `gefs_util.py` dependency)
- `6467c97` — Replace scan_grib with template-based Stage 1 for GEFS Lithops pipeline
- `7da61ee` — Add sewaa-416306 deployment for GEFS Lithops pipeline and AIFS GPU
- `dffb6a5` — Updated the GEFS lithops and changes in the working ECMWF lithops

> Working tree is clean for `devops/lithops_cr_gefs_gik/`, but the local
> branch is **9 commits ahead of `origin/main`**. Since there is no Cloud
> Build trigger wired to GitHub, you can deploy directly from local source
> via `gcloud builds submit` without pushing — the build context is the
> current working directory.

---

## Common Errors & Fixes (carry-over from crma-api-cr lessons)

### `NOT_FOUND: Requested entity was not found`
**Cause**: `gcloud builds submit` called WITHOUT `--service-account`. The
default Cloud Build SA cannot find/access GCR.
**Fix**: Always pass
`--service-account=projects/<PROJECT_ID>/serviceAccounts/gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com`.

### `Permission 'iam.serviceaccounts.actAs' denied`
**Cause**: Self-impersonation grant missing.
**Fix**: Already covered by `google_service_account_iam_member.self_impersonate`
in `terraform/main.tf`. Re-run `terraform apply` if missing.

### Lithops runtime deploy hangs / 403 on bucket
**Cause**: `gcp.credentials_path` in `lithops_config.yaml` still points at
the ECMWF key (which has no IAM on GEFS buckets).
**Fix**: Update to `service_account/gefs-lithops-deployer-key.json` (step 3 above).

### Build succeeds, runtime deploy fails to write to staging bucket
**Cause**: Buckets created out-of-band (not via this terraform), so the GEFS
SA never got `storage.objectAdmin`.
**Fix**: Either re-run `terraform apply` so the bucket-level
`google_storage_bucket_iam_member` resources are created, or grant manually:
```bash
gsutil iam ch \
  serviceAccount:gefs-lithops-deployer@<PROJECT_ID>.iam.gserviceaccount.com:objectAdmin \
  gs://gik-gefs-aws-tf gs://lithops-us-east1-gefs-e4drr
```

---

## Quick Reference

```bash
# Full rebuild + redeploy (e4drr-crafd)
cd devops/lithops_cr_gefs_gik/
gcloud auth activate-service-account \
  --key-file=service_account/gefs-lithops-deployer-key.json
gcloud builds submit . \
  --config=cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
uv run lithops runtime deploy gcr.io/e4drr-crafd/gefs-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Tail Cloud Run logs for one Lithops worker invocation
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name~gefs-lithops-runtime" \
  --project=e4drr-crafd --limit=20 \
  --format="table(timestamp,severity,textPayload)"

# Delete runtime (cleanup)
uv run lithops runtime delete gcr.io/e4drr-crafd/gefs-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml
```
