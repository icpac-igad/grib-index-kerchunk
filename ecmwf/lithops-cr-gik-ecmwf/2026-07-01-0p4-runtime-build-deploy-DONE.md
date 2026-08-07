# 0p4 Lithops runtime — BUILD + DEPLOY DONE

**Date:** 2026-07-01
**Scope:** Build and deploy the `:0p4` runtime image for the pre-49r1 / 0.4° resolution era.
**Status:** ✅ **COMPLETE. Run host can use `:0p4` for 0.4° resolution backfill.**

---

## Why this was needed

The 0.4° resolution era predates 49r1. It uses a different template
(`gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz`, 9-level), a different reference
date (`20230601`), and `enfo` control stream — incompatible with the `:49r1`
or `:50r1` images. A separate image is required.

## What was done

| Step | Result |
|------|--------|
| 1. Auth | Switched to `ecmwf-lithops-deployer@e4drr-crafd` (active was `nka-terraform-access@sewaa-416306` → 403) |
| 2. Build | Cloud Build `73309861-6ce3-4209-a5f9` → **SUCCESS** (1m58s) |
| 3. Deploy 3.6.3 | `lithops runtime deploy` via system lithops → `lithops-worker-363-3fafb82775` ✅ |
| 4. Deploy 3.6.4 | `uv run --python 3.12 --with lithops==3.6.4 --with httplib2 --with google-auth --with google-api-python-client --with google-cloud-storage` → `lithops-worker-364-2d47ab5603` ✅ |
| 5. Confirm | `lithops runtime list` shows `:0p4` under both 3.6.3 + 3.6.4 ✅ |

## The image (verified in GCR)

- **`gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4`**
- Baked template: **`gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz`** (9-level)
- Era env: `ECMWF_REFERENCE_DATE=20230601`, `ECMWF_RESOLUTION=0p4`, `ECMWF_CONTROL_STREAM=enfo`
- Runtime: `lithops==3.6.4` pinned (matches the run host)

## Deployed services

```
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4   2048  3.6.4  lithops-worker-364-2d47ab5603
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4   2048  3.6.3  lithops-worker-363-3fafb82775
```

(`:49r1` and `:50r1` runtimes remain deployed alongside — untouched.)

## Note: uv deploy requires extra --with deps

Unlike system lithops (3.6.3), deploying via `uv run --with lithops==3.6.4`
requires explicitly adding GCP deps that lithops doesn't declare:
`--with httplib2 --with google-auth --with google-api-python-client --with google-cloud-storage`

## ➡️ Run host: to use the 0p4 runtime

1. Set `lithops_config.yaml:32` → `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4`
2. Set env: `ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4 ECMWF_CONTROL_STREAM=enfo`
3. Set `TEMPLATE_URL` → `…/gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz`
4. `export UV_PYTHON=3.12`
5. Run `uv run run_lithops_ecmwf.py` with desired date range
