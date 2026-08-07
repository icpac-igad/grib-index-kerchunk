# 50r1 Lithops runtime — BUILD + DEPLOY DONE, ready for the 00z run

**Date:** 2026-06-26
**Scope:** the deployer half of the 50r1 handoff from the 2026-06-24 run-host session
(`2026-06-24-rebake-gik-par-creation.txt`, "you build+deploy, I run") — *build and
deploy the `:50r1` runtime image so the 50r1 00z par creation can run.*
**Status:** ✅ **COMPLETE. The run host is cleared to launch the 50r1 00z run.**

---

## Why this was needed

The era is **baked into the runtime image at build time** (Dockerfile
`--build-arg TEMPLATE_ARTIFACT`); `ensure_template()` returns the baked path
*before* any client `TEMPLATE_URL` override. The run host proved a 50r1 run
against the deployed `:49r1` image wrote **0 files** (51/51 "Template not found"
— the 49r1 tar's keys are dated `2025051500`, 50r1 needs `2026051300`). So 50r1
needs its **own** image. No code/cloudbuild changes were required — `cloudbuild.yaml`
defaults are already 50r1, and the Dockerfile already carries both prior fixes
(`28867db` lithops==3.6.4 pin, `0714d3a` namegenerator removal).

## What was done

| Step | Result |
|------|--------|
| 0. Auth | `ecmwf-lithops-deployer@e4drr-crafd` activated (active account was wrong project → 403 on registry, as expected) |
| 1. Build | Cloud Build `e0024fca` → **SUCCESS** (1m45s) |
| 2. Config | `lithops_config.yaml:32` → `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1` ✅ |
| 3. Deploy | Registered under **both** lithops versions (see note) ✅ |
| 4. Confirm | `lithops runtime list` shows `:50r1` under 3.6.3 + 3.6.4 ✅ |

## The image (verified in GCR)

- **`gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1`** — digest `c4ce6190dec2`, built 2026-06-26 (also `50r1-e0024fca-…`)
- Baked template: **`gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz`** (14-level, all 51 members)
- Era env: `ECMWF_REFERENCE_DATE=20260513`, `ECMWF_RESOLUTION=0p25`, `ECMWF_CONTROL_STREAM=oper`
- Runtime: `lithops==3.6.4` pinned (matches the run host)

## Deployed under BOTH lithops versions (important)

Lithops encodes its local version into the Cloud Run service name + metadata path:

- This deployer host's CLI is **3.6.3** → `lithops-worker-363-d1cea16f95`
- The run host pins **lithops==3.6.4** + Python 3.12 → `lithops-worker-364-5a680638f5`

Both were deployed (the 3.6.4 one via `uv run --python 3.12 --with lithops==3.6.4 …`),
so the run host finds a **ready** service and need not auto-deploy. `runtime list`:

```
ecmwf-lithops-runtime:50r1   2048   3.6.4   lithops-worker-364-5a680638f5
ecmwf-lithops-runtime:50r1   2048   3.6.3   lithops-worker-363-d1cea16f95
```

(The `:49r1` runtime remains deployed alongside — different image, different
service, untouched.)

---

## ➡️ Run host: launch the 50r1 00z run

Everything 50r1 was pre-verified in the 2026-06-24 session (template, source data
`20260513→20260624` all HTTP 200, era config, the staged `rebake_50r1_00z.sh`).
On the run host:

1. Set **its** `lithops_config.yaml:32` → `…ecmwf-lithops-runtime:50r1`
2. `export UV_PYTHON=3.12`
3. `bash rebake_50r1_00z.sh`  → 43 dates into the same `v20260623_run_par_ecmwf/`:
   - `20260513 → 20260531` (19 dates — starts at the 13th, won't touch 49r1's ≤05-12)
   - `20260601 → 20260624` (24 dates — 06-25+ not yet published)

Re-validate the boundary date `20260513` (expect 51 files) before fanning out.

## Notes / still outstanding (separate work)

- `lithops_config.yaml:32` on **this** host now points at `:50r1` (committed).
- Still out of scope (per the run-host transcript): 06z/12z/18z cycles, and the
  HF re-mirror of the fixed parquets.
