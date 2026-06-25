# ECMWF Era-2 (49r1) Step-2 Lithops Backfill — execution plan

**Date:** 2026-06-08
**Branch:** `ecmwf-50r1-template`
**Scope:** Run the Stage-2/3 Lithops pipeline to produce the **final daily
per-member parquet** for **every 00z date in the 49r1 0.25° era**
(`2024-02-29` → `2026-05-12`, **804 dates**), now that Step-1 (per-era
`scan_grib` + realigner + per-level templates) is complete and verified.
Plan also covers the other two eras (0.4°-beta, 50r1) but **starts with era 2**.

> Run-cycle scope: **00z only** (decided 2026-06-08). Matches the existing
> `run_backfill_00z.sh` and the cGAN consumer. ECMWF open-data publishes the
> full 360h/85-step ENS only at 00z & 12z (06z/18z stop at 144h and would
> under-fill the 85-step template), so 00z is the clean choice for the archive.

---

## 0. Where we are (context up to this point)

**Step 1 is DONE and live** (`grib-index-kerchunk`,
`ecmwf/docs/2026-06-03-three-era-scan-run-verification-and-perlevel-fix.md`):

- Three per-era Coiled `scan_grib` runs complete (340 dumps, verified
  complete + row-count exact).
- Realigner (`ecmwf_par_to_ensemble_members.py`) fixes per-member assignment,
  per-level pl keys (`var/isobaricInhPa/{hPa}`), and real grid shape.
- **Three per-era templates are live on HF `E4DRR/grib-index-kerchunk-templates`:**
  - `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` (13-level, ref `20250515`) — **era 2**
  - `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` (14-level, ref `20260513`) — era 3
  - `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` (9-level, ref `20230601`) — era 1
  - (4th "49r1 9-level" sub-era is covered by the 13-level template as a safe
    superset → no separate build.)

**The Lithops runtime is DONE and wired for per-era deploy** (`cno-e4drr`
`main`, commit `40b6759`):

> **Two-repo topology (important for "run it from there").** The deployment
> lives in **two mirrored copies**:
> - **PRIVATE / operational (source of truth, you run from here):**
>   `cno-e4drr/devops/lithops_cr_ecmwf_gik/` — the only copy that holds
>   credentials (`service_account/`, `terraform/`, and `lithops_config.yaml`
>   pointing at the SA key). The Cloud Run deploy is driven from here.
> - **PUBLIC / reflection (no creds):**
>   `grib-index-kerchunk/ecmwf/lithops-cr-gik-ecmwf/` — an identical clone with
>   `service_account/` and `terraform/` **absent** (verified), so cloud
>   credentials never reach the public repo. Edits land private-first, then are
>   duplicated to public.
>
> Commit `40b6759` **synced the operational `.py` + `Dockerfile` from the
> validated public copy.** This matters in **two distinct ways** (precise
> mechanism in **§2.1**): the **client** `run_lithops_ecmwf.py` — whose function
> body Lithops cloudpickles and ships to every worker at submit time — must be
> the fixed version (per-level keys, resolution-awareness); **and** the
> **`Dockerfile`** must bake the right per-era **template + env**, which is the
> one part that lives *only* in the image and cannot ride along from the client.
> So any era-2 backfill **must (a) run from the current synced `.py` and (b)
> rebuild + deploy the `:49r1` image** (§3) — the legacy 50r1 image bakes the
> wrong (14-level) template, and a worker loads its baked template regardless of
> what the client passes.

- `run_lithops_ecmwf.py` emits per-level keys, is resolution-aware
  (`ECMWF_RESOLUTION` → `ifs/0p25` vs `0p4-beta`), and selects control stream
  (`ECMWF_CONTROL_STREAM` → `enfo`/`oper`).
- `Dockerfile` bakes the per-era template via `--build-arg TEMPLATE_ARTIFACT`
  + matching `ECMWF_*` env. **Defaults are 50r1.**
- `cloudbuild.yaml` selects era via `--substitutions` → per-era image tag
  `ecmwf-lithops-runtime:{era}`.
- `run_backfill_00z.sh` already has `--era {49r1|50r1|0p4}` exporting the
  matching template/resolution/control-stream env; **default `--era 49r1`**,
  default range `2024-03` → `2025-12`.

**Net:** No new code is required for era 2. This is an **operations run**:
build+deploy the **49r1 image** (the deployed default is currently 50r1),
then drive `run_backfill_00z.sh` across the era-2 month range.

### What "from there" means (pre-flight, not assumable)

This sandbox has no `gcloud`/`gsutil`/auth, so the already-done portion can't
be read here. **Step 0 of execution is a GCS audit** to find which era-2 dates
already have 51 parquets, so the backfill only fills gaps (idempotent re-runs
are otherwise harmless — they overwrite).

---

## 1. Era-2 facts (the target)

| Property | Value |
|---|---|
| Era | 49r1 0.25° (covers both 9-level and 13-level sub-spans) |
| Date range (00z) | `2024-02-29` → `2026-05-12` = **804 dates**, **28 monthly batches** |
| 9-level sub-span | `2024-02-29` → `2025-01-13` (320 dates) — extra 4 pl levels left empty by the 13-level template ✅ |
| 13-level sub-span | `2025-01-14` → `2026-05-12` (484 dates) — full 13-level fit ✅ |
| Members | 51 (control bundled in `enfo` as `number=0`) → `ECMWF_CONTROL_STREAM=enfo` |
| Source path | `s3://ecmwf-forecasts/{date}/00z/ifs/0p25/enfo/...` → `ECMWF_RESOLUTION=0p25` |
| Template | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz`, `ECMWF_REFERENCE_DATE=20250515` |
| Output | `gs://gik-ecmwf-aws-tf/run_par_ecmwf/YYYY/MM/YYYYMMDD/00z/{date}00z-{member}.parquet` (51 files/date) |
| Cost | 804 × $0.026 ≈ **$21**; wall ≈ 28 months × ~8.5 min ≈ **~4 hrs** sequential |

> The era boundary is at **2026-05-12 06z** (50r1 starts). So `2026-05-12 00z`
> is the **last** era-2 00z date — consistent with this range. Your message's
> "…20260512 18z" 06z/12z/18z of that day belong to **era 3 (50r1)** and are
> out of scope for the 49r1 image.

---

## 2. Do we need updates to Docker / Cloud Run / backend / credentials?

**Short answer: No code/infra changes are required. Two operational
prerequisites only.**

| Item | Status | Action |
|---|---|---|
| `run_lithops_ecmwf.py` | ✅ per-level, resolution-aware, era-driven | none |
| `Dockerfile` | ✅ per-era build-arg; default 50r1 | **build with `--substitutions=_ERA=49r1`** (see §3.1) |
| `cloudbuild.yaml` | ✅ per-era substitutions | none |
| `run_backfill_00z.sh` | ✅ `--era 49r1` default; range overridable | **set `--from 2024-02 --to 2026-05`** |
| `lithops_config.yaml` | ✅ `max_workers=35`, region `europe-west3`, 2 GB/2 vCPU/3600s | none (35 ≥ 31 dates/month) |
| Cloud Run runtime deployed | ⚠️ **the deployed image is the default 50r1** | **re-deploy the 49r1 tag before backfill** (§3.2) |
| GCS bucket `gik-ecmwf-aws-tf` | ✅ exists (terraform/) | none |
| IAM / service accounts | ✅ `ecmwf-lithops-deployer@…`; bucket IAM via terraform | none |

### 2.1 Why a Docker rebuild is required when the codebase is updated (precise mechanism)

The intuitive worry is *"the code changed, so the running Cloud Run service is
stale."* That is only **half** right, and getting the halves straight tells you
exactly **what** the rebuild is for. There are two independent delivery paths
into a worker, and they update by **different** triggers:

**Path A — the worker function: shipped client-side, NOT baked.**
The `Dockerfile` **never `COPY`s `run_lithops_ecmwf.py`** into the image (verify:
no `COPY`/`ADD` of the script — it only `pip install`s deps + bakes the
template). Instead, `run_with_lithops()` calls
`fexec.map(process_ecmwf_date, …)` (`run_lithops_ecmwf.py:894`). Lithops
**cloudpickles** `process_ecmwf_date` and everything it reaches (all sibling
`__main__` functions: `process_single_member_stage2`, `ensure_template`,
`build_deflated_stores_from_template`, …) **by value**, plus the era globals it
captures — `REFERENCE_DATE`, `ECMWF_RESOLUTION`/`STREAM_PATH`, `CONTROL_STREAM`
(read at import, lines 122/131/139) and the `template_date=REFERENCE_DATE`
default arg — and ships the pickle through GCS at **every** `map()`.

> **Consequence:** a pure **Python-logic** fix (per-level keys,
> resolution-awareness, control-stream selection) reaches workers the moment you
> run the **updated client** `run_lithops_ecmwf.py` — *even on an old image*. It
> does **not**, by itself, require a Docker rebuild. The only thing that must be
> current for Path A is the **`.py` you launch from** (the private copy, made
> current by `40b6759`). This is also why you can rebuild the image on the VM
> and then **come back and submit the run from here** via `uv run` — the worker
> code is whatever this client pickles, not whatever the image holds.

**Path B — the template + era env: baked into the image, NO client override.**
`ensure_template()` (`run_lithops_ecmwf.py:172`) resolves, *at runtime on the
worker*, `os.environ.get('ECMWF_TEMPLATE_PATH')` **first** — and on any built
image that path **always exists** (the `Dockerfile` `curl`s the per-era
tarball to `/opt/ecmwf_templates/…` and sets `ECMWF_TEMPLATE_PATH` to it,
lines 79–93). So a worker **loads the template baked into its image and returns
before the `TEMPLATE_URL` fallback is ever reached.** Critically,
`fexec.map(...)` passes **no `extra_env`** — there is **no client-side way to
override the baked template** a running image uses.

> **Consequence:** the **template is image-bound**. A worker on the **50r1**
> image will load the **50r1 14-level** template for *every* date you send it,
> no matter that the client cloudpickled `49r1` era globals. To put the **49r1
> 13-level** template in front of workers you **must build a `:49r1` image**
> (`--build-arg TEMPLATE_ARTIFACT=…-49r1-perlevel.tar.gz`) and **deploy that
> tag** (§3.1–3.2). **This is the real, non-negotiable reason era 2 needs a
> rebuild** — not the Python logic (Path A), but the baked template (Path B).

**Where the era constants land (belt-and-suspenders).** Reference-date /
resolution / control-stream travel **both** ways: cloudpickled from the client
(Path A) **and** baked as image-env defaults (Path B). When you pass
`--era 49r1` they agree, so they are not the blocker — the template is. (The
§3.2 "both agree on 49r1" note refers to exactly this redundancy.)

**Secondary rebuild trigger — the worker dependency set.** The image pins the
worker-side libs (`lithops`, `pandas`, `pyarrow`, `gcsfs`, `s3fs`, …) and the
lithops Cloud Run proxy (`lithopsproxy.py`). A cloudpickled function can only
call libraries that already exist *in the image*. So if a code change starts
importing a **new** dependency, or you bump the **lithops** version (the
client/runtime wire-format must match), that **does** force a rebuild — over
and above the template. Pure logic changes against the already-baked libs do
not.

**Summary table — what each change actually requires:**

| Change to… | Reaches workers via | Needs Docker rebuild? |
|---|---|---|
| Python processing logic (per-level keys, paths, retries) | cloudpickle at `map()` (Path A) | **No** — just run the updated client `.py` |
| Era constants (ref-date / resolution / control stream) | cloudpickle **+** image env (both) | No (client `--era` carries them) |
| **Per-era template (e.g. 49r1 13-level vs 50r1 14-level)** | **baked image only** (Path B) | **Yes — mandatory** (the reason for era 2) |
| New/upgraded worker dependency or lithops version | baked image only | **Yes** |

### Credentials (the JSON files you must have on the run machine)

> **Run the backfill from the PRIVATE repo** (`cno-e4drr/devops/lithops_cr_ecmwf_gik/`).
> The public mirror has **no** `service_account/` or `terraform/` and its
> `lithops_config.yaml` points at a key path that does not exist there — so
> Lithops cannot authenticate from the public copy. This is by design (keeps
> cloud creds out of the public repo). All §3 commands assume the private dir.

All are **git-ignored** (`.gitignore`: `*.json`) — they live on your local /VM
run machine, **not** in the repo, and are **not** present in this sandbox.
Three JSON keys are involved across the lifecycle:

1. **`service_account/ecmwf-lithops-deployer-key.json`** — the Lithops backend
   key. `lithops_config.yaml` points at it (`gcp.credentials_path`). Used by
   `run_lithops_ecmwf.py` to submit/scale Cloud Run workers and to write GCS.
   Regenerate (if missing) from terraform:
   ```
   cd service_account && terraform output -raw service_account_key_private \
     | base64 -d > ecmwf-lithops-deployer-key.json
   ```
2. **gcloud user/SA auth for Cloud Build** — to *build & push* the 49r1 image
   you need `gcloud auth` with build perms (or activate the deployer SA key:
   `gcloud auth activate-service-account --key-file=service_account/ecmwf-lithops-deployer-key.json`).
3. **Worker-side GCS write** — workers write parquets to GCS via `gcsfs`. On
   Cloud Run this uses the **service identity** of the Lithops-deployed service
   (no JSON shipped into the function); the deployer SA must have
   `roles/storage.objectAdmin` on `gik-ecmwf-aws-tf` (already provisioned in
   `terraform/`). AWS source reads are **anonymous** (`AWS_NO_SIGN_REQUEST=YES`
   baked into the image) — no AWS creds needed.

> Note: `coiled-data.json` (the GCS key used in **Step 1** Coiled scans) is
> **not** used by the Lithops Step-2 path. Don't confuse the two.

**So: the only JSON you must place on the run machine is
`service_account/ecmwf-lithops-deployer-key.json`** (plus a gcloud login for
Cloud Build). Nothing new to author.

---

## 3. Execution runbook — era 2 (49r1)

All commands run from `cno-e4drr/devops/lithops_cr_ecmwf_gik/`.

### 3.0 Pre-flight (verify, don't assume)

```bash
# a) deployer key present?
test -f service_account/ecmwf-lithops-deployer-key.json && echo "key OK"

# b) gcloud authed for Cloud Build (or activate the SA)
gcloud auth list
# gcloud auth activate-service-account --key-file=service_account/ecmwf-lithops-deployer-key.json

# c) what era-2 dates already exist in GCS (defines the gap to fill)
gsutil ls "gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/**/00z/" | wc -l   # spot-check counts
gsutil ls "gs://gik-ecmwf-aws-tf/run_par_ecmwf/2025/" 
gsutil ls "gs://gik-ecmwf-aws-tf/run_par_ecmwf/2026/0*/"
# (51 *.parquet per YYYYMMDD/00z = complete; <51 or missing = (re)run that month)

# d) dry-run the full era-2 month plan (no Cloud Run spend)
bash run_backfill_00z.sh --era 49r1 --from 2024-02 --to 2026-05 --dry-run
```

### 3.1 Build the 49r1 image (Cloud Build)

```bash
gcloud builds submit --config=cloudbuild.yaml --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
  --substitutions=_ERA=49r1,_TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz,_REFERENCE_DATE=20250515,_RESOLUTION=0p25,_CONTROL_STREAM=enfo
# → gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1
```

### 3.2 Point Lithops at the 49r1 tag, then deploy

> **The deploy reference is currently UNTAGGED.** `lithops_config.yaml:32` and
> README Step 3 (`line 266`) both reference the bare
> `gcr.io/e4drr-crafd/ecmwf-lithops-runtime` (→ resolves to `:latest`), **not**
> a per-era tag. Only the new "Three template eras" README section uses
> `:${_ERA}` tags. So the era-tag override below is **mandatory** — without it
> Lithops pulls `:latest` (whatever era was last pushed to it), which is not a
> reliable era selector. (Optional cleanup, not required for this run: pin
> `lithops_config.yaml` + README Step 3 to the explicit era tag.)

Whatever `:latest` currently is, backfilling era 2 against a 50r1/`:latest`
image would read the wrong control stream and a 14-level template. Two ways to
make Lithops use the 49r1 image:

- **Recommended:** set the runtime tag in `lithops_config.yaml`
  (`gcp_cloudrun.runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1`) for
  the duration of the era-2 backfill, then deploy:
  ```bash
  lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1 \
    -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml
  ```
  (Or skip — Lithops auto-deploys on first `fexec.map()`.)
- Revert the tag to `:50r1` when era-2 is finished so daily ops resume on 50r1.

> The image bakes the era env, **and** `run_backfill_00z.sh --era 49r1` exports
> the same env client-side — belt-and-suspenders, both agree on 49r1.

### 3.3 Run the backfill (28 monthly waves, 00z)

```bash
# full era-2 range, live
bash run_backfill_00z.sh --era 49r1 --from 2024-02 --to 2026-05 2>&1 \
  | tee logs/backfill_00z/era2_full.log

# resume a sub-range after a gap audit, e.g.
bash run_backfill_00z.sh --era 49r1 --from 2025-01 --to 2025-12
```

Each month = one Cloud Run wave of ≤31 dates (≤35 workers), ~8.5 min;
per-date retry (3×, 30 s) is internal to `run_lithops_ecmwf.py`.
Per-month + summary logs land in `logs/backfill_00z/`.

> **Note on the script's hard-coded default range** (`FROM_MONTH=2024-03`):
> the script defaults start at **March**, which would **skip 2024-02-29** (the
> single era-start day in Feb). Pass `--from 2024-02` explicitly (the Feb wave
> processes only the 29th, the one valid 0.25° date that month). This is the
> one easy-to-miss boundary in era 2.

### 3.4 Verify completeness

```bash
# Every YYYYMMDD/00z under era-2 months should hold exactly 51 parquets.
for ym in 2024/02 2024/03 ... 2026/05; do
  gsutil ls "gs://gik-ecmwf-aws-tf/run_par_ecmwf/${ym}/**/00z/*.parquet" | wc -l
done
# Expect 51 × (#dates in that month). Re-run any month short of that.
```
Spot-check a 9-level date (e.g. `20240301`) and a 13-level date (e.g.
`20250515`): open one member parquet, confirm `u/isobaricInhPa/{hPa}` keys —
9 vs 13 distinct levels respectively, grid `[1,721,1440]`.

---

## 4. The other eras (same machine, after era 2)

Identical pattern — only the `--substitutions` (build) and `--era` + range
(backfill) differ. Build the era image, point Lithops at its tag, deploy, run.

| Era | Build `--substitutions` | Backfill range (00z) | Dates | Control |
|---|---|---|---|---|
| **2 — 49r1** (this plan) | `_ERA=49r1,_TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz,_REFERENCE_DATE=20250515,_RESOLUTION=0p25,_CONTROL_STREAM=enfo` | `2024-02` → `2026-05` | 804 | enfo |
| **3 — 50r1** (default img) | defaults (`_ERA=50r1`) | `2026-05` → present | ongoing | **oper** |
| **1 — 0.4°-beta** | `_ERA=0p4,_TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz,_REFERENCE_DATE=20230601,_RESOLUTION=0p4,_CONTROL_STREAM=enfo` | `2023-01` → `2024-02` | ~407 | enfo |

**Known `--era` wiring gap (verified on disk).** Only `run_backfill_00z.sh`
(private + public) carries the `--era` block. The **`run_backfill_06z/12z/18z.sh`**
scripts (public-only) and the private **`run_jan2026_backfill.sh`** (all-4-runs,
Jan 2026 — inside era 2) are **not** era-wired (`ERA`-count 0) — that work was
started but interrupted. For this **00z-only** backfill that gap is irrelevant.
If multi-cycle is ever needed, port the identical `--era` block into them first
(their layout differs from 00z: `shift 2` parsing + `RUN_HOUR` +
`GOOGLE_APPLICATION_CREDENTIALS` anchor), and remember 06z/18z ENS only reach
144h (under-fills the 85-step template).

Era boundaries (don't cross with the wrong image):
- 0.4° ends `2024-02-28`; 49r1 starts `2024-02-29 00z`.
- 49r1 ends `2026-05-12 00z`; 50r1 starts `2026-05-12 06z` (and **control moves
  to `oper/fc`** → must use the 50r1 image for ≥ that cycle).

> **Caveat — no in-script era/date guard.** `run_lithops_ecmwf.py` does not
> validate that a requested date falls inside the deployed era's window; it
> trusts the deployed env. Running an out-of-era date silently builds against
> the wrong template/path. Mitigation: keep each era's backfill range inside
> its boundary (tables above) and verify the deployed runtime tag before each
> backfill. (Optional hardening: add a date-range assertion keyed on
> `ECMWF_RESOLUTION`/`REFERENCE_DATE` — not required for this run.)

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Deployed image is 50r1 → wrong template/control for era 2 | Build+deploy `:49r1` and verify the runtime tag before backfill (§3.2) |
| `lithops_config.yaml` runtime is untagged (`:latest`) → ambiguous era | Override to explicit `:49r1` tag for the backfill (§3.2); revert after |
| Running from the public mirror → no creds, auth fails | Run from the **private** repo only (§2 credentials box) |
| `2024-02-29` skipped by default `FROM_MONTH=2024-03` | Pass `--from 2024-02` explicitly (§3.3 note) |
| 9-level dates (pre-2025-01-14) under the 13-level template | Verified safe superset — 4 extra levels left empty (Step-1 doc §4.2) |
| Cold-start HTTP 500 on >35 concurrent dates | `max_workers=35` ≥ 31 dates/month → one wave/month; internal 3× retry |
| Partial month from a transient worker death | Idempotent re-run of that month overwrites; §3.4 count check finds shortfalls |
| Missing `.index` for some old dates | `validate_index_availability` returns failure for that date; result marked unsuccessful, rest proceed |
| Forgot to revert to 50r1 after era-2 | Daily ops would run on 49r1 img → revert `lithops_config.yaml` runtime tag to `:50r1` post-backfill |

---

## 6. One-screen execution checklist (era 2)

1. `service_account/ecmwf-lithops-deployer-key.json` present; `gcloud` authed.
2. `gsutil` audit → list era-2 dates already complete (51 parquets).
3. `gcloud builds submit … _ERA=49r1 …` → image `:49r1`.
4. Set `lithops_config.yaml runtime: …:49r1`; `lithops runtime deploy …:49r1`.
5. `bash run_backfill_00z.sh --era 49r1 --from 2024-02 --to 2026-05 --dry-run` → sanity.
6. Same without `--dry-run` (or per-month resume of gaps) → `tee` a log.
7. `gsutil` count check = 51 × dates/month for every era-2 month; re-run shortfalls.
8. Revert `lithops_config.yaml` runtime tag to `:50r1`; redeploy for daily ops.
