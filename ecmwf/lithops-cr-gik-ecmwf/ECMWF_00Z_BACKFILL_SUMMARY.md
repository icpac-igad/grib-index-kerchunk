# ECMWF IFS/ENS — how the 00z GIK parquet corpus was built (3 eras, 3 Cloud Run runtimes)

**Status:** ✅ COMPLETE — 1,256 contiguous 00z dates, `20230118 → 20260702`, all 51/51 members, zero unexpected gaps.
**Authoritative catalog:** `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/`
**Public mirror:** [`E4DRR/gik-ecmwf-par-v2`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par-v2) — 64,056 parquets, 21 GB
**Completed:** 2026-07-06 (final gap fills + Herbie validation + HF mirror)

This document is the reference for replicating the same corpus at **06z, 12z and 18z**.
Read it before starting any cycle backfill. The companion plan is [`cycles/README.md`](cycles/README.md).

---

## 1. The core design: one baked runtime image per era

The ECMWF model era is **baked into the Lithops runtime image at build time**, not
passed at runtime. `ensure_template()` in `run_lithops_ecmwf.py` returns the image's
baked template path **before** any client-side `TEMPLATE_URL` override is consulted.

> **Consequence:** running era A's dates against era B's deployed image silently
> produces *zero* files (`51/51 "Template not found"`). This was proven when 50r1
> dates were run against the `:49r1` image. **Each era needs its own image.**

Build args (`Dockerfile` / `cloudbuild.yaml`) select the era:

```bash
gcloud builds submit --config=cloudbuild.yaml --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
  --substitutions=_ERA=<tag>,_TEMPLATE_ARTIFACT=<hf.tar.gz>,_REFERENCE_DATE=<yyyymmdd>,_RESOLUTION=<0p25|0p4>,_CONTROL_STREAM=<enfo|oper>
```

---

## 2. The three eras — image, template, S3 path, coverage

| Era | Runtime image | Baked template | Grid | Levels | Control stream | Ref date | S3 stream path | Dates | Window |
|-----|---------------|----------------|------|--------|----------------|----------|----------------|-------|--------|
| **0p4** | `…/ecmwf-lithops-runtime:0p4` | `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` | 451×900 (0.4°) | 9 | `enfo` | `20230601` | `0p4-beta/enfo` | **401** | 2023-01-18 → 2024-02-28 |
| **49r1** | `…/ecmwf-lithops-runtime:49r1` | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` | 721×1440 (0.25°) | 13 | `enfo` | `20250515` | `ifs/0p25/enfo` | **804** | 2024-02-29 → 2026-05-12 |
| **50r1** | `…/ecmwf-lithops-runtime:50r1` | `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` | 721×1440 (0.25°) | 14 | `oper` | `20260513` | `ifs/0p25/oper` | **51** | 2026-05-13 → 2026-07-02 |
| | | | | | | | | **1,256** | **20230118 → 20260702** |

### Era cutovers (S3-proven, not inferred)

- **0p4 → 0p25 at `20240229`.** The 0.4°-beta stream stops; `ifs/0p25` begins.
- **49r1 → 50r1 at `20260512` → `20260513`.** At this boundary `enfo` drops 51→50
  members and pressure levels go 13→14, so the control moves to the `oper` stream.

> ⚠️ **These boundaries hold at 00z only.** The cutovers are **cycle-granular**: at
> 06z, `20240228` is already 0p25/49r1 (the 0.4° stream 404s) and `20260512` is
> already 50r1 (control has moved to `oper`). See `cycles/README.md` §1b — it
> matters for 06z/12z/18z, not for this 00z corpus.

> ⚠️ **Known typo in the record:** an earlier session stated "49r1 ends 20260612".
> That was a `0512` typo. The true cutover is **20260512/13**, which is why the
> 50r1 rebake starting at `20260513` was correct and needed no re-run. Do not
> propagate the 0612 figure into the 06z/12z/18z plans.

### Unbackfillable dates

6 dates in the 0p4 era — **`20230427` → `20230502`** — return 404 on S3. ECMWF never
published them at 0.4°. `401/401` publishable dates are complete; these 6 are the
only permanent holes.

---

## 3. Deployed Cloud Run services (one per era × per lithops version)

Lithops encodes its **local** version into the Cloud Run service name and its GCS
metadata path, so each image is registered twice — 3.6.3 (this deployer host's CLI)
and 3.6.4 (the run host's pinned version). A run host that finds no matching
service auto-deploys one, which wastes several minutes at wave start.

```
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4    2048 MB  3.6.4  lithops-worker-364-2d47ab5603
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4    2048 MB  3.6.3  lithops-worker-363-3fafb82775
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1   2048 MB  3.6.4  lithops-worker-364-41b681e756
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1   2048 MB  3.6.3  lithops-worker-363-e004c694d6
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1   2048 MB  3.6.4  lithops-worker-364-5a680638f5
gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1   2048 MB  3.6.3  lithops-worker-363-d1cea16f95
```

All: **region `europe-west3`** (Frankfurt — closest GCP region to ECMWF's S3 in
`eu-central-1`), **2 GB**, **containerConcurrency = 1**, **maxScale = 35**.

> **Service identity = runtime image + memory.** Two concurrent sessions using the
> same era *and* the same memory hit the *same* Cloud Run service and therefore
> share its 35-instance ceiling. This is the central capacity fact for running
> cycles in parallel — see [`cycles/README.md`](cycles/README.md) §3.

---

## 4. The execution routine

`run_backfill_00z.sh --era {0p4|49r1|50r1}` drives everything: it walks whole
calendar months, and for each month invokes `run_lithops_ecmwf.py` with
`--max-workers 35`, so **one month = one concurrent wave** (28–31 dates).

Per-era env is exported by the script and **must match the deployed image**:

```bash
# 0p4
ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4  ECMWF_CONTROL_STREAM=enfo
# 49r1
ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
# 50r1
ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
```

Plus, always:

```bash
export UV_PYTHON=3.12                              # host↔runtime Python must match
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf  # else writes to the LEGACY prefix
export AWS_NO_SIGN_REQUEST=YES
```

### ⚠️ How era selection actually works — it is TWO independent switches

This is the least obvious thing in the whole system, and the easiest way to
silently produce nothing. **Neither `run_backfill_00z.sh` nor `rebake_50r1_00z.sh`
selects the Cloud Run image.** They only export env vars. Grep them and you will
find no `runtime:` assignment — only a *comment* stating it as a prerequisite.

| # | Switch | Set where | Controls | Read by |
|---|--------|-----------|----------|---------|
| 1 | `--era` → `ECMWF_RESOLUTION`, `ECMWF_CONTROL_STREAM`, `ECMWF_REFERENCE_DATE` | `run_backfill_00z.sh` case block (exported env) | **Which S3 bytes are READ** — `STREAM_PATH` = `0p4-beta` vs `ifs/0p25`, and the `-enfo-ef` vs `-oper-fc` suffix | `run_lithops_ecmwf.py` → `ecmwf_index_url()` |
| 2 | `runtime:` tag | **`lithops_config.yaml:32` — by hand, or `sed`** | **Which image/template DECODES them**, hence which Cloud Run service runs | `run_lithops_ecmwf.py:872` → `FunctionExecutor(config_file=…)` |

**The two are not linked.** `--era 0p4` while the config still says `:50r1` is
accepted without complaint: the driver reads 0.4° S3 paths and hands them to
workers holding the 50r1 14-level template. Every date returns
`51/51 "Template not found"` and **zero files are written**.

Why the mismatch cannot self-correct: inside the worker, `ensure_template()`
returns `ECMWF_TEMPLATE_PATH` — the image's baked template — and returns it
**before** `TEMPLATE_URL` is ever consulted. The `TEMPLATE_URL` exported by the
shell scripts is therefore **dead code on Cloud Run**; it is only a fallback for
local sequential runs. You cannot override the era from the client side.

> **Rule: before every wave, confirm switch 2 matches switch 1.**
> ```bash
> grep -E 'runtime: gcr' lithops_config.yaml
> ```

The only scripts that set both are the gap runners
(`queue_49r1_50r1_gaps.sh`, `run_gaps_49r1-20240229_50r1-20260701-20260702.sh`),
via:

```bash
set_rt(){ sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$1|" lithops_config.yaml; }
```

`lithops_config.yaml` is re-read **fresh at every month boundary**, so the tag may
be flipped between waves — but **never during one**. `queue_49r1_50r1_gaps.sh`
blocks on the running wave's PID before flipping.

---

## 5. Four gotchas that cost real time

| # | Gotcha | Symptom | Fix |
|---|--------|---------|-----|
| 1 | **Driver hangs at interpreter exit** | Work completes, full output prints (through `List all parquets:`), process never exits — a lingering lithops thread | Wrap each wave in `timeout`; **exit 124 = success**, not failure. Always confirm via per-date GCS file counts (51/date), never by process exit code. |
| 2 | **Wrong default GCS prefix** | Parquets land in the legacy `run_par_ecmwf/` catalog | `export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf` — the public `run_lithops_ecmwf.py` still defaults to the legacy prefix |
| 3 | **Wrong active gcloud account** | `403` on the registry / `forbidden from accessing bucket e4drr-crafd_cloudbuild` | `gcloud auth activate-service-account --key-file=service_account/ecmwf-lithops-deployer-key.json --project=e4drr-crafd` |
| 4 | **`uv run --with lithops` misses GCP deps** | `ModuleNotFoundError: httplib2` → `google.auth` → `google.cloud.storage`, one at a time | Add all four: `--with httplib2 --with google-auth --with google-api-python-client --with google-cloud-storage`. The system `lithops` (3.6.3) already has them. |

A fifth, historical: `namegenerator` was removed from PyPI and broke every runtime
Cloud Build. It was a vestigial pin lithops 3.6.x neither requires nor imports —
removed in `0714d3a` (private) / `8efff5c` (mirror).

---

## 6. Observed throughput (the basis for all cycle planning)

| Unit | Observation |
|------|-------------|
| One wave (≤35 dates, concurrent) | **~8–20 min wall**, largely independent of date count |
| 50r1 wave C (6 dates) | 7.5 min work + hang-at-exit |
| Typical 49r1 month (30–31 dates) | 10–21 min |
| Effective concurrency ceiling | **35 per Cloud Run service** (`maxScale`) |
| Full 00z corpus (1,256 dates) | **≈ 6 h sequential** at 35-wide |
| Cost | ~$0.026 / date → ~$33 for 1,256 dates |

---

## 7. Verification — the Herbie gate

Every era was validated against **Herbie** ground truth *before* its full wave was
released, then re-validated across eras afterward.

- **Tool:** `ecmwf/compare_gik_herbie_pressure.py` — reads a GIK parquet from GCS,
  streams a pressure-level variable via `gribberish` byte-range reads, fetches the
  Herbie equivalent, computes correlation / RMSE / max|diff| and plots ensemble
  mean + spread maps.
- **0p4 go-signal:** `20230601`, `t@500` → **r = 0.9999**. This gated the 401-date wave.
- **Cross-era final:** 2 random dates × 3 eras, 500 & 850 hPa → 12 plots.
  - 49r1 / 50r1: **r ≈ 1.0000**
  - 0p4: **r ≥ 0.9997** (RMSE ~0.02–0.05 K; residual is grid-reindexing, expected)
- **Artifacts:** `ecmwf/gik_vs_herbie/` — `GIK_vs_Herbie_Evaluation.md`, `0p4_eval/`,
  `random_3era_eval/`; 23 validation files mirrored alongside the data on HF.

**The pattern to repeat per cycle: validate one date per era first, read the
correlation, and only then release the full wave.**

---

## 8. Where the 00z work actually ran

The backfill was **not** driven from this deployer host. It ran from
`/scratch/notebook/grib-index-kerchunk/ecmwf/lithops-cr-gik-ecmwf` on the Coiled
scheduler VM (`aifs-etl`), with this repo mirrored at `/scratch/notebook/cno-e4drr`.

This host's role is **build + deploy runtimes**; the run host drives waves.
Anyone replicating for 06z/12z/18z should confirm which host they are on and that
`UV_PYTHON=3.12` matches the runtime's Python.

---

## 9. Downstream

- **HF mirror:** `ecmwf/mirror_gcs_to_hf_v2.py` — idempotent and resumable;
  future dates need only `--from-month YYYY-MM`. Layout `par/{YYYY}/{MM}/{date}/00z/`.
- **Icechunk store:** `ecmwf/icechunk-par/backfill_all_eras.py` →
  `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens`.
  ⚠️ **Single writer only** — concurrent writers to one Icechunk store conflict.
  This is a hard constraint on parallelising the cycles at the Icechunk stage.

---

## 10. Reference commits

| Commit | What |
|--------|------|
| `0714d3a` | Remove dead `namegenerator` pin (unblocked all runtime builds) |
| `4ea827c` | Build + deploy `:0p4` runtime (0.4°, pre-49r1 era) |
| `7d915b6` | Extend 50r1 rebake to `20260630` (wave C) |
| `8693ccc` | Add `rebake_50r1_00z.sh` wave driver |

Era docs: `2026-06-10-era2-49r1-rebuild-DONE-ready-for-backfill.md`,
`2026-06-26-50r1-runtime-build-deploy-DONE-ready-for-run.md`,
`2026-07-01-0p4-runtime-build-deploy-DONE.md`.

---

## 11. Current real-time operations — 50r1 only, July 2026 onward

Everything from `20260513` to today is the **50r1 era**. There is no era switching
in day-to-day ops: 0p4 and 49r1 are closed historical windows, so routine work
touches exactly one runtime.

### Which Cloud Run to use

| | |
|---|---|
| **Image** | `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1` |
| **Config line** | `lithops_config.yaml:32` → `runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1` |
| **Service (lithops 3.6.4)** | `lithops-worker-364-5a680638f5` ← the run host pins 3.6.4 |
| **Service (lithops 3.6.3)** | `lithops-worker-363-d1cea16f95` ← this deployer host's system CLI |
| **Region / size** | `europe-west3` · 2 GB · 2 vCPU · concurrency 1 · maxScale 35 |

You do **not** choose the service by name. Lithops derives it from
`runtime` + `runtime_memory` + **its own version**, which is why the same image is
registered twice. Set the tag; the right service follows. If the host's lithops
version has no registered service, lithops silently auto-deploys one mid-wave,
costing several minutes — so keep both registrations alive.

The repo's committed default is already `:50r1` — correct for ongoing ops. Confirm
before every run anyway (see §4, switch 2):

```bash
grep -E 'runtime: gcr' lithops_config.yaml
# expect: runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1
```

If a historical backfill left it on `:0p4` or `:49r1`, restore it:

```bash
sed -i -E 's|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\150r1|' lithops_config.yaml
```

### Running the rolling tail

The catalog's last 00z date is **`20260702`**. Anything after that is an open tail.

```bash
cd devops/lithops_cr_ecmwf_gik

# 1. auth (the deployer SA token expires; re-activate freely)
gcloud auth activate-service-account \
  --key-file=service_account/ecmwf-lithops-deployer-key.json --project=e4drr-crafd

# 2. environment — all four lines matter
export UV_PYTHON=3.12                              # host↔runtime Python must match
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf  # else writes to the LEGACY catalog
export AWS_NO_SIGN_REQUEST=YES
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper

# 3. confirm the runtime tag is :50r1  (§4, switch 2)
grep -E 'runtime: gcr' lithops_config.yaml

# 4. run the tail — chunk to ≤35 dates so each wave is one Cloud Run scale-out
timeout 1500 uv run run_lithops_ecmwf.py \
    --start-date 20260703 --end-date 20260731 \
    --run 00 --max-workers 35 --yes
```

`rebake_50r1_00z.sh` is the same thing wrapped in waves with the `timeout` guard
and a summary log; extend it with another `run_wave` line rather than editing the
existing ones, so past waves stay reproducible.

### Daily operation

One date, ~35 workers unnecessary — use a small pool:

```bash
uv run run_lithops_ecmwf.py --start-date $D --end-date $D --run 00 --max-workers 4 --yes
```

Then mirror to HuggingFace: `ecmwf/mirror_gcs_to_hf_v2.py --from-month YYYY-MM`
(idempotent and resumable, so re-running a month is safe).

### Verifying — the part that trips people up

**A successful run exits non-zero.** The driver finishes the work, prints its full
output through `List all parquets:`, then hangs at interpreter exit on a lingering
lithops thread. Under `timeout` that surfaces as **exit 124, which means success**
(§5, gotcha 1). Never gate on the exit code:

```bash
gsutil ls "gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/2026/07/${D}/00z/**" | wc -l
# expect 51
```

### When 50r1 ends

At the next IFS cycle upgrade, member count and/or pressure levels will shift
mid-stream (49r1→50r1 went 51→50 members, 13→14 levels at `20260512/13`). That
requires a **new era**: new HF template → new `_ERA` Cloud Build → deploy under
both lithops versions → new tag in `lithops_config.yaml`. Follow
`2026-07-01-0p4-runtime-build-deploy-DONE.md` as the worked example, and validate
the first date against Herbie (§7) before releasing any wave.
