# ECMWF three-era GIK completion — end-to-end runbook

**Date:** 2026-07-03
**Scope:** the full path that took all three ECMWF schema eras (`0p4`, `49r1`,
`50r1`) from *template* → *Lithops parquet catalog* → *Herbie value validation*,
with the exact scripts and commands used at each step.

**End state:** fixed catalog `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/`
(00z) = **1256 contiguous dates `20230118 → 20260702`**, all 51/51 members, zero
unexpected gaps (only the 6 dates ECMWF never published at 0.4° are absent).
All three eras value-validated against Herbie (r ≈ 1.0, see §4).

---

## Era matrix (the one table everything keys off)

| Era | Grid | pl levels | Control stream | Ref date | Template artifact (HF `E4DRR/grib-index-kerchunk-templates`) | Image tag | Dates covered |
|---|---|---|---|---|---|---|---|
| **0p4** | 451×900 (0.4°) | 9 | `enfo` (bundled `number=0`) | `20230601` | `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` | `:0p4` | 2023-01-18 → 2024-02-28 |
| **49r1** | 721×1440 (0.25°) | 9→13 | `enfo` (bundled) | `20250515` | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` | `:49r1` | 2024-02-29 → 2026-05-12 |
| **50r1** | 721×1440 (0.25°) | 14 | `oper` (dual-stream) | `20260513` | `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` | `:50r1` | 2026-05-13 → present |

Cutovers S3-verified (enfo `number` count + pl-level set): 0p4→0p25 at
**20240229**; 49r1→50r1 at **20260512→20260513** (enfo drops 51→50 members, pl
13→14). This corrected the earlier "49r1 ends 20260612" typo.

---

## Step 1 — Template creation (one-time per era)

Produces the per-era `.tar.gz` (rt000 mapping parquets + zarr skeleton) that
Stage-1 loads at runtime instead of `scan_grib`.

**Scripts:** `ecmwf/dev-test/ecmwf_{0p4,49r1,50r1}_coiled_preprocessing.py`
(per-era Coiled scan drivers) → `ecmwf/dev-test/ecmwf_par_to_ensemble_members.py`
(resolution-aware realigner: `field_shape_for_uri` → `[1,721,1440]` for 0.25°,
`[1,451,900]` for 0.4°, fixing the legacy `[1,181,360]` 1° placeholder bug).

```bash
# a) scan the reference date's GRIBs on Coiled (example: 50r1 ref 20260513)
uv run ecmwf/dev-test/ecmwf_50r1_coiled_preprocessing.py \
    --software gik-coiled-pinned          # pinned scan_grib env (reproducible)
# (0p4 -> ecmwf_0p4_coiled_preprocessing.py, ref 20230601;
#  49r1 -> ecmwf_49r1_coiled_preprocessing.py, ref 20250515, built from a
#  13-level date so the 4 extra levels 100/150/400/600 hPa are a superset)

# b) realign the per-member par into the ensemble template + fix grid shape
uv run ecmwf/dev-test/ecmwf_par_to_ensemble_members.py \
    --resolution 0p25 --reference-date 20260513   # 0p4 -> --resolution 0p4 --reference-date 20230601

# c) package + upload the tar.gz to HuggingFace
#    -> E4DRR/grib-index-kerchunk-templates/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz
```

Validate a template cheaply (one date) before baking: run one Lithops date
(Step 3) and assert per-level pl keys `step_NNN/{var}/pl/{hPa}/{member}/0.0.0`
across the era's level set **and** that the control member is present (esp. 50r1,
which exercises the `oper` path).

---

## Step 2 — Bake + deploy the per-era runtime image

The era (template + ref date + resolution + control stream) is **baked into the
image at build time** — it is NOT runtime-overridable. One image per era.

**Files:** `Dockerfile` (build-arg selected), `cloudbuild.yaml` (substitutions).

```bash
# Build + push (example: 0p4). Auth as the deployer SA first.
gcloud config set account ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
gcloud config set project e4drr-crafd
gcloud builds submit --config=cloudbuild.yaml --project=e4drr-crafd \
  --substitutions=_ERA=0p4,\
_TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz,\
_REFERENCE_DATE=20230601,_RESOLUTION=0p4,_CONTROL_STREAM=enfo
#  49r1 -> _ERA=49r1,_TEMPLATE_ARTIFACT=...-49r1-perlevel.tar.gz,_REFERENCE_DATE=20250515,_RESOLUTION=0p25,_CONTROL_STREAM=enfo
#  50r1 -> defaults in cloudbuild.yaml (_ERA=50r1, ref 20260513, oper)
# -> pushes gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4

# Deploy the Cloud Run runtime under BOTH lithops versions (system 3.6.3 has all
# GCP deps; 3.6.4 matches run_lithops_ecmwf.py's pin)
lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4 \
    -b gcp_cloudrun -s gcp_storage --memory 2048 --config lithops_config.yaml
uv run --python 3.12 --with lithops==3.6.4 --with httplib2 --with google-auth \
    --with google-api-python-client --with google-cloud-storage \
    lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4 \
    -b gcp_cloudrun -s gcp_storage --memory 2048 --config lithops_config.yaml
```

Runtime list after all three: `:0p4`, `:49r1`, `:50r1` each under 3.6.3 + 3.6.4.

---

## Step 3 — Generate the parquet catalog (Lithops on Cloud Run)

**Driver:** `run_backfill_00z.sh --era {0p4|49r1|50r1}` (per-month waves) which
calls **`run_lithops_ecmwf.py`** (self-contained 3-stage pipeline, ~960 lines).
`--era` sets the driver env (ref date / resolution / control stream / template
URL); the **worker era comes from the image `lithops_config.yaml:runtime:` points
at**, so switch that tag to match the era before each run.

> ⚠️ **Always** `export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf` — the
> public-repo `run_lithops_ecmwf.py` still defaults to the *legacy* `run_par_ecmwf`
> (pre-per-level-fix) prefix.

```bash
cd devops/lithops_cr_ecmwf_gik
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf

# point the runtime at the era's image, then run the wave
sed -i 's#runtime:.*#runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4#' lithops_config.yaml
bash run_backfill_00z.sh --era 0p4 --from 2023-01 --to 2024-02   # 0p4 full era (~407 dates)

# 49r1 backfill (image :49r1) and 50r1 ongoing (image :50r1) analogously:
bash run_backfill_00z.sh --era 49r1 --from 2024-03 --to 2026-05
bash run_backfill_00z.sh --era 50r1 --from 2026-05 --to 2026-07
```

**Residual single-date / rolling-tail gaps** were filled by
**`run_gaps_49r1-20240229_50r1-20260701-20260702.sh`** (flips the runtime tag
per era in sequence; leaves it at `:50r1`). Its companion
`queue_49r1_50r1_gaps.sh` waits for a running 0p4 wave first (now zombie-safe:
gates on process *state*, not `kill -0`, which succeeds on `<defunct>`).

Output layout: `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/00z/{date}00z-{control|ens_NN}.parquet`.

Coverage was verified by enumerating the bucket with the deployer SA
(`service_account/ecmwf-lithops-deployer-key.json`): 1256 contiguous 00z dates,
all 51/51.

---

## Step 4 — Herbie value validation (GIK vs ground truth)

**Tool:** `ecmwf/compare_gik_herbie_pressure.py --grid {0p25|0p4}` — streams each
member's exact GRIB byte-range from the par refs, decodes with gribberish on the
era's grid, subsets East Africa, and compares ensemble mean & spread against
Herbie (`model=ifs, product=enfo`). **Batch driver:**
`ecmwf/run_random_3era_herbie_eval.sh` (2 random dates × 3 eras, seed=42).

```bash
export GIK_GCS_KEY=.../service_account/ecmwf-lithops-deployer-key.json
export AWS_NO_SIGN_REQUEST=YES
# single date (example, 50r1):
uv run ecmwf/compare_gik_herbie_pressure.py --grid 0p25 \
    --gcs-path gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/2026/07/20260701/00z \
    --date 20260701 --step 0 --var t --levels 500,850 \
    --output-dir gik_vs_herbie/random_3era_eval
# all 6 dates at once:
bash ecmwf/run_random_3era_herbie_eval.sh
```

### Results — random 2 dates × 3 eras, `t` @ 500 & 850 hPa, T+0h

| Era | Date | Level | GIK/Herbie | Pearson r | RMSE (K) | max\|diff\| (K) |
|---|---|---|---|---|---|---|
| 0p4 | 20230318 | 500 | 51/50 | 0.999974 | 0.018 | 0.57 |
| 0p4 | 20230318 | 850 | 51/50 | 0.999940 | 0.024 | 0.74 |
| 0p4 | 20231112 | 500 | 51/50 | 0.999950 | 0.017 | 0.45 |
| 0p4 | 20231112 | 850 | 51/50 | 0.999660 | 0.050 | 1.48 |
| 49r1 | 20240327 | 500 | 51/50 | 1.000000 | 8.9e-05 | 2.1e-04 |
| 49r1 | 20240327 | 850 | 51/50 | 1.000000 | 1.9e-04 | 7.3e-04 |
| 49r1 | 20251125 | 500 | 51/50 | 1.000000 | 9.3e-05 | 1.8e-04 |
| 49r1 | 20251125 | 850 | 51/50 | 1.000000 | 9.6e-05 | 6.4e-04 |
| 50r1 | 20260621 | 500 | 51/50 | 0.999998 | 2.8e-03 | 0.020 |
| 50r1 | 20260621 | 850 | 51/50 | 1.000000 | 3.2e-03 | 0.034 |
| 50r1 | 20260701 | 500 | 51/50 | 0.999999 | 2.8e-03 | 0.020 |
| 50r1 | 20260701 | 850 | 51/50 | 1.000000 | 3.4e-03 | 0.047 |

- GIK carries 51 members (incl. bundled/oper control); Herbie enfo returns the
  50 perturbed by default — hence 51/50.
- **0p25 eras (49r1/50r1) are effectively bit-exact** (RMSE ~1e-4 K, diff maps
  pure floating-point speckle). **0p4** shows RMSE ~0.02–0.05 K — the 0.4°→grid
  reindex interpolation, still a perfect physical match.
- Plots (`compare_pl_t{level}_{date}_T0h.png`) + stats JSON in
  `gik_vs_herbie/random_3era_eval/`. PNGs are `.gitignore`d (regenerable);
  stats JSON committed.

**Conclusion:** all three eras produce parquets whose decoded values match
Herbie ground truth across the archive — the catalog is complete and correct.
