# 18z backfill — session C runbook

**Status:** ✅ **Ready to run — no new images required.**
**Why:** 18z is a **short run reaching only +144h** (S3-verified), identical in axis to
06z — and proven on 2026-07-29: a 06z date built from the existing `:50r1` image
yielded **49 steps, max_step 144** (vs 85 / 360 for 00z). Templates are
cycle-agnostic; see [`../README.md`](../README.md) §1a.

**Read first:** [`../README.md`](../README.md) · [`../06z/PLAN.md`](../06z/PLAN.md) · [`../../ECMWF_00Z_BACKFILL_SUMMARY.md`](../../ECMWF_00Z_BACKFILL_SUMMARY.md)

---

## Scope

| Era | Runtime tag | Memory | Dates | Window |
|-----|-------------|--------|-------|--------|
| 0p4 | `:0p4` | **3072 MB** | ~401 | 2023-01-18 → 2024-02-28 |
| 49r1 | `:49r1` | **3072 MB** | ~804 | 2024-02-29 → 2026-05-12 |
| 50r1 | `:50r1` | **3072 MB** | ~51 | 2026-05-13 → 2026-07-02 |
| | | | **~1,256** | |

Writes to `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/18z/`.

⚠️ **6 permanently absent dates:** `20230427`–`20230502` (0p4, 404 on S3).

---

## ⚠️ The one thing unique to this plan: memory = 3072 MB

Lithops derives the Cloud Run **service name from runtime image + memory**. 18z uses
the *same* images as 06z, 12z and 00z, so at the same memory all of them would land on
one service and **share a single 35-instance pool**.

Deploying at **3072 MB** yields distinct service names and an independent scaling pool
(06z stays at 2048, 12z at 2560). The extra memory is incidental; separation is the
point. Set `runtime_memory: 3072` in this session's `lithops_config.yaml` and keep it
consistent between deploy and run — a mismatch makes lithops auto-deploy a fresh
service mid-wave.

---

## Phase 0 — unblock

No images to build. This session only deploys the three **existing** images at 3072 MB:

```bash
for TAG in 0p4 49r1 50r1; do
  lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:$TAG \
    -b gcp_cloudrun -s gcp_storage --memory 3072 --config lithops_config.yaml

  uv run --python 3.12 --with lithops==3.6.4 --with httplib2 --with google-auth \
    --with google-api-python-client --with google-cloud-storage \
    lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:$TAG \
    -b gcp_cloudrun -s gcp_storage --memory 3072 --config lithops_config.yaml
done

lithops runtime list -b gcp_cloudrun --config lithops_config.yaml | grep 3072
```

Confirm six **distinct** service names at 3072 MB, none colliding with session A's
2048 MB set. Then raise `maxScale` per [`../README.md`](../README.md) §2.

---

## Step 1 — isolated checkout & environment

```bash
git worktree add /tmp/gik-18z HEAD
cd /tmp/gik-18z/devops/lithops_cr_ecmwf_gik
sed -i -E 's|(runtime_memory:) *[0-9]+|\1 3072|' lithops_config.yaml

export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
gcloud auth activate-service-account \
  --key-file=service_account/ecmwf-lithops-deployer-key.json --project=e4drr-crafd
```

## Step 2 — `--run` passthrough

Add a `--run` flag (default `00`) to `run_backfill_00z.sh`, pass it to
`run_lithops_ecmwf.py` and log to `logs/backfill_18z/`. The `--era` case block already
has the three tags — no additions needed.

## Step 3 — Herbie verification gate (3 dates, ~25 min)

| Era | Suggested date | Runtime tag | Pass threshold |
|-----|----------------|-------------|----------------|
| 0p4 | `20230601` | `:0p4` | r ≥ 0.9997 |
| 49r1 | `20250115` | `:49r1` | r ≥ 0.9999 |
| 50r1 | `20260601` | `:50r1` | r ≥ 0.9999 |

Run one date at `--run 18`, confirm 51/51 files **and** the axis stops at +144h, then
compare against Herbie with `ecmwf/compare_gik_herbie_pressure.py`.

Session A validates the same images at 06z. **Both must still pass** — passing at 06z
does not prove 18z reads the right S3 keys (the cycle appears in both the path and the
filename). Do not skip this on the assumption that A covered it.

## Step 4 — full waves

```bash
# 0p4
export ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4  ECMWF_CONTROL_STREAM=enfo
bash run_backfill_00z.sh --era 0p4  --run 18 --from 2023-01 --to 2024-02

# 49r1
export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
bash run_backfill_00z.sh --era 49r1 --run 18 --from 2024-03 --to 2026-05

# 50r1
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
bash run_backfill_00z.sh --era 50r1 --run 18 --from 2026-05 --to 2026-07
```

⚠️ Era boundaries are **mid-month** (`20240229`, `20260512/13`) — handle those dates
with explicit `--start-date`/`--end-date` calls.

## Step 5 — verify

**Never judge by exit code** — a successful wave hangs at exit; `timeout` returns
**124**. Count GCS objects: 51/51 per date, contiguous apart from the 6 known 0p4 holes.

## Step 6 — hand off

Report dates written, any `<51` dates, and Herbie stats. **Do not** run the HF mirror
or Icechunk ingest — serial Phase 3, after all three cycles land.

---

## Estimated wall time

~1,256 dates × ~7 min/date (49 steps vs ~85 for a 00z date). At 35-wide ≈ **4 h**;
~85 min only at ~103 concurrent.

## Start offset

Start **~10 min after session A** (A at T+0, B at T+5, C at T+10) so three cold-start
storms do not collide in one region — the HTTP 500 cold-start race that
`max_workers 35` was originally chosen to avoid.
