# 12z backfill — session B runbook

**Status:** ✅ **Ready to run — no new build required.**
**Why:** 12z has the same 0–360h time axis as 00z (S3-verified: `+360h` → HTTP 200),
so the existing `:0p4`, `:49r1`, `:50r1` images, templates and Cloud Run services
apply unchanged.

**Read first:** [`../README.md`](../README.md) · [`../../ECMWF_00Z_BACKFILL_SUMMARY.md`](../../ECMWF_00Z_BACKFILL_SUMMARY.md)

---

## Scope

| Era | Runtime tag | Dates | Window |
|-----|-------------|-------|--------|
| 0p4 | `:0p4` | ~401 | 2023-01-18 → 2024-02-28 |
| 49r1 | `:49r1` | ~804 | 2024-02-29 → 2026-05-12 |
| 50r1 | `:50r1` | ~51 | 2026-05-13 → 2026-07-02 |
| | | **~1,256** | |

Writes to `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/12z/`.

⚠️ **6 permanently absent dates:** `20230427`–`20230502` (0p4, 404 on S3). Expected — not a failure.

⚠️ 12z shares the **same three Cloud Run services** as 00z daily ops. Coordinate so a
rolling 00z job is not running against the same 35-slot pool.

---

## Step 0 — isolated checkout (mandatory)

Three sessions must not share `lithops_config.yaml`.

```bash
git worktree add /tmp/gik-12z HEAD
cd /tmp/gik-12z/devops/lithops_cr_ecmwf_gik
sed -i -E 's|(runtime_memory:) *[0-9]+|\1 2560|' lithops_config.yaml
```

**Memory 2560 MB is deliberate.** All four cycles now share the same three images, so
at a common memory they would share one 35-slot pool per era. Lithops names services
from image + memory, so 2560 gives 12z its own pool (06z stays at 2048 alongside 00z
daily ops; 18z uses 3072). Deploy at that memory once, before Step 3:

```bash
for TAG in 0p4 49r1 50r1; do
  lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:$TAG \
    -b gcp_cloudrun -s gcp_storage --memory 2560 --config lithops_config.yaml
  uv run --python 3.12 --with lithops==3.6.4 --with httplib2 --with google-auth \
    --with google-api-python-client --with google-cloud-storage \
    lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime:$TAG \
    -b gcp_cloudrun -s gcp_storage --memory 2560 --config lithops_config.yaml
done
```

## Step 1 — environment

```bash
export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
gcloud auth activate-service-account \
  --key-file=service_account/ecmwf-lithops-deployer-key.json --project=e4drr-crafd
```

## Step 2 — add `--run` passthrough to the driver

`run_backfill_00z.sh` hardcodes `--run "00"`. Add a `--run` flag defaulting to `00`
and pass it through to `run_lithops_ecmwf.py`. Keep the change backward-compatible so
the 00z rolling job is unaffected. Log to `logs/backfill_12z/`.

## Step 3 — Herbie verification gate (3 dates, ~25 min)

**Do not skip. Do not start Step 4 until all three pass.**

| Era | Suggested date | Runtime tag | Pass threshold |
|-----|----------------|-------------|----------------|
| 0p4 | `20230601` | `:0p4` | r ≥ 0.9997 |
| 49r1 | `20250115` | `:49r1` | r ≥ 0.9999 |
| 50r1 | `20260601` | `:50r1` | r ≥ 0.9999 |

For each: set the runtime tag, run that single date at `--run 12`, confirm 51/51
files in GCS, then compare against Herbie with
`ecmwf/compare_gik_herbie_pressure.py` (`t@500`, and `t@850` if time allows).

A miss means a template/axis mismatch — **stop and report**.

## Step 4 — full waves

Oldest → newest. Flip `lithops_config.yaml` `runtime:` **only between waves**, never
during one:

```bash
sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$TAG|" lithops_config.yaml
```

```bash
# 0p4 — 2023-01 → 2024-02
export ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4  ECMWF_CONTROL_STREAM=enfo
bash run_backfill_00z.sh --era 0p4  --run 12 --from 2023-01 --to 2024-02

# 49r1 — 2024-02-29 → 2026-05-12
export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
bash run_backfill_00z.sh --era 49r1 --run 12 --from 2024-03 --to 2026-05
#   + edge date 20240229 separately (month walker starts at the 1st)

# 50r1 — 2026-05-13 → 2026-07-02
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
bash run_backfill_00z.sh --era 50r1 --run 12 --from 2026-05 --to 2026-07
```

⚠️ Era boundaries are **mid-month** at both ends (`20240229`, `20260512/13`). The
month walker cannot express them — handle the boundary months with explicit
`--start-date` / `--end-date` calls to `run_lithops_ecmwf.py`, exactly as
`run_gaps_49r1-20240229_50r1-20260701-20260702.sh` does.

## Step 5 — verify

**Never judge success by exit code** — the driver hangs at interpreter exit and
`timeout` reports **124 on a fully successful wave**. Verify by counting GCS objects:
every date must have **51/51**. Then confirm the date set is contiguous over
`20230118 → 20260702` apart from the 6 known 0p4 holes.

## Step 6 — hand off

Report to the coordinating session: dates written, any `<51` dates, Herbie stats.
**Do not** run the HF mirror or Icechunk ingest — those are serial Phase 3 steps
after all three cycles land.

---

## Estimated wall time

~1,256 dates ÷ 35-wide × ~12 min/date ≈ **7 h at current capacity**; ~85 min only if
concurrency reaches ~177 (see [`../README.md`](../README.md) §2). 12z is the long pole
of the three cycles.
