# ECMWF GIK Parquet Backfill — 2026-04-08

## Background

The GIK (Grib-Index-Kerchunk) pipeline produces lightweight parquet reference
files for ECMWF IFS ensemble forecasts. Each parquet contains
`[url, byte_offset, byte_length]` triplets pointing into remote GRIB files on
`s3://ecmwf-forecasts/`. There are 51 parquets per date per run hour (1 control
+ 50 ensemble members).

Parquets are generated on **GCS** (`gs://gik-ecmwf-aws-tf/run_par_ecmwf/`) via
Lithops Cloud Run workers, then mirrored to the public **HuggingFace** dataset
(`E4DRR/gik-ecmwf-par`).

## What Was Done

### 1. Gap Analysis

Compared GCS bucket contents against the live HuggingFace repo tree for all 4
run hours (00z, 06z, 12z, 18z), covering the expected date range 2024-03-01 to
2026-04-07.

**Before (2026-04-08 morning):**

| Run | GCS dates | HF dates | Missing on HF |
|-----|-----------|----------|---------------|
| 00z | 737       | 662      | 75            |
| 06z | 737       | 642      | 95            |
| 12z | 736       | 736      | 0             |
| 18z | 736       | 328      | 408           |

Additionally, 31 dates (2026-03-08 to 2026-04-07) were missing from **both**
GCS and HuggingFace for all 4 run hours.

### 2. Lithops Cloud Run — Generate Missing Parquets on GCS

Generated parquets for the 31 dates missing from GCS (2026-03-08 to 2026-04-07)
across all 4 run hours using the Lithops Cloud Run pipeline.

**Script:** `lithops-cr-gik-ecmwf/run_lithops_ecmwf.py`

**Commands run:**

```bash
cd lithops-cr-gik-ecmwf/

uv run run_lithops_ecmwf.py --start-date 20260308 --end-date 20260407 --run 00 --max-workers 35 --yes
# 31/31 successful in 19.9 min (14 first attempt, 17 retry)

uv run run_lithops_ecmwf.py --start-date 20260308 --end-date 20260407 --run 06 --max-workers 35 --yes
# 31/31 successful in 16.8 min (27 first attempt, 4 retry)

uv run run_lithops_ecmwf.py --start-date 20260308 --end-date 20260407 --run 12 --max-workers 35 --yes
# 31/31 successful in 18.7 min (29 first attempt, 2 retry)

uv run run_lithops_ecmwf.py --start-date 20260308 --end-date 20260407 --run 18 --max-workers 35 --yes
# 31/31 successful in 17.3 min (30 first attempt, 1 retry)
```

**Credentials:** `service_account/ecmwf-lithops-deployer-key.json`
(SA: `ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com`)

**Config:** `lithops_config.yaml` — Cloud Run in `europe-west3`, 2 GB / 2 vCPU,
GCS bucket `gik-ecmwf-aws-tf`.

**Fix applied:** Pinned `lithops==3.6.3` in the script's PEP 723 metadata to
match the Cloud Run runtime image version (was `3.6.4`, caused version mismatch
error). Committed as `155afbc`.

### 3. GCS to HuggingFace Upload

Uploaded all parquets present on GCS but missing from HuggingFace across all
run hours.

**Script:** `ecmwf/upload_parquets_to_hf.py`

**Commands run (in parallel where possible):**

```bash
cd ecmwf/

# Historical 00z gaps (scattered across 2024-2025)
uv run upload_parquets_to_hf.py --sync --year 2024 --month 07 --run 00   # 969 parquets, 12.0 min
uv run upload_parquets_to_hf.py --sync --year 2024 --month 12 --run 00   # 816 parquets, 9.9 min
uv run upload_parquets_to_hf.py --sync --year 2025 --month 04 --run 00   # 408 parquets, 5.1 min
uv run upload_parquets_to_hf.py --sync --year 2025 --month 05 --run 00   # 1173 parquets, 15.2 min
uv run upload_parquets_to_hf.py --sync --year 2025 --month 06 --run 00   # 51 parquets, 0.8 min
uv run upload_parquets_to_hf.py --sync --year 2025 --month 10 --run 00   # 153 parquets, 2.4 min

# Remaining 00z stragglers
uv run upload_parquets_to_hf.py --sync --run 00                          # 204 parquets, 4.2 min

# Full 06z sync
uv run upload_parquets_to_hf.py --sync --run 06                          # 3060 parquets, 51.9 min

# Full 18z sync (largest gap: Feb 2025 onwards)
uv run upload_parquets_to_hf.py --sync --run 18                          # 17901 parquets, 209.5 min

# New Lithops-generated dates (all runs)
uv run upload_parquets_to_hf.py --sync --year 2026 --month 03            # 4845 parquets, 49.1 min
uv run upload_parquets_to_hf.py --sync --year 2026 --month 04            # 1428 parquets, 13.5 min

# Re-sync to catch 2 HF server errors from Mar upload
uv run upload_parquets_to_hf.py --sync --year 2026 --month 03            # 51 parquets, 0.7 min
```

**Credentials:** `coiled-data.json`
(SA: `coiled-data@sewaa-416306.iam.gserviceaccount.com`) for GCS read,
`HF_TOKEN` environment variable for HuggingFace write.

### 4. Catalog Rebuild

Rebuilt the `catalog.parquet` index on HuggingFace to reflect all new files.

```bash
uv run upload_parquets_to_hf.py --catalog
# 156,570 entries, 1.93 MB
```

## Result

**After (2026-04-08 evening):**

| Run | GCS dates | HF dates | Missing |
|-----|-----------|----------|---------|
| 00z | 768       | 768      | 0       |
| 06z | 768       | 768      | 0       |
| 12z | 768       | 768      | 0       |
| 18z | 768       | 768      | 0       |

| Metric                | Before    | After     |
|-----------------------|-----------|-----------|
| HF catalog entries    | 144,228   | 156,570   |
| HF date coverage      | to Feb 18 | to Apr 7  |
| GCS date coverage     | to Mar 7  | to Apr 7  |
| Run hours fully synced | 1 (12z)  | 4 (all)   |

## Architecture Reference

```
ECMWF S3 (public)                    Lithops Cloud Run (e4drr-crafd)
  s3://ecmwf-forecasts/        ←───  run_lithops_ecmwf.py
  .grib2 + .index files               reads .index, builds parquets
                                       writes to GCS ↓

                                     GCS (sewaa-416306)
                                       gs://gik-ecmwf-aws-tf/run_par_ecmwf/
                                       51 parquets per date per run
                                              │
                                              ↓
                                     upload_parquets_to_hf.py --sync
                                              │
                                              ↓
                                     HuggingFace (public)
                                       E4DRR/gik-ecmwf-par
                                       catalog.parquet (index)
```

## Git Commits

- `155afbc` — Fix HF parquet filename and pin lithops version for Cloud Run
- `92f094c` — Add source.coop Icechunk pipeline for ECMWF EA total precipitation
