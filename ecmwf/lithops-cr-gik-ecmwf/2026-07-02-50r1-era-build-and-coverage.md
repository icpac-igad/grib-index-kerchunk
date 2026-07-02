# 50r1 era — build state, GCS coverage, and backfill plan

**Date:** 2026-07-02
**Era:** `50r1` (0.25°, 14 pl levels, **dual-stream**: 50 perturbed members from
`enfo/ef` + control from `oper/fc`)
**Verification:** GCS `gik-ecmwf-aws-tf` enumerated with SA
`ecmwf-lithops-deployer@e4drr-crafd`, and S3 cutover probed on `ecmwf-forecasts`,
both 2026-07-02.

---

## Build state — ✅ deployed

`:50r1` Cloud Run runtime deployed (both lithops 3.6.3 / 3.6.4).

| Knob | Value |
|---|---|
| Image tag | `:50r1` |
| Template | `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` |
| `ECMWF_REFERENCE_DATE` | `20260513` |
| `ECMWF_RESOLUTION` | `0p25` (S3 path `ifs/0p25/`) |
| `ECMWF_CONTROL_STREAM` | `oper` (control from `oper/fc`; perturbed from `enfo/ef`) |
| Grid | 721 × 1440 |
| pl levels | 14 (`+10` hPa over the 13-level 49r1 set) |

## Era boundary — S3-verified

00z cutover is **20260512 (49r1) → 20260513 (50r1)**: on 20260513 the enfo
member set drops from 51 to 50 (control lifted out to `oper/fc`) and pl levels
rise from 13 to 14. This confirms the **50r1 rebake correctly started at
20260513** — the earlier worry (session log, "49r1 ends 20260612 → re-run
20260513→20260612 under :49r1") was based on a `20260612`/`20260512` typo and is
**void; no re-run is required.**

## Coverage — complete through 20260630, rolling tail missing

Fixed catalog `v20260623_run_par_ecmwf` (00z) over the 50r1 window:

- **20260513 → 20260630: fully covered** (waves A `20260513-31`, B `20260601-24`,
  C `20260625-30`), 51/51 members every date (50 perturbed + oper control), zero gaps.
- **Missing: `20260701`, `20260702`** — published on S3 (HEAD-verified 2026-07-02)
  but not yet baked. This is the rolling daily tail, not a structural gap.
- **06z / 12z / 18z: absent** from the fixed catalog (00z-only).

## Backfill plan (routine)

**1. Catch up the rolling tail** (each new date, into the fixed catalog):

```bash
cd devops/lithops_cr_ecmwf_gik
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
bash run_backfill_00z.sh --era 50r1 --from 2026-07 --to 2026-07
# processes the whole month; per-date pre-flight skips dates not yet published
```

Or a single explicit date:

```bash
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
uv run run_lithops_ecmwf.py --start-date 20260701 --end-date 20260702 --run 00 --max-workers 4 --yes
```

**2. Ongoing daily** — 50r1 is the current era. Schedule a daily `--era 50r1`
run for `today` (00z lands on S3 by ~mid-morning UTC) to keep the catalog current.

**Not run here** — Cloud Run invocations are paid/outward-facing; launch on the
run host.

---

## Consolidated missing-date summary (all three eras, fixed catalog, 00z)

| Era | Window | Covered (00z) | Missing | Fix |
|---|---|---|---|---|
| 0p4 | 2023-01-18 → 2024-02-28 | none | **entire era (~407 dates)** | `run_backfill_00z.sh --era 0p4 --from 2023-01 --to 2024-02` |
| 49r1 | 2024-02-29 → 2026-05-12 | 20240301 → 20260512 | **20240229** (1 date) | single-date `run_lithops_ecmwf.py --era 49r1` |
| 50r1 | 2026-05-13 → present | 20260513 → 20260630 | **20260701–today** (rolling) | `run_backfill_00z.sh --era 50r1 --from 2026-07` + daily |

All three fixes must set `GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf` so they land
in the per-level-keys-fixed catalog and never in the superseded `run_par_ecmwf`.
