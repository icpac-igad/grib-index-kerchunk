# 49r1 era — build state, GCS coverage, and backfill plan

> **UPDATE 2026-07-03 — DONE.** The single missing edge date `20240229` was
> filled (`:49r1`, enfo, prefix `v20260623_run_par_ecmwf`; 51/51 members). The
> 49r1 window `20240229 → 20260512` is now fully covered at 00z (804 dates, no
> gaps). Off-00z (06/12/18z) remains optional/absent as noted below.

**Date:** 2026-07-02
**Era:** `49r1` (0.25°, control bundled in `enfo` as `number=0`; 9 pl levels
until 2025-01-14 00z, 13 pl levels from 2025-01-14 06z — the 13-level template
is a safe superset that covers both)
**Verification:** GCS `gik-ecmwf-aws-tf` enumerated with SA
`ecmwf-lithops-deployer@e4drr-crafd`, and S3 cutover probed on `ecmwf-forecasts`,
both 2026-07-02.

---

## Build state — ✅ deployed

`:49r1` Cloud Run runtime deployed (both lithops 3.6.3 / 3.6.4).

| Knob | Value |
|---|---|
| Image tag | `:49r1` |
| Template | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` (13-level, per-level-keys fixed) |
| `ECMWF_REFERENCE_DATE` | `20250515` (a 13-level date) |
| `ECMWF_RESOLUTION` | `0p25` (S3 path `ifs/0p25/`) |
| `ECMWF_CONTROL_STREAM` | `enfo` |
| Grid | 721 × 1440 |

## Era boundaries — S3-verified (corrects the 2026-07-01 session note)

The 2026-07-01 session log recorded a user correction that "49r1 ends 20260612".
**That is wrong.** Parsing the enfo `.index` member set on the public bucket:

| Date (00z) | distinct `number` in enfo | pl levels | Era |
|---|---|---|---|
| 20260512 | **51** (control bundled) | 13 | **49r1** |
| 20260513 | **50** (control moved to `oper/fc`) | 14 | **50r1** |

So the 00z cutover is **20260512 → 20260513**, matching CLAUDE.md
("50r1 … 2026-05-12 06z → present") and `2026-06-03-per-era-deploy-prep.md`.
The `20260612` figure was a typo for `20260512`. Consequence: the 50r1 rebake
that started at **20260513 was correct** — no 20260513→20260612 re-run under
`:49r1` is needed (see the 50r1 doc).

Full 49r1 window: **2024-02-29 → 2026-05-12** (0p25).

## Coverage — near-complete at 00z, one edge date missing

Fixed catalog `v20260623_run_par_ecmwf` (00z, per-level-keys fixed) over the
49r1 window:

- **20240301 → 20260512: fully covered, continuous, 51/51 members every date, zero interior gaps.**
- **Missing: `20240229`** — Feb 29 2024 is the *first* 0p25 date (the 0p4→0p25
  cutover), and both catalogs start at 20240301, so it was never processed.
- **06z / 12z / 18z: absent** from the fixed catalog (00z-only). They exist only
  in the legacy pre-fix catalog `run_par_ecmwf` (collapsed pressure-level keys,
  superseded — do not consume).

## Backfill plan (routine)

**1. Fill the single missing edge date** (write into the fixed catalog):

```bash
cd devops/lithops_cr_ecmwf_gik
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
# 49r1 env from --era; single date via run_lithops directly:
ERA=49r1 ; export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
uv run run_lithops_ecmwf.py --start-date 20240229 --end-date 20240229 --run 00 --max-workers 4 --yes
```

(20240229 is 9-level; the 13-level template is a superset, so the 4 extra levels
100/150/400/600 hPa are simply written empty — same behaviour documented for all
pre-2025-01-14 dates.)

**2. (optional) Off-00z runs** — rebuild 06/12/18z into the fixed catalog if a
consumer needs them:

```bash
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
bash run_backfill_06z.sh --era 49r1 --from 2024-03 --to 2026-05
bash run_backfill_12z.sh --era 49r1 --from 2024-03 --to 2026-05
bash run_backfill_18z.sh --era 49r1 --from 2024-03 --to 2026-05
```

(~671 dates × 3 runs; only launch if the off-00z streams are actually consumed.)

**Not run here** — Cloud Run invocations are paid/outward-facing; launch on the
run host.
