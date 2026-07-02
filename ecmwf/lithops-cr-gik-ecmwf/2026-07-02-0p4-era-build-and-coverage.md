# 0p4 era — build state, GCS coverage, and backfill plan

**Date:** 2026-07-02
**Era:** `0p4` (0.4° beta, 9 pressure levels, 51 members with control bundled in `enfo` as `number=0`)
**Verification:** GCS bucket `gik-ecmwf-aws-tf` enumerated with service account
`ecmwf-lithops-deployer@e4drr-crafd` on 2026-07-02; S3 cutover probed against the
public `ecmwf-forecasts` bucket the same day.

---

## Build state — ✅ runtime ready, ❌ no dates run

The `:0p4` Cloud Run runtime was **built and deployed 2026-07-01** (session log
`2026-07-01-49r10p4-gik-build.txt`). Image `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:0p4`,
deployed under **both** lithops 3.6.3 (deploy host) and 3.6.4 (run host).

| Knob | Value |
|---|---|
| Image tag | `:0p4` |
| Template | `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` |
| `ECMWF_REFERENCE_DATE` | `20230601` |
| `ECMWF_RESOLUTION` | `0p4` (S3 path segment `0p4-beta/`) |
| `ECMWF_CONTROL_STREAM` | `enfo` (control bundled as `number=0`) |
| Grid | 451 × 900 |
| pl levels | 9 `[50,200,250,300,500,700,850,925,1000]` |

## Coverage — entire era MISSING

Neither the fixed catalog (`v20260623_run_par_ecmwf`) nor the legacy catalog
(`run_par_ecmwf`) contains a single 0p4-era date. Both catalogs start at
**20240301**, which is already the 49r1 (0p25) era.

**S3-verified era window** (public bucket, HEAD probe 2026-07-02):

- `0p4-beta/` `.index` present through **20240228**; absent from **20240229**.
- `ifs/0p25/` `.index` absent through 20240228; present from **20240229**.

So the 0p4 era is a clean **2023-01-18 → 2024-02-28** window (~407 calendar dates,
00z), none of which is in GCS.

## Backfill plan (routine)

Run the existing per-month driver with `--era 0p4`, **pinning the fixed-catalog
prefix** (the public-reflection `run_lithops_ecmwf.py` still defaults
`GCS_PARQUET_PREFIX=run_par_ecmwf`; the fixed catalog is `v20260623_run_par_ecmwf`):

```bash
# run host, :0p4 runtime must be the deployed image in lithops_config.yaml
cd devops/lithops_cr_ecmwf_gik
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf   # write into the FIXED catalog
bash run_backfill_00z.sh --era 0p4 --from 2023-01 --to 2024-02
```

`run_backfill_00z.sh --era 0p4` exports `ECMWF_REFERENCE_DATE=20230601`,
`ECMWF_RESOLUTION=0p4`, `ECMWF_CONTROL_STREAM=enfo`, and the 0p4-beta template
URL automatically. It processes one calendar month per Cloud Run wave; the
per-date pre-flight in `run_lithops_ecmwf.py` silently skips any date ECMWF
never published at 0.4°.

- **Scope:** 14 months (2023-01 partial from the 18th → 2024-02, ~407 dates).
- **Validate first (cheap):** one date, e.g. `20230601`, assert the parquet has
  the 9-level pl keys `step_NNN/{var}/pl/{hPa}/{member}/0.0.0` and the control
  member is present, before the bulk wave.
- **Cost:** ~407 dates × ~$0.026 ≈ **$11** Cloud Run; ~14 waves × ~8.5 min ≈ 2 h.
- **Not run here** — this is a paid, outward-facing Cloud Run job; launch on the
  run host after the single-date validation passes.

Optional 06z/12z/18z: append `run_backfill_{06,12,18}z.sh --era 0p4` — the fixed
catalog currently holds **00z only** for every era (see the 49r1/50r1 docs).
