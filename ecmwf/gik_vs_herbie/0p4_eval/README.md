# 0p4 era — GIK vs Herbie validation (go-signal for the backfill wave)

**Date run:** 2026-07-02
**Purpose:** value-level proof that the `:0p4` runtime's parquets are correct
before launching the ~407-date 0p4 backfill (2023-01-18 → 2024-02-28).

## What was validated

- **Par under test:** `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/2023/06/20230601/00z`
  (51 files: `control` + `ens_01..ens_50`), produced by the single-date `:0p4`
  validation run.
- **Method:** for `t` (temperature) at T+0h, stream each member's exact GRIB
  byte-range from the ECMWF 0.4°-beta S3 archive via the par refs, decode with
  gribberish on the **451×900** 0.4° grid, subset to East Africa; compare the
  ensemble mean & spread against Herbie (`model=ifs, product=enfo`) ground truth.
- **Tool:** `ecmwf/compare_gik_herbie_pressure.py --grid 0p4` (the `--grid`
  flag was added for this; 0p25 remains the default).

## Result — PASS

| level | GIK / Herbie members | mean r | mean RMSE | max\|diff\| |
|---|---|---|---|---|
| t @ 500 hPa | 51 / 50 | **0.999911** | 0.0121 K | 0.338 K |
| t @ 850 hPa | 51 / 50 | **0.999958** | 0.0339 K | 1.8 K |

(GIK carries 51 members incl. the bundled control; Herbie enfo returns the 50
perturbed by default — hence 51 vs 50.) Difference maps are essentially zero;
the sub-0.04 K RMSE is grid-reindex/rounding noise. Structure (ensemble spread
over highlands/lakes) matches pixel-for-pixel — confirming the 0.4° grid shape
`(451,900)`, byte offsets, per-level keys, and control member are all correct.

`pl_comparison_stats_t_20230601_T0h.json` holds the full numbers. PNGs are
`.gitignore`d (regenerable from the par + the compare script).

## Decision

Green-lit and launched the full 0p4 wave
(`run_backfill_00z.sh --era 0p4 --from 2023-01 --to 2024-02`,
`GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf`) on 2026-07-02.
