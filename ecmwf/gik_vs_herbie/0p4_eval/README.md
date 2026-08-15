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

**Re-measured 2026-08-15** after fixing a defect in the comparison script (see
below). The original figures on this row were 0.999911 / 0.0121 K / 0.338 K at
500 hPa and 0.999958 / 0.0339 K / 1.8 K at 850 hPa; they measured the script,
not the pars.

| level | GIK / Herbie members | mean r | mean RMSE | max\|diff\| |
|---|---|---|---|---|
| t @ 500 hPa | 51 / 50 | **0.99999994** | 9.30e-05 K | 2.14e-04 K |
| t @ 850 hPa | 51 / 50 | **1.0000000** | 1.79e-04 K | 4.27e-04 K |

(GIK carries 51 members incl. the bundled control; Herbie enfo returns the 50
perturbed by default — hence 51 vs 50.) The residual is float32 packing noise:
compared **per member**, GIK and Herbie are bit-identical. Structure (ensemble
spread over highlands/lakes) matches pixel-for-pixel — confirming the 0.4° grid
shape `(451,900)`, byte offsets, per-level keys, and control member are all
correct.

`pl_comparison_stats_t_20230601_T0h.json` holds the full numbers. PNGs are
`.gitignore`d (regenerable from the par + the compare script).

### Why the original numbers were wrong

They were not rounding noise, as this file previously said — they were one
duplicated row of data. `compare_gik_herbie_pressure.py` used to do

```python
da = da.sel(latitude=slice(LAT_MAX, LAT_MIN), ...)      #  25 .. -14
da = da.reindex(latitude=ICPAC_LATS, ..., method="nearest")
```

eccodes reports the 0.4° axis edge as `-14.000000000000057`, so `slice(25, -14)`
returns 97 rows instead of 98, and the bare `nearest` reindex then back-fills
the missing edge row from `-13.6`. One wrong row at the domain boundary, worth
up to 1.8 K in a temperature gradient. The fix selects the target labels
directly with `tolerance=0.01`, so an absent label raises instead of silently
duplicating its neighbour.

Only 0p4 is affected — 0p25 region edges land on exactly representable floats,
which is why the 49r1/50r1 evaluations always came out at ~1e-4. The same
correction applies to the two 0p4 rows in `../random_3era_eval/` (re-measured
in the same pass; both now ~1e-4).

The failure mode is the one from `HANDOVER_LONGITUDE_FIX.md`: a lookup that
cannot fail loudly turns a missing coordinate into a plausible number. It was
visible here for a year, explained away in this file as "grid-reindex noise",
and only found when a **per-member** comparison — which admits no tolerance —
was run against the Icechunk store
(`../icechunk_v4_eval/README.md`).

## Decision

Green-lit and launched the full 0p4 wave
(`run_backfill_00z.sh --era 0p4 --from 2023-01 --to 2024-02`,
`GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf`) on 2026-07-02.
