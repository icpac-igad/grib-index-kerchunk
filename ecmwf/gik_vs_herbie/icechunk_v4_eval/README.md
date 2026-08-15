# Icechunk v4 (00z) vs Herbie

**Run 2026-08-15** against `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4`, the
store rebuilt after the longitude fix. 00z only. Reproduce with
`ecmwf/run_icechunk_v4_herbie_eval.sh`.

## Why this is not the same as the par-vs-Herbie study

`GIK_vs_Herbie_Evaluation.md` validates the pipeline **input** — the par files.
The pars were correct throughout the longitude incident; it was the *conversion*
that mislabelled the longitude axis, and every store built before the fix
returned the eastern Pacific for an East Africa subset. So this run reads the
store the way a consumer does — `xr.open_zarr` then `.sel(latitude=…,
longitude=…)` **by label, never by integer index** — and holds it against the
same GRIB messages fetched independently by Herbie.

It also compares **per member**, which the par study did not. Ensemble mean and
spread are invariant under a permutation of the member axis, so a store that
filed member 7's field under member 12 would pass the old check and fail this
one. Store `number=n` vs Herbie `number=n`, element-wise, same source bytes:
the only permitted difference is float32 packing noise.

## Results — 22 field comparisons, all bit-exact

| era | date | step | var | lev | n | per-mem max\|diff\| | ens r | ens RMSE |
|---|---|---|---|---|---|---|---|---|
| 0p4 | 20230318 | 0 | t | 500 | 50 | **0** | 1.0000000 | 9.18e-05 |
| 0p4 | 20230318 | 0 | t | 850 | 50 | **0** | 1.0000000 | 1.80e-04 |
| 0p4 | 20231112 | 0 | t | 500 | 50 | **0** | 1.0000000 | 9.56e-05 |
| 0p4 | 20231112 | 0 | t | 850 | 50 | **0** | 1.0000000 | 1.82e-04 |
| 0p4 | 20231112 | 48 | u | 700 | 50 | **0** | 0.9999899 | 0.0163 |
| 0p4 | 20231112 | 48 | t2m | – | 50 | **0** | 0.9999974 | 0.00906 |
| 0p4 | 20231112 | 48 | tp | – | 50 | **0** | 0.9999670 | 7.29e-05 |
| 49r1 | 20240327 | 0 | t | 500 | 50 | **0** | 1.0000000 | 8.87e-05 |
| 49r1 | 20240327 | 0 | t | 850 | 50 | **0** | 1.0000000 | 1.86e-04 |
| 49r1 | 20251125 | 0 | t | 500 | 50 | **0** | 1.0000000 | 9.31e-05 |
| 49r1 | 20251125 | 0 | t | 850 | 50 | **0** | 1.0000000 | 9.59e-05 |
| 49r1 | 20251125 | 48 | u | 700 | 50 | **0** | 0.9999981 | 0.0113 |
| 49r1 | 20251125 | 48 | t2m | – | 50 | **0** | 0.9999990 | 0.00655 |
| 49r1 | 20251125 | 48 | tp | – | 50 | **0** | 0.9999538 | 2.60e-05 |
| 50r1 | 20260621 | 0 | t | 500 | 50 | **0** | 0.9999985 | 0.00281 |
| 50r1 | 20260621 | 0 | t | 850 | 50 | **0** | 0.9999998 | 0.00317 |
| 50r1 | 20260701 | 0 | t | 500 | 50 | **0** | 0.9999989 | 0.00276 |
| 50r1 | 20260701 | 0 | t | 850 | 50 | **0** | 0.9999998 | 0.00343 |
| 50r1 | 20260701 | 48 | u | 700 | 50 | **0** | 0.9999972 | 0.0132 |
| 50r1 | 20260701 | 48 | t2m | – | 50 | **0** | 0.9999990 | 0.00702 |
| 50r1 | 20260701 | 48 | tp | – | 50 | **0** | 0.9999518 | 3.98e-05 |
| 49r1 | 20240327 | 36 | t | 500 | 49 | **0** | 0.9999913 | 0.00918 |

`n` is the number of members compared; `max|diff|` is the worst single grid point
across all of them. **Zero** — not "small" — for every case, on all three eras,
at the analysis step and at T+48h, for pressure levels, instantaneous surface
fields and accumulated `tp`.

The ensemble columns are 51 store members against Herbie's 50, so they carry the
control-member offset the original study describes; they are reported only for
continuity with that table. The per-member column is the actual test.

### The last row is deliberate

`verify_store_completeness.py` reports `20240327 ens_24` missing at step 36. That
run compares 49 members instead of 50 and reports exactly one all-NaN member —
number 24 — so it exits FAIL by design. Two independent tools, one reading the
manifest and one reading the data, name the same missing slice.

## A correction to the published 0p4 numbers

The par study reports for 0p4 `t@500 T+0`: mean r 0.99997, RMSE 0.0176,
max_abs_diff **0.569 K**, and similar at 850 hPa. Those figures are an artifact
of the comparison script, not a property of the data.

`compare_gik_herbie_pressure.py` does

```python
da = da.sel(latitude=slice(LAT_MAX, LAT_MIN), ...)      # 25 .. -14
da = da.reindex(latitude=ICPAC_LATS, ..., method="nearest")
```

eccodes reports the 0.4° axis edge as `-14.000000000000057`, so `slice(25, -14)`
drops that row — 97 rows, not 98 — and the bare `nearest` reindex then
back-fills it from `-13.6`. One duplicated row at the domain edge, reading as a
real 0.56 K disagreement. The first version of this evaluation reproduced that
number to four digits before the cause was found; selecting the store's labels
directly with `tolerance=0.01` drops it to 9.18e-05.

0p25 eras are unaffected — their region edges land on exactly representable
floats. Only 0p4 dates are contaminated, in any script using that pattern.

The general shape is the same as the longitude bug: `method="nearest"` with no
tolerance cannot fail loudly, so a missing coordinate becomes a plausible number
instead of an error.

## What was ruled out along the way

Each of these was checked directly rather than assumed, while chasing the 0p4
discrepancy above:

- **decoder disagreement** — gribberish vs eccodes on the same message bytes:
  max\|diff\| 1.5e-05 K over the global field;
- **member permutation** — Herbie's `number=n` matched against all 51 store
  members; the same-numbered member is the best match;
- **wrong byte range** — Herbie's inventory `start_byte`/`end_byte` for the
  probe message are identical to the ECMWF `.index` `_offset`/`_length`;
- **a shifted grid** — the store's field matches a direct S3 byte-range +
  gribberish decode of the `.index`-named message to 1.5e-05, and rolling the
  global field by one cell in any direction raises the difference to ≥ 4 K.

## Files

- `icechunk_v4_{era}_{var}{level}_{date}_T{step}h.png` — Icechunk | Herbie |
  difference, for ensemble mean (top) and spread (bottom)
- `icechunk_v4_{era}_{var}_{date}_T{step}h.json` — full stats, including
  per-member rows
- `sweep.log` — the whole run
