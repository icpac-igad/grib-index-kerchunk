# Published realized EA-SWIO store vs Herbie

**Run 2026-08-21** against the published, anonymously readable store

    https://source.coop/e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3
    s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3

Reproduce with `ecmwf/run_realized_ea_swio_herbie_eval.sh`. No credentials are
used anywhere in this evaluation — reading it the way an outside user would is
part of what is being tested.

## Why this is a different test from the virtual store

`icechunk_v4_eval/` validates the **virtual** store, whose chunks are
`[url, offset, length]` references decoded from ECMWF's GRIB on every read.
This store is **realized**: the values were decoded once, cut to the EA-SWIO
box, and written as real chunks. That is a separate set of failure modes, none
of them covered by the virtual-store check:

- the subset could be taken on the wrong window, or be off by a row;
- flattening `(var, level)` into channel names (`t` @ 850 → `t850`) could
  mis-assign a level — the per-level-keys bug, re-introduced at a later layer;
- dim order is `(time, step, number, lat, lon)` here and
  `(time, number, step, …)` in the virtual store, so a transposed write would
  be silent;
- `step` is stored in **seconds** here and hours there.

Herbie knows about none of this. It fetches the same GRIB messages straight
from ECMWF's open-data mirror and decodes them with eccodes.

## Domain — the extended EA-SWIO box

**15–80°E, 40°S–40°N**, 321 × 261 at 0.25°, taken from the store's own
coordinate arrays and never re-derived. Wider than both the ICPAC box used in
the earlier studies (19–55°E, 14°S–25°N) and the `EA_BOX` in
`verify_grid_geolocation.py` (which stops at 25.25°N).

## Result — 18 comparisons, all bit-exact per member

Per member means store `number=n` against Herbie `number=n`, element-wise, for
n = 1…50. `max|diff| = 0` is the whole field, every member.

| store | date | step | channel | n | per-mem max\|diff\| | ens r | ens RMSE |
|---|---|---|---|---|---|---|---|
| 49r1-mam2024 | 20240315 | T+120h | tp | 50 | **0** | 0.99998053 | 1.01e-04 |
| 49r1-mam2024 | 20240320 | T+0h | u700 | 50 | **0** | 1.00000000 | 1.80e-04 |
| 49r1-mam2024 | 20240327 | T+48h | t850 | 50 | **0** | 0.99999860 | 0.00888 |
| 49r1-mam2024 | 20240402 | T+24h | t2m | 50 | **0** | 0.99999915 | 0.00895 |
| 49r1-mam2025 | 20250315 | T+48h | gh500 | 50 | **0** | 0.99999975 | 0.0424 |
| 49r1-mam2025 | 20250420 | T+240h | tp | 50 | **0** | 0.99996078 | 2.14e-04 |
| 49r1-mam2025 | 20250501 | T+0h | msl | 50 | **0** | 1.00000000 | 0.0243 |
| 49r1-mam2025 | 20250510 | T+72h | q925 | 50 | **0** | 0.99999625 | 1.31e-05 |
| 49r1-mam2026 | 20260310 | T+48h | t850 | 50 | **0** | 0.99999932 | 0.0068 |
| 49r1-mam2026 | 20260401 | T+96h | tcwv | 50 | **0** | 0.99999340 | 0.0498 |
| 49r1-mam2026 | 20260415 | T+24h | v850 | 50 | **0** | 0.99999414 | 0.0133 |
| 49r1-mam2026 | 20260512 | T+0h | u10 | 50 | **0** | 0.99999999 | 8.10e-04 |
| 50r1-mam2026-tail | 20260515 | T+48h | t850 | 50 | **0** | 0.99999921 | 0.0079 |
| 50r1-mam2026-tail | 20260520 | T+120h | tp | 50 | **0** | 0.99998110 | 1.02e-04 |
| 50r1-mam2026-tail | 20260525 | T+72h | vo850 | 50 | **0** | 0.99984748 | 6.90e-07 |
| 50r1-mam2026-tail | 20260531 | T+0h | skt | 50 | **0** | 0.99999992 | 0.00284 |
| **virtual v4** (compensate) | 20240501 | T+0h | u700 | 50 | **0** | 1.00000000 | 1.80e-04 |
| **virtual v4** (compensate) | 20240520 | T+24h | t2m | 50 | **0** | 0.99999890 | 0.00957 |

The ensemble columns compare the store's 51 members against Herbie's 50: `enfo`
does not return the control, so `r < 1` there is the missing control member, not
a defect. `vo850` has the lowest ensemble r and the smallest RMSE at once
because vorticity is ~1e-5 in magnitude — one member out of 51 moves the
correlation while the absolute error stays negligible. Per member it is exact.

Levels, steps and channel kinds were varied deliberately: pressure-level,
instantaneous surface, and accumulated (`tp`); the analysis step and leads out
to T+240h; and `vo850`/`gh500`/`q925`, whose values would be grossly wrong if a
level were mis-assigned during flattening.

## Coverage — `49r1-mam2024` is only one third published

`coverage.json` holds the per-date map, read from the manifest rather than by
sampling data.

| sub-store | dates written | window |
|---|---|---|
| 49r1-mam2024 | **33 / 92** | 2024-03-01 … **2024-04-02** only |
| 49r1-mam2025 | 92 / 92 | complete |
| 49r1-mam2026 | 73 / 73 | complete |
| 50r1-mam2026-tail | 19 / 19 | complete |

The time axis is **preallocated to the whole season**, so an unwritten date is
not missing — it is present on the axis and reads as all-NaN. It raises nothing.
The first version of this sweep picked `20240501` and `20240520`, and both
"compared" against a field of NaN: 0 members matched, and the run then died with
`KeyError: 'corr'` — an error that named the wrong problem entirely.

`compare_realized_herbie.py` now detects this and stops with `RESULT: UNWRITTEN`
before downloading 50 GRIB messages to compare against nothing.

This is the same hazard as the union step axis in `verify_store_completeness.py`
and the same one as the longitude bug: **preallocation makes absence look like
data**, and only an explicit check tells them apart.

## Compensating for the unpublished dates

Rather than drop the two 2024 dates the realized store never wrote, they are
compared against the **virtual** v4 store — which carries every date — over the
identical EA-SWIO box, producing the identical 321 × 261 grid and the same
per-member test. A gap in the published subset does not become a gap in the
validation. Those two rows are marked *compensate* above; their plots are named
`icechunk_v4_*_easwio.png`.

This is what `--lon-min/--lon-max/--lat-min/--lat-max` were added to
`compare_icechunk_herbie.py` for.

## Files

- `realized_{sub}_{channel}_{date}_T{step}h.png` — realized store | Herbie |
  difference, ensemble mean (top) and spread (bottom)
- `icechunk_v4_49r1_*_easwio.png` — the two compensation cases, same layout
- `*.json` — full statistics including a per-member row for every member
- `coverage.json` — which dates each sub-store has actually written
- `sweep.log` — the whole run, including the two failures that found the gap
