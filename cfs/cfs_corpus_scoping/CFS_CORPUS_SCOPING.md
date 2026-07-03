# CFS AWS S3 Corpus Scoping Study

**Bucket:** `s3://noaa-cfs-pds/` (NOAA Open Data, anonymous access)
**Prepared:** 2026-07-03 · direct S3 probes (anonymous `s3fs`)
**Purpose:** scope the CFS dataset layout, ensemble/monthly structure, forecast
length, and corpus stability **before** running the full-corpus mapping-parquet
scan that Stage-1 template creation (GEFS-style) needs. This is the "zero step":
the template scan's task grid and the runtime's timestep axis both depend on the
facts below.

> Sample `.idx` files referenced here are saved under
> [`idx_samples/`](idx_samples/) (4 × `flxf` 6-hourly + 4 × `time_grib` daily,
> for the archive start date and 2026-07-02).

---

## 0. TL;DR

- **Archive extent:** `cfs.` = **2018-10-31 → 2026-07-03** (2,803 dates, ~daily
  continuous). `cdas.` = **2023-04-22 → present** (1,169 dates).
- **Two products in the bucket:** `cfs.{date}` = the **CFSv2 forecast** (what the
  GIK method uses); `cdas.{date}` = the **CDAS real-time analysis** (initial
  conditions / obs assimilation) — *not* used by the seasonal-forecast pipeline.
- **Ensemble = CFSv2 staggered lagged ensemble.** Per day: 4 cycles
  (00/06/12/18z); per cycle: 4 members (`6hrly_grib_01..04`). **Member 01 runs
  long (~7 months); members 02–04 run short (~4 months).**
- **6-hourly cadence**, uniform. **flxf `.idx` coverage is 100%** on every date
  checked (the target files always have an index).
- **Forecast length is date-dependent** (fixed *calendar* horizon, not fixed
  hour count): member 01 = **849–857 steps (212–214 days)** depending on init
  month. The archive's **first day (2018-10-31) is a short anomaly** (729
  steps / 182 d); from 2018-11-01 onward it is the normal length.
- **Schema is stable 2018 → 2026** — flxf `.idx` variable/level sets are
  *identical* across the whole archive; directory layout unchanged.
- **Three grib products per member:** `6hrly_grib` (6-hourly instantaneous),
  `time_grib` (per-variable long series, incl. `tmax`/`tmin`/`prate` directly),
  `monthly_grib` (monthly means). **The seasonal target variables exist directly
  in `time_grib`** — see §7 (a real design decision for the template).

---

## 1. Archive extent & continuity

| Prefix | Range | Count | Cadence |
|---|---|---|---|
| `cfs.`  | `cfs.20181031` → `cfs.20260703` | 2,803 | ~daily, continuous |
| `cdas.` | `cdas.20230422` → present | 1,169 | daily |
| (other) | `index.html` | 1 | — |

`2,803` dates over 2018-10-31 → 2026-07-03 ≈ one prefix per calendar day (no
material gaps). Today's date (`cfs.20260703`) is already present.

## 2. `cfs` vs `cdas`

- **`cfs.{date}`** — CFSv2 **forecast** output. Structured by cycle/member/product
  (§3). This is the corpus the GIK streaming method targets.
- **`cdas.{date}`** — **CDAS** (Climate Data Assimilation System) real-time
  analysis: flat `cdas1.t{HH}z.*` layout mixing analysis GRIB2
  (`ipvgrb{anl,fNN,hNN,lNN}.grib2` + `.idx`) and BUFR observation dumps. It is
  the analysis/IC stream (CFSR successor), present only from 2023-04-22. **Not
  consumed by the seasonal-forecast pipeline** and out of scope for the template.

## 3. Directory structure (identical for the earliest and latest dates)

```
cfs.{YYYYMMDD}/
  {00,06,12,18}/                       # 4 forecast cycles per day
    6hrly_grib_{01,02,03,04}/          # 6-hourly instantaneous fields  <-- GIK uses this
    time_grib_{01,02,03,04}/           # per-variable long time series
    monthly_grib_{01,02,03,04}/        # monthly-mean fields
```

- **4 cycles × 4 members × 3 products = 12 subdirs per cycle**, on both
  `cfs.20181031` and `cfs.20260702` — **no reorganization over the archive.**
- `6hrly_grib_NN` file kinds (GRIB2 + `.idx`):
  `flxf` (surface fluxes — **GIK target**), `pgbf`/`pgba` (pressure levels),
  `ocnf` (ocean), `ipvf`/`ipva` (isentropic PV), `spla` (spectral analysis).
- **`.idx` coverage:** on `cfs.20260702` every grb2 has an `.idx` (3430/3430).
  On older dates some *non-flxf* kinds lack an `.idx` (e.g. `cfs.20181031`:
  2918 grb2 vs 2500 idx) — but **`flxf` is always 100% indexed** (see §5).

## 4. Ensemble & monthly consideration

CFSv2 is a **lagged (time-staggered) ensemble**:

- **Within a day:** 4 cycles (00/06/12/18z). Within a cycle: 4 members
  (`6hrly_grib_01..04`).
- **Staggered forecast lengths** (verified §5): **member 01 is the long run
  (~7 months); members 02–04 are short runs (~4 months).** So only member 01
  covers a full two-season horizon.
- **Monthly seasonal ensemble:** to build an ensemble forecast *for target month
  M*, aggregate the forecasts initialised across the **preceding month's** days ×
  cycles (× members) whose range covers M. `cfs/CLAUDE.md` frames this as
  "31 init_dates × 4 runs = up to 124 members per target month" (using member 01
  per cycle). Members 02–04 can extend the ensemble size for the nearer part of
  the range where their shorter forecasts still reach M.

## 5. Forecast length & timesteps (6-hourly `flxf`, cycle 00z)

| Date | m01 | m02 | m03 | m04 |
|---|---|---|---|---|
| `20181031` *(archive start, anomaly)* | 729 st / 4368 h / **182 d** | 373 / 2232 / 93 d | 373 / 93 d | 373 / 93 d |
| `20260702` | 857 st / 5136 h / **214 d** | 489 / 2928 / 122 d | 489 / 122 d | 489 / 122 d |

- **Interval:** uniform **6 h** (only inter-step delta observed = 6).
- **`.idx` coverage for flxf:** 100% on every member/date checked.
- **Date-dependence (fixed calendar horizon):** member 01 wobbles **849–857
  steps (212–214 d)** by init month (forecast runs to a target ~7 months out, so
  the exact step count follows month lengths). Members 02–04 wobble **481–493
  steps (120–123 d)**. Representative probes:

  | Init | m01 | m02 |
  |---|---|---|
  | 2018-10-31 | 729 / 182 d | 373 / 93 d |
  | 2018-11-01 | 849 / 212 d | — |
  | 2019-03-01 | 857 / 214 d | — |
  | 2019-04-01 … 2025-04-01 | 849–857 / 212–214 d | 481–493 / 120–123 d |
  | 2026-07-02 | 857 / 214 d | 489 / 122 d |

- **Transition:** only **2018-10-31** is short (182 d); **from 2018-11-01 the
  full length is in place**. Treat the first archive day as a special case.
- **"860 timesteps / ~6 months"** in prior notes = **member 01 of the 6hrly_grib
  product** (857 ≈ 860; 214 d ≈ 7 months). Confirmed as the intended target.

## 6. Corpus stability over time

- **flxf `.idx` schema identical 2018 → 2026** — the `VAR:level` set of
  `flxf…f006` on `cfs.20181031` is byte-for-byte the same set as on
  `cfs.20260702`. Same record counts (f000 = 101, f006 = 103), same sizes.
- **Directory layout unchanged** across the archive (§3).
- **Only change:** forecast *length* extended after the 2018-10-31 start (§5).
- **Implication:** a single reference date's template schema is valid for the
  whole archive; only the **timestep count is date-dependent** and must be
  derived per (date, cycle, member) rather than hard-coded.

## 7. The three grib products (which corpus should the template target?)

| Product | Files/member (00z, 2026-07-02) | Structure | Horizon | Contains |
|---|---|---|---|---|
| `6hrly_grib_01` | ~857 `flxf` (+ pgbf/ocnf/…) | one file **per timestep**, all vars | 214 d | instantaneous 6-hourly fields — **current GIK target** |
| `time_grib_01` | 91 (one **per variable**) | one file **per variable**, all steps inside | **303 d** (7272 h, 6-hourly) | `prate, tmax, tmin, tmp2m, dswsfc, uswsfc, dlwsfc, ulwsfc, wnd10m`, … |
| `monthly_grib_01` | 250 | monthly means (`…avrg…{HH}Z`) | seasonal | monthly-mean diurnal fields |

**Key finding:** the seasonal target variables (incl. `tmax`/`tmin`, which
`cfs/CLAUDE.md` assumed had to be *computed* from 6-hourly `TMP`) exist
**directly** in `time_grib` as per-variable series — and `time_grib` runs
**longer (303 d vs 214 d)**. Its `.idx` files are 6-hourly internally despite the
`.daily.grb2` name (e.g. `tmax…daily`: 1,212 records at 6/12/…/7272 h).

- **6hrly_grib** = many files (~857) × all-variables-per-file → the mapping-scan
  task grid is large but matches the existing pipeline/template code.
- **time_grib** = few files (~91) × one-variable-per-file, longer horizon,
  target variables ready-made → far smaller scan, arguably better fit for
  seasonal streaming, but the current pipeline/template code is written for
  6hrly `flxf` and would need adapting.

This is a genuine **decision point** for the template build (see §10).

## 8. `.idx` format (why the kerchunk patch exists)

CFS `.idx` is NCEP colon-separated text: `record:offset:d=YYYYMMDDHH:VAR:level:step:`

```
1:0:d=2026070200:UFLX:surface:anl:
5:195482:d=2026070200:TMP:surface:anl:
36:36.1:1877340:d=2026070200:UGRD:10 m above ground:anl:
37:36.2:1877340:d=2026070200:VGRD:10 m above ground:anl:
```

- **Analysis vs forecast step:** `f000` records are `anl`; `f006+` are
  `N hour fcst`. The **+2 records at f006** (101 → 103) are averaged/accumulated
  fields (precip/radiation, e.g. `CPRAT`, `CSDLF`) that need a non-zero interval
  — this is why the zarr skeleton must scan **f000 + f006** (not f000 alone).
- **Fractional record ids** (`36.1`/`36.2`, `79.1`/`79.2`): UGRD+VGRD packed into
  a single GRIB message (shared byte offset). kerchunk's
  `parse_grib_idx(...).astype(int)` chokes on these → the `_safe_astype` monkey
  patch in `run_cfs_template_creation.py` / `cfs_coiled_preprocessing.py`.
  **4 fractional records** in every flxf idx (10 m wind + 1-hybrid-level wind),
  stable 2018 → 2026. (VGRD sharing UGRD's offset is the known wind-decode
  caveat.)

## 9. Grid

CFS native flux grid is Gaussian **T126 ≈ 384 lon × 190 lat** (confirmed from
`scan_grib`: `latitude[190]`, `longitude[384]`). One GRIB message = one
`[1, 190, 384]` field.

## 10. Implications for the Stage-1 template build (the "zero step")

1. **Reference date:** any date **≥ 2018-11-01** has the stable full schema; the
   schema is date-independent, so one reference date suffices. Avoid
   `2018-10-31` (short anomaly).
2. **Timestep count is per-(date,cycle,member), not constant.** Derive it from
   the `.idx` listing (or the calendar horizon), don't hard-code 857/5160. The
   runtime axis must be built per forecast, not baked into the template.
3. **Member scope:** member 01 = full seasonal length; 02–04 short. Decide
   whether the ensemble uses **m01 only** (simplest, full horizon) or all four
   (larger ensemble, mixed lengths → ragged time axis).
4. **Full-corpus mapping-parquet scan (Coiled):** the GEFS-model template needs
   `build_idx_grib_mapping` over every (member, hour). For **m01 / 6hrly flxf**
   that is ~857 tasks per (date, cycle); all four cycles ≈ 3.4k; add m02–04 for a
   fuller ensemble. This is the scan `cfs_coiled_preprocessing.py` performs — its
   task grid should use the **per-member step counts from §5**, not a flat 5160 h.
5. **Product decision (§7):** confirm whether the seasonal deliverable is built
   from **6hrly_grib flxf** (current code path) or **time_grib** (per-variable,
   longer, target vars ready-made). This changes the scan target, the template
   layout, and the streaming stage.

## 11a. Seasonal-forecast applicability (SPI / SPEI / PET) — product decision

**Recommendation: build the template on `6hrly_grib` `flxf` (member 01).** The
`flxf` file is *self-sufficient* for the decision-relevant drought indices — every
ingredient is present directly (verified in `flxf…f006` idx, 103 records):

| Product | Needs | In `flxf`? |
|---|---|---|
| **SPI** | precipitation | `PRATE:surface` ✅ |
| **PET** (ready-made) | potential evap | `PEVPR:surface` ✅ (no computation) |
| **PET** (Hargreaves) | tmax, tmin, tmean | `TMAX`/`TMIN`/`TMP:2 m` ✅ |
| **PET** (Penman-Monteith) | net radiation, T, wind, humidity, pressure | `DSWRF`/`USWRF`/`DLWRF`/`ULWRF`, `TMP:2 m`, `UGRD`/`VGRD:10 m`, `SPFH:2 m`, `PRES:surface` ✅ |
| **SPEI** | P − PET | all of the above ✅ |

So `flxf` covers SPI, PET and SPEI with one file kind — **`TMAX`/`TMIN` are
present directly** (correcting the `cfs/CLAUDE.md` assumption that they must be
derived from 6-hourly `TMP`). `time_grib` remains useful only if a **longer
horizon** (303 d vs 214 d) or the pre-aggregated daily series is wanted; for
seasonal (3-month) targets `flxf`'s 214 d already spans a full season + lead.

**Ensemble via the lagged/burst design (the "100+ members" is real).** CFSv2's
ensemble is generated by initialization staggering, so a target month's ensemble
is assembled from many init (date, cycle) forecasts of the **preceding** month(s):

| Ensemble definition | Members for a 30-day init month |
|---|---|
| member 01 × 4 cycles/day × 30 days | **120** (all reach ~7 months) |
| all 4 members × 4 cycles × 30 days | up to **480** (m02–04 reach ~4 months only) |

So member-01-only already gives **>100 members** per month at full seasonal
horizon (`cfs/CLAUDE.md`: "31 init_dates × 4 runs = up to 124"). Caveat: lagged
members have **different lead times** to the target month (not exchangeable like a
single burst) — standard for CFSv2/NMME, but calibration must respect it.

**Why streaming (GIK) matters here.** A 120-member monthly ensemble over `flxf`
is ~120 init-parquets; streaming only the ~10 SPI/SPEI/PET variables via
byte-range reads fetches a few % of the data instead of downloading ~120 × 857 ×
(all-variable GRIB) files. That is what makes continuous, large-ensemble
probabilistic monitoring (E4DRR/SEWAA) tractable.

**Method caveats (must-haves for valid indices):** SPI/SPEI fit distributions to a
**baseline climatology** — use the CFSv2 **reforecast/hindcast** archive (or
bias-correct to an observational reference, e.g. CHIRPS/ERA5) rather than raw
forecasts; and single-model GHA seasonal *rainfall* skill is modest (ENSO-driven,
stronger for OND than MAM) — which is exactly why ICPAC combines **nine** models
(§ external context) and CFS is **one calibrated input**, not a standalone forecast.

## 11. Saved sample `.idx` files ([`idx_samples/`](idx_samples/))

| File | Product | Date | Step | Records |
|---|---|---|---|---|
| `flxf_20181031_00_f000.01.idx` | 6hrly flxf | 2018-10-31 | f000 (anl) | 101 |
| `flxf_20181031_00_f006.01.idx` | 6hrly flxf | 2018-10-31 | f006 | 103 |
| `flxf_20260702_00_f000.01.idx` | 6hrly flxf | 2026-07-02 | f000 (anl) | 101 |
| `flxf_20260702_00_f006.01.idx` | 6hrly flxf | 2026-07-02 | f006 | 103 |
| `timegrib_tmax_20181031_00.daily.idx` | time_grib | 2018-10-31 | full series | 1220 |
| `timegrib_tmax_20260702_00.daily.idx` | time_grib | 2026-07-02 | full series | 1212 |
| `timegrib_prate_20181031_00.daily.idx` | time_grib | 2018-10-31 | full series | 1220 |
| `timegrib_prate_20260702_00.daily.idx` | time_grib | 2026-07-02 | full series | 1212 |

---

## Appendix — probe method

All facts from anonymous `s3fs` listings/reads against `s3://noaa-cfs-pds/`
on 2026-07-03 (scripts in the session scratchpad): top-level prefix inventory;
per-cycle/member `ls` of `6hrly_grib`/`time_grib`/`monthly_grib`; `flxf` step
counts & 6-h interval via forecast-timestamp arithmetic; multi-year length
probes (2018-10 → 2026-07); `.idx` downloads for the start date and 2026-07-02.
No GRIB payloads were downloaded (index/listing bytes only).
