# CFS Member-01 Reference Creation — Timing, Commands & Pipeline Internals

**What this documents:** the measured time to build one CFS ensemble-member
parquet (reference creation), the exact commands/scripts used, and the Stage
1→3 internals of `lithops-cr-gik-cfs/run_lithops_cfs.py` (the GEFS-style rewire).
Prepared 2026-07-04 on a 2-core / 7 GB box. Companion: `CFS_ZARR_VS_GIK_COST.md`.

---

## 1. Timing — single full member (member 01, all timesteps)

Command:

```bash
cd cfs/
uv run lithops-cr-gik-cfs/run_lithops_cfs.py --init-date 20260201 --run 00 \
    --sequential --yes \
    --local-template cfs_template/gik-fmrc-v2cfs_fmrc.tar.gz \
    --output-dir cfs_ensemble_202602 --max-forecast-hours 5136
# -> cfs_ensemble_202602/2026020100z.parquet
```

Result (init 20260201 00z, member 01):

| Stage | Time | What ran |
|---|---|---|
| Stage 1 — load skeleton | **0.2 s** | read `rt000.par` from the template tar → 1,705 zarr keys |
| Stage 2 — map timesteps | **~165 s** | 857 tasks: `parse_grib_idx` (fresh S3 `.idx`) + `map_from_index`, 8 threads |
| Stage 3 — build store | **~58 s** | `store_coord_var`/`store_data_var` over 849 steps × 86 groups |
| **Total** | **224.3 s (3.7 min)** | 75,061 zarr keys → **2.6 MB** parquet |

(849 real timesteps; 8 template hours beyond this Feb date's calendar horizon
were absent on S3 and dropped cleanly.)

### Stability across members (4-sample batch)

```bash
for m in "20260201 06" "20260214 00" "20260220 12"; do
  set -- $m
  uv run lithops-cr-gik-cfs/run_lithops_cfs.py --init-date $1 --run $2 \
    --sequential --yes --local-template cfs_template/gik-fmrc-v2cfs_fmrc.tar.gz \
    --output-dir cfs_ensemble_202602 --max-forecast-hours 5136
done
```

| Member | Time |
|---|---|
| 20260201 00z | 224.3 s |
| 20260201 06z | 221.8 s |
| 20260214 00z | 207.9 s |
| 20260220 12z | 201.1 s |
| **Average** | **≈ 213.8 s (±5 %)** |

Per-member time is stable ⇒ month/corpus extrapolation is reliable. On Cloud Run
in `us-east1` (next to the `noaa-cfs-pds` bucket) the `.idx` reads have lower
latency, so per-member likely drops to ~160–200 s.

### Extrapolation

- **Month** (Feb 2026 = 28 days × 4 cycles = 112 members): sequential ≈ 6.7 h;
  Lithops 50 workers ≈ 11 min; 112 workers ≈ 5 min. Cost ≈ **$0.70–1.40**.
- **Full corpus 2018→2026** (11,212 members, ~2.4M vCPU-s): **~$70** (1 vCPU/
  2 GiB) to ~$140 (2 vCPU/4 GiB). Cost is ~fixed by total vCPU-seconds, *not*
  parallelism — smallest container is cheapest; workers only set wall-clock.

---

## 2. Pipeline internals (Stage 1 → 3)

Entry point per member: `process_cfs_member(init_date, run, ...)`. The template
is the two-part tar `gik-fmrc-v2cfs_fmrc.tar.gz` (deflated skeleton +
per-hour idx→grib mapping parquets), published at
`E4DRR/grib-index-kerchunk-templates`.

### Stage 1 — load the deflated skeleton (`build_deflated_store_from_template`)
- Reads one member parquet `gik-fmrc/v2cfs_fmrc/{date}_{run}/cfs-…-rt000.par`
  from the tar into a `{key: value}` zarr-metadata dict (`.zarray/.zattrs/
  .zgroup` for all ~68 variable groups). No GRIB scan.
- Date-independent: if the processing date has no skeleton entry it falls back
  to `REFERENCE_INIT_DATE` (20260702). ~0.2 s.

### Stage 2 — mapped index (`build_final_store_mapped` → `build_cfs_mapped_index`)
- `CFSMappingManager` indexes the 857 mapping parquets in the tar by
  `(member, forecast_hour)`.
- For each 6-hourly timestep (threaded, `PARALLEL_WORKERS=8`):
  1. `parse_grib_idx(flxf_url)` — read the processing date's **fresh** `.idx`
     from S3 (byte offsets for that date). CFS fractional record ids
     (`36.1`/`36.2`, packed UGRD/VGRD) are handled by a `pd.Series.astype`
     patch.
  2. `map_from_index(datestr, template_mapping, idxdf)` — join the template
     mapping (structure, deduped on `attrs`) with the fresh offsets → this
     date's `[uri, offset_grib, length_grib]` per message.
- Concatenate all timesteps, dedup on `(varname, typeOfLevel, level,
  valid_time)`. Dominant cost (~165 s) = one S3 `.idx` read per timestep.

### Stage 3 — build the real-axis zarr store (`process_unique_groups`)
- `generate_axes_cfs` + `calculate_time_dimensions` build the full 6-hourly
  `valid_time` axis (one entry per forecast hour — the real time dimension).
- Deep-copy the skeleton; drop groups not present in the mapped index; for each
  `(varname, stepType, typeOfLevel)` group call `store_coord_var` (time/
  valid_time/step/level) and `store_data_var` (chunk refs) — kerchunk builds a
  proper multi-timestep zarr, **not** a `step_XXXX/` overlay.
- Skeleton values are normalized back to JSON strings (kerchunk's
  `store_*` expects strings, not parsed dicts). ~58 s for 849×86.

### Stage 3 output
- `create_parquet_simple` → `{key, value}` parquet, one per member
  (`{date}{run}z.parquet`). 849-step member = 75,061 keys, 2.6 MB.

---

## 3. Verification performed (commands)

### 3a. Byte-range correctness (chunk refs vs fresh `.idx`)
The chunk reference for a variable/timestep must equal that date's `.idx`
offset. Verified (t2m/prate/tmax at f024, 20260702): exact match, e.g.
`t2m/…/1.0.0 → offset 1900417 == idx record 37 (TMP:2 m)`.

### 3b. Decode validation — gribberish vs Herbie (`cfs_validate_gribberish_herbie.py`)
```bash
uv run cfs_validate_gribberish_herbie.py \
    --parquet cfs_validation/2026070200z.parquet \
    --date 20260702 --run 00 --member 1 --fxx 24 --out cfs_validation
```
Decodes the same message two ways — GIK byte-range + gribberish vs
`Herbie(model="cfs", product="6_hourly", kind="flxf")` full-file cfgrib.
Result (f024): **t2m/sdswrf/prate all corr = 1.000000, max|Δ| at machine
precision.** 3-panel PNGs written to `cfs_validation/`.
(Note: Herbie's cfs model needs `import herbie.models.cfs`; stack is numpy≥2 /
pandas≥2.2 because gribberish requires numpy≥2.)

### 3c. Cross-year template reuse
Ran the pipeline for one date per year 2018–2025 (`--max-forecast-hours 24`) and
compared the t2m chunk-ref offset to each year's own fresh `.idx`:

| Year | GIK offset == fresh idx |
|---|---|
| 2018–2025 | **all exact** (offsets differ per year → real positions injected) |

⇒ **one template (ref 20260702) serves 2018→2026**; no per-year templates.

---

## 4. Key parameters / env

| Item | Value |
|---|---|
| `TEMPLATE_URL` | `…E4DRR/grib-index-kerchunk-templates/…/gik-fmrc-v2cfs_fmrc.tar.gz` |
| `REFERENCE_INIT_DATE` / `REFERENCE_RUN` | `20260702` / `00` |
| `CFS_MEMBER` | `01` |
| `FORECAST_INTERVAL` | 6 h |
| `PARALLEL_WORKERS` (Stage 2 threads) | 8 |
| `--max-forecast-hours` | 5136 (full member 01); use smaller for quick tests |
| deps added for the rewire | `kerchunk==0.2.7`, `xarray`, `zarr>=2.18,<3`, `numcodecs<0.13` |

## 5. Reproduce

```bash
# one member, full length, local parquet
uv run lithops-cr-gik-cfs/run_lithops_cfs.py --init-date <YYYYMMDD> --run <HH> \
    --sequential --yes --local-template cfs_template/gik-fmrc-v2cfs_fmrc.tar.gz \
    --output-dir <dir> --max-forecast-hours 5136

# whole target month on Cloud Run (Lithops)
uv run lithops-cr-gik-cfs/run_lithops_cfs.py --target-month <YYYYMM> --max-workers 50
```
