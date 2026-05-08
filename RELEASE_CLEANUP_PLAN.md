# GIK v1.0 Release — Cleanup Plan

This document inventories every Python / shell script currently in
`ecmwf/` and `gefs/` and proposes whether each stays on the main path
or moves into `<datasource>/dev-test/`. It is the single source of
truth before any file moves so you can edit / approve in one place.

> **Goal**: ship a tagged `v1.0.0` release where the main path of each
> datasource folder contains *only* the latest, working, validated
> scripts that implement the canonical 4-step GIK pipeline plus the
> Herbie / Coiled / dynamical-earth-zarr validators.

---

## 1. The canonical 4-step GIK pipeline

| Step | What it does | Inputs | Outputs |
|------|--------------|--------|---------|
| **1. Template** | One-time per (member, timestep) sweep on a reference date — `build_idx_grib_mapping` over every forecast hour for every ensemble member, packed into a `.tar.gz`. The tar.gz is the *schema + per-timestep idx↔grib mapping* of the dataset; the zarr metadata skeleton is recoverable from any one `rt000` parquet inside it. (For GEFS only, an additional 27 KB `gefs-deflated-store-template-*.parquet` is bundled separately, made one-time by `scan_grib(f000, f003)` — convenience artefact, not load-bearing.) | reference-date GRIB sidecars (`.idx` / `.index`) for all 30×81 / 51×85 (member, hour) pairs | `gik-fmrc-{model}-YYYYMMDD.tar.gz` (~10–30 MB) |
| **2. Index from template** | For any new date/member, read NOAA's `.idx` file (a few KB), look up byte ranges, merge with the template's mapping parquets — **no scan_grib needed**. | template + `.idx` | per-member kerchunk dict |
| **3. Build parquet & save** | Combine step 1 (zarr metadata) + step 2 (byte-range refs) → write one parquet per member-date to GCS / object storage. Serverless via Lithops on Cloud Run. | template + idx outputs | `s3://noaa-gefs-pds/...` references in `.parquet` (~280 KB / file) |
| **4. Stream → xarray** | Read the parquet, reconstruct the kerchunk zstore, fetch only the byte ranges you need, decode with gribberish (or cfgrib fallback), wrap in xarray Dataset. | parquet | `xr.Dataset` (lazy or materialised) |

For step 4 the canonical consumers are:
- **Herbie validator** — bit-equivalence proof against `herbie-data`
- **dynamical-earth zarr validator** — comparison against the published `dynamical.org` ARCO zarr store
- **Coiled Dask cluster** — distributed reads for ensemble-wide analysis

Anything not directly implementing one of these steps moves to `dev-test/`.

---

## 2. Top-level changes

| File | Action | Why |
|------|--------|-----|
| `README.md` | **Edit** — add v1.0 quickstart pointing at `gefs/README.md` and `ecmwf/README.md`; keep "Supported Products" table; mark `gfs/` and `cfs/` as **legacy / not part of v1.0**. | Currently mixes GEFS pipeline doc with the umbrella overview. |
| `gfs/`, `cfs/` | **Leave untouched** for v1.0 | User scoped this to ecmwf + gefs |
| `tutorial/` | **Leave untouched** | Out-of-scope; tutorials reference both datasources |
| `devops/` | **Leave untouched** | Different concern (private deployment tooling) |

---

## 3. ECMWF — proposed actions per file

### 3a. KEEP (canonical, on main path)

| File | Step | Notes |
|------|------|-------|
| `ecmwf_util.py` | shared | All steps depend on this |
| `utils_ecmwf_step1_scangrib.py` | 1 | Template builder |
| `ecmwf_index_preprocessing.py` | 1 | Template + initial index |
| `ecmwf_index_processor.py` | 2 | Index lookups via template |
| `ecmwf_three_stage_multidate.py` | 1+2+3 | Local (non-Lithops) end-to-end runner |
| `lithops-cr-gik-ecmwf/` (folder) | 3 | Cloud Run / Lithops orchestration |
| `run_single_ecmwf_to_zarr_gribberish.py` | 4 | Read one parquet → xr.Dataset |
| `compare_gik_herbie.py` | 4 (validation) | Herbie comparison |
| `fetch_tp_herbie.py` | 4 (validation) | Herbie fetch helper |
| `validate_gik_vs_herbie_2025.py` | 4 (validation) | Latest year-long validation |
| `stream_cgan_variables_coiled_simple.py` | 4 (Coiled Dask) | Single canonical Coiled streamer |
| `upload_parquets_to_hf.py` | 4 (publish) | Mirror parquets to HuggingFace |
| `consolidate_parquets_to_hf.py` | 4 (publish) | Bulk consolidator → keep, but docs needed |
| `hf_README.md` | docs | Dataset card |
| `README.md` | docs | Folder overview |
| `__init__.py` | pkg | Keep (module marker) |
| `.env.example` | config | Keep |
| `docs/` | docs | Keep |
| `gik_vs_herbie/` | results | Keep (validation artefacts) |
| `plots/` | results | Keep (selectively — see below) |
| `dev-test/` | sandbox | Already populated; will receive more |

### 3b. MOVE → `ecmwf/dev-test/` (older / superseded / experimental)

| File | Reason |
|------|--------|
| `aifs-etl.py`, `aifs-etl-v2.py` | AIFS is a separate model; not part of v1.0 GIK pipeline |
| `read_stage3_aifs_all_timesteps.py` | AIFS-specific consumer |
| `compare_ecmwf_opendata_vs_gik.py` | Open-data comparison (one-off; superseded by Herbie validator) |
| `ecmwf_ensemble_par_creator_efficient.py` | Older predecessor of `ecmwf_three_stage_multidate.py` |
| `ecmwf_ensemble_par_creator_efficient_multidate.py` | Older predecessor (same) |
| `ecmwf_ensemble_par_creator_v2.py` | Even older predecessor |
| `ecmwf_ea_tp_icechunk.py` | Icechunk experiment; replaced by source.coop zarr |
| `ecmwf_ea_tp_pencil_zarr.py` | Pencil-tile rechunker (specialised, not core pipeline) |
| `ecmwf_ea_tp_source_coop.py` | source.coop publication pipeline (separate workstream) |
| `ecmwf_ea_tp_source_coop_zarr.py` | source.coop zarr variant |
| `inspect_icechunk_store.py` | Diagnostic for retired Icechunk path |
| `validate_gik_vs_herbie_2024.py` | Earlier vintage of `_2025.py` |
| `validate_gik_vs_herbie_single.py` | Smoke-test variant |
| `run_random_date_test.py` | Superseded by `validate_gik_vs_herbie_2025.py` |
| `run_ecmwf_tutorial.py` | Tutorial entry-point (lives better under `tutorial/`) — **REVIEW** |
| `stream_cgan_variables.py` | Single-machine variant (keep Coiled one) |
| `stream_cgan_variables_coiled.py` | Older Coiled variant (keep `_simple` only) |
| `plot_cgan_maps.py` | Application-specific plot |
| `plot_source_coop_tp_probability.py` | source.coop-specific |
| `plot_tp_maps.py` | Generic but unused — **REVIEW** |
| `test_catalog_virtual.py`, `test_catalog_xarray.py` | Catalog experiments |
| `test_gribberish_vs_scangrib.py` | Decoder benchmark |
| `hf_readme_update.py` | One-off README updater |
| `2026-02-19-gik-validation.txt`, `2026-02-19-herbie-gik-difference.txt` | Session logs |
| `COILED_DASK_IMPLEMENTATION_PLAN.md`, `GRIBBERISH_EXPERIMENT_REPORT.md` | Historical planning docs (alternatively → `docs/archive/`) |

### 3c. REVIEW — RESOLVED

| File | Decision |
|------|----------|
| `run_ecmwf_tutorial.py` | **MOVE → `tutorial/ecmwf/`** (sibling of existing tutorials) |
| `plot_tp_maps.py` | **MOVE → `ecmwf/dev-test/`** |
| `consolidate_parquets_to_hf.py` | **MOVE → `ecmwf/dev-test/`** — replaced by `upload_parquets_to_hf.py --aggregate`. Prerequisite: ECMWF script ported with the same GEFS optimisations (member col + sort + row_group_size=102 + zstd + retry wrapper) — done as a prep commit before the moves. |

---

## 4. GEFS — proposed actions per file

### 4a. KEEP (canonical, on main path)

| File | Step | Notes |
|------|------|-------|
| `gefs_util.py` | shared | All steps depend on this |
| `run_gefs_preprocessing.py` | 1 | Template builder |
| `gefs-deflated-store-template-20241112.parquet` | 1 (artefact) | Pre-built Stage-1 template |
| `lithops-cr-gik-gefs/` (folder) | 3 | Cloud Run / Lithops orchestration |
| `create_gik_tp_netcdf.py` | 4 | Read parquet → TP NetCDF (the canonical step-4 example) |
| `run_single_gefs_to_zarr_gribberish.py` | 4 | Single member → zarr |
| `compare_gik_herbie_gefs.py` | 4 (validation) | Herbie comparison |
| `fetch_tp_herbie_gefs.py` | 4 (validation) | Herbie fetch helper |
| `validate_gik_vs_herbie_2022_2026.py` | 4 (validation) | Multi-year Herbie validator |
| `validate_hf_gik_vs_herbie.py` | 4 (validation) | NEW — pulls refs from HF aggregate, validates vs Herbie |
| `upload_parquets_to_hf.py` | 4 (publish) | Mirror parquets / build aggregates / catalog |
| `process_ensemble_by_variable.py` | 4 (analysis) | Ensemble stats — used in tutorials |
| `run_gefs_24h_accumulation.py` | 4 (analysis) | 24h precip + probability — referenced from README |
| `hf_README.md` | docs | Dataset card |
| `README.md` | docs | Folder overview |
| `GEFS_Three_Stage_Processing.md` | docs | Pipeline description |
| `.env.example`, `.gitignore` | config | Keep |
| `docs/` | docs | Keep |
| `gik_vs_herbie_gefs/` | results | Keep (validation artefacts) |
| `dev-test/` | sandbox | Already populated; will receive more |

### 4b. MOVE → `gefs/dev-test/` (older / superseded / experimental)

| File | Reason |
|------|--------|
| `run_day_gefs_ensemble_full.py` | Single-machine driver superseded by Lithops Cloud Run |
| `run_single_gefs_to_zarr.py` | cfgrib variant of the gribberish one (keep gribberish only) |
| `test_three_stage_gefs_simple.py` | Test-bed for three-stage pipeline |
| `validate_gik_gcs_vs_herbie_cost_test.py` | Cost-test instrumented variant of the validator |
| `upload_validation_pngs_to_hf.py` | One-off PNG uploader |
| `plot_ensemble_east_africa.py` | Application-specific plot — **REVIEW** |
| `ea_ghcf_simple.geojson` | EA region polygon (small artefact) — **REVIEW** keep or move to `assets/` |
| `gik_tp_gefs_output/`, `validation_hf_gik_vs_herbie/`, `plots/`, `logs/` | Runtime output dirs — gitignore + keep empty in repo, or move into `dev-test/runs/` |
| `env/` | Old uv env scratch dir — delete or gitignore |

### 4c. REVIEW — RESOLVED (all → dev-test)

| File | Decision |
|------|----------|
| `process_ensemble_by_variable.py` | **MOVE → `gefs/dev-test/`** |
| `run_gefs_24h_accumulation.py` | **MOVE → `gefs/dev-test/`** |
| `plot_ensemble_east_africa.py` | **MOVE → `gefs/dev-test/`** |
| `gefs-deflated-store-template-20241112.parquet` | **MOVE → `gefs/dev-test/`** (fetched from HF templates dataset at runtime; no need to bundle in repo) |

---

## 5. Commit / tag strategy (locked)

```
Commit  Subject (final)
─────────────────────────────────────────────────────────────────────
1       docs: add RELEASE_CLEANUP_PLAN.md   (already shipped — b929896)
2       ecmwf: port GEFS aggregate optimisations (member/sort/zstd/retry)
        + lock §3c & §4c decisions in RELEASE_CLEANUP_PLAN.md
3       ecmwf: archive legacy parquet creators (v2/efficient/multidate)
4       ecmwf: archive AIFS pipeline (separate workstream)
5       ecmwf: archive icechunk + source.coop / pencil zarr variants
6       ecmwf: archive older validators + run_random_date_test
7       ecmwf: archive plot/test/doc one-offs (CGAN plots, gribberish-vs-scangrib, etc.)
8       ecmwf: keep one Coiled streamer; move tutorial runner to tutorial/ecmwf/
9       ecmwf: archive consolidate_parquets_to_hf.py
        (functionality folded into upload_parquets_to_hf.py --aggregate)
10      gefs: archive single-machine driver + cfgrib variant
11      gefs: archive cost-test, PNG uploader, scratch outputs
12      gefs: archive REVIEW items (process_ensemble, 24h_accum, EA plot,
        bundled .parquet template)
13      docs: refresh top-level README + ecmwf/README + gefs/README for v1.0
14      docs: v1.0.0 release notes
─────────────────────────────────────────────────────────────────────
TAG     v1.0.0 — git tag -a v1.0.0 -m "GIK v1.0 — GEFS + ECMWF GIK pipeline"
```

Each move-commit message follows this template:

```
<area>: archive <category> scripts to dev-test/

Files moved (no functional change):
  - <file-1>     superseded by <replacement>
  - <file-2>     <reason>
  ...

Reason for archive:
  <1-2 sentence explanation>

Files preserved on main path for v1.0 (referenced by README):
  - <list>
```

---

## 6. Top-level README + folder READMEs

After the moves, the **three READMEs** should be tightened:

### `README.md` (top level)
- Lead with: "GIK turns multi-GB GRIB files into ~140 KB virtual references"
- Include the 4-step diagram from §1 above
- "Supported products" table — mark GEFS + ECMWF as **production**, GFS as **legacy**, CFS as **experimental**
- Link to `ecmwf/README.md` and `gefs/README.md` for the deep dives
- Add "Citation" + "License" + "Contributing" sections

### `ecmwf/README.md`
- Same 4-step structure
- Quick-start: one date end-to-end via `ecmwf_three_stage_multidate.py`
- Link to `lithops-cr-gik-ecmwf/` for the Cloud Run path
- Link to `hf_README.md` for the dataset card on HF
- Validation: link to `compare_gik_herbie.py` + `validate_gik_vs_herbie_2025.py`

### `gefs/README.md`
- Already updated (commit `7e0c5a5`); needs only a §-level heading consistency pass after the moves
- Link to `lithops-cr-gik-gefs/BACKFILL.md` for backfill ops
- Link to `validate_hf_gik_vs_herbie.py` for the published-dataset validator

---

## 7. Status

- §3c / §4c review items: **RESOLVED** (see locked decisions above).
- §3a / §4a "KEEP" lists: implicitly confirmed by user reply
  ("move all to dev-test, move parquet file to dev-test as well"
  applies to §4c only — §4a / §3a stay as proposed).
- §5 commit sequence: **LOCKED**.
- README rewrites: drafted **after** the moves, in commit 13 (so each
  README references the post-move file layout).
