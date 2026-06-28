# CFS Template Work — Session Handoff (2026-06-28)

A self-contained summary of the CFS GIK template work done in this session, so a
new session can resume without re-deriving anything. Read this first, then the
two reference docs it points to.

---

## 1. Where the CFS plan stands

CFS GIK is at: **Phase-1 parquet creation implemented; Stage-1 template NOT yet
built/published; no streaming or validation phase.**

Verified Step-1 (template) status across the three products — by direct probe of
HuggingFace `E4DRR/grib-index-kerchunk-templates` (HTTP, this session):

| Product | Template artifact | On HF? |
|---|---|---|
| ECMWF | `gik-fmrc-v2ecmwf_fmrc.tar.gz` (+ per-era `-49r1-perlevel`, `-50r1`, `-0p4-beta`) | ✅ 126 MB |
| GEFS | `gik-fmrc-gefs-20241112.tar.gz` | ✅ 3.2 MB |
| **CFS** | `gik-fmrc-v2cfs_fmrc.tar.gz` | ❌ **404 (not built)** |

Consequence: `run_lithops_cfs.py` Cloud Run path cannot run today — its
`TEMPLATE_URL` (line ~107) points at
`Nishadhka/gfs_s3_gik_refs/.../gik-fmrc-v2cfs_fmrc.tar.gz`, which **404s on both
that repo and E4DRR**. Only the local `--local-template` path works (after a
local `run_cfs_template_creation.py --zarr-template`).

---

## 2. Key correction made this session (was "Lesson 2")

An earlier claim — *"CFS is GEFS-like, so the template only needs a 2-file
`scan_grib`; no Coiled needed"* — was **WRONG**. Corrected and documented in
`cfs/GIK_TEMPLATE_BUILD_CORPUS_SCAN.md`. The findings (all code-verified):

- The GEFS production template is built by `build_idx_grib_mapping` over **every
  (member, hour) = 30 × 81 = 2,430 GRIB files** (`gefs/README.md:23`;
  `gefs/gefs_index_preprocessing_fixed.py:163,187,305` — Dask-on-Coiled fan-out).
  `build_idx_grib_mapping` runs `scan_grib` internally, so the template build IS
  a whole-corpus scan, run on a cluster — same shape as the ECMWF per-(stream,
  hour) Coiled scan. **A whole-corpus scan on Coiled/Dask is mandatory.**
- GEFS has **two distinct `scan_grib` roles**, previously conflated: (A) a cheap
  2-file runtime skeleton scan, and (B) the expensive whole-corpus template
  build. The earlier claim looked at (A) and missed (B).
- The ECMWF **pressure-level collapse bug is absent in GEFS**: GEFS uses
  kerchunk's native `store_data_var` which builds a real level axis
  (`gefs/lithops-cr-gik-gefs/run_lithops_gefs.py:818-833`). The bug was
  ECMWF-specific (it re-implemented store-building and keyed on level *type*).
  CFS keys on the level *string value* (`run_lithops_cfs.py:468-469`), so it
  also avoids collapse — but builds neither a real axis nor has any validation.

---

## 3. What was built this session

Two new scripts in `cfs/` (committed; see §6):

### `cfs_coiled_preprocessing.py` — Stage-1 corpus scan (Coiled)
Adapted from `ecmwf/dev-test/ecmwf_50r1_coiled_preprocessing.py`. Parallel
(Coiled) replacement for the sequential `process_cfs_member` loop. Per task =
one (member, forecast-hour): `build_idx_grib_mapping` (deduped on `attrs`) with
the CFS `.idx` fractional-record patch (`36.1`/`36.2`) and `validate=False`.

- Task grid: members `01[,02,03,04]` × 6-hourly hours `0..MAX` (861 hours for a
  215-day run). All-members full-range = **3,444 tasks**.
- Output: `cfs-time-{date}-{run}-rt{hhhh}.parquet` →
  `gs://gik-cfs-aws-tf/cfs/time_idx/{year}/{date}/{run}/{member}/`.
- Coiled config mirrors ECMWF (`gik-coiled-pinned`→`gik-coiled-v6` fallback,
  `n2-standard-2`, `us-east1`, `gcp-sewaa-nka`, anon-S3 forcing,
  `coiled-data.json` shipped to workers). `--dry-run` is free.
- Verified: compiles; dry-run plans correct; generated S3 URL resolves live
  (`flxf...grb2` + `.idx`, HTTP 206). **Not runnable by the assistant** (needs
  Coiled auth + `coiled-data.json`).

### `cfs_package_template.py` — packaging + HuggingFace upload
Mirrors the ECMWF runbook §6.3-6.4. Bundles the zarr **skeleton** into
`gik-fmrc-v2cfs_fmrc.tar.gz` in the exact layout the runtime expects, then
optionally uploads to HF.

- Tar member path (matches `run_lithops_cfs.py:230-233` exactly):
  `gik-fmrc/v2cfs_fmrc/{date}_{run}/cfs-{date}{run}-member01-rt000.par`.
- `--replicate-month YYYYMM` / `--replicate-dates` write the same skeleton under
  every (date,run) of an init month (see §4 for why this is needed).
- `--upload --repo E4DRR/grib-index-kerchunk-templates` (HF_TOKEN from env or
  `.env`). `--dry-run` writes nothing.
- Verified end-to-end on a synthetic skeleton: tar write, self-check
  (re-open + read the reference member), and replicate-month expansion (120
  members for Nov) all pass.

---

## 4. Critical architecture note for the next session

There are **two template models**, and CFS currently mixes signals:

| | GEFS model | ECMWF model |
|---|---|---|
| Template content | per-(member,hour) mapping parquets | one deflated **zarr skeleton** per member |
| Runtime Stage-2 | kerchunk `map_from_index` | direct `.idx` parse |

**`run_lithops_cfs.py` as written follows the ECMWF model**: at runtime it loads
only the zarr skeleton (`build_deflated_store_from_template`) and parses `.idx`
directly (`build_refs_from_indices`); `merge_with_template` is a pure dict
overlay (`run_lithops_cfs.py:582-586`). **It does NOT consume the per-hour
mapping parquets** that `cfs_coiled_preprocessing.py` produces.

So, today:
- `cfs_package_template.py` produces what the runtime needs (the skeleton tar).
- `cfs_coiled_preprocessing.py`'s mapping parquets are **not consumed at
  runtime**. Their value is (a) **schema-completeness verification** — proving
  the 2-file (`f000`+`f006`) skeleton scan actually covers every
  variable/level/stepType across the full run (it may not — accumulated vs
  instantaneous fields and some variables appear only at later hours), or
  (b) the basis if CFS is switched to the GEFS-style `map_from_index` path.

**Decision the next session must make:** keep the ECMWF-style skeleton path (then
the coiled scan is a *validation/completeness* tool, and only the skeleton needs
packaging), or move CFS to the GEFS-style mapping path (then wire
`map_from_index` into `run_lithops_cfs.py` and package the mapping parquets).

### Lookup-by-processing-date quirk
`build_deflated_store_from_template(template_path, init_date, run)` is called
with the **processing** date (`run_lithops_cfs.py:790`), and that date is
embedded in the tar path. A pre-built tar therefore needs one entry per
processed (init_date, run) — hence `--replicate-month` in the packaging script.
Cleaner long-term fix: have the runtime look the skeleton up by a fixed
`REFERENCE_INIT_DATE` (the skeleton structure is date-independent), so one entry
serves all dates. Replication is the no-runtime-change stopgap.
**Caveat:** replication reuses the reference skeleton's coordinate metadata for
all dates — validate a non-reference-date decode before trusting it.

---

## 5. End-to-end runbook to finish the CFS template (next session)

```bash
cd cfs/

# (1) Build the zarr skeleton (one-time, ~24s). Consider a LATER --init-date /
#     larger --max-forecast-hours so the scan sees all variables/stepTypes,
#     not just f000+f006. (Schema-completeness — see §4.)
uv run run_cfs_template_creation.py --zarr-template \
    --init-date 20251101 --run 00 \
    --local-dir ./cfs_test/template --max-forecast-hours 48
# -> cfs_test/template/cfs-zarr-template-20251101-00.parquet

# (2) OPTIONAL completeness check: corpus scan on Coiled (needs Coiled auth +
#     coiled-data.json). Compare the variable/level/step set in the mapping
#     parquets against the skeleton. --dry-run first (free).
python cfs_coiled_preprocessing.py --date 20251101 --run 00 --members 01 --dry-run
# python cfs_coiled_preprocessing.py --date 20251101 --run 00 --members 01   # paid

# (3) Package + upload the skeleton. Replicate across the init month you will
#     process so the runtime's per-date lookup resolves.
uv run cfs_package_template.py \
    --skeleton cfs_test/template/cfs-zarr-template-20251101-00.parquet \
    --ref-date 20251101 --run 00 --replicate-month 202511 \
    --out gik-fmrc-v2cfs_fmrc.tar.gz \
    --upload --repo E4DRR/grib-index-kerchunk-templates

# (4) Point the runtime at it (currently 404):
#     edit run_lithops_cfs.py:107 TEMPLATE_URL, or set env:
export TEMPLATE_URL="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main/gik-fmrc-v2cfs_fmrc.tar.gz"

# (5) Smoke-test a real date locally, then validate a decode.
uv run lithops-cr-gik-cfs/run_lithops_cfs.py \
    --init-date 20251101 --run 00 --sequential \
    --local-template cfs_test/template/cfs-zarr-template-20251101-00.parquet \
    --output-dir ./cfs_output --max-forecast-hours 48 --yes
```

---

## 6. Open items / backlog (priority order)

1. **Build + publish the CFS template** (runbook §5) — fixes the 404; unblocks
   Cloud Run.
2. **Resolve TEMPLATE_URL repo mismatch** — runtime points at `Nishadhka/...`,
   the other templates live at `E4DRR/...`. Pick one; update line ~107.
3. **Validate a decode** — there is NO CFS GIK-vs-Herbie check (ECMWF has
   `compare_gik_herbie*.py`). Highest-risk gap: parquets are produced but never
   proven to decode to correct values.
4. **Schema completeness** of the skeleton — confirm a 2-file scan covers all
   variables/levels/stepTypes over the full forecast range (use the coiled scan
   from §3 / runbook step 2).
5. **Decide the template model** (§4) — ECMWF-skeleton vs GEFS-mapping.
6. **TMIN/TMAX/sfcWind derivation** and **ensemble members 02-04** — documented
   gaps in `cfs/CLAUDE.md`, still unimplemented.

---

## 7. File / commit reference

| Path | Role |
|---|---|
| `cfs/cfs_coiled_preprocessing.py` | NEW — Stage-1 corpus scan on Coiled (GEFS-style mapping parquets) |
| `cfs/cfs_package_template.py` | NEW — package skeleton → `gik-fmrc-v2cfs_fmrc.tar.gz` → HF |
| `cfs/GIK_TEMPLATE_BUILD_CORPUS_SCAN.md` | NEW — why whole-corpus scan is mandatory; GEFS level-bug analysis |
| `cfs/CFS_TEMPLATE_WORK_HANDOFF.md` | NEW — this handoff |
| `cfs/run_cfs_template_creation.py` | existing — builds the skeleton (`--zarr-template`) and mapping parquets |
| `cfs/lithops-cr-gik-cfs/run_lithops_cfs.py` | existing — runtime; ECMWF-style; `TEMPLATE_URL`@107 (404), tar lookup @230-233, merge @582 |
| `cfs/GIK_THREE_DATASET_COMPARISON.md` | existing — CFS/ECMWF/GEFS pipeline comparison |

Commits on branch `ecmwf-50r1-template` (identity nishadhka <nishadhka@gmail.com>):
- `d5714d0` — coiled preprocessing driver + corpus-scan doc
- (this commit) — packaging script + this handoff doc
