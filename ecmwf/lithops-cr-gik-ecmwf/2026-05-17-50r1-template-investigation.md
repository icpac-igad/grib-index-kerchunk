# ECMWF GIK — 50r1 Break & scan_grib Template-Creation Investigation

**Branch:** `ecmwf-50r1-template`   **Date:** 2026-05-17

This record captures *how* the problem was identified, the dead-ends and
corrections along the way, and what must be built. It is the rationale behind
`2026-05-17-50r1-template-rebuild-plan.md`.

---

## 1. How the problem surfaced

1. Backfilled the GCS gap 2026-04-08 → 2026-05-15 (all run hours) +
   2026-03-07 12z/18z. All complete **except** a recent-date tail showing
   **50/51 parquets**.
2. First hypothesis (**wrong**): ECMWF S3 *publishing lag* — "the 51st member
   isn't published yet, re-run later." An audit of the `.index` files
   disproved this: the source itself had exactly 50 members for those dates,
   structurally, not transiently.
3. User flagged the ECMWF IFS Cycle **50r1 / AIFS v2** implementation
   (forum + Confluence). That reframed 50/51 as a **model-cycle breaking
   change**, not a data gap.

## 2. Pinning the 50r1 break (evidence-based)

`.index` deltas verified per run hour:

| | 49r1 (≤2026-05-12 00z) | 50r1 (≥2026-05-12 **06z**) |
|---|---|---|
| Transition | — | exactly **2026-05-12 06z** (05-12 00z still 49r) |
| enfo members | 51 (`pf` 1..50 + `cf` control) | **50** (`pf` 1..50 only) |
| Control | `enfo/cf` in `*-enfo-ef` | **moved** → separate `…/ifs/0p25/**oper**/{ts}-{H}h-**oper-fc**` |
| Pressure levels | 13 | **14** (new **10 hPa**) |
| enfo msgs 0h | 8211 | 8500 (sfc 1836→1800, sol 408→400, pl 5967→6300) |

Key correction: **control is NOT removed — it moved open-data streams**
(MARS `class=od,stream=oper,type=fc`, was `enfo/cf` + `scda/fc`). It is a
**superset** of a perturbed member: `oper/fc` 0h = 187 msgs / 50 params vs
perturbed `enfo/ef` = 170 / 47 — control has extra static fields
`z`, `sdor`, `slor`. 51-member output remains achievable; GIK just never read
the `oper` stream. Confluence documents only MARS; **S3 is ground truth**.

## 3. The scan_grib template-creation issue (provenance)

CLAUDE.md claims the ECMWF template was built by
`ecmwf_index_preprocessing.py` (`build_idx_grib_mapping`). Investigation
showed this is **inaccurate**, via several dead-ends:

- `ecmwf_index_preprocessing.py` *does* run scan_grib (kerchunk 0.2.7
  `build_idx_grib_mapping` → `_map_grib_file_by_group` → `scan_grib`) but
  writes a different artifact (`ecmwf/{member}/…rt{HHH}.parquet` mapping
  tables) that `run_lithops_ecmwf.py` never reads.
- `ecmwf_util.py`'s scan_grib path: a faithful dry-run on 20260513 00z 0h
  showed its `identify_ensemble_members()` returns **0 members** — scan_grib
  groups carry no `number` in `attrs`. That path is **dead code** for this
  data.
- The **working** builder is the **idx-position-join** in
  `utils_ecmwf_step1_scangrib.py` / `fmrc_utils.py`
  (`ecmwf_filter_scan_grib`: `for i,group in enumerate(scan_grib(...))` →
  `idx_mapping[i]`), driven by the Coiled notebook
  `99o-coiled-function-ecmwf-scan_grib_store_fmrc.ipynb`.
- **No repo script tars/uploads** the template (zero `tarfile.open(...,'w')`)
  — the per-member parquets are produced to GCS, then `tar`+HF-upload
  **manually**.
- Pulling the live 126 MB `gik-fmrc-v2ecmwf_fmrc.tar.gz` proved the template
  interior is **51 members × 85 `rt{hhh}.par` = 4,335** per-(member,hour)
  deflated zarr stores (full step set 000..360) — the notebook's
  **all-85-hours** scan, *not* a 2-file/`rt000`-only build. (An earlier
  assumption that only `rt000` mattered was wrong; `run_lithops_ecmwf.py`
  reads only `rt000` but the canonical HF template ships the full set for
  index-based recreation / tutorial / virtualizarr consumers.)

Dry-run timing: scan_grib ≈ **12.6 min for one enfo file** (8500 groups) ⇒
85 files ÷ 9 Coiled workers ≈ **~2 h** — matches the historical run.

## 4. What must be built for the new (50r1) template

Decisions locked: reference **2026-05-13 00z** (first 50r1 date that is also a
full 85-step run — 06z/18z only reach 144h); **Option A** — keep 51 members,
build the control from `oper/fc` retaining its superset schema.

1. **Builder fixes** (`utils_ecmwf_step1_scangrib.py`):
   - `ecmwf_idx_unique_dict`: `pressure_levels` allowlist lacks `'10'` →
     would silently drop the new 10 hPa level. Add `'10'`.
   - `ecmwf_duplicate_dict_ens_mem` / `organize_ensemble_tree`: 51-member
     hardcodes (`np.insert(arange(1,51),0,-1)`, `np.arange(-1,50)`,
     `shape/chunks:[51]`) → perturbed = **50** (1..50, `[50]`).
2. **enfo + oper/fc split (one pass):** perturbed `ens_01..50` from
   `enfo/ef`; `ens_control` from `oper/fc` (separate scan, no member filter,
   superset schema kept).
3. **Full-85-step** scan for 2026-05-13 00z (perturbed: 85 enfo files;
   control: 85 oper/fc files) → `gik-fmrc/v2ecmwf_fmrc/{member}/
   ecmwf-2026051300-{member}-rt{hhh}.par`.
4. Package `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz`, upload to HF
   `E4DRR/grib-index-kerchunk-templates` (new name; keep 49r tar.gz for
   ≤2026-05-12 00z history).
5. **Then** (separate, after template exists): `run_lithops_ecmwf.py`
   `REFERENCE_DATE→20260513`, new template URL, control via `oper/fc` in
   Stage 2; rebuild/redeploy Cloud Run runtime; reprocess 2026-05-12 06z →
   present (with the cleanup-hang watchdog); re-mirror to HF.

## 5. Sequence rule learned

Template is created **first**; `run_lithops_ecmwf.py` is operational only
*after* the template exists — do not wire the consumer to a non-existent
template (an earlier premature edit was reverted).

## 6. Security note

`service_account/*.json` and `ecmwf/coiled-data.json` were **not**
gitignored (SETUP_NEW_MACHINE.md wrongly states they are). `.gitignore`
hardened in this commit; staging is always explicit, never `git add -A`.
