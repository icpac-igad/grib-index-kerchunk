# ECMWF GIK Template Rebuild for IFS Cycle 50r1 — Implementation Plan

**Date:** 2026-05-17
**Trigger:** IFS Cycle 50r1 / AIFS v2 went live in the open-data `enfo-ef`
stream at **2026-05-12 06z**. The production template
`gik-fmrc-v2ecmwf_fmrc.tar.gz` (ref 2024-05-29, Cycle ≤49r1) is stale for every
run ≥ 2026-05-12 06z.

## 1. Verified facts driving this plan

| Fact | Evidence |
|---|---|
| 50r1 live at exactly 2026-05-12 06z (05-12 00z still 49r) | `.index` deltas pinned per run hour |
| Control (`type:cf`/number 0) **removed** → 50 members | `.index` types: 49r `{pf,cf}` → 50r1 `{pf}` only |
| New **10 hPa** pressure level added (13→14 levels) | pl levelist 49r `[50..1000]` → 50r1 `[10,50..1000]` |
| Msgs 8211 → 8500 (sfc 1836→1800, sol 408→400, pl 5967→6300) | per-step `.index` scan |
| Ref date must be a **full-85-step 00z** 50r1 date | 06z/18z ENS only run to 144h (49 steps) → 2026-05-12 06z unusable |
| **Chosen reference: 2026-05-13 00z** | first 50r1 date with full 0–360h (85 steps) |
| Template interior = **51 members × 85 `rt{hhh}.par` = 4,335** per-(member,hour) deflated stores (full 000..360) | verified: pulled live 126 MB tar.gz. Notebook all-85-hours scan; `run_lithops_ecmwf.py` reads only rt000 but HF template must ship full set |
| Built by scan_grib path in `ecmwf_util.py`, **not** `ecmwf_index_preprocessing.py` | provenance trace; no `tarfile.open(...,'w')` exists → manual tar+HF upload |

## 2. Scope of code changes

> **CORRECTION (2026-05-17), supersedes "drop control" framing below.**
> The control is **not removed in 50r1 — it moved open-data streams**:
> perturbed 1..50 stay in `…/ifs/0p25/enfo/{ts}-{H}h-enfo-ef`; the control
> is now `…/ifs/0p25/**oper**/{ts}-{H}h-**oper-fc**` (separate file the
> pipeline never reads). Empirically: control 0h = 187 msgs / 50 params,
> a **superset** of a perturbed member (170 msgs / 47 params) — extra
> static fields `z`, `sdor`, `slor`; both have the new 10 hPa level.
> ⇒ 51-member output is still achievable. **Open decision:**
> **A (recommended)** keep 51 → builder + Stage 2 gain an `oper/fc` read
> path for `ens_control`; perturbed unchanged (`enfo/ef`, 50).
> **B** drop to 50 perturbed-only (simpler; loses control; breaks
> downstream that expects 51 / a `control` member).
> **DECIDED 2026-05-17: Option A** — keep 51, control from `oper/fc` with
> its superset schema retained (z/sdor/slor kept). Concrete A change-set:
> 1. Builder: perturbed `ens_01..50` from `enfo/ef` (50, idx-join, +`'10'`
>    pl fix); `ens_control` built **separately** by scanning the `oper/fc`
>    file (single forecast, no member filter) → its own rt000.par.
> 2. `run_lithops_ecmwf.py` Stage 2: keep `enfo/ef` `.index` path for
>    perturbed; add an `oper/fc` `.index` path
>    (`…/ifs/0p25/oper/{ts}-{H}h-oper-fc.index`) for `ens_control`.
> 3. Stage 1 member list keeps `ens_control`; tar.gz keeps `ens_control/`.
> 4. Control parquet carries 3 extra vars (z/sdor/slor) — accepted as-is.
> Sections 2.1–2.3 below = the shared (perturbed) mechanics; everything
> there still applies, plus the dedicated control path above.

### 2.1 Stage-1 template builder (`ecmwf/ecmwf_util.py` + orchestrator)
- `ECMWF_CONTROL_NUMBER=-1`, `ECMWF_ALL_NUMBERS=[-1]+1..50` → drop control:
  `ECMWF_ALL_NUMBERS = list(range(1,51))` (50 members).
- `identify_ensemble_members()` normalizes `0→-1`; with no control this just
  yields 1..50 — harmless, but the `fixed_ensemble_grib_tree()` loop over
  `ECMWF_ALL_NUMBERS` will log "Member -1 not found" unless the list is fixed.
- `save_ecmwf_parquet()` member-dir logic keeps `ens_control` branch — unused
  once control is dropped; leave or remove.
- **The working builder is the idx-join path** (`utils_ecmwf_step1_scangrib.py`
  / `fmrc_utils.py`), NOT `ecmwf_util.py` (its attrs-based member ID yields 0
  members — see §4). The two concrete code edits are in that module:
  `ecmwf_idx_unique_dict` pressure-level allowlist (+`'10'`) and the 51→50
  member hardcodes in `ecmwf_duplicate_dict_ens_mem` / `organize_ensemble_tree`.
- Confirm which orchestrator actually drives the production tar.gz build
  (`ecmwf_ensemble_par_creator_efficient*.py` imports `utils_ecmwf_step1_scangrib`;
  the `99o-…ipynb` uses `fmrc_utils.py`) and apply the two edits there.

### 2.2 Production pipeline (`run_lithops_ecmwf.py`)
- `REFERENCE_DATE = '20240529'` → `'20260513'`.
- `build_deflated_stores_from_template()` iterates
  `['ens_control'] + [f'ens_{i:02d}' for i in range(1,51)]` (51) and reads
  `ecmwf-{date}00-{member}-rt000.par`. Drop `ens_control` → 50 members.
- `parse_grib_index()`: `member_num=int(get('number',0)); if 0 -> 'control'`.
  In 50r1 no entry has number 0, so 'control' is simply never produced —
  no code change strictly required, but the Stage-1 member list must match.
- `validate_index_availability()`: `if n_members < 50: return False`. 50r1 has
  exactly 50 → passes. Consider tightening to `!= 50`.
- `TEMPLATE_URL` / `ECMWF_TEMPLATE_PATH` → new 50r1 template artifact name.
- Filenames/paths (`{date}{run}z-{member}.parquet`) unaffected.

### 2.3 Versioning strategy (avoid breaking ≤2026-05-12-00z)
- New artifact name, e.g. `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` (do **not**
  overwrite the 49r tar.gz on HF).
- The 49r template stays valid for ≤2026-05-12 00z; 50r1 template for
  ≥2026-05-12 06z. Since `REFERENCE_DATE` is a single global, the simplest
  operational model: cut over fully (all future runs are 50r1) and treat
  ≤05-12-00z as already-complete history (it is). No dual-template runtime
  logic needed unless we must *re-backfill* a 49r date later (then run the
  old image/commit).

## 3. Build & deploy steps

1. **Env**: `uv run --with kerchunk==0.2.7 --with cfgrib --with eccodes
   --with s3fs --with zarr==2.18.7 --with pandas --with numpy` (verified
   buildable; eccodes 2.46.2).
2. **Scan**: run the scan_grib Stage-1 builder for **2026-05-13 00z** across
   **all 85 forecast hours** (the deflated store's `step`/`valid_time` coords
   need the full forecast structure — single-file scan is NOT sufficient for
   the template, only for the §4 smoke test). This is the ~2 hr / 9-worker
   Coiled job (`99o-…ipynb` is the parallel front-end; or run
   `ecmwf_three_stage_multidate.py` Stage-1 path on a big VM).
   Prereq for the notebook: reassemble `utils.py` (≈ `ecmwf/dev-test/fmrc_utils.py`)
   and `dynamic_zarr_store.py`; `coiled-data.json` already in `ecmwf/`.
3. **Per-member deflated stores** → `gs://gik-ecmwf-aws-tf/v2ecmwf_fmrc/ens_{NN}/
   ecmwf-2026051300-{member}-rt000.par` (50 members, no `ens_control`).
4. **Package** (manual — no repo script does this):
   `tar czf gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz gik-fmrc/v2ecmwf_fmrc/`.
5. **Upload** to HF template repo `E4DRR/grib-index-kerchunk-templates`
   (needs HF write token).
6. **Patch** `run_lithops_ecmwf.py` (§2.2), rebuild Cloud Run runtime image
   (template baked in), redeploy (`cloudbuild.yaml` + `lithops runtime
   deploy`), per `SETUP_NEW_MACHINE.md §5`.
7. **Reprocess** 2026-05-12 06z → present (all run hours) via the gap-backfill
   driver **with the cleanup-hang watchdog** ([[ecmwf-lithops-cleanup-hang]]).
8. **Re-mirror** to HF `E4DRR/gik-ecmwf-par` (`upload_parquets_to_hf.py`,
   needs `ecmwf/coiled-data.json` + HF token).

## 4. Dry-run RESULTS (2026-05-17, 20260513 00z 0h) — decisive

**scan_grib works but is slow:** 8500 groups in **756 s (~12.6 min) for ONE
file** → 85 files sequential ≈ 18 h ⇒ Coiled 9-worker parallelism is mandatory
(matches the historical ~2 hr figure).

**`ecmwf_util.py` member identification is BROKEN (dead code), not the builder:**
all 8500 scan_grib groups have **no `number` in `attrs`** →
`identify_ensemble_members()` → 0 members → empty tree (1 ref). The real,
working Stage-1 is the **idx-position-join** in `utils_ecmwf_step1_scangrib.py`
/ `fmrc_utils.py` (`ecmwf_filter_scan_grib`: `for i,group in enumerate(scan_grib)
… idx_mapping[i]`). Plan targets that path; ignore `ecmwf_util.py`'s
attrs-based member logic.

**Two concrete 50r1 breakages confirmed in the idx-join builder (index-only
check, no scan needed):**
1. `ecmwf_idx_unique_dict` (`utils_ecmwf_step1_scangrib.py:143`) hardcodes
   `pressure_levels=['50'…'1000']` (13, **no '10'**). 50r1 source HAS 10 hPa →
   builder would **silently drop the entire 10 hPa level**. Fix: add `'10'`.
2. `ecmwf_duplicate_dict_ens_mem:178` `np.insert(np.arange(1,51),0,-1)` (51,
   incl. phantom control -1) and `organize_ensemble_tree` hardcodes
   `numbers=np.arange(-1,50)`, `shape:[51]`, `chunks:[51]` for the `number`
   coord. 50r1 = members 1..50 only. Fix: 50 members, `np.arange(1,51)`,
   `shape/chunks:[50]`, drop control slot.

scan_grib + idx-join structurally fine otherwise (50 members, number 1..50,
no 0). Logs: `logs/gap_2026/dryrun_50r1.log`.

## 5. Risks

- **Step aggregation**: template needs all 85 steps' structure; verify the
  deflated store's `step`/`valid_time` arrays are full-length (85) after a
  multi-file build, not just 0h.
- **10 hPa coordinate correctness** in the zarr skeleton (chunk shape / level
  array) — validate by opening a reprocessed 50r1 parquet as xarray.
- **Coiled cost/time** ~2 hr, 9 workers, paid (`gcp-sewaa-nka`).
- **Cleanup-hang** during the reprocess backfill — watchdog mandatory.
- **CLAUDE.md is wrong** about the builder ([[ecmwf-template-provenance]]) —
  do not follow it; fix it as part of this work.
- Old 49r tar.gz must remain on HF for historical 49r re-backfills.
