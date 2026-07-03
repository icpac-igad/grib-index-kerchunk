# Identified: the actual producer of the legacy ECMWF GIK template

**Date:** 2026-05-30
**Branch:** `ecmwf-50r1-template`
**Related docs (all sibling):**
  - `2026-05-17-50r1-template-investigation.md` — original investigation (commit `b1d96b7`)
  - `2026-05-17-50r1-template-rebuild-plan.md` — 50r1 chain
  - `2026-05-29-per-level-keys-fix-and-template-implications.md` — code fix + why rebuild needed
  - `2026-05-29-49r1-perlevel-reprocess-plan.md` — 49r1 chain

## Why this MD exists

Three sources gave conflicting accounts of who built
`gik-fmrc-v2ecmwf_fmrc.tar.gz` on HuggingFace (the production 49r1
template, ref date `20240529`):

| Source | Claimed producer |
|---|---|
| `ecmwf/README.md` line 23 | `ecmwf_index_preprocessing.py` (via `build_idx_grib_mapping`) |
| `2026-05-17-50r1-template-investigation.md` | notebook `99o-coiled-function-ecmwf-scan_grib_store_fmrc.ipynb` → `ecmwf_par_to_ensemble_members.py` (realigner) |
| What `4ca1c21` patched as "the aggregator" | `fixed_ensemble_grib_tree` in `utils_ecmwf_step1_scangrib.py` |

Gate A0 (`ecmwf/dev-test/gate_a0_legacy_template.py`, commit `59b89dd`)
pulled the live tar.gz and showed the rt000.par structure is
**neither** what `ecmwf_index_preprocessing.py` writes (`ecmwf-time-…`
prefix, `.parquet` extension) **nor** what `fixed_ensemble_grib_tree`
produces (Gate B1 returned 2 keys for what should have been hundreds).

This MD records which source was right.

## The actual producer

**`ecmwf/dev-test/ecmwf_par_to_ensemble_members.py`**, function
`create_member_parquet(self, member_groups)` at **lines 265–329**, with
the file written by `save_parquet` at line 331 (called from
`process_forecast_hour` at lines 440–441).

The investigation MD was right about the chain (notebook → realigner);
it just under-described the realigner. The realigner doesn't merely
"split by member" — it **assembles the consolidated zarr store** that
ends up in the tar.gz.

## Why the legacy template looks the way it does

Three discoveries from reading the producer code:

### 1. The realigner deliberately bypasses `grib_tree`

Line 272–273 contains an explicit comment:

```python
# Create a simple zarr store structure without using grib_tree
# since we don't have the complete scan_grib format
```

`grib_tree` and `strip_datavar_chunks` are **imported** at lines 29–30
but never called. The producer hand-rolls a stub-style zarr store
instead.

### 2. The `.zarray` shape is a hardcoded placeholder

Lines 309 and 315:

```python
'chunks': [1, 181, 360],  # Example chunking
'shape':  [1, 181, 360],  # Example shape
```

This is why Gate A0 saw `[1, 181, 360]` (1° resolution) in the live
template's `u/isobaricInhPa/0.0/.zarray`, despite the source GRIB being
0.25° (721 × 1440). The producer never derived the real grid shape;
it just stamped a placeholder.

### 3. The key construction at line 296 IS per-level-aware

```python
level = group.get('level', 0)
if level_type == 'surface':
    key = f"{varname}/surface"
else:
    key = f"{varname}/{level_type}/{level}"
```

So the producer **would** emit per-(member, level) keys — IF the
upstream code feeding `member_groups` provided correct `level`
values. The fact that Gate A0 observed 51 sequential indices
`0.0..50.0` per pl variable (not 51 members × 13 levels = 663
slots) means the upstream code is pre-collapsing (member, level)
into a single sequential axis before the realigner sees the groups.

Where exactly that upstream collapse happens has not been investigated
yet (the user scoped this MD to producer identification only).

## Where the per-level fix belongs in the rebuild chain

| Layer | Status | What needs to happen |
|---|---|---|
| **Coiled scan dump producer** (`fmrc_utils.s3_ecmwf_scan_grib_storing` + raw `scan_grib`) | Row-level per GRIB message — looks fine | Verify the dumps actually carry all 13/14 levels × 51 members, not pre-collapsed |
| **Realigner upstream** (whatever builds `member_groups` from dumps inside `process_forecast_hour`) | **Likely buggy** — produces sequential-index `level` values | Trace and fix: pass through the real isobaric values + keep member identity separate |
| **`create_member_parquet`** lines 290–296 | Per-level-aware, would pass through correctly if upstream is fixed | No change needed if upstream is fixed |
| **`create_member_parquet`** lines 309/315 (`[1, 181, 360]` stub) | **Buggy** — placeholder shape | Derive shape from actual GRIB metadata (721 × 1440 for spatial dims; add a `level` axis sized to the number of pl levels) |
| **`fixed_ensemble_grib_tree`** in `utils_ecmwf_step1_scangrib.py` | **NOT** the producer — patches in `4ca1c21` + `c78e449` apply to dead/unused code for this purpose | Stop spending effort here (per CLAUDE.md it "was never used to build the production template") |

## Two repair options for the 50r1 rebuild

### Option A: Patch `create_member_parquet` in place

- Replace the hardcoded `[1, 181, 360]` chunk/shape with values derived
  from actual GRIB metadata in `member_groups`.
- Fix the upstream code that fills `member_groups` so `level` carries
  the true isobaric value and members are tracked on a separate axis.
- Add a `level` coordinate dimension to the per-variable `.zarray`
  for pl groups.
- ~50–100 LOC. Keeps the realigner's existing architecture.
- Risk: other stub-style assumptions in the file may surface; the
  whole "without using grib_tree" framing is fragile.

### Option B: Replace with `grib_tree` + `strip_datavar_chunks`

- Rewrite `create_member_parquet` to call the kerchunk helpers that
  are already imported (lines 29–30) but unused.
- This is the pattern GEFS uses per CLAUDE.md and what kerchunk's
  upstream maintainers intended for this use case.
- The "without using grib_tree" comment in the realigner suggests
  this was a known-quick workaround; reverting it returns the code
  to a maintainable state.
- ~150 LOC delta, but most of it is replacing custom logic with
  kerchunk API calls.
- Risk: changes the rt000.par shape further (still incompatible with
  legacy template — same as the per-level fix already would be).

## Impact on prior plan/synthesis MDs

The 50r1 rebuild plan (§3 of `2026-05-17-50r1-template-rebuild-plan.md`)
and the per-level synthesis MD (`2026-05-29-per-level-keys-fix-…md`)
reference `fixed_ensemble_grib_tree` as the aggregator the rebuild
runs through. That's wrong — the rebuild actually runs through
`create_member_parquet` in the realigner. Before any paid Coiled
preprocessing run, those plan MDs should be revised to:

  - Point at the realigner as the actual template producer.
  - Drop assertions that `fixed_ensemble_grib_tree` ever fed the
    template tar.gz.
  - State explicitly which repair path (A or B above) will be applied
    to the realigner before Step 1 runs.

The runtime fix (`59b89dd`) is unaffected by this finding — it
patches a parallel code path (`run_lithops_ecmwf.py` inline copies of
parse_grib_index + create_references_from_index) that handles fresh
`.index` files at runtime, never touching the realigner.

## What this finding does NOT decide

- Whether Option A or Option B is the right repair path. Deferred.
- Whether the upstream `member_groups` builder is actually collapsing
  (member × level) or just mis-labeling levels. Needs a small trace
  (~30 min, no scan_grib needed) before A/B is chosen.
- Whether the 49r1 reprocess plan stays valid or needs revising.
  Likely needs revising — same realigner is used per the chain.

## Next concrete action (when user resumes)

1. Trace `process_forecast_hour` upstream of line 440 to see how
   `member_groups` is built; locate the actual (member, level) collapse.
2. Pick A or B based on what step 1 reveals.
3. Update the rebuild plan MDs to reference the correct producer.
4. Then — and only then — run the paid Coiled preprocessing (Step 1 of
   the 50r1 or 49r1 plan), since the rebuild is now correctly targeted.
