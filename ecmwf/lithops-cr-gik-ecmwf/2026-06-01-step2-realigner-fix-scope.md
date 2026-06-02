# Step 2 realigner fix — scope (closes the producer MD's "next action #1")

**Date:** 2026-06-01
**Branch:** `ecmwf-50r1-template`
**Scope target:** `ecmwf/dev-test/ecmwf_par_to_ensemble_members.py`
(the Step 2 "realigner" = the actual template producer, per
[`2026-05-30-template-producer-identified.md`](2026-05-30-template-producer-identified.md)).

## Why this MD exists

The producer-identification MD (`2026-05-30`) found *which* code builds
the template tar.gz and sketched two repair options (A: patch in place;
B: rewrite on `grib_tree`), but explicitly deferred the trace:

> "Where exactly that upstream collapse happens has not been
> investigated yet… Next concrete action #1: Trace `process_forecast_hour`
> upstream of line 440 to see how `member_groups` is built; locate the
> actual (member, level) collapse."

This MD records that trace, names the exact defects with line numbers,
and pins a concrete fix. It supersedes the Options A/B framing only in
that it now has enough evidence to **recommend a refined Option A**.

## The chain, traced (no scan_grib needed)

```
process_forecast_hour              L376  (driver, one scan-dump parquet = one forecast hour)
  ├─ load_parquet_groups           L112  → (df, grib_uri)        # df = scan dump rows
  ├─ parse_index_file              L156  → idx_to_member {idx:int → member:int}
  ├─ create_member_groups_from_index L213 → member_dict {member → [group,…]}
  └─ _process_single_member        L437
        └─ create_member_parquet   L265  → zarr_store            # per-member, keyed var/levtype/level
        └─ save_parquet            L331  → {hour}h/{member}.par
```

## Defect 1 — member mis-assignment (`create_member_groups_from_index`, L213–254)

The function iterates dump rows but **does not** assign row `idx` to its
member. It computes the *set* of members and replicates **every row to
every member**:

```python
for row_idx, row in df.iterrows():
    unique_members = set(idx_to_member.values())   # L228
    for member_num in unique_members:              # L230  ← replicate-to-all
        group = { …, 'level': row.get('level', 0), 'ens_number': member_num }
        member_dict[member_num].append(group)
```

`idx_to_member` (the entire reason `parse_index_file` exists) is reduced
to its value-set and the per-row mapping is thrown away. Every member
ends up with a copy of all ~8 500 messages instead of its own ~170.

**Correct shape:** the dump is indexed by `idx` (`_map_grib_file_by_group`
does `.set_index("idx")`), and `.index` line *i* describes GRIB message
*i*. So row `idx` belongs to exactly `idx_to_member[idx]`:

```python
for idx, row in df.iterrows():
    member_num = idx_to_member.get(idx)
    if member_num is None:
        continue
    member_dict.setdefault(member_num, []).append(build_group(row, idx, member_num))
```

## Defect 2 — level handling (CORRECTED 2026-06-01 after empirical check)

> **Correction.** The first draft of this MD (commit `a58b47a`) claimed
> the level value is "silently dropped" because it lives only in the
> `group["refs"]["isobaricInhPa/0"]` binary blob that the dump never
> carries. **That root cause is wrong for the current kerchunk.** An
> empirical scan + source read on 2026-06-01 (see "Empirical validation"
> below) shows the scan dump **does** carry the real isobaric level as a
> proper `level` column. Gate B1's "level is in a blob" was about *raw*
> `scan_grib` dattrs — but `_map_grib_file_by_group` does not store raw
> dattrs; it stores the output of `_extract_single_group`, which decodes
> that blob into a coordinate value.

What actually happens (verified):

- `_extract_single_group` → `extract_datatree_chunk_index(dt, …, grib=True)`
  → `extract_dataset_chunk_index(grib=True)` (kerchunk `_grib_idx.py`).
  With `grib=True` it renames the vertical coordinate to `level`
  (`cname = "level" if cname in ECCODES_VERTICAL_LEVELS`; `isobaricInhPa`
  ∈ that set — confirmed) and writes its **value** into the row:
  `coord_vals[cname] = cvar.to_numpy()[coord_index]`.
- So each dump row is `dict(varname=…, **attrs, level=<real hPa>, uri, offset, length, idx)`.
  `build_idx_grib_mapping` even indexes on `["varname","typeOfLevel","stepType","level","valid_time"]`,
  which only works because `level` is a real column.

Consequence for the realigner: `create_member_groups_from_index`'s
`row.get('level', 0)` (L238) and `create_member_parquet`'s
`key = f"{varname}/{level_type}/{level}"` (L296) **already pass the true
level through** when fed a current-kerchunk dump. So the per-level
collapse seen in the *legacy* HF template is an artifact of the **older
kerchunk** used for that 2024 build (where the dump lacked a usable
`level`), not of the realigner logic as it stands. A rebuild on current
kerchunk + the Defect-1 member fix emits per-level keys without any
change to L296.

### Hardening: cross-check level against the `.index` (authoritative)

Even though the dump carries `level`, the `.index` carries `levelist`
**redundantly and authoritatively** — empirically present on 100% of pl
lines and absent on 100% of non-pl lines (validated below). Since the
realigner **already reads the `.index`** in `parse_index_file`
(L156–211), it costs nothing to capture `levelist` there and use it as
the source of truth (and a guard against a dump that was built by an old
kerchunk with a missing/zero `level`):

```python
# in parse_index_file, per line:
idx_meta[idx] = {
    'member':   ens_number,                         # existing logic
    'levtype':  data.get('levtype'),                # 'pl' | 'sfc' | 'sol'
    'levelist': data.get('levelist'),               # real hPa for pl, else None
}
```

Then `create_member_groups_from_index` sets
`level = int(levelist) if levtype=='pl' and levelist else row.get('level', 0)`.
This makes the level correct regardless of which kerchunk produced the
dump, keeps the whole thing inside the cheap Step 2 realigner, and
triggers **no** paid re-scan. (`number`/member identity is likewise in a
`number/0` blob in the raw group, which is exactly why the realigner
already takes members from the `.index` rather than the dump — see
Defect 1.)

## Defect 3 — resolution stub (`create_member_parquet`, L309/315)

```python
'chunks': [1, 181, 360],  # Example chunking   ← 1°, wrong
'shape':  [1, 181, 360],  # Example shape
```

ECMWF 0p25 is 721 × 1440. Replace with the real spatial geometry. Two
sub-cases:

- **sfc/sol** vars: `shape/chunks = [1, 721, 1440]`.
- **pl** vars: the per-level keying already gives one zarr group per
  (var, level), so each pl group is still a 2-D field
  `[1, 721, 1440]`. A separate `level` *dimension* inside one `.zarray`
  is **not** required by the runtime — the consolidated store addresses
  levels via distinct keys (`var/pl/{hPa}/…`), matching `59b89dd`'s
  runtime key shape. (If a future consumer wants a single 3-D pl array,
  that's a separate enhancement, not part of unblocking cGAN.)

Source the `721/1440` from GRIB metadata where available; a constant is
acceptable as long as a guard asserts the source grid is 0p25 (the
template is cycle/grid-specific anyway).

## Recommended repair: refined Option A

Option A (patch in place) over Option B (rewrite on `grib_tree`),
because:

- The level + member values are recoverable from the `.index` the
  realigner already reads — Option A becomes ~40–60 LOC across three
  functions (`parse_index_file`, `create_member_groups_from_index`,
  `create_member_parquet`), no new scan format, no re-scan.
- Option B (`grib_tree` + `strip_datavar_chunks`) still has to solve the
  blob-decoding of level/number itself, and `c78e449` showed
  `fixed_ensemble_grib_tree` returns 2 keys where hundreds are expected
  — the `grib_tree`/MultiZarrToZarr path is currently broken for this
  data, so Option B is higher-risk for the same outcome.

### Concrete change-set (Option A)

1. `parse_index_file` (L156) → return `idx_meta` dict carrying
   `{member, levtype, levelist}` per idx (superset of today's
   `idx_to_member`).
2. `create_member_groups_from_index` (L213) → assign row `idx` to
   `idx_meta[idx]['member']` (fixes Defect 1, with the off-by-one
   reconciliation below); set `level` from the `.index` `levelist` for
   pl rows, falling back to the dump's `level` column (Defect-2
   hardening).
3. `create_member_parquet` (L309/315) → derive `[1,721,1440]` per var
   (fixes Defect 3); L296 key construction unchanged.

## Empirical validation (2026-06-01) — the three live checks, now run

Run in a throwaway `uv` venv against the live 49r1 reference file
`s3://ecmwf-forecasts/20240529/00z/ifs/0p25/enfo/20240529000000-0h-enfo-ef.{grib2,index}`
(anonymous S3). Scripts in `/tmp/gate_*.py` during the session.

1. **Dump carries a real `level` column — CONFIRMED (overturns the
   original Defect 2).** `extract_dataset_chunk_index(grib=True)` emits
   `level`; `isobaricInhPa ∈ ECCODES_VERTICAL_LEVELS`. Empirically, the
   real scan's `isobaricInhPa/0` coordinate for the first pl message
   decodes (`base64:` → `<f8`) to **850.0 hPa** — i.e. the level value is
   present and correct in the scanned group, not lost.
2. **`.index` `levelist` coverage — CONFIRMED.** 5 559 messages in the
   0h file; `levtype` counts `sfc=1428, pl=4131`; **pl lines missing
   `levelist`: 0**, **non-pl lines with `levelist`: 0**. The
   `levtype=='pl'` guard is safe. (51 members incl. control `-1`, uniform
   per-member counts.)
3. **Off-by-one — CONFIRMED REAL.** `.index` line **528** (0-based) is
   `u @ 850 hPa`; the scan's group **529** (1-based, `enumerate(..., start=1)`)
   decodes to 850.0 hPa — same message, indices differ by one. The
   `idx_meta` join in `create_member_groups_from_index` must map dump
   `idx` (1-based) to `.index` line `idx-1` (0-based), or every row
   shifts by one member.

Net effect on the fix: the level-recovery logic stays (now justified as
authoritative cross-check + old-kerchunk guard, not as the sole source),
and the off-by-one becomes a hard requirement rather than a caution.

## Gates (free) before any paid Coiled run

- **Structural gate — BUILT and PASSING:** `ecmwf/dev-test/gate_step2_realigner.py`
  runs the real patched realigner against a faithful dump (pl `level`
  zeroed) + the live `.index`, and asserts: member count == `.index`,
  per-level pl keys, **full** published-level coverage, no level-0
  collapse, `[1,721,1440]` shape, and no replicate-to-all. Parameterised
  `--date/--run/--hour/--stream`. Results (2026-06-01), all **8/8**:
  - `--date 20240529 --hour 0  --stream enfo` → 49r1 9-level, 51 members
  - `--date 20250515 --hour 12 --stream enfo` → 49r1 **13-level**, 51 members
  - `--date 20260513 --hour 12 --stream enfo` → 50r1 **14-level**, 50 perturbed
  - `--date 20260513 --hour 12 --stream oper` → 50r1 **14-level**, control only
- Only after the structural gate is green does Step 1 (the paid Coiled
  preprocessing) become worth running.

## Pressure-level coverage is DATE-DEPENDENT within 49r1 (new finding)

Empirically (anon `.index` scan across the archive, 2026-06-01) the
open-data ENS pl set is **not** constant across the 49r1 era:

| Window | pl levels | set |
|---|---|---|
| 2024-03-01 → **2025-01-14 00z** | **9** | 50, 200, 250, 300, 500, 700, 850, 925, 1000 |
| **2025-01-14 06z** → 2026-05-12 00z | **13** | + 100, 150, 400, 600 |
| 2026-05-12 06z → present (50r1) | **14** | + 10 |

The 9→13 expansion is pinned to **2025-01-14 06z** (00z that day = 9,
06z = 13). Every pl variable (`d gh q r t u v vo w`) carries the full
set for its window; the level set is uniform across members and forecast
hours within a date.

**This invalidates two assumptions in the existing plans:**

1. **The 49r1 reprocess plan's reference date `20240529` is a 9-level
   date.** A template built from it has `.zarray`/coords for only 9
   levels. Used at runtime for a **13-level** date (≥2025-01-14 06z —
   which includes all of MAM 2025 and MAM 2026, the cGAN-critical
   windows), the 4 extra levels' chunk-refs (100/150/400/600 hPa) would
   be **orphaned** — exactly the per-level breakage this whole effort is
   fixing, just one level up. **Fix: build the 49r1 template from a
   13-level reference date (e.g. `20250515`), which is a superset.** A
   13-level template used for a pre-2025-01-14 (9-level) date is benign —
   those 4 levels simply carry no chunk-refs (empty), because the older
   `.index` never lists them, so nothing is orphaned.
2. **The "13→14, +10 hPa" framing in the 50r1 plan** is only correct
   relative to *late* 49r1. Relative to the template's stale ref era
   (2024, 9 levels) the jump is 9→14. Immaterial to 50r1 (which has its
   own 14-level ref `20260513`), but the 49r1 plan must not describe its
   own era as a single 13-level block.

`gate_b1_real_scangrib.py` hardcodes `EXPECTED_LEVELS_49R1` = the 13-level
set and asserts equality — that assertion is **false for any 49r1 date
before 2025-01-14 06z** and should be made window-aware or dropped.

## Effort & risk

- **Effort:** ~40–60 LOC, one file, three functions. ~half a day incl.
  the three live checks and the structural gate.
- **Risk:** low-moderate. The off-by-one (check #3) is the only sharp
  edge. Everything else is additive (richer `.index` parse, real grid
  shape) and validated by a free structural gate before any spend.

## Sibling-MD updates (per producer MD §"Impact") — DONE 2026-06-01

- `2026-05-29-49r1-perlevel-reprocess-plan.md`: GATE A rewritten to point
  at the automated `gate_step2_realigner.py`; reference date corrected
  from the 9-level `20240529` to a 13-level date (`20250515`); the
  "13 levels" facts corrected to the 9/13 date-dependent reality.
- `2026-05-17-50r1-template-rebuild-plan.md`: Step-2 GATE rewritten to the
  automated gate (both enfo + oper streams, 14-level), replacing the
  "diff vs legacy 20240529 control rt000" gate (wrong 1°/9-level baseline).
- `2026-05-30-template-producer-identified.md`: "Next action #1 (trace)"
  **done** (this MD); #2 (pick A/B) **decided** — refined Option A,
  **implemented** in `ecmwf_par_to_ensemble_members.py`; #3 (update plan
  MDs) **done**. #4 (run the paid Coiled scan) remains, now correctly
  targeted (13-level 49r1 ref + 14-level 50r1 ref).
- `gate_b1_real_scangrib.py`: its hardcoded 13-level `EXPECTED_LEVELS_49R1`
  assertion is false pre-2025-01-14 06z — flagged for window-awareness or
  removal (not yet changed).
