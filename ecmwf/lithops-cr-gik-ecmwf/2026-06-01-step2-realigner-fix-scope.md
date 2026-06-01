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

## Defect 2 — level collapse (root cause confirmed via Gate B1)

`gate_b1_real_scangrib.py:82-83` already established the mechanism:

> "cfgrib's scan_grib does NOT put `GRIB_level` in dattrs; level lives in
> `group["refs"]["isobaricInhPa/0"]`" — a binary `<f8` blob.

Consequence: kerchunk's `_extract_single_group` (used by
`_map_grib_file_by_group` in `fmrc_utils.py:85`) surfaces
`typeOfLevel="isobaricInhPa"` as a column but **never the level value**.
So in the dump, pl rows have no usable `level`; `row.get('level', 0)`
returns `0` for all of them. In `create_member_parquet` (L296):

```python
key = f"{varname}/{level_type}/{level}"   # level==0 for every pl message
```

⇒ all 13 (49r1) / 14 (50r1) isobaric levels of a variable collapse into
one key `u/isobaricInhPa/0`. This is the exact per-level bug the cGAN
port reported. (`number`/member identity lives in the sibling `number/0`
blob and is likewise absent — which is *why* the realigner reaches for
the `.index` instead; see Defect 1.)

### Recovery source — no re-scan required

The realigner **already reads the `.index`** in `parse_index_file`
(L156–211), and ECMWF `.index` JSON-lines carry `levelist` and `levtype`
per message — the same fields the runtime fix `59b89dd` uses. So both
the member **and** the true level can be recovered from one `.index`
read, keyed by line index. Extend `parse_index_file` to return richer
per-idx metadata:

```python
# in parse_index_file, per line:
idx_meta[idx] = {
    'member':   ens_number,                         # existing logic
    'levtype':  data.get('levtype'),                # 'pl' | 'sfc' | 'sol'
    'levelist': data.get('levelist'),               # real hPa for pl, else None
}
```

Then `create_member_groups_from_index` stamps the recovered level onto
each row's group (overriding the dump's missing/0 value for pl), and
`create_member_parquet`'s existing L296 key construction — already
per-level-aware — emits `u/isobaricInhPa/250`, `…/500`, … correctly with
**zero change to L296 itself**.

This keeps the entire level fix inside the cheap Step 2 realigner. The
Coiled scan dump producer (Step 1) is **not** touched, so no paid
re-scan is triggered by this fix.

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
   `idx_meta[idx]['member']` (fixes Defect 1); stamp
   `level = levelist if levtype=='pl' else 0` onto the group
   (fixes Defect 2).
3. `create_member_parquet` (L309/315) → derive `[1,721,1440]` per var
   (fixes Defect 3); L296 key construction unchanged.

## What still needs a live check before coding (cheap, no paid scan)

These cannot be confirmed in this environment (kerchunk not installed,
no dump fixture present), and must be checked against one real dump +
one real `.index` before/while implementing:

1. **Dump columns** — confirm `df.columns` from one real
   `e_sg_mdt_*_enfo_*.parquet` actually lacks a usable pl `level`
   (expected: present as 0/NaN). `load_parquet_groups` already logs
   `df.columns.tolist()` — capture it.
2. **`.index` `levelist` coverage** — confirm every pl line has a
   `levelist` and sfc/sol lines do not (so the `levtype=='pl'` guard is
   safe). Trivially checkable on any S3 `.index`.
3. **idx alignment** — confirm the dump's `idx` index is 0-based and
   matches `.index` line order 1:1 (both derive from the same
   `scan_grib` enumeration; `_map_grib_file_by_group` uses
   `enumerate(scan_grib(...), start=1)` while `parse_index_file` uses
   `enumerate(f)` starting at 0 — **watch this off-by-one**; reconcile
   the index base before trusting the join).

Item 3 is the one real implementation hazard surfaced by the trace: the
dump is 1-based (`start=1`, then `.set_index("idx")`), the index map is
0-based. The join must align them or every row shifts by one member.

## Gates (free) before any paid Coiled run

- **Structural gate:** run the patched realigner on ONE existing scan
  dump + its `.index`; open `control/rt000.par`; assert pl `.zarray`
  shapes are `[1,721,1440]`, keys are `var/isobaricInhPa/{hPa}` across
  the full level set (13 for 49r1, 14 for 50r1), and member count is
  51 (49r1) / 51-with-oper (50r1). This is the GATE A / GATE B1 already
  described in the 49r1 and 50r1 plans — now it can actually pass.
- Only after the structural gate is green does Step 1 (the paid Coiled
  preprocessing) become worth running.

## Effort & risk

- **Effort:** ~40–60 LOC, one file, three functions. ~half a day incl.
  the three live checks and the structural gate.
- **Risk:** low-moderate. The off-by-one (check #3) is the only sharp
  edge. Everything else is additive (richer `.index` parse, real grid
  shape) and validated by a free structural gate before any spend.

## Sibling-MD updates implied (per producer MD §"Impact")

- `2026-05-17-50r1-template-rebuild-plan.md` and
  `2026-05-29-49r1-perlevel-reprocess-plan.md`: their GATE A / B1 steps
  now have a concrete pass condition (above). The "aggregator =
  `fixed_ensemble_grib_tree`" framing remains wrong; the producer is the
  realigner, fixed via Option A here.
- `2026-05-30-template-producer-identified.md`: its "Next concrete
  action #1 (trace)" is **done** (this MD). #2 (pick A/B) is **decided**:
  refined Option A. #3 (update plan MDs) and #4 (then run paid scan)
  remain.
