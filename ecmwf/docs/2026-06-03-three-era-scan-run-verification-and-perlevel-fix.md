# Three-era ECMWF scan_grib run — execution, verification, and how it fixes the per-level bug

**Date:** 2026-06-03
**Branch:** `ecmwf-50r1-template`
**Scope:** the Stage-1 (Coiled `scan_grib`) preprocessing run for every ECMWF
template era, how the output was verified, the "sequential-index level"
property of the dumps and how the downstream realigner reconciles it, and
how this whole chain closes the per-pressure-level bug (incl. the **missed
13-level era**) in the current operational parquet.

---

## 1. What was run

Three Coiled drivers (`ecmwf/dev-test/ecmwf_{0p4,49r1,50r1}_coiled_preprocessing.py`),
one per schema era, each scanning every forecast-hour GRIB and writing a
per-(stream, hour) all-member dump to GCS.

| Era | Driver / Coiled fn | Ref date | Tasks | GCS prefix |
|---|---|---|---|---|
| 0.4°-beta (9-level) | `func-ecmwf-0p4` | 20230601 | 85 (enfo) | `gs://gik-ecmwf-aws-tf/fmrc/scan_grib20230601/` |
| 49r1 0.25° 13-level | `func-ecmwf-49r1` | 20250515 | 85 (enfo) | `…/scan_grib20250515/` |
| 50r1 0.25° 14-level | `func-ecmwf-50r1` | 20260513 | 170 (85 enfo + 85 oper) | `…/scan_grib20260513/` |

Each dump file: `e_sg_mdt_{date}_{stream}_{hour}h.parquet`. Total **340** dumps.

### 1.1 Environment integration (what it took to run on `gik-coiled-v6`)

The drivers run a local Coiled **client** that submits to the prebuilt
**`gik-coiled-v6`** container (`gik-coiled:v4`, built 2024-10-30). Four
issues had to be reconciled:

1. **Client↔cluster version skew.** Latest `coiled`/`distributed` couldn't
   talk to the older scheduler (`Scheduler.identity() got an unexpected
   keyword argument 'n_workers'`). Fixed by a version-matched client env
   (`micromamba … python=3.12 distributed=2024.10.0 dask=2024.10.0`).
2. **No `kerchunk._grib_idx` on the worker.** The image ships **kerchunk
   0.2.6** (no `_grib_idx`), so the newer `fmrc_utils.py` refactor can't
   import there. Switched `scan_one` to the image's **baked-in**
   `dynamic_zarr_store._map_grib_file_by_group` (the version-matched scan
   primitive).
3. **403 on the public bucket.** The image bakes in stale AWS creds, so
   `scan_grib` made a *signed* request → `PermissionError: Forbidden`.
   Forced anonymous S3 via `fsspec.config.conf["s3"]={"anon":True}` (the
   primitive opens the URL with no storage_options).
4. **No GCS key on the worker.** Shipped `coiled-data.json` to workers via
   `Client(cluster).upload_file(...)`; the worker resolves it from its
   `local_directory`.

These are committed in `6e215d7` (all three drivers). The fixes are
identical across eras; only the stream set (enfo vs enfo+oper) and Coiled
function name differ.

---

## 2. How it was verified

### 2.1 Completeness + size sanity (GCS)

Checked the **full expected `(stream, hour)` filename set** per era (not
just a count), plus sizes:

| Era | present / expected | missing | size range | zero/partial |
|---|---|---|---|---|
| 20230601 | 85 / 85 | 0 | 74–79 KB | none |
| 20250515 | 85 / 85 | 0 | 190–199 KB | none |
| 20260513 | 170 / 170 | 0 | 14.5 KB ↔ 212 KB | none |

The 50r1 bimodal sizes are correct: 85 small **oper** (1 control member)
+ 85 large **enfo** (50 perturbed). The Coiled dashboard showed transient
errors (6 on 50r1, 1 on 49r1) that **retried to success** — completeness
confirms no gaps remain.

### 2.2 Content sanity (row counts vs `.index`, columns)

For one dump per era, row count was compared to the live `.index` message
count, and columns checked:

| Dump | rows = `.index` msgs | cols present |
|---|---|---|
| 0p4 enfo 0h | 4233 = 4233 ✅ | ✅ |
| 49r1 enfo 0h | 8007 = 8007 ✅ | ✅ |
| 50r1 enfo 0h | 8500 = 8500 ✅ | ✅ |
| 50r1 oper 0h | 187 = 187 ✅ | ✅ |

Exact row-count match = **no messages dropped during the scan**. Columns
include `varname, typeOfLevel, level, step, valid_time, uri, offset,
length` — the byte-range references plus the metadata the realigner needs.

---

## 3. The "sequential-index level" property — and how it's reconciled

A subtle but important observation from the content sanity:

- In **enfo (multi-member)** dumps, the `isobaricInhPa` `level` column is a
  **sequential index** `[0, 1, 2, 3, …]`, **NOT** the real hPa value.
- In the **oper (single control member)** dump, `level` is the real hPa
  `[10, 50, 100, 150, …]`.

### Why

This is `kerchunk 0.2.6`'s `_extract_single_group` behavior on a
many-member file: with the ensemble `number` dimension present, the level
coordinate is collapsed onto a sequential axis rather than carrying the
isobaric value. (Single-member `oper` has no member axis to collapse, so
the true hPa survives.) This is exactly the "51 sequential `N.0` slots"
pattern previously identified in the legacy template producer.

### Why it does NOT break the template

The Stage-2 **realigner** (`ecmwf_par_to_ensemble_members.py`,
`create_member_groups_from_index`) does **not** trust the dump's `level`
column for pl rows. It recovers the **true** level from the **`.index`
`levelist`**, keyed by message position:

```
dump row idx (1-based, from enumerate(scan_grib, start=1))
   ↔ .index line idx-1 (0-based)            # off-by-one reconciled
   → levtype/levelist from that .index line
   → level = int(levelist) if levtype=='pl' else dump level
```

So whether the dump's `level` is `0`, a sequential index, or a real hPa is
**irrelevant** — the realigner overrides pl levels with the authoritative
`.index` value. This was proven by `gate_step2_realigner.py`, which forces
`level=0` in a synthetic dump and still recovers all 9 / 13 / 14 levels
(8/8 across every era + both streams; log:
`ecmwf/dev-test/gate_step2_realigner_validation.log`).

> Net: the scan dumps are **complete and correct as scan output**; the
> per-level correctness is established downstream from the `.index`, by
> design. Re-scanning with a newer kerchunk is **not required**.

---

## 4. How this closes the per-level bug (and the missed 13-level era)

### 4.1 The operational bug (what cGAN reported)

The current operational parquet collapses every pressure level of a
variable into **one** key per `(var, step, member)` —
`step_NNN/{var}/pl/{member}/0.0.0` — whose byte-range points at whichever
isobaric message is *first* in the source GRIB for that step. Because
ECMWF reorders messages across steps, `gh/pl` silently mixes 300/400/1000
hPa across lead times. The 5 pl channels the Xu et al. (2026) EP-cGAN port
needs (`u/v`, `ub/vb`, `gh`) are unrecoverable → the port falls back to
surface-only inputs.

Two parts to the fix (both required for whole-archive consumers):

1. **Runtime (`59b89dd`)** — `run_lithops_ecmwf.py` now emits per-level
   chunk-ref keys `var/pl/{hPa}/{member}/0.0.0`. ✅ done earlier.
2. **Template** — the zarr metadata skeleton must have a per-level
   structure for those keys to land in. This is the realigner +
   per-era template rebuild — the subject of this run.

### 4.2 The missed 13-level era

The original 49r1 plan used reference date **`20240529`**, which is a
**9-level** date. Empirically the 49r1 era is **not homogeneous**: it
publishes **9** pl levels before **2025-01-14 06z** and **13** levels
(`+100,150,400,600 hPa`) after — through 2026-05-12 00z. A template built
from a 9-level date **orphans the 4 extra levels for every ≥2025-01-14
date**, including the cGAN-critical MAM 2025 / MAM 2026 windows.

→ The 49r1 template is therefore rebuilt from a **13-level** reference
date (**`20250515`**), a safe superset (9-level dates just leave those 4
levels empty). This is why the 49r1 scan above targets `20250515`, not the
legacy `20240529`.

Combined with the 0.4°-beta and 50r1 eras, the archive needs **four**
per-era templates (0.4°/9-lvl, 0.25°/9-lvl, 0.25°/13-lvl, 0.25°/14-lvl),
not the single legacy one — see CLAUDE.md "ECMWF spans FOUR template eras"
and `ecmwf/docs/2026-06-02-ecmwf-era-variable-inventory.md`.

### 4.3 The realigner fixes (Option A, validated)

`ecmwf_par_to_ensemble_members.py:create_member_parquet` /
`create_member_groups_from_index` (commit on this branch):

- **Per-member assignment** by `.index` `number` (no more replicate-to-all).
- **Per-level keys** `var/isobaricInhPa/{hPa}` via the `.index` `levelist`
  recovery above.
- **Real 0.25° / 0.4° grid shape** (`[1,721,1440]` / `[1,451,900]`),
  replacing the legacy `[1,181,360]` 1° placeholder.

---

## 5. Next: realigner (Stage 2) per era

For each era's GCS dumps, run the realigner to produce per-(member, hour)
`rt` stores, then package into the per-era template tar.gz and upload to
HuggingFace. **Starting with the 49r1 13-level era (`20250515`)** — the
cGAN blocker — then 0.4°-beta and 50r1.

Chain per era:
```
GCS dumps  →  ecmwf_par_to_ensemble_members.py  →  {H}h/{member}.par (51 or 50+1 members)
           →  package to gik-fmrc/v2ecmwf_fmrc/{member}/ecmwf-{date}-{m}-rt{hhh}.par
           →  tar czf gik-fmrc-v2ecmwf_fmrc-<era>.tar.gz
           →  upload to HF E4DRR/grib-index-kerchunk-templates
```
