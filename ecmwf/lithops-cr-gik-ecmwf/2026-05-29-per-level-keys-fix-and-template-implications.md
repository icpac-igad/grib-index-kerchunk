# Per-Pressure-Level Keys — issue, fix, and the template implication

**Date:** 2026-05-29
**Branch:** `ecmwf-50r1-template`
**Code fix landed:** `4ca1c21`
**Downstream reports:** `cGAN_tutorial/example_notebooks/pytorch_cgan/`
  - `GIK_MAINTAINER_REQUEST.md`
  - `GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`

---

## 1. What the downstream reports said

The PyTorch EP-cGAN port for East Africa long-rains training found that the
ECMWF GIK parquets (`E4DRR/gik-ecmwf-par` on HuggingFace) emit **one `pl`
reference per `(variable, step)`** — single key shape:

```
step_NNN/{var}/pl/{member}/0.0.0
```

When they decoded the GRIB header that the byte-range under that key
resolves to, the `isobaricInhPa` value was different at every step,
because ECMWF reorders GRIB messages across forecast hours and the
parquet captured whichever message was *first* per (var, step).

The smoking gun (probed on `20260301-00z-control.parquet`):

| Lead step (h) | `gh/pl` level | `u/pl` level | `v/pl` level |
|---:|---:|---:|---:|
|  6 | 400  | 250 | 250 |
|  9 | 300  | 500 | 500 |
| 12 | 300  | 500 | 500 |
| 24 | 1000 | 250 | 250 |
| 27 | 1000 | 250 | 250 |

`gh @ 1000 hPa` at step >=24 is the orographic-surface height (~0 m),
stacked under the same key as `gh @ 300/400 hPa` (~9000 m). A consumer
that builds a `lead_time` axis from those gets a `gh` array mixing
three different physical quantities. Same logic for `u/v`: a "u-wind"
array that silently mixes 250 / 500 / 1000 hPa.

**Why it blocks the cGAN port:** Xu et al. (2026) EP-cGAN uses 5
pressure-level channels out of 11 (paper §2.b) — `u/v` at low-trop +
boundary-layer + `gh` at mid-trop. Those channels carry the Somali
jet (850–925 hPa), Turkana jet, 500/300 hPa trough/ridge, and vertical
wind shear signals — exactly what the paper's CSI improvements at
>50 mm/3 h depend on. Without per-level keys the cGAN port falls back
to surface-only inputs and cannot reproduce the paper.

The reports' explicit ask: emit one reference per `(var, step, level)`
with key shape

```
step_NNN/{var}/pl/{level_hPa}/{member}/0.0.0
```

---

## 2. Root cause — two collapse sites, same flaw

Two independent code paths in `grib-index-kerchunk` build the parquet
references, and **both** drop the level value, keeping only the
*type* of level:

| File | Site | What it kept | What it dropped |
|---|---|---|---|
| `ecmwf/utils_ecmwf_step1_scangrib.py` | `fixed_ensemble_grib_tree()` path builder (line ~465) | `stepType`, `typeOfLevel` (e.g. `instant/isobaricInhPa`) | `GRIB_level` (the actual hPa value) |
| `ecmwf/ecmwf_index_processor.py` | `parse_grib_index()` (line ~95) + `create_references_from_index()` (line ~155) | `levtype` (`"pl"` / `"sfc"`) | `levelist` (the hPa value in the JSON-line) |

The `.index` sidecars and the scan_grib output both contain the level
value — it was simply being thrown away during key construction.

---

## 3. What `4ca1c21` changed

Surgical, single-commit fix on `ecmwf-50r1-template`:

| Change | File | Effect |
|---|---|---|
| Path builder appends `GRIB_level` when `typeOfLevel == "isobaricInhPa"` | `ecmwf/utils_ecmwf_step1_scangrib.py` + `ecmwf/dev-test/` copy (kept byte-identical per `7bac214`) | scan_grib / template-build path emits one zarr group per (var, stepType, isobaricInhPa, **level**, member) |
| `parse_grib_index` exposes `level_value`; `create_references_from_index` emits `var/pl/{level_hPa}/{member}/0.0.0` for pl rows | `ecmwf/ecmwf_index_processor.py` | runtime / Lithops path produces correct per-level keys in the HF parquets |
| New `ecmwf/dev-test/validate_per_level_keys.py` | Index-only regression test | `uv run` checks both fixes against the real `20260513 00z f009` .index |

**Validation against live ECMWF 50r1 (`20260513 00z f009`):**

```
8500 index entries -> 6300 pl + 1800 sfc + 400 sol references
every pl entry now carries level_value (was 0% before)
every pl key has shape var/pl/{level}/{member}/0.0.0
u/v/gh/t/q/w/r/d/vo -> 14 distinct pl levels per member, 50 members
  (14 = the 13 pre-50r1 levels + the new 10 hPa added in 50r1)
sfc keys keep var/sfc/{member}/0.0.0 -- no consumer break outside pl
```

The MD's smoking-gun probe (gh@step24/27/30 collapsing to 1000 hPa)
is now impossible by construction: each level lives at its own key.

**Backwards compat:** `sfc` / `sol` keys are unchanged. `pl`-key
consumers MUST update — but since the old `pl` keys were physically
incoherent, breakage is the correct outcome (the report agrees).

---

## 4. The template implication — why the code fix is **not enough on its own**

This is the part that wasn't in the original reports and matters most
for shipping the fix end-to-end.

### What the template carries

The production template archive
`gik-fmrc-v2ecmwf_fmrc.tar.gz` (HF: `E4DRR/grib-index-kerchunk-templates`)
is **not** just text metadata — it is the serialized **zarr-store
skeleton** for the deflated reference store. The tar.gz holds per-member
per-hour `rt{hhh}.par` files, each containing:

- The `.zarray` for every variable (chunk shape, dims, dtype, fill value).
- The `.zattrs` (CF/GRIB attributes, coordinate references).
- The coordinate arrays (`latitude`, `longitude`, `step`, `valid_time`,
  `number`, etc.).
- The *key shape* under which data chunks are addressable.

`run_lithops_ecmwf.py` Stage 1 (`build_deflated_stores_from_template`)
loads one `rt000.par` per member and that **fixes the zarr structure**
for the whole forecast. Stage 2 then reads the fresh `.index` files
and inserts byte-range references into the existing chunk slots.

### The mismatch the code fix alone creates

The current `gik-fmrc-v2ecmwf_fmrc.tar.gz` was built on `2024-05-29`
with the **buggy** `fixed_ensemble_grib_tree` — so every `rt000.par`
inside it has:

- pl-variable `.zarray` shaped **without a `level` dimension**
  (e.g. `u.shape = [85, 721, 1440]` — step × lat × lon, no levels).
- ONE chunk slot per (var, step, member): `u/instant/isobaricInhPa/control/0.0.0`.

After the code fix, Stage 2 produces 14 keys per (var, step, member)
for pl variables: `u/instant/isobaricInhPa/{50,100,...,1000}/control/0.0.0`.

**These 14 keys have nowhere to land** in a deflated store whose
zarr metadata only provides one chunk slot per (var, step, member).
Result: the references either silently overwrite each other (last-write
wins — same wrong-level bug, different mechanism) or the zarr reader
rejects them as out-of-shape chunk addresses.

### Why patching at runtime won't work

A tempting shortcut is to "expand" the template's single-slot per
(var, step) into 14 slots in `build_deflated_stores_from_template` at
runtime. This won't work because:

- The `.zarray` is the contract that downstream readers (xarray,
  zarr-python, virtualizarr) use to compute chunk addresses. Faking a
  level dimension at runtime means rewriting every variable's
  `.zarray` *and* synthesising a `level` coordinate array consistent
  with the data — the work of a template rebuild, but spread across
  every backfill run.
- Coordinate variables in the template (`step`, `valid_time`,
  `number`) are written assuming the no-level structure. Adding a
  `level` axis requires emitting a new `level` coord array (the 14
  hPa values, ordering, attributes).
- It conflates "what the archive looks like" (template) with "where
  the bytes are" (Stage 2 byte-range insertion). The clean separation
  is what makes the GIK design work.

**Conclusion: the template must be rebuilt against the fixed
preprocessing.** No runtime band-aid is correct.

---

## 5. Roadmap to ship the fix end-to-end

The `ecmwf-50r1-template` branch already carries a documented template
rebuild plan in
[`2026-05-17-50r1-template-rebuild-plan.md`](2026-05-17-50r1-template-rebuild-plan.md)
(§3 "Build & deploy steps", commit `f746e3b`). With `4ca1c21` landed,
that chain now produces a **per-level-correct** template automatically
— no further plan changes needed. Order of operations:

1. **Run Coiled preprocessing** (`ecmwf_50r1_coiled_preprocessing.py
   --date 20260513 --run 00`) — 170 tasks (85 enfo + 85 oper), ~2 h,
   paid. Step 1 in `ecmwf/README.md`'s terminology; renamed from
   "coiled scan" 2026-05-29 to align with that. Uses
   `fmrc_utils.s3_ecmwf_scan_grib_storing` which is unaffected by the
   level fix (it dumps raw scan_grib output per message). User-only
   step, requires Coiled auth.
2. **Realign with `ecmwf_par_to_ensemble_members.py`** — per-member
   `rt000.par` files. Unaffected by the level fix (row-level reorg).
3. **Aggregate to per-member deflated stores** with the **fixed**
   `fixed_ensemble_grib_tree()`. This is where the per-level structure
   gets baked into each `rt{hhh}.par`: pl variables gain a `level`
   dimension, the chunk grid enumerates the 14 isobaric levels, the
   coordinate array `level = [10, 50, 100, 150, 200, 250, 300, 400,
   500, 600, 700, 850, 925, 1000]` is written into the store.
4. **Package** as `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` (new name,
   distinct from the 49r legacy tar.gz).
5. **Upload to HF** `E4DRR/grib-index-kerchunk-templates` (user-only;
   needs HF write token).
6. **Wire** `run_lithops_ecmwf.py` to the new template URL
   (`REFERENCE_DATE=20260513`, `TEMPLATE_URL` / `ECMWF_TEMPLATE_PATH`
   bumped), rebuild Cloud Run runtime image, redeploy.
7. **Re-bake HF parquets** for the 50r1 era (>= 2026-05-12 06z) by
   re-running the Lithops backfill with the new image. The
   `GIK_MAINTAINER_REQUEST.md` Request-2 gap (2026-04-08 onward)
   gets closed in this same backfill pass.

### A cheap gate before the paid scan

Steps 3 and 6 can be validated *without* re-running the ~2 h Coiled
scan: take any existing scan_grib dump (the old 49r template's
`rt000.par` or a single-hour dump from a development scan), feed it
through the **fixed** `fixed_ensemble_grib_tree`, and check:

- `u/instant/isobaricInhPa/.zarray` has a `level` axis.
- The deflated store has 14 chunk slots per (step, member) for pl vars.
- Coordinate `level` is `[10, 50, 100, ..., 1000]`.
- sfc/sol stores are byte-identical to the pre-fix output.

If that gate passes on existing dumps, the only risk left in the paid
scan is data availability — not correctness.

---

## 6. What about the existing HF parquets?

The current `E4DRR/gik-ecmwf-par` parquets (dates up to 2026-04-07)
were generated by `run_lithops_ecmwf.py` against the **buggy**
template + **buggy** runtime. Their pl keys are physically incoherent
for the reasons in §1. Treatment by date range:

- **Pre-50r1 dates (≤ 2026-05-12 00z)** — covered by the 49r template
  `gik-fmrc-v2ecmwf_fmrc.tar.gz`. Same per-level bug. Re-baking those
  requires also rebuilding the 49r template against fixed code
  (re-run scan_grib on a 49r reference date, e.g. 2024-05-29) **OR**
  accepting that 49r-era pl keys stay broken and only fix going forward.
  Decision deferred — coordinate with downstream consumers.
- **50r1 era (≥ 2026-05-12 06z)** — gets fixed by the rebuild roadmap
  above; the Lithops backfill re-bake produces correct per-level
  parquets in one pass.
- **The gap 2026-04-08 → present** (Request 2 of the maintainer MD)
  — closed by the same backfill run.

A `available_dates.json` manifest in the HF dataset (Request 2 bonus
ask) is a small follow-up — the backfill driver can write it as the
last step of each daily run.

---

## 7. Summary

- **Code fix landed** on `ecmwf-50r1-template`@`4ca1c21`. Two collapse
  sites patched; sfc/sol untouched; index-only validation green.
- **Template rebuild is required** because the current tar.gz bakes
  the old (no-level) zarr `.zarray` into every `rt000.par`. No
  runtime patch can compensate.
- **The rebuild plan we already have** (§3 of the 50r1 plan MD)
  produces a correct template now that the code is fixed — no plan
  changes needed.
- **A cheap dry-run gate** can validate the rebuild on existing dumps
  before the paid Coiled preprocessing run.
- The `GIK_MAINTAINER_REQUEST.md` Request-2 backfill gap closes
  automatically during the post-fix re-bake.

The next concrete action gated on the user: run the ~2 h paid Coiled
preprocessing when ready (`ecmwf_50r1_coiled_preprocessing.py --date
20260513 --run 00`),
and feed the dumps back here for the realigner + packaging + cheap
template-structure gate.
