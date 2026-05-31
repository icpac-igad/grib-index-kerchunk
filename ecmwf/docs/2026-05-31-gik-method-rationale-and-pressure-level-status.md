# Why GIK? — and the per-pressure-level bug status

**Date:** 2026-05-31
**Branch:** `ecmwf-50r1-template`
**Audience:** new contributors trying to understand whether GIK is worth
the complexity, and engineers tracking the per-level-keys bug fix in
progress.

Two questions answered here:

1. **Why does the GIK method exist when simpler "fetch + decode +
   pickle" scripts can produce a realized dataset directly?** Concrete
   reference: `ea-aifs/s3_grib_pkl/s3_grib_pkl_input_aifsens.py`.
2. **What is the current status of the per-pressure-level keys bug
   reported by the cGAN port — why it was identified, what's been
   fixed, what's still open?**

---

## Part 1 — Why GIK exists

### The minimalist alternative: direct fetch → decode → pickle

`ea-aifs/s3_grib_pkl/s3_grib_pkl_input_aifsens.py` is a complete,
working ECMWF ENS input-state builder that **does not use GIK at
all**. Its method, in one paragraph:

> Read the tiny `.index` JSON-lines sidecar (~2 MB) next to each ECMWF
> GRIB on `s3://ecmwf-forecasts/`, filter to the exact ~89 messages
> one AIFS ensemble member needs (8 sfc + 4 sfc-constant + 2 soil + 6
> pl × 13 levels), HTTP byte-range fetch only those messages, decode
> each with `gribberish`, regrid to N320, pickle into a single
> `(2, 542080)` input state.

This produces a **realized** dataset — the bytes are downloaded,
decoded, regridded, and serialized into a NumPy-backed pickle. No
zarr, no parquet manifest, no kerchunk. Total download per member
is ~90 MB; the pickle is ~800 MB.

For the AIFS inference use case, this works perfectly. So why does
the project's whole GIK infrastructure exist? Because the AIFS
script and GIK are solving **different problems**.

### What each method optimizes for

| Aspect | Direct pkl (`s3_grib_pkl_input_aifsens.py`) | GIK method (parquet → zarr) |
|---|---|---|
| **Output shape** | Realized NumPy arrays in a pickle | Lightweight parquet of `(url, byte_offset, byte_length)` triplets |
| **Output size per member-date** | ~800 MB pickle | ~140 KB parquet |
| **When is data fetched?** | Immediately during script run | Lazily, at consumer query time |
| **When is data decoded?** | Immediately, by this script | Lazily, by each consumer (`xarray.open_dataset`) |
| **Number of consumers per output** | One — tied to AIFS's specific 92-field schema | Many — every consumer can pull a different subset |
| **What survives long-term** | The pickle (if you keep it; ~40 GB for 50 members) | The parquet (~7 MB total; trivially mirrorable on HuggingFace) |
| **Re-use across products** | Re-fetch + re-decode for any other consumer | Same parquet serves cGAN, ICPAC operational, climate analysts, etc. |
| **Streaming / lazy slicing** | Not possible (data is already realized) | Standard `xarray + dask` patterns |
| **Cost across a 1000-day archive** | 1000 × 40 GB pickles = 40 TB, recomputed per consumer | 1000 × 7 MB parquets = 7 GB total, shared across all consumers |
| **Dependency on `kerchunk`** | No (pure `gribberish` + `obstore`) | Yes (`kerchunk._grib_idx`) |

The key insight: the direct-pkl approach **bakes the consumer's
schema into the artifact**. The AIFS pickle is unusable for cGAN
because cGAN needs a different field set, a different grid (East
Africa subset at 0.25°, not global N320), different stacking, etc.
Producing it requires re-fetching, re-decoding, re-pickling — even
though the same source bytes on `s3://ecmwf-forecasts/` would serve
both.

GIK inverts this. The byte-range manifest is **consumer-agnostic**.
A single GIK parquet for one member-hour serves:

- **AIFS inference**: select the 89 messages, decode, regrid to N320,
  pickle.
- **cGAN training**: select the 11 channels Xu et al. (2026) defines,
  subset to East Africa, normalise, stream to PyTorch DataLoader.
- **ICPAC operational forecast**: select TP, accumulate 24h windows,
  compute exceedance probabilities against CMORPH thresholds.
- **A climate researcher's xarray notebook**: `xr.open_dataset(par)`,
  slice by `[time, level, lat, lon]`, plot.

Each consumer pays only the byte-range S3 cost for the messages they
actually read. Nobody pays for the messages they don't.

### The streaming analogy (worth restating)

GIK applies the same trick HTTP video streaming uses:

| Video streaming (HLS) | Weather data streaming (GIK) |
|---|---|
| `.m3u8` manifest lists segment URLs + byte ranges | Parquet manifest lists GRIB URLs + byte ranges |
| Player fetches only visible segments on demand | Reader fetches only needed messages on demand |
| Full video file never downloaded | Full multi-GB GRIB never downloaded |
| Manifest is tiny (KBs); segments are MBs | Manifest is tiny (KBs); messages are MBs |

You wouldn't watch a 4K film by downloading the entire MKV — and for
the same reason, an archive-scale weather workflow shouldn't fetch
3-6 GB GRIB files whole when the consumer only needs a few MB of one
variable.

### When the direct pkl approach is the right choice

The AIFS script's design IS correct for its problem:

- **Single, well-defined consumer schema** (92 fields, fixed).
- **One-shot use** (run inference once per analysis time, discard).
- **No infrastructure for hosting parquets** (no HuggingFace, no
  Cloud Run runtime, no Lithops).
- **Minimum dependency surface** (no `kerchunk`, easier to vendor).
- **Already paying the decode cost anyway** (you're going to decode
  everything before AIFS runs).

For those constraints, GIK adds complexity without value. The two
methods are complementary, not competing. The presence of the AIFS
script doesn't argue against GIK; it argues that **the right method
depends on the consumer pattern**.

### When GIK is the right choice (the project's actual use case)

The ICPAC pipeline targets:

- **An archive** (1000+ days, growing daily).
- **Multiple consumers** (cGAN training, operational forecast,
  research notebooks, calendar-map web app, BN flood-risk model).
- **Lazy, partial reads** (e.g. one variable for one month over East
  Africa).
- **Cost sharing** (the parquets are public on HuggingFace; cGAN and
  ICPAC and outside researchers all benefit from one set of manifests).

For that pattern, GIK is mandatory. The 7 GB-of-parquets vs
40 TB-of-pickles math is the project-level decision.

---

## Part 2 — The per-pressure-level keys bug

### What the cGAN authors reported

Documented in:
- `cGAN_tutorial/example_notebooks/pytorch_cgan/GIK_MAINTAINER_REQUEST.md`
- `cGAN_tutorial/example_notebooks/pytorch_cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`

The bug, in their words:

> The parquet currently emits **one `pl` reference per
> `(variable, step)`** keyed as `step_NNN/{var}/pl/{member}/0.0.0`. The
> underlying byte-range points at whatever GRIB message is *first* in
> the source `.grib2` file for that variable at that step — which is a
> different isobaric level depending on step.

The smoking-gun probe (on `20260301-00z-control.parquet`):

| Lead step (h) | `gh/pl` level | `u/pl` level | `v/pl` level |
|---:|---:|---:|---:|
|  6 | **400** | **250** | **250** |
|  9 | 300 | 500 | 500 |
| 12 | 300 | 500 | 500 |
| 24 | **1000** | **250** | **250** |
| 27 | **1000** | **250** | **250** |

`gh` at step 24+ resolves to 1000 hPa — that's the orographic-surface
height (~0 m), stacked under the same key as the 300/400 hPa heights
(~9000 m mid-troposphere). A consumer that builds a `lead_time` axis
from those keys gets a `gh` array silently mixing three different
physical quantities.

### Why this matters

For the Xu et al. (2026) EP-cGAN port (DOI 10.1175/WAF-D-24-0199.1)
the inputs include **5 pressure-level channels out of 11 total**:
`u/v` at low-trop, `ub/vb` at boundary layer, `gh` at mid-trop. These
channels carry:

- **Somali jet** (850–925 hPa cross-equatorial flow, East Africa
  long-rains moisture source).
- **Turkana jet** (low-level wind through the Ethiopian–Kenyan highland
  gap; controls Lake Victoria convection).
- **500/300 hPa trough/ridge identification** for synoptic-scale
  precipitation organisation.
- **Vertical wind shear** between two wind levels (drives squall lines,
  deep convection).

Without per-level keys, none of these signals are recoverable. The
cGAN port falls back to surface-only inputs and cannot reproduce the
paper's CSI improvements at the >50 mm/3 h extreme threshold.

### Root cause — at the right level of abstraction

The GIK parquet generator does (logically):

```
for each .index line:
    key = f"{var}/{level_type}/{member}/0.0.0"
    references[key] = [url, offset, length]
```

The `level_type` field is the typeOfLevel name (`"pl"`, `"sfc"`,
`"sol"`), NOT the level value. The `levelist` field carries the
actual hPa value (250, 500, 850, etc.) and was being silently dropped.

So 13 distinct GRIB messages — `u @ 50 hPa`, `u @ 100 hPa`, ..., `u
@ 1000 hPa` for one member at one step — all collided into the same
key `u/pl/control/0.0.0`. Whichever was written last (= whichever
came first in the source GRIB's message order) survived. ECMWF
reorders messages across steps, so the surviving level varies by step.

The fix is structurally simple: include the level value in the key for
pl rows.

```
if level_type == 'pl' and levelist:
    key = f"{var}/pl/{levelist}/{member}/0.0.0"
else:
    key = f"{var}/{level_type}/{member}/0.0.0"
```

The hard part wasn't the fix. It was finding which copy of the
generator the production pipeline actually runs.

---

## Part 3 — Investigation, in chronological order

### `4ca1c21` (2026-05-29) — first fix attempt, wrong targets

Two code sites looked like the parquet-key generators:

1. `ecmwf/ecmwf_index_processor.py:create_references_from_index`
2. `ecmwf/utils_ecmwf_step1_scangrib.py:fixed_ensemble_grib_tree`

Both were patched to extract `levelist` and emit per-level keys for
pl. Validation against a real 50r1 `.index` showed 6300 per-level pl
keys with the correct shape — looked like a win.

### `1d2cd7d` (2026-05-29) — synthesis MD with template implications

Wrote `ecmwf/lithops-cr-gik-ecmwf/2026-05-29-per-level-keys-fix-and-template-implications.md`
documenting why the code fix alone wasn't sufficient: the **template
tar.gz** on HuggingFace contains zarr metadata (`.zarray` shapes,
coord arrays) that was built without level awareness. New per-level
chunk-ref keys would have nowhere to land in a `.zarray` with no
level dimension. Template rebuild required.

### `9950fa3` (2026-05-29) — 49r1 reprocess plan

Wrote a parallel plan to rebuild the 49r1-era template (2024-03
onwards) since cGAN training spans MAM 2024, 2025, 2026. The chain:
Coiled preprocessing → realigner → package → upload → re-bake HF
parquets, gated by a paid ~1 h Coiled scan.

### `59b89dd` (2026-05-29) — production runtime fix (the real fix)

The breakthrough. `run_lithops_ecmwf.py` is **self-contained** per
CLAUDE.md's documented design: it does NOT import from
`ecmwf/ecmwf_index_processor.py`. It has its **own inline copies** of
`parse_grib_index` (line 243) and `create_references_from_index` (line
285). Those copies are what the Cloud Run runtime actually executes.

**So the `4ca1c21` fix to `ecmwf_index_processor.py` had zero
production impact.** The Cloud Run workers were still emitting the
buggy single-key-per-(var, step, member) pl shape.

`59b89dd` mirrors the `4ca1c21` fix into the inline copy. Validated
live against the 50r1 20260513 00z f9 `.index`:

- 8500 entries → 6300 pl + 1800 sfc references
- pl key shape `var/pl/{level_hPa}/{member}/0.0.0` (5 segments)
  uniformly
- sample (q, ens46) has all 14 distinct 50r1 isobaric levels
  `{10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}`
- sfc keys unchanged

**This is the production-impacting fix.** Every HF parquet produced
by a Cloud Run image built from `59b89dd` or later will have per-level
pl keys. The cGAN port's blocker is closed for new backfills.

### `c78e449` (2026-05-29) — aggregator fix revision

The `4ca1c21` patch to `fixed_ensemble_grib_tree` had a subtle bug:
it read `dattrs.get("GRIB_level")`, but cfgrib's `scan_grib` does
NOT populate `GRIB_level` in the per-variable `.zattrs` dict it
returns. Diagnostic verified the level value actually lives in a
**binary blob** at `group["refs"]["isobaricInhPa/0"]` (dtype `<f8`).
`c78e449` decodes that blob the same way the existing
`ensemble_member` extraction nearby decodes `number/0` — using
`np.frombuffer` on the latin1-encoded bytes.

Honest finding even after the revision: Gate B1 still failed.
`fixed_ensemble_grib_tree` returns only 2 keys for what should be
hundreds, even when fed real scan_grib output with correctly-decoded
level values. The MultiZarrToZarr step downstream of the path
construction silently drops every aggregation. Surrounding function is
broken in a way separate from the per-level extraction.

### `3ddb98a` (2026-05-30) — actual template producer identified

Per Gate A0 (commit `59b89dd`), the live HF template tar.gz contains
rt000.par files with structure `{var}/{typeOfLevel}/{N.0}/.zarray`
and 51 sequential indices per pl var. Neither
`ecmwf_index_preprocessing.py` (README's claim) nor
`fixed_ensemble_grib_tree` (the aggregator we patched) produces that
shape.

The actual producer is
`ecmwf/dev-test/ecmwf_par_to_ensemble_members.py:create_member_parquet`
(lines 265–329). Three discoveries:

1. Line 272–273 has an explicit comment: *"Create a simple zarr store
   structure without using grib_tree since we don't have the complete
   scan_grib format."* `grib_tree` and `strip_datavar_chunks` are
   imported at lines 29–30 but never called.
2. Lines 309 and 315 hardcode `chunks=[1, 181, 360]` and
   `shape=[1, 181, 360]` as `# Example` values. That's 1° resolution,
   which is why Gate A0 saw `[1, 181, 360]` in the live template
   despite the source GRIBs being 0.25° (721 × 1440).
3. The key construction at line 296 is per-level-aware
   (`f"{varname}/{level_type}/{level}"`), so the producer **would**
   emit per-level keys if upstream provided correct `level` values.
   The 51 sequential `N.0` slots mean upstream pre-collapses (member,
   level) into a single axis before the realigner sees the groups.

The `4ca1c21` and `c78e449` aggregator patches apply to code that
never built the production template. The work isn't lost — it's just
not in the production critical path.

---

## Part 4 — Current status

### What's fixed in production

- **`59b89dd`**: Every fresh `.index` file processed by the Cloud Run
  runtime now emits per-level pl keys. Verified on a real
  2026-05-13 50r1 .index. Once a Cloud Run image is rebuilt from this
  branch and redeployed, every backfill produces a per-level-correct
  parquet.

### What's NOT yet fixed

- **The template tar.gz on HuggingFace
  (`E4DRR/grib-index-kerchunk-templates/gik-fmrc-v2ecmwf_fmrc.tar.gz`).**
  The runtime fix produces correct chunk-ref keys (e.g.
  `u/pl/250/control/0.0.0`), but those keys still need a per-level
  zarr metadata skeleton in the template to be openable as a single
  consolidated zarr store. The current template's `.zarray` for pl
  variables has no level dimension at all.
- **The template producer** (`ecmwf_par_to_ensemble_members.py:create_member_parquet`).
  Two repair options sketched in `3ddb98a`'s MD: patch in place
  (A, ~50–100 LOC) or replace with `grib_tree` + `strip_datavar_chunks`
  (B, ~150 LOC, matches GEFS pattern). Decision deferred.
- **The 50r1 and 49r1 rebuild plan MDs.** Both reference
  `fixed_ensemble_grib_tree` as the aggregator the rebuild runs
  through — that's wrong. They need revising before any paid Coiled
  preprocessing run.

### Practical impact on the cGAN port today

- For dates produced by a Cloud Run image built from `59b89dd` onwards
  → cGAN can consume per-level pl keys directly (`step_NNN/u/pl/250/control/0.0.0`,
  etc.). The MAM 2026 training pass is unblocked for these dates.
- For dates produced before `59b89dd` (everything currently on HF) →
  pl keys are still collapsed. cGAN must either (a) re-bake those
  parquets with the new image, or (b) fall back to surface-only inputs
  for the broken dates.
- For the full archive view (`xarray.open_dataset` of the consolidated
  template) → still broken until the template tar.gz is rebuilt with
  per-level metadata. Single-key consumers work; whole-archive
  consumers don't.

### What the next concrete actions are (when work resumes)

1. Trace `process_forecast_hour` upstream of line 440 in
   `ecmwf_par_to_ensemble_members.py` to find where the (member,
   level) collapse happens.
2. Pick repair Option A or B based on what that trace reveals.
3. Revise the 50r1 / 49r1 rebuild plan MDs to point at the realigner
   (`create_member_parquet`), not `fixed_ensemble_grib_tree`.
4. Build a fresh Cloud Run image from the current branch tip (so the
   `59b89dd` runtime fix reaches production) and redeploy.
5. Run the paid Coiled preprocessing for 50r1 (and later 49r1) per the
   revised plan, producing the per-level template tar.gz.
6. Re-bake HF parquets, prioritising MAM 2026 (the cGAN blocker).

---

## References

- The direct-pkl script for AIFS:
  [`ea-aifs/s3_grib_pkl/s3_grib_pkl_input_aifsens.py`](../../../ea-aifs/s3_grib_pkl/s3_grib_pkl_input_aifsens.py)
- The cGAN bug reports:
  `cGAN_tutorial/example_notebooks/pytorch_cgan/GIK_MAINTAINER_REQUEST.md`
  + `GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`
- The per-level synthesis MD (rebuild rationale):
  [`../lithops-cr-gik-ecmwf/2026-05-29-per-level-keys-fix-and-template-implications.md`](../lithops-cr-gik-ecmwf/2026-05-29-per-level-keys-fix-and-template-implications.md)
- The producer-identification MD:
  [`../lithops-cr-gik-ecmwf/2026-05-30-template-producer-identified.md`](../lithops-cr-gik-ecmwf/2026-05-30-template-producer-identified.md)
- Xu et al. (2026), *Wea. Forecasting* 41:381–401, DOI 10.1175/WAF-D-24-0199.1.
- CLAUDE.md (project root) for the streaming-vs-download benchmarks
  and Lithops self-contained design rationale.
