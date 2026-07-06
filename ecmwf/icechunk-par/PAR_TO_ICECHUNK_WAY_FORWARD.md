# From GIK par files to an Icechunk store: what actually changes, and the way forward

Research note, 2026-07-06. Everything marked *measured* was verified in this
folder's scripts (see `README.md` and `CONVERSION_ESTIMATE.md`); the Herbie
facts were checked against the Herbie source/docs on this date.

## 1. The major change, in one sentence

The par file is a **private format** — a `[key, value]` DataFrame whose chunk
keys (`step_000/2t/sfc/control/0.0.0`) live in a different tree than its zarr
metadata, whose `.zarray` entries declare raw float32 while the bytes are GRIB2,
and whose URLs omit the `.grib2` suffix. Nothing can open it except code in
this repository. The Icechunk store is the **same information re-expressed in
the community's format**: a Zarr v3 hierarchy whose virtual chunk refs point at
the same `s3://ecmwf-forecasts` byte ranges, decoded on read by the gribberish
codec that registers itself via the standard `zarr.codecs` entry point.

| | par (today) | Icechunk store (measured here) |
|---|---|---|
| open | custom parser in this repo | `xr.open_zarr(...)` / `xr.open_datatree(...)` |
| reader deps | this repo's scripts | `pip install icechunk xarray gribberish` |
| dask | hand-rolled ThreadPoolExecutor loops | native: every GRIB message is a dask chunk |
| append a day | write 51 new files, no transaction | one commit, atomic, with history |
| schema change | new folder convention | new group (`{era}/{run}z`) |
| integrity | none (refs can silently rot) | per-commit snapshots; optional `last_updated_at` checksums |
| size | 20.6 GB pars (1,256 dates) | ~13 GB manifests, same coverage |

## 2. The DAG argument — is the assumption right?

**Yes, with one honest nuance.** The claim: with a single Icechunk store, a
dask cluster can run, as ONE graph — per date: EA subset → realized store →
hydrobasin zonal masking → threshold exceedance — and scale to 1000+ days;
with pars the same is possible but needs a convoluted custom abstraction.

Why it holds (all steps already demonstrated piecemeal here):

```
xr.open_zarr(repo, group="49r1/00z", chunks={})        # 1 task per GRIB message
  .tp.sel(latitude=slice(25,-14), longitude=slice(19,55))   # EA subset: graph-level slice,
                                                            # only EA-relevant messages ever fetched
  →  materialize to a realized zarr (measured: 8,670 msgs → 306 MB in 470 s,
     coordinator 308 MB, workers 280 MB)                    # per-date "realized store"
  →  apply hydrobasin mask (regionmask/xr.where on the realized array —
     plain xarray from here on)
  →  threshold exceedance (tp > T).mean("number") etc.      # ensemble probability
  →  reduce/append per date; repeat over the time axis      # 1000+ days = a bigger
                                                            # time slice of the SAME array
```

Every arrow is stock xarray/dask — no GIK-specific code anywhere in the
analysis. Dask sees chunk-level tasks (fetch byte range → gribberish decode →
subset → mask → compare), fuses them, and schedules them per worker; the
icechunk readonly session and the codec pickle cleanly to workers (*measured*:
multi-process cluster, bit-exact results, no OOM). The 1000-day scale-up is
literally `isel(time=slice(0, 1000))` — same graph, more tasks.

With pars, each of those arrows needs bespoke machinery: a par parser, a
key-convention walker, a manual byte-fetch + decode pool, hand-stitched numpy
arrays with hand-built coordinates, and then — only after all that — you reach
standard xarray for masking/thresholds. That is exactly the "custom parser and
convoluted abstraction" in the assumption. It works (the cGAN and probability
pipelines prove it) but every new analysis pays the abstraction tax again, and
nobody outside this repo can pay it at all.

The nuance: **graph size still needs planning at 1000+ days.** 1,000 dates ×
51 members × 85 steps of one variable is ~4.3 M chunk-tasks — a single naive
graph would strain the scheduler. The optimization the DAG enables (and pars
don't) is that pruning happens *declaratively*: select the variable, steps and
region first and dask never materializes tasks for anything else; batch the
time axis (`.map_blocks` per date, or loop `to_icechunk(region=...)` writes)
and the scheduler stays light. The precedent already runs in production:
the CMORPH/RFE2 threshold-trigger pipelines in `ibf-thresholds-triggers` do
exactly subset → mask → exceedance on icechunk/zarr stores.

**So: par → Icechunk is the right move for long-term storage and community
usability** — not because pars can't do the work, but because the Icechunk
store deletes the private-abstraction layer between the archive and the
standard stack, adds transactions/versioning/append that a pile of parquet
files cannot have (VirtualiZarr #964's argument), and makes the archive
readable by anyone with three pip packages. The remaining honest caveats:
readers must install gribberish (codec resolves by entry point — *measured*,
no import needed), virtual refs die if ECMWF ever rewrites the source objects
(mitigate with `last_updated_at` checksums), and one store = one commit
stream (fine for sequential backfill + 4 appends/day).

## 3. Herbie integration — GIK as a plugin

Checked against Herbie's source (2026-07-06):

- Herbie ships model templates for **ecmwf (IFS/AIFS), gefs, gfs, cfs** and ~20
  more (`herbie/models/*.py`) — the exact corpora GIK targets. Templates are
  small classes (SOURCES, PRODUCTS, IDX_SUFFIX...) and third-party templates
  load via `importlib.metadata.entry_points` — **a real plugin mechanism**.
- `Herbie(...).inventory()` parses both wgrib2-style (`.idx`) and
  eccodes-style (`.index`) files into a DataFrame with `grib_message`,
  `start_byte`, `end_byte`, `range`, variable/level metadata and a regex
  `search_this` column. **This is byte-for-byte the primitive GIK Stage 2
  extracts** — Herbie already implements GIK's index-reading layer, for far
  more models, with an actively maintained community.

So the pieces line up naturally:

```
Herbie                          GIK                         Icechunk
──────                          ───                         ────────
model templates    ──────────►  (replaces per-product       one store,
+ .idx/.index inventory()        path/parse logic in         {era}/{run}z groups
  = [msg, start_byte,            Stage 2)
     end_byte, var, level]
                                era/template table   ─────► array shapes, dims,
                                (this repo's contribution)   level supersets
                                inventory row → VirtualChunkSpec
                                [time, member, step, level, 0, 0]
                                                     ─────► set_virtual_refs
                                                            + commit per date
```

What GIK adds that Herbie doesn't have: the **zarr-model assembly** (mapping
inventory rows to chunk indices in an FMRC array), the **era/template
knowledge** (grids, level supersets, schema eras), and the **Icechunk writer**
(manifest splitting, append discipline). What Herbie adds that GIK hand-rolls:
maintained discovery/paths/inventory for every model and source, so the GEFS
`.idx` and CFS parsing code in this repo could shrink to Herbie calls.

Three integration options, in increasing ambition:

| option | shape | effort | recommendation |
|---|---|---|---|
| (a) companion package `herbie-icechunk` (or `gik`) | `to_icechunk(H_list, repo, group=...)` consuming `H.inventory()` | small; no upstream buy-in needed | **start here** |
| (b) Herbie accessor upstream | `Herbie(...).to_icechunk(repo)` like `.xarray()` | PR + maintainer discussion | after (a) proves the API |
| (c) full merge of GIK into Herbie | GIK becomes Herbie's virtual-store backend | large | only if (b) lands well |

Concrete first step (one afternoon): `gik_from_herbie(H, repo, group)` that
takes one GEFS date via `FastHerbie`, builds `VirtualChunkSpec`s from the
inventory DataFrame, commits, and asserts bit-identical output against the
existing par-based builder for the same date. This repo already contains the
validation harness (`compare_gik_herbie*.py`, the CFS gribberish-vs-Herbie
work), so the equivalence test is nearly free. If it holds for GEFS
(one-file-per-member, wgrib2 idx) and ECMWF (all-members-per-file, eccodes
index), CFS follows the same pattern and the **pars stop being an intermediate
at all** — Herbie inventory → Icechunk directly, with the daily operational
append being a ~30 s job per model.

## 4. Way forward, ordered

1. **Backfill the unified GCS store** (`gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens`)
   from the existing pars — `backfill_all_eras.py`, ~13 h sequential (see
   `CONVERSION_ESTIMATE.md`). Mirror finished store → source.coop (plain copy).
2. **Add `last_updated_at` checksums** on refs during backfill or as a
   follow-up pass, so source-object rewrites fail loudly.
3. **Hand Team 1 the DAG demo**: one notebook — open group, EA subset,
   hydrobasin mask, exceedance probability, 100 days on a small cluster.
4. **Prototype `gik_from_herbie`** for one GEFS date; assert equivalence vs
   the par pipeline; then propose the plugin package to the Herbie community
   (option a), with (b) as the goal.
5. Retire pars as the operational product once the icechunk daily append runs
   alongside Lithops for a probation month; keep pars as the archival
   intermediate on HF (they remain the provenance record).
