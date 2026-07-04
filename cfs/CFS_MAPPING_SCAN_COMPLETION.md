# CFS Stage-1 Mapping Scan — Completion Record

**Date:** 2026-07-04 · **Reference:** `cfs.20260702` `00z`, member `01`, `flxf`
**Producer:** `cfs/cfs_coiled_preprocessing.py` (`--backend local`)
**Status:** ✅ complete — **857 / 857** files mapped

This records the completion of the full-corpus **mapping-parquet scan**, the
"mapping half" of the GEFS-model CFS template. It complements the deflated
**skeleton** built earlier (see `CFS_TEMPLATE_WORK_HANDOFF.md`) — together they
form the two-part template that `run_lithops_cfs.py` will consume once rewired to
the GEFS path (`map_from_index` + `generate_axes`).

---

## 1. What ran

```bash
cd cfs/
uv run cfs_coiled_preprocessing.py --date 20260702 --run 00 \
    --backend local --workers 4 --out ./cfs_template
```

- **Backend:** `local` (this machine, via `uv`) — no Coiled, no GCS key.
- **Task grid:** discovered from S3 (not a flat range) → member 01's **857**
  6-hourly `flxf` steps, `0 … 5136 h` (214 d). See `CFS_CORPUS_SCOPING.md` §5.
- **Per task:** `build_idx_grib_mapping(flxf_url, validate=False)` → dedup on
  `attrs` → parquet, returned to the client and bundled into one tar.

## 2. Result

| Metric | Value |
|---|---|
| Files mapped | **857 / 857** (no failures) |
| Wall time | **78.4 min** (`--workers 4` on a 2-core / 7 GB box) |
| Effective rate | ~5.5 s/file — threading gave ~no speedup (eccodes is CPU-bound; 2 cores) |
| Output tar | `cfs/cfs_template/gik-fmrc-v2cfs_fmrc-mappings-20260702-00.tar.gz` |
| Tar size | **1.77 MB** gzip (≈ 15.8 MB uncompressed) |
| Entries | 857 parquets, `rt0000 … rt5136` |

> The `cfs_template/` dir is **gitignored** — this is a build artifact, not
> source. Publish it to HuggingFace, don't commit it.

## 3. Artifact layout & contents

Tar member paths (GEFS-style, so a `LocalTarGzMappingManager`-equivalent can look
up by member + forecast hour):

```
cfs/time_idx/2026/20260702/00/01/cfs-time-20260702-00-rt0000.parquet
cfs/time_idx/2026/20260702/00/01/cfs-time-20260702-00-rt0006.parquet
...
cfs/time_idx/2026/20260702/00/01/cfs-time-20260702-00-rt5136.parquet
```

Each parquet is the deduped `build_idx_grib_mapping` output (~101–103 rows,
~19 KB), with columns:

```
offset_idx, date, attrs, length_idx, idx_uri, grib_uri, varname, typeOfLevel,
stepType, name, step, level, time, valid_time, uri, offset_grib, length_grib,
inline_value
```

`map_from_index(datestr, deduped_mapping, idxdf)` joins this template mapping with
a fresh date's parsed `.idx` (on `attrs`) to produce that date's byte-range
references — which is why one reference date suffices (see §5).

## 4. Where this fits (two-part GEFS-model template)

| Part | Artifact | Built by | Status |
|---|---|---|---|
| Deflated **skeleton** (zarr metadata) | `…/cfs-{date}{run}-member01-rt000.par` in `gik-fmrc-v2cfs_fmrc.tar.gz` | `build_cfs_template.py` (f000+f006 scan) | ✅ built earlier |
| **Mapping** parquets (idx→grib) | this `…-mappings-20260702-00.tar.gz` (857 files) | `cfs_coiled_preprocessing.py --backend local` | ✅ **this run** |

## 5. Why one reference date is enough

The mapping schema is **date-independent**: the deduped `attrs` → grib-message
structure is identical across init dates (verified in `CFS_CORPUS_SCOPING.md` §6 —
flxf idx schema is byte-identical 2018→2026). Fresh byte offsets for any forecast
date come from that date's own `.idx` at runtime via `map_from_index`. This mirrors
GEFS reusing its single `20241112` reference for all dates. So this ~78-min scan is
a **one-time** job, not repeated per forecast date.

## 6. Verification performed

- `completed 857/857` in the run log; tar re-listed = 857 entries `rt0000…rt5136`.
- A sample parquet (`rt0006`) re-opened: 103 rows, expected 18-column schema,
  correct `grib_uri`/`offset_grib`/`length_grib` triplets.

## 7. Reproduce / extend

```bash
# other reference date or cycle (member 01)
uv run cfs_coiled_preprocessing.py --date <YYYYMMDD> --run <00|06|12|18> --backend local
# all four members (ragged lengths; ~2,300 files -> hours on this box)
uv run cfs_coiled_preprocessing.py --date 20260702 --run 00 --members all --backend local
# large multi-date/member jobs: use Coiled instead (parallel)
python cfs_coiled_preprocessing.py --date 20260702 --run 00               # sink=local-tar
python cfs_coiled_preprocessing.py --date 20260702 --run 00 --sink gcs    # to GCS
```

## 8. Next steps (not blocking; neither done yet)

1. **Package + publish** — combine the skeleton `rt000.par` and these 857 mapping
   parquets into a single `gik-fmrc-v2cfs_fmrc.tar.gz`; upload to HuggingFace
   `E4DRR/grib-index-kerchunk-templates` (needs `HF_TOKEN`). Fix
   `run_lithops_cfs.py:107` `TEMPLATE_URL` (currently a 404).
2. **Rewire the daily run** — port the GEFS Stage-2/3 path into
   `run_lithops_cfs.py`: `LocalTarGzMappingManager` (read these parquets) +
   `parse_grib_idx` + `map_from_index` + `generate_axes` + `store_data_var` /
   `store_coord_var`, so it builds a real 857-step time axis instead of the
   current `step_XXXX/` overlay.
