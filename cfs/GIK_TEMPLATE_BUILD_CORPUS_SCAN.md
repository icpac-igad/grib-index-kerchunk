# GIK Template Build: the whole-corpus `scan_grib` requirement

**Status:** definitive correction
**Scope:** how the Stage-1 template is *built* (not read) for GEFS vs ECMWF,
whether the pressure-level collapse bug exists in GEFS, and what both imply for
CFS. Corrects an earlier claim ("Lesson 2") that the CFS template needs only a
2-file `scan_grib`.

---

## 0. Verdict (TL;DR)

| Claim under review | Verdict |
|---|---|
| "CFS is GEFS-like, so the template only needs a 2-file `scan_grib`; you don't need Coiled." | **WRONG.** |
| The GEFS production template is built by `scan_grib` over the **whole corpus** (every member × every forecast hour), run on a distributed cluster. | **CORRECT** (`gefs/README.md:23`). |
| The GEFS template is stored following the **same routine as ECMWF** (per-member archive of pre-scanned per-hour parquets, tarred, on the same HuggingFace dataset). | **CORRECT.** |
| The pressure-/multi-level collapse bug that hit ECMWF is also present in GEFS. | **NO** — GEFS avoids it (uses kerchunk's native `store_data_var`, a real level axis). The bug was ECMWF-specific, caused by ECMWF re-implementing store-building. |
| Therefore the "Coiled function + `scan_grib` over all GRIB files in the corpus" is mandatory for the GEFS-style template, exactly as for ECMWF. | **CORRECT.** |

The earlier reasoning conflated **two different `scan_grib` roles in GEFS**.
They are not the same operation and only one of them is cheap.

---

## 1. The two `scan_grib` roles in GEFS (the source of the error)

GEFS invokes `scan_grib` in **two completely separate places**, with opposite
cost and purpose:

| | A. Runtime skeleton scan | B. Template-build corpus scan |
|---|---|---|
| Where | `run_lithops_gefs.py` (per backfill date) + `gefs-deflated-store-template-20241112.parquet` | `gefs/gefs_index_preprocessing_fixed.py` (one-time) |
| What | `scan_grib(f000, f003) → grib_tree → strip_datavar_chunks` | `build_idx_grib_mapping(url)` on **every** GRIB file |
| Files scanned | **2** | **2,430** (30 members × 81 hours) |
| Cost | ~5.5 s (eliminable; baked to a 26 KB parquet) | hours of compute → **needs a cluster** |
| Produces | the zarr **metadata skeleton** (variable/dim/chunk-shape schema) | the per-`(member,hour)` **idx→grib mapping parquets** that ARE the template |
| Parallelism | none needed | **Dask on Coiled** (`@dask.delayed` + `dask.compute`) |

The earlier "Lesson 2" looked at **A** (the cheap 2-file scan), saw GEFS "only
scans 2 files at runtime," and wrongly concluded that a GEFS-like product needs
no corpus scan and no Coiled. But the artifact that actually *makes the GIK
method work* — the template archive on HuggingFace — is produced by **B**, the
whole-corpus scan.

> **A** is a convenience/skeleton; **B** is the template. You cannot skip **B**.

---

## 2. Evidence: the GEFS template IS a whole-corpus scan

`gefs/README.md:23` (production artifact table):

> `gik-fmrc-gefs-20241112.tar.gz` … built by `gefs/dev-test/gefs_index_preprocessing_fixed.py`
> running **`build_idx_grib_mapping` over every (member, timestep) = 30 × 81 = 2,430 GRIB files**.

The builder loop (`gefs/gefs_index_preprocessing_fixed.py`):

```python
@dask.delayed                                    # line 163 — distributed task per file
def process_gefs_time_idx_data(s3url, bucket_name):
    mapping = build_idx_grib_mapping(s3url, storage_options={"anon": True})   # line 187
    deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
    ...
    parquet_path = f"gefs-time-{date_str}-{member}-rt{int(forecast_hour):03}.parquet"  # line 194
    deduped_mapping.to_parquet(parquet_path)
    # → GCS time_idx/gefs/{year}/{date}/{member}/...  (line 199)

# fan-out over all URLs of all members, computed on the cluster:
final_results = dask.compute(*results)           # line 305
```

- One `build_idx_grib_mapping` call **per GRIB file**, over the full
  member × forecast-hour grid (`create_gefs_full_ensemble_mappings`,
  members `gep01..gep30`, line 338; URLs per member, line 295).
- `@dask.delayed` + `dask.compute` = distributed fan-out. The worker
  credential path is `coiled-data-key.json` and it calls
  `distributed.get_worker()` (lines 28, 122–124) — i.e. the Dask cluster is a
  **Coiled** cluster. This is the GEFS analogue of
  `ecmwf/dev-test/ecmwf_*_coiled_preprocessing.py`.

### `build_idx_grib_mapping` *is* `scan_grib` over each file

`build_idx_grib_mapping` (kerchunk `_grib_idx`) internally runs `scan_grib` on
the GRIB file and joins the result to the parsed `.idx`, producing the
`attrs ↔ grib-message-metadata` mapping. So 2,430 mapping parquets = 2,430
`scan_grib` decodes. (`gefs/docs/GEFS_Three_Stage_Processing.md:45,121` and
`gefs/docs/GEFS_SETUP_DOCUMENTATION.md:17,39` document the same call.)

**Conclusion:** the GEFS template build is a whole-corpus `scan_grib`, run on a
distributed (Coiled) cluster — structurally identical to ECMWF's per-`(stream,
hour)` Coiled scan, only the parallel framework differs (Dask-on-Coiled vs
`coiled.function`).

---

## 3. Evidence: GEFS stores the template following the ECMWF routine

| | ECMWF | GEFS |
|---|---|---|
| Build driver | `ecmwf_{0p4,49r1,50r1}_coiled_preprocessing.py` (Coiled fan-out) | `gefs_index_preprocessing_fixed.py` (Dask-on-Coiled fan-out) |
| Per-task output | `e_sg_mdt_{date}_{stream}_{h}h.parquet` → GCS | `gefs-time-{date}-{member}-rt{h}.parquet` → GCS |
| Packaged as | `gik-fmrc-v2ecmwf_fmrc*.tar.gz` | `gik-fmrc-gefs-20241112.tar.gz` |
| Hosted on | HF `E4DRR/grib-index-kerchunk-templates` | **same** HF dataset |
| Archive contents | per-member, per-hour `rt{hhh}.par` slices | per-member, per-hour `rt{hhh}.parquet` slices |

Both: **fan-out scan over the corpus → per-`(member/stream, hour)` parquet →
tar.gz → HuggingFace**. Verified live (HTTP 200): ECMWF tar = 126 MB, GEFS tar
= 3.2 MB, both on `E4DRR/grib-index-kerchunk-templates` (mirrored on
`Nishadhka/gfs_s3_gik_refs`). The CFS equivalent
`gik-fmrc-v2cfs_fmrc.tar.gz` returns **404 on both repos** — i.e. the CFS
Stage-1 template has **not** been built/published yet, so the CFS cloud path
(`run_lithops_cfs.py:107` `TEMPLATE_URL`) currently points at a non-existent
file.

---

## 4. The multi-level collapse bug: present in GEFS? — **No**

### 4.1 What the bug was (ECMWF)

ECMWF's self-contained runtime/realigner re-implemented zarr-store building and
keyed pressure-level data as `step_NNN/{var}/pl/{member}/0.0.0` — keyed on the
level **type** (`pl`), not the level **value**. All isobaric levels of a
variable therefore mapped to **one** key whose byte-range pointed at whichever
isobaric message was *first* in the GRIB. Result: levels silently mixed across
lead times; the 4 extra 13-level levels were orphaned. (See
`ecmwf/docs/2026-06-03-three-era-scan-run-verification-and-perlevel-fix.md`
§4.1, and `ecmwf/lithops-cr-gik-ecmwf/2026-05-29-per-level-keys-fix-and-template-implications.md`.)

### 4.2 Why GEFS does NOT have it

GEFS does **not** re-implement store-building — it uses kerchunk's native
`store_data_var`/`store_coord_var` with a real level axis
(`run_lithops_gefs.py:process_unique_groups`, lines 793–833):

```python
for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
    lvals = group.level.unique()
    if len(lvals) == 1:
        lvals = lvals.squeeze();  dims[key[2]] = 0          # scalar level
    elif len(lvals) > 1:
        lvals = np.sort(lvals);   dims[key[2]] = len(lvals) # REAL level axis
        coords["datavar"] += (key[2],)                      # level becomes a coord
    store_coord_var(key=f"{base_path}/{key[2]}", ..., data=lvals)
    store_data_var(key=f"{base_path}/{key[0]}", ..., lvals=lvals if lvals.shape else None)
```

When a variable has multiple levels of the same `typeOfLevel`, kerchunk builds a
proper level **dimension** and `store_data_var` indexes each level as its own
chunk (`…/0.N.0`). No collapse, no mixing. (In practice the GEFS `pgrb2sp25`
subset is mostly surface/2 m/10 m single-level fields, so multi-level groups are
rare — but the mechanism is correct when they occur.)

The bug was therefore **ECMWF-specific**, a consequence of ECMWF dropping
kerchunk at runtime and hand-rolling the store. GEFS keeps kerchunk's store
machinery and is safe.

### 4.3 CFS status

CFS is **self-contained like ECMWF** (no kerchunk at runtime) but builds keys
differently — it embeds the **level string value** in the path
(`run_lithops_cfs.py:468–469`):

```python
level_clean = level.replace(' ', '_')
key = f"{var_lower}/{level_clean}/0.0.0"     # e.g. soilw/0-0.1_m_below_ground/0.0.0
```

Because the distinct `.idx` level strings (`SOILW` depth layers, `TCDC` cloud
layers, `PRES` levels) yield **distinct keys**, CFS does **not** collapse levels
the way ECMWF did — but it also does **not** build a real zarr level axis (each
level becomes a separate flat group). This is functionally safe for byte-range
streaming, but should be validated against a known-good decode (no CFS
GIK-vs-Herbie validation exists yet), and the *template* must actually contain
the per-level metadata for those keys to resolve.

---

## 5. Corrected guidance

| Question | Earlier (wrong) answer | Correct answer |
|---|---|---|
| Does a GEFS-style template need a whole-corpus `scan_grib`? | No, 2 files | **Yes** — every `(member, hour)`, 2,430 files |
| Is Coiled/distributed compute needed to build it? | No | **Yes** — Dask-on-Coiled fan-out (ECMWF uses `coiled.function`; GEFS uses `@dask.delayed`) |
| Is the 2-file runtime scan the template? | (implicitly yes) | No — it's only the metadata **skeleton** (role A); the template is the corpus mapping (role B) |
| Does the level-collapse bug affect GEFS? | (not considered) | **No** — kerchunk `store_data_var` gives a real level axis |
| Does CFS need the same corpus scan + Coiled? | No | **Depends on the chosen template model** (see §6) |

---

## 6. Implications for CFS

The corpus-scan requirement is a property of the **template model**, not the
product:

- **GEFS model** (per-`(member,hour)` `build_idx_grib_mapping` parquets):
  **mandatory** whole-corpus `scan_grib` on a Coiled/Dask cluster. For CFS this
  is the larger job — ~861 files/member × members (and 4 `6hrly_grib_0{1..4}`
  ensemble streams), i.e. thousands of `scan_grib` decodes. A
  `cfs_coiled_preprocessing.py` adapted from `ecmwf_*_coiled_preprocessing.py`
  (swap the task plan to CFS `flxf` URLs over all hours/members) is the right
  vehicle.
- **ECMWF model** (one deflated zarr skeleton + direct `.idx` parsing, which is
  what `run_lithops_cfs.py` already implements): you still must scan **enough of
  the corpus to capture the full schema** — every variable, every `typeOfLevel`,
  and every `stepType` (accumulated vs instantaneous fields appear at different
  forecast hours). A 2-file (`f000`+`f006`) scan is **not** guaranteed to be a
  superset of the whole run's variable/level/step layout. This is the direct
  analogue of why ECMWF scans **all 85 hours** rather than just the first two,
  and of the 9-vs-13-level "era" lesson (a too-narrow reference scan orphans
  levels/variables that only appear later).

**Net for CFS:** do not assume a 2-file scan suffices. Either (a) adopt the
GEFS-style corpus mapping (Coiled fan-out, mandatory), or (b) keep the
ECMWF-style skeleton but **prove** that the chosen reference scan covers every
variable/level/stepType across the full forecast range before publishing
`gik-fmrc-v2cfs_fmrc.tar.gz` — and add a CFS GIK-vs-Herbie decode validation.

---

## 7. File / line reference

| File | Lines | What it shows |
|---|---|---|
| `gefs/README.md` | 23–24 | template = `build_idx_grib_mapping` over 2,430 files; 26 KB skeleton is the *separate* 2-file scan |
| `gefs/gefs_index_preprocessing_fixed.py` | 163, 187, 194, 199, 295, 305, 338 | per-file `build_idx_grib_mapping`, Dask fan-out, per-`(member,hour)` parquet → GCS |
| `gefs/gefs_index_preprocessing_fixed.py` | 28, 122–124 | Coiled worker creds (`coiled-data-key.json`, `get_worker`) |
| `gefs/lithops-cr-gik-gefs/run_lithops_gefs.py` | 793–833 | native kerchunk level axis → no collapse bug |
| `ecmwf/dev-test/ecmwf_50r1_coiled_preprocessing.py` | 50–66, 119–176 | ECMWF Coiled corpus scan (the pattern GEFS mirrors with Dask) |
| `ecmwf/docs/2026-06-03-three-era-scan-run-verification-and-perlevel-fix.md` | §4 | the ECMWF per-level bug + why it's downstream-specific |
| `cfs/lithops-cr-gik-cfs/run_lithops_cfs.py` | 107, 468–469 | CFS `TEMPLATE_URL` (404), level-string key (no collapse, no axis) |
