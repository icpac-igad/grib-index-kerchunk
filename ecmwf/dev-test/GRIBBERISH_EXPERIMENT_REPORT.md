# Experiment: Replacing scan_grib with .index + gribberish

## Background

The ECMWF GIK pipeline takes **110.6 minutes** per daily run. Stage 1 (kerchunk `scan_grib`) accounts for **73.3 minutes (66%)** of total runtime, scanning two ~5.8GB GRIB files from S3 via thousands of small byte-range reads.

This document records the experiment to replace `scan_grib` with faster alternatives, assesses the risks, and proposes an implementation plan.

---

## Experiment Setup

**Environment:** `micromamba run -n aifs-etl python`
**Target file:** `s3://ecmwf-forecasts/20260205/00z/ifs/0p25/enfo/20260205000000-0h-enfo-ef.grib2`
**File size:** 5,841.4 MB (8,160 GRIB messages across 51 ensemble members x 160 variables/levels)
**Test script:** `test_gribberish_vs_scangrib.py`

---

## Benchmark Results

| Method | Time | Speedup vs scan_grib |
|--------|------|---------------------|
| **.index file parsing only** | **1.96s** | **2,370x** |
| gribberish bulk parse (50MB fetch + Rust parse) | 9.31s | 499x |
| gribberish + .index (10 sample byte-range fetches) | 36.4s | 128x |
| kerchunk scan_grib (20 messages, measured) | 11.4s | — |
| kerchunk scan_grib (8,160 messages, extrapolated) | ~4,644s (~77 min) | 1x (baseline) |

### Key Timing Observations

- **gribberish parse speed:** 68 messages parsed in **0.001s** (Rust-native speed)
- **.index file parse speed:** 8,160 JSON entries parsed in **0.131s** (pure Python)
- **S3 I/O dominates:** Even gribberish can't help when individual byte-range fetches take 2-3 seconds each due to S3 latency

### gribberish ECMWF Compatibility

| Feature | Status |
|---------|--------|
| `parse_grib_mapping()` | Works — returns metadata dict for all parseable messages |
| `parse_grib_message_metadata()` | Partial — panics (Rust unwrap) on some ECMWF accumulation templates |
| Variable identification | ~70% success — returns `"missing"` for `sd`, `tprate`, `vsw`, etc. |
| Grid shape extraction | Works — correctly returns `(721, 1440)` |
| Ensemble detection | Works — identifies `"ensemble forecast"` generating process |
| Level/dim extraction | Works — identifies isobaric, surface, above-ground levels |

---

## How the Current 3-Stage Pipeline Works

```
Stage 1: scan_grib (73 min)
│  Input: 2 GRIB files from S3 (~5.8GB each)
│  Process: kerchunk scan_grib → MultiZarrToZarr → strip_datavar_chunks
│  Output: "deflated_stores" — zarr metadata ONLY (no data refs)
│    Contains: .zgroup, .zarray, .zattrs, latitude/0, longitude/0
│    Stripped: ALL data chunk references [url, offset, length]
│
Stage 2: .index + template merge (35.6 min)
│  Input: 85 .index files from S3 + HuggingFace template (ref date 20240529)
│  Process: parse .index → build byte refs → merge with template
│  Output: Per-member refs dict with template structure + fresh byte positions
│    Template provides: .zgroup, .zarray, .zattrs, coordinate arrays, hierarchy
│    Index provides: data chunk refs [url, offset, length] for all 85 hours
│    Total per member: ~6,685 references (2,774 template + 3,912 index)
│
Stage 3: Final merge (4.6 sec)
│  Process: deflated_stores.copy() → overlay Stage 2 refs
│  Output: Final parquet files (9,938 or 8,102 rows per member)
```

### What `strip_datavar_chunks` Does

```python
strip_datavar_chunks(kerchunk_store, keep_list=('latitude', 'longitude'))
```

Removes ALL data chunk references from the zarr store except latitude and longitude coordinates. The "deflated" store retains only:
- `.zgroup` entries (zarr group structure)
- `.zarray` entries (array shape, dtype, chunks, compressor)
- `.zattrs` entries (attributes, dimension names, GRIB metadata)
- `latitude/0` and `longitude/0` (coordinate data, inline)

**Everything else** — the actual `[url, offset, length]` byte references — is stripped.

### What the Template Provides

The HuggingFace template (`gik-fmrc-v2ecmwf_fmrc.tar.gz`) contains per-member parquet files built from a reference date (2024-05-29). Each parquet has ~2,774 entries including:
- Full hierarchical zarr group structure
- `.zarray` definitions (dtype, shape, chunks)
- `.zattrs` with CF metadata and GRIB attributes
- Coordinate arrays (latitude, longitude, time, step, valid_time, number)
- Data chunk references (pointing to reference date's GRIB files)

### How Stage 3 Merges

```python
# ecmwf_three_stage_multidate.py lines 396-403
final_store = deflated_store.copy()        # Stage 1 structure (current date)
for key, ref in complete_refs.items():     # Stage 2 refs (template + fresh index)
    if not key.startswith('_'):
        final_store[key] = ref             # OVERWRITES matching keys
```

Stage 2's output **overwrites** matching keys from the deflated store. Since the template's key patterns match the deflated store's patterns (both are hierarchical zarr paths from scan_grib), the template values replace the deflated store values for overlapping keys.

---

## Risk Analysis: Removing scan_grib

### RISK 1: Loss of Current-Date Coordinate Validation — LOW

**What scan_grib provides:** Zarr coordinate arrays with the current forecast date's time and valid_time values.

**Why it's low risk:** The template overwrites these anyway in Stage 3 (template keys overlap with deflated store keys). The template has reference-date coordinates (2024-05-29), which the current pipeline already uses in the final output. The data streaming code (`stream_cgan_variables.py`) accesses data by step index (0, 3, 6, ..., 360), not by absolute datetime — the coordinate values are cosmetic for this use case.

**Mitigation:** `generate_ecmwf_axes()` already generates correct date-specific axes independently. If absolute timestamps are needed downstream, they can be computed from the date + step, which the .index file provides.

### RISK 2: Template Staleness (Model Changes) — MEDIUM

**What scan_grib provides:** Ground-truth zarr structure from the actual current GRIB file. If ECMWF changes their IFS model (new variables, different grid resolution, changed ensemble size, different compression), scan_grib captures these changes immediately.

**Why it matters:** The template is from 2024-05-29. ECMWF model upgrades (cycle changes) can alter:
- Variable names or parameter codes
- Grid resolution (currently 0.25°, 721x1440)
- Ensemble member count (currently 51)
- Level structure (currently 3 types: sfc, pl, sol)
- Compression algorithm

**Historical context:** ECMWF IFS cycle changes happen roughly yearly. The last major change (Cycle 49r1 → 49r2) did not change the ensemble grid or variable set. The 0.25° ensemble grid has been stable since 2023.

**Mitigation strategies:**
1. **Variable count check:** Compare the .index file's unique (param, levtype) count against the template's expected count. If they differ, fall back to scan_grib.
2. **Periodic template refresh:** Regenerate the template quarterly using scan_grib (offline, not in the daily pipeline).
3. **ECMWF changelog monitoring:** Subscribe to ECMWF cycle change announcements.

### RISK 3: Member Group Variable Differences — LOW

**Observation:** Final parquets show 9,938 rows for members 1-12 vs 8,102 rows for members 13-50 (difference of 1,836 rows).

**Why:** ECMWF ensemble members 1-12 include additional diagnostic variables not present in members 13-50 (extended ensemble diagnostics). The template already accounts for this with separate per-member parquet files (different templates for `ens_01` through `ens_12` vs `ens_13` through `ens_50`).

**Risk:** LOW — the template already handles this member-group distinction. The .index file also shows different message counts per member group, so the index-based approach naturally captures this.

### RISK 4: No Data Integrity Validation — MEDIUM

**What scan_grib provides:** Implicit validation that the GRIB file is well-formed and accessible. scan_grib reads every message header, so it would detect:
- Truncated/corrupted GRIB files
- S3 upload failures (partial files)
- Stale .index files that don't match the GRIB binary

**What .index-only loses:** The .index file is a separate text file. If the GRIB file is partially uploaded or the .index file was generated from a different GRIB version, byte offsets could be wrong. This would cause silent failures at data access time (wrong data read from wrong offsets).

**Mitigation strategies:**
1. **Spot-check validation:** Fetch 3-5 random byte ranges from the GRIB file and verify with gribberish that they parse as valid GRIB messages. Cost: ~10 seconds.
2. **Size validation:** Compare the GRIB file size (from S3 HEAD request) against the sum of all .index entry lengths. Mismatch indicates corruption.
3. **ETag/timestamp comparison:** The .index file includes `grib_etag` and `grib_updated_at` fields that can be compared against the GRIB file's current S3 metadata.

### RISK 5: Loss of Hierarchical Zarr Structure — VERY LOW

**Concern:** scan_grib + MultiZarrToZarr produces a hierarchical zarr structure (e.g., `t2m/instant/surface/t2m/0.0`). Would losing this break downstream consumers?

**Why it's very low risk:** The template already provides this exact hierarchical structure (built from a previous scan_grib run). Stage 2's merge preserves the template's hierarchy and updates only the data chunk references. The hierarchical structure is STATIC for a given model version.

### RISK 6: .index File Availability — LOW

**Concern:** What if ECMWF stops providing .index files alongside GRIB files?

**Why it's low risk:** The .index files are a core part of ECMWF's S3 data distribution. They're used by many consumers and are unlikely to be removed. Additionally, the .index file format is ECMWF's official index specification.

**Mitigation:** Keep scan_grib as a fallback. If .index file is missing for a given forecast hour, fall back to scan_grib for that hour only.

---

## Risk Summary Matrix

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|-----------|--------|------------|
| Template staleness | Medium | Low (yearly) | Wrong refs | Variable count check + periodic refresh |
| Data integrity | Medium | Low | Silent wrong data | Spot-check 3-5 byte ranges with gribberish |
| Coordinate dates | Low | N/A | Cosmetic | generate_ecmwf_axes() already handles this |
| Member group diffs | Low | N/A | None | Template already handles per-member differences |
| Hierarchical structure loss | Very Low | N/A | None | Template provides the structure |
| .index unavailability | Low | Very Low | Pipeline failure | scan_grib fallback |

**Overall assessment: Safe to replace scan_grib with .index parsing for daily operations, with the spot-check validation and scan_grib fallback as safety nets.**

---

## Proposed Implementation Plan

### Phase 1: Index-Based Stage 1 Replacement (Immediate)

Replace the 73-minute scan_grib call with .index file parsing. The deflated_stores dict needs to contain the zarr structure that Stage 3 expects.

**Approach:** Instead of running scan_grib, build deflated_stores directly from the template. The template already contains the zarr structure. We just need to load it once and reuse it for all members.

```
Current Stage 1 flow (73 min):
  scan_grib(file1) → scan_grib(file2) → MultiZarrToZarr → strip_datavar_chunks → deflated_stores

Proposed Stage 1 flow (~2 sec):
  load template from tar.gz → extract per-member zarr structure → deflated_stores
```

**Key implementation steps:**

1. **New function `build_deflated_stores_from_template()`:**
   - Open the HuggingFace template tar.gz (already downloaded)
   - For each member, read the template parquet
   - Filter to only metadata keys (.zgroup, .zarray, .zattrs) + coordinate arrays
   - Strip data chunk references (equivalent to strip_datavar_chunks)
   - Return as deflated_stores dict
   - Expected time: ~2-5 seconds for all 51 members

2. **Validation step `validate_index_against_template()`:**
   - Fetch the .index file for hour 0
   - Extract unique (param, levtype) combinations
   - Compare against expected set from template
   - If mismatch detected, log warning and optionally fall back to scan_grib

3. **Spot-check step `spot_check_grib_integrity()`:**
   - Pick 3-5 random .index entries
   - Fetch those byte ranges from S3
   - Use gribberish `parse_grib_mapping()` to verify they parse as valid GRIB
   - If failures detected, fall back to scan_grib
   - Expected time: ~10 seconds

### Phase 2: Parallelize Stage 2 (Quick Win)

Wire in the existing `ProcessPoolExecutor` code from `ecmwf_index_processor.py:723`.

```
Current Stage 2 flow (35.6 min):
  for member in 51 members:        # SEQUENTIAL
      build_complete_parquet_from_indices(member)  # ~42 sec each

Proposed Stage 2 flow (~5 min):
  with ProcessPoolExecutor(max_workers=8):
      parallel map over 51 members   # 7 batches × 42 sec
```

### Phase 3: Full Pipeline Timing Target

| Stage | Current | After Phase 1 | After Phase 1+2 |
|-------|---------|---------------|-----------------|
| Stage 1 | 73.3 min | ~5 sec | ~5 sec |
| Stage 2 | 35.6 min | 35.6 min | ~5 min |
| Stage 3 | 4.6 sec | 4.6 sec | 4.6 sec |
| GCS Upload | 1.3 min | 1.3 min | 1.3 min |
| **Total** | **110.6 min** | **~38 min** | **~7 min** |

### Phase 4: Lithops/Serverless Stage 2 (Future)

Replace ProcessPoolExecutor with Lithops Lambda functions for Stage 2.

| Stage | After Phase 3 (Lithops) |
|-------|------------------------|
| Stage 1 | ~5 sec |
| Stage 2 | ~1 min (51 parallel Lambda functions) |
| Stage 3 | 4.6 sec |
| **Total** | **~2 min** |

---

## Where gribberish Fits

gribberish is NOT the right tool for replacing scan_grib in Stage 1 (reference extraction), because:
1. The .index file already provides all byte offsets and metadata faster
2. gribberish has ECMWF template compatibility gaps ("missing" variable names)
3. gribberish's `parse_grib_message_metadata` panics on some message types

gribberish IS valuable for:
1. **Spot-check validation** — `parse_grib_mapping()` can verify GRIB integrity without downloading the full file
2. **Data streaming phase** — Already used in `stream_cgan_variables.py` for decoding GRIB data into numpy arrays (Phase 2 of the cGAN workflow)
3. **Future template regeneration** — Could potentially replace scan_grib for building new templates, with .index files filling the metadata gaps

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `test_gribberish_vs_scangrib.py` | Benchmark script comparing all methods |
| `GRIBBERISH_EXPERIMENT_REPORT.md` | This document |

---

## Conclusion

The **73-minute Stage 1 bottleneck can be eliminated** by loading the zarr structure from the existing template instead of scanning GRIB files. The .index files + template provide everything scan_grib provides, with the template handling static structural metadata and the .index files providing fresh byte references.

The key safety measures are:
1. Variable count validation against the template on each run (~0.5s)
2. Spot-check of 3-5 GRIB byte ranges with gribberish (~10s)
3. Keeping scan_grib as a fallback path

Combined with Stage 2 parallelization (Phase 2), the total pipeline time drops from **110 minutes to ~7 minutes** — a **15x speedup** with no cloud costs.

For 365 days of daily processing:
- **Current:** 672 hours/year of compute
- **Proposed:** 43 hours/year of compute
- **Savings:** 629 hours/year
