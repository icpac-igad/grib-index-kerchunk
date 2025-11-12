# ECMWF Implementation Critical Evaluation and Action Plan

## Visual Summary

```
Current ECMWF Pipeline (BROKEN - Only 2-3 timesteps):
┌─────────────┐     ┌─────────────┐                        ┌─────────────┐
│  Stage 0    │     │  Stage 1    │                        │  Stage 3    │
│ GCS Templates│     │ Scan GRIB   │                        │ Create Zarr │
│    ✅        │     │  (2-3 files)│                        │ (2-3 steps) │
│ IMPLEMENTED │     │     ✅       │     ❌ MISSING ❌      │     ⚠️      │
│ Ready to Run│     │   Working   │      Stage 2           │  Limited    │
└─────────────┘     └─────────────┘                        └─────────────┘

Required ECMWF Pipeline (TARGET - All 85 timesteps):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Stage 0    │     │  Stage 1    │     │  Stage 2    │     │  Stage 3    │
│ GCS Templates│────▶│ Scan GRIB   │────▶│Index + GCS  │────▶│ Create Zarr │
│  (85 files) │     │  (2 files)  │     │ (85 files)  │     │ (85 steps)  │
│     ✅       │     │     ✅       │     │   ❌ TODO   │     │     ✅       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Executive Summary

After comprehensive analysis of both GEFS and ECMWF implementations, the critical missing component is **Stage 2: The Integration of Index Files with GCS Templates**. ECMWF has Stage 0 (GCS template creation) and Stage 1 (scan_grib) implemented, but lacks the crucial Stage 2 integration that combines fresh index files with pre-built GCS templates to efficiently process all 85 timesteps.

**Key Finding**: ECMWF's current implementation only processes 2-3 timesteps via scan_grib, missing 82-83 of the required 85 timesteps. Stage 0 (template creation) is already available in `ecmwf_index_preprocessing.py`, but Stage 2 (the integration layer) is completely missing.

---

## 1. Current State Analysis

### 1.1 GEFS Implementation (Working - 81 Timesteps)

The GEFS system successfully processes all **81 timesteps** using a three-stage pipeline:

#### Stage 0: One-Time GCS Template Creation
- **File**: Not shown, but referenced in `gefs_util.py`
- **Process**: Creates reusable parquet mappings in GCS bucket
- **Location**: `gs://gik-fmrc/gefs/{member}/`
- **Frequency**: One-time per member (30 members)
- **Output**: 81 parquet files per member (one per timestep)

#### Stage 1: GRIB Scanning (Minimal)
- **Files**: `run_day_gefs_ensemble_full.py`, `gefs_util.py`
- **Process**: Scans only 2 GRIB files (0h, 3h) for structure
- **Function**: `filter_build_grib_tree(gefs_files, forecast_dict)`
- **Output**: Deflated GRIB tree store with variable references

#### Stage 2: Index-Based Mapping (Critical - 81 timesteps)
- **Files**: `gefs_util.py`
- **Key Function**: `cs_create_mapped_index()` → `process_gefs_files_in_batches()`
- **Process**:
  ```python
  # Process ALL 81 forecast hours (0-240h at 3h intervals)
  valid_time_index = pd.date_range(start_date, end_date, freq="180min")  # 81 steps

  # For each timestep, combines:
  # 1. Fresh index file from target date (binary positions)
  # 2. Pre-built GCS template from reference date (variable structure)
  ```
- **Output**: Complete mapped index DataFrame with all 81 timesteps

#### Stage 3: Final Zarr Assembly
- **Function**: `process_unique_groups()` with `time_dims={'valid_time': 81}`
- **Output**: Complete parquet with all 81 timesteps

### 1.2 ECMWF Current Implementation (Incomplete - Only 2-3 Timesteps)

#### What Exists:
1. **Stage 0 Implementation** (`ecmwf_index_preprocessing.py`) ✅
   - **FULLY IMPLEMENTED**: Creates GCS parquet templates
   - Processes all 85 forecast hours (3h intervals 0-144h, 6h intervals 150-360h)
   - Saves to GCS bucket with service account access (`coiled-data-e4drr_202505.json`)
   - Output structure: `gs://gik-ecmwf-aws-tf/ecmwf/{member}/ecmwf-time-{date}-{member}-rt{hour:03d}.parquet`
   - **Status**: Ready to use - just needs to be run once per member

2. **Stage 1 Implementation** (`ecmwf_ensemble_par_creator_efficient.py`) ✅
   - Successfully scans 2-3 GRIB files
   - Uses `ecmwf_filter_scan_grib()` and `fixed_ensemble_grib_tree()`
   - Creates deflated store for parquet
   - **CRITICAL LIMITATION**: Only processes files explicitly scanned (2-3 timesteps)

3. **Index Processor Started** (`ecmwf_index_processor.py`) ⚠️
   - Has basic index parsing logic
   - Defines all 85 forecast hours correctly
   - **BUT**: Not integrated with GCS templates or main pipeline

4. **Data Streaming** (`aifs-etl.py`) ✅
   - Successfully extracts data from parquet to numpy/pkl
   - Handles S3 byte-range fetching
   - Works with whatever timesteps are available

#### What's Missing:

1. **Stage 2: Complete Integration Layer** ❌
   - No `cs_create_mapped_index()` equivalent for ECMWF
   - No function to combine fresh index files with GCS templates
   - No async batch processing integration for 85 timesteps
   - Missing the critical `map_from_index()` integration

2. **Time Dimension Handling** ❌
   - No `generate_ecmwf_axes()` function for 85 timesteps
   - Missing variable interval handling integration (3h then 6h)

3. **Main Runner Script** ❌
   - No `run_day_ecmwf_ensemble_full.py` that integrates all stages
   - No orchestration of Stage 1 → Stage 2 → Stage 3 pipeline

---

## 2. Critical Gap Analysis

### 2.1 The Core Problem

**ECMWF is missing the entire Stage 2 pipeline that would give it all 85 timesteps.**

Current flow:
```
ECMWF: Stage 1 (2-3 files) → Stage 3 (2-3 timesteps only) ❌
GEFS:   Stage 1 (2 files) → Stage 2 (81 index files) → Stage 3 (81 timesteps) ✅
```

### 2.2 Specific Missing Components

| Component | GEFS (Has) | ECMWF (Status) | Impact |
|-----------|------------|----------------|---------|
| **GCS Template Creation** | Pre-built templates for all 81 hours | ✅ **IMPLEMENTED** in `ecmwf_index_preprocessing.py` | Ready to use |
| **Index File Processing** | `process_gefs_files_in_batches()` | ⚠️ Basic parser only, no batch processing | Cannot process 85 hours efficiently |
| **Integration Layer** | `cs_create_mapped_index()` | ❌ **MISSING** - Critical gap | Cannot combine index + templates |
| **Async Batch Processing** | Processes 81 files in parallel batches | ❌ Not implemented | Would take hours instead of minutes |
| **Time Axis Generation** | `generate_axes()` for 81 steps | ❌ No equivalent for 85 steps | Cannot create proper time dimensions |
| **map_from_index() Usage** | Combines index + template | ❌ Not integrated | Cannot merge fresh positions with structure |
| **Reference Date System** | Uses reference_date_str for templates | ⚠️ Structure exists, not integrated | Cannot reuse templates efficiently |

### 2.3 Why Current ECMWF Approach Fails

The current ECMWF implementation tries to use `scan_grib` for everything, but:

1. **Performance**: Scanning 85 GRIB files (500MB-1GB each) takes 60-120 minutes
2. **Memory**: Requires 16-32 GB RAM to hold all scanned data
3. **Network**: Downloads 40-80 GB of GRIB data
4. **Scalability**: Cannot process 51 members × 85 timesteps efficiently

The GEFS solution using index files + GCS templates:
1. **Performance**: 3-5 minutes for all timesteps
2. **Memory**: < 1 GB RAM
3. **Network**: Downloads only KB of index files
4. **Scalability**: Can process all members in parallel

---

## 3. Detailed Action Plan

### Phase 0: Run Existing GCS Template Creation (Prerequisite)

**Status**: ✅ **Already Implemented** in `ecmwf_index_preprocessing.py`

**Action Required**: Run once to create GCS templates

```bash
# Create templates for all 51 members (one-time, 25-40 hours total)
python ecmwf_index_preprocessing.py \
  --date 20240529 \
  --bucket gik-ecmwf-aws-tf \
  --all-members

# Or test with single member first (30-50 minutes)
python ecmwf_index_preprocessing.py \
  --date 20240529 \
  --bucket gik-ecmwf-aws-tf \
  --member control
```

**Service Account**: Use `coiled-data-e4drr_202505.json` for GCS access

**Expected Output**:
```
gs://gik-ecmwf-aws-tf/ecmwf/{member}/ecmwf-time-20240529-{member}-rt{hour:03d}.parquet
```
- 85 parquet files per member (one for each forecast hour)
- Total: 51 members × 85 hours = 4,335 template files

### Phase 1: Create ECMWF Time Generation (Priority: CRITICAL)

**File to enhance**: `ecmwf_util.py` (add missing function)

def generate_ecmwf_axes(date_str: str) -> List[pd.Index]:
    """Generate 85 timestep axes for ECMWF (0-360h forecast)."""
    start_date = pd.Timestamp(date_str)

    # 3-hour intervals from 0-144h (49 steps)
    times_3h = pd.date_range(start_date,
                            start_date + pd.Timedelta(hours=144),
                            freq="3h")

    # 6-hour intervals from 150-360h (36 steps)
    times_6h = pd.date_range(start_date + pd.Timedelta(hours=150),
                            start_date + pd.Timedelta(hours=360),
                            freq="6h")

    # Combine for total 85 steps
    valid_time_index = times_3h.append(times_6h)
    time_index = pd.Index([start_date], name="time")

    return [valid_time_index, time_index]
```

### Phase 2: Implement Stage 2 - Complete Index Integration Layer

**File to enhance**: `ecmwf_index_processor.py`

Add the critical async batch processing:

```python
async def process_single_ecmwf_file(
    target_date: str,
    hour: int,
    member: str,
    gcs_bucket: str,
    reference_date: str,
    semaphore: asyncio.Semaphore
) -> pd.DataFrame:
    """Process single ECMWF hour using index + GCS template."""

    async with semaphore:
        # 1. Parse fresh index for target date
        idx_url = f"s3://ecmwf-forecasts/{target_date}/00z/ifs/0p25/enfo/{target_date}000000-{hour}h-enfo-ef.index"
        idxdf = parse_grib_idx(basename=idx_url, storage_options={"anon": True})

        # Filter for member
        member_num = 0 if member == 'control' else int(member.replace('ens', ''))
        idxdf = idxdf[idxdf['attrs'].str.contains(f"number={member_num}")]

        # 2. Load GCS template from reference date
        gcs_path = f"gs://{gcs_bucket}/ecmwf/{member}/ecmwf-time-{reference_date}-{member}-rt{hour:03d}.parquet"
        gcs_fs = gcsfs.GCSFileSystem()
        template = pd.read_parquet(gcs_path, filesystem=gcs_fs)

        # 3. Map fresh positions with template structure
        from kerchunk._grib_idx import map_from_index
        mapped = map_from_index(
            run_time=pd.Timestamp(target_date),
            mapping=template,
            idxdf=idxdf
        )

        return mapped

async def cs_create_ecmwf_mapped_index(
    axes: List[pd.Index],
    gcs_bucket: str,
    target_date: str,
    member: str,
    reference_date: str,
    max_concurrent: int = 10
) -> pd.DataFrame:
    """Create complete mapped index for all 85 ECMWF timesteps."""

    # All 85 hours
    hours_3h = list(range(0, 145, 3))
    hours_6h = list(range(150, 361, 6))
    all_hours = hours_3h + hours_6h

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []

    # Process in batches
    batch_size = 10
    for i in range(0, len(all_hours), batch_size):
        batch = all_hours[i:i+batch_size]

        tasks = [
            process_single_ecmwf_file(
                target_date, hour, member, gcs_bucket,
                reference_date, semaphore
            )
            for hour in batch
        ]

        results = await asyncio.gather(*tasks)
        all_results.extend(results)

        print(f"✅ Processed batch {i//batch_size + 1}/{len(all_hours)//batch_size + 1}")

    # Combine all 85 timesteps
    ecmwf_kind = pd.concat(all_results, ignore_index=True)
    return ecmwf_kind
```

### Phase 4: Create Main Runner

**File to create**: `run_day_ecmwf_ensemble_full.py`

```python
#!/usr/bin/env python3
"""
ECMWF Full Ensemble Processing - Complete 85 Timesteps
Follows GEFS pattern but adapted for ECMWF's 85 timesteps
"""

import asyncio
from ecmwf_util import (
    generate_ecmwf_axes,
    filter_build_ecmwf_grib_tree,
    calculate_time_dimensions,
    prepare_zarr_store,
    process_unique_groups
)
from ecmwf_index_processor import cs_create_ecmwf_mapped_index

async def process_ecmwf_member(
    member: str,
    target_date: str,
    reference_date: str = '20240529'
) -> None:
    """Process single ECMWF member with all 85 timesteps."""

    print(f"🎯 Processing {member} for {target_date}")

    # Stage 1: Scan minimal GRIB files (just for structure)
    ecmwf_files = [
        f"s3://ecmwf-forecasts/{target_date}/00z/ifs/0p25/enfo/{target_date}000000-0h-enfo-ef.grib2",
        f"s3://ecmwf-forecasts/{target_date}/00z/ifs/0p25/enfo/{target_date}000000-3h-enfo-ef.grib2"
    ]

    _, deflated_store = filter_build_ecmwf_grib_tree(
        ecmwf_files, forecast_dict, member
    )
    print(f"✅ Stage 1: GRIB tree built")

    # Stage 2: Get all 85 timesteps via index + GCS templates
    axes = generate_ecmwf_axes(target_date)

    ecmwf_kind = await cs_create_ecmwf_mapped_index(
        axes=axes,
        gcs_bucket='gik-ecmwf-aws-tf',
        target_date=target_date,
        member=member,
        reference_date=reference_date
    )
    print(f"✅ Stage 2: Mapped {len(ecmwf_kind)} entries for 85 timesteps")

    # Stage 3: Create final zarr store with all timesteps
    time_dims = {'valid_time': 85}  # All 85 forecast hours
    time_coords = {
        'valid_time': 'valid_time',
        'time': 'time',
        'step': 'step'
    }

    time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
    zstore, chunk_index = prepare_zarr_store(deflated_store, ecmwf_kind)

    final_zstore = process_unique_groups(
        zstore, chunk_index, time_dims,
        time_coords, times, valid_times, steps
    )

    # Save parquet
    create_parquet_file_fixed(final_zstore, f"{member}_{target_date}.parquet")
    print(f"✅ Stage 3: Created {member}_{target_date}.parquet with 85 timesteps")

# Main execution
async def main():
    target_date = '20250101'
    members = ['control'] + [f'ens{i:02d}' for i in range(1, 51)]

    # Process all members
    for member in members:
        await process_ecmwf_member(member, target_date)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Implementation Timeline

### Immediate Action: Run Stage 0 (Can Start Now)
- [ ] **Day 0**: Run Stage 0 for control member to test (30-50 minutes)
- [ ] **Background Task**: Run Stage 0 for all 51 members (25-40 hours total, can run while developing)
  ```bash
  python ecmwf_index_preprocessing.py --date 20240529 --bucket gik-ecmwf-aws-tf --all-members
  ```

### Week 1: Foundation (CRITICAL PATH)
- [ ] Day 1: Implement `generate_ecmwf_axes()` in ecmwf_util.py
- [ ] Day 2: Test time dimension generation with 85 timesteps
- [ ] Day 3-4: Implement `process_single_ecmwf_file()` function
- [ ] Day 5: Test single file processing with GCS template integration

### Week 2: Core Integration
- [ ] Day 1-2: Implement `cs_create_ecmwf_mapped_index()` with async batching
- [ ] Day 3: Test batch processing for all 85 timesteps
- [ ] Day 4: Integration testing between Stage 1, Stage 2, and Stage 3
- [ ] Day 5: Performance optimization and error handling

### Week 3: Production Ready
- [ ] Day 1-2: Create `run_day_ecmwf_ensemble_full.py` main runner
- [ ] Day 3: Test full pipeline with single member
- [ ] Day 4: Parallel processing for multiple members
- [ ] Day 5: Full system validation and documentation

---

## 5. Validation Checklist

### Correct Implementation Indicators:
- [ ] `ecmwf_kind` DataFrame contains entries for all 85 timesteps
- [ ] Processing time per member: 8-10 minutes (not 60+ minutes)
- [ ] Memory usage: < 2 GB (not 16+ GB)
- [ ] Final parquet contains `valid_time` dimension with shape (85,)
- [ ] GCS bucket has 85 template files per member
- [ ] Index parsing shows different dates (target vs reference)

### Error Prevention:
- [ ] Member filtering working (number=0 for control, number=1-50 for ensemble)
- [ ] Time intervals correct (3h then 6h, not uniform)
- [ ] All 85 hours processed (not just 81 like GEFS)
- [ ] Template paths match actual GCS structure
- [ ] Async semaphores prevent memory overload

---

## 6. Performance Comparison

### Current ECMWF (Broken):
- Timesteps: 2-3 only ❌
- Time per member: N/A (incomplete)
- Memory: 8-16 GB
- Network: 2-3 GB

### After Implementation:
- Timesteps: 85 ✅
- Time per member: 8-10 minutes
- Memory: < 2 GB
- Network: < 100 MB (index files only)
- Speedup: 8-10x faster than full scan_grib

---

## 7. Risk Mitigation

### High Risk Areas:
1. **GCS Template Creation Failure**
   - Mitigation: Implement retry logic, save progress incrementally

2. **Index File Format Changes**
   - Mitigation: Add format validation, version checking

3. **Memory Overflow in Async Processing**
   - Mitigation: Semaphore limits, batch size tuning

4. **Network Timeouts**
   - Mitigation: Exponential backoff, connection pooling

---

## 8. Success Metrics

The implementation will be considered successful when:

1. **Functional**: Process all 85 timesteps for all 51 members
2. **Performance**: < 10 minutes per member
3. **Scalability**: Can process multiple members in parallel
4. **Reliability**: < 1% failure rate with retry logic
5. **Validation**: Output matches expected dimensions and can be opened with xarray

---

## Conclusion

The ECMWF implementation has **Stage 0 (GCS template creation) already implemented** in `ecmwf_index_preprocessing.py`, but is critically missing **Stage 2 (the integration layer)** that combines fresh index files with these templates to efficiently process all 85 timesteps. Without this integration, the system cannot scale beyond 2-3 timesteps and is fundamentally broken for production use.

The solution is clear:
1. **Run the existing Stage 0** to create GCS templates (one-time operation)
2. **Implement Stage 2 integration** following the GEFS pattern (`cs_create_mapped_index()`)
3. **Create the main runner** to orchestrate all stages

**Current Assets**:
- ✅ Stage 0: `ecmwf_index_preprocessing.py` (ready to run)
- ✅ Stage 1: `ecmwf_ensemble_par_creator_efficient.py` (working for 2-3 files)
- ✅ GCS Access: `coiled-data-e4drr_202505.json` service account

**Critical Missing Piece**: Stage 2 integration layer that uses `map_from_index()` to combine:
- Fresh index files from target date (binary positions)
- Pre-built GCS templates from reference date (variable structure)

**Estimated Implementation Time**: 2-3 weeks (Stage 0 can run in parallel)
**Expected Performance Gain**: 8-10x faster, 85 timesteps vs current 2-3
**Critical Success Factor**: Stage 2 integration must properly map all 85 timesteps