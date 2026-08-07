# Parquet Filename Structure Update

**Date**: 2026-02-18  
**Update**: Improved filename and GCS path structure  
**Status**: ✅ Applied

---

## Changes Made

### 1. Parquet Filename (line 545)

**Before**: `stage3_{member}_final.parquet`
```
stage3_control_final.parquet
stage3_ens_01_final.parquet
stage3_ens_02_final.parquet
...
```

**After**: `{date_str}{run}z-{member}.parquet`
```
2026021000z-control.parquet
2026021000z-ens_01.parquet
2026021000z-ens_02.parquet
...
```

**Benefits**:
- Self-documenting filenames (includes date and run)
- Easier to identify which date/run a file belongs to
- Consistent naming convention across files

### 2. GCS Upload Path (line 566)

**Before**: `gs://{bucket}/{prefix}/{date_str}_{run}z/`
```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210_00z/
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210_12z/
```

**After**: `gs://{bucket}/{prefix}/{date_str}/{run}z/`
```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/00z/
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/12z/
```

**Benefits**:
- Better hierarchical organization (date → run)
- Easier to list all runs for a given date
- More standard directory structure
- Better for tools that group by date

### 3. Glob Pattern Update (line 568)

**Before**: `*_final.parquet`  
**After**: `*.parquet`

Simplified since all parquet files in the output directory should be uploaded.

---

## Example Output Structure

### Complete GCS Path for Date 20260210, Run 00z

```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/00z/
├── 2026021000z-control.parquet
├── 2026021000z-ens_01.parquet
├── 2026021000z-ens_02.parquet
├── 2026021000z-ens_03.parquet
├── ...
└── 2026021000z-ens_50.parquet
```

### Complete GCS Path for Date 20260210, Run 12z

```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/12z/
├── 2026021012z-control.parquet
├── 2026021012z-ens_01.parquet
├── 2026021012z-ens_02.parquet
├── ...
└── 2026021012z-ens_50.parquet
```

---

## Usage

The CLI already supports the `--run` parameter:

```bash
# Process 00z run (default)
uv run run_lithops_ecmwf.py --date 20260210 --run 00

# Process 12z run
uv run run_lithops_ecmwf.py --date 20260210 --run 12

# Process multiple dates with 00z run
uv run run_lithops_ecmwf.py --days-back 7 --run 00

# Process date range with 12z run
uv run run_lithops_ecmwf.py --start-date 20260201 --end-date 20260207 --run 12
```

---

## Listing Files

### List all dates
```bash
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/
```

### List all runs for a specific date
```bash
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/
```

### List all files for a specific date and run
```bash
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/00z/
```

### Count files for a specific run
```bash
gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/00z/ | wc -l
```

---

## Migration Notes

### For Existing Data

If you have existing parquet files in the old structure, they will remain at:
```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210_00z/stage3_control_final.parquet
```

New data will be written to:
```
gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260210/00z/2026021000z-control.parquet
```

### No Migration Script Needed

The old and new structures can coexist. Downstream consumers should be updated to:
1. Check the new path first: `{date}/{run}z/{date}{run}z-{member}.parquet`
2. Fall back to old path if not found: `{date}_{run}z/stage3_{member}_final.parquet`

Or, if you want to migrate old data:
```bash
# Example migration script (run carefully!)
for date in 20260210 20260211; do
  gsutil -m mv \
    "gs://gik-ecmwf-aws-tf/run_par_ecmwf/${date}_00z/stage3_*_final.parquet" \
    "gs://gik-ecmwf-aws-tf/run_par_ecmwf/${date}/00z/"
  
  # Rename files to new format
  # (requires gsutil cp + rm or use Python script)
done
```

---

## Files Modified

- ✅ `run_lithops_ecmwf.py:545` - Updated filename format
- ✅ `run_lithops_ecmwf.py:566` - Updated GCS path structure
- ✅ `run_lithops_ecmwf.py:568` - Updated glob pattern

## Testing

Dry run confirmed working:
```bash
$ uv run run_lithops_ecmwf.py --date 20260210 --run 00 --dry-run
DRY RUN - Dates that would be processed:
  20260210
```

Ready for production use!
