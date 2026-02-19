# GIK ECMWF Pipeline: Architecture Evolution

Critical evaluation of the shift from `tutorial/ecmwf/run_ecmwf_tutorial.py`
to `cno-e4drr/devops/lithops_cr_ecmwf_gik/run_lithops_ecmwf.py`.

The inflection point is commit `e0a4167` ("Replace scan_grib with template-based Stage 1
for 68x speedup"), which triggered a chain of structural consequences that culminate in
the Lithops version being a fundamentally different beast.

---

## 1. Stage 1: The Defining Break

This is the single most important architectural change in the entire lineage.

### Tutorial (`run_ecmwf_tutorial.py`, line 159)

```python
from ecmwf_ensemble_par_creator_efficient import (
    process_ecmwf_files_efficiently,
    extract_individual_member_parquets,
    save_processing_metadata
)
```

Stage 1 scans live GRIB2 files from S3 byte-by-byte using kerchunk. The result is
a zip of per-member deflated parquet files that encode the zarr reference structure.
This is expensive: **~73 min for 51 members × 85 timesteps**.

The zip file then has to be passed as an artifact into Stage 2 via disk. The tutorial
wraps this in a check (`if zip_file exists: skip Stage 1`) which betrays how painful
the step is to re-run.

### Lithops (`run_lithops_ecmwf.py`, lines 161–236)

```python
def build_deflated_stores_from_template(template_tar_path, ...):
    with tarfile.open(template_tar_path, 'r:gz') as tar:
        # extract per-member .par files from HuggingFace archive
        template_df = pd.read_parquet(io.BytesIO(parquet_bytes))
        # build zstore dict directly in memory
```

Stage 1 is now pure in-memory parsing of a pre-built tar.gz (~5 seconds). No S3
GRIB reads, no kerchunk scanning, no zip artifact on disk. The zarr metadata structure
comes from a reference archive baked at `REFERENCE_DATE = '20240529'` and is reused
for every future date.

**Critical implication:** Stage 1 no longer produces date-specific output. It produces
a *template* of the zarr structure that is valid for any ECMWF IFS date because the
variable/dimension/chunk layout is fixed by the model configuration.

---

## 2. Stage 2: Index Parsing Replaces Parquet Merging

### Tutorial (via `ecmwf_three_stage_multidate.py` + `ecmwf_index_processor.py`)

Stage 2 reads the Stage 1 zip, unzips per-member parquet files, then fetches GCS
parquet templates and merges byte offsets from the existing `ecmwf_index_processor.py`
logic. This involves:
- Unzipping the Stage 1 artifact on disk
- Reading per-member parquet DataFrames
- Fetching template parquets from GCS (requires service account credentials)
- A complex merge of zarr keys and byte-range references

The GCS dependency (`gik-fmrc` bucket, `coiled-data-e4drr_202505.json`) is a hard
operational coupling that requires credential management in every deployment environment.

### Lithops (`run_lithops_ecmwf.py`, lines 243–344)

```python
def parse_grib_index(idx_url, member_filter=None):
    # reads ECMWF .index files (JSON-lines) from S3 directly
    entry = {
        'byte_offset': entry_data['_offset'],
        'byte_length': entry_data['_length'],
        ...
    }

def build_refs_from_indices(date_str, run, member_name, hours):
    idx_url = f"s3://{S3_BUCKET}/{date_str}/{run}z/.../...{hour}h-enfo-ef.index"
    # anonymous S3 read - no credentials
```

Stage 2 directly reads ECMWF's own `.index` files (JSON-lines with `_offset` /
`_length` byte ranges) from the same public S3 bucket as the GRIB files. No GCS,
no service account, no zip artifact from Stage 1. The byte ranges are assembled per
member per forecast hour using `ThreadPoolExecutor` for per-member parallelism.

**Critical implication:** The `.index` files that ECMWF publishes alongside every
GRIB2 file contain exactly the byte-range information that kerchunk was being used to
extract in Stage 1. The old architecture was re-deriving what ECMWF already provides.
The Lithops version simply reads it.

---

## 3. Self-Containment: The Cloudpickle Constraint

### Tutorial

```
run_ecmwf_tutorial.py
    └── imports ecmwf_ensemble_par_creator_efficient.py  (Stage 1, ~565 lines)
    └── imports ecmwf_three_stage_multidate.py           (Stage 2+3, ~700 lines)
            └── imports ecmwf_util.py                    (variable defs, ~1000 lines)
            └── imports ecmwf_index_processor.py         (merge logic, ~900 lines)
```

Five files, ~3200 lines, with GCS credential files required at runtime. The `sys.path`
manipulation (tutorial line 54-55) is a sign that the import structure is fragile:

```python
ECMWF_DIR = SCRIPT_DIR.parent.parent / "ecmwf"
sys.path.insert(0, str(ECMWF_DIR))
```

This breaks if the script is run from any directory other than its own.

### Lithops

**One file. Zero imports from the GIK codebase.**

```python
# All dependencies are standard packages (pandas, fsspec, requests, ...)
# No: from ecmwf_util import ...
# No: from ecmwf_index_processor import ...
# No: sys.path manipulation
```

The reason is architectural, not stylistic. Lithops serializes the worker function
(`process_ecmwf_date`) using `cloudpickle` and uploads it to GCS. `cloudpickle` can
capture closure variables and module-level constants, but it **cannot capture external
local Python files** that are not installed packages. Any import of a local `.py` file
would silently fail inside the Cloud Run container.

Self-containment is therefore a hard requirement of the Lithops execution model, not
an aesthetic choice.

---

## 4. Execution Model: Local vs Cloud-Native

| Dimension | Tutorial | Lithops |
|---|---|---|
| Execution target | Local machine | Google Cloud Run containers |
| Date scope | Single date per invocation | Ranges, `--days-back N`, `--start/end-date` |
| Member parallelism | `ProcessPoolExecutor` (local cores) | `ThreadPoolExecutor` within worker + N workers in parallel |
| Date parallelism | None | `fexec.map(process_ecmwf_date, dates)` |
| Output | Local `.parquet` files | GCS bucket `gs://gik-ecmwf-aws-tf/run_par_ecmwf/{date}/{run}z/` |
| Template source | Downloaded to CWD | Pre-baked into Docker image or cached at `/tmp` |
| Credentials | GCS service account JSON + anon S3 | Anon S3 only (GCS for upload uses ADC / workload identity) |
| Local test mode | Default | `--sequential` flag |

The `--sequential` flag in Lithops is the equivalent of the tutorial's full execution
path — it calls `process_ecmwf_date()` in a plain for-loop without Lithops overhead.
This makes the Lithops script fully testable locally.

---

## 5. Validation: Shift from Post-Hoc to Pre-Flight

### Tutorial

No index availability check before processing. Failures surface deep inside Stage 1
GRIB scanning, after 20-30 minutes of compute.

### Lithops (`run_lithops_ecmwf.py`, lines 595–631)

```python
def validate_index_availability(date_str, run):
    # opens the 0h .index file from S3
    # counts members, checks n_members >= 50
    # returns (bool, n_messages, n_members)
```

`validate_index_availability` is called as the **first step** inside `process_ecmwf_date`
before any template loading or Stage 2 processing begins. A date with missing or
incomplete index files fails fast with a clear message, not after wasted compute.

This matters even more in the cloud context: a Lithops worker is a billed Cloud Run
container. Failing fast on index unavailability avoids paying for a container that
will inevitably fail.

---

## 6. What the Tutorial Still Provides That Lithops Doesn't

The tutorial is not simply superseded. It retains value for:

- **Stage 1 with GRIB scanning** (`--run-stage1`): If the HuggingFace template is
  outdated (e.g., ECMWF changes the model grid or variable set), the tutorial pipeline
  can regenerate the template from scratch by re-scanning live GRIB files. The Lithops
  version has no fallback if `REFERENCE_DATE = '20240529'` template becomes stale.

- **Local data streaming to cGAN** (`stream_cgan_variables.py`): The tutorial ecosystem
  includes streaming the parquet output into cGAN-compatible variables on a local
  machine. The Lithops version terminates at GCS upload; what consumes the parquets
  afterward is out of scope.

- **Debugging and introspection**: The multi-file architecture exposes intermediate
  objects (zarr store dicts, per-member DataFrames) that are useful for debugging.
  The Lithops version discards all intermediate state in `/tmp` after completion.

---

## 7. Summary: What the Template-Based Stage 1 Actually Changed

The commit `e0a4167` said "68x speedup" but the deeper consequence was that it made
the pipeline *stateless with respect to the GRIB scan*. Once the zarr metadata
structure is fixed in a template, every future date just needs:

1. Template parquet (static, from HuggingFace, ~5s to load)
2. Fresh `.index` files (dynamic, from S3, ~5-15 min to fetch for 51 members × 85 hours)
3. A merge function (no disk, pure in-memory)

That is a pipeline that can run entirely inside a serverless container with no shared
filesystem, no artifact staging, and no service account for template access. The Lithops
architecture is the direct and logical consequence of that insight.

The tutorial's multi-file import structure (line 159 and its siblings) represents
the *pre-template* world where Stage 1 was a heavy, modular computation that needed
its own dedicated file and its output artifact passed forward. Once Stage 1 became a
5-second in-memory operation, the justification for that modularity collapsed.
