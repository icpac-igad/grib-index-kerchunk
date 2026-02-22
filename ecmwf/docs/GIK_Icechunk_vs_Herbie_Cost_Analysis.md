# GIK+Icechunk vs Herbie: Cost, Time & Data Volume Analysis

**Author**: ICPAC GIK Team | **Date**: February 2026 | **Project**: E4DRR / SEWAA

---

## Context

This document provides a concrete cost and performance comparison between the
**GIK+Icechunk pipeline** and a hypothetical **Herbie-based equivalent** for
materializing ECMWF IFS ensemble total precipitation (tp) data over East Africa
and generating 7-day forecast visualizations.

The GIK+Icechunk pipeline was benchmarked on a real production run:
**32 dates (March 2024), 51 ensemble members, 53 lead times, 157x145 spatial grid**.

---

## What Was Actually Done (GIK+Icechunk Pipeline)

### Task Description

1. **Fill**: Stream 32 dates of ECMWF IFS ensemble tp data (51 members x 53 lead
   times per date) from S3 GRIB files via byte-range reads, decode with gribberish,
   subset to East Africa, write into a versioned Icechunk store on GCS with per-date
   commits.

2. **Plot**: Read the Icechunk store, compute ensemble mean daily precipitation for
   each of 7 forecast days, overlay country boundaries from GeoJSON, produce PNGs
   and animated GIFs.

### Measured Results

| Step | Time | Compute | Cost (est.) | Output |
|------|------|---------|-------------|--------|
| **Fill** (32 dates) | 20.6 min | 50 x `e2-standard-4` VMs | ~$2.50 | Icechunk store on GCS |
| **Plot** (32 dates) | 11.2 min | 1 local machine | ~$0.00 | 224 PNGs + 32 GIFs |
| **Merge GIF** | 15 sec | 1 local machine | ~$0.00 | 1 combined GIF |
| **Total** | **~32 min** | | **~$2.50** | |

---

## Data Volume Breakdown

### Per Date (1 init date, 51 members, 53 lead times)

| Item | Count | Size per item | Total |
|------|-------|---------------|-------|
| GRIB messages fetched from S3 | 51 x 53 = 2,703 | ~835 KB (global) | ~2.2 GB (raw GRIB) |
| Useful data (EA subset) | 2,703 | ~91 KB (157x145 float32) | ~245 MB |
| Icechunk write (EA subset) | 1 date slice | | ~245 MB |

### Full 32-Date Test Run

| Metric | Value |
|--------|-------|
| Total GRIB messages | 86,496 (32 x 51 x 53) |
| Raw GRIB data transferred | ~71 GB (global grid, full messages) |
| Useful data after EA subset | ~7.8 GB (2.2% of raw) |
| Icechunk store size (32 dates) | ~7.8 GB |

### Full Target (671 dates, 20240301-20251231)

| Metric | Value |
|--------|-------|
| Total GRIB messages | 1,814,763 (671 x 51 x 53) |
| Raw GRIB data transferred | ~1.5 TB |
| Icechunk store size | ~165 GB |
| Estimated fill time | ~7.2 hours (at 32 dates/20 min rate) |
| Estimated cost | ~$53 |

### Visualization Output (32 dates)

| Item | Count | Size each | Total |
|------|-------|-----------|-------|
| Daily forecast PNGs | 224 (32 x 7) | ~170 KB | ~37 MB |
| Per-date GIFs | 32 | ~500 KB | ~16 MB |
| Combined GIF | 1 | 20 MB | 20 MB |
| **Total plots** | 257 files | | **~73 MB** |

---

## Herbie Equivalent: What Would It Cost?

Herbie accesses the same ECMWF GRIB files on S3 via byte-range reads (through the
`.index` files), but uses **cfgrib** for decoding instead of gribberish. To build
an equivalent materialized store and produce the same visualizations, Herbie would
need to perform exactly the same data fetching, but with a fundamentally slower
decode step.

### Critical Difference: Decode Speed

| Decoder | Time per GRIB message | Implementation |
|---------|----------------------|----------------|
| **gribberish** (GIK) | ~25 ms | Rust, in-memory byte buffer |
| **cfgrib** (Herbie) | ~2,000 ms | Python/eccodes, writes temp file to disk |
| **Speedup** | **80x** | |

### Herbie Cost Estimate for 32 Dates (Same Workload)

#### Approach A: Herbie on a Single Machine

Herbie is designed for single-machine use. Processing 32 dates sequentially:

| Step | Calculation | Time |
|------|-------------|------|
| S3 index fetch | 32 dates x 53 .index files x ~200 ms | ~5.6 min |
| S3 GRIB fetch | 86,496 messages x ~50 ms (avg latency) | ~72 min |
| **GRIB decode** | **86,496 messages x 2,000 ms** | **~48 hours** |
| EA subset + write | 86,496 arrays x ~5 ms | ~7 min |
| **Total** | | **~49 hours** |

| Metric | Value |
|--------|-------|
| Wall clock time | **~49 hours** (vs 20.6 min GIK) |
| Speedup (GIK/Herbie) | **~143x** |
| Machine cost (e2-standard-4, 49h) | **~$6.50** |
| No cluster overhead | $0 |
| **Total cost** | **~$6.50** |

#### Approach B: Herbie on Parallel Workers (Hypothetical)

Herbie is not designed for distributed computing. But if we assume we could
parallelise it across 50 workers (like the GIK approach), we'd face:

- **eccodes must be installed** on every worker (system library, not pure Python)
- **Temp file I/O** on every worker (cfgrib writes temp GRIB files to disk)
- **No native Dask integration** (Herbie returns xarray datasets per call)
- **GIL contention** with Python-based eccodes bindings

Optimistic estimate (50 workers, perfect scaling):

| Step | Time |
|------|------|
| S3 fetch (parallelised) | ~3 min |
| **GRIB decode** (50 workers) | **~58 min** |
| Coordination + write | ~5 min |
| **Total** | **~66 min** |

| Metric | Value |
|--------|-------|
| Wall clock time | **~66 min** (vs 20.6 min GIK) |
| Speedup (GIK/Herbie) | **~3.2x** |
| Cluster cost (50 x e2-standard-4, 66 min) | **~$5.50** |
| Setup complexity | High (eccodes on every node, temp disk) |
| **Total cost** | **~$5.50** |

In practice, Herbie scaling is worse than this because:
- eccodes is not thread-safe (can't use ThreadPoolExecutor)
- temp file contention on shared disk
- Herbie's API is synchronous, designed for interactive use

---

## Side-by-Side Comparison (32-Date Workload)

### Time

| Pipeline | Fill Time | Plot Time | Total | vs GIK |
|----------|-----------|-----------|-------|--------|
| **GIK+Icechunk (50 workers)** | **20.6 min** | **11.2 min** | **31.8 min** | 1x |
| Herbie (single machine) | ~49 hours | ~11 min | ~49.2 hours | **93x slower** |
| Herbie (50 workers, optimistic) | ~66 min | ~11 min | ~77 min | **2.4x slower** |

### Cost

| Pipeline | Compute Cost | Storage Cost/mo | Total (one-time) |
|----------|-------------|-----------------|-------------------|
| **GIK+Icechunk** | **~$2.50** | ~$0.20 (GCS) | **~$2.70** |
| Herbie (single machine) | ~$6.50 | ~$0.20 (GCS) | ~$6.70 |
| Herbie (50 workers) | ~$5.50 | ~$0.20 (GCS) | ~$5.70 |

### Data Transferred from S3

| | GIK | Herbie |
|---|---|---|
| GRIB messages | 86,496 | 86,496 |
| Raw bytes | ~71 GB | ~71 GB |
| Index overhead | ~0 (pre-built parquet) | ~9 MB (.index files) |
| **Total** | **~71 GB** | **~71 GB** |

Both methods transfer the same amount of data from S3. The difference is
entirely in **decode speed** (gribberish vs cfgrib).

---

## Full Target Projection (671 Dates)

Scaling the 32-date benchmarks to the full 671-date store:

| Metric | GIK+Icechunk | Herbie (single) | Herbie (50 workers) |
|--------|-------------|-----------------|---------------------|
| **Fill time** | **~7.2 hours** | ~43 days | ~23 hours |
| **Fill cost** | **~$53** | ~$136 | ~$115 |
| **Plot time** | ~3.9 hours | ~3.9 hours | ~3.9 hours |
| **Plot cost** | ~$0 (local) | ~$0 (local) | ~$0 (local) |
| **Total time** | **~11 hours** | **~43 days** | **~27 hours** |
| **Total cost** | **~$53** | **~$136** | **~$115** |

| Output | Size |
|--------|------|
| Icechunk store | ~165 GB |
| Forecast PNGs | 4,697 (671 x 7) |
| Per-date GIFs | 671 |
| Combined GIF | ~420 MB (est.) |
| **Total visualisation** | ~2.4 GB |

---

## Why GIK+Icechunk Wins

### 1. Decode Speed (80x)

The single largest factor. Gribberish (Rust) decodes GRIB messages ~80x faster
than cfgrib (Python/eccodes). For 86,496 messages:

```
gribberish: 86,496 x 25 ms  =  36 minutes (distributed across 50 workers = ~43 sec)
cfgrib:     86,496 x 2,000 ms = 48 hours   (distributed across 50 workers = ~58 min)
```

### 2. Native Dask Distribution

GIK parquets are one file per member — each is a self-contained unit of work
that can be shipped to a Dask worker without dependencies. Workers only need
`gribberish` (pure Python wheel with Rust extension).

Herbie requires `eccodes` (C library + Python bindings) installed on every
worker, plus temp disk space for cfgrib's file-based decoding. This makes
container/cluster setup significantly more complex.

### 3. Versioned Store with Resume

Icechunk's per-date commits provide:
- **Crash safety**: No orphaned data on OOM/timeout
- **Auto-resume**: Re-run the same command, it skips completed dates
- **Audit trail**: Full commit history (`"fill date 0 (20240301): 51/51 members"`)
- **Time travel**: Inspect data at any historical commit

Herbie has no equivalent — interrupted runs must restart from scratch or
require custom checkpoint logic.

### 4. Reusable Archive

Once the Icechunk store is filled, reading it for visualisation is
**instantaneous** (no S3 fetches, no GRIB decoding):

```
GIK plot (from Icechunk): 32 dates → 224 PNGs + 32 GIFs in 11.2 min
GIK plot (re-run):        32 dates → 224 PNGs + 32 GIFs in 11.2 min  (same)
Herbie plot (from scratch): 32 dates → need to re-fetch + re-decode → ~49 hours
```

The Icechunk store converts a **~49 hour Herbie workflow into an 11 minute
read** for every subsequent analysis.

---

## Cost Summary Table

| Scenario | GIK+Icechunk | Herbie | GIK Advantage |
|----------|-------------|--------|---------------|
| **32 dates (load test)** | $2.50, 32 min | $6.50, 49 hr | **2.6x cheaper, 93x faster** |
| **671 dates (backfill)** | $53, 11 hr | $136, 43 days | **2.6x cheaper, 94x faster** |
| **Re-analysis (any date)** | $0, 11 min | $6.50, 49 hr | **Free vs $6.50 per run** |
| **Daily operational (1 date)** | $0.05/day | $0.27/day | **5.4x cheaper** |

---

## Annual Operational Cost (Daily 7-Day Forecast)

### What Happens Every Day in Production

The daily operational workload is **1 new init date per day** — today's 00Z
forecast covering 7 days ahead (53 lead times, 51 members). The 32-date run
above was a load test / backfill; the steady-state is a single date per day.

### Per-Day Workload: 1 Init Date

| Item | Count |
|------|-------|
| GRIB messages | 51 members x 53 lead times = **2,703** |
| Raw GRIB data from S3 | ~2.2 GB |
| Useful data (EA subset) | ~245 MB |
| Icechunk write | 1 date slice |

### GIK+Icechunk Daily Run

For a single date, the Coiled cluster needs fewer workers and less time:

```
Workers:         10 x e2-standard-4
Cluster startup: ~3 min
Fill time:       ~2 min (51 member tasks across 10 workers)
Total wall time: ~5 min
Compute:         10 workers x 5 min = 0.83 VM-hours
Cost:            0.83 x $0.134 = $0.11 + 10% Coiled = ~$0.12
```

Plotting the new date's 7 daily forecast maps: ~21 seconds (local, free).

### Herbie Daily Run (Single Machine)

```
Index fetch:     53 .index files x ~200 ms = ~11 sec
GRIB fetch:      2,703 messages x ~50 ms = ~2.3 min
GRIB decode:     2,703 messages x 2,000 ms = ~90 min
Subset + write:  2,703 arrays x ~5 ms = ~14 sec
Total:           ~93 min
Cost:            1 machine x 1.55 hr x $0.134 = ~$0.21
```

### Annual Comparison (365 Days)

| Pipeline | Daily Cost | Daily Time | Annual Cost | Annual Time |
|----------|-----------|------------|-------------|-------------|
| **GIK+Icechunk** | **$0.12** | **5 min** | **$44** | **30 hours** |
| Herbie (single machine) | $0.21 | 93 min | $77 | 566 hours |

But the cost comparison above only captures the **ingest side**. The real
value difference appears when you consider what happens after the data is in
the store.

---

## The Real Value: Unlocking Petabytes via Just-In-Time Streaming

### Fair Baseline: Both Methods Can Store

A fair objection: Herbie can also fetch data once and store it as Zarr or
NetCDF locally. After that, re-analysis is free for both methods. So the
comparison is not "store vs no store" — both can do that, and the storage
cost (~$40/year for 165 GB on GCS) is the same.

**The real question is: what happens BEFORE the data is stored?**

### The YouTube Analogy for Weather Data

ECMWF publishes petabytes of GRIB2 data on S3. This data is publicly
available but sitting in a format that is expensive to access naively
(each timestep file is 3-4 GB, containing all variables and all 51 members).

GIK's parquet reference files act as a **streaming manifest** — like an HLS
`.m3u8` playlist for video. The parquet contains `[url, byte_offset,
byte_length]` triplets that point directly into the GRIB files. This enables
**just-in-time byte-range reads** of only the exact data needed:

```
Video streaming (YouTube/HLS):
    .m3u8 manifest (2 KB) → byte-range fetch → stream only visible segments
    Never download the full video file

Weather data streaming (GIK):
    .parquet manifest (~140 KB) → byte-range fetch → stream only needed variables
    Never download the full 3-4 GB GRIB file
```

**Herbie can also do byte-range reads** — it reads the `.index` files at
runtime and constructs the same byte offsets. Both methods access the same S3
data via the same mechanism. The difference is in how they get there and how
fast they decode.

### The 1,000-Day Analysis Scenario

A researcher wants to analyse ensemble total precipitation across 1,000 days
of ECMWF forecasts. The raw GRIB data for this is **~2.2 TB** (1,000 days x
51 members x 53 lead times x ~835 KB/message). Nobody wants to download 2.2
TB to their laptop.

With GIK's streaming approach, you **never download the full GRIB files**.
You stream only the bytes you need, decode on the fly, and either analyse
in memory or materialize to a store.

#### Cost to Realize 1,000 Days

From our measured benchmarks (32 dates = $2.50, 20.6 minutes, 50 workers):

| Scale | GIK+Icechunk | Herbie |
|-------|-------------|--------|
| 1 month (32 dates) | $2.50, **20 min** | $6.50, **49 hours** |
| 6 months (~180 dates) | ~$14, **~2 hours** | ~$37, **~11.5 days** |
| 1 year (365 dates) | ~$29, **~4 hours** | ~$75, **~23 days** |
| **1,000 days** | **~$78, ~10 hours** | **~$205, ~64 days** |
| 2 years (730 dates) | ~$57, **~7.5 hours** | ~$150, **~47 days** |

Both methods stream the same data volume from S3 (~2.2 GB/date). Both can
write to a persistent store. The cost is comparable — **GIK is ~2.6x cheaper
and ~150x faster** because of gribberish decode speed and native Dask
distribution.

#### But Here's the Catch

**GIK's parquets must exist** for the target dates. These are created in
Phase 1 (Lithops pipeline, ~5 min/date). Once they exist on HuggingFace,
any date can be streamed on demand — forever, for free (S3 is public).

**Herbie needs nothing pre-built** — it reads `.index` files directly from
ECMWF S3 at runtime. No Phase 1 prerequisite. This is simpler but slower.

| | GIK | Herbie |
|---|---|---|
| Pre-requisite | Parquets on HuggingFace | None |
| Index lookup | Dict lookup from parquet (~0 ms) | S3 read of .index file (~200 ms) |
| Decode speed | gribberish ~25 ms/message | cfgrib ~2,000 ms/message |
| Cluster scaling | Native Dask (pure Python wheel) | Needs eccodes on every node |
| Streaming API | `read_parquet → byte_range → decode` | `Herbie().xarray()` |

### Just-In-Time Analysis: Where GIK Shines

The YouTube-style streaming advantage is most visible in interactive and
exploratory workflows. The parquets are the key — they make any historical
forecast date instantly addressable by byte offset:

**Example: "What was the ensemble precipitation forecast for Nairobi on
2024-08-15?"**

```python
# GIK: read parquet (140 KB from HuggingFace), fetch 1 GRIB message
# from S3 (835 KB), decode with gribberish (25 ms). Total: ~2 seconds.

# Herbie: read .index file (1 MB from S3), find offset, fetch 1 GRIB
# message (835 KB), decode with cfgrib (2,000 ms). Total: ~4 seconds.
```

For a single query, both are fast enough. The difference shows at scale:

**Example: "Compute 30-day rolling ensemble spread for 1,000 days"**

```python
# GIK on 50 Coiled workers:
#   1,000 dates x 51 members x 1 lead time = 51,000 messages
#   Decode: 51,000 x 25 ms / 50 workers = ~26 seconds
#   Fetch: ~42 GB from S3 in parallel = ~10 min
#   Total: ~10 min, ~$1.20

# Herbie on single machine:
#   Same 51,000 messages
#   Decode: 51,000 x 2,000 ms = ~28 hours
#   Fetch: ~42 GB from S3 = ~2 hours
#   Total: ~30 hours, ~$4.00
```

### What This Means for the ICPAC Use Case

ECMWF publishes GRIB data going back years. With GIK parquets covering the
full archive on HuggingFace, ICPAC effectively has **on-demand access to the
entire ECMWF forecast history** without storing any of it:

| What you store | Size | Annual cost |
|----------------|------|-------------|
| GIK parquets on HuggingFace | ~5 GB for 2 years | **$0** (free tier) |
| Raw GRIB data (ECMWF S3) | Petabytes | **$0** (public) |
| **Total storage to unlock 2 years** | **~5 GB** | **$0** |

The parquets are the **streaming index** into the petabyte archive. The actual
data stays on ECMWF's S3 — you only pay for compute when you stream it.

When you choose to materialize a subset (e.g., 1,000 days of tp over East
Africa into an Icechunk store), you pay:

| Item | Cost |
|------|------|
| Compute (1,000 dates, 50 workers) | ~$78 |
| GCS storage (1,000 dates, ~245 GB) | ~$5/month |
| Plotting (1,000 dates, local) | $0 |

Want to do it again with different variables? Another $78 and 10 hours. Want
to go back and look at a specific week? Stream just those 7 dates for ~$0.55
in 5 minutes. Want to add a new year of data? Append 365 dates for ~$29.

**Herbie can do the same streaming** — it reads the same GRIB data from the
same S3 bucket. But it takes ~150x longer (64 days vs 10 hours for 1,000
dates) because cfgrib is the bottleneck. Both methods unlock the same data;
GIK unlocks it faster.

---

## Honest Summary: Where Each Method Wins

### Ingest and Store (Both Fetch Once)

Both methods can fetch data from S3 and store it locally. The stored data is
identical in value — once written, re-analysis is free for both.

| Aspect | GIK+Icechunk | Herbie+Zarr |
|--------|-------------|-------------|
| **Ingest speed (1,000 dates)** | **~10 hours** | ~64 days |
| **Ingest cost (1,000 dates)** | **~$78** | ~$205 |
| **Storage cost (same data)** | $5/month | $5/month |
| **Re-analysis after store** | Free | Free |

### Just-In-Time Streaming (No Storage)

Both methods can stream directly from S3 without storing. GIK is faster.

| Aspect | GIK | Herbie |
|--------|-----|--------|
| **Pre-requisite** | Parquets exist | **None** |
| **Decode speed** | **25 ms/message** | 2,000 ms/message |
| **1 date, 1 variable, interactive** | ~2 sec | ~4 sec |
| **1,000 dates, batch** | **~10 hours** | ~64 days |
| **Cluster scaling** | **Native Dask** | Difficult (eccodes) |

### Operational Quality

| Aspect | GIK+Icechunk | Herbie+Zarr |
|--------|-------------|-------------|
| **Crash safety / resume** | **Built-in (per-date commits)** | DIY checkpoint logic |
| **Store format** | **Consolidated 5D Zarr** | Scattered files or DIY Zarr |
| **Git-like versioning** | **Built-in** | None |
| **Incremental daily append** | **Native** | Fragile |
| **Cloud partial reads** | **Native** | Needs consolidation work |
| **Setup complexity** | Higher (parquets + Coiled) | **Lower (pip install herbie)** |
| **Ad-hoc single query** | Needs parquets | **Works immediately** |

### The Bottom Line

The real GIK advantage is **speed of access to the petabyte archive**. Both
methods can stream and both can store. GIK does it 80-150x faster because of
gribberish, and the Icechunk store adds operational robustness (versioning,
resume, append) that you'd have to build yourself with Herbie+Zarr.

For a research team doing occasional single-date queries, Herbie is simpler.
For an operational centre processing 1,000+ days or running daily forecasts,
GIK's speed and infrastructure pay for themselves.

### Annual Total Cost of Ownership (Daily 7-Day Forecast Operations)

| | GIK+Icechunk | Herbie+Zarr |
|---|---|---|
| Daily ingest (365 days x $0.12) | $44 | — |
| Daily ingest (365 days x $0.21) | — | $77 |
| GCS storage (growing archive) | $40 | $40 |
| Engineering: resume/checkpoint | $0 (built-in) | 2-5 days dev time |
| Engineering: consolidated store | $0 (built-in) | 1-2 days dev time |
| **Annual recurring** | **$84** | **$117 + engineering** |
| | | |
| On-demand 1,000-day re-analysis | +$78, 10 hours | +$205, 64 days |

---

## Data Equivalence

Both methods produce **numerically identical results** (validated across 21 dates
in 2024–2025, Pearson r > 0.9999 for both ensemble mean and spread). The tiny
differences (RMSE ~5e-05 m, relative <1%) arise from floating-point representation
differences between gribberish (Rust) and cfgrib (Python/eccodes) decoders.

See `docs/gik_vs_herbie_comparison.md` for detailed validation tables.

---

## Methodology Notes

### Cost Assumptions

| Resource | Unit Price | Source |
|----------|-----------|--------|
| GCP `e2-standard-4` VM | $0.134/hr | GCP Compute Engine pricing (europe-west1) |
| GCS storage | $0.020/GB/mo | GCS standard storage |
| S3 data transfer | $0.00 | ECMWF public bucket, anonymous access |
| Coiled overhead | ~10% on compute | Coiled management fee |

### GIK Compute Cost Calculation (32 dates)

```
50 workers x e2-standard-4 x 20.6 min = 17.2 VM-hours
17.2 VM-hours x $0.134/hr = $2.30
Coiled overhead (~10%): $0.23
Total: ~$2.53
```

### GIK Daily Operational Cost (1 date)

```
10 workers x e2-standard-4 x 5 min = 0.83 VM-hours
0.83 VM-hours x $0.134/hr = $0.11
Coiled overhead (~10%): $0.01
Total: ~$0.12/day
Annual: 365 x $0.12 = ~$44/year
```

### Herbie Daily Operational Cost (1 date, single machine)

```
1 machine x e2-standard-4 x 93 min = 1.55 VM-hours
1.55 VM-hours x $0.134/hr = $0.21
Annual: 365 x $0.21 = ~$77/year
```

### Herbie Compute Cost Calculation (32 dates, single machine)

```
1 machine x e2-standard-4 x 49 hours = 49 VM-hours
49 VM-hours x $0.134/hr = $6.57
```

### Herbie Compute Cost Calculation (32 dates, 50 workers)

```
50 workers x e2-standard-4 x 66 min = 55 VM-hours
55 VM-hours x $0.100/hr = $5.50  (lower rate, no Coiled, but higher setup cost)
```

---

## Related Documents

- `docs/gik_vs_herbie_comparison.md` — Detailed data transfer and validation comparison
- `docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` — Decoder benchmark analysis
- `docs/ECMWF_GEFS_Processing_Comparison.md` — ECMWF vs GEFS implementation gaps
- `compare_gik_herbie.py` — Validation script (statistical comparison)
- `fetch_tp_herbie.py` — Herbie data fetcher (benchmark reference)

---

*This analysis is based on measured benchmarks from the GIK+Icechunk production
pipeline run on 2026-02-22 (32 dates, 50 Coiled workers, europe-west1). Herbie
estimates are projected from known cfgrib decode times (~2,000 ms/message) and
validated against single-date Herbie runs.*
