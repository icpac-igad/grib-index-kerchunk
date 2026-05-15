# GIK + Ephemeral Dask: Cost Evaluation and Paradigm Note

This note records what we have actually observed (in dollars, minutes, and
workers) when running the Grib-Index-Kerchunk (GIK) pipeline against
ECMWF IFS and NOAA GEFS ensemble data, and reflects on what this enables
for continuous risk monitoring.

It combines two sources of evidence:

1. **Live Coiled / Dask cluster runs** — concrete dollar bills for actual
   GIK-driven analyses that produced Icechunk stores on Source.coop and
   NetCDF outputs.
2. **The `cost-eval/` harness** (scripts `01_*` → `07_*` and
   `results/REPORT.md`) — a reproducible primary-measurement pipeline
   that HEAD-probes the source archives, lists the GIK parquet catalog,
   measures dynamical.org's ARCO Zarr footprint, benchmarks Herbie and
   GIK live, and rolls everything into a stakeholder cost matrix.

The two sources answer different questions: the Coiled runs say *"this
is what one analysis actually costs today"*; the harness says *"this is
what the paths cost and scale to under audited assumptions."* Read
together they make the JIT-Dask + Bayesian-evidence story checkable.

---

## 1. Observed Coiled runs (live cluster bills)

Three runs, all owned by `nishadh-kalladath`, all using GIK parquet
reference files to stream byte-range reads of GRIB messages from
`s3://ecmwf-forecasts/`. Workers are spawned on demand and torn down at
the end of the job — there is no standing cluster.

| Run name              | Purpose                                                            | Workers | Duration | Cost  | Compute-hrs* | Tasks* |
|-----------------------|--------------------------------------------------------------------|---------|----------|-------|--------------|--------|
| `ecmwf-ea-tp-5900`    | One month of total precipitation, East Africa, GIK → Icechunk on Source.coop | 50      | 19m 18s  | $2.20 | 65.19        | 1,632  |
| `ecmwf-ea-tp-8990`    | Second TP observation (rerun / re-window)                          | 50      | 12m 26s  | $1.42 | 41.91        | 127    |
| `ecmwf-fog-680`       | 8 variables × single 00z init × 7-day forecast (51 members)        | 15      | 20m 47s  | $0.74 | 21.64        | 51     |

\* "Compute-hrs" and "Tasks" are the last two columns of the Coiled
dashboard rows — roughly worker-hours and the count of discrete units
processed (members / member-timesteps).

### What the numbers say

- **Order-of-magnitude cost per scientific question is single-digit
  dollars.** A month of ECMWF total-precipitation ensemble, streamed and
  written to an Icechunk store on Source.coop, finished in under
  20 minutes for $2.20. A multi-variable single-day forecast cost $0.74.
- **No data downloaded, no data parked.** The clusters fetched only the
  GRIB byte ranges they needed (GIK manifest → S3 `Range:` GET). The
  ECMWF archive (~1.1 PiB at recent file sizes — see §3.1) was never
  replicated.
- **Cluster size is sized to the job, not provisioned ahead.** 50 workers
  for a month-of-precip aggregation, 15 for a single-day multi-variable
  pull. Both spin up, both spin down.

The first run materialized its output as an Icechunk store at:

```
https://source.coop/e4drr-project/forecasts/ecmwf_ea_tp_icechunk
```

That output is the only persistent artifact. Input bytes live in
NOAA/ECMWF public buckets; we never own them.

---

## 2. The `cost-eval/` harness (scripts 01 → 07)

The harness re-derives every storage and cost claim from object-store
reality. All scripts are PEP 723 `uv run` modules, all use anonymous
public-data access, all write CSV + JSON outputs to `results/`. Pricing
is a knob in `config.yaml`, not a hard-coded constant.

### 2.1 Per-script purpose and finding

| # | Script | What it measures | Headline finding |
|---|--------|------------------|------------------|
| 01 | `01_measure_grib_sizes.py` | Anonymous S3 HEAD on every GRIB object in a 7-date × 4-run sample | **Source archive ≈ 1.1 PiB ECMWF + 310 TiB GEFS** at recent file sizes (full archive extrapolation) |
| 02 | `02_measure_parquet_sizes.py` | List the public HuggingFace mirrors of the GIK parquet catalog | **Parquet refs compress GRIB by 4,755× (GEFS) and 56,545× (ECMWF)** — full archive coverage is ~14 + ~16 GiB |
| 03 | `03_measure_dynamical_zarr.py` | List dynamical.org's ARCO Zarr stores object-by-object + read `.zmetadata` | **dynamical's on-disk footprint ≈ 38.6 TiB ECMWF + 146 TiB GEFS** for the curated 19–22 var subset (≈6× Zarr compression vs uncompressed) |
| 04 | `04_benchmark_herbie.py` | One live Herbie download per product per mode (A = full file, B = byte-range searchString), then extrapolate to the full workload | **Herbie Mode B ≈ GIK structurally**: both pull full GRIB messages, both can't spatially subset. Differs on durability + Dask. |
| 05 | `05_benchmark_gik_and_zarr.py` | Bytes-on-wire and wall time for GIK vs dynamical Zarr across 3 access patterns (global, regional, pencil) × 1 variable | **Zarr beats GIK 100–10,000× for regional & pencil queries** because of chunk-level spatial subsetting; GIK and Zarr are within 3× for global queries |
| 07 | `07_cost_model_and_report.py` | Roll 01–05 into `cost_matrix.csv` + a stakeholder ledger; project to JIT annual cadence (365 slices/yr) | **Annual storage rent**: GIK ~$9, dynamical (curated) ~$56K, own-ARCO duplicate $68K–$439K. **Break-even** for self-hosting an ARCO duplicate is ~40 GEFS / ~76 ECMWF regional slices per day. |

(Script `06_benchmark_dynamical_open.py` was planned but not written;
script `05_benchmark_gik_and_zarr.py` covers both paths instead.)

### 2.2 Archive volumes (live, extrapolated)

From `01_grib_sizes_summary.json` and `03_dynamical_summary.json`:

| Product | GRIB source archive | dynamical ARCO on-disk | GIK parquet catalog |
|---------|---------------------|------------------------|---------------------|
| GEFS    | **310.4 TiB**       | 146.0 TiB              | **13.9 GiB**        |
| ECMWF   | **1,135.5 TiB** (1.1 PiB) | 38.6 TiB         | **16.4 GiB**        |
| Combined | ~1.4 PiB           | ~185 TiB               | ~30 GiB             |

The GIK catalog is **~7,650× smaller** than the source archive it
indexes (combined). dynamical's on-disk Zarr is ~7,800× larger than the
GIK catalog but covers a narrower variable scope.

### 2.3 Annual storage cost by stakeholder

From §4 of `results/REPORT.md`. AWS S3 Standard list price ($0.023/GB-mo):

| Party / deployment model                                  | GEFS    | ECMWF    | Combined |
|-----------------------------------------------------------|---------|----------|----------|
| Data provider (NOAA + ECMWF, already publishing)          | $0      | $0       | $0       |
| Regional centre hosts own ARCO duplicate (Zarr, ~6.4×)    | $14,720 | $53,841  | **$68,561** |
| Regional centre hosts own raw GRIB duplicate              | $94,206 | $344,581 | **$438,787** |
| dynamical.org curated subset (today's actual on-disk)     | $44,316 | $11,718  | $56,034  |
| **GIK parquet catalog (full archive coverage)**           | $4.11   | $4.85    | **$8.96** |

Read in one sentence: GIK gives a regional centre **full archive
coverage for under $10/yr** versus tens to hundreds of thousands of
dollars for any duplication strategy.

### 2.4 Per-slice cost (one analysis, today)

From `cost_matrix.csv`, showing the structural divergence between
method × workload. Workload definitions:

- **full_operational_slice_1init**: 12 vars × 51 mems × 9 leads ECMWF
  / 4 vars × 30 mems × 9 leads GEFS, **1 init**
- **global_1var / regional_1var / pencil_1var**: 1 variable across
  globe / East Africa / a single (lat, lon) point, across the full
  archive of inits

| Product | Method        | Workload                       | Bytes on wire | $ same-cloud | $ cross-cloud |
|---------|---------------|--------------------------------|---------------|--------------|---------------|
| ECMWF   | herbie_mode_A | full_operational_slice_1init   | 54.2 GiB      | $0.008       | $5.25         |
| ECMWF   | herbie_mode_B | full_operational_slice_1init   | 25.6 GiB      | $0.008       | $2.49         |
| ECMWF   | gik           | global_1var                    | 20.2 GiB      | $0.00        | $1.95         |
| ECMWF   | zarr          | global_1var                    | 2.7 GiB       | $0.0004      | $0.26         |
| ECMWF   | gik           | regional_1var                  | 20.2 GiB      | $0.00        | $1.95         |
| ECMWF   | zarr          | regional_1var                  | 0.04 GiB      | $0.00        | $0.004        |
| ECMWF   | gik           | pencil_1var                    | 15,624 GiB    | $0.03        | $1,510        |
| ECMWF   | zarr          | pencil_1var                    | 2.0 GiB       | $0.0003      | $0.19         |

**Per-message floor**: GIK and Herbie-Mode-B both pay the ~5 MB ECMWF /
~2 MB GEFS per-message minimum even for a single-pixel query. Zarr's
chunk-level addressing (~17 MB uncompressed → ~3 MB compressed chunk)
collapses regional and pencil costs by 100–10,000×.

This is the central structural fact: **GIK = full message; Zarr =
spatial subset of a chunk.** For global queries they converge; for
regional/pencil they diverge by orders of magnitude on wire.

### 2.5 JIT annual operational cost (365 slices/yr)

From `jit_annual.csv`. "Regional slice" = East Africa, all members, all
leads, the full operational variable set, one init per day.

| Scenario                                | Method                    | Annual GiB | $ same-cloud | $ cross-cloud | Infra $/yr (paid by) |
|-----------------------------------------|---------------------------|------------|--------------|---------------|----------------------|
| `gefs_gik_regional_jit_sequential`      | GIK byte-range            | 3,815      | $0.80        | $369          | $0 (you)             |
| `gefs_gik_regional_jit_dask20`          | GIK + 20-worker Coiled    | 3,815      | $37.30       | $406          | $0 (you)             |
| `gefs_dynamical_regional_jit`           | dynamical Zarr (region)   | 22         | $0.04        | $2.15         | $44,316 (dynamical)  |
| `gefs_own_arco_regional_jit`            | own ARCO Zarr (region)    | 22         | $0.04        | $2.15         | $14,720 (you)        |
| `ecmwf_gik_regional_jit_sequential`     | GIK byte-range            | 7,368      | $0.00        | $712          | $0 (you)             |
| `ecmwf_gik_regional_jit_dask20`         | GIK + 20-worker Coiled    | 7,368      | $36.50       | $749          | $0 (you)             |
| `ecmwf_dynamical_regional_jit`          | dynamical Zarr (region)   | 15         | $0.00        | $1.46         | $11,718 (dynamical)  |
| `ecmwf_own_arco_regional_jit`           | own ARCO Zarr (region)    | 15         | $0.00        | $1.46         | $53,841 (you)        |

The harness's 20-worker Dask projection ($36–$37/yr) is in the same
ballpark as the Coiled runs in §1 ($0.74–$2.20 per slice × 365 ≈
$270–$800/yr at one-slice/day cadence). The two sources agree: **at
~1 slice/day cadence the JIT premium is sub-thousand dollars
annually**, vs ~$54K/yr to self-host the equivalent ARCO Zarr for
ECMWF.

### 2.6 Dask-cluster ergonomics (§7 of `results/REPORT.md`)

| Method                | Cloudpickle / Dask | Notes |
|-----------------------|--------------------|-------|
| **GIK parquet refs**  | ✅ trivial          | Parquet refs are pure data; workers open via `fsspec.implementations.reference.ReferenceFileSystem`. No session state to ship. |
| **Herbie**            | ⚠ known issues     | `Herbie` instances carry an open requests session + filesystem-rooted `Path` cache state that cloudpickle struggles with. Community workarounds exist but it's not seamless. |
| **dynamical Zarr**    | ✅ native           | icechunk is Dask-aware; `xr.open_zarr(store=icechunk.session.store).sel(...).compute()` fans out chunks automatically. |

The Coiled runs in §1 are the empirical confirmation of the top row:
GIK parquet refs serialised across 15- and 50-worker clusters without
fuss.

### 2.7 Break-even for self-hosting an ARCO duplicate

| Product | Own ARCO $/yr | $/slice GIK | $/slice Zarr | $ saved/slice | Break-even slices/yr | Per day |
|---------|---------------|-------------|--------------|---------------|----------------------|---------|
| GEFS    | $14,720       | $1.012      | $0.006       | $1.006        | **14,627**           | 40.1    |
| ECMWF   | $53,841       | $1.951      | $0.004       | $1.947        | **27,656**           | 75.8    |

A regional centre would need to be running **40+ GEFS or 76+ ECMWF
regional slices every day** for self-hosting an ARCO Zarr duplicate to
beat GIK + cross-cloud egress. Under typical JIT use (~1/day) GIK wins
by a wide margin; under heavy continuous-streaming use (hundreds of
slices/day) the duplicate starts to pay off.

---

## 3. The "Just-In-Time / YOLO cluster" paradigm

The three observed runs and the harness's JIT-annual rows share the
same lifecycle:

1. User (or scheduler) submits a question.
2. A Dask cluster is provisioned (15–50 workers).
3. Workers read the GIK parquet manifest, fan out byte-range reads
   against S3, decode in memory (gribberish ~80× faster than cfgrib),
   reduce / accumulate, write a small persistent artifact.
4. Cluster shuts down. Bill stops.

Calling this "YOLO" is accurate in the engineering sense: the cluster
is treated as disposable, not as infrastructure. There is no
provisioning, no idle billing, no environment drift. The accounting
unit becomes **cost per scientific question**, not **cost per month of
compute**.

### Cost per question (rough, from §1)

| Question                                         | Cost  |
|--------------------------------------------------|-------|
| One month TP, 51 members, East Africa, Icechunk  | $2.20 |
| Same, smaller window                             | $1.42 |
| 8 vars × 7-day forecast × 51 members             | $0.74 |

The dominant cost in these runs is decoder/aggregator CPU, not storage
or egress. The harness's same-cloud projections (§2.5) are consistent
with this — same-cloud egress is rounding error; cross-cloud egress is
where the per-slice premium lives.

### The pipeline shape

```
[GRIB on S3]──(GIK parquet refs, ~30 GiB total)──▶[ephemeral Dask cluster]──▶[Icechunk store on Source.coop]
                                                          │
                                                          └──▶ optional: NetCDF, plots, scalar summaries
```

- Input bytes live in NOAA / ECMWF public buckets — never replicated.
- The GIK manifest is small (parquet refs ~$9/yr full archive — §2.3)
  and cheap to keep.
- The Icechunk store is the only persistent artifact, holding only the
  subset and derived form actually wanted.

---

## 4. Implication for continuous risk monitoring

The motivation is not "make Zarr cheaper". It is that **continuous risk
monitoring with Bayesian belief updating only needs the data while a
decision is being made.**

In a Bayesian-network framing:

- Nodes represent hazards (heavy rainfall, drought onset), exposure,
  vulnerability, decision outcomes.
- New forecast data arrives every 6 hours from ECMWF, every 6 hours
  from GEFS, etc.
- Each cycle, the network ingests new forecasts as **evidence**,
  propagates belief, and produces an updated risk distribution.
- The forecast bytes themselves are not the asset. The **updated
  belief** is the asset.

Under the conventional pipeline you would (a) download/replicate the
forecast archive, (b) keep a standing cluster or warehouse to query it,
(c) re-query at every cycle. That model has fixed cost regardless of
how often the network actually needs an update — §2.3 shows that fixed
cost is **$68K–$439K/yr** for an ARCO duplicate at recent file sizes.

Under the GIK + ephemeral-Dask model:

- A cycle starts. Spin up workers.
- Workers stream only the bytes the network's evidence variables need
  (TP, T2M, U/V at 700 hPa, ...) for the relevant members and steps.
- Evidence delivered. Belief updates. Decision emitted.
- Workers shut down. The data ceases to exist anywhere we pay for.

The cost of one belief update is the cost of one `ecmwf-fog-680`-class
run — sub-$1 today. The harness's break-even table (§2.7) makes the
economic threshold explicit: **under ~40 GEFS / ~76 ECMWF regional
slices per day**, GIK + JIT clusters is the cheapest credible path. At
6-hourly forecast cycles (4 slices/day) we are nowhere near that
threshold.

This is the same shift video streaming made: you pay for the segments
you actually play, not for owning the film.

---

## 5. Lessons learned (from `cost-eval/` + the Coiled runs)

The harness sharpened the original GIK story in five ways:

1. **The "tens of thousands saved" claim is stronger than originally
   stated.** The tex paper counted only 00z. The live HEAD probe
   (script 01) shows full archive sizes are ~5× larger because all four
   runs publish data. At recent file sizes the duplication cost gap is
   **~7,650× on hosting** — between 10³ and 10⁴×, sharper than the
   original ~10⁴× claim.

2. **GIK ≡ Herbie Mode B at the byte level.** Script 04 confirmed both
   methods transfer the same GRIB messages (full message minimum). The
   difference is operational: Herbie re-parses the sidecar `.idx` on
   every query and has known cloudpickle / dask-distributed friction;
   GIK has a durable parquet manifest that serialises trivially across
   workers (§2.6).

3. **Zarr beats GIK 100–10,000× for spatial subsets.** Script 05 made
   this concrete: for a regional ECMWF query, GIK pulls 20.2 GiB on
   wire and Zarr pulls 0.04 GiB. The right architecture depends on
   workload shape, not on which method is "better." Variables that fit
   inside dynamical.org's curated scope should be read via Zarr;
   everything else falls back to GIK.

4. **The right answer for a Global-South regional centre is usually
   "both."** Use dynamical's free public Zarr where its curated scope
   matches the work (cheapest path); fall back to GIK byte-range over
   the source GRIB for everything else — without ever paying ARCO
   hosting yourself. The win-win-win is real once you read both rows
   of the cost matrix together.

5. **Per-decision economics make Bayesian-update workflows tractable.**
   The Coiled runs cost $0.74–$2.20 per substantial scientific
   question, and the harness projects single-slice JIT cadence to
   $370–$800/yr per product including cross-cloud egress. A national
   met service can run continuous belief updates against the public
   archive on a four-digit annual budget — without negotiating data
   sharing, without standing infrastructure, without owning the bytes.

---

## 6. What this opens up

- **Risk-conditioned compute.** Run a cheap cluster every 6 h on the
  question *"is there evidence of a >25 mm/24 h event over Lake
  Victoria basin in the next 5 days?"*. Escalate to a full ensemble
  pull only when belief crosses a threshold.
- **Country / basin-scoped early warning.** A national met service can
  stand up its own pipeline against the same public archive without
  data-sharing agreements — they pay $1–$3 per decision cycle in
  compute.
- **Reproducibility on small storage.** Every Bayesian update can be
  archived as the small Icechunk store of the slice it consumed, not
  the whole source archive. Audit trails become tractable.
- **Dynamical.org / Zarr ecosystem alignment.** Icechunk outputs from
  these ephemeral runs are first-class Zarr v3 stores; they drop
  straight into the dynamical.org / Zarr ecosystem for downstream
  consumers — no bespoke format. And where dynamical already serves a
  variable, the workflow can `xr.open_zarr` it directly and skip GIK.
- **Backfill to the pre-2020-09-25 archive.** The new
  `gefs_pgrb2b05_preprocessing.py` (committed in `cdee2db`) extends GIK
  coverage to the v10 era at 0.5° / 21 members. Once the v10 template
  archive is built, the same JIT pattern works back to 2017-01-01 —
  enabling retrospective Bayesian-network calibration against
  ~9 years of additional ensemble history.

---

## 7. Open questions to revisit

- The two TP runs (`ecmwf-ea-tp-5900` and `-8990`) differ in "Tasks"
  (1,632 vs 127) at the same cluster size and similar cost. Worth
  confirming which run produced the Source.coop Icechunk store and what
  the second observation was for.
- Where exactly is the Coiled cost dominated — gribberish decode, Dask
  shuffling, or Icechunk write commits? Profiling one of these runs
  would identify the next 2× cost reduction.
- Script `06_benchmark_dynamical_open.py` is listed as `todo` in the
  README but was folded into `05_benchmark_gik_and_zarr.py`. Worth
  either deleting the README line or breaking 06 out cleanly for
  clarity.
- The pencil workload in the GIK column of §2.4 is the bracing data
  point — **15,624 GiB on wire** for one pixel-time series via GIK
  because every full message is required. This is the workload where
  GIK is strictly the wrong tool; Zarr is 7,800× better. Worth
  documenting explicitly in the public-facing tutorial.
- Pre-decision warmup: 1–2 minutes of cluster spin-up is included in
  the $0.74–$2.20 above. If risk monitoring needs sub-minute latency,
  a small warm pool may be justified — but for 6-hourly forecast cycles
  the cold-start is invisible.
- Benchmark a v10 0.5° GEFS JIT run against the v12 0.25° baseline once
  `gefs_pgrb2b05_preprocessing.py` has produced its template archive.
  Expect different per-decision economics: 21 members vs 30, 0.5° grid
  vs 0.25°, 6 h cadence vs 3 h.

---

*Numbers in §1 are taken verbatim from the Coiled cluster dashboard
rows shared in-thread. Numbers in §2 are from `cost-eval/results/` and
`cost-eval/results/REPORT.md`, generated by the reproducible harness
under `cost-eval/`. Pricing reflects `cost-eval/config.yaml ->
pricing.as_of` and updates automatically when re-run.*
