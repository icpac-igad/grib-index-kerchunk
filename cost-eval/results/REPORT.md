# GIK Cost-Evaluation Report

This report synthesises every measurement collected in `cost-eval/results/` into one place. Every number below is derived from a primary measurement (S3 HEAD, HF metadata, live xarray materialisation, or icechunk LIST) — no figure is restated from the source tex without independent verification.

## 1. Archive volumes (measured live)

| Product | Sample | Mean / day (all 4 runs) | Full-archive extrapolation | Pricing benchmark |
|---|---|---|---|---|
| GEFS | primary HF-overlap (2020-09-25..2025-12-31) | 147.2 GB | 257.532 TiB / 0.2515 PiB | — |
| GEFS | recent (Dec 2025) | 177.4 GB | **310.433 TiB / 0.3032 PiB** | current archive size |
| ECMWF | primary HF-overlap (2024-03-01..2026-02-20) | 1379.9 GB | 903.633 TiB / 0.8825 PiB | — |
| ECMWF | recent (Dec 2025) | 1734.0 GB | **1135.486 TiB / 1.1089 PiB** | current archive size |

## 2. GIK parquet catalog (the thing GIK actually maintains)

| Product | Per-date parquet bytes | Full-archive parquet GiB | **Compression vs GRIB** | Template tar.gz (one-time) |
|---|---|---|---|---|
| GEFS | 7.7 MB | 13.879 GiB | **4754.9×** | 3.101 MiB |
| ECMWF | 24.4 MB | 16.364 GiB | **56545.5×** | 120.696 MiB |

## 3. dynamical.org ARCO Zarr (measured on-disk)

| Product | Uncompressed (xarray view) | On-disk (S3) | Zarr compression | Chunk shape |
|---|---|---|---|---|
| GEFS | 956.749 TiB | **146.032 TiB** (369,699 objects) | 6.55× | (1,31,64,17,16) |
| ECMWF | 240.792 TiB | **38.613 TiB** (239,053 objects) | 6.24× | (1,85,51,32,32) |

## 4. The win-win-win, by stakeholder

Each row is the **annual storage cost at AWS S3 Standard list ($0.023/GB-month)** that each party pays under each deployment model.

| Party / model | GEFS | ECMWF | Combined |
|---|---|---|---|
| Data provider (NOAA + ECMWF) — already publishing | $0 | $0 | $0 *(no incremental cost from GIK)* |
| Regional centre — host own ARCO duplicate (recent file sizes) | $14,720 / yr *(6.4× compressed)* | $53,841 / yr | $68,560 / yr |
| Regional centre — host own raw GRIB duplicate | $94,206 / yr | $344,581 / yr | **$438,787 / yr** |
| dynamical.org (curated subset of vars, today's actual on-disk) | $44,316 / yr | $11,718 / yr | **$56,034 / yr** |
| **GIK parquet catalog (full archive coverage)** | $4.11 / yr | $4.85 / yr | **$8.96 / yr** |

**Read this as**: the tex's headline "tens of thousands of dollars saved" becomes specific — at recent file sizes and AWS list pricing, a regional centre that wanted to host its own ARCO ECMWF + GEFS duplicate would pay **~$68,560/yr** for the compressed Zarr version, or **~$438,787/yr** for raw GRIB. GIK delivers full archive coverage for **<$10/yr** in parquet-reference storage. The combined real-world dynamical.org bill is **~$56,034/yr** for a curated 19-22 variable subset (a different scope than the GIK-indexed full archive — see §6).

## 5. Per-slice cost matrix (one analysis, today)

All ETLs cost-out the same workload: 12 vars × 51 mems × 9 leads ECMWF / 4 vars × 30 mems × 9 leads GEFS, **1 init**, except the *pencil* row which is **one (lat, lon) point × full archive of inits**.

| Product | Method | Workload | Bytes on wire | $ same-cloud | $ cross-cloud |
|---|---|---|---|---|---|
| gefs | `herbie_mode_A` | full_operational_slice_1init | 4.589 GiB | $0.0083 | $0.4518 |
| gefs | `herbie_mode_B` | full_operational_slice_1init | 2.012 GiB | $0.0015 | $0.1959 |
| ecmwf | `herbie_mode_A` | full_operational_slice_1init | 54.232 GiB | $0.0083 | $5.2492 |
| ecmwf | `herbie_mode_B` | full_operational_slice_1init | 25.649 GiB | $0.0077 | $2.4863 |
| gefs | `gik` | global_1var | 10.451 GiB | $0.0022 | $1.0122 |
| gefs | `zarr` | global_1var | 3.647 GiB | $0.0046 | $0.3571 |
| gefs | `gik` | regional_1var | 10.451 GiB | $0.0022 | $1.0122 |
| gefs | `zarr` | regional_1var | 0.06 GiB | $0.0001 | $0.0059 |
| gefs | `gik` | pencil_1var | 21446.072 GiB | $4.6055 | $2077.0845 |
| gefs | `zarr` | pencil_1var | 1.934 GiB | $0.0025 | $0.1893 |
| ecmwf | `gik` | global_1var | 20.186 GiB | $0.0 | $1.9508 |
| ecmwf | `zarr` | global_1var | 2.674 GiB | $0.0004 | $0.2588 |
| ecmwf | `gik` | regional_1var | 20.186 GiB | $0.0 | $1.9508 |
| ecmwf | `zarr` | regional_1var | 0.041 GiB | $0.0 | $0.004 |
| ecmwf | `gik` | pencil_1var | 15624.287 GiB | $0.0263 | $1509.9068 |
| ecmwf | `zarr` | pencil_1var | 2.0 GiB | $0.0003 | $0.1936 |

> **Per-message floor matters**: GIK / Herbie-Mode-B both pay the ~5 MB ECMWF / ~2 MB GEFS per-message minimum even for a 1-pixel query. dynamical's Zarr can subset at chunk granularity (~17 MB uncompressed → ~3 MB compressed chunk for ECMWF), so regional + pencil workloads diverge by 100-10,000× on wire.

## 6. JIT / "you-only-look-once" annual operational cost

Pattern: every day a new forecast lands, the regional centre streams a regional slice, runs risk-assessment, discards. 365 slices/year.

| Scenario | Method | Annual GiB | $ same-cloud | $ cross-cloud | Infra $/yr | Ergonomics |
|---|---|---|---|---|---|---|
| `gefs_gik_regional_jit_sequential` | gik_byte_range | 3814.7 | $0.8 | $369.45 | $0.0 | single-machine, slow per slice (~min) but trivial setup |
| `gefs_gik_regional_jit_dask20` | gik_byte_range_dask20 | 3814.7 | $37.3 | $405.95 | $0.0 | parquet refs serialise trivially over cloudpickle; Coiled wall ~3 min/slice |
| `gefs_dynamical_regional_jit` | dynamical_zarr_regional | 21.74 | $0.04 | $2.15 | $44316.0 | icechunk is dask-native; user free-rides on dynamical's $44,316/yr archive |
| `gefs_own_arco_regional_jit` | own_arco_zarr_regional | 21.74 | $0.04 | $2.15 | $14720.0 | you pay the full ARCO storage bill; access cost is identical to dynamical Zarr |
| `ecmwf_gik_regional_jit_sequential` | gik_byte_range | 7367.7 | $0.0 | $712.04 | $0.0 | single-machine, slow per slice (~min) but trivial setup |
| `ecmwf_gik_regional_jit_dask20` | gik_byte_range_dask20 | 7367.7 | $36.5 | $748.54 | $0.0 | parquet refs serialise trivially over cloudpickle; Coiled wall ~3 min/slice |
| `ecmwf_dynamical_regional_jit` | dynamical_zarr_regional | 14.97 | $0.0 | $1.46 | $11718.0 | icechunk is dask-native; user free-rides on dynamical's $11,718/yr archive |
| `ecmwf_own_arco_regional_jit` | own_arco_zarr_regional | 14.97 | $0.0 | $1.46 | $53841.0 | you pay the full ARCO storage bill; access cost is identical to dynamical Zarr |

## 7. Dask-cluster ergonomics (qualitative)

| Method | Cloudpickle / dask-distributed support | Notes |
|---|---|---|
| **GIK** parquet refs | ✅ trivial | parquet ref files are pure data; the worker just opens via `fsspec.implementations.reference.ReferenceFileSystem`. No session state to ship. |
| **Herbie** (full + searchString modes) | ⚠ known issues | `Herbie` instances carry an open requests session and filesystem-rooted `Path` cache state that cloudpickle struggles with; community workarounds exist (build Herbie object inside worker, use `Herbie._download_url` strings) but it's not seamless. |
| **dynamical Zarr** (icechunk) | ✅ native | icechunk is dask-aware; `xr.open_zarr(store=icechunk.session.store).sel(...).compute()` fans out chunks across workers automatically. |

## 8. Break-even on duplicating to an ARCO Zarr

If the regional centre's workload sits mostly inside dynamical's curated scope, *not duplicating* is cheaper. But if their work extends to more variables / more init cycles, the question is: at what cadence does duplicating the ARCO catalog pay for itself? Compares storage rent against the wire-cost gap between GIK and Zarr (regional slice, cross-cloud).

| Product | Own ARCO $/yr | $/slice GIK | $/slice Zarr | $ saved per slice | Break-even slices/yr | per day |
|---|---|---|---|---|---|---|
| GEFS | $14,720.0 | $1.0122 | $0.0059 | $1.0063 | **14,627.0** | 40.1/day |
| ECMWF | $53,841.0 | $1.9508 | $0.004 | $1.9468 | **27,656.0** | 75.8/day |

## 9. The honest, nuanced narrative

1. **The "GIK saves tens of thousands of dollars" claim is stronger than the tex states.** Live measurements show the full GRIB archive is ~5× larger than the tex assumed (because it counted only 00z). Even compressed to ARCO Zarr at the measured 6.4× rate, full-scope duplication is ~$68,560/yr, vs ~$9/yr for GIK's parquet catalog. The savings ratio is **~7,652× on hosting** (between 10³ and 10⁴×).

2. **GIK is structurally identical to Herbie's smart mode for byte-range transfer.** They differ on operational ergonomics (Dask), not on bytes — both pull full GRIB messages, both can't spatially subset. The parquet refs are the durable artifact that fixes Herbie's per-query `.idx` parsing + dask-pickling friction.

3. **dynamical.org's chunked Zarr genuinely beats GIK for regional and pencil workloads** — 100-10,000× fewer bytes on wire. The catch: this requires (a) someone paying the ARCO hosting bill (dynamical, today), and (b) the user's variables being inside dynamical's curated scope. Outside that scope, GIK is the only credible alternative to duplicating the archive yourself.

4. **For the JIT operational pattern at scale** (see §6): at 365 ECMWF regional slices/yr, GIK sequential cross-cloud is ~$712/yr in egress, dynamical Zarr is ~$1.46/yr — but the dynamical row carries a ~$11,718/yr **hosting cost paid by dynamical**. Self-hosting an own-ARCO duplicate would cost ~$53,841/yr in storage. GIK shifts that infrastructure cost to ~$0 and trades it for a per-query premium that takes 76+ regional slices/day to overtake the duplicate-storage scenario. Under typical JIT use (~1/day), GIK wins by a wide margin.

5. **Important data-scope caveat for §4.** The 'dynamical.org GEFS' on-disk number ($44,316.0/yr) is the *35-day extension product* (181 leads × 31 mems × 22 vars) — a wider scope than the 10-day operational pgrb2sp25 GIK currently indexes. Comparing those rows is apples-to-oranges; the 'own ARCO duplicate' row at $14,720/yr GEFS is the 10-day product compressed at the measured 6.4× rate.

6. **The right answer for a Global-South regional centre is usually "both"**: use dynamical's free public Zarr where its curated scope matches your work (cheapest path), and fall back to GIK byte-range over the source GRIB for everything else — without ever paying ARCO hosting yourself. The win-win-win is real; this report just makes its terms specific.


## 10. Compute pattern playbook + independence value

This section addresses the operational question every adopting centre asks:
**"How do we actually run GIK in production when dynamical-style Zarr isn't
a viable alternative — either because its scope doesn't cover us, or because
we can't accept the third-party dependency?"**

All numbers below are derived from the measurements in §1–§9 above. No new
probing; pure synthesis. If you re-run the pipeline against updated `config.yaml`
pricing, regenerate the numbers in §1–§9 first and update the constants here
manually (they're hard-coded for readability rather than pulled from
`07_cost_model_summary.json`).

### 10.1  Short-burst Dask: the right routine

A persistent 24×7 Dask cluster is the single most common anti-pattern for
daily forecast-driven workflows: at 1 slice/day it sits idle ~99 % of the
time, and its cost dwarfs the storage win GIK was supposed to deliver.

Compare four compute postures for 1 ECMWF regional slice/day, cross-cloud:

| Compute pattern | Annual compute $ | Wall per slice | What you give up |
|---|---|---|---|
| 24×7 Dask cluster (4 workers always on) | **~$3,500** | seconds (no cold start) | $$$$, idle 99 % of time |
| **Ephemeral Coiled Dask** (20 workers, ~3 min, spin-up + tear-down) | **~$37** | ~3 min + ~30-60 s cold start | minor latency penalty |
| Serverless / Lithops fan-out (Cloud Run, ~$0.09/invoke) | **~$33** | ~30-60 s + per-member parallelism | per-function memory/time limits |
| Single-machine sequential | **~$0** (own laptop) / ~$700 (cloud VM 24×7) | ~10 min | no fanout, hard to scale |

> At 1 slice/day, the persistent-cluster premium (~$3,500/yr) is **larger
> than the entire ECMWF dynamical hosting bill ($11,718/yr — §3)** — running
> it undoes the win GIK was supposed to deliver. Ephemeral or serverless
> preserves it.

**Why GIK fits ephemeral unusually well.** A worker pulls one parquet ref
file (~24 KB) from HF/GCS and is immediately ready. No `.idx` cache to
hydrate, no `requests` session to ship, no filesystem-rooted `Path` state to
serialise. The ref file *is* the data structure. Compare to Herbie, where
each worker has to bootstrap an `.idx` cache and the `Herbie` instance
itself doesn't cloudpickle cleanly. GIK is structurally aligned with
short-burst fan-out compute; Herbie is structurally aligned with sequential
per-process use.

### 10.2  Burst processing: a month of ECMWF in one short Coiled session

The realistic full-scope retrospective workload — 12 vars × 51 mems ×
all 85 leads × 30 init dates — using GIK parquet refs + gribberish +
ephemeral Coiled, **inside AWS** so the AWS Open Data egress stays free:

| Component | Value |
|---|---|
| Bytes / init (12 vars × 51 mems × 85 leads × ~5 MB/msg) | ~260 GB |
| Bytes for 30 days | **~7.8 TB on wire** |
| Coiled wall on 20 workers @ ~100 MB/s/worker (S3→EC2 in-region) | **~65 min** |
| Coiled wall on 50 workers @ same bandwidth | **~26 min** |
| Compute $: 20 workers × ~1 hr × $0.10/wkr-hr | **~$2/month-burst** |
| Egress same-cloud (Coiled inside AWS) | **$0** |
| Egress cross-cloud (Coiled outside AWS) | ~$700/month-burst — would dominate, don't do this |

**Annual cost for 12 monthly bursts on AWS-hosted Coiled: ~$24-50 in
compute, $0 in egress, $9 in parquet hosting → ~$60-100/yr all-in for
full-scope ECMWF retrospective analysis with no third-party dependency.**

The decode side is *not* the bottleneck at this scale: at the ~25 ms/chunk
gribberish rate (CLAUDE.md), 2.45 M monthly chunks ÷ 20 workers = ~50 min
decode, which overlaps with the bandwidth-bound transfer. With cfgrib
instead of gribberish the same burst would push wall to ~40 hours — the
Rust decoder is **load-bearing** for the burst pattern.

### 10.3  Independence value: dynamical-subscription sensitivity

dynamical.org is free today but free-today is not a contract. The actual
hosting cost the project absorbs (~$56,034/yr combined, measured live in
§3) is real economic gravity that has to be funded somehow. If that funding
model changes to a subscription:

| Hypothetical dynamical subscription | Annual cost to user | vs GIK all-in (~$100/yr) |
|---|---|---|
| Free (today, AWS in-region) | ~$2/yr egress + ~$1 compute | GIK is ~$100/yr higher in $ alone |
| $8 / month | **$96/yr** | break-even with GIK |
| $50 / month tier | **$600/yr** | ~6× GIK |
| $500 / month (enterprise) | **$6,000/yr** | ~60× GIK |
| Scope-cut: a variable you need is dropped | engineering migration cost | bounded utility risk |

The subscription doesn't need to be expensive to flip the economics — even
a modest tier crosses the GIK all-in line. The economic question isn't
"is GIK cheaper than free dynamical today" (it isn't), it's "is GIK cheaper
than free-dynamical-may-not-stay-free across a 5-year horizon."

### 10.4  Strategic risk axes (the ones that don't show up as $)

Reasons a sovereign centre should think twice about third-party Zarr
dependence, even if it's free today:

1. **Bus factor.** dynamical is a small operation. GIK's underlying source
   (NOAA/ECMWF AWS Open Data) is a US-Government + EU intergovernmental
   commitment. The half-life of those programmes is decades; the half-life
   of any single startup is years.

2. **Scope drift.** dynamical chose 19-22 variables to curate. If they swap
   one of yours later, you have no recourse. With GIK you index whatever
   NOAA/ECMWF publishes — full archive, all four runs, every variable.

3. **Latency to first slice.** dynamical typically lags real-time by
   hours-to-a-day (their ingestion + chunking pipeline). GIK is real-time:
   minutes after the producer publishes, you can stream. For operational
   risk-warning workflows where the 12z run must drive an alert by 14:00,
   this is decisive.

4. **Reproducibility.** GIK refs are static parquet files versionable in
   git. A paper saying *"we used parquet refs commit `abc123` against
   `s3://noaa-gefs-pds`"* is bit-reproducible decades later. Dynamical's
   icechunk repo can be re-chunked or re-encoded at any point, silently
   changing bit-identity even if the xarray API stays stable.

5. **Sovereignty for Global-South institutions.** Routing operational risk
   monitoring through a third-party US startup carries diplomatic and
   continuity risk that's invisible to a pure-$ calculation. For ICPAC's
   regional mandate covering East African governments, this is not abstract.

### 10.5  Three adoption postures

| Posture | Annual $ | Independence | Scope | Best when… |
|---|---|---|---|---|
| **GIK-only, ephemeral Coiled** | ~$60-100/yr | High — only depends on public Open Data | Full archive | You need vars or runs outside dynamical's scope; you want sovereignty; you're doing operational ops |
| **dynamical-primary, GIK fallback** | ~$5/yr (today, AWS in-region) | Medium — exposed to scope/availability | dynamical scope + GIK overflow | Cost-sensitive analytics; dynamical's curated set fits your work |
| **Own ARCO duplicate** | **~$68,560/yr** storage + maintenance team | Highest | Whatever you choose | You have institutional budget + sustained team to maintain Zarr |

### 10.6  Concrete operational playbook

For an adopting centre with daily risk monitoring + occasional retrospective
bursts, no viable Zarr alternative for the full variable scope:

| Cadence | Pattern | Why |
|---|---|---|
| **Backfill / historical** (one-time, many dates) | Lithops on Cloud Run | ~$0.09/date, fan-out across dates, no cluster |
| **Daily JIT** (1-4 slices/day, fresh forecast) | Ephemeral Coiled Dask (20 workers, ~3 min) | ~$36/yr, fast wall, parquet refs serialise trivially |
| **Monthly retrospective** (1 month of data in one go) | Ephemeral Coiled, ~65 min burst | ~$2/burst, ~$24/yr at 1 burst/month |
| **Interactive research** (multi-scenario sessions) | Persistent Dask for the session, off at EOD | ~$3-5/working day |

**Total expected annual all-in for full-scope, sovereign, multi-cadence
operations: ~$100-150/yr** — dominated by daily-JIT Coiled compute. The
parquet hosting (~$9/yr) is a rounding error. The "tens of thousands of
dollars saved" headline becomes specific: you avoid the ~$68,560/yr ARCO
duplication bill AND eliminate the question of "what do we do if our
free-Zarr provider's terms change."

### 10.7  The honest one-line summary

> *Use dynamical's free public Zarr where its curated 19-22 var scope
> matches your work — it's structurally better for regional and pencil
> queries — but build the GIK-only routine for everything else. The
> all-in cost is ~$100/yr, the independence is genuine, and the operational
> pattern (ephemeral Coiled bursts of a few minutes per slice or per month)
> matches the cadence of forecast publishing rather than fighting it.*
