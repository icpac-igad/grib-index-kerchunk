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
