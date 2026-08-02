# EWC usage to date, and the resource plan for the East Africa cGAN corpus

**What we have actually run on the EWC cluster and object store so far, what
the production workload looks like, and how the ~7.6-day / ~4 TB figures for
the full ECMWF ensemble extraction were derived.**

_Written 2026-07-31. Measured on the live EWC cluster — **6 worker VMs, each
4 vCPU and 16.77 GB RAM** (Dask `memory_limit` 13.94 GB after its own
headroom), 24 vCPU and ~84 GB usable in total — against the published ECMWF
IFS ensemble Icechunk store. Every figure is labelled **measured** or
**modelled**; nothing here is a vendor estimate._

Companion material in this directory:
- `materialize_ea_icechunk_ewc.py` — the extraction tool (`plan`, `corpus`,
  `probe`, `run`, `size`)
- `EA_MATERIALIZATION_PLAN.md` — the engineering detail behind every number

---

## 1. Answer up front

| Question | Answer |
|---|---|
| How much have we processed and stored so far? | **Almost nothing — 3.38 GB stored, all synthetic benchmark data, and ~6 GB read from the real ECMWF source during calibration.** No production data has been materialised. §2 |
| Why so little? | We were establishing feasibility first, and the link to the data has been unreliable throughout. |
| What is the production workload? | Read the published virtual ECMWF store (chunks live on AWS), crop to East Africa, keep all 51 ensemble members, write a realized Icechunk store on EWC. §3 |
| How much will it store? | **~4.4 TB compressed** (7.37 TB uncompressed) for 1,246 forecast dates. §5 |
| How long will it take? | **3.5 to 60 days** — set entirely by the AWS→EWC link, which is the open question. §5.4 |
| What does it need per worker? | **4 vCPU and 16 GB is not known to be insufficient.** An earlier draft claimed 128 GB was needed for most of the archive; that rested on a manifest model since disproven. §6 |
| What is the unusual cost? | **~61 TB of cross-cloud ingress** from AWS `eu-central-1` into EWC, to write 4.4 TB — a 14× read amplification, and unavoidable. §5.3 |
| Biggest risk to the schedule | **The AWS→EWC link.** Measured single-stream throughput varies **0.74 – 14.5 MB/s** and was timing out entirely during testing. That range alone spans **3 days to 5 months**. §5.4 |

---

## 2. Real usage to date — the honest picture

### 2.1 Stored in `s3://must-icechunk`

Measured 2026-07-31 by listing the bucket:

| prefix | GB | objects | what it is |
|---|---:|---:|---|
| `long-test` | 3.361 | 1,283 | `test-icechunk-long.py` — synthetic 0.25° fields, plumbing/throughput test |
| `realized-test` | 0.022 | 40 | `test-icechunk-write.py` — synthetic write/commit correctness test |
| `_probe` | 0.000 | 6 | endpoint capability probe |
| **total** | **3.383** | **1,329** | |

**All of it is synthetic.** Not one byte of real ECMWF forecast data has been
materialised into the store. The `long-test` figure is `da.random.normal` and
`da.random.gamma` arrays shaped like the IFS grid, written specifically to
prove that Icechunk commits work on the Ceph RadosGW backend.

### 2.2 Read from the real ECMWF archive

All reads so far were calibration, not production. Cumulative across every
benchmark in this investigation:

| activity | GRIB messages decoded | ≈ bytes read |
|---|---:|---:|
| availability probes (all 30 channels × 2 runs) | ~55 | ~0.08 GB |
| manifest RAM measurements (4 arrays) | ~5 | ~0.01 GB |
| throughput benchmarks (`t2m`, `u`) | ~3,700 | ~5.5 GB |
| **total** | **~3,800** | **~5.6 GB** |

For scale, the production corpus is **91.5 M messages / ~61 TB** — so we have
so far read about **0.01 %** of what the full job requires.

### 2.3 What that testing established

Small as the volume is, it settled the things that decide whether the job is
viable at all:

| finding | status |
|---|---|
| Icechunk commits work on Ceph RGW (conditional writes / `If-None-Match`) | ✅ **measured** — 8/8 checks pass, incl. stale-session rejection |
| Anonymous read of the source store from EWC workers | ✅ **measured** — after fixing an `AWS_*` environment clash |
| All 30 required channels present and finite | ✅ **measured** — finite = 1.00 for every channel in the newest era |
| Sustained read throughput off AWS | ✅ **measured** — 21.6 messages/s per task |
| Worker memory required per source array | ⚠️ **earlier figure withdrawn** — the store is split one manifest per date, largest 1.1 MB — §6 |
| Whether the full archive is reachable on the current cluster | ⚠️ **no memory obstacle known**; never demonstrated end to end because the link kept failing — §6 |

---

## 3. How the Icechunk store is used with the Dask cluster

### 3.1 The two stores

```
  SOURCE (read, anonymous)                    SINK (write, EWC credentials)
  ────────────────────────                    ────────────────────────────
  source.coop / e4drr-project                 EWC Ceph RadosGW
  ecmwf_ifs_ens_aws_s3_icechunk_vd            s3://must-icechunk/<prefix>
                                     
  VIRTUAL Icechunk store:                     REALIZED Icechunk store:
  holds only chunk *references*               holds real compressed chunks
  pointing into the public GRIB               (time, step, number, lat, lon)
  archive s3://ecmwf-forecasts                float32, zstd
  on AWS eu-central-1                         ~15 GB of metadata -> 4.4 TB
```

The source store is ~15 GB of metadata standing in for ~620 TB of GRIB. It is
read-only and public. **Nothing we do writes to it.**

### 3.2 Dataflow per forecast date

```
   ┌────────── EWC Dask cluster ──────────┐
   │                                       │
   │  worker 1 ── holds `u` manifest ──┐   │      AWS eu-central-1
   │  worker 2 ── holds `v` manifest ──┤   │   s3://ecmwf-forecasts
   │  worker 3 ── holds `w` manifest ──┼───┼──► range-GET one whole
   │  worker 4 ── holds `r`,`t` ───────┤   │    GLOBAL GRIB message
   │  worker 5 ── holds `gh` + surface ┤   │    per member/step/level
   │  worker 6 ── holds surface vars ──┘   │
   │         │                             │
   │         │ decode (gribberish) → crop to East Africa → float32
   │         ▼                             │
   │      client ── writes region ─────────┼──► s3://must-icechunk
   │                 one commit per date   │    (same DC, no egress)
   └───────────────────────────────────────┘
```

Three properties of this shape matter for resourcing:

1. **One store variable is pinned to one worker for the whole run.** It keeps
   the opened era cached and each read task large — the shape the throughput
   was measured on. (Originally a memory workaround; §6 explains why that
   reasoning was withdrawn.)

2. **Cropping to East Africa does not reduce what we read.** A virtual chunk
   is one whole *global* GRIB message — 0.788 MB on average, measured from
   ECMWF's `.index` sidecars. We download a 721×1440 field and keep 163×147 of
   it, 2.3 % of the bytes. Inherent to the archive's layout, not a shortcoming
   of the tool.

3. **The write is local.** The sink is in the same data centre as the
   cluster, so the 4.4 TB written costs no egress. All the network cost is on
   the read side, inbound from AWS.

### 3.3 What lands in the sink

```
s3://must-icechunk/<prefix>/                icechunk repo, branch `main`
  <channel>   (time, step, number, latitude, longitude)   float32 + zstd
```

30 channels — 10 surface (`tp pw sp msl t2m skt ssr ttr tcw mucape`) and 20
pressure-level (`u`,`v` at 925/850/700/500/200; `w` at 925/850/700/500; `r` at
850/700; `t` at 850/700/500; `gh500`) — over a 163 × 147 grid at 0.25°
covering 25.25 … −15.25 °N, 18.5 … 55 °E.

One commit per forecast date, so the store is queryable and consistent while
the job is still running, and a failed run resumes from the last commit rather
than restarting.

---

## 4. How the ~7.6-day figure was derived

Two different "7 days" are involved and they are easy to confuse:

- **7-day forecast lead** — what we extract *per date*: every published step
  from 0 to 168 h. The store's step axis is 3-hourly to 144 h then 6-hourly,
  so this is **53 steps** (measured off the store's own axis, not assumed).
- **~7.6 days wall clock** — how long the *job runs*, **if** the link sustains
  ~92 MB/s aggregate. It has not been shown to. See §5.4 for the real band.

### 4.1 The arithmetic chain

**Step 1 — how many GRIB messages.** One message = one global field for one
(member, step, level). So per forecast date:

```
30 channels × 51 members × 53 steps = 81,090 messages/date
```

Across the archive, with per-era channel counts (§5.1):

```
0p4        401 dates × 22 ch × 51 × 53 = 23.85 M
49r1-pre   320 dates × 29 ch × 51 × 53 = 25.08 M
49r1-post  474 dates × 30 ch × 51 × 53 = 38.44 M
50r1        51 dates × 30 ch × 51 × 53 =  4.14 M
                                  total = 91.50 M messages
```

**Step 2 — how fast one task reads. MEASURED.** Three shapes, warm manifest,
on this cluster:

| shape | messages | seconds | rate |
|---|---:|---:|---:|
| `t2m`, 11 members × 11 steps | 121 | 12.4 | 9.8 msg/s |
| `t2m`, 51 members × 3 steps | 153 | 13.2 | 11.6 msg/s |
| **`u`, 5 levels, 51 members × 12 steps** | **3,060** | **176** | **21.6 msg/s** |

Larger reads are roughly **2× more efficient** — per-request overhead
amortises. The production task shape is the largest one, so 21.6 msg/s is the
rate to plan with, and it is measured at production shape rather than
extrapolated from a toy.

**Step 3 — the critical path.** Because variables are pinned one per worker,
wall clock is *not* `total messages ÷ total threads`. The per-date critical
path is the single heaviest task — `u`, which carries 5 pressure levels:

```
5 levels × 51 members × 53 steps = 13,515 messages
13,515 ÷ 21.6 msg/s            = 626 s ≈ 10.4 min per date
```

**Step 4 — multiply out**, with 0p4 credited 2× for its smaller 0.4° fields
(451×900 against 721×1440):

| group | dates | h |
|---|---:|---:|
| 0p4 | 401 | 34.8 |
| 49r1-pre | 320 | 55.6 |
| 49r1-post | 474 | 82.4 |
| 50r1 | 51 | 8.9 |
| **total** | **1,246** | **181.7 h = 7.6 days** *(⇔ ~92 MB/s aggregate)* |

### 4.2 What could make this wrong

Stated plainly, because a 7.6-day estimate off a 176-second measurement
deserves scepticism:

| risk | direction | note |
|---|---|---|
| **Network ceiling** | ⬆⬆ dominates everything | 60.6 TB at 75 MB/s is 9.4 days; at the worst observed 11.8 MB/s it is 59 days. Measured per-stream throughput was 0.74–14.5 MB/s and later failed outright. **This is not a risk to the estimate, it *is* the estimate.** §5.4 |
| Cold manifest loads | ⬆ slightly | 11–215 s per (worker, variable), paid once per run. Negligible over 1,246 dates, dominant over 1. |
| source.coop 5xx responses | ⬆ slightly | sporadic; the tool retries with backoff |
| Larger workers allow more concurrency | ⬇ faster | the critical path is one task; more RAM per worker would let us split it |
| Rate measured on a quiet cluster | ⬆ | contention with other tenants not accounted for |

**We would not commit to 7.6 days on this basis alone — and §5.4 shows why
not.** The task-rate model above says 7.6 days, but it presumes the per-task
rate holds under 16-way concurrency. Measured single-stream throughput since
ranged 0.74–14.5 MB/s and then failed outright, putting the honest band at
**3.5 to 60 days**. The plan (§7) starts with a one-date run that converts the
estimate into a measurement before any large allocation is requested.

---

## 5. Storage, compute and network

### 5.1 Storage — where 4 TB comes from

Per date per channel: `53 steps × 51 members × cells × 4 bytes`. The East
Africa window is a different number of cells per era because 0p4 is a 0.4°
grid — read from the store's coordinates, not assumed:

| group | dates | channels | EA grid | cells | GB/date | **TB** |
|---|---:|---:|---|---:|---:|---:|
| `0p4` | 401 | 22 | 102 × 91 @ 0.4° | 9,282 | 2.21 | 0.885 |
| `49r1`-pre | 320 | 29 | 163 × 147 @ 0.25° | 23,961 | 7.51 | 2.404 |
| `49r1`-post | 474 | 30 | 163 × 147 @ 0.25° | 23,961 | 7.77 | 3.684 |
| `50r1` | 51 | 30 | 163 × 147 @ 0.25° | 23,961 | 7.77 | 0.396 |
| **total** | **1,246** | | | | | **7.37 TB** |

- **7.37 TB uncompressed → ~4.4 TB after zstd** (the 40 % assumption is the
  one number here we have *not* measured; the first real run will settle it).
- Channel counts differ by era because the archive does: `0p4` publishes no
  `w`, `ssr`, `ttr` or CAPE family at all, and the `49r1` group spans a schema
  change at 2025-01-14 where `cape` gives way to `mucape` + `tcw`.

**If we stored the ensemble mean + standard deviation instead of all 51
members, this would be 289 GB — 25.5× smaller.** That is the form the
downstream model actually consumes. Keeping individual members is a deliberate
choice to preserve optionality for future work, and it is *purely* a storage
cost: see §5.3.

### 5.2 Compute — vCPU and RAM per worker

**Neither vCPU nor RAM is the constraint — the network is.** The workload is
network-bound; 4 vCPU per worker is ample and the current cluster's 24 threads
are not saturated.

An earlier draft of this section claimed RAM was the binding constraint, with
a table requiring 128 GB per worker for the `49r1` era. **That is withdrawn**
— it rested on a manifest model since disproven (§6). The current position:

| | workers | vCPU/worker | RAM/worker | total | vCPU-hours |
|---|---:|---:|---:|---|---:|
| **Current cluster** | 6 | 4 | 16.77 GB | 24 vCPU / ~84 GB usable | ~1,300–21,000 |

The vCPU-hour range is wide only because the *duration* is wide (§5.4), not
because the shape is uncertain. No requirement for larger workers has been
demonstrated.

### 5.2.1 The cluster the estimate was actually measured on

| | value |
|---|---|
| workers | **6** (separate VMs) |
| vCPU per worker | **4** (24 total) |
| RAM per worker | **16.77 GB** physical / 15.61 GiB |
| Dask `memory_limit` per worker | **13.94 GB** — the operative ceiling |
| usable manifest budget per worker | **~9.8 GB** (70 % of `memory_limit`, where Dask begins spilling) |

The budget is taken against Dask's `memory_limit`, not the physical 16.77 GB,
because that is what Dask enforces: it spills at 70 %, pauses at 80 % and
kills the worker at 95 % of `memory_limit` regardless of physical headroom.

**Six workers is a floor, not a tuning choice.** There are six pressure-level
store variables (`u`, `v`, `w`, `r`, `t`, `gh`) and each one's manifest must
sit on its own worker (§6). With five workers, two of those manifests share a
worker and it dies.

**More than six workers does not make it faster.** The per-date critical path
is a *single task* — `u`, 13,515 messages, 626 s — and adding workers does not
split it. Throughput scales with RAM per worker (which would let one
variable's levels run across several threads) or with the upstream manifest
fix, not with worker count.

This cluster is sufficient for the newest era only — 51 of 1,246 dates.

### 5.3 Network — the part worth flagging

| | volume |
|---|---:|
| **Inbound, AWS `eu-central-1` → EWC** | **~61 TB** |
| Outbound from EWC | 0 |
| Written to EWC object store (same DC) | 4.4 TB compressed (7.37 TB raw) |
| **Read amplification** | **8× vs raw, 14× vs compressed** |

The 61 TB is now **measured, not assumed**. ECMWF publishes a `.index`
sidecar next to every GRIB file listing `_offset` and `_length` for each
message; averaged over the 1,500 messages that make up the channels we
extract, the true size is **0.788 MB per message** (0.307 MB for the coarser
0p4 grid). An earlier draft of this plan assumed 1.5 MB and therefore
overstated the ingress as 115 TB.

| group | messages | MB/message | read TB |
|---|---:|---:|---:|
| `0p4` | 23.85 M | 0.307 | 7.33 |
| `49r1`-pre | 25.08 M | 0.788 | 19.77 |
| `49r1`-post | 38.44 M | 0.788 | 30.29 |
| `50r1` | 4.14 M | 0.788 | 3.26 |
| **total** | **91.50 M** | | **60.64** |

We download ~61 TB to keep ~4.4 TB. This is not inefficiency in our code — it
is that the archive's atom is a whole global GRIB message and our region is
2.3 % of the globe. There is no server-side subsetting available on the
public archive.

Two things follow: `ecmwf-forecasts` is in the AWS Open Data programme so
there is **no egress charge on the AWS side**, but the sustained inbound rate
(**~93 MB/s to finish in 7.6 days**) is the crux of the whole schedule — and
is well above what we have measured. See §5.4.

### 5.4 The link is the schedule — and it is not currently healthy

This is the most important measurement in this document, and the least
comfortable. Plain HTTP range GETs from the six workers to the public bucket
on AWS `eu-central-1`, no Dask, no Icechunk, no decoding in the path:

| worker | single-stream MB/s |
|---|---:|
| 192.168.1.105 | 9.42 |
| 192.168.1.120 | 10.80 |
| 192.168.1.155 | **14.46** |
| 192.168.1.196 | **0.74** |
| 192.168.1.20 | **0.74** |
| 192.168.1.74 | 9.28 |

A **20× spread across workers at the same moment**, and during the same
session the Icechunk read path failed outright with
`HTTP connect timeout occurred after 3.1s`.

What that does to the schedule, against the measured 60.6 TB:

| per-stream | × 16 concurrent | corpus wall clock |
|---|---:|---:|
| 14.5 MB/s (best observed) | 232 MB/s | **3.0 days** |
| 12.5 MB/s (implied by the Icechunk benchmark) | 200 MB/s | **3.5 days** |
| 0.74 MB/s (worst observed) | 11.8 MB/s | **59 days** |

**The honest range is 3 days to 2 months, and the network alone decides
which.** The 7.6-day figure in §4 assumes the healthy end and per-task rates
that hold under concurrency — an assumption we have not yet been able to test,
because the link degraded while we were trying to test it.

This is the single thing worth resolving before any allocation is committed,
and it is a question about the EWC↔AWS path rather than about our code.

**Note the network cost is identical whether we store 51 members or the mean+sd pair.**
All 51 members must be *read* either way; the reduction only changes what
lands on disk. The ensemble decision is a storage decision, not a
network or runtime one.

---

## 6. A blocker we reported, and then disproved

An earlier version of this document stated that resolving any chunk loads that
array's entire manifest, that a `49r1` pressure-level manifest therefore needs
**89.5 GB** of worker RAM, that **96 % of the archive was unreachable**, and
that the fix was to rebuild the source store with `ManifestSplittingConfig`.

**All of that is withdrawn.** It is recorded rather than deleted because we
had already raised it, and because the reasoning failure is worth being
explicit about.

### What was measured, and what was wrongly inferred

The measurement was real — worker RSS grew 0.52–6.95 GB across the first read
of an array — but the inference was not. Dividing by the array's *total*
chunk-reference count gave ~2000 B/ref, and that product was extrapolated
across whole arrays.

**Listing the store's manifest objects disproves the extrapolation:**

```
46,154 manifest objects
11.11 GB total
largest 1.1 MB     median 0.071 MB     none above 1.1 MB
```

and the repository's own **persisted** configuration, read with
`icechunk.Repository.fetch_config()`, already declares

```
splitting: [(any_array, [(dimension_name("time"), 1)])]
```

**The store is already split one manifest shard per forecast date.** No
per-array manifest exists to exhaust memory on, no era is blocked by RAM, and
the `ManifestSplittingConfig` work we were about to ask for **is already done
upstream**.

The `49r1/u` read that "died after 40 minutes" is now re-attributed to the
network failure that was already in progress, not to memory.

### What we still do not know

- **What the 6.25 GB RSS on a first read actually was.** Not a per-array
  manifest, but unexplained, and worth understanding before a multi-day run.
- **Whether `49r1` and `0p4` pressure-level reads work end to end.** Never
  demonstrated — every attempt coincided with the link failing.

### What replaces it as the constraint

The AWS→EWC link (§5.4). That is a schedule question rather than a feasibility
one, but at the time of writing it is also an availability question: reads are
failing outright.

---

## 7. Proposed phasing

We would rather convert estimates into measurements before requesting a large
allocation. Each phase is a decision point.

| phase | scope | storage | ingress | wall clock @75 MB/s | cluster |
|---|---|---:|---:|---:|---|
| **0. Calibrate** | 1 date, all channels | 7.8 GB | 63.9 GB | ~14 min | current |
| **1. Prove** | `50r1`, 51 dates | 396 GB | 3.26 TB | ~12 h | current |
| — *decision* — | is the link stable enough to commit to phases 2–4? | | | | |
| **2. Recent** | `49r1`-post, 474 dates | 3.68 TB | 30.3 TB | ~112 h | current |
| **3. Backfill** | `49r1`-pre, 320 dates | 2.40 TB | 19.8 TB | ~73 h | current |
| **4. Archive** | `0p4`, 401 dates | 0.89 TB | 7.3 TB | ~27 h | current |
| | **total** | **7.37 TB** *(≈4.4 TB compressed)* | **60.6 TB** | **~225 h** | |

Wall clock scales inversely with the link, so every figure in that column is
÷2.7 at 200 MB/s and ×6.4 at the worst observed 11.8 MB/s. The cluster column
now reads "current" throughout — no larger workers are required (§6).

**Phase 0 is the one that matters for planning.** It converts the two soft
numbers — sustained read throughput and compression ratio — into measured
ones, at a cost of one date. Everything after it can be re-forecast from real
data. It cannot be run until the link recovers.

### What we would ask for

1. **Confirmation of what the EWC↔AWS path can sustain, and why it is
   currently failing.** We measured 0.74–14.5 MB/s per stream across workers
   at the same instant, then connect timeouts on every read. This is the whole
   schedule: the corpus is 3.5 days at 200 MB/s and 60 days at 11.8 MB/s.
2. **~4.5 TB of object storage** in `must-icechunk` (or a decision to store
   mean+sd instead, at 289 GB).
3. **No larger workers are needed** on current evidence — 6 × 4 vCPU × 16 GB
   stands. (The earlier request for 128 GB workers, and for a source-store
   manifest rebuild, are both withdrawn — §6.)
4. Agreement on phasing, so the allocation follows the measurements rather
   than the estimates.

---

## 8. Summary for the impatient

- **Used so far: 3.38 GB stored (all synthetic), ~6 GB read.** No production
  data yet. We were proving feasibility.
- **Planned: ~4.4 TB stored (7.37 TB raw), ~60.6 TB read, 3.5–60 days** for
  1,246 forecast dates × 7-day lead × 51 members over East Africa. The
  duration is a linear function of the link and nothing else.
- **Workers: 6 × 4 vCPU × 16 GB, unchanged.** Neither CPU nor RAM has been
  shown to be a limit; an earlier claim that 128 GB was needed is withdrawn.
- **The schedule is a linear function of one unknown** — what the AWS→EWC
  link sustains. Measured 0.74–14.5 MB/s per stream, currently failing. That
  spans 3.5 to 60 days for the corpus.
- **An earlier "96 % unreachable" finding is withdrawn** (§6). The store is
  already manifest-split; no era is blocked by memory. What blocks progress
  today is simply that the link to the data keeps failing.
