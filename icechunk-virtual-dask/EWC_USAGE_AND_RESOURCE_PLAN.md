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
| Why so little? | We were establishing feasibility first. That work turned up a hard blocker (§6) which we would rather report now than after consuming a large allocation. |
| What is the production workload? | Read the published virtual ECMWF store (chunks live on AWS), crop to East Africa, keep all 51 ensemble members, write a realized Icechunk store on EWC. §3 |
| How much will it store? | **~4.4 TB compressed** (7.37 TB uncompressed) for 1,246 forecast dates. §5 |
| How long will it take? | **~182 h ≈ 7.6 days** of continuous cluster time. §4 |
| What does it need per worker? | **4 vCPU is enough; RAM is the constraint — 16 GB works only for the newest era, 128 GB is needed for the bulk of the archive.** §6 |
| What is the unusual cost? | **~115 TB of cross-cloud ingress** from AWS `eu-central-1` into EWC, to write 4.4 TB. A 26× read amplification, and it is unavoidable. §5.3 |

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

For scale, the production corpus is **91.5 M messages / ~115 TB** — so we
have so far read about **0.004 %** of what the full job requires.

### 2.3 What that testing established

Small as the volume is, it settled the things that decide whether the job is
viable at all:

| finding | status |
|---|---|
| Icechunk commits work on Ceph RGW (conditional writes / `If-None-Match`) | ✅ **measured** — 8/8 checks pass, incl. stale-session rejection |
| Anonymous read of the source store from EWC workers | ✅ **measured** — after fixing an `AWS_*` environment clash |
| All 30 required channels present and finite | ✅ **measured** — finite = 1.00 for every channel in the newest era |
| Sustained read throughput off AWS | ✅ **measured** — 21.6 messages/s per task |
| Worker memory required per source array | ✅ **measured** — ~2000 B per chunk reference |
| Whether the full archive is reachable on the current cluster | ❌ **it is not** — §6 |

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

1. **One store variable is pinned to one worker for the whole run.** Not a
   performance choice — a memory one. Resolving any chunk of a source array
   loads that array's *entire* manifest into RAM (§6), so a worker can hold
   only a small number of them. Pinning also means each manifest is loaded
   once per run instead of once per date.

2. **Cropping to East Africa does not reduce what we read.** A virtual chunk
   is one whole *global* GRIB message. We download a 721×1440 field and keep
   163×147 of it — 2.3 % of the bytes. This is inherent to the archive's
   layout, not a shortcoming of the tool.

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
- **~7.6 days wall clock** — how long the *job runs*.

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
| **total** | **1,246** | **181.7 h = 7.6 days** |

### 4.2 What could make this wrong

Stated plainly, because a 7.6-day estimate off a 176-second measurement
deserves scepticism:

| risk | direction | note |
|---|---|---|
| **Network ceiling** | ⬆ could be much slower | 7.6 days for 115 TB implies **~176 MB/s sustained** from AWS into EWC. If the link caps below that, bandwidth sets the time and no tuning helps. **This is the single largest uncertainty.** |
| Cold manifest loads | ⬆ slightly | 11–215 s per (worker, variable), paid once per run. Negligible over 1,246 dates, dominant over 1. |
| source.coop 5xx responses | ⬆ slightly | sporadic; the tool retries with backoff |
| Larger workers allow more concurrency | ⬇ faster | the critical path is one task; more RAM per worker would let us split it |
| Rate measured on a quiet cluster | ⬆ | contention with other tenants not accounted for |

**We would not commit to 7.6 days on this basis alone.** The plan (§7) starts
with a one-date run that converts the estimate into a measurement before any
large allocation is requested.

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

**vCPU is not the constraint.** The workload is network- and
decode-bound; 4 vCPU per worker is sufficient and the current cluster's 24
threads are not saturated. **RAM is the constraint**, and it is set by the
source store's manifest structure (§6):

| era | surface array | pressure array | minimum worker RAM |
|---|---:|---:|---|
| `50r1` (newest) | 0.44 GB | 6.19 GB | **16 GB** ✅ current cluster works |
| `0p4` (2023) | 3.48 GB | 31.3 GB | **48 GB** |
| `49r1` (2024–26) | 6.88 GB | **89.5 GB** | **128 GB** |

Two provisioning scenarios:

| | workers | vCPU/worker | RAM/worker | total | vCPU-hours |
|---|---:|---:|---:|---|---:|
| **A. As-is (source store unchanged)** | 6 | 4 | **128 GB** | 24 vCPU / 768 GB | ~4,400 |
| **B. After source-store fix** (§6) | 6–12 | 4 | 16 GB | 24–48 vCPU / 96–192 GB | ~4,400–8,700 |

Scenario A is expensive in memory for a workload that uses almost none of it
for actual data — the 128 GB exists solely to hold an index. **Scenario B is
strongly preferred** and is a change to how the source store was written, not
to the cluster.

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
| **Inbound, AWS `eu-central-1` → EWC** | **~115 TB** |
| Outbound from EWC | 0 |
| Written to EWC object store (same DC) | 4.4 TB |
| **Read amplification** | **~26×** |

We download ~115 TB to keep ~4.4 TB. This is not inefficiency in our code — it
is that the archive's atom is a whole global GRIB message and our region is
2.3 % of the globe. There is no server-side subsetting available on the
public archive.

Two things follow: `ecmwf-forecasts` is in the AWS Open Data programme so
there is **no egress charge on the AWS side**, but the sustained inbound rate
(~176 MB/s for 7.6 days) is worth confirming against the EWC link budget
before we start.

**Note this is identical whether we store 51 members or the mean+sd pair.**
All 51 members must be *read* either way; the reduction only changes what
lands on disk. The ensemble decision is a storage decision, not a
network or runtime one.

---

## 6. The blocker, and why we are reporting it now

Resolving **any** chunk of a source array loads that array's **entire**
manifest into RAM — the reference count is
`era_dates × members × steps × levels` for the whole era, regardless of how
little is selected. Measured by RSS delta on a worker:

| array | references | RAM | bytes/ref | load time |
|---|---:|---:|---:|---:|
| `50r1/t2m` | 0.22 M | 0.52 GB | 2195 | 11 s |
| `0p4/t2m` | 1.74 M | 3.61 GB | 2055 | 164 s |
| `49r1/t2m` | 3.44 M | 6.95 GB | 2006 | 155 s |
| `50r1/u` (14 levels) | 3.10 M | 6.25 GB | 2008 | 179 s |
| **`49r1/u` (13 levels)** | **44.75 M** | **~89.5 GB** | — | **worker killed** |

On 13.9 GB workers that makes **51 of 1,246 dates — 4 % of the archive —
extractable today.** The other 96 % fails before reading a single byte of
data. No amount of task splitting or scheduling helps, because the manifest
loads whole.

**The fix is upstream and cheap:** rewrite the source store with
`icechunk.ManifestSplittingConfig` split along `time`, so a read loads one
shard rather than the whole array index. That removes the constraint for every
era at once and drops the worker requirement from 128 GB to ~16 GB.

It is also *time-sensitive in the other direction*: `50r1` is only cheap
because it is currently short (51 dates). At ~250 dates its pressure-level
manifest reaches ~30 GB and it joins the infeasible list. **The window in
which any of this works on modest workers is closing.**

---

## 7. Proposed phasing

We would rather convert estimates into measurements before requesting a large
allocation. Each phase is a decision point.

| phase | scope | storage | ingress | wall clock | cluster |
|---|---|---:|---:|---:|---|
| **0. Calibrate** | 1 date, all channels | 8 GB | 122 GB | ~15 min | current |
| **1. Prove** | `50r1`, 51 dates | 396 GB | 6.2 TB | ~9 h | current (6 × 4 vCPU × 14 GB) |
| — *decision* — | fix the source store (§6) before going further | | | | |
| **2. Recent** | `49r1`-post, 474 dates | 3.68 TB | 57.7 TB | ~82 h | 6 × 4 vCPU × 16 GB *(post-fix)* |
| **3. Backfill** | `49r1`-pre, 320 dates | 2.40 TB | 37.6 TB | ~56 h | as phase 2 |
| **4. Archive** | `0p4`, 401 dates | 0.89 TB | 14.0 TB | ~35 h | as phase 2 |
| | **total** | **7.37 TB** *(≈4.4 TB compressed)* | **115 TB** | **182 h** | |

**Phase 0 is the one that matters for planning.** It converts the two soft
numbers — read throughput and compression ratio — into measured ones, at a
cost of one date. Everything after it can be re-forecast from real data.

### What we would ask for

1. **Confirmation of the sustained inbound rate** EWC can carry from AWS
   `eu-central-1` — ~176 MB/s for several days. This is the largest
   uncertainty in the estimate.
2. **~4.5 TB of object storage** in `must-icechunk` (or a decision to store
   mean+sd instead, at 289 GB).
3. **Either** 128 GB workers, **or** — much preferably — support for
   rewriting the source store with manifest splitting, after which 16 GB
   workers suffice.
4. Agreement on phasing, so the allocation follows the measurements rather
   than the estimates.

---

## 8. Summary for the impatient

- **Used so far: 3.38 GB stored (all synthetic), ~6 GB read.** No production
  data yet. We were proving feasibility.
- **Planned: ~4.4 TB stored, ~115 TB read, ~7.6 days** for 1,246 forecast
  dates × 7-day lead × 51 members over East Africa.
- **Workers: 4 vCPU is plenty; RAM decides everything.** 16 GB after a
  source-store fix, 128 GB without it.
- **The 7.6 days rests on one solid measurement** (21.6 messages/s at
  production task shape) **and one open question** (whether the AWS→EWC link
  sustains ~176 MB/s). One calibration date settles it.
- **96 % of the archive is currently unreachable** on this cluster for reasons
  that have nothing to do with storage or time, and everything to do with how
  the source store's index is laid out.
