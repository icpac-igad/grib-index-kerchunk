# What one daily initialisation costs

**2026-09-29.** `REALIZATION_COST.md` costs the **backfill** — 887 dates, once.
This costs the **operation** — one initialisation, every day, forever.

Read §1 first. The largest physical quantity in this pipeline does not appear
on any bill, and a cost table that opens with the stored volume repeats that
omission.

> **Where these numbers come from.** Everything in §1–§3 is **measured on the
> ECMWF Weather Cloud** (6 VMs, 48 readers), from the build logs of every season
> in the published store. §4 onwards is a **GCP price estimate applied to that
> measured profile** — no GCP run has been done. EWC is where this actually
> runs; GCP is a costing exercise for a centre that would have to buy the
> compute.

---

## 1. The quantity that is not on the bill

Per date, measured (`REALIZATION_COST.md` §3):

| | 1 cycle/day | all four cycles |
|---|---|---|
| global GRIB **fetched from AWS** | **99.3 GB/day** → **36.2 TB/yr** | 313.8 GB/day → **114.5 TB/yr** |
| EA-SWIO **written to the store** | 16.8 GB/day | 53.1 GB/day |
| **kept** | **17%** | 17% |
| **fetched and discarded** | **30.1 TB/yr** | **95.2 TB/yr** |

Every day, ~100 GB of **global** fields cross the Atlantic so that ~17 GB of an
East Africa window can be written. The ECMWF archive sells whole GRIB messages
— a message is one global field — and the EA-SWIO box is 83,781 of 1,038,240
grid points, about 8%. **There is no way to order the corner**, so 92% of every
message is decoded and thrown away.

### Why it reads as free

Two independent policies, neither of which this project controls:

1. `s3://ecmwf-forecasts` is **AWS Open Data** — the bucket owner pays egress,
   so the reader is not billed and the bucket is not requester-pays;
2. **GCP charges nothing for ingress.**

Remove either and the invisible line becomes the largest one:

| | 1 cycle/day | four cycles |
|---|---|---|
| today | **$0** | **$0** |
| at a sponsored $0.05/GB | $1,812/yr | $5,727/yr |
| at ordinary $0.09/GB egress | **$3,262/yr** | **$10,308/yr** |

Against storage at $876–$4,650/yr (§5), **the fetch would dominate the budget**.
Any costing that omits §1 is describing a subsidy, not an architecture.

### And it is never free on a constrained link

The same bytes, per **one** initialisation, on a Nairobi link:

| link | realize locally (fetch global GRIB) | pull the realized store |
|---|---|---|
| **20 Mbps retail** | 11.0 h — **18.4 h at 60% duty** | 1.9 h — 3.1 h |
| 160 Mbps institutional | 1.4 h — 2.3 h | 0.2 h — 0.4 h |

**At 20 Mbps, one initialisation takes longer than the day it belongs to.** The
operation never catches up, and four cycles is arithmetically impossible.
Pulling the finished store is 3.1 h — comfortable. This is the operational
argument for the realized store, and it is the reason the fetch is paid **once,
in Europe, next to the data** rather than by every consumer over their own link.

---

## 2. Wall-clock for one initialisation — measured on EWC

From the per-season build logs behind the published store, which covers
**2023-03-01 → 2026-08-23**, 1,264 of 1,271 date slots across 16 sub-stores:

| | s/date | min/date | mean msg/s |
|---|---|---|---|
| **best season** (`49r1-mam2024`) | 590 | **9.8** | 206 |
| **mean over 887 dates** | **781** | **13.0** | **161** |
| typical recent (`sond`, `jja`) | 798–847 | 13.3–14.1 | 153–160 |
| **worst season** (`50r1-mam2026-tail`) | 1,547 | **25.8** | 87 |

**Plan on ~13 minutes per initialisation, and budget ~26 minutes.** The spread
is not workload — every date is the same 125,970 messages — it is upstream: S3
latency, throttling, and source.coop wobbles. Two effects worth carrying into an
operational schedule:

- **Cluster age costs ~25%.** Rates drift down over days of uptime and recover on
  restart; the runbook restarts every 12 hours. For a daily job that starts a
  fresh cluster, expect the *better* end of the range.
- **Retries are rare but holes are not.** 654 retries across 111.7 M messages
  (5.85 per million), yet ~0.45% of dates finished with committed NaN blocks that
  every date-level check passed. **A chunk-level verification pass belongs in the
  daily schedule**, counting against `39 × 65` per date.

All four cycles: 00z/12z carry the full step axis, 06z/18z about 0.58 of it, so
a full day is ~3.16 × one 00z date ≈ **41 min**.

---

## 3. What the cluster actually needs — measured on EWC

Mid-build, 48 readers active (`REALIZATION_COST.md` §A1):

| | per VM | headroom |
|---|---|---|
| vCPU | 4, load **~1.2** | **~70% idle** |
| RAM | 16 GB, **~4.5 GB used** | ~11.4 GB free |
| worker RSS | **~2.0 GB** steady, **~3.4 GB peak**, 2 per VM | limit 10 GiB, never approached |
| network in | ~15 MB/s | far below any NIC cap |

**The job is I/O-bound with idle CPU and spare RAM.** Throughput is set by how
many requests are in flight; the binding resource is RAM per worker, not cores.

### The cluster was reader-count bound, not source bound

161 msg/s ÷ 6 VMs = **26.8 msg/s per VM** = exactly **8 readers × 3.35 msg/s**.
An independent sweep from an EWC node reached **81.3 msg/s at 48 readers on one
box, with zero retries at 96** — so the source had headroom the cluster was not
using. Measured per-reader rates of 3.32 and 3.27 msg/s at 6 and 12 readers
confirm `REALIZATION_COST.md`'s 3.35 from a second direction.

**More readers per VM should shorten the 13 minutes.** Untested at 6 × 48 = 288
concurrent against `ecmwf-forecasts`; the aggregate may throttle where one box
does not. Treat as a hypothesis worth one afternoon.

*(Measurements: `grib-index-kerchunk/ecmwf/gik_vs_herbie/audit_crosscheck/09_ewc_48readers.py`.)*

---

## 4. Machine shape, if this were bought on GCP

**Do not buy CPU.** Both CPU and memory are provisioned ~3× above need.

| shape | vCPU / RAM | workers | verdict |
|---|---|---|---|
| **`e2-standard-2`** | 2 / 8 GB | **1 × 4 threads** | **recommended.** Twelve give the same 48 readers |
| `e2-standard-4` | 4 / 16 GB | 2 × 4 threads | matches the EWC cluster exactly; the safe choice |
| **`e2-custom-2-4096`** | 2 / **4 GB** | 1 × 4 threads | **no — see below** |
| `n2-standard-4` | 4 / 16 GB | 2 × 4 threads | **no.** 45% dearer for CPU this job does not use |

### Why not 4 GB

Worker RSS is **2.0 GB steady but reaches 3.4 GB** during heavy phases. One
worker on a 4 GB VM sits at ~85% at peak *before the OS*. `REALIZATION_COST.md`
§A2 already records the consequence: **the OOM killer takes a worker and the
supervisor loses a date.** In a backfill that is a retry; in a daily operation
it is a missed initialisation.

And the saving is not there. **The VMs run ~15 minutes a day**, so unused RAM
costs cents per month (§5). Trading an operational failure mode for pocket
change is a bad exchange. **Keep 8 GB per worker.**

### Why twelve VMs rather than six

At 0.25 h/day the bill is identical — 12 × `e2-standard-2` and 6 ×
`e2-standard-4` both come to $0.806/h on-demand. The reasons are operational:
losing one VM costs **1/12 of the cluster instead of 1/6**, which matters on
Spot, where the work is idempotent (`--resume`, `--repair`) and a preemption
costs only the date in flight.

---

## 5. The GCP estimate

Bases, all derived from measured figures: cluster **$0.806/h** on-demand and
**$0.239/h** Spot (from $155 / $46 over 192.4 cluster-hours); GCS Standard
**$0.020/GB/month** (from 14.9 TB = $298/month).

| | 1 cycle/day | all four cycles |
|---|---|---|
| runtime | 13 min/day (billed ~16) | 41 min/day (billed ~44) |
| **compute, Spot** | **$1.91/mo** | **$5.28/mo** |
| compute, on-demand | $6.45/mo | $17.78/mo |
| store grows by | 6.13 TB/yr | 19.4 TB/yr |
| **storage, end of year 1** | **$123/mo** | **$388/mo** |
| storage, 2-year rolling | $245/mo | $775/mo |
| **fetch (§1)** | **$0** — *policy, not physics* | **$0** |

**Compute is two to six dollars a month. Storage is 20–60× that and grows every
day.** If the daily output is also mirrored off GCP, egress at $0.11/GB adds
**~$56/month** at 16.8 GB/day — on its own comparable to the storage bill.

### Levers, in order of size

1. **Lifecycle the storage.** Nearline ($0.010/GB/mo) after 30 days, Coldline
   ($0.004/GB/mo) after 90 — halves, then quarters, the accumulating cost for
   dates the cGAN reads rarely.
2. **Choose the retention window deliberately.** Two years at four cycles is
   $775/month; a 90-day hot window is under $50.
3. **Coarsen the atmospheric channels.** `crma-bn-6node-res/…/COMPUTE_COST.md`
   §7 measured 0.5° atmospheric + 0.25° `tp` at **2.78 GB against 10.21 GB — 3.7×**
   for the downstream stage, with the caveat that 1° breaks two things.
4. **Compute where the output lives.** The EWC setup writes to Ceph on the same
   network, so only the final mirror crosses a boundary. That property is worth
   more than any machine-shape choice.

---

## 6. Caveats

- **§1–§3 are measured on EWC. §4–§5 are not measured anywhere** — they price
  the EWC profile against GCP's catalogue. No GCP run has been done.
- Prices are approximate us-central1 list as of 2026-09. Sustained-use,
  committed-use and current Spot rates are not modelled and all of them move.
- The 16.8 GB/date figure is 38 channels. At 39 it is slightly higher; a quoted
  "10 GB/date" does not match the published store, which is 9.42 TB over 544
  days (**17.3 GB/day**) and ~17.7 TB over 1,264 days (**14.0 GB/day**).
- **EWC is allocated, not billed per hour**, so the §5 table is what this would
  cost a centre buying compute — not what it costs today.
- The Nairobi figures in §1 are arithmetic on assumed bandwidth. One `iperf3`
  from the ICPAC machine replaces them with a measurement.
