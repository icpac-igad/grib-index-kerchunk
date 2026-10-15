# What the realization cost, and how it compares to downloading GRIB

2026-09-02. Measured from the build logs of every season realized into
`ecmwf-ifs-ea-swio-realized-v3`, plus a structural comparison against the
obvious alternative: fetching the same fields with Herbie from the raw GRIB
archive.

Read the two halves differently. **§1–4 are measurement** — every number comes
from a build log or an object listing, and can be re-derived. **§5 is
analysis** — a like-for-like Herbie benchmark was never run here, so that
section reasons from structure and states its assumptions rather than quoting
timings it does not have.

## 1. What was built

| season | dates | hours | s/date | mean msg/s | min | max | retries |
|---|---|---|---|---|---|---|---|
| `49r1-mam2024` | 92 | 15.1 | 590 | 206 | 142 | 235 | 27 |
| `49r1-mam2025` | 92 | 19.0 | 745 | 173 | 61 | 231 | 135 |
| `49r1-mam2026` | 73 | 13.6 | 673 | 185 | 102 | 240 | 87 |
| `50r1-mam2026-tail` | 19 | 8.2 | 1547 | 87 | 50 | 172 | 14 |
| `49r1-jja2024` | 92 | 18.4 | 721 | 176 | 130 | 192 | 10 |
| `49r1-jja2025` | 92 | 20.7 | 809 | 158 | 108 | 178 | 57 |
| `50r1-jja2026` | 33 | 7.0 | 759 | 167 | 141 | 182 | 0 |
| `50r1-jja2026-tail` | 51 | 12.0 | 847 | 153 | 91 | 175 | 11 |
| `49r1-sond2024` | 122 | 27.1 | 800 | 160 | 107 | 193 | 71 |
| `49r1-sond2025` | 122 | 27.0 | 798 | 160 | 99 | 179 | 48 |
| `49r1-jf2026` | 56 | 13.6 | 875 | 147 | 92 | 176 | 110 |
| `49r1-jf2025` | 43* | 10.7 | 895 | 144 | 84 | 171 | 84 |
| **total** | **887** | **192.4** | **781** | | | | **654** |

\* JF 2025 still running at time of writing; JF 2026 shows 56 rather than 59
because three dates were completed by `--repair`, which logs differently.

**192.4 hours of cluster time — a little over eight days** — for 887 forecast
days. The seasons ran across three weeks of wall-clock, but that includes
credential expiries, a VM reboot, and the idle gaps between them.

## 2. The compute

| | |
|---|---|
| cluster | 6 VMs x 2 frisky workers x **4** threads = **48 concurrent readers** |
| cluster the runbook specifies | 6 x 2 x **8** = **96 readers** -- see §2a |
| VM-hours | **1,458** (1,154 at the time of writing) |
| unit of work | one GRIB message -> one 321x261 float32 slab |
| messages per date | 38-39 channels x 51 members x 65 steps = **~126,000** |
| messages total | **~156 million** |
| mean | 694 s/date, 197 msg/s (166 for the 0.25 deg seasons alone) |

Two things the table above shows that matter more than the totals.

**Variance is upstream; the *level* was not.** The spread within a single
season (min 84, max 240 msg/s) is wider than the spread between seasons, and
nothing about the workload changes date to date -- so the date-to-date variation
is upstream: S3 latency, throttling, source.coop wobbles.

An earlier version of this section went further and said *"adding workers past
48 does not help."* **That was wrong**, and §2a corrects it. 48 was not a
ceiling that had been found; it was the `--nthreads` default of 4 taking effect
because the deploy command omitted the flag. Per-reader throughput turns out to
be flat at ~3.46 msg/s, which is the signature of a reader-starved cluster, not
a saturated source.

**Cluster age costs about 25%.** Rates drift down over days of uptime and
recover on restart: JF 2026 opened at 175 msg/s and reached 132 before a
restart put the next season back at 165. The pattern is consistent enough that
a restart every 12 hours is written into the runbook. Distinguishing it from
upstream throttling matters and is easy to get wrong — a *monotonic* decline
with **zero retries** is the cluster; a rate drop **with** retries that heals
itself is the source.

## 2a. The corpus ran at half its provisioned concurrency

> The DAG itself is tagged **v1** in [`DAG_VERSIONS.md`](DAG_VERSIONS.md)
> (diagram: [`dag/realize_dag_v1.drawio`](dag/realize_dag_v1.drawio)). Reader
> count is a deployment parameter, not a version.

Every build in this record used `--workers-per-vm 2` **without `--nthreads`**,
so it took the default of 4: `6 x 2 x 4 = 48` readers. The configuration
settled in `RUNBOOK_REALIZE.md` and `SCALING.md` is `--nthreads 8`, i.e. **96**.

The evidence that 48 was starvation rather than saturation is that per-reader
throughput does not sag as readers are added:

| source | readers | msg/s per reader |
|---|---|---|
| this corpus, 902 0.25 deg dates | 48 | **3.46** |
| `REALIZATION_COST` §2, earlier count | 48 | 3.35 |
| independent EWC sweep, 6 and 12 readers | 6, 12 | 3.32, 3.27 |

A saturated source would show the per-reader figure falling as concurrency
rises. It is flat from 6 to 48, so the aggregate was set by how many readers
were running -- and half of them were not.

**Recoverable time, 0.25 deg seasons only** (902 dates, 196.1 of the 243.0
hours). Doubling readers at a flat per-reader rate halves the time; discount it
for the throttling that `SCALING.md` §4 already sees *at* 96:

| | hours |
|---|---|
| measured at 48 readers | **196.1** |
| at 96, if per-reader held exactly | 98.1 |
| at 96, with ~15% lost to SlowDown retries | **~113** |

Across the whole corpus that is **243 -> roughly 140-155 hours**, on the same
six VMs, by passing one flag. The 0p4 seasons are a smaller prize and are left
out of the projection: their messages are 39% the size, they already ran at
222-307 msg/s, and they are correspondingly less reader-bound.

### What this does *not* mean

**More VMs still will not work.** The ceiling above 96 is real and it is
shared, not per-VM: `SCALING.md` §4 pushed 12 workers x 16 threads = 192 in
flight and got `SlowDown: Please reduce your request rate` on 56 of 1,590
blocks, with zero Ceph errors -- entirely the AWS read side. Per-VM throughput
tops out near 50 MB/s at two processes and is flat past that. So the honest
shape of the ceiling is:

- **below ~96 readers** -- reader-starved, add threads
- **at ~96** -- intermittent SlowDown, the practical operating point
- **above ~96** -- throttled, and adding VMs makes it worse because it raises
  the request rate against the same quota

Sustained high throughput past that is not a tuning problem; it needs reading
from `eu-central-1`, which is `BLOCKERS.md` §10.

### Before relying on the projection

The bandwidth half is measured (`REALIZATION_CONCURRENCY.md` §4a-4b,
2026-10-04). The shared EWC path is **not** the ceiling: two probe processes per
VM reached **216.5 MB/s (1.73 Gbps) with zero 503s**, against 123.4 MB/s at one.

But the gain depends on the workload. A single-variable probe gives **1.75x**
for doubled readers; a realistic 38-channel read gives **1.27x**. So 243 hours
becomes **139-191**, and the corpus's own locality should put it toward the
better end. Pushing to ~200 readers rather than 96 measures **1.83x**, i.e.
**~135 hours**.

Two ideas that looked promising and are **refuted** in §4b: AWS request-rate
throttling (zero errors at 100/200/400 readers -- `SCALING.md`'s 192-reader
`SlowDown` did not reproduce) and per-object hot-keying (concentrating reads on
one object is **15% faster** than spreading them, so the DAG's ordering is
right as it stands).

What remains unmeasured is the *full pipeline* at 96 readers: the probe reads
and discards, where a build also decodes, forks, merges and commits. One season
at `--nthreads 8` settles it; watch the **per-reader** rate, not the aggregate.
Note also that one VM had dropped out of the cluster by 2026-10-04, so the
six-VM figures are an extrapolation from five.


## 3. The data

| | per date | total (887 dates) |
|---|---|---|
| GRIB read from AWS | 99.3 GB | **88.0 TB** |
| written to the store | 16.8 GB | **14.9 TB** |
| amplification | **5.9x** | |

That 5.9x is the central fact of this pipeline. Every date reads ~99 GB of
packed GRIB to keep ~17 GB of float32 over a 321x261 box. The store is
*smaller* than what was read, despite float32 being less compact than GRIB
packing, because the domain is a small window on a global field: 83,781 of
1,038,240 grid points, about 8%.

The read cost is therefore irreducible for any approach that starts from these
GRIB files. **A message is the smallest thing the archive will sell you.** You
cannot fetch the East Africa corner of a global field; you fetch the field and
throw away 92% of it. Everything in §5 follows from this.

Sustained read across the whole exercise: **127 MB/s**, or ~21 MB/s per VM.

## 4. Reliability

**654 retries across 111.7 million messages — 5.85 per million.** Effectively
all were transient S3 or source.coop faults absorbed by the backoff.

What the retry counter does *not* capture is more interesting. Four dates
finished with **holes**: blocks that failed after their retries were exhausted,
committed anyway, and left the date on the time axis reading as NaN. Three in
JF 2026, one so far in JF 2025 — roughly **0.45% of dates**.

Those cost one block each and `--repair` filled them in 1.6 minutes. But they
are invisible to every date-level check: the store reported 59/59 dates, all
committed, `ds.sizes["time"] == 59`. Only counting chunks against
`38 x dates x 65` exposed them. **Budget for a chunk-level verification pass on
every season; it is not a formality.**

## 5. Against a Herbie / raw-GRIB setup

No like-for-like benchmark was run. Herbie was used here for **correctness**
validation — 21 dates, r > 0.9999 against the realized store — not for timing.
What follows is structural, and the assumptions are stated so they can be
argued with or measured later.

> **Scope of the logs behind this document.** They establish one thing: what the
> **GIK virtual store → frisky** realization cost. Every build read
> `forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd` on source.coop and nothing else —
> there is no `herbie` import anywhere in the write path (`build_corpus.py`,
> `frisky_daily_dag.py`, `sink_icechunk.py`, `parallel_write.py`), and from
> 2026-09-26 every build log records the source store and its **snapshot id**,
> so the record stands without reference to a source-code constant that may
> since have changed.
>
> They cannot establish what a Herbie routine would have cost, because none was
> measured here. **That half is audited in Claude Code session
> `25e05358-82fe-4411-ae92-d2fc00290e1d`.** Read §5 below as the structural
> argument it is, and take the Herbie numbers from there.

### What is identical

**Egress.** Both approaches fetch the same GRIB messages by byte range from the
same bucket. 88 TB is a floor set by the archive's granularity, not by the
tooling. Any claim that one approach moves less data than the other is wrong
unless it fetches fewer messages.

**Decode.** Both run the same GRIB decoder over the same bytes. gribberish here,
eccodes/cfgrib under Herbie; a constant factor, not a structural difference.

### Where they diverge

**1. Index acquisition.** Herbie locates a message by fetching the `.index`
sidecar for its GRIB file and parsing it. The 00z cycle has 87 files per date,
each with an index listing every message. A run that touches all of them pays
that cost *every time it runs*. The GIK virtual store resolves the same lookup
from a manifest built once and shared by every consumer — the byte range is
already known before the job starts.

This is the difference that compounds. Realizing 887 dates re-reads nothing;
887 Herbie runs would re-fetch and re-parse the same indexes 887 times.

**2. Unit of parallelism.** Frisky schedules **one task per message** — 125,970
per date, 48 in flight. Herbie's natural unit is a file or a variable subset,
which is coarser: a slow file blocks its whole batch, and the tail of a date is
one straggler rather than one message. Fine granularity is also what makes
`--repair` possible: a lost block is re-fetched alone, at a cost of seconds.

**3. Recovery.** Because the manifest records what *should* exist, a partial
date is detectable and repairable at block granularity (§4). A download-driven
flow has no such record; the natural recovery is to redo the date, ~1,908
blocks instead of one.

**4. What the consumer gets.** The realized store is the real output here. A
Herbie run yields files a consumer must then index, subset and concatenate; the
store yields `xr.open_zarr` over 544+ days with no GRIB decoder in the reader's
environment at all. That is the point of the exercise, and it is not a
performance claim.

### The honest summary

For **one date, once**, Herbie is simpler and the difference is small — both
fetch ~99 GB and decode it. The virtual-store approach earns its complexity at
**scale and on repetition**: a shared prebuilt index instead of per-run index
parsing, message-level parallelism and message-level repair, and an output
other people can read without any of this machinery.

### What would settle it

A fair benchmark is one date, both paths, same box and channels, same VM,
measuring: index acquisition, decode, and total wall-clock separately. That is
a few hours of work and has not been done. Until it is, treat §5 as reasoning
about structure, not a measured result.

## 6. Planning numbers

For sizing future seasons, from §1–3:

| | |
|---|---|
| build | **~13 min/date** at 48 readers on a fresh cluster |
| a 92-day season | **~20 h**, ~1.55 TB |
| storage | **16.8 GB/date** |
| egress | **99.3 GB/date** |
| mirror to source.coop | **~430 GB/h**, so ~3.6 h per 92-day season |
| verify | manifest scan ~3 min/season; do not use `check_complete.py`, it reads chunk bytes and times out |

A season is therefore about **a day of cluster time end to end** — build,
verify, mirror, reconcile — and the wall-clock is dominated by the build.

---

# Appendix: running this on GCP

Added 2026-09-02. Sizing is derived from the measured load on the current VMs;
the prices are approximate us-central1 list prices and **must be re-checked
before anyone commits money to them**.

## A1. What the workload actually demands

Measured on the six worker VMs mid-build, 48 readers active:

| | per VM | headroom |
|---|---|---|
| vCPU | 4, load **~1.2** | **~70% idle** |
| RAM | 16 GB, **~4.5 GB used** | ~11.4 GB free |
| frisky worker RSS | ~2.0 GB each x 2 (peak ~3.4 GB) | limit set to 10 GiB, never approached |
| network in | **~15 MB/s** | far below any cloud NIC cap |
| network out | ~7 MB/s | |

**The job is I/O-bound with idle CPU and spare RAM.** A reader thread spends
its life waiting on an S3 round trip, not computing: 48 threads keep four cores
at 30%. Both CPU and memory are provisioned roughly 3x above need.

That has a direct consequence for sizing. **Do not buy CPU.** The thing that
sets throughput is how many requests are in flight, and the thing that limits
how many you can run is RAM per worker, not cores.

### The one real constraint

Worker RSS was ~2.0 GB steady but reached **~3.4 GB** during heavy phases. Two
workers per VM therefore need **~7 GB at peak** plus the OS. An 8 GB VM running
two workers is at ~85% — survivable, but the OOM killer takes a worker and the
supervisor loses a date. Either give two workers ~12 GB, or run one worker per
smaller VM.

## A2. Machine recommendation

| shape | vCPU / RAM | workers | verdict |
|---|---|---|---|
| **`e2-standard-2`** | 2 / 8 GB | **1 x 4 threads** | **recommended.** RAM comfortable for one worker; CPU still only ~15% used. Twelve of these give the same 48 readers. |
| `e2-standard-4` | 4 / 16 GB | 2 x 4 threads | matches the current cluster exactly. Proven, and the safe choice if you would rather not re-tune. |
| `e2-custom-2-12288` | 2 / 12 GB | 2 x 4 threads | best fit on paper — the RAM two workers actually need, none of the cores they do not. |
| `n2-standard-4` | 4 / 16 GB | 2 x 4 threads | **no.** 45% dearer for CPU this job does not use. |
| `e2-highcpu-*` | high CPU, low RAM | — | **no.** Backwards: starves the binding resource to buy the idle one. |

Prefer **more small VMs over fewer large ones**, for the same money: twelve
`e2-standard-2` cost the same as six `e2-standard-4` and give the same 48
readers, but losing one costs 1/12th of the cluster instead of 1/6th — and a
lost VM costs the in-flight date.

**Use Spot.** The work is idempotent and resumable: `--resume` skips committed
dates, `--repair` fills partial ones. A preempted VM costs the date it was on,
and the supervisor already retries. That is a ~70% discount for a risk the
pipeline is built to absorb.

## A3. Cost

**Build — 192.4 cluster-hours for 887 forecast days:**

| configuration | | cost | per date |
|---|---|---|---|
| 6 x `e2-standard-4` on-demand | | **$155** | $0.17 |
| 12 x `e2-standard-2` on-demand | | **$155** | $0.17 |
| 6 x `e2-standard-4` **Spot** | | **$46** | $0.05 |
| 12 x `e2-standard-2` **Spot** | | **$46** | $0.05 |
| 6 x `n2-standard-4` on-demand | | $224 | $0.25 |

**Storage and transfer:**

| | |
|---|---|
| AWS -> GCP ingress, 88 TB read | **$0** — ingress is free, and `ecmwf-forecasts` is AWS Open Data, so no requester-pays egress either |
| GCS Standard, 14.9 TB | **~$298 / month** |
| GCS -> internet egress, 14.9 TB | **~$1,639 one-time** |

### The number that matters

**Egress dwarfs compute by more than 10x.** The whole 887-day build costs $46
on Spot; shipping the 14.9 TB result off GCP to source.coop costs ~$1,639 —
**36x the compute**, and it recurs every time the store is republished
elsewhere.

Reading 88 TB is free, because AWS Open Data sponsors it and GCP does not
charge ingress. Writing 14.9 TB out is not. So the economics say: **do the
compute anywhere; be deliberate about where the output lives.**

Three ways that plays out:

- **Publish from GCS itself.** No egress at build time; consumers pay their own
  reads. Cheapest, but moves the store off source.coop.
- **Compute on GCP, publish to source.coop.** ~$1,639 per full republish. Fine
  once, painful if repeated.
- **Compute where the output already lives** — which is what the current EWC
  setup does. The Ceph store is on the same network as the workers, so the
  build writes for free and only the final mirror crosses a boundary.

### Per 92-day season

| | on-demand | Spot |
|---|---|---|
| compute (6 x `e2-standard-4`, ~20 h) | $16 | **$5** |
| egress of 1.55 TB to source.coop | **$170** | $170 |

A season's compute is lunch money. Its egress is not.

## A4. Caveats

- **Prices are approximate us-central1 list prices as of 2026-09.** Sustained-use
  discounts, committed-use discounts and current Spot rates are not modelled, and
  all of them move. Re-check before budgeting.
- **Region matters for latency, not for the ingress bill.** `ecmwf-forecasts` is
  in `eu-central-1`; a European GCP region will see lower round-trip times, and
  since this job is latency-bound rather than bandwidth-bound, that may be worth
  more than the per-hour price difference. Untested here.
- **192.4 h is measured on the current cluster**, which sustained 161 msg/s mean.
  A GCP cluster of the same reader count should land near it because the
  bottleneck is upstream (§2), but that is a prediction, not a measurement.
- **No GCP run has been done.** Everything above is the current cluster's
  measured profile priced against GCP's catalogue.
