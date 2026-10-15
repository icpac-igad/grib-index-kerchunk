# Reader concurrency: what limits the realization, and what is still on the table

2026-10-03. Re-derived from the build logs of every season in
`ecmwf-ifs-ea-swio-realized-v3`. Written because the question "can we add
workers to cut the 243 hours?" has a two-part answer, and the two parts
point in opposite directions.

**Short version.** Adding VMs or threads *past 96 readers* will not work — the
binding limit is a request-rate quota on `s3://ecmwf-forecasts` that is shared
across the cluster and already pushes back at 96. But the corpus was built at
**48** readers, not the 96 the runbook specifies, because a deploy command
omitted `--nthreads`. That is roughly **90–100 hours of recoverable time on the
existing six VMs**, and it costs one flag.

**The DAG that produced these hours is tagged v1** in
[`DAG_VERSIONS.md`](DAG_VERSIONS.md), with its shape, counts and the measured
facts drawn in [`dag/realize_dag_v1.drawio`](dag/realize_dag_v1.drawio). The
48-reader figure below is a *deployment parameter*, not a property of v1 --
that distinction is the point of tagging it.

## 1. What was measured

1,260 dates, **243.0 cluster-hours**, 934 retries, across 16 seasons:

| season | dates | hours | s/date | mean msg/s | retries |
|---|---|---|---|---|---|
| `49r1-sond2024` | 122 | 27.1 | 800 | 160 | 71 |
| `49r1-sond2025` | 122 | 27.0 | 798 | 160 | 48 |
| `49r1-jja2025` | 92 | 20.7 | 809 | 158 | 57 |
| `49r1-mam2025` | 92 | 19.0 | 745 | 173 | 135 |
| `49r1-jja2024` | 92 | 18.4 | 721 | 176 | 10 |
| `49r1-mam2024` | 92 | 15.1 | 590 | 206 | 27 |
| `0p4-mam2023` | 85 | 14.4 | 612 | 222 | 23 |
| `49r1-jf2025` | 58 | 14.3 | 890 | 145 | 92 |
| `49r1-mam2026` | 73 | 13.6 | 673 | 185 | 87 |
| `49r1-jf2026` | 56 | 13.6 | 875 | 147 | 110 |
| `0p4-sond2023` | 122 | 13.6 | 401 | 307 | 69 |
| `0p4-jja2023` | 92 | 12.2 | 477 | 262 | 141 |
| `50r1-jja2026-tail` | 51 | 12.0 | 847 | 153 | 11 |
| `50r1-mam2026-tail` | 19 | 8.2 | 1547 | 87 | 14 |
| `50r1-jja2026` | 33 | 7.0 | 759 | 167 | 0 |
| `0p4-jf2024` | 59 | 6.7 | 409 | 298 | 39 |
| **total** | **1,260** | **243.0** | **694** | | **934** |

Split by grid, because the two behave differently:

| | dates | hours | mean msg/s |
|---|---|---|---|
| 0.25 deg (`49r1`, `50r1`), 38–39 ch | 902 | **196.1** | 166 |
| 0.4 deg (`0p4`), 36 ch | 358 | 46.9 | ~270 |

The 0p4 seasons run faster per message because a 0p4 global field is 405,900
points against 1,038,240 — 39% of the bytes for the same one-message unit of
work.

## 2. The configuration that actually ran

```
deploy_workers.py --start --workers-per-vm 2          # --nthreads omitted
```

`--nthreads` defaults to **4** (`deploy_workers.py:312`), so this gave
`6 VMs x 2 workers x 4 threads = 48` concurrent readers. The settled
configuration in `RUNBOOK_REALIZE.md` and `SCALING.md` §5 is:

```
deploy_workers.py --scheduler-on <ip> --start \
    --workers-per-vm 2 --nthreads 8 --memory-limit 6GB    # = 96 readers
```

Nothing failed and nothing looked wrong, which is why it ran this way for
1,260 dates. The rates were plausible, the retry count was low, and the
seasons completed.

## 3. Why 48 was starvation, not saturation

Per-reader throughput is **flat** as readers are added:

| source | readers | msg/s per reader |
|---|---|---|
| this corpus, 902 0.25 deg dates | 48 | **3.46** |
| `REALIZATION_COST.md` §2 | 48 | 3.35 |
| independent EWC sweep | 6 | 3.32 |
| independent EWC sweep | 12 | 3.27 |

That flatness is the whole argument. A source at its limit would show
per-reader throughput *falling* as concurrency rises, because the readers would
be queueing behind each other. From 6 readers to 48 it does not move.

A second, blunter confirmation: an independent sweep on **one** EWC box reached
**81.3 msg/s at 48 readers with zero retries at 96**. The entire six-VM cluster
averaged 166 msg/s on the 0.25 deg seasons. One box, driven properly, was doing
half the work of six.

Supporting evidence that the cluster was not short of anything local
(`DAILY_OPERATION_COST.md` §3): per VM, **~70% of vCPU idle**, 4.5 of 16 GB RAM
used, network in ~15 MB/s against a per-VM ceiling near 50. Idle on every axis
except readers.

## 4. What is recoverable

Doubling readers at a flat per-reader rate halves the time. Discount for the
throttling `SCALING.md` §4 observes *at* 96 — 6 of 1,590 blocks failing on one
date, SlowDown counts 40/36/30 across workers:

| | 0.25 deg hours | whole corpus |
|---|---|---|
| measured at 48 readers | **196.1** | **243.0** |
| at 96, per-reader rate holding | 98.0 | ~121 |
| at 96, ~15% lost to SlowDown retries | **~113** | **~140–155** |

The 0p4 seasons are deliberately not projected: already at 222–307 msg/s with
39%-size messages, they are much less reader-bound and the gain there is
smaller and less certain.

**This was arithmetic when first written; §4a now measures the bandwidth half
of it.** The shared EWC path carries 1.73 Gbps with zero 503s, and doubling
in-flight readers returns 1.75x -- so the figures above are revised to that
rather than a naive 2x. What is still unmeasured is the *full pipeline* at 96
readers: the probe reads and discards, where a build also decodes, forks,
merges and commits. One season at `--nthreads 8` settles that, and the number
to watch is the **per-reader** rate: if it holds near 3.46 the projection
stands; if it sags, the source is pushing back.

## 4a. The shared EWC path is not the ceiling -- measured 2026-10-04

The projection in §4 needed the EWC->AWS path to carry ~260 MB/s, and until now
the highest aggregate anyone had demonstrated was **165.2 MB/s**. That figure
came from `deploy_workers.py --bandwidth`, which launches **one** probe process
per VM -- so each VM was capped by the per-process icechunk connection limit
(~25-31 MB/s) and the result said nothing about the link.

Re-run with two processes per VM, all VMs firing at once, total in-flight held
at the corpus and runbook configurations rather than pushed to 384 (above ~96
the AWS quota returns SlowDown and would confound a bandwidth measurement with
a policy one):

| config | readers in flight | per VM | **aggregate** | 503s |
|---|---|---|---|---|
| 1 proc x 8 threads x 5 VMs | 40 | 24.7 MB/s | **123.4 MB/s** (0.99 Gbps) | **0** |
| 2 procs x 8 threads x 5 VMs | 80 | 43.3 MB/s | **216.5 MB/s** (1.73 Gbps) | **0** |

Three things follow.

**The link carries at least 1.73 Gbps.** That is 31% above the previous best
demonstrated figure and 70% above the 127 MB/s the corpus sustained. The shared
path was never the constraint at the rates this corpus ran.

**Doubling in-flight readers returns 1.75x, not 2x.** 87% efficiency; the
missing 13% is contention on the shared path, the same effect the older
one-process test measured as 11%. Per-VM 24.7 -> 43.3 MB/s also reproduces
`procs_probe`'s single-VM 25.0 -> 46.8 from a second direction, which is worth
more than either measurement alone.

**Zero 503s at 80 readers.** The AWS quota is not engaged at this level, so the
1.75x is a clean bandwidth-and-concurrency result rather than a throttled one.

Extrapolated to six VMs at 16 readers each: **6 x 43.3 = ~260 MB/s**, which is
what §4 assumed. The projection stands, with 1.75x substituted for the naive 2x:

| | 0.25 deg hours | corpus |
|---|---|---|
| measured at 48 readers | 196.1 | 243.0 |
| at 96 readers, scaling at 1.75x | **112** | **~139** |

### Two things this test does not settle

**It reads one variable.** `bandwidth_probe.py` reads `t2m` over and over, so
manifest and object locality flatter it. §4b measures a realistic 38-channel
read and gets materially less.

**It ran on five VMs, not six.** One VM had dropped out of the cluster by
2026-10-04, so the 260 MB/s six-VM figure is an extrapolation from five.
Recovering that VM is independently worth ~20%.

## 4b. The concurrency curve on a realistic read -- and two refuted ideas

§4a's 1.75x came from a single-variable probe. Re-measured on the workload the
pipeline actually runs -- 38 mixed channels including pressure levels, step
rotated so reads scatter over 85 objects, 1,800 messages per point, 5 VMs:

| readers in flight | msg/s | MB/s | per reader | errors |
|---|---|---|---|---|
| 50 | 113 | 89 | 2.26 | 0 |
| 100 | 143 | 112 | 1.43 | **0** |
| 200 | 207 | 163 | 1.03 | **0** |
| 400 | 227 | 179 | 0.57 | **0** |

**Doubling 50 -> 100 returns 1.27x, not 1.75x.** The honest range for 48 -> 96
on the real pipeline is **1.27-1.75x** -- 1.27 from this curve, 1.75 from the
single-variable probe -- so **243 hours becomes 139-191**. The corpus has more
locality than this test (it reads 51 consecutive members out of one object), so
it should land toward the better end, but the top of that range was never a
measured figure for mixed-channel work and §4a overstated it.

Pushing further is worth more than stopping at 96: 50 -> 200 readers is
**1.83x** and 50 -> 400 is **2.0x**, putting the practical floor around
**~135 hours** at ~200 readers.

### Refuted: AWS request-rate throttling as the ceiling

`SCALING.md` §4 reports 192 readers failing with `SlowDown` on 56 of 1,590
blocks (2026-08-03). **That did not reproduce.** Zero errors at 100, 200 and
400 concurrent readers on 2026-10-04. Either the behaviour changed, the failure
needed longer exposure than these ~10 s bursts, or it was specific to a full
build rather than a read probe. **Until it reproduces, "the AWS quota blocks
horizontal scaling" should not be relied on** -- it is the premise behind every
"do not add VMs" conclusion in this repo, including an earlier version of §5.

### Refuted: per-object concentration as the ceiling

One GRIB object per step holds all 51 members and every variable (~6.5 GB), and
`build_and_submit` submits channel-major then step then member -- so the whole
in-flight set lands on one object. That looked like an obvious hot key, and the
proposed fix was to interleave submissions across step-files. Tested at 200
readers, same message count, same bytes:

| spread | msg/s | MB/s | errors |
|---|---|---|---|
| **concentrated**, 1 object | **175** | 138 | 0 |
| spread, 85 objects | 152 | 120 | 0 |

**Concentration is 15% faster.** The locality is a feature, not a hot key, and
the proposed change would have made the pipeline slower. The DAG's existing
ordering is correct and should not be altered on this reasoning -- see
`DAG_VERSIONS.md`.

### What the ceiling actually is -- RESOLVED 2026-10-05: it was a cold cache

This section stood unresolved for a day and the answer turned out to be a
measurement error in it. See `DAG_VERSIONS.md` §4.

The read path warms up 2.56x within a process (140 -> 359 msg/s over five
passes, 200 readers, zero failures), and crucially it warms on **byte ranges it
has never fetched** -- an alternating disjoint-set test gives 196 cold then 272
on new bytes. So it is not S3 byte caching; it is per-process manifest and/or
connection state, and it generalises to the read-each-message-once case a build
is permanently in.

**The 36 MB/s-per-VM asymptote below was cold.** Warm is 57 MB/s per VM, which
*exceeds* the ~50 MB/s `procs_probe` figure, so the contradiction the section
below puzzles over does not exist. The original text follows for the record.

### What the ceiling appeared to be -- superseded

The curve asymptotes near **227 msg/s / 179 MB/s on five VMs = 36 MB/s per
VM**, and that is *below* the ~50 MB/s per-VM figure from `procs_probe`. So on
present evidence it is:

- **not** reader-bound -- 400 readers buys 10% over 200
- **not** per-VM bound -- 36 < 50 MB/s
- **not** the AWS request quota -- zero errors at 400
- **not** the shared EWC path -- it carried 216 MB/s in §4a

The remaining candidate is **manifest resolution on source.coop**. Every read
is two hops: source.coop for the byte range, then AWS for the bytes. The
metadata hop has never been isolated, and it would produce exactly this
signature -- an asymptote that moves with none of the above. **Isolating it is
the test to run before buying machines**, because if source.coop metadata is
the wall then more VMs change nothing, and the fix is a local manifest cache or
in-region reads rather than hardware.

## 5. Why more VMs is the wrong lever

There are two ceilings and only one of them scales (`SCALING.md` §1–4):

| # | limit | value | scales with VMs? |
|---|---|---|---|
| 1 | icechunk connections per **process** | ~25–31 MB/s | yes — 2 procs/VM reaches ~50 MB/s |
| 2 | **AWS request rate**, `s3://ecmwf-forecasts` | begins below 96 readers | **no — one shared quota** |

Limit 1 is already satisfied: `procs_probe` gives 25.0 → 46.8 → 50.4 → 48.2
MB/s at 1/2/4/8 processes, so two per VM reaches the ceiling and more buy
nothing.

Limit 2 is the one that decides the sizing question, and it was found by pushing
until it broke:

| in flight | result |
|---|---|
| 96 (6 workers x 16 threads) | passed, repeatedly |
| **192** (12 workers x 16 threads) | **56 of 1,590 blocks failed — `SlowDown`** |
| 96 (12 workers x 8 threads) | 364.5 s, 0 failures |
| 96, during a corpus run | 6 of 1,590 failed; SlowDown 40/36/30 |

Every failure carried an `aws_request_id` and **zero** Ceph errors, so it is
entirely the AWS read side. Because the quota is shared, **adding VMs raises
the request rate against it**: 20 VMs is ~320 readers, well past where 192
already failed. Two earlier conclusions in this repo — 36 VMs (`610fa42`) and
later ~20 VMs — both assumed per-VM bandwidth was the only ceiling, and both
are wrong for this reason.

So the shape of the ceiling:

- **under ~96 readers** — reader-starved; add threads, this is free
- **at ~96** — intermittent SlowDown; the practical operating point
- **over ~96** — throttled; more VMs make it worse, not better

Past that, throughput is not a tuning problem. It needs reading in
`eu-central-1` (`BLOCKERS.md` §10), which is what the ~100 Gbps reports are
doing.

## 6. What the realization cost

| | |
|---|---|
| cluster-hours | **243.0** |
| compute, Spot | **$58** |
| compute, on-demand | **$196** |
| AWS fetch | **$0** |
| read from AWS | ~103 TB |
| written to the store | ~17.9 TB |

Compute uses this repo's measured basis — $0.239/h Spot and $0.806/h on-demand
for the whole six-VM cluster, derived from $46 / $155 over 192.4 hours, which
reproduces these figures exactly.

**The $0 fetch is policy, not physics.** `s3://ecmwf-forecasts` is in the AWS
Open Data programme, so the bucket owner pays egress. At an ordinary $0.09/GB
those ~103 TB would be **~$9,270** — about 47x the on-demand compute. The
dominant cost of this pipeline is one someone else is absorbing, and that is
worth knowing before anyone plans around it.

At 96 readers the compute roughly halves (~$29 Spot). It is not the reason to
do it; the reason is 243 hours becoming ~145.

## 7. Caveats

- **Cluster-hours, not wall-clock.** The 1,260 dates spanned weeks of
  calendar time including credential expiries, a worker-VM reboot that changed
  every DHCP address, and idle gaps.
- **The 5.9x read amplification is irreducible.** Each date reads ~99 GB of
  global GRIB to keep ~17 GB over a 321x261 window, because a message is the
  smallest unit the archive sells. Concurrency does not change it.
- **Cluster age costs ~25%** independently of all of this; rates drift down
  over days of uptime and recover on restart, which is why the runbook restarts
  every 12 hours. A *monotonic* decline with **zero** retries is the cluster; a
  drop **with** retries that heals itself is the source. Reading those two the
  wrong way round sends you after the wrong fix.
