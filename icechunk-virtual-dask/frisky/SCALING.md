# What actually limits this read, measured

Every number here was measured on the EWC cluster against the published ECMWF
Icechunk store on 2026-08-03/04. Where an earlier conclusion in this repo was
wrong, it is named and corrected rather than quietly replaced.

**There are two ceilings, not one, and they are hit in this order:**

| # | limit | value | fixed by |
|---|---|---|---|
| 1 | icechunk connections **per process** | ~25–31 MB/s | 2 worker processes per VM → ~50 MB/s |
| 2 | **AWS request rate** on `s3://ecmwf-forecasts` | starts below 96 concurrent readers | nothing local; only reading in-region |

Limit 1 is a per-VM property and scales with VM count. **Limit 2 is shared
across the whole cluster and does not.** That distinction decides every sizing
question below.

---

## 1. Concurrency inside one process saturates at ~31 MB/s

`bandwidth_probe.py`, one process, threads raised:

| concurrency | MB/s | p50 latency | 503s |
|---|---|---|---|
| 4 | 14.4 | 0.20 s | 0 |
| 16 | 27.8 | 0.46 s | 0 |
| 32 | 29.9 | 0.83 s | 0 |
| 64 | 30.6 | 1.60 s | 0 |
| 128 | 30.9 | 2.83 s | 0 |

Flat past 32, with latency rising linearly — extra threads only queue.

### The wrong conclusion drawn from this

Commit `610fa42` read that curve as *"the limit is bandwidth, not in-flight
count"* and sized 1 GB/s at ~36 VMs. **That did not follow.** Raising threads
inside one process cannot distinguish a saturated network from a per-process
connection cap. Both produce exactly this curve.

## 2. It is a per-PROCESS cap. Two processes nearly double it

`procs_probe.py`, independent processes, 32 threads each:

| procs | in flight | MB/s | per-proc |
|---|---|---|---|
| 1 | 32 | 25.0 | 25.0 |
| 2 | 64 | **46.8** | 23.4 |
| 4 | 128 | 50.4 | 12.6 |
| 8 | 256 | 48.2 | 6.0 |

**~50 MB/s is the per-VM ceiling, and two processes reach it.** Past two it is
flat, so more workers per VM buy nothing.

> This probe was itself wrong on its first run and reported 4.9 MB/s. It timed
> from the parent process, so ~15 s of spawn, import and store-open swamped ~4 s
> of actual reading. A same-time single-process control still returning 23–31
> MB/s is what exposed it. It now uses the children's own clocks.

## 3. Per-VM throughput adds — up to a point

`deploy_workers.py --bandwidth`, all six VMs at once, one process each:

```
per-VM peaks : 25.4  27.0  27.5  27.9  28.5  28.9  MB/s
AGGREGATE    : 165.2 MB/s (1.32 Gbps)   mean 27.5 MB/s
```

Only 11% below what one VM reaches alone, so egress is not a shared border
ceiling at this scale.

## 4. But AWS throttles, and that ceiling IS shared

This went undetected through every run until pushed into:

| config | in flight | result |
|---|---|---|
| 6 workers × 16 threads | 96 | passed, repeatedly |
| 12 workers × 16 threads | 192 | **56 of 1,590 blocks failed** — `SlowDown: Please reduce your request rate` |
| 12 workers × 8 threads | 96 | 364.5 s, 0 failures |
| 12 workers × 8 threads (corpus) | 96 | 6 of 1,590 failed on date 2; SlowDown counts 40/36/30 across three workers |

So 96 is not safe, only *safer*. Throttling begins **below** 96 and is
intermittent. Every failure carried `aws_request_id`; **zero** Ceph errors, so
this is entirely the AWS read side.

### Why this breaks the VM-count answer

Per-VM throughput scales. Total request rate against `s3://ecmwf-forecasts`
does not — it is one shared quota, and it is already pushing back at 96
concurrent readers on six VMs. **Adding VMs raises the request rate against
that same quota.** 20 VMs would be ~320 concurrent readers, well past where
192 already failed.

**1 GB/s from EWC is therefore not reachable by adding VMs.** The earlier
answers of 36 VMs (`610fa42`) and then ~20 VMs both assumed the only ceiling
was per-VM bandwidth. Neither accounted for a shared rate limit.

`BLOCKERS.md` §10 — reading in `eu-central-1` — is not an optimisation. It is
the only route to sustained high throughput, and it is what the ~100 Gbps
reports are doing.

---

## 5. Settled configuration

```bash
deploy_workers.py --scheduler-on 192.168.1.74 --start \
    --workers-per-vm 2 --nthreads 8 --memory-limit 6GB
```

- **2 workers/VM** — captures the per-process gain (§2)
- **8 threads each = 96 in flight** — the most that has ever completed a date
- **scheduler on a worker VM** — it was OOM-killed on the 8 GiB gateway
- **retry at the block level**, 4 rounds, 45 s pause — see below

Measured: **317.8 s/date, 255.2 msg/s**, against 494.6 s before repacking.

### Retry must be at the block level

A read that exhausts its 8 attempts fails one block. Failing the *date* for
that discards 81,090 messages and ~64 GB of egress to avoid redoing ~300.
`build_corpus.py` resubmits failed blocks over 4 rounds with a 45 s pause —
the pause matters, because resubmitting into an active throttling window just
burns the retries again.

### The backoff, and how long it went untested

`read_message` retries `SlowDown`. It was written early and **never once fired**
until 192 concurrent readers provoked AWS. Two things were then wrong with it:

- `_open_era` sat *outside* the retry. It fetches the manifest — an S3 call
  like any other — and 12 worker processes open the store simultaneously.
- 6 attempts at 0.5–16 s is ~31 s total, which a sustained throttle outruns,
  and with no jitter every throttled reader retried in lockstep.

Now 8 attempts capped at 30 s with full jitter, covering the open.

**A retry path that has never executed is not a working retry path.** The
`retried` counter in `build_corpus.py` exists so throttling pressure is visible
rather than silent.

---

## 6. On CPU and memory being "underused"

At 6% CPU and 2–4% RAM the cluster looks idle. It is not, and this is not
waste: `frisky observe` measures `worker.exec.gil` at **7.2 s against 3,632 s**
of execution. The workers are blocked on S3 essentially the whole time.

Low CPU is the *correct* state for this workload, and raising it is not a
goal — the only number worth moving is MB/s. What moved it was **process
count** (§2), not threads, memory, or worker size. Neither Frisky nor Dask can
see icechunk's Rust allocations either way: `managed` reads 0 B against 2.17 GiB
resident, which is why the bounded task shape matters more than any memory
manager.
