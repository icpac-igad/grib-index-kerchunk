# Blockers: why the ECMWF → EWC realization does not scale

**The cross-cloud read works. It just does not work more than one request at a
time, and the job needs roughly a hundred thousand times that.**

_Written 2026-08-03, against the live EWC Dask cluster (Bonn, non-AWS — 6
worker VMs, 4 vCPU / 16.77 GB each, Dask `memory_limit` 13.94 GB) reading the
published ECMWF IFS ensemble Icechunk store whose virtual chunks live on AWS
`s3://ecmwf-forecasts` in `eu-central-1`._

---

## 1. Summary

| | status |
|---|---|
| Opening the published store from EWC | ✅ works, 5.2 s |
| Reading a single field cross-cloud | ✅ works, 3.5–6 s per global message |
| Writing a realized Icechunk store to EWC S3 | ✅ works, 8/8 correctness checks |
| EWC object-store credentials | ✅ work from client and all 6 workers |
| **Reading anything at all, right now** | ❌ **AWS returns 503 SlowDown to a SINGLE request** — not rate-related |
| **Reading many fields sequentially on a worker** | ❌ **~7 GB RSS per array, never released** |
| **Dask seeing that memory** | ❌ **`managed` reports 0.00 GB of 33.6 GB — it schedules blind** |
| **Running the whole thing on the notebook instead** | ❌ **8 GiB cgroup cap, SIGKILL** |
| Net result of the last full attempt | **1 of 30 channels in ~9 minutes** |

**The link is not the blocker.** Everything that reads *one thing at a time*
succeeds. Every failure appears only when we try to go faster.

---

## 2. What demonstrably works — the notebook

`OPENING_PUBLISHED_ECMWF_ICECHUNK.ipynb` runs clean from this EWC session,
against the AWS-hosted chunks, today:

| cell | operation | result |
|---|---|---|
| open repo | source.coop metadata | **5.2 s** |
| `t2m` @ 2026-07-02, member 0, f000 | one global 721×1440 message | **3.54 s**, finite 1.000, 199.2 / 281.5 / 313.1 K |
| `gh` @ 500 hPa | one global message | **5.25 s**, 4579 / 5561 / 5998 m |
| 14-level `t` profile | 14 messages | **6.0 s** → 2.33 msg/s |

Physically sensible values, full coverage, no errors. The anonymous
cross-cloud virtual-chunk path from Bonn to `eu-central-1` is sound.

This is the thing to hold onto: **there is no fundamental connectivity or
credential problem.** Which is precisely why the failure mode is confusing —
it only appears under load.

---

## 3. The scaling wall, in one table

The notebook's best sustained rate is **2.33 messages/s**. The job needs:

| workload | messages | at 2.33 msg/s (notebook rate) | at 21.6 msg/s (best batch ever measured) |
|---|---:|---:|---:|
| June 2026, 30 dates | 2.43 M | **12.1 days** | 1.3 days |
| Full corpus, 1,246 dates | 91.5 M | **454 days** | 49 days |

Single-stream is not a slow option, it is not an option. The job is only
feasible with **one to two orders of magnitude of concurrency** — and
concurrency is exactly what breaks. Everything below is about that.

---

## 4. Blocker 1 — AWS returns 503 SlowDown, and it is not our request rate

```
|-> error fetching virtual reference
|-> service error
|-> unhandled error (SlowDown)
`-> Error { code: "SlowDown", message: "Please reduce your request rate." }
```

### The measurement that settles the cause

An earlier version of this document said the throttle was **self-inflicted**,
caused by our own hammering during investigation, and would decay if we backed
off. **That attribution was wrong.**

A fresh Python process on the Jupyter host — no Dask, no icechunk, not a
worker — issuing **one** 1 MB range GET, three times, 3 s apart:

```
jupyter client #1        HTTP 503  SlowDown
jupyter client #2        HTTP 503  SlowDown
jupyter client #3        HTTP 503  SlowDown
```

And via the working lazy Dask pattern on a LocalCluster, **8 chunks total**:

```
vars  chunks    wall  chunk/s   peakRSS alive  finite  status
   1       8   31.7s      0.3      205M     3     -    IcechunkError (SlowDown)
```

**One request cannot be a request-rate problem.** We had also backed off
substantially by this point, and it had not decayed.

### What actually fits the evidence

The throttle is applied to the **egress IP**, which on EWC is very likely a
shared NAT address for the tenancy. Our requests compete with everything else
leaving that address, and AWS rate-limits the address, not our process.

It is **intermittent, not absolute**, which is exactly what a contended shared
egress looks like:

- the notebook read global fields fine earlier the same day (§2)
- two of six workers got 5.0 and 5.2 MB/s minutes before the single-request
  test failed
- the other four got 503 at the same instant

**This is an EWC ↔ AWS infrastructure question, not something to engineer
around.** The concrete thing to ask the EWC team: *does the tenancy share an
egress IP to the internet, and is that address being rate-limited by AWS S3?*

### icechunk makes it worse than it needs to be

icechunk surfaces 503 as `unhandled error (SlowDown)` — no internal backoff,
no retry. A single 503 anywhere in a read fails the entire read.

Retry with exponential backoff and jitter (implemented in
`realize_smoke_test.py`) does help when the throttle is partial — the one
channel that completed shows `503s 1`. But retrying cannot fix a
requester-level block, and retrying harder is precisely the wrong response.

### Consequence for every throughput number in these documents

**All of our rate measurements were taken while throttled.** The
"0.74–14.5 MB/s" figures, and everything derived from them in
`EWC_USAGE_AND_RESOURCE_PLAN.md` §5.4, are a floor on a rate-limited requester
and say nothing reliable about what the link can do. Until a clean measurement
is possible, the schedule estimates are unfounded in both directions.

---

## 5. Blocker 2 — worker memory accumulates per source array and is never released

This is the more damaging one, because it caps concurrency *before* AWS does.

Measured RSS on a worker after reading **one channel**, 2 members × 2 steps
(4 GRIB messages, ~3 MB of useful data):

| situation | worker RSS |
|---|---:|
| `skt` on a freshly restarted worker | **0.75 GB** |
| `skt` on a worker that had already read other channels | **6.87 GB** |

And with **nothing running at all**, after the run was stopped:

```
dask-worker-01    7.52 GB / 13.94 GB
dask-worker-05    7.39 GB / 13.94 GB
dask-worker-06    7.32 GB / 13.94 GB
dask-worker-03    7.30 GB / 13.94 GB
dask-worker-02    2.75 GB / 13.94 GB
dask-worker-04    2.51 GB / 13.94 GB
                 ---------------------
     34.8 GB held across the cluster, idle
```

### The mechanism: Dask cannot see this memory at all

`cluster_status.py` on an idle cluster, minutes after everything stopped:

```
worker           rss      managed   limit    use   exec
dask-worker-01   7.38G    0.00G    13.9G    53%     0
dask-worker-05   7.28G    0.00G    13.9G    52%     0
dask-worker-06   7.17G    0.00G    13.9G    51%     0
dask-worker-03   7.16G    0.00G    13.9G    51%     0
dask-worker-02   2.44G    0.00G    13.9G    18%     0
dask-worker-04   2.17G    0.00G    13.9G    16%     0
TOTAL           33.60G    0.00G
```

**`managed` is 0.00 GB on every worker against 33.6 GB resident.** Dask's
`managed_bytes` counts data it holds references to in the Python heap. This
workload's memory is allocated inside the Rust icechunk / object-store layer
and the gribberish decoder — outside anything Dask accounts for.

That single fact explains the whole failure cascade:

1. Dask inspects a worker holding 7.4 GB and sees *0 bytes managed, idle*.
2. It therefore assigns up to `nthreads` = **4** concurrent tasks.
3. Each adds ~7 GB of memory Dask cannot see.
4. The worker passes `memory_limit`, the nanny kills it.
5. The client gets `scheduler-connection-lost` and every in-flight future dies.

This is not a Dask bug and not a misconfiguration. **Dask's memory model has
no visibility into this workload**, so its scheduling decisions are made on
information that is simply wrong. Spill-at-70% / pause-at-80% never fire,
because as far as Dask is concerned the worker is empty.

### What this means operationally

- A worker has 13.94 GB and **4 threads**. At ~7 GB per array touched it can
  safely run **one** — not four.
- **Concurrency must be capped externally.** Dask will never self-limit here,
  so the caller has to enforce one array-read per worker. `nthreads=4` is
  actively harmful for this workload; a cluster of 1-thread workers would be
  safer.
- Memory is **not reclaimed** between tasks — the numbers above are from an
  idle cluster with `exec 0` on every worker. Even strictly serial reads walk
  a worker into the wall after a handful of channels, so workers must be
  **restarted between waves** (`fix_worker_credentials.py restart`).
- This also explains why dropping to `--max-workers 2` did not rescue the last
  run: those workers were already holding ~7 GB from earlier, so even one task
  each started near the ceiling.

### Relationship to the withdrawn manifest claim

An earlier document claimed a `49r1` pressure-level manifest needs 89.5 GB and
that 96 % of the archive was unreachable. **That specific claim stays
withdrawn** — the store is already split one manifest shard per date, 46,154
objects, largest 1.1 MB.

But the underlying *observation* was real and has now appeared three times
independently: **touching a source array costs several GB of resident memory
that is never returned.** What was wrong was the cause and the magnitude, not
the existence. It is a flat per-array cost, not something that scales with era
length, and it is the binding limit on concurrency.

**Cause not yet established.** It is not the manifest (too small). Candidates:
the gribberish decoder, the zarr/icechunk chunk cache, or an allocator that
does not return freed pages. This is the single most valuable thing left to
diagnose.

---

## 6. Blocker 3 — the notebook cannot be the workaround

The obvious response to "the workers are unstable" is "run it in the notebook,
that works". It does not scale, for a hard reason:

```
/sys/fs/cgroup/system.slice/jupyter-nishadhka.service/memory.max
    = 8589934592   (8 GiB)
```

The JupyterHub session is capped at **8 GiB**. Two client-side runs were
**SIGKILLed (exit 137)**, both at channel 11 — `u925`, the first
pressure-level channel. `free` reports 32 GB because that is the *host*; the
cgroup is what kills you.

So: the notebook works for one field at a time and dies on the eleventh. It
is a good demonstration and a bad execution environment.

---

## 7. Where it actually fails, in sequence

```
  notebook, 1 field at a time        ->  WORKS      3.5-6 s per message
  client, 30 channels sequential     ->  SIGKILL    at channel 11, 8 GiB cap
  6 workers, 30 channels concurrent  ->  OOM        ~7 GB x 4 threads > 13.94 GB
                                                    nanny kills, scheduler drops
  6 workers, 1 channel each (waves)  ->  503        AWS SlowDown, icechunk
                                                    does not retry
  2 workers, waves, with backoff     ->  1/30       throttled + workers already
                                                    holding 7 GB from before
```

Every step that increases parallelism hits a different wall. That is why "the
notebook works" and "the job does not" are both true.

---

## 8. What we do not know

Stated plainly, because these gate any commitment:

1. **What causes the ~7 GB per-array RSS**, and whether it can be released.
   Everything else is downstream of this. We now know *where* it is not — it is
   not in the Python heap, since Dask reports 0.00 GB managed — which points at
   the Rust object-store layer, the gribberish decoder, or an allocator not
   returning freed pages. A first cheap test: `MALLOC_TRIM_THRESHOLD_`/
   `malloc_trim`, and reading two different arrays in one process to see
   whether the cost is per-array or one-off.
2. **Why the egress address is being throttled**, whether it is shared across
   the EWC tenancy, and what rate it would tolerate if it were not. All our
   rate measurements were taken while throttled, so none of them are
   trustworthy. This is a question for the EWC operators, not a tuning
   exercise.
3. **Whether `49r1` and `0p4` pressure-level reads work end to end.** Never
   demonstrated — every attempt coincided with throttling or a worker death.
4. **The compression ratio of the realized store.** Still the assumed 40 %;
   the calibration run has never completed.

---

## 9. How to check the state

Three scripts, all in this directory:

| script | question it answers | cost |
|---|---|---|
| `cluster_status.py` | what state are the workers in *right now* — rss vs managed, headroom, credentials, what is executing | ~2 s, no side effects |
| `fix_worker_credentials.py` | `check` / `restore` / `restart` — repairs workers left stripped or holding leaked memory | seconds; `restart` kills running work |
| `realize_smoke_test.py` | end-to-end: published store → EA subset → realized Icechunk store on EWC | minutes; `--dask` to use workers, `--max-workers` to throttle |
| `test_single_date.py` | single date, ramps the variable count, on the EWC **or** a local cluster. Built on the read pattern that already passes in `grib-index-kerchunk`. `--eager` reproduces the bad pattern for comparison | seconds to minutes |
| `stop_work.py` | `status` / `cancel` / `restart` — stop work and reclaim worker memory. `client.restart()` is unreliable here | seconds |

`../../test-icechunk-long.py` is a **workload** test, not a status check — it
writes 2 GB of synthetic data and exercises worker-to-worker shuffle. Useful
for proving the EWC half works (it never touches AWS), but it cannot tell you
worker state and costs minutes.

---

## 10. Options

**A. Run the extraction inside AWS `eu-central-1`, ship ~4.4 TB back.**
Addresses blockers 1 and 2 simultaneously: in-region reads are not subject to
the same cross-region request penalties, and worker size can be chosen for the
~7 GB-per-array behaviour rather than fought against a 13.94 GB ceiling.
Inverts the transfer problem — 4.4 TB out instead of 60.6 TB in. **This is the
recommendation.**

**B. Stay on EWC and work around the memory behaviour.** One array per worker,
restart workers between waves, aggressive backoff, low concurrency. 30
channels per date becomes 5 waves plus 5 worker restarts. Viable for a proof
on a handful of dates; the 12-day June estimate assumes a concurrency we have
not yet sustained for more than one channel.

**C. Diagnose the RSS growth first.** If the ~7 GB is a cache that can be
bounded or an allocator artifact that can be released, option B becomes far
more attractive and the whole picture changes. Cheapest experiment of the
three, and it does not need the AWS link to be healthy.

Whatever the outcome, one change is worth making regardless: **run the workers
with `nthreads=1`**. Dask cannot see this memory, so the only safe
over-subscription factor is 1, and 4 threads per worker simply gives it four
ways to kill itself.

**D. Ask EWC about the egress address.** Since a single request from a clean
process is refused, backing off does not help and no amount of client-side
tuning will. If the tenancy shares a NAT address that AWS is rate-limiting,
the fix is operational — a different egress path, or a request to AWS — and it
gates options A and B alike.

**Immediate:** restart the workers to reclaim leaked memory
(`stop_work.py restart` — measured 33.3 GB → 0.52 GB). Do **not** plan around
"waiting for the throttle to decay"; that was based on the withdrawn
self-inflicted theory and is not supported by the single-request test.
