# Realization DAG: versions, and which one produced the corpus

The realization has run on **one** DAG shape throughout. This file tags it, so
that any future change has something explicit to be a change *from*, and so the
243-hour record is attributable to a named design rather than to "the pipeline".

| version | status | produced | diagram |
|---|---|---|---|
| **v1** | **current, validated** | 1,260 dates / 243.0 cluster-hours / 16 seasons | [`dag/realize_dag_v1.drawio`](dag/realize_dag_v1.drawio) |
| **v2** | **proposed** — warm the read path, §4 | — | [`dag/realize_dag_v2_proposed.drawio`](dag/realize_dag_v2_proposed.drawio) |

---

## 0. v0: the Dask attempt (2026-07-31 → 08-03), and why it was abandoned

Before frisky, the realization was tried as a Dask collection graph on the
same six VMs. It hung. The records of that period are not kept in this
folder; the findings that still matter are below.

**Root cause (`e77f2aa`): graph construction, not the cluster.**
`xr.open_zarr(..., chunks={})` makes every array a dask array, so the graph is
built over **every chunk in the store** before `.sel()` narrows it:

| array | store chunks | tasks for a 2-chunk read | result |
|---|---|---|---|
| `t2m` | 221,085 | 665,966 | built in 1.4 s |
| `u` (14 levels) | 3,097,290 | ~9.3 M | **never finishes** |

The hang is **in the client**, building the graph while holding the GIL. The
dashboard is empty, there is no traceback, and the distributed and threaded
schedulers fail identically.

- **Fix:** open without `chunks`, `isel(isobaricInhPa=…)` first, then
  `.sel()`, then `.chunk({"number": 1, "step": 1})`. With it, `u` passes at
  1,329 MB and `t2m` at 637 MB.
- Select-first still builds 3.76 M tasks for `u` (in 2 s), so **graph size
  tracks the store, not the request**.
- **Rule:** measure the graph first (`store_chunks ≈ prod(ceil(shape/chunk))`).
  Under 10 k tasks is fine; 100 k–1 M is fragile; over 1 M expect trouble;
  over ~5 M expect a hang.

**Secondary failures.** These were consequences of the eager pattern, not
separate causes:

- `.compute()` inside a task gave ~7 GB RSS per array.
- Dask's `managed` memory read 0.00 GB against 33.6 GB resident, because the
  memory sits in the Rust icechunk/gribberish layer. Dask therefore
  over-subscribed `nthreads=4` workers (memory limit 13.94 GB).
- The nanny killed those workers and left 236 orphaned tasks (`processing 7,
  waiting 140, executing 0`). Only `sudo systemctl restart dask-scheduler`
  cleared them; `client.restart()` raised `assert not self.tasks`.
- The JupyterHub client cgroup is 8 GiB. Client-side runs were SIGKILLed (137)
  at channel 11 (`u925`).

**Other anti-patterns found:**

- `workers=[w], allow_other_workers=False`: a dead worker's tasks never
  reschedule.
- Module-level caches in cloudpickled functions never hit.
- `pure=True` on I/O tasks.
- `AWS_*` popped without restore (and `finally` does not run on a kill).

**Icechunk 2.1.1 constraints.** `to_zarr` is unsupported with the distributed
and multiprocessing schedulers. Distributed writes need `fork → write →
merge → commit`. Distributed reads are undocumented.

**The decision.** The work is embarrassingly parallel per (date, channel). So
v1 uses one session per process, a `ForkSession` pickled to the workers, and
futures instead of a collection graph (§1).

**Claims made and withdrawn during that period** (so they are not re-argued):

1. "89.5 GB of manifest RAM, 96% of the archive unreachable". Wrong: the store
   is already split one manifest per date (`splitting: time=1`); 46,154
   objects, the largest 1.1 MB.
2. "The 503 is self-inflicted". Wrong: one GET from a clean process also got
   503, because the throttle is on the shared egress IP
   ([`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) §2).
3. "A 0.74–14.5 MB/s link ceiling". Measured while throttled; not a ceiling.

---

## 1. What v1 is

Defined in `frisky_daily_dag.py`, submitted by `build_corpus.py`. It is a
**Frisky futures DAG**, not a Dask collection graph, and the distinction is the
reason it works at this scale.

```
for channel in 38:
    for step in 65:
        read_message(ch, step, member=0)  ─┐
        read_message(ch, step, member=1)  ─┤
        ...  49 more  ...                  ├─> stack_members(*51) ─> write_block
        read_message(ch, step, member=50) ─┘
```

| | |
|---|---|
| unit of work | **one GRIB message → one 321×261 float32 slab** |
| leaves per date | 38 × 51 × 65 = **125,970** |
| fan-ins per date | 38 × 65 = **2,470** blocks |
| chunk written | `(1, 1, 51, 321, 261)` = 17.1 MB |
| fan-in kind | **stack, not reduce** — every member is kept, including control |
| write path | `ForkSession` pickled to each worker; one merge, one commit per date |

### Why Frisky and not Dask

Three properties of this workload make the choice structural rather than
preferential:

**Task count.** 125,970 leaves per date. Dask pays roughly 1 ms of scheduler
overhead per task, so ~97 s per date of pure bookkeeping — about 13% of an
800-second date. Frisky pays ~3 µs, so ~0.25 s.

**No nested event loop.** `frisky_daily_dag.py` §4.3 forbids `.compute()` inside
a task; leaves read numpy directly. A zarr `sync()` inside a worker thread,
which a Dask collection graph invites, deadlocks against icechunk's own async
runtime.

**Fork/merge writes.** Icechunk needs a session per writer and one merge. The
`ForkSession` pickles to workers, so each writes independently and the parent
merges once — there is no Dask equivalent that does not serialise the writes.

### The ordering is load-bearing

Channel-major, then step, then member. Every concurrent reader therefore works
inside the **same ~6.5 GB step object**, because one GRIB object per step holds
all 51 members and every variable.

This looks like a hot key and is not. Measured 2026-10-04 at 200 readers, same
message count, same bytes (`REALIZATION_CONCURRENCY.md` §4b):

| spread | msg/s |
|---|---|
| **concentrated, 1 object** | **175** |
| spread over 85 objects | 152 |

**Concentration is 15% faster.** Do not "fix" this.

---

## 2. What v1 has earned

The case for treating v1 as settled is the volume it has survived, not its
design:

| | |
|---|---|
| dates | **1,260** across 16 seasons, four seasons × three years plus 0p4 2023 |
| cluster-hours | **243.0** |
| messages | **~156 million** |
| retries | **934** — 5.9 per million |
| dates lost | **0** |
| dates needing `--repair` | 4 (0.3%), 1–36 blocks each, all filled in minutes |
| eras | `0p4` (0.4°, 36 ch), `49r1`, `50r1` (0.25°, 38–39 ch) |
| grids | 201×164 and 321×261 — same DAG, no change |

It also survived, without data loss: a credential rotation mid-season, a
worker-VM reboot that changed every DHCP address and wiped `/tmp`, a transient
Ceph auth outage, and upstream gaps (six dates absent from `0p4`, one missing
`+186 h` GRIB).

**The four `--repair` dates are the one thing to keep in view.** They committed
with holes that every date-level check passed — `ds.sizes["time"]` was correct,
all dates present. Only counting chunks against `channels × dates × steps`
exposed them. That is a property of v1's commit-per-date design and the reason
chunk-level verification is not optional.

---

## 3. Where the time actually goes

`REALIZATION_CONCURRENCY.md` §4b, measured on v1's own workload:

| readers in flight | msg/s | per reader |
|---|---|---|
| 48 (what ran) | 166 | 3.46 |
| 100 | 143 | 1.43 |
| 200 | 207 | 1.03 |
| 400 | 227 | 0.57 |

The corpus ran at **48 readers because a deploy command omitted `--nthreads`**,
taking the default of 4 instead of the runbook's 8. That is a deployment
parameter, **not a DAG property** — which is the whole point of this section.

Reaching ~200 readers is worth about **1.8×**: 243 hours → **~135**. It needs
one flag and no change to v1.

---

## 4. v2, proposed: warm the read path

### What was refuted first

Two candidate redesigns were examined and **both are refuted by measurement**.
They are kept here because each looked compelling and the next person will
think of them too:

**Interleaving reads across step-files**, to avoid a presumed S3 hot key — 175
msg/s concentrated on one object against 152 spread over 85, so v1's locality is
worth **15%** and the "fix" is a regression.

**Sharding to dodge the AWS request quota** — the quota did not reproduce. Zero
errors at 100, 200 and 400 concurrent readers on 2026-10-04, where `SCALING.md`
§4 reports 192 readers failing with `SlowDown` in August.

### What the measurement actually found

The read path **warms up**, and v1 never benefits from it.

Five passes over an identical 1,800-message set, one process, 200 readers,
rates from successful reads only (2026-10-05):

| pass | msg/s | GRIB MB/s |
|---|---|---|
| 1 | 140 | 110 |
| 2 | 266 | 210 |
| 3 | 329 | 260 |
| 4 | 346 | 273 |
| 5 | **359** | **283** |

**2.56x between cold and warm, zero failures throughout.**

The obvious objection is that re-reading identical bytes could simply be hitting
an S3 cache, which would be worthless to a build that reads each message once.
Tested by alternating two **disjoint** message sets over the same 38 channels:

| pass | set | msg/s |
|---|---|---|
| 1 | X, cold | 196 |
| 2 | **Y — byte ranges never fetched** | **272** |
| 3 | X again | 296 |
| 4 | Y again | 338 |

**Pass 2 is 39% faster on bytes it has never seen.** The warmup is therefore a
property of the *process*, not of the data, and it generalises to the
read-each-message-once case that a build is permanently in.

It also resolves a contradiction `REALIZATION_CONCURRENCY.md` §4b could not:
that the curve asymptoted at 36 MB/s per VM, *below* the ~50 MB/s `procs_probe`
measures. There was no mystery ceiling — §4b was measuring cold reads against a
warm benchmark. Warm steady state is **57 MB/s per VM**, which exceeds it.

### The proposal

**Warm the read path, and keep it warm.** The DAG's shape, leaf unit, fan-in and
write path are unchanged — which is why this is v2 and not a rewrite.

The specific suspect is in our own configuration. Every call site disables
manifest preload:

```python
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))
```

That was a deliberate fix — the default open prefetches thousands of manifests
in parallel and source.coop returns sporadic 500s, so one failure aborted the
whole open. The cost of that fix is that every chunk reference is then resolved
lazily, one at a time, mid-read. A **bounded** preload — enough to cover the
working set of one date, not thousands of refs at once — is the first thing to
try.

| | |
|---|---|
| available from warmth alone, fixed concurrency | 196 -> 338 msg/s = **1.7x** |
| upper bound if cold reads reached the warm plateau | 243 h -> **~95 h** |
| realistic | less; see the caveat below |

### The caveat, stated plainly

**Manifest resolution and TCP/TLS connection reuse were not separated.** Both
warm up per-process and both would produce this curve. The fix differs:

- if it is **manifests** -> bounded `ManifestPreloadConfig`, possibly a shared
  cache across the 51 members of a (channel, step) group
- if it is **connections** -> raise the per-process pool and hold keep-alives;
  nothing to do with icechunk at all

An attempt to separate them by comparing raw `boto3` range GETs against the
icechunk path produced unusable numbers — rates computed from task count rather
than successes, and msg/s compared across variants whose messages were
different sizes. **Separate them before building anything**, because the two
answers have no implementation in common.

### What this retires

Four candidate ceilings are now measured and ruled out: the AWS request quota,
per-object hot-keying, the shared EWC path, and per-VM bandwidth. **Adding
machines would not have helped** — the bottleneck was never throughput. It was
that each of ~156 million reads paid a setup cost that 51 of them could have
shared.

## 5. Tagging convention

- **v1** = the DAG in `frisky_daily_dag.py` as of commit `3db93fe`, diagram
  `dag/realize_dag_v1.drawio`. Any change to the leaf unit, the fan-in shape,
  the submission order or the write path is a new version.
- Changes to **concurrency, worker count, thread count, retry budget or
  scheduler address are not versions** — they are deployment parameters, and
  conflating the two is how 48 readers came to look like a property of the
  design rather than a flag that was left off.
