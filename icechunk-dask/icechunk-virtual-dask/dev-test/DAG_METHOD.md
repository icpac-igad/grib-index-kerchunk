# Method: attacking a Dask + Icechunk pipeline that hangs

**Given an Icechunk store and a Dask cluster, in what order do you make a read
work?** Written after spending a session attacking it in the wrong order.

The short version: **measure the graph before you run anything.** Every other
diagnostic we ran — worker RSS, 503 rates, egress addresses, scheduler
states — was downstream of one number nobody had looked at.

Tool: [`graph_size.py`](graph_size.py). Evidence: [`BLOCKERS.md`](BLOCKERS.md) §0.

---

## 1. Why start with the graph

A Dask read has three phases, and they fail differently:

| phase | runs on | failure looks like |
|---|---|---|
| **1. graph construction** | **client**, pure Python, holding the GIL | process frozen, **dashboard empty**, no traceback, watchdogs cannot fire |
| 2. scheduling | scheduler | tasks stuck in `processing`, `executing == 0` |
| 3. execution | workers | tasks `erred`, worker RSS climbs, nanny kills |

**Phase 1 failures are the ones that waste days**, because every symptom points
at phases 2 and 3. An empty dashboard reads as "the cluster is broken" when the
truth is that nothing was ever submitted to it.

So: measure phase 1 first. It costs seconds and needs no cluster.

```bash
P=/opt/mamba/envs/dask/bin/python
$P graph_size.py --vars t2m u --members 2 --steps 1
```

---

## 2. What we measured on this store

```
variable  pattern                   store chunks    raw tasks   build   verdict
t2m       chunks={} then .sel()          221,085      665,966   1.87s   fragile
t2m       .sel() then .chunk()           221,085      665,966   1.34s   fragile
u         chunks={} then .sel()                   (no result)   >100s   **HANG**
u         .sel() then .chunk()         3,095,190    3,761,156   1.96s   expect a hang
```

Read that carefully, because it is not the tidy story:

- `chunks={}` on the 6-D `u` array **never finishes constructing**. That is the
  hang, and it is the thing to avoid.
- Selecting first and chunking after **builds in 2 s** — but it still produces
  a **3.76 million task** graph. It is tractable, not small.
- On the 5-D `t2m` both patterns are identical, because there is no extra
  dimension to blow up.

**So "select first" fixes the hang. It does not give you a small graph.** That
distinction is worth keeping: the ladder in §3 still applies, and §6 records
what remains unsolved.

---

## 3. The ladder — one rung at a time

Do not start at the top. Each rung is a real test with a clear pass condition,
and a failure tells you exactly which layer is at fault.

### Rung 0 — does the store read at all? *(no dask, no cluster)*

```bash
$P quick-run.py
```

Pass: a field decodes with sensible values. On this store, 0.2–2 s per global
message. **If this fails, nothing above it can work** — it is credentials,
network or the store, not Dask.

### Rung 1 — how big is the graph? *(no cluster)*

```bash
$P graph_size.py --vars <the ones you actually want>
```

Pass: every variable builds a graph without hanging. Thresholds on the **raw**
count, since that is what the client constructs:

| raw tasks | meaning |
|---|---|
| < 10k | fine |
| 10k–100k | slow build, works |
| 100k–1M | seconds to minutes of pure graph building, fragile |
| **> 1M** | **expect trouble; > ~5M expect a hang** |

**Do not proceed while a variable hangs here.** No scheduler change, worker
size or retry policy will help — it never reaches them.

### Rung 2 — compute it in one process, threaded

```bash
$P test_single_date.py --cluster threads --threads 8 --vars t2m --members 4 --steps 2
```

One process, one session, nothing pickled, no scheduler, no nannies. Icechunk's
own docs say the multithreaded scheduler is the one that works
([`ICECHUNK_DASK_GUIDANCE.md`](ICECHUNK_DASK_GUIDANCE.md)).

Pass: correct values, bounded peak RSS. Measured here: 637 MB for `t2m`,
1329 MB for `u`.

**If rung 1 passes and rung 2 hangs**, the problem is execution, not graph
construction — go to §4.

### Rung 3 — ramp the variable count, still threaded

```bash
$P test_single_date.py --cluster threads --threads 8 --ramp
```

Pass: RSS grows roughly linearly. Measured: 674 MB → 1152 MB → 2018 MB for
1, 2, 4 variables — about 340 MB each. Extrapolate before you scale: 30
channels ≈ 10 GB, which needs a **worker VM (16.77 GB)**, not the JupyterHub
session (8 GiB cgroup).

### Rung 4 — one process per shard

Only now introduce parallelism across processes, and do it with **independent
processes that each hold one session** — not a shared scheduler. The workload
is embarrassingly parallel: every `(date, channel)` is independent, no shuffle,
no cross-chunk dependencies.

### Rung 5 — `dask.distributed`, if you still want it

Treat as unsupported territory (the icechunk docs cover distributed *writes*
only, and say the distributed scheduler does not work with the standard zarr
write path). Constrain it hard: `nthreads=1`, open the session **on the
worker**, never `.compute()` inside a task, explicit 503 retry.

---

## 4. If rung 2 passes but the cluster still hangs

Then it is genuinely a scheduling or execution problem. In order:

1. **`stop_work.py status`** — is the dashboard showing *your* run, or stale
   wreckage? Ours sat at a frozen `236 tasks {processing: 7, waiting: 140}` for
   hours, identical between checks. Orphaned tasks from dead clients cannot be
   released from the client and need a scheduler service restart.
2. **`cluster_status.py`** — compare `rss` against `managed`. If `managed` is
   ~0 while RSS is GBs, Dask is scheduling blind: the memory lives in Rust
   extensions it cannot see, so spill/pause/kill never fire and it
   over-subscribes until the nanny kills workers. Then `nthreads=1`.
3. **Check for `.compute()` inside a submitted task** — nests a synchronous
   asyncio wait inside a worker thread.
4. **Check pinning** — `workers=[w], allow_other_workers=False` means a dead
   worker's tasks never reschedule.

[`FABLE_DAG_ANALYSIS_NOTE.md`](FABLE_DAG_ANALYSIS_NOTE.md) is the full
anti-pattern list, ordered by severity.

---

## 5. Patterns that blow up a graph

### `chunks={}` on a large array, selection afterwards

```python
ds = xr.open_zarr(store, chunks={})     # WHOLE array becomes dask
da = ds["u"].sel(time=..., number=[...])  # graph already built over everything
```

The graph covers every chunk in the store before the selection narrows it.
Instead:

```python
ds = xr.open_zarr(store)                 # lazily-indexed xarray, NOT dask
da = ds["u"]
da = da.isel(isobaricInhPa=0)            # extra dims FIRST
da = da.sel(**sel)                       # metadata only, no graph yet
da = da.chunk({"number": 1, "step": 1})  # dask over the subset
```

### Selecting the extra dimension last

`ds["u"].sel(...).isel(isobaricInhPa=0)` builds across all 14 levels, then
throws 13 away. Reversing costs nothing and removes a 14× factor.

### Estimating before you run

```
store_chunks ≈ prod(ceil(shape[i] / chunk[i]) for i in dims)
```

For this store: `t2m` is 51×51×85 = 221,085 chunks; `u` adds ×14 = 3,095,190.
`graph_size.py` prints both the shape and the chunk shape so you can do this by
eye for any new variable.

---

## 6. What is still unsolved

**The select-first pattern builds a 3.76 M-task graph for `u`.** It builds in
2 s and computes correctly, because Dask culls unreachable tasks before
scheduling — but the client is constructing millions of tasks and discarding
nearly all of them. That is wasted work, it scales with the store rather than
with the request, and it will get worse as the `50r1` era grows.

Worth trying, in order:

1. **Positional indexing** — `.isel()` with integer positions instead of
   `.sel()` with label lists, which may let xarray narrow before dask sees it.
2. **Index the zarr array directly** and wrap the result, bypassing xarray's
   lazy-index machinery entirely.
3. **`--cull`** in `graph_size.py` reports the post-`dask.optimize` count, so
   you can see how much is being thrown away. It is slow — optimizing a
   multi-million-task graph can itself exceed the timeout — which is itself a
   finding.

A graph whose size tracks the *request* rather than the *store* is the goal. We
are not there; we have only made it not hang.
