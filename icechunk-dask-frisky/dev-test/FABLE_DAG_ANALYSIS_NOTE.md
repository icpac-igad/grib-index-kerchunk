# Note for Fable — static analysis of Dask scripts for hangs

**Task: read a Python script, reconstruct the Dask task graph it builds, and
say where it will block or hang on our cluster. Static reading only — do not
run anything.**

Reference input: `test_single_date.py` in this directory. Others:
`realize_smoke_test.py`, `materialize_ea_icechunk_ewc.py`,
`where_does_it_fail.py`.

---

## 1. The cluster you are reasoning about

```
client   JupyterHub VM, cgroup memory.max = 8 GiB   (hard kill, SIGKILL/137)
scheduler tcp://127.0.0.1:8786
workers  6 VMs x 4 vCPU x 16.77 GB, Dask memory_limit 13.94 GB, nthreads 4
```

Three facts that drive most findings:

1. **The client is capped at 8 GiB.** Anything materialised client-side above
   that is killed outright, no traceback.
2. **Dask cannot see this workload's memory.** Reads go through icechunk and
   gribberish, both Rust extensions; `managed_bytes` reports **0.00 GB** while
   workers hold 7 GB. So Dask's spill/pause/kill thresholds never fire and it
   over-subscribes workers until the nanny kills them.
3. **`nthreads=4`.** Dask will run 4 tasks per worker concurrently unless the
   code prevents it.

---

## 2. The symptom we are chasing

```
scheduler tasks : 236  {processing: 7, erred: 39, released: 50, waiting: 140}
executing across cluster: 0
```

Scheduler thinks tasks are processing; no worker is executing anything; it
never recovers. Distinguish carefully:

| observation | means |
|---|---|
| `erred` grows, wall time short | a **fast failure** — exception, e.g. HTTP 503. Not a hang. |
| `processing > 0` and `executing == 0` | **a hang.** Worker died, or a task is blocked in a nested wait. |
| `waiting` large and static | downstream of a hung task. Symptom, not cause. |
| worker ports change between checks | workers **died and restarted**; their tasks are orphaned. |

Your job is to find the code that produces the middle two.

---

## 3. Procedure

**Step 1 — find the execution boundary.** For every function, decide: does it
run on the **client** or on a **worker**? A function runs on a worker iff it is
passed to `client.submit`, `client.map`, `client.run`, or appears inside a
`dask` graph. Everything else is client-side. Write the list out; most bugs
here are a function running somewhere the author did not intend.

**Step 2 — find graph construction.** Look for:
- `xr.open_zarr(..., chunks={})` or `chunks=...` → **lazy**, one task per chunk
- `xr.open_zarr(...)` with no `chunks` → **eager**, `.values`/`.compute()`
  pulls everything in the calling process
- `da.mean(dim)`, `.std(dim)`, arithmetic on dask arrays → adds graph layers
- `.persist()`, `.rechunk()` → materialisation / shuffle points

**Step 3 — find submission points.** `client.compute(...)`, `.compute()`,
`client.gather`, `client.submit`, `.result()`. For each, note whether it is
called on the client or inside a worker task (see Step 1).

**Step 4 — size one task.** For the heaviest task, compute what is live at its
peak: `n_chunks_it_touches × chunk_bytes`, plus decode buffers. Compare against
13.94 GB. Remember Dask does not see this memory.

**Step 5 — check failure propagation.** If a worker dies, can its tasks be
rescheduled? Look at `workers=[...]` pinning and `allow_other_workers`.

---

## 4. Anti-patterns to flag, highest severity first

### A0. `chunks={}` on a huge array, selection applied afterwards — **hang, in the client**

**This was the actual bug in this codebase. Check for it first.**

```python
ds = xr.open_zarr(store, chunks={})       # WHOLE array becomes dask
da = ds["u"].sel(time=..., number=[...], step=[...])   # graph already built
```

`chunks={}` makes the entire array a dask array, so the graph is constructed
over **every chunk in the store** before the selection narrows it. Measured
here, for a read of two chunks:

```
t2m    221,085 store chunks ->   665,966 graph tasks   1.4 s
u    3,097,290 store chunks -> ~9,300,000 graph tasks   never finishes
```

Signature: process appears frozen, **dashboard shows nothing at all** (no tasks
submitted), no traceback, and a watchdog thread cannot report because graph
building holds the GIL.

To size it: `prod(array.shape[:-2]) / prod(chunk_shape[:-2])` ≈ number of store
chunks; graph tasks are a small multiple of that. **Anything over ~10^6 is a
hang.** Multiply by 14 for a pressure-level variable (`isobaricInhPa`).

Correct form — select while still lazy xarray, chunk afterwards:

```python
da = ds[name]                            # opened WITHOUT chunks
if "isobaricInhPa" in da.dims:
    da = da.isel(isobaricInhPa=0)        # level first
da = da.sel(**sel)                       # metadata only
da = da.chunk({"number": 1, "step": 1})  # dask over the SUBSET
```

### A. `.compute()` / `.result()` inside a worker task — **hang**

```python
def read_channel(...):          # submitted with client.submit
    ...
    cube = da.compute()          # <-- nested synchronous execution
```

Nests a synchronous scheduler/asyncio wait inside a worker thread. With zarr v3
this also drives `zarr.core.sync.sync()` on a background event loop from
several worker threads at once. Classic deadlock shape. **This is the bug in
`realize_smoke_test.py`.**

Flag any `.compute()`, `.result()`, `.gather()`, `client.*` call reachable from
a function that is submitted to workers.

### B. Worker pinning with `allow_other_workers=False` — **permanent hang**

```python
client.submit(fn, ..., workers=[w], allow_other_workers=False)
```

If that worker dies, the task cannot be rescheduled. It sits in `processing`
forever and everything downstream sits in `waiting`. Given (2) above, workers
here *do* die. Flag every pinned submit and ask what happens if that worker is
killed.

### C. One task that internally reads N chunks — **invisible over-subscription**

```python
da.sel(number=range(51), step=all_53).compute()   # 2703 messages, ONE task
```

Dask sees one task and schedules 3 more alongside it. Actual concurrent memory
and request burst are `4 × N`. Flag any single task whose selection spans many
chunks.

### D. Client-side materialisation above 8 GiB — **SIGKILL, no traceback**

Any `.compute()`, `.values`, `.load()` on the client over a large selection.
Symptom is exit code 137 with no Python error.

### E. Mutating `os.environ` in a task without restoring — **breaks other jobs**

```python
for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)            # never restored
```

Worker processes are long-lived and shared. Also note: a `finally` restore does
**not** run if the task is cancelled or the worker is killed. Flag both the
missing restore and the assumption that `finally` always runs.

### F. Module-level cache in a submitted function — **silently ineffective**

```python
_SRC = {}
def read(...):
    if "ds" not in _SRC: _SRC["ds"] = open_store()   # never persists
```

Functions shipped by cloudpickle **by value** get a fresh module namespace per
task, so the cache never hits and the store is re-opened every time. Costs ~5 s
per task and hides as "slow". Flag `globals()` / module-dict caching in worker
functions.

### G. No retry on transient errors — **avoidable failure**

icechunk reports AWS throttling as `unhandled error (SlowDown)` and does **not**
retry internally. One 503 anywhere fails the whole read. Flag reads with no
backoff around them.

### H. `pure=True` (the default) on side-effecting functions

Dask deduplicates by key and may return a cached result instead of re-running.
Flag `client.submit(fn, ...)` without `pure=False` where `fn` does I/O or has
side effects.

### I. Blocking loop over `sync=True` submissions

```python
for item in items:
    client.compute(graph, sync=True)     # blocks; one hang stalls everything
```

Flag as a resilience issue: no timeout, no per-item recovery.

---

## 5. Output format

For each script, produce:

```
## <script>

### Graph
<one paragraph: what DAG this builds — how many tasks, of what shape,
 what the reduction structure is, lazy or eager>

### Execution boundary
  client-side : <functions>
  worker-side : <functions>

### Findings
| # | severity | pattern | location | why it hangs/blocks |
|---|----------|---------|----------|---------------------|
| 1 | HANG     | A       | line 157 | .compute() inside submitted task ... |

### Heaviest task
  <what is live at peak, vs the 13.94 GB worker limit>

### If a worker dies mid-run
  <can tasks reschedule? what ends up in `processing`/`waiting`?>
```

Severities: **HANG** (cluster stalls, needs restart) > **OOM** (worker or
client killed) > **FAIL** (task errors, recoverable) > **SLOW** (works, wasteful).

---

## 6. Calibration — what a correct answer looks like

For `test_single_date.py` **as committed at `e77f2aa` or later** (it now opens
without `chunks={}` and chunks after selection, so A0 is *absent* — if you
report A0 against the current file you have misread it; A0 applies to the
version before that commit, and to any other script still using `chunks={}`):

- **Lazy path** (`--eager` absent): `build_graph` returns xarray objects built
  from a `chunks={}` dataset; `client.compute([...], sync=True)` submits them
  as one graph. Tasks = `n_vars × members × steps` chunk reads plus reduction
  layers. Nothing pinned. No nested compute. **This path is expected to be
  sound** — flag only D (client holds the results) and I (no per-item timeout).
- **Eager path** (`--eager`): line ~157, `[da.compute() for ...]` runs on the
  **client**, not a worker — so it is **D (client OOM at 8 GiB)**, not A. Getting
  this distinction right is the test of whether you applied Step 1.
- `open_store()` is submitted to a worker at line ~234, and its return value —
  an `xr.Dataset` wrapping an icechunk store — is pulled back to the client.
  Flag as a **serialisation risk**: the script already has a `try/except`
  fallback for it, which tells you the author expected it to be fragile.
- `open_store()` mutates `os.environ` (pattern **E**) with no restore.

If your analysis of the eager path says "A, nested compute in a worker", it is
wrong — re-do Step 1.

---

## 7. Ground rules

- **Do not run anything.** Static reading only. The cluster is in a bad state
  and additional load makes diagnosis harder.
- **Do not propose a rewrite** unless asked. Locate and explain first.
- Cite **line numbers**.
- If a pattern is ambiguous, say so rather than guessing. "This may run on a
  worker depending on how `ds` was built" is a useful finding.
- Prior wrong turns to avoid re-deriving: manifest RAM is *not* the constraint
  (store is split, largest manifest 1.1 MB); AWS 503 is *intermittent and not
  our request rate*; the link is fine when healthy (0.23 s per global message);
  and the worker-OOM/nanny-kill chain is real but was a *consequence* of the
  eager pattern, not the cause of the hangs. The cause was A0. See
  `BLOCKERS.md` §0.
- **A0 outranks everything else.** If a script uses `chunks={}` on one of these
  arrays, report that and stop — the other findings are downstream of it.
