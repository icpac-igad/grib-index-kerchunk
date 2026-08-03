# Icechunk's own Dask guidance, checked against what we are doing

Source: <https://icechunk.io/en/stable/guides/dask/> and
<https://icechunk.io/en/stable/guides/xarray/>, read 2026-08-03 against
icechunk **2.1.1** (our installed version).

**Verdict: our write path is within spec. The read path is in territory the
documentation does not cover at all — and that is where we hang.**

---

## 1. What the docs actually say

### The one hard limitation, quoted

> "it's not possible to use the existing `Dask.Array.to_zarr` or
> `Xarray.Dataset.to_zarr` functions with either the Dask **multiprocessing or
> distributed** schedulers. **(It is fine with the multithreaded scheduler.)**"

And from the Xarray guide:

> "In a distributed context, e.g. writes orchestrated with `multiprocessing` or
> a `dask.distributed.Client` … you **must** use `to_icechunk`."

### Why: sessions do not cross process boundaries by themselves

A writable `Session` accumulates a change-set. Several processes cannot each
hold one and have the result be coherent. The supported shape is
**fork → write remotely → merge → commit**:

```python
fork = session.fork()
zarray = zarr.open_array(fork.store, path="array")
remote_session = icechunk.dask.store_dask(
    sources=[dask_array],
    targets=[zarray]
)
session.merge(remote_session)
session.commit("wrote a dask array!")
```

`to_icechunk` does this for you: *"to_icechunk takes care of handling the
forking."*

### Two constraints that are easy to miss

1. **No `compute` kwarg** on either API.
2. **"chunks in the store are a divisor of the dask chunks"**, so each write
   task is independent — *"It is your responsibility to ensure that such
   conflicts are avoided."*

### Confirmed present in 2.1.1

```
icechunk.dask.store_dask(*, sources, targets, regions=None, split_every=None, **store_kwargs) -> ForkSession
Session.fork()  -> ForkSession
Session.merge(*others: ForkSession) -> None
icechunk.xarray.to_icechunk(obj, session, *, mode, append_dim, region, encoding, split_every, ...)
```

---

## 2. Checked against our code

| our code | verdict |
|---|---|
| `realize_smoke_test.py` → `to_icechunk(ds, session, mode="w", encoding=...)` from the **client**, numpy-backed data | ✅ **within spec.** `to_icechunk` is the documented API and handles forking; with numpy-backed arrays the write is local anyway |
| `materialize_ea_icechunk_ewc.py` → `to_icechunk(..., append_dim="time")` and `region={...}` from the client | ✅ within spec |
| Sink chunking `(1 date, all steps, all members, full box)`, one chunk per (date, channel) | ✅ satisfies the divisor rule — no two write tasks touch a chunk |
| **Reads: open the store client-side, let Dask pickle it into the graph, execute on 6 worker processes** | ⚠️ **undocumented.** Not endorsed, not forbidden |

**So the docs do not explain our hang, but they do tell us we are off the
supported path** — and in the one place they *do* speak about the distributed
scheduler, the verdict is "does not work, use the multithreaded scheduler".

That the write-side limitation is precisely *"distributed and multiprocessing
bad, multithreaded fine"* is a strong hint about where the sharp edges are:
**one session per process**, and the distributed scheduler is the case they had
to build special machinery for.

---

## 3. The way forward this suggests

### 3.1 First — the cheapest test we have not run

> *"It is fine with the multithreaded scheduler."*

Run the whole single-date read in **one process** with Dask's threaded
scheduler. One session, one process, nothing pickled, no scheduler, no nannies:

```python
import dask
ds = open_store()                      # chunks={} -> lazy
with dask.config.set(scheduler="threads", num_workers=8):
    out = dask.compute(*[ds[v].sel(...).mean("number") for v in VARS])
```

- Removes every failure mode in `BLOCKERS.md` §5 and §7 at a stroke: no
  unmanaged-memory mis-scheduling, no nanny kills, no orphaned tasks, no
  zombie clients.
- Keeps the parallelism that makes the job feasible.
- Memory stays bounded because the reduction streams — the reference test
  asserts peak RSS under a 2 GB limit doing exactly this.
- **Caveat: run it on a worker VM (16.77 GB), not the JupyterHub session
  (8 GiB cgroup).**

This is a ~10 line change to `test_single_date.py` and directly tests the
documented "multithreaded is fine" claim.

### 3.2 Then — one process per shard, threaded inside

Scale out by **processes that do not share sessions**, each doing 3.1
internally:

```
for each (date, channel-group):
    a fresh OS process
      opens its own session          (quick-run.py recipe, ~5 s)
      threaded-scheduler read of its shard
      writes its own region with to_icechunk, then commits
```

This is `NEXT_SESSION.md` Option B, and the icechunk docs now support it: each
process has exactly one session, which is the shape the library is built
around. Coordination is a work queue, not a Dask graph.

Sizing from measured numbers (0.25 s/message, 81,090 messages/date):

| processes | one date | June 2026 |
|---|---:|---:|
| 6 | 56 min | 1.2 days |
| 24 | 14 min | 7 h |

### 3.3 If we keep `dask.distributed` for reads

Then treat it as unsupported territory and constrain it hard:

- `nthreads=1` per worker (`NEXT_SESSION.md` §4.2)
- open the session **on the worker**, do not pickle a client-side store into
  the graph — the store handle crossing the process boundary is the untested
  part
- never `.compute()` inside a task (`BLOCKERS.md` / `FABLE_DAG_ANALYSIS_NOTE.md`
  pattern A)
- retry 503 explicitly; icechunk does not

### 3.4 If workers ever need to *write*

Use the documented shape, not `to_zarr`:

```python
fork = session.fork()                      # on the client
# ... workers write through fork.store ...
session.merge(remote_session)
session.commit("...")
```

We do not need this today — the client writes and the payload is ~305 MB/date
— but `--store-members` at 7.8 GB/date may change that, and then this is the
required API.

---

## 4. Recommendation

**Do 3.1 next session.** It is the smallest change, it tests a claim the
library authors make explicitly, and if it works it removes the entire class of
failures we have been fighting — because it deletes the distributed scheduler
from the read path rather than trying to tame it.

If 3.1 works and 3.2 scales it, we never need to solve the distributed-read
problem at all. That is worth more than another week of diagnosing nanny kills.

**Also worth noting for the record:** the icechunk docs are silent on
distributed *reads* — no guidance on session pickling, `chunks={}` across
workers, or thread safety. Given they were careful enough to document the write
limitation, that silence is not reassuring. If 3.1 and 3.2 both fail, an issue
on the icechunk tracker asking whether a readonly session is safe to pickle
into `dask.distributed` workers would be a reasonable next move — it is a
question the documentation currently cannot answer.
