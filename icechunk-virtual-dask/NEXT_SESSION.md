# Next session: getting the virtual store realized, without Dask fighting us

**ROOT CAUSE FOUND AND FIXED (`e77f2aa`).** `xr.open_zarr(..., chunks={})`
built the task graph over every chunk in the store before any selection —
665,966 tasks for `t2m`, ~9.3 M for `u`, never finishing. The hang was in the
**client**, during graph construction, before anything reached the scheduler.
Fix: select first, `.chunk()` after. `BLOCKERS.md` §0.

Much of the ordering below was written before that was known. §4.1 (restart the
scheduler) still stands. §4.2–4.5 are now secondary.

_Written 2026-08-03. Companion to `BLOCKERS.md` (evidence) and `CHANGELOG.md`
(how the diagnosis evolved, including two of my own wrong turns)._

---

## 1. Where we are

| | state |
|---|---|
| Reading the published store, single machine | ✅ **works, reliably** — 0.23–1.29 s per global GRIB message |
| Writing a realized Icechunk store to EWC S3 | ✅ works — `test-icechunk-write.py` 8/8 |
| EWC credentials, client and workers | ✅ 6/6, `.env` verified |
| AWS 503 SlowDown | ⚠️ **intermittent nuisance** — 0/6 one minute, 4/4 at 0.23 s the next |
| Reading a pressure-level variable through dask | ✅ **fixed** — was ~9.3 M graph tasks, now passes in seconds at 1.3 GB |
| Realized data actually produced | **0 bytes** — the fix landed after the last write attempt |

**Nothing has been materialized yet.** Every attempt has died on the Dask path.

---

## 2. What is proven to work — build on this

`quick-run.py`, run on the JupyterHub VM, no Dask:

```
repo opened in 5.2s
t2m @ 2026-07-02, member 0 (control), f000
  shape (721, 1440) decoded in 1.95s
  finite fraction 1.000
  min/mean/max = 199.2 / 282.1 / 313.1 K
```

Repeated single-message reads, same path, four in a row:

```
1  OK 0.92s   2  OK 0.25s   3  OK 0.24s   4  OK 0.23s
```

**0.23–0.25 s per whole global message once warm.** That is the number to
build a plan on — not the 2.33 msg/s in `BLOCKERS.md` §3, which was measured
during a throttled window and is pessimistic.

Also proven, in `grib-index-kerchunk/ecmwf/icechunk-par/`:
- `test_dask_read.py` — LocalCluster, **one** variable, lazy `chunks={}`, passes
- `test_dask_read_ewc.py` — this EWC cluster, **one** variable, passes

So Dask is not inherently broken here. **One variable, lazy, works. Our
30-channel eager arrangement does not.**

---

## 3. What hangs, and the evidence

Reproducible every time we submit real work:

```
scheduler tasks : 236   {'processing': 7, 'erred': 39, 'released': 50, 'waiting': 140}
executing across cluster: 0
```

The scheduler believes 7 tasks are processing and 140 are waiting. **No worker
is executing anything.** Workers sit at 0.09–0.22 GB, idle. Worker ports change
between checks, meaning they died and were restarted mid-run.

**AWS cannot cause this.** A 503 returns in milliseconds and the task *errors*
— it does not park tasks in `processing` with no worker running them. I spent
a long time attributing this to AWS and was wrong; see `CHANGELOG.md`.

### The mechanism — SUPERSEDED, see `BLOCKERS.md` §0

The chain below is a real description of what the *eager* pattern does to
workers, and it is why the cluster is full of wreckage. But it is **not** why
runs hung: they hung in the client before submitting anything. Kept for the
record.

1. A task allocates several GB that **Dask cannot see** — `managed` reports
   0.00 GB against 33 GB resident (`BLOCKERS.md` §5), because the memory lives
   in the Rust icechunk/object-store layer and the gribberish decoder.
2. Dask therefore thinks the worker is empty and schedules `nthreads=4`
   concurrent tasks onto it.
3. The worker exceeds `memory_limit`; the nanny kills it.
4. The scheduler keeps the dead worker's tasks in `processing`, and everything
   downstream waits forever.
5. Stale tasks accumulate across runs (236 now, owned by **zombie clients**
   whose processes are gone) and cannot be released — `client_releases_keys`
   does not clear another client's keys, and `client.restart()` raises
   `assert not self.tasks`.

---

## 4. Low-hanging fruit, in the order to do it

### 4.1 Restart the scheduler service — 5 minutes, unblocks diagnosis

236 stale tasks and 2 zombie clients are wedged in the scheduler. `stop_work.py
restart` bounces the *workers* but cannot clear scheduler state. Until this is
clean, every measurement is contaminated.

```bash
# on the gateway, whatever the unit is called
sudo systemctl restart dask-scheduler
# then confirm
python cluster_status.py && python stop_work.py status   # expect 0 tasks
```

### 4.2 Run workers with `nthreads=1` — the single highest-value change

Dask cannot see this workload's memory, so its only safe over-subscription
factor is **1**. Four threads per worker simply gives it four ways to kill
itself. Six single-threaded workers will be *faster* than six four-threaded
ones that keep dying.

Change in the worker cloud-init / service definition:
`--nthreads 1` (and consider `--memory-limit 12GB` to leave headroom).

### 4.3 Never call `.compute()` inside a Dask task

This is the specific arrangement bug. `realize_smoke_test.py` does:

```python
# INSIDE a dask task:
cube = da.sel(number=all, step=all).compute()   # nested sync + asyncio
```

That nests zarr's `sync()` event loop inside a Dask worker thread, holds every
member × step live at once, and fires the whole request burst from one task.
The working reference tests never do this.

Use the lazy form and let Dask schedule chunk-level tasks:

```python
ds  = xr.open_zarr(..., chunks={})          # 1 dask task per GRIB message
out = client.compute([ds[v].sel(...).mean("number") for v in vars])
```

`test_single_date.py` already implements both, with `--eager` to A/B them.
**That comparison is still unrun** — it is the first experiment for next
session.

### 4.4 Wrap reads in 503 retry

icechunk reports AWS throttling as `unhandled error (SlowDown)` and does not
retry. One 503 anywhere fails the whole read. The backoff in
`realize_smoke_test.py` works (`503s 1` on the one channel that completed) —
lift it into whatever the final reader is.

### 4.5 Prove one variable before thirty

`test_dask_read_ewc.py` passes with one variable. Start there on a clean
cluster, then ramp. If it passes at 1 and fails at 8, that localises the
problem precisely.

---

## 4.6 Read `ICECHUNK_DASK_GUIDANCE.md` — the library documents this

Icechunk's own Dask guide states, verbatim:

> "it's not possible to use the existing `Dask.Array.to_zarr` or
> `Xarray.Dataset.to_zarr` functions with either the Dask **multiprocessing or
> distributed** schedulers. **(It is fine with the multithreaded scheduler.)**"

That is about writes, and our writes are already within spec (`to_icechunk`).
But it tells us where the sharp edges are: one session per process, and the
distributed scheduler is the case they had to build `fork`/`merge` machinery
for. Distributed *reads* are not documented at all.

**The cheapest untried experiment follows directly from it:** run the whole
single-date read in ONE process with Dask's threaded scheduler. No pickling, no
scheduler, no nannies — and it keeps the parallelism. See
`ICECHUNK_DASK_GUIDANCE.md` §3.1. Run it on a worker VM, not the 8 GiB
JupyterHub session.

---

## 5. On your question: design the DAG from the single-machine test?

**Yes — and go further: consider not using Dask for the read at all.**

### Why Dask buys us little here

The workload is **embarrassingly parallel**. Every `(date, channel)` is
independent:

- no inter-task communication
- no shuffle
- no cross-chunk dependencies
- the only reduction is over `number`, entirely within one channel

Dask's scheduler, shuffle machinery and memory manager are all overhead we are
paying for and, in the memory manager's case, actively harmed by — because it
schedules on memory numbers that are wrong.

### Option A — build the DAG properly (stay with Dask)

Take `quick-run.py`'s proven open, then build **one lazy graph per date** and
submit it once:

```python
ds = xr.open_zarr(sess.store, group="50r1/00z", chunks={},      # LAZY
                  consolidated=False, zarr_format=3)
graph = []
for var, lev, name in CHANNELS:
    da = ds[var].sel(time=d, number=MEMBERS, step=STEPS,
                     latitude=slice(25.25, -15.25),
                     longitude=slice(18.5, 55.0))
    if lev: da = da.sel(isobaricInhPa=lev)
    graph += [da.mean("number"), da.std("number")]
results = client.compute(graph, sync=True)     # ONE submission, no nesting
```

Dask then owns chunk-level scheduling and releases each message as it folds
into the reduction. Pair with `nthreads=1` and 503 retry.

### Option B — drop Dask for the read (recommended to try first)

Since the work is independent, a plain process pool does the same job with far
fewer failure modes:

```
for each (date, channel) shard:
    fork a process
    open the store  (quick-run.py recipe, ~5 s)
    read members x steps sequentially, with 503 backoff
    reduce to mean+sd
    write its own region into the Icechunk store, or a .npy to be assembled
```

- No scheduler to wedge, no zombie clients, no unmanaged-memory mis-scheduling
- A crashed shard loses one shard, not the run
- Memory bounded by construction: one process, one channel at a time
- Restart granularity is one shard

**Sizing from measured numbers** (0.25 s/message, 81,090 messages/date):

| parallelism | one date | June 2026 (30 dates) |
|---|---:|---:|
| 1 process | 5.6 h | 7 days |
| 6 processes | 56 min | 1.2 days |
| 24 processes | 14 min | 7 h |

Six worker VMs × 4 vCPU = 24 processes, which is the same hardware, just
without Dask in the middle.

### Option C — run it in AWS `eu-central-1`

Still the structural answer: read 60.6 TB in-region, ship ~4.4 TB back. Removes
the cross-cloud 503s entirely. `BLOCKERS.md` §10.

---

## 6. Concrete checklist for next session

1. [ ] Restart the **scheduler** service; confirm `stop_work.py status` shows
       0 tasks and no zombie clients
2. [ ] Re-run `cluster_status.py` — expect ~0.09 GB/worker, 6/6 credentials
3. [ ] `python test_single_date.py --vars t2m --members 4 --steps 2`
       — one variable, lazy. **Must pass** before anything else
4. [ ] `python test_single_date.py --ramp --members 4 --steps 2`
       — 1, 2, 4, 8 variables. Find where it breaks
5. [ ] `python test_single_date.py --vars t2m --members 4 --steps 2 --eager`
       — confirm the eager pattern is what breaks it
6. [ ] If Dask still hangs at low variable counts, **switch to Option B** and
       stop spending time on the scheduler
7. [ ] First real target: **one date**, 30 channels, 51 members, 53 steps →
       measure the compression ratio, which is still assumed at 40 %
8. [ ] Then June 2026, 30 dates

**Do not** start with 30 channels on 6 workers again. That is what has failed
every time.

---

## 7. Tooling in this directory

| script | use |
|---|---|
| `quick-run.py` | the known-good single-machine read. Start here when anything is confusing |
| `cluster_status.py` | worker rss vs managed, headroom, credentials. ~2 s |
| `stop_work.py` | `status` / `cancel` / `restart` / `kill` |
| `fix_worker_credentials.py` | `check` / `restore` / `restart` |
| `test_single_date.py` | single date, `--ramp` variable count, `--eager` to A/B the bad pattern |
| `where_does_it_fail.py` | client vs worker, egress IPs, raw HTTP vs icechunk |
| `realize_smoke_test.py` | end-to-end realization. **Uses the eager pattern — fix or replace before reusing** |
| `materialize_ea_icechunk_ewc.py` | the full extraction tool. `plan` / `corpus` are useful now; `run` inherits the eager pattern |

---

## 8. Things I got wrong, so they are not re-litigated

1. **"Manifest RAM is the constraint, 89.5 GB, 96 % of the archive
   unreachable."** Wrong. The store is already split one manifest per date,
   largest object 1.1 MB.
2. **"The AWS 503 is self-inflicted by our hammering and will decay."** Wrong.
   A single request from a clean process was also refused, and it recovers in
   minutes without us doing anything.
3. **"The AWS→EWC link is the constraint, 3.5–60 days."** Measured entirely
   during throttled windows. When the path is healthy it is 0.23 s per global
   message, which is fine.
4. **"The workers are the problem because they are throttled."** Client and
   workers have different egress addresses (136.156.130.111 vs
   136.156.131.254) and both fluctuate. Neither is reliably worse.

The one that has held up under every test: **the Dask arrangement is the
blocker**, and it is ours to fix.
