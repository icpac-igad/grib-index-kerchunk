# Frisky: a DAG whose size tracks the request

`DAG_METHOD.md` §6 leaves one thing unsolved:

> **The select-first pattern builds a 3.76 M-task graph for `u`.** [...] it is
> wasted work, it scales with the store rather than with the request [...]
> A graph whose size tracks the *request* rather than the *store* is the goal.
> We are not there; we have only made it not hang.

This directory is an answer to that: `frisky_daily_dag.py` builds the same
daily reduction as a **futures DAG** — one task per GRIB message — so there is
no dask-array graph to construct, and the task count is exactly what was asked
for.

|  | tasks to read 2 members × 1 step of `u` |
|---|---|
| `chunks={}` then `.sel()` | never finishes constructing — **hang** |
| `.sel()` then `.chunk()` | 3,761,156 |
| Frisky futures | **3** |

Both dask numbers are `graph_size.py`'s, measured on this store. The full
production shape (30 channels × 51 members × 53 steps, one date) is
**82,710 Frisky tasks** — 1.02 tasks per message, sized by the request.

---

## What Frisky does and does not fix

**Does not.** Frisky cannot rescue a dask collection from §6. `frisky.hijack`
and `client.compute` both receive the graph *after* the client has built it, so
the 3.76 M tasks — or the hang — happen before Frisky is involved.
**`graph_size.py` stays the right tool for measuring a dask graph.**

Frisky's `submit_expression` does expand graphs scheduler-side in Rust rather
than materialising them in the client, which is the one path that would help.
It needs `dask@main` plus the `dask-array` Rust backend, which would break the
client/worker version parity `README.md` §1 depends on — and it still expands
to 3.76 M tasks, just elsewhere. Not worth it while the futures form is
available.

**Does.** Measured on this VM, 2026-08-03:

| | |
|---|---|
| Worker sizing reads the **cgroup** limit | Frisky sees the real 8 GiB cap here; psutil — and so Dask — reports the host's 31.3 GiB. That gap is the exit-137 client kills in `BLOCKERS.md` |
| `--threads 1` is a flag | The §4.2 fix without a cloud-init change, and no nanny in the loop |
| Fresh scheduler state | No 236 stale tasks, no zombie clients, and `client.restart()` works — no `assert not self.tasks` |
| ~3 µs/task vs Dask's ~1 ms | 0.25 s vs ~83 s of pure submission per date at 82,710 tasks |

Frisky does **not** solve the root memory problem: icechunk and gribberish
allocate in Rust, so `managed` reads ~0 against GBs resident, and Frisky sizes
from managed memory just as Dask does. What fixes that here is the task shape —
one message in, ~0.1 MB out, ~5 MB held per reduction — not the scheduler.

---

## Against icechunk's own limitation (`81affcd`, `ICECHUNK_DASK_GUIDANCE.md`)

The documented hard limit is a **write** limit:

> "it's not possible to use the existing `Dask.Array.to_zarr` or
> `Xarray.Dataset.to_zarr` functions with either the Dask multiprocessing or
> distributed schedulers. (It is fine with the multithreaded scheduler.)"

`81affcd` reads the *shape* of that limitation correctly: **one session per
process** is the assumption the library is built around, and the untested thing
in our read path was "open the store client-side and let Dask pickle it into
six worker processes."

This DAG never does that. `_open_era()` runs **inside the worker process** and
caches at module level; the session is never an argument to a task and never
crosses a process boundary. One session per process, which is the supported
shape — reached by construction rather than by care.

Two related rules, and where this differs from `test_single_date.py`:

| rule | `test_single_date.py` | here |
|---|---|---|
| never `chunks={}` before `.sel()` | ✅ opens lazily-indexed, selects, then `.chunk()` | ✅ opens lazily-indexed and **never calls `.chunk()`** — there is no dask array at any point, so `.sel().values` is a direct single-chunk icechunk read |
| level before `.sel()` | ✅ `.isel(isobaricInhPa=0)` first | ✅ `.sel(isobaricInhPa=level)` first |
| one session per process | ⚠️ distributed mode pickles a client-side store | ✅ opened on the worker |

**On writes, the limitation still binds and is not yet addressed here.** When
the sink write is added it must use `icechunk.xarray.to_icechunk` (which
handles the fork), not `to_zarr` on a `zarr.open_array`, and the sink chunking
must keep dividing the dask chunks. `--out` currently writes `.npz`, so nothing
here is on that path yet.

---

## Environment

Deliberately **not** `/opt/mamba/envs/dask`. That env is version-pinned to all
six workers (`README.md` §1) and must not drift. Frisky lives in an isolated
venv here:

```bash
cd ~/cGAN_tutorial/icechunk-virtual-dask/frisky
/opt/mamba/envs/dask/bin/python -m venv .venv
.venv/bin/pip install "frisky>=0.3.0" icechunk==2.1.1 zarr==3.2.1 \
                      xarray==2026.7.0 gribberish==1.6.0 numpy
```

Installed and verified: frisky 0.7.2, icechunk 2.1.1, zarr 3.2.1,
xarray 2026.7.0, gribberish 1.6.0.

> The upstream "Try it out" recipe (`dask-array`, `dask@main`, `xarray@main`,
> `dask_array.xarray.register()`) is for accelerating **dask collections**.
> This DAG uses futures, so it needs none of it — which is just as well, since
> those git-main pins are exactly what would break worker parity.

---

## Usage

```bash
P=~/cGAN_tutorial/icechunk-virtual-dask/frisky/.venv/bin/python

$P frisky_daily_dag.py --count --channels 30 --members 51 --steps 53   # size only
$P frisky_daily_dag.py --synthetic --channels 30 --members 51 --steps 6 # shape, no network
$P frisky_daily_dag.py --date 2026-07-02 --channels 1 --members 4 --steps 2
$P frisky_daily_dag.py --date 2026-07-02 --ramp --members 4 --steps 2
```

Dashboard defaults to `127.0.0.1:8790` — 8787 is the Dask scheduler's on this VM.

For a cluster, run Frisky standalone rather than hijacking the Dask one
(`hijack` inherits `worker.state.nthreads`, which is 4 here — the
over-subscription `NEXT_SESSION.md` §4.2 warns about):

```bash
frisky scheduler                              # gateway
frisky worker <sched>:8786 --nthreads 1       # each worker VM, AWS_* unset
$P frisky_daily_dag.py --scheduler tcp://<sched>:8786 --date 2026-07-02
frisky observe overview                       # diagnostics
```

## Status

### `ea-cgan/v2-7day` — a full date with members, via fork/merge

```
30 channels x 51 members x 53 steps, 2026-07-02, era 50r1
  81,090 reads + 1,590 chunk writes
  schema        15.0 s   (metadata only, 30 arrays)
  submit       476.4 s   <-- 1,590 x session.fork()
  read+write   477.0 s   (overlapped with submit)
  merge          2.6 s   1,590 changesets -> 4BX3QNEG92Q7
  TOTAL        494.6 s   ONE commit, 0 failed blocks
```

Verified: 30/30 channels, finite 1.000, and `gh500` spread 1.4285 → 5.8914
from +0h to +168h — **the same numbers the client-side v1 store produced**, so
two independent write paths agree.

#### Compression: the 40% assumption holds

| | |
|---|---|
| raw float32 | 7.77 GB |
| on Ceph | **3.09 GB** (3.01 GB chunks, 6,400 objects) |
| ratio | **39.8%** |

**This corrects an earlier entry in this file.** A previous run reported 77.2%
and called the 40% assumption in `NEXT_SESSION.md` §6.7 badly wrong. That
measurement was `numpy.savez_compressed` (zlib) over mean/sd fields — not the
store's codec, and not the store's data. Measured through icechunk on the real
payload, 40% was right. June 2026 with members is ~93 GB, not ~180 GB.

#### The bottleneck is now `session.fork()`, not I/O

476 s of the 494 s was **submission** — `session.fork()` called once per write
block, at ~0.30 s each. Every block was written within a second of submission
finishing, so the cluster was starved the whole run and the true read time was
never exposed.

`--fork-once` forks a single session and pickles it to every task; each
unpickled copy records its own changes, so the merge is unchanged. Verified
correct on 8 blocks, where submission went 1.80 s → **0.23 s**. At 1,590 blocks
that should take the run to roughly the read-bound ~390 s, and matters far more
for 30 dates, where per-block forking would cost ~4 hours of pure overhead.

**Not yet verified at 1,590 shared-fork copies** — only the mechanism is
proven, at 8.

### One full date, mean/sd, client-side — 2026-08-03

```
30 channels x 51 members x 53 steps, date 2026-07-02, era 50r1
  81,090 leaves + 1,590 reductions + 30 stacks = 82,710 tasks
  submitted in 2.02 s
  682.4 s  (11 min 22 s)   118.8 messages/s   22.7 s per channel
  30/30 channels, every one finite 1.000
  erred 0   spilled 0 B
```

Peak memory, per `frisky observe workers`:

```
Address              Memory              Managed   Unacct
192.168.1.74:34511   374.64 MiB / 10 GiB   0 B     374.64 MiB
...                  ~372 MiB each         0 B     ~372 MiB
TOTAL                  2.17 GiB / 60 GiB   0 B       2.17 GiB
```

**`Managed 0 B` against 2.17 GiB resident is the `BLOCKERS.md` §5 blindness,
reproduced exactly under Frisky.** Frisky cannot see icechunk's Rust
allocations any better than Dask can. It did not matter: the task shape caps
each worker at ~372 MiB — of which ~332 MiB is the pinned per-process store
session — so there was nothing for a memory manager to get wrong. That is the
argument of this whole directory in one table.

`--nthreads 4` was the right call. 24 concurrent readers, 4% of the memory
budget, no spilling, no worker deaths.

Output: `out/2026-07-02.npz`, **235.3 MB** against 304.8 MB of raw float32
(30 channels x 2 arrays x 53 x 163 x 147 x 4 B) — **77.2%**. `NEXT_SESSION.md`
§6.7 carried an assumed 40%; the measured figure is much worse than that.
**Caveat: this is `numpy.savez_compressed` (zlib), not the icechunk/zarr codec
path**, so treat it as indicative of the payload's compressibility, not as the
store's ratio. The real number needs the sink write.

Two things this run did **not** establish:

- **Whether any 503s occurred.** `erred 0` only means no task failed
  permanently. The backoff in `read_message` retries silently, so a throttled
  read that succeeded on retry leaves no trace anywhere. That is an
  instrumentation gap, not a clean bill of health — the retry should count and
  report.
- **AWS ingress volume.** The ~64 GB figure is derived from `MSG_MB = 0.788`
  in `materialize_ea_icechunk_ewc.py`, not measured here. Frisky's byte
  counters cover task output and worker-to-worker transfer, not the S3 fetch.

### Earlier, local, on the JupyterHub VM (8 GiB cgroup)

| | |
|---|---|
| `frisky demo`, 6,295 tasks | ✅ 15.7 s |
| DAG shape, synthetic, 30 channels / 9,390 tasks | ✅ 33.0 s, 6 workers |
| **Real read of `u700`**, 4 members × 2 steps | ✅ 29.0 s — dominated by 4 cold store opens |
| **Real read of `u700 u850 u500`**, 16 members × 4 steps | ✅ **192 messages, 19.4 s, 6 workers** |
| Write to `must-icechunk` | ⬜ not implemented; `--out` writes `.npz` only |

The `u` runs are the point: `u` is the array that **never finishes graph
construction** under `chunks={}`, and still costs 3.76 M tasks select-first.
Read as futures it is 207 tasks and 19.4 s.

```
tasks:   31/s   waiting=2 processing=8 finished=197
cluster: workers=6  memory=878.2 MiB / 8.0 GiB (11%)
sched:   busy=0.1%
```

- **9.9 messages/s** aggregate including cold opens; **31 tasks/s** warm on 6
  single-threaded workers, i.e. ~0.19 s/message — consistent with the
  0.23–0.25 s single-machine figure in `NEXT_SESSION.md` §2.
- **878 MiB peak across all six workers** (~146 MB each), against
  `test_single_date.py`'s 1,329 MB for `u` under the threaded scheduler.
  Bounded by the task shape, not by a memory manager.
- Scheduler **0.1% busy**. Submission of the whole graph took 0.01 s.
- **No 503s** on this run; the backoff was never exercised, so it remains
  untested against a real `SlowDown`.
- The first run's 29 s for 8 messages is the store-open cost: ~5 s per worker
  process, paid once per process, not per task.

Extrapolating 31 tasks/s to the full 82,710-task date gives ~45 min on 6
workers, ~11 min on 24. Not yet measured at that scale.
