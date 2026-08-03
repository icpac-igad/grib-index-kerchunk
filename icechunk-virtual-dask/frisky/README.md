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

| | |
|---|---|
| `frisky demo`, 6,295 tasks | ✅ 15.7 s |
| DAG shape, synthetic, 30 channels / 9,390 tasks | ✅ 33.0 s, 6 workers |
| Real read through Frisky | ⬜ **not yet run** — the next step |
| Write to `must-icechunk` | ⬜ not implemented; `--out` writes `.npz` only |
