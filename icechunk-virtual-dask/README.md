# icechunk-virtual-dask

Materialising an East Africa subset of the published **ECMWF IFS ensemble
virtual Icechunk store** into a **realized Icechunk store** on the ECMWF
European Weather Cloud (EWC) object store, using the EWC Dask cluster.

> **Status: the hang is fixed; no data materialized yet.**
> The blocker was `xr.open_zarr(..., chunks={})` building a task graph over
> every chunk in the store before any selection — 665,966 tasks for `t2m`,
> ~9.3 M for `u`, never finishing, **in the client**. Fix: select first,
> `.chunk()` after (`e77f2aa`). See [`BLOCKERS.md`](BLOCKERS.md) §0.
>
> Still outstanding: the scheduler holds ~236 orphaned tasks from the failed
> runs and needs `sudo systemctl restart dask-scheduler`; and no realization
> run has been attempted since the fix. [`NEXT_SESSION.md`](NEXT_SESSION.md)
> has the order to work in.

---

## 1. Environment

Everything runs in a **micromamba** environment named `dask`, at
`/opt/mamba/envs/dask`. **The Dask workers use the same environment**, at the
same path, on their own VMs — verified identical:

```
client python : 3.12.13   /opt/mamba/envs/dask/bin/python
worker python : 3.12.13   /opt/mamba/envs/dask/bin/python3.12
version parity: IDENTICAL on all 6 workers
```

That parity matters more than it looks. Dask ships task functions by pickle; if
the client and workers disagree on `dask`, `distributed`, `zarr` or
`cloudpickle`, the failure is **not** an `ImportError` — it is an opaque
deserialization error deep inside a task. `test_dask_read_ewc.py` in
`grib-index-kerchunk/ecmwf/icechunk-par/` checks parity explicitly for exactly
this reason.

### Pinned versions (client and all workers)

| package | version | | package | version |
|---|---|---|---|---|
| python | 3.12.13 | | icechunk | 2.1.1 |
| dask | 2026.7.1 | | gribberish | 1.6.0 |
| distributed | 2026.7.1 | | s3fs | 2026.6.0 |
| zarr | 3.2.1 | | netCDF4 | 1.7.4 |
| xarray | 2026.7.0 | | cloudpickle | 3.1.2 |
| numpy | 2.5.1 | | pandas | 3.0.3 |

### Always call the interpreter by full path

There is no `conda activate` in these scripts, and the login shell does **not**
have the env on `PATH`. Use the absolute path everywhere:

```bash
P=/opt/mamba/envs/dask/bin/python
$P cluster_status.py
```

Every example below assumes `P` is set like that.

### Recreating the environment elsewhere

```bash
micromamba create -n dask -c conda-forge \
    python=3.12 \
    dask=2026.7.1 distributed=2026.7.1 \
    zarr=3.2.1 xarray=2026.7.0 numpy=2.5.1 pandas=3.0.3 \
    s3fs netcdf4 matplotlib cloudpickle

micromamba run -n dask pip install icechunk==2.1.1 gribberish==1.6.0
```

`icechunk` and `gribberish` come from PyPI — both are Rust extension modules,
which is directly relevant to the memory behaviour documented in
`BLOCKERS.md` §5 (their allocations are invisible to Dask).

**If you rebuild the workers, rebuild them from the same spec**, or the parity
check will start failing.

---

## 2. Credentials

Two different sets, and keeping them apart is essential:

| | store | credentials |
|---|---|---|
| **Source** (read) | `source.coop` → virtual chunks on AWS `s3://ecmwf-forecasts` | **anonymous**, `from_env=False` |
| **Sink** (write) | EWC Ceph RadosGW, `s3://must-icechunk` | `AK` / `SK` from `.env` |

Create `.env` in this directory (gitignored via `*.env`):

```
AK=<ewc access key>
SK=<ewc secret key>
```

### The trap: `AWS_*` must be absent when reading the source

The Dask workers carry `AWS_ENDPOINT_URL` (the Ceph endpoint) and
`AWS_DEFAULT_REGION=RegionOne` so they can write to `must-icechunk`. The
object-store client icechunk builds for the **virtual chunk container** picks
those up too, and then tries to resolve a hostname that does not exist:

```
error fetching virtual reference -> dispatch failure -> io error
  -> client error (Connect) -> dns error
  -> failed to lookup address information: Name or service not known
```

Three rules follow, all learned the hard way:

1. `AWS_*` must be **absent before the process builds its first S3 client** —
   popping it afterwards does not help, the config is cached process-wide.
2. It must stay absent for the **whole read**, not just the open. The
   virtual-chunk client is not constructed until the first chunk is fetched,
   i.e. inside `.compute()`.
3. It must be **put back** afterwards, or you strip the workers' Ceph
   credentials for every other job on the cluster.

`fix_worker_credentials.py` repairs workers left stripped by earlier versions
of this tooling.

---

## 3. Cluster

```
JupyterHub gateway VM  (this machine)     -> DASK_SCHEDULER_ADDRESS=tcp://127.0.0.1:8786
  dask-worker-01..06   192.168.1.x         4 vCPU / 16.77 GB each
                                           Dask memory_limit 13.94 GB, nthreads 4
```

Dashboard: `https://jupyter-ewc-must.e4drr-cloud.work/user/$JUPYTERHUB_USER/proxy/8787/status`

Two limits worth knowing before you debug anything:

- **This JupyterHub session is capped at 8 GiB** —
  `/sys/fs/cgroup/system.slice/jupyter-<user>.service/memory.max`. `free` shows
  32 GB because that is the host. Client-side reads of many channels get
  SIGKILLed (exit 137).
- **Workers should be run with `nthreads=1`**, not 4. Dask cannot see this
  workload's memory (`managed` reports 0.00 GB against 33 GB resident), so it
  over-subscribes and the nanny kills them. See `NEXT_SESSION.md` §4.2.

---

## 4. Quick start

```bash
P=/opt/mamba/envs/dask/bin/python
cd ~/cGAN_tutorial/icechunk-virtual-dask
```

**1. Does the store read at all?** No cluster, no credentials needed. This is
the known-good path — come back here whenever something is confusing.

```bash
$P quick-run.py
```

Expect the three eras to list, and a `t2m` field to decode in ~0.2–2 s.

**1b. How big a graph does your read build?** Also no cluster. Do this before
submitting anything — a graph over ~1 M tasks hangs the *client*, which looks
exactly like a cluster failure and is not one.

```bash
$P graph_size.py --vars t2m u
```

**2. What state is the cluster in?**

```bash
$P cluster_status.py            # rss vs managed, headroom, credentials
$P stop_work.py status          # what the SCHEDULER thinks is running
```

**3. Stuck? Clear it.**

```bash
$P stop_work.py restart         # cancel + bounce workers, reclaims memory
```

`client.restart()` on its own is unreliable here — with tasks registered it
raises `assert not self.tasks` inside dask and leaves the cluster running.

**4. Single-date read test** — the current focus.

```bash
$P test_single_date.py --vars t2m --members 4 --steps 2      # must pass first
$P test_single_date.py --ramp   --members 4 --steps 2        # 1,2,4,8 vars
$P test_single_date.py --vars t2m --members 4 --steps 2 --eager   # the bad pattern
```

**5. Sizing, no cluster, no cost.**

```bash
$P materialize_ea_icechunk_ewc.py plan --days 30 --lead-days 7
$P materialize_ea_icechunk_ewc.py corpus --store-members
```

---

## 5. Scripts

| script | what it does | needs cluster | writes |
|---|---|---|---|
| `quick-run.py` | known-good single-machine read of the published store | no | no |
| `cluster_status.py` | worker rss vs managed, headroom, credentials, what is executing | yes | no |
| `stop_work.py` | `status` / `cancel` / `restart` / `kill` | yes | no |
| `fix_worker_credentials.py` | `check` / `restore` / `restart` for stripped workers | yes | no |
| `test_single_date.py` | one date, `--ramp` variable count, `--eager` to A/B the bad pattern | either | no |
| `graph_size.py` | **how many dask tasks does this read build?** Measure this BEFORE running anything | no | no |
| `where_does_it_fail.py` | client vs worker, egress IPs, raw HTTP vs icechunk | yes | no |
| `realize_smoke_test.py` | end-to-end realization to `must-icechunk` | optional `--dask` | **yes** |
| `materialize_ea_icechunk_ewc.py` | full extraction tool: `plan` `corpus` `probe` `run` `size` | `run`/`probe` | `run` only |

> `realize_smoke_test.py` and `materialize_ea_icechunk_ewc.py run` both use the
> **eager** read pattern that is currently hanging the cluster. Fix or replace
> that before reusing them — `NEXT_SESSION.md` §4.3.

---

## 6. Documents

| file | contents |
|---|---|
| **`NEXT_SESSION.md`** | **start here.** What works, what hangs, the order to attack it, and two design options |
| **`DAG_METHOD.md`** | **the method.** Given a store and a cluster, the order to make a read work — measure the graph first, then climb the ladder |
| `BLOCKERS.md` | evidence: the 503s, the unmanaged memory, the 8 GiB client cap, where it fails in sequence |
| `EA_MATERIALIZATION_PLAN.md` | extraction design: channels, extent, steps, members, output layout |
| `EWC_USAGE_AND_RESOURCE_PLAN.md` | what has actually been used on EWC, and the resource ask |
| `CHANGELOG.md` | how the diagnosis evolved, **including the wrong turns** |
| `OPENING_PUBLISHED_ECMWF_ICECHUNK.ipynb` | annotated walkthrough of opening the store |

Several claims in these documents were made and later **withdrawn** — the
manifest-RAM figure, the self-inflicted-throttle theory, and the schedule
estimates derived from throttled measurements. They are marked as withdrawn in
place rather than deleted, because they were shared before being disproven.
`CHANGELOG.md` and `NEXT_SESSION.md` §8 list them.

---

## 7. Related work elsewhere in the repo

| path | why it matters |
|---|---|
| `../../test-icechunk-write.py` | proves Icechunk commits work on Ceph RGW (8/8) |
| `../../test-icechunk-long.py` | longer synthetic workload; exercises worker-to-worker traffic. Never touches AWS, so useful when the source path is unhappy |
| `../docs/ecmwf_icechunk_dask_variable_extraction.md` | which variables, which levels, which era |
| `../../grib-index-kerchunk/ecmwf/icechunk-par/test_dask_read.py` | **passes** — LocalCluster, one variable, lazy `chunks={}` |
| `../../grib-index-kerchunk/ecmwf/icechunk-par/test_dask_read_ewc.py` | **passes** — this cluster, one variable. The pattern to copy |

Those last two are the reference implementations. One variable, read lazily,
works. Thirty channels read eagerly does not — which is the whole of the
current problem.
