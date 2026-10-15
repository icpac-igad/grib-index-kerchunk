# How the Ceph → source.coop mirror uses the six VMs

**Written 2026-08-19, during the live run.** Answers one question that is easy
to get wrong: *frisky is stopped, so what is running on the VMs?*

**Short answer: the same six machines, a different cluster.** The realization
used **Frisky** workers. The mirror uses the **Dask** cluster, which has been
up the whole time and is used purely as a remote-execution channel. Stopping
frisky freed the VMs' CPU and network; it did not remove the way to reach them.

---

## 1. Two clusters, one set of machines

| | Frisky | Dask |
|---|---|---|
| what it runs | the realization (`build_corpus.py`) | the mirror's shard dispatch |
| workers | 2 per VM × 8 threads = 96 readers | 1 per VM = 6 |
| lifecycle | started/stopped by `deploy_workers.py` | long-lived, up since Aug 3 |
| status now | **0 across 6 VMs — stopped** | **6 workers — in use** |

`deploy_workers.py --stop` deliberately leaves Dask alone; its output even says
so (`Dask workers untouched`). That is why the mirror can use the VMs while
MAM 2024 is paused: the realization workers are gone, the exec channel is not.

The Dask cluster was never the compute for the realization either — it has
always been the remote-exec channel, because there is no SSH key on the
gateway. Both `deploy_workers.py` and the mirror reach the VMs the same way.

## 2. What is actually running, right now

```
frisky:  0 worker processes across 6 VMs
dask:    6 workers, one per VM
         dask-worker-01   3 procs   58:41  28.6% CPU  412 MB
         dask-worker-02   3 procs   58:41  28.3% CPU  400 MB
         dask-worker-03   3 procs   58:41  28.1% CPU  430 MB
         dask-worker-04   3 procs   58:41  29.8% CPU  322 MB
         dask-worker-05   3 procs   58:41  27.8% CPU  366 MB
         dask-worker-06   3 procs   58:41  27.4% CPU  491 MB
```

Even across all six — same uptime, same CPU, same memory. That evenness is the
sign the sharding is balanced; a skewed shard shows up here first.

## 3. The dispatch chain, end to end

```
gateway:  mirror_all_to_source_coop.sh          (chain: 3 stores, deadline-aware)
            └─ mirror_ceph_to_source_coop.py --via-dask tcp://127.0.0.1:8786
                 ├─ lists Ceph source + source.coop destination
                 ├─ computes `todo` (skips anything already the same size)
                 ├─ shards todo[i::6] -- one slice per VM
                 └─ client.submit(_dask_shard, shard, cfg, creds, workers=[addr])
                        │
VM (x6):                └─ dask worker runs _dask_shard
                              └─ subprocess: /tmp/frisky-ea/.venv/bin/python -c ...
                                   └─ ThreadPoolExecutor(48)
                                        GET  from Ceph  (LAN)
                                        PUT  to source.coop (WAN)
```

**Concurrency: 6 VMs × 1 process × 48 threads = 288 concurrent transfers.**

### Why a subprocess, and not just the dask worker

The dask workers run `/opt/mamba/envs/dask`, which **has no boto3 and must not
be given any** — it is version-pinned to all six workers and drift there breaks
the realization. So the shard runner shells out to `/tmp/frisky-ea/.venv`, the
venv this project owns, into which boto3 was installed for exactly this.

### Why `workers=[addr]` and not `client.run`

`client.run` sends the *same* arguments to every worker. Each VM needs a
*different* slice, so the mirror uses `submit` pinned to one address per shard.

### Credentials

The Ceph `AK`/`SK` and the source.coop STS triple travel as **function
arguments** and then as subprocess environment. Nothing is written to disk on
any VM. The shard list is the only thing that touches VM disk — a temp JSON,
deleted in a `finally`.

## 4. Why processes, not more threads

Measured before building this (see `mirror_ceph_to_source_coop.py` docstring):

| | MB/s |
|---|---|
| 48 threads, 1 process, gateway | 55 |
| **96 threads**, 1 process, gateway | 78 — *and 2 errors* |
| 48 threads × 2 processes, gateway | 78 |
| 48 threads × 4 processes, gateway | 90 |
| 48 threads × 6 VMs | 166 |

Threads plateau and start erroring; processes keep scaling. Same per-process
connection cap `SCALING.md` records for icechunk reads on the way in. So the
lever is process/VM count, and `--threads 48` is left alone.

## 5. Live rate, and why it differs from the benchmark

| | GB/h |
|---|---|
| ad-hoc benchmark, 6 VMs | 599 |
| **50r1 tail, real run** | **426** |
| `49r1-mam2026`, in flight | ~437 |

The benchmark timed only the upload loop against blobs already in memory. A
real run **reads every object from Ceph first**, so ~426–437 GB/h is the
number to plan with. The chain script uses 426.

## 6. Safety properties worth not breaking

- **`repo` is uploaded LAST.** A pass that leaves any error does not write it,
  so a half-finished mirror is an *incomplete* store, never a complete-looking
  one pointing at absent objects. On the 50r1 tail, 25 of 45,229 objects failed
  on pass 1; the pointer was withheld and pass 2 cleared them in 5 s.
- **Resume is by size comparison**, so re-running is cheap and idempotent. This
  is what makes credential expiry survivable and retries free.
- **Nothing is deleted from Ceph.** The mirror is copy-only; deletion is a
  separate, deliberate step, currently on hold.
- **The chain checks credential life before each store** and refuses to start
  one it cannot finish, rather than dying half-way unattended.

## 7. What this blocks

The mirror and the realization **cannot run at once** — they want the same six
VMs, and the realization is the one that saturates them. MAM 2024 is paused at
33/92 for exactly this reason and resumes only after the mirror finishes:

```bash
$D deploy_workers.py --scheduler-on 192.168.1.74 --start \
     --workers-per-vm 2 --nthreads 8 --memory-limit 6GB
tmux new-session -d -s seasons ./run_all_seasons.sh    # skips finished seasons
```

## 8. Commands, for reference

```bash
# credentials (12 h, browser OAuth -- see CLUSTER notes for the headless trick)
~/.local/bin/source-coop login --duration 12h --format env
source <this folder>/.env

# one store
$P mirror_ceph_to_source_coop.py --via-dask tcp://127.0.0.1:8786 --threads 48 \
    --src-prefix ea-swio/v3-49r1-mam2026 \
    --dest forecasts/ecmwf-ifs-ea-swio-realized-v3/49r1-mam2026 --env .env

# the whole chain, deadline-aware
tmux new-session -d -s mirror ./mirror_all_to_source_coop.sh
tail -f mirror_all.log
```

Destination naming drops the `v3-` from each season, since the repository name
`ecmwf-ifs-ea-swio-realized-v3` already carries it.
