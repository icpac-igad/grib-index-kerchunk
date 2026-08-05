# Realizing the ECMWF IFS ensemble with Frisky + Icechunk

Reads the published virtual Icechunk store on `source.coop` (chunks live on AWS
`s3://ecmwf-forecasts`) and writes a **realized** store to EWC Ceph, using a
[Frisky](https://getfrisky.dev/) futures DAG across the six EWC worker VMs.

**Status: working.** `must-icechunk/ea-cgan/v3-june2026` holds June 2026 —
30 dates × 30 channels × 51 members × 53 steps, 47,700 chunks, 94.6 GB, verified
complete. Built in 176 minutes.

| document | contents |
|---|---|
| this file | the A-to-Z runbook |
| [`SCALING.md`](SCALING.md) | what limits throughput, measured — read before asking for more VMs |
| `dev-test/DAG_METHOD.md` | why the Dask path hung, and how to measure a graph before running it |
| `dev-test/BLOCKERS.md` | the earlier evidence trail |

---

## 1. Throughput actually achieved

June 2026, six VMs, 2 worker processes each × 8 threads = 96 concurrent readers:

| | |
|---|---|
| wall clock | **176.3 min** (10,578 s) for 29 dates |
| messages | **2,432,700** GRIB messages |
| pulled from AWS | **1.92 TB** |
| **pull rate** | **181.2 MB/s = 1.45 Gbps = 0.18 GB/s** |
| messages/s | 230 |
| per VM | 30.2 MB/s |
| per worker process | 15.1 MB/s |
| written to Ceph | 94.6 GB at 8.9 MB/s |

### On the 1 GB/s target

**We are at 0.18 GB/s. 1 GB/s is 5.5× further.** Worth separating two units that
are easy to conflate: we reached **1.45 Gbps**, which is already past 1 Gbps —
but 1 GB/s is 8 Gbps, a different target entirely.

Two things stand in the way, and only one of them is fixable by adding hardware:

- **Per-VM ceiling ~50 MB/s** (icechunk caps connections per process; two
  processes reach it). We are running at 30.2 MB/s per VM, so there is ~1.6×
  of headroom here.
- **AWS request rate**, which is *shared across the whole cluster* and already
  refuses at 96 concurrent readers. This does not improve with more VMs — more
  VMs means more requests against the same quota. See `SCALING.md` §4.

So 1 GB/s is **not** reachable from EWC by adding VMs. Reading in-region
(`eu-central-1`) is the only route. `BLOCKERS.md` §10.

### The 95% you cannot avoid

**Only 4.9% of the bytes pulled are kept** — 94.6 GB retained out of 1,917 GB
pulled. This is not waste in the tuning sense: a GRIB message is the atomic
unit of the store, so fetching the East Africa box (163×147) still means
pulling the whole global message (721×1440) and discarding the rest. No amount
of concurrency or chunk tuning changes it; only a differently-encoded source
would.

It does mean the *useful* output rate is 8.9 MB/s while the *network* rate is
181 MB/s. Size the network for the latter.

---

## 2. Environment

Deliberately **not** `/opt/mamba/envs/dask`. That env is version-pinned to all
six Dask workers (`dev-test/README.md` §1) and must not drift.

```bash
cd ~/grib-index-kerchunk/icechunk-dask/icechunk-virtual-dask

/opt/mamba/envs/dask/bin/python -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install "frisky>=0.3.0" icechunk==2.1.1 zarr==3.2.1 \
                      xarray==2026.7.0 gribberish==1.6.0 numpy==2.5.1 \
                      dask==2026.7.1
```

`dask` is needed only on the **client**, and only by `create_schema`, which
uses `dask.array.zeros` to declare the arrays without materialising them.

Verified: frisky 0.7.2, icechunk 2.1.1, zarr 3.2.1, xarray 2026.7.0,
gribberish 1.6.0, Python 3.12.13.

> The upstream Frisky "Try it out" recipe (`dask-array`, `dask@main`,
> `xarray@main`, `dask_array.xarray.register()`) accelerates **dask
> collections**. This DAG uses futures and needs none of it — which is as well,
> since those git-main pins are exactly what would break worker parity.

Set once per shell:

```bash
P=~/grib-index-kerchunk/icechunk-dask/icechunk-virtual-dask/.venv/bin/python
D=/opt/mamba/envs/dask/bin/python     # only for deploy_workers.py
```

`deploy_workers.py` runs under `$D` because it talks to the **Dask** cluster,
which is used purely as a remote-exec channel (there is no SSH key on the
gateway).

---

## 3. Credentials

Two sets, and keeping them apart is the single most error-prone part.

| | store | credentials |
|---|---|---|
| **source** (read) | `source.coop` → virtual chunks on AWS | **anonymous**, `from_env=False` |
| **sink** (write) | EWC Ceph, `s3://must-icechunk` | `AK`/`SK` from `.env` |

`.env` (gitignored, alongside these scripts):

```
AK=<ewc access key>
SK=<ewc secret key>
```

### How the workers get write permission

**They are never given the keys, and never read `.env`.**

The client opens the sink with explicit `access_key_id` / `secret_access_key`,
then calls `session.fork()`. The resulting `ForkSession` carries its storage
configuration — including those credentials — and is **pickled to the worker**
as a task argument. The worker writes through `fork.store` and returns the
changeset. Nothing else is distributed.

This matters because of the `AWS_*` trap (`dev-test/README.md` §2): the Dask workers
carry `AWS_ENDPOINT_URL` and `AWS_DEFAULT_REGION` for Ceph, and if the
icechunk **virtual-chunk** client sees those it tries to resolve a hostname
that does not exist. So:

- `frisky_daily_dag.py` pops every `AWS_*` **at import**, before anything
  builds an S3 client — the config is cached process-wide, so popping later
  does not help.
- `deploy_workers.py` launches each `frisky worker` with `AWS_*` scrubbed from
  **that child process only**. The Dask worker keeps its own environment
  untouched, so no other job on the VM is affected.
- The Ceph write therefore *cannot* use env vars — it uses the credentials
  embedded in the pickled `ForkSession`. The two paths never contend.

---

## 4. A-to-Z runbook

### Step 1 — build the venv on all six VMs (~40 s, once)

```bash
$D deploy_workers.py --install
```

Creates `/tmp/frisky-ea/.venv` on each VM from the pinned interpreter,
installing nothing into `/opt/mamba/envs/dask`. Also ships
`frisky_daily_dag.py`, which the workers need because **Frisky pickles task
functions by reference** — the defining module must be importable there.

### Step 2 — scheduler on a worker VM, then the workers

```bash
$D deploy_workers.py --scheduler-on 192.168.1.74 --start \
     --workers-per-vm 2 --nthreads 8 --memory-limit 6GB
```

One command does both, in order. What each flag is for:

| flag | why |
|---|---|
| `--scheduler-on 192.168.1.74` | **not** the gateway. The JupyterHub session is capped at 8 GiB and the scheduler was OOM-killed there mid-run; the client saw only `Failed to send: channel closed` and the store kept nothing but its initial commit. A worker VM has 15 GB. Also sets `FRISKY_TRACING_CAPACITY=20000`, since the default 200k span buffer is what grew across runs. |
| `--workers-per-vm 2` | icechunk caps in-flight connections **per process**: one process tops out at ~25–31 MB/s, two reach ~47. Threads cannot substitute. `SCALING.md` §2. |
| `--nthreads 8` | 12 × 8 = **96 concurrent readers**. 192 drew `SlowDown` from AWS and failed 56 of 1,590 blocks. |
| `--memory-limit 6GB` | 2 workers × 6 GB fits the 15 GB VM. Each holds a pinned ~332 MB store session. |

`--start` always kills existing workers first. Skipping that once left 12
workers registered, six of them running a stale copy of the task module.

Check and tear down:

```bash
$D deploy_workers.py --status
$D deploy_workers.py --stop      # Dask workers untouched
```

### Step 3 — dashboard (optional)

```bash
setsid nohup $P dashboard_forward.py --to 192.168.1.74:8791 --listen 8791 \
    > dashboard_forward.log 2>&1 < /dev/null &
```

Then `https://<hub>/user/<user>/proxy/8791/` — **trailing slash, no `/status`**
(that is a Dask path; Frisky serves at `/`).

The relay is needed because jupyter-server-proxy's bare `/proxy/<port>/` route
always targets `127.0.0.1`, while the scheduler is bound to the LAN address the
workers must reach; its `/proxy/<host>:<port>/` route is gated by
`host_allowlist`, which defaults to localhost. Raw TCP, so the dashboard's
WebSocket upgrade passes through.

No proxying needed for the terminal view:

```bash
$P -m frisky observe overview http://192.168.1.74:8791
```

### Step 4 — one day

```bash
$P parallel_write.py --sink ea-cgan/v2-7day --date 2026-07-02 \
     --channels 30 --members 51 --steps 53 --fork-once --env .env
```

`--fork-once` forks a single session and pickles it to every task. Forking per
block costs ~0.30 s each — 476 s of the 494 s a date took before.

Expect ~340 s, ~1,590 blocks, one commit.

### Step 5 — a month

```bash
$P build_corpus.py --sink ea-cgan/v3-june2026 \
     --start 2026-06-01 --days 30 --env .env
```

Preallocates the whole `time` axis, so every write is a region write and dates
are independent. One commit per date. Add `--resume` to skip dates already in
the commit log — a date costs 81,090 messages and ~64 GB, so it should never be
redone because a later one failed.

Failed blocks are resubmitted over 4 rounds with a 45 s pause. The pause is the
point: resubmitting into an active throttling window just burns the retries.

### Step 6 — verify

```bash
$P check_chunks.py   --prefix ea-cgan/v3-june2026 --env .env   # completeness
$P check_members.py  --prefix ea-cgan/v3-june2026 --env .env   # ensemble is real
$P check_complete.py --prefix ea-cgan/v2-7day     --env .env   # single date only
```

**Do not trust "0 failed blocks".** `create_schema` pre-fills with zeros, so a
write that never happened is indistinguishable from data: no error, right
shape, right dtype, finite fraction 1.000.

- `check_chunks.py` reads the **manifest** (~17 MB) and lists what was actually
  written. Use this on a corpus.
- `check_complete.py` reads every block. Right for one date (7.8 GB), hopeless
  for 30 (233 GB).
- `check_members.py` catches a `number` selection that silently broadcast, by
  checking that ensemble **spread grows with lead time** — physics a broadcast
  bug cannot fake.

Expected on June 2026:

```
47,700 / 47,700 chunks, every channel 1,590, none missing
gh500 spread 1.3073 -> 5.9305 from +0h to +168h
VERDICT: members are REAL
```

---

## 5. Reading the result

```python
import icechunk as ic, xarray as xr
st = ic.s3_storage(bucket="must-icechunk", prefix="ea-cgan/v3-june2026",
                   region="RegionOne",
                   endpoint_url="https://object-store.os-api.cci1.ecmwf.int",
                   access_key_id=AK, secret_access_key=SK,
                   force_path_style=True, from_env=False)
ds = xr.open_zarr(ic.Repository.open(st).readonly_session("main").store,
                  consolidated=False, zarr_format=3, decode_timedelta=True)
# (time: 30, step: 53, number: 51, latitude: 163, longitude: 147) x 30 channels
```

Chunks are `(1, 1, number, lat, lon)` — one chunk per (date, step), 4.9 MB.

---

## 6. Files

| file | |
|---|---|
| `frisky_daily_dag.py` | task functions: `read_message`, `write_block`, the channel table. Shipped to workers |
| `parallel_write.py` | one date via fork/merge |
| `build_corpus.py` | many dates, one commit each, resumable |
| `sink_icechunk.py` | schema creation, merge+commit. Client only |
| `deploy_workers.py` | install / scheduler / start / status / stop / bandwidth |
| `check_chunks.py` | completeness from the manifest |
| `check_complete.py` | completeness by reading blocks |
| `check_members.py` | is the ensemble dimension real |
| `bandwidth_probe.py` | throughput vs thread concurrency |
| `procs_probe.py` | throughput vs **process** count — the one that found the real ceiling |
| `dashboard_forward.py` | TCP relay so jupyter-server-proxy can serve the dashboard |

---

## 7. Corrections

Claims made in this repo and later disproven, kept so they are not re-litigated:

1. **"Compression is 77.2%, the 40% assumption was badly wrong."** Wrong. That
   measured `savez_compressed` over mean/sd, not the store. Measured three
   times through icechunk on real payloads: 39.8%, 40.6%. The assumption held.
2. **"The limit is bandwidth, not in-flight count; ~36 VMs for 1 GB/s."**
   Wrong. Raising threads in one process cannot distinguish a saturated network
   from a per-process connection cap. It was the latter.
3. **"~20 VMs for 1 GB/s."** Also wrong, for a different reason: it counted
   only the per-VM ceiling and ignored the shared AWS request-rate quota, which
   already refuses at 96 concurrent readers.
4. **"0 failed blocks means the date is complete."** No — the zero-fill hides
   missing writes. Hence `check_chunks.py`.
5. **A retry path that has never executed is not a working retry path.** The
   `SlowDown` backoff went untested through every run until 192 concurrent
   readers provoked AWS, and then had two bugs: the store open sat outside the
   retry, and 6 attempts with no jitter meant every reader retried in lockstep.
