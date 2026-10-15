# Setup from scratch: the EWC tenancy, VMs, Ceph, environments, credentials

Everything that has to exist before [`02_LONG_TERM_REALIZATION.md`](02_LONG_TERM_REALIZATION.md)
or [`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md) can run. Sections 1–2
describe the setup as it stands. Wherever a provisioning step was never
written down, this page says so; nobody has rebuilt the cluster from zero yet.

## 1. What runs where

| role | host | spec | runs |
|---|---|---|---|
| **gateway** (driver) | `jupyterhub-gateway`, `192.168.1.129` | 8 cores, 31 GB RAM, **76 GB disk** (keep ≥ 2.5 GB free) | the Dask scheduler `:8786` (remote-exec channel only), the frisky client, the mirrors, Lithops launch |
| **workers** ×6 | `dask-worker-01..06`: `192.168.1.10, .99, .164, .165, .230, .248` | flavour `4cpu-16gbmem-40gbdisk` | one Dask worker each (`/opt/mamba/envs/dask`), plus frisky workers started by `deploy_workers.py` |
| **frisky scheduler** | on worker `192.168.1.164` | — | `:8796`, dashboard `:8791` |
| **Ceph object store** | `https://object-store.os-api.cci1.ecmwf.int` (RegionOne) | bucket `must-icechunk` | the realized stores (`ea-swio/…`) and CRMA Layer 3 |

Networking: the VMs are **outbound-only through router NAT**, and all of them
share **one public egress IP**. That matters for AWS throttling (see
[`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) §2). There is
**no SSH key on the gateway, by design**: `deploy_workers.py` reaches the VMs
through Dask `client.run()`, so the Dask scheduler on the gateway is the only
management channel. If the gateway is retired, re-point the VMs' Dask workers
first ([`history/HANDOVER_NEW_GATEWAY.md`](history/HANDOVER_NEW_GATEWAY.md)).

## 2. Provisioning the tenancy (EWC / OpenStack)

**Gap: this was done by hand and never recorded.** What is known:

- Six instances of flavour `4cpu-16gbmem-40gbdisk` on the tenancy's private
  network `192.168.1.0/24`, behind a router with NAT, with no floating IPs on
  the workers.
- `/opt/mamba/envs/dask` on every worker, built from the same spec as the
  gateway (§4). Dask workers dialled to `tcp://192.168.1.129:8786`, spilling to
  `/scratch/dask`.
- A Ceph S3 bucket `must-icechunk`, with EC2-style access keys created from an
  **OpenStack application credential** (§3).

To fill this gap, record the `openstack server create` / image / network /
security-group commands the next time a VM is rebuilt.
`history/EWC_USAGE_AND_RESOURCE_PLAN.md` holds the original allocation request.
Its ~4.4 TB storage figure predates the long-term run; the real figure is about
17.9 TB.

## 3. Credentials (three sets, never mixed)

| what | file (gitignored) | lifetime | used by |
|---|---|---|---|
| **EWC Ceph** `AK`/`SK` | `.env` in this folder | as long as the **OpenStack application credential** behind it | the realize sink, mirror ③ source, check scripts |
| **source.coop** STS (`AWS_*` ×5) | the same `.env` | refreshed from a 30-day token (`source-coop` CLI ≥ v0.3.0) | mirror ② and mirror ③ destination |
| **GCP service account** | `../lithops-cr-gik-ecmwf/service_account/ecmwf-lithops-deployer-key.json` | long-lived | Lithops pars, the virtual store on GCS |

- **Ceph failure mode.** When the application credential lapses, the error is
  `InvalidAccessKeyId`, even though the key itself is fine (2026-10-02). Renew
  the credential, not the key.
- **The source.coop CLI omits two variables.** `AWS_ENDPOINT_URL` and
  `AWS_DEFAULT_REGION` must be kept when you merge in new credentials. See
  [`SOURCE_COOP_CREDENTIALS.md`](SOURCE_COOP_CREDENTIALS.md) and
  [`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md) §1.
- **Workers are never given keys.** The client opens the Ceph sink, calls
  `session.fork()`, and pickles the fork to the workers.
- **Reads are anonymous.** The source (source.coop, then virtual chunks on
  `s3://ecmwf-forecasts`) is read with `from_env=False`.
- **The `AWS_*` trap.** If `AWS_ENDPOINT_URL` is set, the virtual-chunk client
  resolves a host that does not exist, and the sink can be aimed at the wrong
  endpoint. `frisky_daily_dag.py` pops `AWS_*` at import, and
  `deploy_workers.py` scrubs them from each worker. **`unset AWS_*` before
  stage ③.**
- **`chmod 600 .env`, and keep no `.env` backups.** A `.env.bak-*` once
  escaped the ignore rule.
- **Direct boto3 calls to Ceph** need
  `export AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`. Without it, the
  missing ECMWF CA shows up as a credentials error.

## 4. Environments

**Rule: never `pip install` into `/opt/mamba/envs/dask`.** The workers'
frisky `.venv` symlinks to it, so installing there changes what the workers
run, and client/worker parity is what keeps pickled tasks working.

| env | where | for | build |
|---|---|---|---|
| `dask` (micromamba) | `/opt/mamba/envs/dask`, identical on gateway and workers | Dask workers, `deploy_workers.py` (`D=/opt/mamba/envs/dask/bin/python`) | spec in [`history/README.md`](history/README.md) §1: python 3.12, dask/distributed 2026.7.1, zarr 3.2.1, xarray 2026.7.0, numpy 2.5.1, + pip icechunk 2.1.1, gribberish 1.6.0 |
| frisky client `.venv` | this folder (`P=.venv/bin/python`) | `build_corpus.py`, mirrors, checks | see below |
| frisky worker venv | on each VM, built by `deploy_workers.py --install` | frisky workers | pinned list in `deploy_workers.py` |
| `uv` + Python 3.12 | gateway, `~/.local/bin` | stages ①–② (Lithops pars, virtual store) | `pip install --user uv` |
| `geo-venv` | `/home/ubuntu/geo-venv` | CRMA layers only (geopandas/pyproj) | not needed for ECMWF realization |

Client venv. It must be a real venv. On the gateway `python3 -m venv` has no
`ensurepip`, so build it from the mamba interpreter:

```bash
/opt/mamba/envs/dask/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install "frisky>=0.3.0" icechunk==2.1.1 zarr==3.2.1 \
    xarray==2026.7.0 gribberish==1.6.0 numpy==2.5.1 dask==2026.7.1 boto3
```

Pinned stack in operation (2026-10): frisky 0.7.2, icechunk 2.1.1, zarr 3.2.1,
xarray 2026.7.0, gribberish 1.6.0, dask 2026.7.1. Lithops needs local Python
3.12 to match its Cloud Run runtime. Known gaps that have bitten: `boto3` was
absent from both the client venv and the dask env. On the gateway, `uv`,
`gcloud` and `source-coop` must be installed by hand. **No gcloud SDK is
needed**: GCS is reached through the service-account key.

## 5. Bring up the workers

```bash
D=/opt/mamba/envs/dask/bin/python
$D deploy_workers.py --install                       # once per VM rebuild: worker venvs
$D deploy_workers.py --scheduler-on 192.168.1.164 --start \
     --workers-per-vm 1 --nthreads 4 --memory-limit 12GB    # daily shape: 24 readers
$D deploy_workers.py --status
$D deploy_workers.py --stop                          # Dask workers untouched
```

- **`--start` always kills existing workers first.** It also re-ships
  `frisky_daily_dag.py`, so run it after **any** edit to the domain or the
  channel list. Workers hold their own copy of that module.
- **Run the scheduler on a worker VM, not the gateway.** The gateway's cgroup
  OOM-killed it mid-run.
- The long-term shape was `--workers-per-vm 2 --nthreads 4` (48 readers).
  Whether 96 is safe is open; see
  [`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) §1–2.
- `--status` truncates each worker's log to ~70 characters. For a real error,
  read the full log through `client.run()`.

Dashboard: `dashboard_forward.py --to 192.168.1.164:8791 --listen 8791`, then
`https://<hub>/user/<user>/proxy/8791/`. Terminal alternative:
`.venv/bin/frisky observe overview http://192.168.1.164:8791`.

## 6. Smoke test, in this order

```bash
1. ip -4 addr                                                   # on 192.168.1.0/24?
2. curl -s http://192.168.1.164:8791/api/workers | head -c 200   # frisky up, worker count
3. timeout 4 bash -c "</dev/tcp/192.168.1.129/8786"              # Dask channel
4. $D deploy_workers.py --dask tcp://192.168.1.129:8786 --status # per-VM last log line
5. .venv/bin/python frisky_daily_dag.py --scheduler 192.168.1.164:8796 \
       --date <newest published date> --era 50r1 --vars d2m tp --members 4 --steps 2
```

Step 5 is the one that proves the whole path. It covers anonymous source.coop,
virtual chunks on AWS, GRIB decode on the workers, and the remote scheduler.
Finite numbers mean the cluster can realize that date.
[`history/run_20260301.md`](history/run_20260301.md) is a full A-to-Z record of
one real run.
