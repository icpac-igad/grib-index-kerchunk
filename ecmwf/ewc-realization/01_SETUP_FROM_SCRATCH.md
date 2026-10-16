# Setup from scratch: the EWC tenancy, VMs, Ceph, environments, credentials

Everything that has to exist before [`02_LONG_TERM_REALIZATION.md`](02_LONG_TERM_REALIZATION.md)
or [`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md) can run. Sections 1–2
describe the setup as it stands and how it was built. The full build record
(JupyterHub gateways, TLS, users) is in the private repository
`cno-e4drr`, `devops/ecmwf-weather-cloud-jh/RUNBOOK.md`. §2 below keeps the
part this pipeline depends on: VMs, network, scheduler, workers.

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
management channel.

**Retiring or replacing the gateway:**

- First re-point the VMs' Dask workers, which dial `192.168.1.129:8786`, or
  stand up a Dask scheduler elsewhere and restart them against it. Otherwise
  the frisky workers can no longer be restarted.
- From another host, use `deploy_workers.py --dask tcp://192.168.1.129:8786`.
  The default (127.0.0.1) only works on the gateway itself.
- `/opt/mamba/envs/dask` sits on the gateway's local disk (`/dev/sda1`), not
  a shared mount, so a new gateway needs its own copy.
- Credentials travel as a zip. Extract it with `unzip` (not
  `python -m zipfile`) so file modes survive, then `chmod 600 .env`.
- A local smoke test without `--scheduler` proves nothing about the cluster;
  use §6 step 5.

## 2. Provisioning the tenancy (EWC / OpenStack)

Built 2026-07-27/28 with the OpenStack CLI and cloud-init only: no
Kubernetes, no images to build, no orchestration layer. The source is
`cno-e4drr/devops/ecmwf-weather-cloud-jh` (RUNBOOK §2, §4, §9; worker
cloud-init in `cloud-init-dask-worker.yaml`).

**a. CLI access.**
- EWC logs in through federated SSO, so create one **application credential**
  in Horizon (Identity → Application Credentials). Use "Download
  clouds.yaml", then strip `project_name`, `project_id`, `username` and
  `user_domain_name` from it.
- Leave Roles blank so the credential inherits compute, volume and network.
- Run `export OS_CLOUD=openstack`, then check with `openstack limits show
  --absolute` (not `token issue`, which prints a live token).
- Application credentials expire. EC2 (S3) keys minted from them do not.

**b. Private network for Dask.** `--remote-group` admits only members of the
same group:

```bash
openstack security group create dask-internal --description "Dask scheduler/worker (private only)"
openstack security group rule create --proto tcp --dst-port 8786 --remote-group dask-internal dask-internal
openstack security group rule create --proto tcp --dst-port 8787 --remote-group dask-internal dask-internal
openstack security group rule create --proto tcp --dst-port 1024:65535 --remote-group dask-internal dask-internal
```

- The high-port range is required: workers connect to each other directly.
  Without it, small jobs pass and large ones hang.
- The scheduler port has **no authentication**. So only the gateway gets a
  floating IP, and it exposes only 80/443.

**c. Scheduler: one systemd unit on the gateway.**

```bash
printf '[Unit]\nDescription=Dask scheduler\nAfter=network-online.target\n\n[Service]\nExecStart=/opt/mamba/envs/dask/bin/dask scheduler --host 0.0.0.0 --port 8786 --dashboard-address :8787\nRestart=always\nRestartSec=10\nLimitNOFILE=65535\n\n[Install]\nWantedBy=multi-user.target\n' \
  | sudo tee /etc/systemd/system/dask-scheduler.service
sudo systemctl daemon-reload && sudo systemctl enable --now dask-scheduler
```

**d. Workers: six VMs from one cloud-init file.**

```bash
sed 's|${SCHEDULER_IP}|192.168.1.129|g' cloud-init-dask-worker.yaml > /tmp/worker.yaml
for i in 01 02 03 04 05 06; do
  openstack server create --image "Ubuntu-22.04" --flavor "4cpu-16gbmem-40gbdisk" \
    --network <tenancy private network> --security-group dask-internal \
    --key-name <keypair> --user-data /tmp/worker.yaml dask-worker-$i
done
```

- The cloud-init file installs micromamba and the pinned `dask` env (§4). It
  then starts `dask worker` against the scheduler, sized at boot: ~4 threads
  per process, total memory minus 2 GB headroom.
- **Create one worker first** and wait for `bootstrap complete` in its
  console log, then create the other five. Otherwise a cloud-init bug fails
  six times over.
- Workers hold no state; rebuilding one takes ~2 min.

**e. Storage.** The Ceph S3 bucket `must-icechunk` (10 TiB quota), with EC2
access keys (§3).

**f. Frisky, on top.** Once the Dask workers are up, `deploy_workers.py`
(§5) uses them only as a remote-exec channel. It builds each VM's frisky venv
and starts the frisky scheduler (on a worker VM) and the frisky workers. The
realization itself never goes through Dask.

Not yet scripted: Terraform covers only the second JupyterHub gateway. The
workers and gateway 1 are CLI-built, as above.

**The original allocation request (2026-07-31), against what happened:**

| | requested / planned | actual |
|---|---|---|
| VMs | 6 × 4 vCPU × 16.77 GB (Dask memory limit 13.94 GB), "no larger workers" | the same six |
| `must-icechunk` storage | ~4.5 TB (or 289 GB if only mean+sd were stored) | ~17.9 TB (raw members, 38–39 channels) |
| corpus | 1,246 dates × 30 ch × 51 members × 53 steps = 91.5 M messages | 1,260 dates × 38–39 ch × 51 × 65 steps |
| read volume | ~60.6 TB (0.788 MB/msg at 0.25°, 0.307 at 0.4°), from the `.index` sidecars | ~103 TB |
| written | 7.37 TB raw, ~4.4 TB after zstd | ~17.9 TB |
| duration | 3.5–60 days, depending on the link | 243 cluster-h (8–10 days of compute) |
| order | calibrate 1 date → 50r1 → 49r1 (13-level) → 49r1 (9-level) → 0p4 | newest season first (§3 of [`02_LONG_TERM_REALIZATION.md`](02_LONG_TERM_REALIZATION.md)) |

Usage before the request was 3.38 GB, all synthetic.

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
| `dask` (micromamba) | `/opt/mamba/envs/dask`, identical on gateway and workers | Dask workers, `deploy_workers.py` (`D=/opt/mamba/envs/dask/bin/python`) | spec below |
| frisky client `.venv` | this folder (`P=.venv/bin/python`) | `build_corpus.py`, mirrors, checks | see below |
| frisky worker venv | on each VM, built by `deploy_workers.py --install` | frisky workers | pinned list in `deploy_workers.py` |
| `uv` + Python 3.12 | gateway, `~/.local/bin` | stages ①–② (Lithops pars, virtual store) | `pip install --user uv` |
| `geo-venv` | `/home/ubuntu/geo-venv` | CRMA layers only (geopandas/pyproj) | not needed for ECMWF realization |

The `dask` env, built identically on the gateway and every worker:

```bash
micromamba create -n dask -c conda-forge python=3.12 dask=2026.7.1 distributed=2026.7.1 \
  zarr=3.2.1 xarray=2026.7.0 numpy=2.5.1 pandas=3.0.3 s3fs netcdf4 matplotlib cloudpickle
micromamba run -n dask pip install icechunk==2.1.1 gribberish==1.6.0
```

Verified pins (client = all six workers): python 3.12.13, s3fs 2026.6.0,
netCDF4 1.7.4, cloudpickle 3.1.2, pandas 3.0.3. Parity matters because a
dask/distributed/zarr/cloudpickle mismatch shows up as an opaque
deserialization error inside a task, not as an `ImportError`. Rebuild workers
from this spec. Observed skew in 2026-10: worker python 3.12.14, msgpack
1.2.2, tornado 6.5.8 (nothing traced to it).

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
[`EWC_REALIZE_NOTES.md`](EWC_REALIZE_NOTES.md) has the measured figures for a
single real date.
