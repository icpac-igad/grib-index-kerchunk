# EWC realization: from GRIB on AWS to an 18 TB realized ensemble, on plain VMs

This folder holds the method, code and records for turning ECMWF IFS ensemble
forecasts into an analysis-ready dataset for East Africa and the SW Indian
Ocean. The dataset is all 51 raw members, 0–240 h, 39 channels, built both as
a ~18 TB multi-season corpus and as a daily operational run. It runs on the
**ECMWF Weather Cloud (EWC)**, on a Dask/frisky cluster made of **plain VMs,
with no Kubernetes**.

```
 GRIB on AWS ──① pars──▶ GIK parquets ──② append──▶ virtual Icechunk (GCS) ──mirror ②──▶ source.coop
 s3://ecmwf-forecasts   (Lithops, ../lithops-   (../icechunk-par/)                         (anonymous read)
                         cr-gik-ecmwf/)                                                          │
                                                                                                 ▼
          source.coop ◀──mirror ③── EWC Ceph  s3://must-icechunk/ea-swio/… ◀──③ realize── 6 EWC VMs, frisky
          ecmwf-ifs-ea-swio-realized-v3/          (this folder)                              one task = one GRIB message
```

## The method, in five steps

### 1. GIK parquet references (stage ①)

The **Grib-Index-Kerchunk** method reads only the small `.index` sidecar that
ECMWF publishes beside each GRIB file. It combines that with a per-era
template, and writes parquets of `[url, offset, length]` for every message
(51 members × 85 steps per run). No GRIB is downloaded. This runs serverless
on Lithops/Cloud Run, one container per date: see
[`../lithops-cr-gik-ecmwf/`](../lithops-cr-gik-ecmwf/) and the repository
[`README`](../../README.md).

### 2. The virtual Icechunk store (stage ②)

The references are appended into one **virtual** Icechunk store
(`ecmwf-ens-v4` on GCS). It has one group per era and cycle, and grows by one
`time` slot per date. Every chunk is still a byte range in
`s3://ecmwf-forecasts`, so the whole archive is about 400 k small objects, not
terabytes. It is mirrored to source.coop for anonymous reading
([`../icechunk-par/`](../icechunk-par/)).

### 3. A Dask/frisky cluster on plain VMs, without Kubernetes

**This is the step that made everything after it possible.** The working
assumption had been that Dask workers in the cloud come from Kubernetes:
JupyterHub on GKE, dask-gateway or dask-kubernetes, Helm, ArgoCD. ICPAC's 2025
attempt went that way (`cno-e4drr/jh-k8s`, private). It needed a GKE cluster,
ArgoCD, `kubectl` auth plugins and service accounts. It stalled at cluster
creation, on a service-account precondition, before any Dask work ran.

On EWC the cluster was built instead from **ordinary OpenStack VMs**, in two
days (2026-07-27/28):

| piece | how | Kubernetes equivalent it replaces |
|---|---|---|
| Dask scheduler | one `systemd` unit on the gateway VM: `dask scheduler --port 8786` | scheduler pod and service |
| 6 workers | `openstack server create` × 6 with a **cloud-init** file that installs the pinned env and starts `dask worker` pointing at the scheduler | worker pods / autoscaler |
| networking | one security group (`dask-internal`) between the VMs; workers have no public IP; only the gateway is reachable | cluster network and ingress |
| environment | the same micromamba spec on every VM (`/opt/mamba/envs/dask`) | the container image |
| dashboard | `jupyter-server-proxy` through the JupyterHub login | ingress plus auth |

On top of that, the realization itself moved from a Dask collection graph to
**frisky** futures ([`DAG_VERSIONS.md`](DAG_VERSIONS.md) §0–1). A Dask graph
over the whole store reached millions of tasks and hung in the client. Frisky
submits one task per GRIB message (~126 k per date) at ~3 µs overhead each.

- The **frisky scheduler** runs on a worker VM; the gateway's 8 GiB cgroup
  OOM-killed it there.
- **Frisky workers** are started on every VM by `deploy_workers.py`, through
  the existing Dask workers' `client.run()`. So Dask stays as the management
  channel: there is no SSH key on the gateway, by design.

The result is a scheduler on one VM and the other VMs as workers. It is a
simple pattern, and it has none of Kubernetes' moving parts. It carried the
243-hour corpus build and has run the daily routine since 2026-10-01. Build steps: [`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) §2 and §5.

### 4. Realizing the ~18 TB corpus (stage ③, long-term)

The cluster read the virtual store, decoded each global GRIB message with
gribberish, cut the 321 × 261 box, and wrote one Icechunk store per season to
EWC Ceph:

- **1,260 dates, 16 seasons**, March 2023 → 2026, across three ECMWF eras;
- **243 cluster-hours**, ~103 TB read from AWS, **~17.9 TB written**;
- 127 MB/s sustained.

Each season was verified, mirrored to source.coop, and only then deleted from
Ceph. See [`02_LONG_TERM_REALIZATION.md`](02_LONG_TERM_REALIZATION.md) and
[`04_VERIFY_AND_PUBLISH.md`](04_VERIFY_AND_PUBLISH.md).

### 5. The daily run, and what it ran into

Since 2026-10-01 each new 00z forecast goes through ① → ② → mirror ② → ③ →
mirror ③ (and on to the CRMA layers) in ~72–80 min on a clean day:
[`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md). The issues of the first
fifteen days are recorded in [`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md)
and [`stats/`](stats/DAILY_RUNS_2026-10-01_to_10-15.md). In short:

- **AWS throttling on the shared NAT IP** stalled 80 readers completely, so
  the daily run uses 24 (robust, but 4–5× slower). The 10-11 → 10-12
  throttling (974 retries) cleared by itself.
- **Stale per-process store caches** broke every block until workers were
  restarted after each mirror. This became RULE 1.
- **Credentials:** the Ceph application credential lapsed, and the source.coop
  token expired (fixed by its refresh token from 10-13).
- **Gateway disk:** 76 GB, and the 2.5 GB gate was nearly hit.
- Silent source.coop chunk stalls in the downstream layers.

The EWC location is part of the picture too. Compute and Ceph are in Bologna,
but the data is read from AWS Frankfurt; the trade-offs, and running the same
job on GCP Spot or Google Colab, are in
[`06_ALTERNATIVES_GCP_SPOT_AND_COLAB.md`](06_ALTERNATIVES_GCP_SPOT_AND_COLAB.md).

## Read in this order

| | document | for |
|---|---|---|
| 1 | [`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) | hosts, VMs, Ceph, the three credential sets, environments, bringing up workers, the smoke test |
| 2 | [`02_LONG_TERM_REALIZATION.md`](02_LONG_TERM_REALIZATION.md) | the ~18 TB, 1,260-date, 243 h multi-season corpus: how it was driven, what it cost, how to size the next one |
| 3 | [`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md) | the daily routine, one forecast from pars to source.coop: commands, gates, rules, what to do when it stops |
| 4 | [`04_VERIFY_AND_PUBLISH.md`](04_VERIFY_AND_PUBLISH.md) | manifest/physics checks, the two mirrors, anonymous verification, deleting from Ceph |
| 5 | [`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) | what limits throughput (with the measured numbers), why the daily run uses 24 readers, every worker issue and fix, open items |
| 6 | [`06_ALTERNATIVES_GCP_SPOT_AND_COLAB.md`](06_ALTERNATIVES_GCP_SPOT_AND_COLAB.md) | running the daily job without EWC: GCP Spot layouts, Google Colab (CLI + compute units) as the daily routine, failure and restart, costs |
| — | [`stats/DAILY_RUNS_2026-10-01_to_10-15.md`](stats/DAILY_RUNS_2026-10-01_to_10-15.md) | per-date timings, retries and failures for the first 15 operational days ([CSV](stats/daily_runs_2026-10.csv)) |

## Numbers at a glance

| | |
|---|---|
| per date | ~126 k GRIB messages (39 ch × 51 members × 65 steps) → 2,535 blocks → **~17 GB** |
| long-term corpus | **1,260 dates, 243.0 cluster-h, ~103 TB read → ~17.9 TB written**, 127 MB/s sustained at 48 readers |
| daily, clean day | realize **39–55 min** at 24 readers; whole chain 72–78 min |
| daily, throttled day | realize ~110 min, with 600–1,000 retries |
| highest EWC bandwidth measured | 216.5 MB/s (1.73 Gbps) |

## Reference documents

| document | what |
|---|---|
| [`RUNBOOK_REALIZE.md`](RUNBOOK_REALIZE.md) | the step-by-step runbook for realizing one season |
| [`REALIZE_REFERENCE.md`](REALIZE_REFERENCE.md) | stage ③ reference: domain, channels, credentials, cluster flags |
| [`REALIZATION_COST.md`](REALIZATION_COST.md) · [`REALIZATION_CONCURRENCY.md`](REALIZATION_CONCURRENCY.md) | the long run's cost and the reader-concurrency analysis |
| [`DAILY_OPERATION_COST.md`](DAILY_OPERATION_COST.md) · [`DAG_VERSIONS.md`](DAG_VERSIONS.md) | cost of one daily date; DAG v1 vs the proposed v2 |
| [`SCALING.md`](SCALING.md) · [`EWC_REALIZE_NOTES.md`](EWC_REALIZE_NOTES.md) | per-process connection cap, probes; earlier realize notes |
| [`CLUSTER_RESTART_ROUTINE.md`](CLUSTER_RESTART_ROUTINE.md) · [`MIRROR_RUN_ARCHITECTURE.md`](MIRROR_RUN_ARCHITECTURE.md) · [`SOURCE_COOP_CREDENTIALS.md`](SOURCE_COOP_CREDENTIALS.md) | operations how-tos |
| [`OPENING_PUBLISHED_ECMWF_ICECHUNK.md`](OPENING_PUBLISHED_ECMWF_ICECHUNK.md) · [`USING_REALIZED_EA_SWIO_STORE.ipynb`](USING_REALIZED_EA_SWIO_STORE.ipynb) | for users of the published stores |

## Scripts

| script | role |
|---|---|
| `frisky_daily_dag.py` | the DAG: domain, the 39-channel table, `read_message`, `write_block`. **Workers hold a copy; redeploy after any edit** |
| `build_corpus.py` | realize a date window into a season store (`--resume`, `--repair`) |
| `sink_icechunk.py` · `parallel_write.py` | Ceph sink; the parallel writer |
| `append_channels.py` | add channels to an existing season (the `d2m` fragments) |
| `deploy_workers.py` · `dashboard_forward.py` · `cluster_watchdog.sh` | worker lifecycle; dashboard relay; watchdog |
| `run_corpus_supervised.sh` · `run_all_seasons.sh` | supervised season build; season sequence |
| `check_chunks.py` · `check_complete.py` · `check_members.py` · `check_geolocation.py` · `inspect_store.py` · `validate_domain.py` | verification |
| `mirror_gcs_to_source_coop.py` · `mirror_ceph_to_source_coop.py` · `mirror_all_to_source_coop.sh` | mirror ② and mirror ③ |
| `bandwidth_probe.py` · `procs_probe.py` | throughput probes |
| `purge_wrong_stores.py` | delete stores built on the wrong region |
| `smoke_test_published_ecmwf.py` | anonymous smoke test of the published virtual store |

Shell scripts `cd` into this folder, and take the client interpreter from
`P` (default `.venv/bin/python`). Python scripts read Ceph keys from the
gitignored `.env` here.

## Where the live copy runs

Day-to-day, these scripts run from the crma repo
(`github.com/nishadhka/crma`), in two checkouts on the gateway:

- **The driver and the run record:** `crma/`, branch
  `crma-one-scale-one-trigger` (latest `4fb83cd`, 2026-10-16). It holds
  `medium-range-forecast/3-realize-frisky/daily_run.sh`, the per-day pages
  `RUNBOOK_V4_YYYYMMDD.md`, and the stage logs in `logs/`. Its own copy of the
  realize scripts is older (36 channels) and is **not** what runs.
- **The executed stage ①–③ scripts:** `refill-crma-routine/`, branch
  `realize39channel-upto-sep2026` @ `420c096`. `daily_run.sh` calls them
  through `$REFILL`.

**This folder's scripts were synced from the `refill-crma-routine` copy at
`420c096` on 2026-10-15.** `deploy_workers.py` additionally carries
this repo's stage-2 package pins. If you change a script here, port the
change there too (or the reverse) until the daily run is pointed at this
folder.
