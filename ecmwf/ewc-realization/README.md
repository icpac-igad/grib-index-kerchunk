# EWC realization: virtual ECMWF ENS store → realized East Africa / SW Indian Ocean store

This folder turns the **virtual** ECMWF IFS ensemble Icechunk store into a
**realized** one. The virtual store holds GIK byte-range references into
`s3://ecmwf-forecasts`; the realized store holds decoded float32 arrays for
the EA/SW-IO box, all 51 raw members, 0–240 h. The work runs on the
**ECMWF Weather Cloud (EWC)**: six worker VMs run a [frisky](https://getfrisky.dev/)
futures DAG and write to EWC Ceph, and the results are mirrored to
source.coop.

```
 GRIB on AWS ──① pars──▶ GIK parquets ──② append──▶ virtual Icechunk (GCS) ──mirror ②──▶ source.coop
 s3://ecmwf-forecasts   (Lithops, ../lithops-   (../icechunk-par/)                         (anonymous read)
                         cr-gik-ecmwf/)                                                          │
                                                                                                 ▼
          source.coop ◀──mirror ③── EWC Ceph  s3://must-icechunk/ea-swio/… ◀──③ realize── 6 EWC VMs, frisky
          ecmwf-ifs-ea-swio-realized-v3/          (this folder)                              one task = one GRIB message
```

## Read in this order

| | document | for |
|---|---|---|
| 1 | [`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) | hosts, VMs, Ceph, the three credential sets, environments, bringing up workers, the smoke test |
| 2 | [`02_LONG_TERM_REALIZATION.md`](02_LONG_TERM_REALIZATION.md) | the ~18 TB, 1,260-date, 243 h multi-season corpus: how it was driven, what it cost, how to size the next one |
| 3 | [`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md) | the daily routine, one forecast from pars to source.coop: commands, gates, rules, what to do when it stops |
| 4 | [`04_VERIFY_AND_PUBLISH.md`](04_VERIFY_AND_PUBLISH.md) | manifest/physics checks, the two mirrors, anonymous verification, deleting from Ceph |
| 5 | [`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) | what limits throughput (with the measured numbers), why the daily run uses 24 readers, every worker issue and fix, open items |
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
| [`history/`](history/) | dated records: the dev-test era, the dask era (`history/dask-era/`), the season drivers, worker-issue and gateway-handover reports |

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
