# Bottlenecks, throughput, and the worker issues

What limits the realization, what has been measured, and what broke on the
Dask and frisky workers. The detailed evidence is in the documents linked
from each section. This page is the summary, with the numbers in one place.

## 0. The figures to quote, and the ones not to

| claim | what is measured | source |
|---|---|---|
| "18 TB / ~200 h" | **887 dates, 192.4 cluster-hours, 88.0 TB read → 14.9 TB written**. Extended record: **1,260 dates, 243.0 h, ~17.9 TB written** | [`REALIZATION_COST.md`](REALIZATION_COST.md) §1–2 |
| long-run throughput | **127 MB/s sustained** aggregate (~21 MB/s per VM), 161–166 msg/s | `REALIZATION_COST.md` §2 |
| peak EWC → AWS bandwidth | **216.5 MB/s (1.73 Gbps)**, 2 procs × 8 threads × 5 VMs, 0 errors (2026-10-04) | [`REALIZATION_CONCURRENCY.md`](REALIZATION_CONCURRENCY.md) §4a |
| source.coop read path | **~150–180 MB/s** shared ceiling (1→3 procs/VM: 148 → 164 → 179 MB/s) | crma `4-bn-preprocess/COMPUTE_COST.md` §5 |
| mirror ③, Ceph → source.coop | **300–430 GB/h** (83–120 MB/s), one host | [`stats/`](stats/DAILY_RUNS_2026-10-01_to_10-15.md) |
| "~600 MB/s" | **No measurement supports this.** It appears in none of the logs or docs. The highest EWC figure ever measured is 216.5 MB/s. | — |

The 5.9× read amplification comes from the domain. A GRIB message is
global (1,038,240 points), and the EA/SW-IO box keeps 83,781 of them, about
8%. Every byte read is fetched for the whole globe. That is the price of
reading the original archive, and no worker setting changes it.

## 1. The long-term realization: where its 200 hours went

Configuration that actually ran: 6 VMs × 2 workers × **4** threads =
**48 readers**. `--nthreads` was left unset and defaulted to 4, although the
runbook intended 8 (96 readers).

- **Starvation, not saturation.** Per-reader rate was flat at ~3.46 msg/s, so
  48 readers were under-feeding the pipe, not hitting a wall
  (`REALIZATION_CONCURRENCY.md` §3).
- **Per-process connection cap.** Within one process, icechunk saturates at
  ~25–31 MB/s. Two processes per VM reach ~50 MB/s, and more add nothing
  (`procs_probe.py`: 25.0 / 46.8 / 50.4 / 48.2 MB/s at 1/2/4/8 procs). Threads
  can't substitute for processes ([`SCALING.md`](SCALING.md) §1).
- **Cold cache, not a ceiling.** The apparent ~36 MB/s-per-VM asymptote was a
  cold per-process manifest and connection state. Warm, it reaches 2.56× more
  (140 → 359 msg/s, ~57 MB/s per VM). [`DAG_VERSIONS.md`](DAG_VERSIONS.md) §4
  proposes a DAG v2 that warms the read path first.
- **Workers lose throughput with age**, ~25% over a long run, and get it back
  on restart, hence the 12 h restart cadence. The heuristic is: a steady
  decline with 0 retries points to the cluster; a drop with retries that heals
  itself points to the source.
- **Refuted ideas.** Hot-keying a single GRIB object is not a problem
  (concentrated reads were 15% *faster*). The shared EWC network path is not
  the ceiling (216.5 MB/s measured).
- **Diminishing returns past 100 readers** on a realistic 38-channel read:

  | readers | msg/s | MB/s |
  |---|---|---|
  | 50 | 113 | 89 |
  | 100 | 143 | 112 |
  | 200 | 207 | 163 |
  | 400 | 227 | 179 |

- **Projections [est].** 96 readers would have taken ~139–191 h for the
  243 h corpus, and ~200 readers ~135 h. More VMs is the wrong lever
  (`REALIZATION_CONCURRENCY.md` §5). Doubling readers buys ~1.3×, not 2×.

## 2. The daily run: why 24 readers

| readers | outcome |
|---|---|
| 96 / 80 | 640–750 s/date on a quiet archive (10-01, 10-04), but **total failure** on a busy same-day one (10-05: four passes stalled at `0 B/s`) |
| **24** | Succeeded on every one of 10 runs (10-05 → 10-14). 2,350–3,800 s on a normal day, 6,500–6,600 s on throttled days |

The 10-05 diagnosis: AWS returned `SlowDown` 503s on `s3://ecmwf-forecasts`
to the workers' **single shared NAT egress IP** (4 of 5 workers got 503s
while the gateway got 206s). A same-day archive is under heavy load from
everyone downloading the new forecast, and 80 readers from one IP crossed
the per-requester limit.

**Unsettled.** The 24-reader choice was argued as being "under the 48-reader
knee". A later commit (`3db93fe` in crma) concluded that 48 "was an
accident", which removes that argument even though the observation stands.
Not yet measured: 24 readers on a quiet archive, any point between 24 and 80,
and whether backoff on the GRIB read path (none exists today, see §4) would
let 80 readers survive a busy day. **That experiment is the biggest available
win, worth up to ~4× on stage ③.**

## 3. The 10-11 → 10-13 slowdown (transient, cause unconfirmed)

| | 10-10 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 |
|---|---|---|---|---|---|---|
| pars (min) | 6.7 | 18.4 | 23.1 | 6.0 | 5.0 | — |
| throttle probe | 10/10 | 6/10 | 4/10 | 5/10 | 8/10 | 9/10 |
| stage ③ (s) | 2,353 | 6,497 | 6,621 | 3,809 | 3,129 | 2,821 |
| stage ③ retries | 22 | 613 | 974 | 162 | 124 | **15** |
| L2→3 (s) | 109 | 105 | **404** | **949** | 102 | 88 |

It cleared on its own by 10-15, with no configuration change.

Pars, probe and stage ③ all read `s3://ecmwf-forecasts`, and they degrade
together. Layer 2→3 reads source.coop, a *different* provider, and it was
slow on different days. Three hypotheses were offered: ECMWF load, the shared
NAT IP, and silent source.coop chunk stalls. None was confirmed for the AWS
side. For the L2→3 times, the documented account is silent source.coop stalls:
some chunk requests trickle at a few kB/s without erroring (one 7.5 MB chunk
took 174.5 s), and `_with_timeout` re-issues the unit after 600 s without
logging it (crma `RUNBOOK_V4_20261012.md` §3). Ceph was
checked and was healthy (0.17 s list, 17–20 MB/s). **Next step:** log every
re-issue in the `_with_timeout` path, so the next slow day can be measured
instead of explained after the fact.

## 4. Worker issues and their fixes

Full write-up: [`history/WORKER_ISSUES_20261002.md`](history/WORKER_ISSUES_20261002.md).

| issue | symptom | fix |
|---|---|---|
| Stale per-process store cache (`_open_era`, `open_store_cached`) | `KeyError: "not all values found in index 'time'"` on every block | **RULE 1**: restart frisky after mirror ②, and Dask after mirror ③. Built into the daily driver |
| Too many readers against a busy archive | `error fetching virtual reference … 0 B/s`, 503 SlowDown | 24 readers (`--workers-per-vm 1 --nthreads 4 --memory-limit 12GB`), plus the read and throttle gate |
| No retry/backoff on the virtual-chunk GRIB read | each throttled read fails a block | only block-level `--rounds 6 --pause 120`. The icechunk `RepositoryConfig` retry settings are proposed, **not done** |
| `AWS_*` env collision | sourcing source.coop creds aims the Ceph sink at source.coop, and `AWS_ENDPOINT_URL` breaks the virtual-chunk client | `unset AWS_*` before stage ③. `frisky_daily_dag.py` pops `AWS_*` at import, and workers start with them scrubbed |
| `open_sink` bare `except` → `Repository.create` | a credential error turned into an attempt to create a new repository | `63fedbc`: `create=False` on repair |
| `deploy_workers.py --status` truncates the log to ~70 chars | hours of misdiagnosis on 10-05 | read the full worker log via `client.run(...)` |
| VM drops after `Client.restart()` | `CommClosedError` in `register_worker_plugin` | retry on 5 workers; `.165` rejoined later. There is no SSH key on the gateway by design, so `client.run()` is the only remote-exec channel |
| `upload_file` imports modules on arrival | an unordered upload list kills worker startup | upload in dependency order, with the driver module last |
| Positional write index | the `--start/--days` window and the dates-file order *are* the sink slot index, so a wrong window or out-of-order dates silently overwrite slots | always `--start 2026-09-01 --days 122` for SOND 2026, and run dates in order |
| `--repair` header overstated the work | alarming "223,080 chunks" | `8412ddf` prints `TO BUILD: 2,535 chunk(s) across 1 date(s)` |
| Environment gaps | `boto3` missing from the stage-3 venv and the Dask env; `gribberish` missing from a PEP-723 header; `uv`, `gcloud`, `source-coop` missing on the gateway | see [`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) |
| Version skew between client and workers | msgpack / python / tornado patch-level warnings | recorded; nothing traced to it |
| Scheduler on the gateway OOM-killed | client sees only `channel closed` | run the frisky scheduler on a worker VM (`--scheduler-on 192.168.1.164`) |

## 5. Open items (2026-10-16)

1. Find the 24 ↔ 80 reader trade-off on quiet and busy days, ideally with GRIB-read backoff added first (§2).
2. Instrument `_with_timeout` re-issues (§3).
3. Add retry/backoff on the virtual-chunk read path (§4).
4. Mirror ②: incremental listing (~9 min/day of listing for 7–18 s of upload).
5. DAG v2 (warm read path) for the next long-term run ([`DAG_VERSIONS.md`](DAG_VERSIONS.md) §4).
6. Gateway disk headroom: 2.5–3.9 GB free of 76 GB; the gate threshold is 2,500 MB.
7. The GCP Spot alternative (`CRMA_SINK=gcs`, cno-e4drr `devops/crma-gik-spot-vm`) is validated but has never been applied. Estimate: ~2–2.5 h per date on 2 × e2-standard-2 [est].
