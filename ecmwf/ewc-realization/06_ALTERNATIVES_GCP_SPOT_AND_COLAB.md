# Alternatives to EWC: GCP Spot, and Google Colab as the daily routine

The daily realization (stage ③ and mirror ③) runs today on six EWC VMs
writing to EWC Ceph. This page looks at running the same daily job without EWC:

- two GCP Spot layouts;
- Google Colab driven by the Colab CLI and paid for in compute units. This is
  the main subject (§4).

**Status (2026-10-16): nothing here has been run.**
- GCP prices are us-central1 list rates, taken from
  [`DAILY_OPERATION_COST.md`](DAILY_OPERATION_COST.md) §5. europe-west3, next
  to the data in AWS Frankfurt, is somewhat dearer.
- Colab facts come from the official CLI README and the Colab FAQ (sources at
  the end).
- **[est]** marks an estimate, and **[untested]** a step nobody has run.

## 1. Why look at all

EWC compute and Ceph sit in Bologna. The data being read sits on AWS in
Frankfurt (`s3://ecmwf-forecasts`, eu-central-1). So EWC brings no locality on
the read side:

- every date pulls ~99 GB of global GRIB across the internet;
- all six VMs leave through one shared NAT IP;
- AWS throttles that IP: 503 SlowDown at 80 readers on 10-05, and 974 retries
  on 10-12 at 24 readers. See
  [`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) §2–3.

What EWC does give is $0 compute, the Ceph quota, and $0 egress to
source.coop. On GCP, staging and egress cost more than compute (§5).

## 2. What a failure loses, on any platform

`build_corpus.py` commits **once per date** (`build_corpus.py:223-235`). Every
finished block's fork is held in the **client's** memory until that commit.

| what dies | what is lost | recovery |
|---|---|---|
| a worker | only its in-flight blocks | resubmitted in the next round (`--rounds 6 --pause 120`) |
| the client or scheduler | **all work on the current date**, 40–110 min | rerun `build_corpus.py --repair` |
| the store | nothing | Icechunk commits are atomic, so the branch stays at the last snapshot (or a labelled `PARTIAL` commit). Uncommitted chunk objects are orphans that cost storage until expired and garbage-collected, never a corrupt read |

Restart rules:

1. Rerun `--repair` with the **same window**, `--start 2026-09-01 --days 122`
   for SOND 2026. The slot index is positional.
2. Run **one writer at a time**. A second run's commit conflicts and wastes
   its work.
3. Mirror ③ is restart-safe: it syncs only missing objects, and the `repo`
   pointer goes last.
4. `daily_run.sh` advertises `--from STAGE` but it was never implemented
   (crma `RUNBOOK_V4_20261011.md` §3), so a chain that dies mid-way is resumed
   by hand.

**Change recommended before any preemptible platform [untested]:** commit a
labelled `PARTIAL` snapshot every ~500 blocks instead of once per date.
`--repair` already fills partial dates, and a lost client then costs minutes
instead of a whole date.

## 3. GCP Spot

The layout already exists as Terraform: `cno-e4drr/devops/crma-gik-spot-vm`.
It is a gateway plus Spot workers in europe-west3 with a GCS sink, selected
with `CRMA_SINK=gcs` (refill `10595a2`). It validates but has **never been
applied**.

| | scenario 1: all Spot | scenario 2: on-demand gateway + Spot workers |
|---|---|---|
| compute **[est]** | gateway ~2.5 h/day + 6 workers ~1.2 h/day, all Spot: **~$12/mo** | gateway only during the run (~$10/mo) or always on (~$98/mo), plus Spot workers ~$9/mo: **~$19–107/mo** |
| worker preempted | blocks retried | blocks retried; the managed instance group or Terraform recreates the VM and it re-dials the scheduler |
| gateway preempted | **the date in flight is lost**; nothing restarts it unless an outside watchdog (Cloud Scheduler, a tiny always-on VM) restarts the VM and reruns `--repair` | does not happen; `run_corpus_supervised.sh` on the gateway handles retries |
| suits | the backfill, where a lost date is a retry | **the daily routine** |

**Single-node variant [est, untested].** At 24 readers a date reads ~99 GB in
~47 min, about 35 MB/s. The EWC workers were ~70% CPU-idle: ~3–4 cores busy
and ~20 GB RSS at peak across the cluster. So one **e2-standard-8**
(8 vCPU, 32 GB) running the scheduler and six 4-thread workers may be enough:
~$6/mo on Spot, ~$20 on-demand. It also removes the gateway/worker split.

Non-compute costs on GCP, either scenario **[est]**:
- GCS staging of the current season (≤2.1 TB): ~$21–42/mo;
- egress to source.coop (~516 GB/mo): ~$60/mo.

## 4. Google Colab as the daily routine

### 4.1 What Colab and its CLI provide

- **The [Colab CLI](https://github.com/googlecolab/google-colab-cli)**
  (Apache-2.0, June 2026). Install with `uv tool install`; Linux/macOS only.
  - `colab new`, `colab sessions`, `colab status`, `colab stop`: session lifecycle;
  - `colab run`: runs a local script on a fresh VM, then releases it unless `--keep`;
  - `colab exec`, `colab repl`, `colab console`, `colab ssh`: execution;
  - `colab upload`, `colab download`, `colab ls`: files;
  - `colab usage`: unit balance and burn rate; `colab pay`.
- **Runtimes.** CPU is the default; GPUs and TPUs exist but this job needs
  none. `--high-mem` asks for the high-RAM shape (Pro/Pro+). The CLI syntax
  differs between third-party guides, so **confirm flags with `colab --help`**.
- **Limits (Colab FAQ):**
  - paid sessions run at most **12 h**, and runtimes time out when idle;
  - an exhausted balance falls back to free-tier rules, and the backend may
    terminate the runtime;
  - **"running distributed computing workers"** and SSH are restricted only on
    free runtimes or with no positive balance;
  - using several accounts to get round limits is forbidden.
- **Not documented:** inbound connections to a runtime, the number of
  concurrent runtimes per account, and the runtime's region. Expect the US,
  which means every byte-range read to Frankfurt crosses the Atlantic.
- **Price:** 100 compute units for $9.99, either pay-as-you-go (units expire
  after 90 days) or as Colab Pro's monthly allowance.
- **No published CPU burn rate.** It shows in a connected runtime and through
  `colab usage`. GPU rates for reference: T4 ~1.2–1.8 units/h.

### 4.2 Recommended shape: one high-RAM CPU runtime, no remote cluster

```
 gateway cron (or a tiny always-on VM)
   │  stage ① pars ── Lithops/Cloud Run (unchanged)
   │  stage ② extend + backfill + mirror ② (unchanged, or moved into Colab)
   │
   └─ colab run --high-mem  colab_daily.sh <YYYYMMDD>
        ┌──────────── one Colab CPU runtime ───────────────────────────┐
        │ frisky scheduler (127.0.0.1:8796)                            │
        │ 6 × frisky worker, 4 threads  = 24 readers                   │
        │ build_corpus.py --repair  ──reads──▶ source.coop (virtual)   │
        │                                     ──▶ s3://ecmwf-forecasts │
        │                           ──writes─▶ sink (§4.3)             │
        │ check_chunks.py  →  mirror ③ (if the sink is not source.coop)│
        └──────────────────────────────────────────────────────────────┘
```

**Not recommended: Dask/frisky workers spread over several Colab runtimes.**
- The FAQ allows it on a paid balance.
- But runtimes accept no documented inbound connections. Workers would have to
  dial out to a scheduler on a VM with a public IP (an extra always-on cost,
  and an open port).
- The concurrent-runtime limit is undocumented.
- And the daily load fits one machine (§3, single-node variant).

### 4.3 Where it writes

| sink | how | cost | trade-off |
|---|---|---|---|
| **A. source.coop directly** | **new backend needed** in `sink_icechunk.py`: today only `ceph` and `gcs` exist; this would be S3 with the source.coop endpoint and the STS | **~$0**, since Colab egress is not billed to the user | readers only ever see committed snapshots, so a crash publishes nothing half-done. But a `PARTIAL` commit **is** public until repaired, and the check-on-Ceph-then-publish step ([`04_VERIFY_AND_PUBLISH.md`](04_VERIFY_AND_PUBLISH.md)) is lost |
| B. GCS | `CRMA_SINK=gcs CRMA_SINK_BUCKET=…`, plus an SA key uploaded to the session; then mirror ③ from GCS | ~$21–42/mo staging + ~$60/mo egress **[est]** | keeps check-then-publish; costs more than Colab itself |
| C. EWC Ceph (hybrid) | the current default sink, over Ceph's public HTTPS endpoint with `AWS_CA_BUNDLE` | $0 | keeps the publish flow, but Colab replaces only the VMs, and the run still depends on the EWC application credential. **Reachability of Ceph from outside EWC is untested** |

### 4.4 The daily routine, step by step [untested]

Run order and gates as in [`03_DAILY_OPERATIONS.md`](03_DAILY_OPERATIONS.md);
only stage ③ and mirror ③ move.

1. **Trigger** (gateway cron, after stage ② and mirror ② pass):
   `colab run --high-mem colab_daily.sh 20261016`. Use `run`, not `new`: it
   releases the VM at the end, and **an idle VM keeps burning units**.
2. **Bootstrap in the runtime** (~2–3 min per session; nothing persists):
   - fetch this folder (git clone, or `colab upload`);
   - write `.env` from Colab secrets, or upload it: source.coop STS, plus the
     GCS SA key or Ceph keys, depending on the sink;
   - `chmod 600 .env`, then `unset AWS_*` before stage ③ (the
     [`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) §3 trap);
   - install the pinned stack into a venv: frisky 0.7.2, icechunk 2.1.1,
     zarr 3.2.1, xarray 2026.7.0, gribberish 1.6.0, numpy 2.5.1,
     dask 2026.7.1, boto3. Client/worker parity holds automatically, because
     both run from the same venv.
3. **Start the cluster locally**, using the same commands `deploy_workers.py`
   issues on each VM:
   ```bash
   .venv/bin/frisky scheduler --address 127.0.0.1:8796 &
   for i in 1 2 3 4 5 6; do
     env -u AWS_ENDPOINT_URL .venv/bin/frisky worker 127.0.0.1:8796 --nthreads 4 &
   done
   ```
   Keep 24 readers. Whether a Colab runtime has its **own** egress IP, which
   would ease the AWS throttling, is unknown. Measure before raising the count.
4. **Gate:** the single-read and throttle probe from `daily_run.sh`.
5. **Realize:** `build_corpus.py --sink <prefix> --repair --era 50r1 --start
   2026-09-01 --days 122 --vars <39> --members 51 --steps 65 --rounds 6
   --pause 120 --scheduler 127.0.0.1:8796`.
6. **Verify:** `check_chunks.py` against `39 × 65` for the date.
7. **Publish:** mirror ③ for sinks B and C; nothing extra for sink A.
8. **Release:** `colab run` exits. Record `colab usage` in the day's ledger.

Downstream CRMA layers (Layer 2→3, Layer 4, export, upload) stay on the gateway
as now. They could move into the same runtime later.

### 4.5 Failures on Colab

| failure | effect | restart |
|---|---|---|
| runtime terminated (no availability guarantee, balance exhausted, 12 h cap) | the client dies, so **the date in flight is lost** (§2); the store is intact | the trigger reruns `colab run …` with `--repair`; only missing blocks are rebuilt. Add the partial-commit change from §2 first |
| no runtime available | the date waits | retry from cron, e.g. every 30 min up to a deadline, then alert |
| AWS throttling | retries, as on EWC | the same gate and 24-reader rule |
| credentials | STS expiry mid-run | refresh with `source-coop creds` on the gateway before triggering; pass the fresh `.env` in |

### 4.6 Cost

| CPU burn rate (**measure it**) | 2 h/day uses | 100 units ($9.99) last |
|---|---|---|
| 0.1 unit/h | ~6 units/mo | ~16 months of use, but pay-as-you-go units expire at 90 days |
| 0.5 unit/h | ~30 units/mo | ~3 months |
| 1 unit/h | ~60 units/mo | ~1.7 months |

The $10 plan very likely covers the daily routine **if** one CPU runtime
completes a date in 1–2 h. With sink A the all-in cost is ~$10–15/mo **[est]**.
With sink B, add GCS staging and egress (~$80–100/mo), and Colab saves little
over GCP Spot.

**Not for the backfill.** 243 cluster-hours on six VMs is far beyond one 12 h
runtime per session.

### 4.7 First test session (decides whether this is viable)

1. `colab new --high-mem`, then `colab status`: record vCPU, RAM and region.
2. Run the [`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) §6 step 5
   smoke test against a scratch sink: finite numbers mean the read and decode
   path works.
3. Realize one full date into a scratch GCS prefix. Record wall time, retries,
   MB/s and `colab usage` before and after.
4. `colab stop`. The numbers from step 3 fill in §4.6 and settle sink A vs B.

## 5. Side by side

| | compute / mo | staging + egress / mo | daily reliability |
|---|---|---|---|
| EWC today | $0 | $0 | good; throttled on the shared NAT IP |
| GCP scenario 1, all Spot | ~$12 | ~$80–100 | lowest: a gateway preemption loses the date unless a watchdog reruns it |
| **GCP scenario 2**, on-demand gateway + Spot workers (or one Spot node) | ~$19–107 (~$6–20 single node) | ~$80–100 | good |
| Colab, 100 units, sink A | ~$10 (rate unmeasured) | ~$0–5 | fair: no availability guarantee |

All non-EWC rows are **[est]**. For a GCP move, choose scenario 2 (or the
single node) plus the partial-commit change. Colab is worth one test session
(§4.7) before any decision.

## Sources

- [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): limits and restricted activities
- [MarkTechPost: Google's new Colab CLI](https://www.marktechpost.com/2026/06/06/googles-new-colab-cli-lets-developers-and-ai-agents-run-python-on-remote-colab-gpus-and-tpus-from-the-terminal/)
- [Colab GPUs features & pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/): GPU burn rates
- [TechRadar: Colab pay-as-you-go](https://www.techradar.com/news/googles-python-writing-code-system-is-going-pay-as-you-go): $9.99 / 100 units
- `cno-e4drr/devops/crma-gik-spot-vm/README.md`; refill `10595a2` (the GCS sink)
