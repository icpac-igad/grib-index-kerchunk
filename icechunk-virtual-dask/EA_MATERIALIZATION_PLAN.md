# Materialising the EA cGAN predictor set into an Icechunk store on EWC

**Can the `test-icechunk-long.py` pattern — Dask cluster, `to_icechunk`, one
commit per day, appending along `time` — be pointed at a new path and used to
write the *real* East Africa predictor subset instead of synthetic arrays?
Yes. Two things had to be fixed first, and one of them rules out most of the
intended training window on this cluster.**

_Written 2026-07-31. Every number below was measured on the live EWC cluster
(6 workers × 4 threads × 13.9 GB) against the live source.coop store, not
estimated._

Companion script: `materialize_ea_icechunk_ewc.py`
Variable/extent spec: `../docs/ecmwf_icechunk_dask_variable_extraction.md`

---

## 0. Answers up front

| Question | Answer |
|---|---|
| Can `test-icechunk-long.py`'s write pattern be reused verbatim? | **Yes.** Per-date `writable_session` → `to_icechunk(append_dim="time")` → `commit` works unchanged. Only the *source* of the arrays changes. §3 |
| Does it work as-is? | **No — it failed on the first read.** The workers' `AWS_*` environment hijacks the virtual-chunk fetch. §1 |
| **What does a month cost?** | 30 dates × 7-day lead (53 steps) × 51 members × 30 channels → **9.1 GB written (~5.5 GB compressed), 2.43 M GRIB messages / 1.9–4.9 TB read, 4.6–9.2 h.** §4 |
| Can a month be done? | **Yes on `50r1` — but the store only holds 51 dates (2026-05-13 … 2026-07-02),** so "a month" means a month inside that window. §2.2 |
| What dominates the cost? | Read, by ~400×. 2.43 M *global* GRIB messages are decoded to write 5.5 GB of East Africa. Cropping saves storage, not time. §4 |
| Can MAM 2024 / MAM 2025 be done? | **No, not on this cluster.** A single `49r1` pressure-level manifest needs **89.5 GB of worker RAM**. Measured, not inferred. §2 |
| What is the fix for that? | Upstream: rebuild the source store with `icechunk.ManifestSplittingConfig` split along `time`. §6 |
| **Full corpus, all 51 members stored?** | 1,246 dates over three eras → **7.37 TB written (~4.4 TB compressed), 91.5 M messages / ~115 TB read, ~182 h ≈ 7.6 days.** §7 |
| Does keeping members cost more runtime? | **No.** All 51 are read either way; the reduction only changes what lands on disk. It is a storage decision, not a runtime one. §7 |
| How much of the corpus is reachable today? | **4 % — 51 of 1,246 dates.** The rest is blocked on manifest RAM, not storage or time. §7.2 |
| Biggest surprise | Manifest RAM is **~2000 B/ref in memory**, not the ~200 B/ref quoted in the extraction doc. 10× the budget. §2 |

---

## 1. The blocker that had to be cleared first: `AWS_*` poisons the read

The EWC workers carry the Ceph credentials so they can write to
`must-icechunk`:

```
AWS_ENDPOINT_URL   = https://object-store.os-api.cci1.ecmwf.int
AWS_DEFAULT_REGION = RegionOne
AWS_ACCESS_KEY_ID  = ...
```

The source store's virtual chunks live on **AWS**, at
`s3://ecmwf-forecasts/`, and the repo's own container config is correct
(`region: eu-central-1`, `anonymous: true`). But the object-store client
icechunk builds for that container reads the process environment too, and the
environment wins. The fetch then goes to a hostname that does not exist:

```
error fetching virtual reference -> dispatch failure -> io error
  -> client error (Connect) -> dns error
  -> failed to lookup address information: Name or service not known
```

Confirmed by resolving the candidates directly from a worker:

| hostname | resolves |
|---|---|
| `ecmwf-forecasts.s3.eu-central-1.amazonaws.com` | ✅ 3.5.122.124 |
| `ecmwf-forecasts.s3.amazonaws.com` | ✅ 3.5.136.57 |
| `ecmwf-forecasts.s3.RegionOne.amazonaws.com` | ❌ `gaierror` |
| `ecmwf-forecasts.object-store.os-api.cci1.ecmwf.int` | ❌ `gaierror` |

Three fixes were tried. Only one works:

| attempt | result |
|---|---|
| `cfg.set_virtual_chunk_container(..., s3_store(region="eu-central-1"))` | ❌ still DNS-fails — the env overrides the explicit config |
| `os.environ.pop("AWS_*")` inside a worker that had already read | ❌ still DNS-fails — the S3 config is cached process-wide on first use |
| **scrub `AWS_*` before the process builds its first S3 client** | ✅ **2.4 s read, `t2m` mean 299.99 K** |

So the ordering is the whole fix, and the script enforces it three ways: a
`WorkerPlugin` that scrubs at worker startup, a `client.restart()` so the
plugin lands on a virgin process, and a scrub inside `open_source_era()`
before the icechunk import.

**The client keeps its `AWS_*`.** That is deliberate: workers only *read* the
anonymous AWS source, the client is what *writes* the EWC sink. The
credential split falls out of the architecture rather than being worked
around.

---

## 2. The constraint that shapes everything: manifest RAM

Resolving any chunk loads that array's **entire** manifest — refs =
`era_dates × members × steps × levels`, regardless of how little you select.
The extraction doc put this at ~200 B/ref. **Measured on a worker, by RSS
delta across the first read, it is ~2000 B/ref** (that ~200 B is the packed
on-disk figure):

| array | refs | RSS delta | B/ref | manifest load |
|---|---:|---:|---:|---:|
| `50r1/t2m` (2-D) | 0.22 M | 0.52 GB | 2195 | 11 s |
| `0p4/t2m` (2-D) | 1.74 M | 3.61 GB | 2055 | 164 s |
| `49r1/t2m` (2-D) | 3.44 M | 6.95 GB | 2006 | 155 s |
| `50r1/u` (14 levels) | 3.10 M | 6.25 GB | 2008 | 179 s |
| `49r1/u` (13 levels) | 44.75 M | **~89.5 GB** | — | **killed after 40 min** |

### 2.1 What fits on this cluster

Worker budget is ~9.7 GB (70 % of 13.9 GB before dask spills, then kills).

| era | surface var | pressure var | verdict |
|---|---:|---:|---|
| `50r1` | 0.44 GB | **6.19 GB** | ✅ 16 store vars over 6 workers, worst worker 7.07 GB — **one pass** |
| `0p4` | 3.47 GB | 31.2 GB | ❌ pressure infeasible (and `0p4` has no `w`/`ttr`/`cape` anyway) |
| `49r1` | 6.88 GB | **89.49 GB** | ❌ **pressure infeasible** — one manifest is 9× a whole worker |

`49r1` is not a scheduling problem. No task decomposition helps, because the
manifest is loaded whole before the first byte of data. Even the *surface*
vars at 6.88 GB allow only one per worker, so `49r1` surface-only would need
two passes.

### 2.2 What the store actually contains right now

Read live, not assumed:

| group | dates | window | levels |
|---|---:|---|---:|
| `0p4/00z` | 401 | 2023-01-18 … 2023-12-07 | 9 |
| `49r1/00z` | 794 | 2024-02-29 … **2026-05-06** | 13 |
| `50r1/00z` | **51** | 2026-05-13 … **2026-07-02** | 14 |

Two things to note: `50r1` holds **51 dates**, so "at least a month" fits but
"three months" does not yet; and the store is **~4 weeks behind** today
(2026-07-31) — it is not being appended daily.

**Consequence for the plan:** the benchmark runs on `50r1`, and the
`50r1` manifests stay cheap *only while the era is short*. At ~250 dates the
pressure-level manifest reaches ~30 GB and `50r1` joins `49r1` on the
infeasible list. This window is temporary.

---

## 3. The design, and how it maps onto `test-icechunk-long.py`

| `test-icechunk-long.py` | here |
|---|---|
| `day_dataset()` builds synthetic `da.random.*` | 16 pinned Dask tasks read the real source, crop, reduce over `number` |
| `repo.writable_session("main")` per day | same |
| `to_icechunk(ds, session, mode="w" \| append_dim="time")` | same |
| `session.commit(f"day {d}")` | same, message carries era + channel count |
| one commit per day | same — commits stay ordered, reads pipeline ahead |

The read side is what is new, and it is driven entirely by §2:

- **One task = (one store variable, all its levels, one date).** Reads want to
  be big: all five `u` levels in a single `.compute()` measured 21.6 msg/s,
  against ~10 msg/s for small reads. Splitting them was tried and is worse —
  §4.2.
- **Each store variable is pinned to one worker for the whole run** and the
  opened era is cached there. The manifest is paid once (11–215 s) rather than
  once per date. Over 30 dates that is the difference between ~5 minutes and
  ~14 hours of pure manifest loading.
- **The driver assembles and writes.** Per date the gathered payload is
  ~305 MB, so funnelling it through the client is cheap next to the ~122 GB
  read that produced it, and it keeps the sink credentials off the workers.
- **A pre-flight budget check refuses to start** a run whose manifests cannot
  fit, with the `--vars` tiering to use instead. Better than watching workers
  get killed 20 minutes in.
- **Manifests are warmed in batches of 3, not all at once.** Firing all 16
  variables simultaneously means 16 concurrent multi-GB manifest loads; that
  was observed to starve the workers until the scheduler dropped the client
  (`scheduler-connection-lost`) and every in-flight result was lost. The
  warm-up costs one cheap read per variable and the main loop then starts
  fully warm. `--batch` tunes it; `--no-warmup` disables it.
- **Level splitting is available but off.** `--levels-per-task 1` would spread
  a 5-level variable across its worker's threads; measured, it killed the
  worker and restarted the scheduler, and the unsplit task is faster per
  message anyway. §4.2.

### 3.1 What comes out

```
s3://must-icechunk/<prefix>/            icechunk repo, branch main
  <channel>   (time, step, stat, latitude, longitude)   float32
              stat = ["mean", "sd"]        <- the 2 channels load_fcst() reads
              chunk = (1, 11, 2, 163, 147) = 2.1 MB
```

30 channels × 1 chunk per date → the **sink manifest is ~900 refs for a
month**. That is the point of materialising: the output store is trivially
cheap to open, where the source is not.

Accumulated fields (`tp`, `ssr`, `ttr`) are stored **as accumulated** — the
step-differencing, the `olr = -ttr/Δt` conversion, `sst = skt` masked by
`lsm`, and the derived `kindex`/`thetaw850` all stay downstream, as in the
existing `materialize_ea_icechunk_dask.py`.

### 3.2 Variables and extent — unchanged from the spec

Extent is the §5.2 superset box, `25.25 … -15.25 N`, `18.5 … 55.0 E`,
**163 × 147 at 0.25°**, which strictly contains both the 0.1° TF training
frame and the PyTorch EP box with a 2-cell halo.

**Steps cover the full 7-day forecast: 0 … 168 h, 53 values.** Read off the
store's own axis, not assumed — it is 3-hourly to 144 h then 6-hourly beyond,
so 0…168 h is 49 + 4 = **53 steps** (`--lead-days 7`). The narrower
24…54 h band the extraction doc specs for the cGAN's accumulation windows is
a subset, available as `--cgan-steps` (11 steps).

**Members: all 51** (`number` 0…50, 0 = control), reduced on the worker to
mean + sd.

Channels published in `50r1`, **30 from 16 store variables**:

| kind | channels |
|---|---|
| surface (10) | `tp pw sp msl t2m skt ssr ttr tcw mucape` |
| `u`,`v` (10) | 925, 850, 700, 500, 200 |
| `w` (4) | 925, 850, 700, 500 |
| `r` (2) | 850, 700 |
| `t` (3) | 850, 700, 500 |
| `gh` (1) | 500 |

`cape` is `49r1`-only (before 2025-01-14); `50r1` carries `mucape`. The script
fixes the channel schema on the first date and **aborts** if a later date
disagrees, so the all-NaN union trap cannot silently enter the store.

### 3.3 Availability, verified not assumed

`probe --eras 50r1` on 2026-07-02, member 0, +24 h. **All 30 channels read
back finite = 1.00** — no all-NaN channel in `50r1`:

| store var | seconds (cold) | channels |
|---|---:|---|
| `u` | 209.7 | `u925 u850 u700 u500 u200` |
| `v` | 135.1 | `v925 v850 v700 v500 v200` |
| `w` | 209.1 | `w925 w850 w700 w500` |
| `r` | 126.2 | `r850 r700` |
| `t` | 143.7 | `t850 t700 t500` |
| `gh` | 214.3 | `gh500` |
| `tp tcwv sp msl t2m skt ssr ttr tcw mucape` | 19–35 each | one each |

Those seconds are almost entirely manifest load — the payload is one message
per channel. They are the one-off cost the pinning-plus-warm-up design pays
once instead of once per date.

---

## 4. Sizing: one month of 00z dates × 7-day lead × 51 members

**Per forecast date:** 30 channels × 53 steps × 2 stats × 163 × 147 × 4 B =
**304.8 MB** written. Read side: 30 × 51 members × 53 steps = **81,090 global
GRIB messages**.

| | 1 date | 7 dates | **30 dates** | 51 dates (all of `50r1`) |
|---|---:|---:|---:|---:|
| written, uncompressed | 0.30 GB | 2.13 GB | **9.14 GB** | 15.5 GB |
| written, after zstd (est.) | ~0.18 GB | ~1.28 GB | **~5.5 GB** | ~9.3 GB |
| sink chunk-refs | 30 | 210 | 900 | 1,530 |
| GRIB messages decoded | 81,090 | 567,630 | **2.43 M** | 4.14 M |
| bytes read @ 0.8–2 MB/msg | 65–162 GB | 0.45–1.1 TB | **1.9–4.9 TB** | 3.3–8.3 TB |
| **wall clock** | 9–18 min | 1.1–2.1 h | **4.6–9.2 h** | 7.8–15.6 h |

Note the asymmetry: **~2.4 million global GRIB messages are decoded to write
5.5 GB.** The read is ~400× the write, because a virtual chunk is one whole
global field and the EA box is 0.2 % of it. Cropping saves storage, not time.

### 4.1 Where the wall clock comes from

Not `messages / threads` — the pinning forbids that. Every store variable is
one task on one worker (§2.1), so the per-date critical path is the single
**heaviest task**: `u` at 5 levels × 51 members × 53 steps = **13,515
messages**.

Measured rates, warm, on this cluster:

| shape | messages | seconds | rate |
|---|---:|---:|---:|
| `t2m`, 11 members × 11 steps | 121 | 12.4 | 9.8 msg/s |
| `t2m`, 51 members × 3 steps | 153 | 13.2 | 11.6 msg/s |
| **`u`, 5 levels, 51 members × 12 steps** | **3,060** | **176** | **21.6 msg/s** |

**Bigger reads are faster**, roughly 2× from the smallest to the largest shape
— the per-request overhead amortises. So the production-shaped task is the
well-measured one:

- 13,515 messages ÷ 21.6 msg/s = **626 s ≈ 10.4 min per date**
- **→ 5.2 h for 30 dates.** The 4.6–9.2 h band in the table is that ±2×.

Plus a one-off ~5 min manifest warm-up.

### 4.2 One thing that was tried and must not be

The obvious optimisation is to split `u`'s five levels into five tasks so the
worker's four threads are used. **It took the cluster down.** Five decode
pipelines running on top of a 6.2 GB resident manifest exceeded the worker,
which was killed, and every in-flight result was lost:

```
FutureCancelledError: cancelled for reason: scheduler-restart.
```

Since the unsplit task is *also* the faster one per message (21.6 vs ~10
msg/s), there is no trade-off here — one task per variable is both safer and
quicker. `--levels-per-task` defaults to 0 and should stay there unless the
workers get materially more RAM.

**The remaining open question is bandwidth, not CPU.** 5.2 h for 2.43 M
messages implies ~130 msg/s aggregate ≈ **194 MB/s sustained from AWS
`eu-central-1` into EWC**. If the link caps below that, bandwidth sets the
time and no task tuning helps. The 1-date run settles it — that is why it is
step 3 and not an afterthought.

### 4.3 Two knobs, both large

| change | wall clock, 30 dates | storage, 30 dates | note |
|---|---|---|---|
| *(baseline: 51 members, 53 steps)* | **5.2 h** | **9.14 GB** | |
| `--cheap-members` (11 not 51) | **1.1 h** (÷4.6) | 9.14 GB — unchanged | mean/sd from 11 members is close for smooth predictor fields |
| `--cgan-steps` (11 not 53) | **1.1 h** (÷4.8) | **1.90 GB** (÷4.8) | if only the 24–54 h accumulation windows are needed |
| both | **~14 min** | 1.90 GB | |

Storage is dominated by steps, wall clock by members × steps. If the 7-day
lead is wanted for a *forecast* product but the cGAN only trains on 24–54 h,
those are two different extractions and it is worth being explicit about which
one this store is for.

**If individual members are stored instead of mean+sd**, 30 dates is **233 GB**
rather than 9.14 GB (×25.5). See §7 for the full corpus at that setting, and
for the write-path change it forces.

---

## 5. Next steps, in order

```bash
cd ~/cGAN_tutorial/icechunk-virtual-dask
P=/opt/mamba/envs/dask/bin/python

# The CLIENT needs the EWC credentials -- it is what writes the sink.
export AWS_ENDPOINT_URL=https://object-store.os-api.cci1.ecmwf.int
export AWS_DEFAULT_REGION=RegionOne
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

**1 — sanity, no cluster, no cost.** Confirms the channel list, the box, and
that the manifests fit before anything is spent:

```bash
$P materialize_ea_icechunk_ewc.py plan --days 30 --lead-days 7
$P materialize_ea_icechunk_ewc.py plan --days 30 --cheap-members  # the ÷4.6
$P materialize_ea_icechunk_ewc.py plan --era 49r1 --days 30       # INFEASIBLE
```

**2 — availability, ~20 min. Already run — all 30 channels finite (§3.3).**
Re-run it if the store is updated or the date moves:

```bash
$P materialize_ea_icechunk_ewc.py probe --eras 50r1 --batch 2
```

**3 — one date, one variable, ~5 min. This is the one that matters.** It
proves the *write* path against Ceph end to end (repo create → chunk write →
commit) and, more importantly, measures the real messages/s so the 4.6–9.2 h
band above collapses to a number:

```bash
$P materialize_ea_icechunk_ewc.py run --days 1 --vars t2m \
    --prefix ea-cgan/calib
$P materialize_ea_icechunk_ewc.py size --prefix ea-cgan/calib
```

**4 — one full date, all 30 channels, ~10–20 min.** The first honest
per-date cost, and the first real compression ratio:

```bash
$P materialize_ea_icechunk_ewc.py run --days 1 \
    --prefix ea-cgan/v1-1day 2>&1 | tee run-1day.log
$P materialize_ea_icechunk_ewc.py size --prefix ea-cgan/v1-1day
```

Read off the log: `s/date` (extrapolations to 30/90/276 dates are printed for
you), `peak worker GB` (must stay under ~9.7), and the `ratio` from `size` —
the only trustworthy storage figure.

**5 — the month, 4.6–9.2 h.** `--days 30` ends at `2026-07-02`, so this covers
2026-06-03 … 2026-07-02. Run it detached; it outlives a terminal:

```bash
nohup $P materialize_ea_icechunk_ewc.py run --days 30 \
    --prefix ea-cgan/v1-30day > run-30day.log 2>&1 &
$P materialize_ea_icechunk_ewc.py size --prefix ea-cgan/v1-30day
```

If step 4 shows the link is the bottleneck rather than the threads, run
`--cheap-members` first (÷4.6, ~1 h) and treat the 51-member pass as the
follow-up.

**6 — decide whether 51 members are worth 4.6×.** Re-run the same dates with
`--cheap-members` and compare the `sd` channel against the 51-member pass. If
the difference is inside the normalisation noise, 11 members stands and every
future extraction is 4.6× cheaper.

### 5.1 Acceptance checks before any of this feeds training

Carried over from the extraction doc §5.10, and all cheap on the materialised
store:

1. finite fraction > 0.99 for every channel — and check nothing was *skipped*
2. `gh500` ≈ 5850–5900 m over the tropics — the cheapest proof that level
   selection is not silently off
3. `olr` (from `ttr`) ≈ 200–300 W m⁻²; `r850` in 10–100 %; `w700` within
   ±3 Pa s⁻¹; `skt` ≈ 297–303 K over the western Indian Ocean
4. lat/lon strictly bracket −13.65…24.65 N / 19.15…54.25 E on all four sides
5. `u925` shows the cross-equatorial southerly Somali-jet signature

---

## 6. The thing this plan cannot deliver, and what would fix it

**MAM 2024 and MAM 2025 — the actual training window — are in `49r1`, whose
pressure-level manifests need 89.5 GB per worker.** This is not a tuning
problem; nothing in this script or this cluster can work around it.

Three ways forward, in order of leverage:

1. **Rebuild the source store with manifest splitting** (the real fix, already
   flagged as out-of-scope in the extraction doc §2.6). With
   `icechunk.ManifestSplittingConfig` split along `time`, a read loads one
   shard instead of the whole array, and the RAM constraint disappears for
   every era at once. This also protects `50r1`, which will cross the same
   threshold at ~250 dates.
2. **Bigger workers.** `49r1` pressure needs ≥128 GB per worker for one
   variable — 6 of those to run one pass. Expensive, and it only postpones the
   problem as the era grows.
3. **Surface-only from `49r1`,** two passes at 6.88 GB per variable. Gets
   `tp/pw/sp/msl/t2m/skt/ssr/ttr/tcw/mucape` for MAM 2024–25, but no winds, no
   `w`, no `r` — i.e. not the set the doc argues for.

Until (1) lands, the honest scope is: **the pipeline is proven, the sizing is
measured, and the data it can actually produce today is the 51 dates of
`50r1`.**

---

## 6a. Credentials: where they are, and the trap in this script

**Verified 2026-07-31.** `test-icechunk-write.py` was re-run end to end and
**all 8 checks pass**, so the write path this plan depends on is proven, not
assumed:

```
[PASS] all workers have S3 credentials -- 6/6
[PASS] repository open_or_create on Ceph RGW
[PASS] commit succeeded (conditional writes supported) -- snapshot G3T8TMS7CXHMMEP2
[PASS] readback has expected variables / shape / t2m mean 288.01 K
[PASS] work ran on worker VMs, not locally -- 6 workers
[PASS] stale session rejected (commits serialised) -- ConflictError
```

The bucket currently holds **1,329 objects / 3.38 GB** under `_probe`,
`long-test` and `realized-test` — leftovers from those tests, worth clearing
before a real run so `size` reports only the new store.

### Where the credentials actually are

| location | has `AWS_*` |
|---|---|
| the dask **workers** (service environment) | ✅ all 4 vars |
| your **login shell** | ❌ none |
| `~/.aws/credentials`, `~/.env`, `/etc/dask/*` | ❌ do not exist |

So the credentials are only ever in the worker service environment. `run` and
`test-icechunk-write.py` both need them **on the client**, because the client
is what creates the repo and commits. To get them into a shell:

```bash
python - <<'PY' > ~/.ewc-creds.env
import os; from dask.distributed import Client
c = Client(os.environ["DASK_SCHEDULER_ADDRESS"])
for k, v in sorted(list(c.run(
        lambda: {k: v for k, v in os.environ.items()
                 if k.startswith("AWS_")}).values())[0].items()):
    print(f"export {k}={v}")
PY
chmod 600 ~/.ewc-creds.env && source ~/.ewc-creds.env
```

### The trap — a defect this script had and now guards against

`prepare_cluster()` registers a `WorkerPlugin` that strips `AWS_*` before the
worker touches icechunk (§1). **That registration is sticky on the
scheduler.** It is re-applied to every worker that starts afterwards —
including workers belonging to unrelated jobs — so after an early version of
this script ran, `client.restart()` brought the workers back up with **no
credentials at all**, and nothing could write to `must-icechunk`.

That is exactly what happened here and it took a `client.unregister_worker_plugin`
to clear. The script now wraps `run` and `probe` in `try/finally` around
`release_cluster()`, which unregisters the plugin and restarts the workers so
the service environment comes back. Confirmed:

```
INFO unregistered the AWS_* scrub plugin
INFO workers restarted with credentials restored: 6/6
```

If a run is killed hard enough to skip the `finally` (SIGKILL), clear it by
hand before anything else uses the cluster:

```python
client.unregister_worker_plugin("scrub-aws-env"); client.restart()
```

`--leave-scrubbed` skips only the closing restart; the plugin is unregistered
either way.

---

## 7. The full corpus, storing individual members

All three eras, every published 00z date, 7-day lead (53 steps), **all 51
members kept** rather than reduced to mean + sd. Reproduce with
`materialize_ea_icechunk_ewc.py corpus --store-members`.

Grid sizes and date counts are read from the store's own coordinate arrays,
not assumed. The 0p4 era is a **0.4° grid**, so the same lat/lon window is a
much smaller array — 102 × 91 = 9,282 cells against 163 × 147 = 23,961.

| group | dates | channels | EA cells | GB/date | **write TB** | msgs M | read TB | hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0p4` | 401 | 22 | 9,282 | 2.21 | 0.885 | 23.85 | 13.95 | 34.8 |
| `49r1`-pre | 320 | 29 | 23,961 | 7.51 | 2.404 | 25.08 | 37.63 | 55.6 |
| `49r1`-post | 474 | 30 | 23,961 | 7.77 | 3.684 | 38.44 | 57.65 | 82.4 |
| `50r1` | 51 | 30 | 23,961 | 7.77 | 0.396 | 4.14 | 6.20 | 8.9 |
| **total** | **1,246** | | | | **7.37 TB** | **91.5 M** | **115 TB** | **182 h** |

- **Storage: 7.37 TB uncompressed, ~4.4 TB after zstd.** Against 289 GB
  (~173 GB compressed) for mean+sd — a **25.5×** difference.
- **Read: 91.5 M GRIB messages, ~115 TB — identical in both modes.** All 51
  members must be *read* either way; the reduction only changes what lands on
  disk. So the ensemble decision is a storage decision, not a runtime one.
- **Time: 182 h ≈ 7.6 days** of continuous cluster at the measured 21.6 msg/s,
  again the same in both modes.

### 7.1 Why the channel counts differ per group

| group | surface | pressure | total |
|---|---:|---:|---:|
| `0p4` | 6 — no `ssr`, `ttr`, `tcw`, `cape`, `mucape` (none published) | 16 — **no `w` at all** | **22** |
| `49r1`-pre | 9 — has `cape`, no `tcw`/`mucape` yet | 20 | **29** |
| `49r1`-post | 10 — `mucape` + `tcw`, `cape` now all-NaN | 20 | **30** |
| `50r1` | 10 | 20 | **30** |

Verified against each group's variable list. `0p4` has `d gh lsm msl q r ro
skt sp st t t2m tcwv tp u u10 v v10 vo` — no `w`, no radiation, no CAPE
family, which is what makes it a different channel set rather than a shorter
one.

### 7.2 What actually blocks this today

| era | surface manifest | pressure manifest | on a 9.7 GB budget |
|---|---:|---:|---|
| `0p4` | 3.48 GB | **31.3 GB** | pressure INFEASIBLE |
| `49r1` | 6.88 GB | **89.5 GB** | pressure INFEASIBLE |
| `50r1` | 0.44 GB | 6.19 GB | ✅ fits |

**51 of 1,246 dates — 4 % of the corpus — are extractable on this cluster.**
The other 96 % are blocked on manifest RAM, not on storage or time. Nothing in
the extraction code can change that; the fix is `ManifestSplittingConfig` on
the source store (§6).

### 7.3 Two consequences of keeping members that are easy to miss

**1. The write path has to change.** At 51 members a single date is ~7.8 GB —
far too much to gather into the driver and append as one Dataset, which is how
the mean+sd path works. `--store-members` therefore switches to
**preallocate + region writes**: the full `(time, step, number, lat, lon)`
schema is created NaN-filled up front (zarr does not write all-fill chunks, so
this is nearly free), then each variable is streamed into its own region as it
lands and released. Peak driver memory becomes one variable (~1.3 GB for a
5-level one) instead of a whole date.

**2. The output store inherits the manifest problem.** Chunked one per
`(date, step)` with all members together, the full corpus is **1.79 M refs →
~3.6 GB** of manifest for the *materialised* store — the very cost
materialising was meant to escape. Chunking one per `(date, all steps, all
members)` keeps it at 0.03 M refs / 0.07 GB, but makes each chunk ~259 MB,
which is too coarse to read efficiently.

Neither extreme is right. **Split the output store per era (or per year) and
write it with `icechunk.ManifestSplittingConfig`** — the same fix the source
store needs, applied on the way out.

> **Untested:** neither write path has been executed. My shell has no EWC
> credentials and writing would create objects in `must-icechunk`. The read
> side, the manifest measurements, and the channel availability are all
> measured on the live cluster; the append and region-write paths are
> designed and syntax-checked only. Step 3 in §5 is what proves them.
