# Changelog — EA cGAN Icechunk materialisation on EWC

Work log for `icechunk-virtual-dask/`, newest first. Each entry records what
changed, **why**, and which numbers moved — so the reasoning lives here rather
than only in `git log`.

All work dated 2026-07-31. Measurements are from the live EWC cluster
(6 worker VMs × 4 vCPU × 16.77 GB, Dask `memory_limit` 13.94 GB) against the
published ECMWF IFS ensemble Icechunk store on source.coop.

---

## Unreleased — the manifest diagnosis was wrong, and is withdrawn

**Status: in the working tree, not yet committed.**

### The headline correction

Earlier entries in this log — and the two planning documents — claimed that
the source store's manifests were the binding constraint: that a `49r1`
pressure-level manifest needed **89.5 GB** of worker RAM, that **96 % of the
archive was unreachable**, that workers needed **128 GB**, and that the fix
was to rebuild the source store with `ManifestSplittingConfig`.

**All of that is withdrawn.** Listing the store's manifest objects:

```
46,154 manifest objects
11.11 GB total
largest 1.1 MB      median 0.071 MB      none above 1.1 MB
```

and the repository's own **persisted** config, via
`icechunk.Repository.fetch_config()`:

```
splitting: [(any_array, [(dimension_name("time"), 1)])]
```

**The store is already split one manifest shard per date.** No per-array
manifest exists. The `ManifestSplittingConfig` rebuild we were about to
request is already done upstream — there was nothing to fix.

### Where the error came from

The measurement was real (worker RSS grew 0.52–6.95 GB on a first read); the
inference was not. Dividing by the array's *total* chunk-reference count gave
~2000 B/ref, and I extrapolated that product across whole arrays. That step
assumed manifests scale with the array, which the object listing disproves.

The `49r1/u` read that "died after 40 minutes" is re-attributed to the network
failure already in progress, not to OOM. Worker RSS during the later failed
probes never exceeded 0.06 GB — nothing was being loaded.

**Still unexplained:** what the 6.25 GB RSS on a first read actually was.
Worth understanding before a multi-day run.

### Bytes per GRIB message: measured, not assumed

The ingress estimate rested on an assumed 1.5 MB/message. ECMWF publishes a
`.index` sidecar beside every GRIB file with `_offset` and `_length` per
message; averaged over the 1,500 messages making up our channels:

| | assumed | **measured** |
|---|---:|---:|
| bytes per message | 1.5 MB | **0.788 MB** (0.307 MB on 0p4) |
| corpus ingress | 115 TB | **60.6 TB** |
| read amplification | 26× | **8× vs raw, 14× vs compressed** |

Per-variable means: `w` 1.211 MB, `ttr` 1.240, `mucape` 0.891, `tp` 0.836,
`v` 0.752, `u` 0.717, `2t` 0.655, `t` 0.608, `skt` 0.530, `msl` 0.526,
`r` 0.392, `gh` 0.377. Cross-check: 8,500 messages × 0.794 MB ≈ 6.75 GB for
one step file, matching the 6,754 MB object in the bucket.

### The real constraint: the AWS→EWC link

Plain HTTP range GETs, no Dask, no Icechunk, no decode:

| worker | single-stream MB/s |
|---|---:|
| 192.168.1.155 | **14.46** |
| 192.168.1.120 | 10.80 |
| 192.168.1.105 | 9.42 |
| 192.168.1.74 | 9.28 |
| 192.168.1.196 | **0.74** |
| 192.168.1.20 | **0.74** |

A 20× spread at the same instant, and shortly afterwards every read failed
with `HTTP connect timeout after 3.1s` — all three eras, surface and pressure
alike. Against 60.6 TB:

| aggregate | corpus |
|---|---:|
| 200 MB/s | 3.5 days |
| 75 MB/s | 9.4 days |
| 11.8 MB/s | 59.5 days |

### Corrected corpus figures

| group | dates | ch | write TB | msgs M | read TB |
|---|---:|---:|---:|---:|---:|
| `0p4` | 401 | 22 | 0.885 | 23.85 | 7.32 |
| `49r1`-pre | 320 | 29 | 2.404 | 25.08 | 19.77 |
| `49r1`-post | 474 | 30 | 3.684 | 38.44 | 30.29 |
| `50r1` | 51 | 30 | 0.396 | 4.14 | 3.26 |
| **total** | **1,246** | | **7.37** | **91.50** | **60.63** |

Realized store: **7.37 TB uncompressed, ~4.4 TB after zstd** (ratio still
unmeasured). Storing mean+sd instead would be 289 GB.

### Files touched

- `materialize_ea_icechunk_ewc.py` — manifest model replaced with measured
  constants; `check_manifest_budget` demoted from gate to no-op; `plan` and
  `corpus` reworked around measured message size and a `--agg-mb-s` bandwidth
  parameter that now sets the schedule.
- `EA_MATERIALIZATION_PLAN.md` — §2 rewritten as a correction; §6 and §7.2
  withdrawn and replaced; sizing tables updated to 0.788 MB/message.
- `EWC_USAGE_AND_RESOURCE_PLAN.md` — §5.2 compute requirement reduced to the
  current cluster; §6 rewritten; phasing and asks updated.

---

## `20d12c7` — Document real EWC usage to date and the resource plan

Answers *"how much have you actually processed and stored, beyond the early
benchmark"* with measured figures rather than projections.

**Usage to date, stated plainly.** 3.38 GB in `must-icechunk`, **all of it
synthetic**: `long-test` 3.361 GB from `test-icechunk-long.py`,
`realized-test` 0.022 GB from `test-icechunk-write.py`, `_probe` ~0. Plus
~3,800 GRIB messages / ~5.6 GB read from the real archive during calibration.
No forecast data has been materialised — the feasibility work turned up a
blocker worth reporting before an allocation is consumed.

**Cluster recorded precisely**, since the earlier `6 × 4 × 13.9 GB` shorthand
was ambiguous about which number was which:

```
6 worker VMs, 4 vCPU each (24 total)
16.77 GB physical per worker, Dask memory_limit 13.94 GB
manifest budget 9.76 GB/worker (70% of memory_limit, where dask spills)
```

Budgets are taken against `memory_limit`, not physical, because that is what
Dask enforces — spill at 70 %, pause at 80 %, kill at 95 %. Using the full
16.77 GB changes no feasibility verdict. *(Both those manifest figures were
later disproven — see the Unreleased entry.)*

**Two properties of the estimate made explicit**, previously implicit:

- Six workers is a **floor**, not a preference — six pressure-level variables,
  each manifest needs its own worker. With five, two share and it dies.
- More than six workers does **not** make it faster. The per-date critical
  path is a single task (`u`, 13,515 messages, 626 s) and adding workers does
  not split it.

**Also disambiguates the two "7 days"**: the forecast lead that fixes 53 steps
per date, and the wall clock of the job itself.

---

## `a6ce562` — Verify the EWC credentials, and stop the scrub plugin leaking

**`test-icechunk-write.py` re-run end to end: 8/8 checks pass**, including the
conditional-write commit on Ceph RGW and stale-session rejection. The write
path is proven rather than assumed.

**Where the credentials live:** only in the Dask workers' service environment.
The login shell has none; `~/.aws/credentials`, `~/.env` and `/etc/dask/*` do
not exist. Both `run` and `test-icechunk-write.py` need them **on the client**
— that is what creates the repo and commits — so the plan now carries a recipe
for lifting them off a worker into a 0600 env file.

**Fixes a defect this tool introduced.** `prepare_cluster()` registers a
`WorkerPlugin` that strips `AWS_*` before the worker touches Icechunk, and
that registration is **sticky on the scheduler**: re-applied to every worker
that starts afterwards, including workers for unrelated jobs. After an early
version ran, `client.restart()` brought all six workers back with no
credentials at all and nothing could write to `must-icechunk`.

`run` and `probe` now wrap their work in `try/finally` around
`release_cluster()`, which unregisters the plugin and restarts the workers.
Verified: *"unregistered the AWS_* scrub plugin"* → *"workers restarted with
credentials restored: 6/6"*, confirmed independently after the process exited.

Also fixes an `AttributeError` that made `probe` and `size` unrunnable —
`main()` derived `args.steps` from `args.lead_days`, which only the
step-taking subcommands define.

---

## `ba7995e` — Store individual ensemble members, cost the full corpus

The predictor set has to keep all 51 members rather than the mean+sd pair the
cGAN consumes, so both sizing and the write path change.

**New `corpus` subcommand.** Grid sizes and date counts read off the store's
coordinate arrays rather than assumed, which corrected two things:

- **0p4 is a 0.4° grid**, so the same EA window is 102 × 91 = 9,282 cells, not
  163 × 147 = 23,961. Its per-date cost is a third of the 0.25° eras.
- The 49r1 union splits **320 dates before** 2025-01-14 and **474 after**, and
  0p4 genuinely lacks `w`/`ssr`/`ttr`/`cape`/`mucape` — **22 channels against
  29 and 30**.

Full corpus, 1,246 dates × 7-day lead × 51 members stored:

```
write  7.37 TB uncompressed, ~4.4 TB after zstd   (289 GB as mean+sd)
read   91.5 M GRIB messages                       -- IDENTICAL either way
time   182 h = 7.6 days continuous                -- IDENTICAL either way
```

So the ensemble decision is **purely a storage decision**: all 51 members are
read regardless, and only what lands on disk changes, by 25.5×.

**Keeping members forces the write path.** A single date is ~7.8 GB, too much
to gather into the driver and append as one Dataset, so `--store-members`
switches to **preallocate + region writes**: the full
`(time, step, number, lat, lon)` schema is created NaN-filled up front and
each variable streams into its region as it lands. Peak driver memory becomes
one variable, not one date.

**The output store inherits the input's problem** — chunked per `(date, step)`
the corpus is 1.79 M refs / ~3.6 GB of manifest. Split it per era and write it
with `ManifestSplittingConfig`.

---

## `30997ba` — Materialize the EA predictor set into an Icechunk store

Extends the `test-icechunk-long.py` write pattern (per-date `writable_session`,
`to_icechunk(append_dim="time")`, one commit per date) to write the real East
Africa predictor subset, reading from the published virtual ECMWF store.

**Two blockers found by running it:**

1. **The workers' `AWS_*` environment hijacks the virtual-chunk fetch.** The
   EWC Ceph endpoint and `RegionOne` leak into the S3 client Icechunk builds
   for `s3://ecmwf-forecasts`, which then resolves a hostname that does not
   exist and dies on DNS. Setting the container config explicitly does not help
   (env wins); popping `AWS_*` after first use does not either (cached
   process-wide). **Scrubbing before the process builds any S3 client does** —
   2.4 s read, `t2m` mean 299.99 K.

2. **A manifest-RAM constraint that turned out not to exist.** RSS growth on
   first reads (0.52–6.95 GB) was divided by whole-array reference counts to
   get ~2000 B/ref and extrapolated to 89.5 GB for `49r1/u`. **See the
   Unreleased entry — this was withdrawn.** The store is already split one
   manifest per date, largest object 1.1 MB.

**Design follows from those:** one store variable pinned to one worker for the
whole run with the era cached there, so each manifest is paid once rather than
once per date; a staggered warm-up, because 16 concurrent manifest loads drops
the scheduler connection; and one task per variable rather than per level —
the big read measured **21.6 msg/s** where splitting it killed the worker.

All 30 channels probed finite in 50r1.

---

## Open items

| item | status |
|---|---|
| **Why the AWS→EWC link is failing** | **open, and blocking everything.** 0.74–14.5 MB/s per stream, then connect timeouts. Decides 3.5 days vs 60 days. |
| Sustained throughput under concurrency | **unmeasured** — could not be tested while the link was down |
| Compression ratio of the written store | **assumed 40 %** — one calibration date settles it |
| `append` and region-write paths | **never executed** — designed and syntax-checked only |
| `49r1` / `0p4` pressure-level reads | **never demonstrated end to end** — every attempt coincided with the link failing. No memory obstacle is known. |
| What the 6.25 GB first-read RSS actually was | **unexplained** — not a per-array manifest, but worth understanding |
| Running the extraction in AWS `eu-central-1` instead | **worth evaluating** — 4.4 TB out instead of 60.6 TB in |
