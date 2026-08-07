# icechunk-dask

Two stages of one pipeline, and it is worth keeping them apart:

| stage | what it produces | credentials |
|---|---|---|
| **1. publish + read** | *virtual* Icechunk stores on source.coop — metadata only; the GRIB bytes stay in the public AWS buckets | none, anonymous |
| **2. realize** | a *real* store on EWC Ceph holding decoded arrays for a chosen subset | EWC key for the write |

Stage 1 is cheap and global: 15 GB of metadata standing in for ~620 TB of GRIB.
Stage 2 is where the bytes actually move — reading a subset out of stage 1 and
writing it down as arrays you can train on.

**Status:** stage 2 is working. Two realized stores, both East Africa,
30 channels × 51 members × 53 steps:

| store | era | dates | chunks | built |
|---|---|---|---|---|
| `must-icechunk/ea-cgan/v3-june2026` | 50r1 | 30 (Jun 2026) | 47,700 | 176 min, 181 MB/s |
| `must-icechunk/ea-cgan/v4-mar2026-49r1` | 49r1 | **21 of 31** (Mar 2026) | 33,390 | 134 min, ~215 msg/s |

Both verified complete for every date they claim. March is 21 of 31 because the
*source* store is missing ten dates (see "Known defects" below); their `time`
slots are reserved and unwritten, so `--resume` fills them in place once Stage 1
publishes them. The two stores are separate because a `time` axis is fixed at
schema creation and 49r1/50r1 are different model versions — read them together
with `xr.concat([mar, jun], dim="time").sortby("time")`.

| document | contents |
|---|---|
| this file | both stages, end to end |
| [`SCALING.md`](SCALING.md) | what limits read throughput, measured — read before asking for more VMs |
| [`dev-test/`](dev-test/) | how stage 2 was arrived at: the Dask path that hung, the diagnoses made and withdrawn, and the probes that settled them |

---

# Stage 1 — the published virtual stores

Publishing the GIK Icechunk stores to **source.coop** and reading them back
anonymously. The stores hold *virtual* references: metadata lives on source.coop,
the actual GRIB bytes stay in the public AWS buckets (`noaa-gefs-pds`,
`ecmwf-forecasts`). A reader needs **no credentials**.

Published stores (bucket `e4drr-project`, endpoint `https://data.source.coop`):

| product | prefix | group(s) | dates (00z) |
|---|---|---|---|
| GEFS  | `forecasts/noaa_gefs_aws_s3_icechunk_vd`     | `0p25/00z`                     | 2031 |
| ECMWF | `forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd` | `0p4/00z`, `49r1/00z`, `50r1/00z` | 401 / 794 / 51 |

#### Known defects in the published store (measured 2026-08-05)

Two problems, both in the ECMWF store, neither of which blocks a realization
run but both of which will bite a reader. **Not yet fixed.**

**1. Two of the three time axes are not sorted.** A date appended after a later
batch stays where it was written, and nothing sorts it afterwards:

| group | monotonic | where it breaks |
|---|---|---|
| `0p4/00z` | **no** | `… 2024-02-27, 2024-02-28, 2023-12-06, 2023-12-07` — an 83-date block landed ahead of two earlier dates |
| `49r1/00z` | **no** | `… 2026-05-11, 2026-05-12, 2026-05-06` — one date backfilled after a later batch |
| `50r1/00z` | yes | clean |

Consequences, both verified:

- `ds.sel(time=slice(a, b))` **raises** `KeyError: Value based partial slicing
  on non-monotonic DatetimeIndexes` on `0p4` and `49r1`. It does not silently
  return the wrong span — the failure is loud.
- `t[0]` and `t[-1]` are **not** the min and max. `49r1`'s last positional entry
  is 2026-05-06 while its true maximum is 2026-05-12. Any code deriving a date
  range from the ends of the axis is wrong on these two groups; sort first.
- Exact-label `ds.sel(time=np.datetime64(d))` is **unaffected** — ordering does
  not enter an exact lookup. That is why `read_message` works and why the June
  2026 realization (50r1, clean) never surfaced this.

**The fix, for a reader, is one line** — no re-scan, no rewrite:

```python
ds = ds.sortby("time")     # monotonic; slicing works; reads still resolve
```

**Cause, confirmed from the store's own commit log.** Each commit message
carries `<group> <YYYYMMDD>: ... (time index N)`, so the log is a full record of
what went where. Auditing all 1,246 date-commits:

| | `0p4` | `49r1` | `50r1` |
|---|---|---|---|
| two different dates on one time index | **0** | **0** | **0** |
| dates committed more than once | 0 | 0 | 0 |
| `index == write order` | yes | yes | yes |
| backfills (written after a later date) | 2 | 1 | 0 |

**Nothing was lost or overwritten.** The whole defect is three backfill commits
made on 2026-07-07:

```
0p4   20231206 -> index 399   (written after 20240228)
0p4   20231207 -> index 400   (written after 20240228)
49r1  20260506 -> index 793   (written after 20260512)
```

Those three displaced 85 dates in `0p4` and 7 in `49r1` from their sorted
position. A backfill of an *earlier* date has nowhere to go but the tail — that
is all this is. It is **not** a speed/ordering tradeoff; nothing about read
throughput requires it.

`build_corpus.py` never takes that path: it preallocates the whole coordinate
sorted and writes regions into it, so the sink stores cannot acquire this defect
no matter what order dates are committed in.

**source.coop is NOT behind the GCS original.** The counts reconcile exactly:

```
0p4    407 days - 6 never published = 401    store has 401
49r1   804 days - 10 defective pars = 794    store has 794
50r1    51 days                     =  51    store has  51
                                      1,246  = 1,256 buildable - 10 defective
```

and `ecmwf/icechunk-par/era_check.py` §F puts the full 00z corpus at 1,256. So
the mirror is complete and the ten March dates are the only absentees.

**Why those ten are missing — they were rejected, not missed.**
`ecmwf/icechunk-par/BACKFILL_RETRY_RUNBOOK.md` §4–5 documents them: the pars for
`20260319`, `20260322`–`20260327`, `20260329`–`20260331` have **pl chunk keys
missing the level segment**, so all 13 pressure levels collapsed onto one
arbitrary message (decoded: `t`→300 hPa, `gh`→400 hPa — random survivors), and
12 of 13 levels are simply absent. `build_ecmwf_icechunk.py:103-111` detects
this up front and refuses the date rather than writing silently-wrong refs.

> ### UPDATE 2026-08-07 — the ten dates are FIXED in the GCS store
>
> Root cause confirmed: the pars were generated 2026-04-08 against the
> **pre-per-level-fix** template, so every pl chunk key lost its level segment.
> The three March dates that *were* in the store (`20260320/21/28`) had been
> regenerated 2026-06-22/23, after the fix — same window, different template.
>
> Regenerated all ten with `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz`:
>
> | | refs | pl refs | malformed | levels |
> |---|---|---|---|---|
> | `20260320` (known good) | 13,175 | 9,945 | 0 | 13 |
> | the ten, **before** | 3,995 | 765 | **765** | **0** |
> | the ten, **after** | 13,175 | 9,945 | 0 | 13 |
>
> Then folded into `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens`:
> **1,257 commits** (was 1,247), **49r1 axis 804** (was 794),
> **March 2026 31/31**, **zero interior gaps in the whole era**.
>
> Verified by decode, not just by structure — `t` at 500/700/850 hPa returns
> **266.7 / 282.3 / 290.1 K**: distinct, and increasing with pressure. A
> collapsed-level par returns three *identical* values, so this is the check
> that actually proves the fix.
>
> **Two things this did NOT fix:**
> - **source.coop still lacks them.** The table above describes the *published*
>   store, which is unchanged until `mirror_gcs_to_source_coop.py` is re-run.
> - **HuggingFace `E4DRR/gik-ecmwf-par-v2` still holds the defective pars** —
>   verified: `20260319` there is still `malformed=765, levels=0`. Anyone
>   rebuilding from HF reintroduces the bug. This is also why
>   `backfill_all_eras.py` could **not** be used: it fetches pars from HF. The
>   fix was applied by driving `build_ecmwf_icechunk.py` directly — the same
>   builder that driver shells out to — with pars pulled from the GCS path the
>   HF catalog itself records as `gcs_path`. Re-mirroring to HF needs a write
>   token.
>
> Rollback: reset `main` to `2PAFARG4ADKVXFKWHDD0` to undo all ten commits.
>
> Remaining to make `ea-cgan/v4-mar2026-49r1` complete: re-mirror to
> source.coop, then `build_corpus.py --resume` fills the ten reserved slots.

**They cannot be filled from the existing pars.** The chain is:

1. regenerate the pars — `run_lithops_ecmwf.py` for those 10 dates at 00z, era
   49r1 (`ECMWF_REFERENCE_DATE=20250515`, `ECMWF_RESOLUTION=0p25`,
   `ECMWF_CONTROL_STREAM=enfo`, runtime tag `:49r1`), mirror to HF/GCS
2. `backfill_all_eras.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens`
   — resume finds the gaps and folds them in as out-of-order appends
3. re-mirror the store to source.coop
4. `build_corpus.py --resume` fills the ten reserved slots in
   `ea-cgan/v4-mar2026-49r1`

Step 1 needs the **deployer** identity
(`ecmwf-lithops-deployer@e4drr-crafd`); the `coiled-data-e4drr` key is 403 on
`gik-ecmwf-aws-tf` and `gik-gefs-aws-tf` and cannot list buckets in the project.

**2. Date coverage, all three eras.** Full audit against `s3://ecmwf-forecasts/`:

| era | span | dates | interior gaps | at source? | channels usable |
|---|---|---|---|---|---|
| `0p4` | 2023-01-18 → 2024-02-28 | 401/407 | **6** — 2023-04-27…05-02 | **no** — upstream | 22/30 |
| `49r1` | 2024-02-29 → 2026-05-12 | 794/804 | **10** — March 2026 | **yes** — ours | 30/30 |
| `50r1` | 2026-05-13 → 2026-07-02 | 51/51 | 0 | — | 30/30 |

**The ten March 2026 dates are the only recoverable gap in the archive:**

```
03-19, 03-22, 03-23, 03-24, 03-25, 03-26, 03-27, 03-29, 03-30, 03-31
```

Each has 177 objects and 87 `.index` files at source, identical to a date that
resolves — so this is a Stage 1 gap. Fixing it means re-scanning those dates and
re-mirroring. Until then `ea-cgan/v4-mar2026-49r1` holds 21 of 31, the other ten
reserved as unwritten slots.

Everything else is not fixable or not a gap:

- **`0p4`'s six days are upstream.** `20230427/` … `20230502/` return zero
  objects even at the bare date prefix. Verified not to be a wrong-path
  artifact: 2023-04-26 and 2023-05-03 both return 177 objects at
  `{date}/00z/0p4-beta/enfo/`.
- **Era boundaries are contiguous** — no date falls between eras.
- **~35 days of tail staleness** after 2026-07-02. The last commit is
  2026-07-07; the scan simply has not run since. Not corruption.

**`0p4` also publishes only 19 variables** (49r1 has 59), so 8 of the 30
channels do not exist there — `ssr`, `ttr`, `tcw`, `mucape`, and all four `w`
levels (`w` is absent entirely). Its 9-level set covers every level the channel
table uses, so the losses are variables, not levels. Realizing 0p4 would need
era-aware channel trimming, which 49r1 and 50r1 do not.

#### How big are they? (three different numbers -- don't conflate them)

| measure | GEFS | ECMWF | what it means |
|---|---|---|---|
| store objects on source.coop | 5.4 GB | 15 GB | what is actually hosted (manifests/snapshots) |
| **referenced GRIB** (packed) | ~92 TB | **~620 TB** | bytes you'd pull reading the store once |
| dense float32 if materialized | 697 TB | 2.79 PB | `ds.nbytes` -- **misleading**, assumes every cell exists, uncompressed |

`ds.nbytes` overstates real data volume ~7x (GEFS) to ~4.5x (ECMWF): GRIB2 is packed,
and the dense product counts cells that have no GRIB message. The ECMWF store
references 97-100% of what ECMWF publishes in `enfo` for those dates (~627 TB).

### Open a store (anonymous smoke test)

Each script opens the store with zero credentials and decodes one field to prove
it resolves end to end. Deps are declared inline (PEP 723), so `uv run` handles
everything — run with **no** `AWS_*` env vars set:

```bash
uv run smoke_test_published_gefs.py     # GEFS  (group 0p25/00z)
uv run smoke_test_published_ecmwf.py    # ECMWF (all three eras)
```

Open takes ~60 s (eager manifest preload is disabled so it survives source.coop's
sporadic 500s — see the docs below). Both print the virtual dataset size and end
with `RESULT: PASS`.

See **`OPENING_PUBLISHED_GEFS_ICECHUNK.md`** and
**`OPENING_PUBLISHED_ECMWF_ICECHUNK.md`** for the minimal open snippet to drop
into your own code, plus gotchas (group paths, the `gribberish.zarr` codec import,
`force_path_style`, per-era variable names).

### Publish / mirror a store to source.coop

`mirror_gcs_to_source_coop.py` copies a GCS-hosted store to source.coop by plain
object copy (resumable; skips objects already in sync). Needs GCS read creds and a
source.coop STS token (`.env`, ~1 h lifetime):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./coiled-data.json   # GCS read
source .env                                                # source.coop STS
uv run mirror_gcs_to_source_coop.py \
    --gcs-store gs://gik-gefs-aws-tf/icechunk/gefs-ens \
    --dest      forecasts/noaa_gefs_aws_s3_icechunk_vd \
    --sa-key    ./coiled-data.json --threads 32
```

If the STS token expires mid-run, refresh `.env` and re-run — it resumes.

---

# Stage 2 — realizing a subset on EWC

Reads the ECMWF store above and writes a realized Icechunk store to EWC Ceph,
using a [Frisky](https://getfrisky.dev/) futures DAG across the six EWC worker
VMs.

### Throughput actually achieved

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

#### On the 1 GB/s target

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

#### The 95% you cannot avoid

**Only 4.9% of the bytes pulled are kept** — 94.6 GB retained out of 1,917 GB
pulled. This is not waste in the tuning sense: a GRIB message is the atomic
unit of the store, so fetching the East Africa box (163×147) still means
pulling the whole global message (721×1440) and discarding the rest. No amount
of concurrency or chunk tuning changes it; only a differently-encoded source
would.

It does mean the *useful* output rate is 8.9 MB/s while the *network* rate is
181 MB/s. Size the network for the latter.

---

### Environment

Deliberately **not** `/opt/mamba/envs/dask`. That env is version-pinned to all
six Dask workers (`dev-test/README.md` §1) and must not drift.

```bash
cd ~/grib-index-kerchunk/icechunk-dask

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
P=~/grib-index-kerchunk/icechunk-dask/.venv/bin/python
D=/opt/mamba/envs/dask/bin/python     # only for deploy_workers.py
```

`deploy_workers.py` runs under `$D` because it talks to the **Dask** cluster,
which is used purely as a remote-exec channel (there is no SSH key on the
gateway).

---

### Credentials

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

#### How the workers get write permission

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

### A-to-Z runbook

#### Step 1 — build the venv on all six VMs (~40 s, once)

```bash
$D deploy_workers.py --install
```

Creates `/tmp/frisky-ea/.venv` on each VM from the pinned interpreter,
installing nothing into `/opt/mamba/envs/dask`. Also ships
`frisky_daily_dag.py`, which the workers need because **Frisky pickles task
functions by reference** — the defining module must be importable there.

#### Step 2 — scheduler on a worker VM, then the workers

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

#### Step 3 — dashboard (optional)

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

#### Step 4 — one day

```bash
$P parallel_write.py --sink ea-cgan/v2-7day --date 2026-07-02 \
     --channels 30 --members 51 --steps 53 --fork-once --env .env
```

`--fork-once` forks a single session and pickles it to every task. Forking per
block costs ~0.30 s each — 476 s of the 494 s a date took before.

Expect ~340 s, ~1,590 blocks, one commit.

#### Step 5 — a month

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

#### Step 6 — verify

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

### Reading the result

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

### Files

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

### Corrections

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
