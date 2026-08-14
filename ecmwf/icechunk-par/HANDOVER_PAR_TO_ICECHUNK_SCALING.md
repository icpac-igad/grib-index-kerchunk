# Handover — making the par → Icechunk conversion fast enough to redo

**Written 2026-08-14.** What was tried, what the clock actually said at each
step, and which changes moved it. The goal throughout: **distribute the par
conversion so the whole archive can be rebuilt in hours, not days** — because a
rebuild stopped being hypothetical the moment the longitude bug landed
(`HANDOVER_LONGITUDE_FIX.md`).

Read that document first if you have not. This one is only about speed and the
machinery that delivers it.

---

## 1. Where the estimates started, and where they ended

| stage | scope | rate | total |
|---|---|---|---|
| `CONVERSION_ESTIMATE.md` projection (2026-07-05) | 1,256 dates, 00z | 11–18 s/date | **5.2 h** |
| **actual serial run** (`backfill_all_eras.py`, 2026-07-06/07) | 1,252 dates, 00z | **59.8 s/date** | **20.64 h** |
| parallel driver, local pool, no pipelining | 50r1 | 14.7 s/date | — |
| \+ pipelined commit | 0p4 | 7.5 s/date | — |
| \+ frisky (6 EWC VMs) | 0p4 | **6.2 s/date** | — |
| \+ member streaming (chunk 8) | 0p4, local bench | **4.5 s/date** | — |
| **scope also grew 4×** | 5,023 date-runs, all four runs | | |

The July estimate was 4× optimistic because it measured a **local** store. Every
commit in the real run goes to GCS, and that is where the time is.

Corpus for scale: **5,023 (date, run) pairs, 256,173 pars, 2,215 M virtual
refs.** 00z alone is 1,257 dates / 64,107 pars.

---

## 2. What actually moved the clock

### 2.1 Preallocated time axis — the precondition for everything

`--create-schema` writes the whole `time` coordinate up front, sorted. A date
then writes at *its own index* rather than appending. This is not itself a
speedup; it is what makes dates independent:

- no derived `ti` from `tarr.shape[0]`, so two dates cannot collide;
- no per-date resize of ~40 shared arrays (metadata, which chunk-disjointness
  does not protect);
- restart is exact, so a killed run resumes rather than redoing;
- the out-of-order time axis that plagued the old store becomes impossible.

### 2.2 Fan-out via fork/merge

`session.fork()` per date → worker sets virtual refs at its own indices →
coordinator `merge()`s and commits once per batch. Legal because every index
`[ti, number, step, level, 0, 0]` is unique per date.

### 2.3 Pipelining the coordinator (~2×)

Fanout and merge+commit measured ~27 s each at batch 4 and were strictly serial.
The commit now runs on a single-thread executor while the next batch fans out, so
batch time is `max(fanout, commit)` instead of their sum. Concurrent sessions
conflict on commit and rebase cleanly — verified directly, two sessions writing
different indices, second rebases, both survive.

### 2.4 Frisky workers (memory, not throughput)

`--executor frisky` runs the parse on six EWC VMs (15.6 GB each) instead of the
gateway, whose cgroup caps at **8 GB** — `free` reports 31 GB and is misleading.
Fanout 20–24 s → 8–9 s, and gateway memory stopped climbing into OOM.

It does **not** remove the ceiling: changesets still return to the coordinator to
be merged, ~110 bytes/ref. Batch size is bounded by *coordinator* memory, not
worker memory.

### 2.5 Member streaming — the stage-1 grain

Stage 1 (Lithops) emits **one par per member**, because that is the grain of the
data. `load_date_refs` concatenated all 51 into a 654,585-row frame, and
`ref_specs` re-split it by variable one line later:

    peak 1,328 MB for a 101 MB resident result

`stream_date_refs` yields a few members at a time. One member per yield is the
wrong extreme — 46 `set_virtual_refs` calls become 2,346:

    members/chunk    1      4      8     17     51
    spec build s   7.1    4.4    4.0    3.8    3.7
    RSS delta MB     0     13     18     32    105

Parse time is flat; the concat was never the time cost, only the memory. **8 is
the default**: ~83% of the saving for ~8% of the time.

---

## 3. What is measured, not guessed

| quantity | value | how |
|---|---|---|
| gateway cgroup cap | **8 GB** | `memory.max`; `free` says 31 GB |
| fork changeset | **110 bytes/ref** | pickling a real `ForkSession` |
| store on disk | **21.8 bytes/ref** | 15.31 GB / 703 M refs |
| par corpus | **67.17 GB** | full bucket listing |
| old 00z store | **15.31 GB** | = **0.74×** its own pars |
| commit throughput | **~12 MB/s**, flat vs batch size | 1/2/4-date commits |
| manifests per date | **37** (one per array) | 46,664 / 1,257 |
| single-stream upload | 13.2 MB/s; 4 parallel 35.7 | timed PUTs |

**There is no storage blow-up.** An earlier claim of 244 GB came from
extrapolating a *pickle* to on-disk size and was wrong by 5×. The virtual store
is smaller than the pars it references.

---

## 4. What does NOT help (each tested)

- **More workers, as configured.** `--batch` sets both commit granularity and
  fan-out width, so batch 2 means 2 of 48 slots busy. Decouple them before
  adding workers.
- **Distributing the commit.** Concurrent commits to one branch *degrade*:
  2.32/s at 1 committer → 1.59/s at 8, retries 0 → 160, and at 16 GCS rejected
  the branch-ref PUT outright. The ref is a single hot object.
- **Bigger batches for commit throughput.** Flat at ~12 MB/s from 1 to 4 dates.
- **Moving workers to cut network.** A fork returns 72 MB per 49r1 date;
  ~245 GB across the corpus. Stage 3's fork/merge keeps bulk *off* the client —
  stage 2's changeset *is* the bulk.

---

## 5. Failure modes, all hit at least once

| symptom | cause | fix |
|---|---|---|
| OOM at 8 GB | `free` reports the host, not the cgroup | `memory_budget_workers()` reads `memory.max` |
| one OOM became four | `ProcessPoolExecutor` spawn children survive a SIGKILLed parent — 18 orphans, 2.9 GB | `PR_SET_PDEATHSIG` in `init_worker` |
| memory to 8.19 GB over hundreds of batches | `frisky.Future` holds its result — a whole changeset — and only drops on GC | `f.release()` in a `finally` |
| every date `EACCES` | `--staging` was a *coordinator* path; it is used **on the worker** | default `/tmp/frisky-ea/staging` |
| 3 groups "succeeded" with 0 dates | driver returned 0 even when every date failed | `main()` returns 1; driver script aborts |
| pool hang, no output, no error | `ProcessPoolExecutor` defaults to `fork`; icechunk holds a tokio runtime | `mp_context=spawn` |
| refs silently dropped | worker returned counts, not the fork — the changeset lives in it | return the fork |
| `KeyError: 'backfill_parallel'` on every worker, then scheduler channel resets | `push_worker_code.py` popped `sys.modules` while fanned out 36-wide over 4-thread processes | `flock` (serialises processes *and* threads) |
| 2 of 6 hosts silently ran stale code | second worker process on a host saw `changed=False` and skipped its own eviction | evict unconditionally; report a per-PID SHA |
| a dead run looked alive for 25 h | `pgrep -f run_rebuild.sh` matches its own wrapper; buffered stdout died with the process | pidfile + `python -u` |
| healthy run killed twice | pushing code / running experiments **mid-flight** | deploy only at a recycle boundary |

Two of those were self-inflicted disruptions of a working run. **Do not push code
or run heavy experiments while a conversion is in flight.** The recycle loop
gives frequent, free boundaries.

---

## 6. How to run it

Working dir is `~/ecmwf-rebuild` (logs, date lists, `run_rebuild.sh`). The code
lives in `grib-index-kerchunk/ecmwf/icechunk-par`.

### Step 0 — cluster, code, credentials (once, and after any VM reboot)

```bash
# deploy_workers.py imports `distributed`, which the gateway does not have.
# Install it in a THROWAWAY venv, never into /opt/mamba/envs/dask -- the frisky
# .venv is a symlink to that shared conda env.
python3 -m venv /tmp/dw-venv && /tmp/dw-venv/bin/pip install -q distributed

# --start clears any running workers first, so it IS the restart. It leaves the
# scheduler alone. 6GB, not 10: two workers per 15.6 GB VM must not oversubscribe.
/tmp/dw-venv/bin/python deploy_workers.py --start --scheduler 192.168.1.74:8796 \
    --workers-per-vm 2 --nthreads 4 --memory-limit 6GB

python push_worker_key.py     # GCS key -> /tmp/frisky-ea/gcs-key.json, 0600
python push_worker_code.py    # modules -> /tmp/frisky-ea/gik
```

`push_worker_code.py` prints a **SHA per worker PID** and exits non-zero if any
is stale. Read that output. Both worker processes on a VM must appear, both `OK`
— an earlier version reported "unchanged" and left half the cluster on old code.

The coordinator needs the key at the **same path the workers use**, because a
`ForkSession` carries the coordinator's `service_account_file` path:

```bash
install -m 600 icechunk-par/coiled-data.json /tmp/frisky-ea/gcs-key.json
mkdir -p /tmp/frisky-ea/staging          # --staging is used ON THE WORKER too
```

### Step 1 — date lists, from the bucket

```bash
uv run make_run_date_lists.py --out ~/ecmwf-rebuild/dates/
```

Not from `catalog.parquet` — it lists 00z only. Era is assigned per (date, run),
because every era transition happens at 06z.

### Step 2 + 3 — schema then convert, all twelve groups

```bash
cd ~/ecmwf-rebuild
LIMIT=200 RETRIES=3 setsid nohup ./run_rebuild.sh > logs/rebuild.log 2>&1 < /dev/null &
```

`setsid` is not optional. A plain `nohup ... &` from a short-lived shell here gets
reaped when the parent exits — two launches died that way before this was found.

The script loops **run-outer**: all of 00z across 50r1/0p4/49r1, then 06z, and so
on, so the run everyone consumes finishes first. It recycles the coordinator every
`LIMIT` dates, retries a group `RETRIES` times on a transient frisky channel drop,
and aborts the sequence on a persistent failure.

To do one group by hand:

```bash
uv run build_ecmwf_icechunk.py --era 49r1 --run 06 --create-schema \
    --dates-file dates/dates-49r1-06z.txt --store gs://.../ecmwf-ens-v4 \
    --sa-key /tmp/frisky-ea/gcs-key.json

uv run backfill_parallel.py --era 49r1 --run 06 --store gs://.../ecmwf-ens-v4 \
    --par-source gcs --sa-key /tmp/frisky-ea/gcs-key.json \
    --executor frisky --scheduler 192.168.1.74:8796 \
    --worker-sa-key /tmp/frisky-ea/gcs-key.json \
    --staging /tmp/frisky-ea/staging --batch 4 --limit 200
```

### Step 4 — check on it

```bash
kill -0 $(cat ~/ecmwf-rebuild/rebuild.pid) && echo alive     # NOT pgrep
grep -E "^=== |^    rc=|ABORT|retry|^done\." logs/rebuild.log
awk '/VmRSS/{print int($2/1024)" MB"}' /proc/$(pgrep -f "backfill_parallel.py --era"|head -1)/status
```

**Use the pidfile.** `pgrep -f run_rebuild.sh` matches its own shell wrapper and
will report a long-dead run as alive — that cost 25 hours of believing a stopped
job was progressing. Logs are unbuffered (`python -u`), so a killed process leaves
its error behind; without that, buffered output dies with the process and the log
stops mid-batch with no explanation.

### Knobs

| var | default | raise/lower when |
|---|---|---|
| `LIMIT` | 200 | lower to 100 if the pre-recycle peak nears 8 GB; costs one resume probe (~0.25 s/date) |
| `BATCH_0P4` | 6 | 0p4 dates are ~355k refs |
| `BATCH_BIG` | 4 | 49r1/50r1 are ~654-712k refs; batch bounds COORDINATOR memory, not worker |
| `RETRIES` | 3 | transient channel drops |
| `ERAS` / `RUNS` | `$1` / `$2` | `./run_rebuild.sh "49r1" "00 06"` for a subset |

### Rules learned the hard way

- **Never touch the cluster while a conversion runs.** Pushing code, sampling
  pars, or running experiments against the same scheduler killed three healthy
  runs. Deploy at a recycle boundary or against a stopped run.
- **Never edit a `.py` a live coordinator is executing.** Python reads source at
  traceback time, so the traceback will quote lines that were never run.
- Everything under `/tmp/frisky-ea` on the VMs is lost on reboot — key, modules,
  staging. Re-push all three.
- `--par-source gcs`, never `hf`: **HuggingFace still holds the defective pars**
  for 20260319 / 20260322-27 / 20260329-31 (collapsed pl levels).

---

## 7. Where it stands and what is next

Conversion restarted **2026-08-14 on a fresh prefix `icechunk/ecmwf-ens-v4`**, so
every group is built by one method. `ecmwf-ens-v3` mixed concat-built and
stream-built groups; the outputs were verified byte-identical, but a store with
one provenance is worth more than the 8 groups it saved. `ecmwf-ens` (the
original, longitude-displaced) and `-v3` are both left in place — nothing has
been deleted or swapped.

Projection: **0p4 ~2.7 h, 49r1 ~7–10 h, 50r1 ~0.7 h.**

Next, in order of leverage:

1. **Decouple fan-out width from commit batch.** Batch 2–4 uses 2–4 of 48 slots.
   Submit K dates, commit in groups as they land. Biggest single win left.
2. **Move the coordinator to a 15.6 GB VM.** The 8 GB gateway is what forces the
   small batch. Needs GCS creds there and a way to launch a long-lived process —
   there is no SSH key on the gateway, only `client.run()`.
3. **Member-grained tasks.** One date is 51 members; at that grain a single date
   saturates the cluster and each task returns ~1.4 MB instead of 72 MB.
4. **Twelve stores, one per group** — the only way past the serial commit, at the
   cost of twelve URLs instead of one.

### Retracted: the "5,114 short member-pars" figure

An earlier version of this document claimed **5,114 member-pars (2.0%) are
missing at least one step, across 78% of dates**. **That was an artifact of the
screening method. Do not act on it.**

The screen compared each par's file size against the median of its (date, run)
group. Par size varies for two benign reasons that swamp the signal:

- **Control is always the smallest par.** A par stores `[url, offset, length]`
  as text, and control is the first message block in each GRIB, so its byte
  offsets are the smallest numbers in the file. Shorter strings, smaller
  parquet. That is why control was 155 of 233 flagged pars at 00z where uniform
  would predict 4.6, and why the ranking tracked member number.
- **Par size tracks the pl-level regime.** Normalising each member against
  itself across dates instead flags every 9-level date inside the 49r1 era at
  ~0.55x a 13-level median — 37.9% of the corpus.

Direct measurement of step counts, which is what the claim was about:

| test | result |
|---|---|
| 8 pars from the top of the flagged 00z list | **all 85/85 steps, complete** |
| 48 random (date, run, member) pars corpus-wide | **0 short** |
| 95% CI on the true short-par rate | **0.0% – 0.0%** |

The compounding error: the "8 of 8 sampled were genuinely short" check that gave
the figure its confidence sampled the *most extreme* flagged ratios — the tail,
not the population. It confirmed the screen's worst cases and said nothing about
the base rate.

**What is actually confirmed**, by reading step counts rather than sizes:
`20240229 18z control` is missing step 0, and that data IS present in the source
`.index` as a `type=cf` record — so it is a par-generation gap, not an ECMWF gap.
A handful of 06z/18z members at 44–48 steps were seen while validating the
screen. Real, but few, and **there is no defensible estimate of how many more
exist.**

**The right way to answer this** is the store's own manifest, not par sizes:
`Session.chunk_coordinates(array_path)` enumerates exactly which chunks exist, so
one pass over a surface variable per group yields every missing
`(time, number, step)` directly — no sampling, no proxy. That check is not built
yet and is the outstanding work here.

The general lesson, and it is the same one as the longitude bug: a proxy
measurement that correlates with the thing you care about is not the thing you
care about. Verify against the quantity itself before quoting a number.
