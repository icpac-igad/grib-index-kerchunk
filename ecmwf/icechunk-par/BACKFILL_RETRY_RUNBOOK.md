# Backfill & retry-missing-dates runbook (ECMWF + GEFS Icechunk stores)

How to run the par→Icechunk backfills, monitor them, and retry the dates that
fail — including how to tell a retryable failure from a defective par that
needs upstream regeneration. Everything here was exercised live on the first
full ECMWF run (2026-07-06/07, observed results at the bottom).

## The two stores and their drivers

| | ECMWF | GEFS |
|---|---|---|
| store | `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4` (12 groups: `{0p4,49r1,50r1}/{00,06,12,18}z`) | `gs://gik-gefs-aws-tf/icechunk/gefs-ens` (group `0p25/00z`) |
| pars source | `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf` (**GCS, not HF** — HF still holds the defective March-2026 pars) | `gs://gik-gefs-aws-tf/run_par_gefs` (GCS tree is authoritative) |
| driver | `backfill_all_eras.py` (sequential) or `backfill_parallel.py` (frisky/local fan-out) | `gefs/backfill_gefs_icechunk.py` |
| log | whatever you redirect to; `~/ecmwf-rebuild/logs/` by convention | `gefs/backfill_gefs.log` (written by the script) |
| corpus | 5,023 (date, run) pairs — 1,257 dates × 4 runs, minus gaps | 2,031 dates |
| measured | sequential 34–60 s/date; parallel 15–29 s/date | ~10–17 s/date — ~7–8 h total |

Which ECMWF driver: `backfill_parallel.py` is ~2–4× faster but needs the frisky
cluster (6 EWC VMs), a pushed key and pushed code, and it is the one that has to
be babysat — see `HANDOVER_PAR_TO_ICECHUNK_SCALING.md` §5. `backfill_all_eras.py`
needs nothing but this host: one date per subprocess, one commit per date, flat
coordinator memory, so no recycle loop and nothing to keep alive.

## 1. Launch (upload)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/frisky-ea/gcs-key.json

# ECMWF, sequential -- run-outer, smallest era first, resumable
cd grib-index-kerchunk/ecmwf/icechunk-par
setsid nohup uv run backfill_all_eras.py \
    --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
    --sa-key /tmp/frisky-ea/gcs-key.json \
    --era 50r1,0p4,49r1 --run 12,18 > ~/ecmwf-rebuild/logs/serial.log 2>&1 &

# GEFS (logs itself to backfill_gefs.log)
cd grib-index-kerchunk/gefs
nohup uv run backfill_gefs_icechunk.py \
    --store gs://gik-gefs-aws-tf/icechunk/gefs-ens > /dev/null 2>&1 &
```

Every ECMWF group must already exist with a **preallocated** time axis; that is
what gives each date a fixed index. Create it once per (era, run):

```bash
uv run build_ecmwf_icechunk.py --era 49r1 --run 12 --create-schema \
    --dates-file ~/ecmwf-rebuild/dates/dates-49r1-12z.txt --store gs://...-v4
```

All drivers are **safe to kill at any point**: one commit per date, and on
restart they skip what is already there. Disk use is one date of pars (~25 MB),
deleted after each commit. `setsid` is not optional — a plain `nohup ... &` from
a short-lived shell gets reaped when the parent exits.

## 2. Monitor

```bash
tail -f serial.log              # per-date: "[N/M] era/RUNz DATE (t=i): ok in Ns (ETA H h)"
grep -c "ok in" serial.log      # dates completed
grep "FAILED" serial.log        # failures so far (driver continues past them)
kill -0 $(cat ~/ecmwf-rebuild/serial.pid) && echo alive   # pidfile, NOT pgrep

# per-group holes, straight from the manifest (the ground truth):
uv run verify_store_completeness.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
    --era 49r1 --run 12 --sa-key /tmp/frisky-ea/gcs-key.json

# full store health (read-only, safe beside the live writer):
uv run check_store_health.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
    --expected-dates 1256 --decode --log serial.log
# GEFS: --store gs://gik-gefs-aws-tf/icechunk/gefs-ens \
#       --container s3://noaa-gefs-pds/ --expected-dates 2031
```

The health check verifies: commits advancing (+rate/ETA), per-group time axes
unique, **every array sized to the group's time axis** (catches mid-era
schema-drift bugs), a spot decode of the newest date's refs, and a FAILED scan
of the log. At the end of a run the driver prints the failed-dates list:

```
done: 1239 built, 13 failed in 20.57 h
failed dates (re-run to retry): ['20231206', '20231207', '20260319', ...]
```

## 3. Retry the missing dates

**The retry is the same command as the launch.** For each group the driver reads
the preallocated time axis, probes the manifest at every index
(`build_ecmwf_icechunk.date_written`), and rebuilds only the slots with no refs:

```bash
uv run backfill_all_eras.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
    --sa-key /tmp/frisky-ea/gcs-key.json --era 49r1 --run 06
# GEFS identical: uv run backfill_gefs_icechunk.py --store gs://gik-gefs-aws-tf/icechunk/gefs-ens
```

A missed date is refilled **at its own index**, so nothing goes out of order —
the sorted-axis defect that the legacy append-mode store had is impossible here.
The scan costs ~0.1 s/date (115 s for an 804-date group), and it is the only
thing that runs before work starts.

Two traps that cost days on the parallel driver, both still live here:

- **A running coordinator holds its code in memory.** Fixing `date_written` on
  disk changed nothing for the process already running: it kept re-declaring the
  same 18 dates unwritten, rebuilt them, exited, re-probed, and looped — 44 times
  on 49r1/00z, and again on 49r1/06z until it was stopped on 2026-08-15. After
  any code change, **stop and relaunch**; do not assume a live run picked it up.
- **A spinning run looks healthy.** Each loop logs a fresh batch of the same
  dates. Read the `N to build / M already written` line across restarts — if the
  same N keeps reappearing, it is looping, not progressing.

## 4. Triage: not every failure is retryable

Read the FAILED lines before retrying. Two classes seen in practice:

| symptom in log | cause | action |
|---|---|---|
| `503 Server Error`, `Connection broken`, timeouts | transient HF/GCS network | **just re-run the driver** — succeeds on retry |
| `DEFECTIVE PAR ...: pl chunk keys lack the level segment` | upstream par-generation bug: the level was dropped from the chunk key, so all 13 pressure levels collapsed onto one arbitrary message (verified by decoding the refs: t→300 hPa, gh→400 hPa, essentially random survivors). 12 of 13 levels are simply absent from the par. | **cannot be fixed here** — regenerate that date's pars upstream (`run_lithops_ecmwf.py` for the date, mirror to HF/GCS), then re-run the driver |

The ECMWF builder detects the defective-par case up front and exits with the
explicit message above (instead of a pandas traceback), so the two classes are
distinguishable straight from the log.

## 5. Observed on the first full ECMWF run (2026-07-06/07)

- 1,252 attempted → **1,239 built, 13 failed** in 20.57 h.
- 3 transient: `20231206`, `20231207` (one HF CDN 503 window), `20260506`
  (connection reset) → recovered by the retry run as out-of-order gap fills.
- 10 defective pars: `20260319`, `20260322`–`20260327`, `20260329`–`20260331`
  (a contiguous late-March-2026 par-generation bug window) → **pending
  upstream regeneration**; the retry run now fails them cleanly with the
  DEFECTIVE PAR message. After regenerating, the same retry command folds
  them in.
- Mid-run schema drift (49r1 vars appearing/disappearing across 2024→2026)
  was absorbed automatically: the builders resize every array on each append,
  so groups stay openable throughout.

## 6. The v4 rebuild (2026-08-14 →)

`icechunk/ecmwf-ens-v4` is the post-longitude-fix store, 12 preallocated groups,
built from GCS pars. State on 2026-08-15:

| run | 50r1 (52) | 0p4 (~400) | 49r1 (804) |
|---|---|---|---|
| 00z | done | done | done |
| 06z | done | done | done — last 18 dates finished by the sequential driver |
| 12z | done | in progress | queued |
| 18z | done | queued | queued |

The 12z/18z remainder (2,406 dates) is running under `backfill_all_eras.py`
rather than the frisky driver, after 49r1/06z spun on a stale-code coordinator.
Measured on the same store, same pars: **37 s/date** for 49r1/06z (49 steps),
34 s/date for 0p4/12z (85 steps) — about 28 h for what is left, against roughly
10 h for the parallel path with the cluster kept healthy.
