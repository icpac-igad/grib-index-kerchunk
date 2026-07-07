# Backfill & retry-missing-dates runbook (ECMWF + GEFS Icechunk stores)

How to run the par→Icechunk backfills, monitor them, and retry the dates that
fail — including how to tell a retryable failure from a defective par that
needs upstream regeneration. Everything here was exercised live on the first
full ECMWF run (2026-07-06/07, observed results at the bottom).

## The two stores and their drivers

| | ECMWF | GEFS |
|---|---|---|
| store | `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens` (groups `0p4/00z`, `49r1/00z`, `50r1/00z`) | `gs://gik-gefs-aws-tf/icechunk/gefs-ens` (group `0p25/00z`) |
| pars source | HF `E4DRR/gik-ecmwf-par-v2` (catalog.parquet) | `gs://gik-gefs-aws-tf/run_par_gefs` (GCS tree is authoritative) |
| driver | `ecmwf/icechunk-par/backfill_all_eras.py` | `gefs/backfill_gefs_icechunk.py` |
| log | `ecmwf/icechunk-par/backfill.log` (stdout redirect) | `gefs/backfill_gefs.log` (written by the script) |
| corpus | 1,256 dates | 2,031 dates |
| measured | ~25–120 s/date (grows with era refs) — 20.6 h total | ~10–17 s/date — ~7–8 h total |

## 1. Launch (upload)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json

# ECMWF
cd grib-index-kerchunk/ecmwf/icechunk-par
nohup uv run backfill_all_eras.py \
    --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens > backfill.log 2>&1 &

# GEFS (logs itself to backfill_gefs.log)
cd grib-index-kerchunk/gefs
nohup uv run backfill_gefs_icechunk.py \
    --store gs://gik-gefs-aws-tf/icechunk/gefs-ens > /dev/null 2>&1 &
```

Both drivers are **safe to kill at any point**: one commit per date, and on
restart they re-read the store's per-group time axes and skip everything
already committed. Disk use is one date of pars (~25 MB), deleted after each
commit. First-time note: the store must be built in chronological order, so
never hand-commit ad-hoc test dates into a group before its backfill.

## 2. Monitor

```bash
tail -f backfill.log            # per-date lines: "[N/M] era DATE: ok in Ns (ETA H h)"
grep -c "ok in" backfill.log    # dates completed
grep "FAILED" backfill.log      # failures so far (driver continues past them)

# full store health (read-only, safe beside the live writer):
uv run check_store_health.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens \
    --expected-dates 1256 --decode --log backfill.log
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

**The retry is the same command as the launch.** Resume logic finds the gaps
(catalog/GCS dates not in the store's time axes) and rebuilds only those:

```bash
uv run backfill_all_eras.py --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens
# GEFS identical: uv run backfill_gefs_icechunk.py --store gs://gik-gefs-aws-tf/icechunk/gefs-ens
```

Because the group tip has moved past the missing dates, the builder appends
them **out of order as gap fills** (it prints a loud NOTE). Consequences:

- the group's time axis is no longer sorted — consumers do `ds.sortby("time")`
  (exact `.sel(time="YYYY-MM-DD")` lookups work regardless; only slices need
  sorting);
- `check_store_health.py` reports the unsorted axis as a **WARN**, not a
  failure, and keeps checking uniqueness/shapes as usual.

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
