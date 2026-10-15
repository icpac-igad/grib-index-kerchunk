# Restart the Frisky cluster on a schedule

**A long realization run degrades. Restart the workers every ~16 h.**

Measured 2026-08-18, after ~24 h of continuous running: throughput had fallen
to **37% of fresh**, and a restart recovered it in under a minute. Over the two
seasons still queued that was worth about **55 hours**.

This is not a crash. Nothing fails, no date is lost, no error is logged. The
run simply gets slower and slower, and the only way to notice is to look at the
`msg/s` column and remember what it used to be.

---

## 1. What it looks like

Rate history across one continuous run, all on the same cluster:

| season | era | first date | last date |
|---|---|---|---|
| `v3-49r1-mam2026` | 49r1 | 231 msg/s | 179 |
| `v3-50r1-mam2026-tail` | 50r1 | ~64 | 64 |
| `v3-49r1-mam2025` | **49r1** | 67 | 67 |

> **This was a correction, and then a correction to the correction.** The slow
> 50r1 tail was first written up as "50r1 reads ~2.5x slower than 49r1". That
> looked wrong when 49r1 ran at the same 67 msg/s on the same tired cluster, so
> it was retracted as degradation. **The retraction over-corrected.**
>
> The controlled measurement finally came from the channel-append run on
> 2026-08-20, where both eras did identical work on a freshly restarted
> cluster:
>
> | era | s/date (2 channels x 51 members x 65 steps) | msg/s |
> |---|---|---|
> | `50r1` | 158.8 | **41.7** |
> | `49r1` | 40.7 | **163.0** |
>
> **Both things are true.** Cluster degradation is real -- 49r1 fell from 180 to
> 67 msg/s on a tired cluster, which is what this whole document is about. And
> 50r1 is intrinsically ~4x slower to read than 49r1, independent of it.
>
> The transferable rule survives either way: **a per-era throughput figure
> measured on a long-uptime cluster tells you nothing about the era.** Compare
> eras only on a fresh cluster, doing the same shape of work.

## 2. What is actually wrong

`frisky observe overview` and `observe workers`, before and after a restart:

| | degraded (24 h uptime) | fresh |
|---|---|---|
| tasks `processing` | **12** | **197** |
| workers idle | 12 of 12 | 0 of 12 |
| `queued` at scheduler | 14,191 | 99,061 |
| total memory | 34.39 GiB | 3.13 GiB |
| per worker RSS | 2.86 GiB | 245–277 MiB |
| per worker `ArenaFree` | **2.59 GiB** | ~200 MiB |
| per worker `Managed` | 1.3 MiB | 7–21 MiB |
| per worker `Proc` | **1** | 18–19 |
| worker CPU | 9–25% | 80–96% |

Two things, and the first is the one that costs throughput:

**Each worker runs one task instead of eight.** `Proc 1`, `Ready 0`, `Wait 0`
on every worker while 14,191 tasks sit queued at the scheduler. That is 12
concurrent readers where the deployment is sized for 96 — the entire 2.7x
shortfall, with no error anywhere.

**31 of 34 GiB is `ArenaFree`.** Memory the allocator has freed but never
returned to the OS, from a day of churning 4 MB global GRIB messages and 17 MB
output blocks. `Managed` — actual live data — is 1.3 MiB. The workers look
memory-pressured to anything watching RSS while holding almost nothing.

A related nuisance: `spans 200000` sits pinned at the default cap even though
`deploy_workers.py --scheduler-on` sets `FRISKY_TRACING_CAPACITY=20000`. **That
setting is not taking effect** and is worth chasing separately; a restart clears
the buffer either way.

## 3. When to restart

**Default: every 16 hours of continuous running.** That is roughly one 92-date
49r1 season, so in practice it means *between seasons plus once mid-season*.

Restart early if any of these show up — check with
`frisky observe overview http://192.168.1.74:8791`:

| symptom | healthy | restart |
|---|---|---|
| `processing` | 100–200 | **< 50** |
| workers idle | 0 | **most of them** |
| total memory | 3–8 GiB | **> 20 GiB** |
| `msg/s` in the build log | 150–235 | **< 120 sustained** |

The `msg/s` column of the build log is the cheapest signal — it is printed once
per date and needs no tooling.

## 4. How to restart safely, mid-run

The supervisor is built for this. `run_corpus_supervised.sh` retries with
`--resume` up to 10 times, and `--resume` skips dates already in the commit log.
So the whole cost is the single date in flight.

```bash
cd ecmwf/ewc-realization      # the deployed checkout of this folder on the EWC VM
D=/opt/mamba/envs/dask/bin/python

# 1. stop the build. Do NOT kill the supervisor or run_all_seasons.sh --
#    they are what brings it back.
pkill -f "build_corpus.py --sink"

# 2. restart scheduler + workers (also re-ships frisky_daily_dag.py)
$D deploy_workers.py --scheduler-on 192.168.1.74 --start \
     --workers-per-vm 2 --nthreads 8 --memory-limit 6GB

# 3. do nothing. The supervisor waits 120 s, then re-runs with --resume.
tail -f build_corpus_ea-swio_<sink>.log
```

Kill order matters. Killing the supervisor or the season driver instead of
`build_corpus.py` stops the whole run and nothing resumes it.

Restarting when the cluster is **idle** — between seasons — costs nothing at
all, so prefer that timing when the choice exists.

### Budget

Each restart consumes one of the supervisor's 10 attempts *for that season*.
At a 16 h interval a 92-date season triggers at most one or two, so the budget
is ample. `run_all_seasons.sh` gives every season a fresh supervisor.

## 5. Doing it automatically

`cluster_watchdog.sh` runs the procedure above on a timer:

```bash
tmux new-session -d -s watchdog './cluster_watchdog.sh 16'   # hours
tmux attach -t watchdog
tail -f cluster_watchdog.log
```

It only acts when a build is actually running, so it is harmless between
seasons, and it logs every restart with the `msg/s` reading immediately before
and after — which is what makes the next person's diagnosis a five-second job
rather than a two-day one.

Stop it with `tmux kill-session -t watchdog`. **Always stop it before doing
anything manual to the cluster**, or it will restart the workers underneath you.

## 6. Verifying the restart worked

Within a minute:

```bash
.venv/bin/frisky observe overview http://192.168.1.74:8791
```

`processing` should jump into the hundreds and workers should stop being idle.
Do not wait for the next date to complete to find out — that is 10+ minutes,
and `observe` answers immediately.

Then confirm on the next completed date that `msg/s` is back above 150.

---

## Related

| | |
|---|---|
| `run_corpus_supervised.sh` | the retry loop this procedure relies on |
| `run_all_seasons.sh` | season driver; one fresh supervisor per season |
| `deploy_workers.py` | `--start` always kills existing workers first |
| `SCALING.md` | why 96 readers and not more — the ceiling this is about preserving |
