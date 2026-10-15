# Long-term realization: the multi-season EA/SW-IO corpus

How the ~18 TB realized corpus was built: every season from MAM 2023 to SOND
2026, all three ECMWF eras, 51 raw members. It is a reference for redoing it
or extending it. The step-by-step runbook for one season is
[`RUNBOOK_REALIZE.md`](RUNBOOK_REALIZE.md). This page covers the whole run:
order, sizing, cost, and what to change next time.

## 1. What it produces

| | |
|---|---|
| source | virtual store on source.coop (`forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd`), read anonymously; its chunks are byte ranges into `s3://ecmwf-forecasts` |
| sink | EWC Ceph `s3://must-icechunk/ea-swio/v3-<era>-<season>[-tail]`, one Icechunk store per season |
| published | source.coop `forecasts/ecmwf-ifs-ea-swio-realized-v3/<era>-<season>` |
| domain | `ea-sw-io40`: lat 40 → −40, lon 15 → 80, **321 × 261** at 0.25° (0.4° for `0p4`) |
| dims | `(time, step, number, latitude, longitude)`; 65 steps (0–240 h), members 0–50 (control at 0) |
| channels | **38–39**, era-portable (18 base variables common to `0p4`/`49r1`/`50r1`). `d2m` exists only in 49r1/50r1 from 2024-03-06 |
| per date | ~126,000 GRIB messages read → 2,535 blocks, **~17 GB** stored |

The store layout and its known gaps are documented in
[`../icechunk-par/REALIZED_EA_SWIO_STORE.md`](../icechunk-par/REALIZED_EA_SWIO_STORE.md).
Opening it is covered by [`USING_REALIZED_EA_SWIO_STORE.ipynb`](USING_REALIZED_EA_SWIO_STORE.ipynb).

## 2. What it cost (measured)

All 16 seasons, from [`REALIZATION_CONCURRENCY.md`](REALIZATION_CONCURRENCY.md) §1:

| season | dates | hours | s/date | msg/s | retries |
|---|---|---|---|---|---|
| `0p4-mam2023` | 85 | 14.4 | 612 | 222 | 23 |
| `0p4-jja2023` | 92 | 12.2 | 477 | 262 | 141 |
| `0p4-sond2023` | 122 | 13.6 | 401 | 307 | 69 |
| `0p4-jf2024` | 59 | 6.7 | 409 | 298 | 39 |
| `49r1-mam2024` | 92 | 15.1 | 590 | 206 | 27 |
| `49r1-jja2024` | 92 | 18.4 | 721 | 176 | 10 |
| `49r1-sond2024` | 122 | 27.1 | 800 | 160 | 71 |
| `49r1-jf2025` | 58 | 14.3 | 890 | 145 | 92 |
| `49r1-mam2025` | 92 | 19.0 | 745 | 173 | 135 |
| `49r1-jja2025` | 92 | 20.7 | 809 | 158 | 57 |
| `49r1-sond2025` | 122 | 27.0 | 798 | 160 | 48 |
| `49r1-jf2026` | 56 | 13.6 | 875 | 147 | 110 |
| `49r1-mam2026` | 73 | 13.6 | 673 | 185 | 87 |
| `50r1-mam2026-tail` | 19 | 8.2 | 1,547 | 87 | 14 |
| `50r1-jja2026` | 33 | 7.0 | 759 | 167 | 0 |
| `50r1-jja2026-tail` | 51 | 12.0 | 847 | 153 | 11 |
| **total** | **1,260** | **243.0** | **694** | | **934** |

- **~103 TB read from AWS → ~17.9 TB written.** That is the "18 TB / ~200 h".
  An earlier snapshot of 887 dates (0.25° only, JF 2025 part-built) recorded
  192.4 h and 88.0 TB → 14.9 TB ([`REALIZATION_COST.md`](REALIZATION_COST.md) §1).
- The read amplification is 5.9×, because a GRIB message is global and the box
  keeps 8% of it. The 0.4° seasons are about twice as fast per date, since
  their messages are smaller.
- 127 MB/s sustained, 161–197 msg/s, and 5.85 retries per million messages.
- The run took about three weeks of wall-clock, because of credential
  expiries, a VM reboot, and idle gaps between seasons. The actual compute was
  8–10 days.

## 3. How it was driven

Each season's `time` axis is **fixed at creation** (preallocated by
`create_schema`). A season can be filled incrementally with `--resume` and
`--repair`, but it can't be grown. A season that spans an era seam needs two
stores (`49r1-mam2026` + `50r1-mam2026-tail`), which are read together with
`xr.concat(..., dim="time")`.

```bash
# one season, supervised: retries with --resume up to 10x, 120 s pause between
tmux new-session -d -s mam2025 \
  './run_corpus_supervised.sh ea-swio/v3-49r1-mam2025 2025-03-01 92 49r1 65 38'

# a sequence of seasons; each waits for the previous one's success marker
tmux new-session -d -s seasons ./run_all_seasons.sh
```

The drivers that actually ran each season and each channel append are kept
in [`history/season-drivers/`](history/season-drivers/). Order matters:

1. **Build the newest, best-verified season first.** It tests the write path
   in hours, not days. MAM 2026 went first because 2026-03-01 had already
   been verified byte-exact.
2. **Restart the cluster every ~12 h.** Throughput falls ~25% with worker age
   and comes back on restart. A steady decline with **0 retries** is the
   cluster; a drop **with** retries is the source.
3. **Run `--repair`** after the supervisor finishes. It rebuilds only the
   missing blocks.
4. **Verify, then mirror, then purge.** See
   [`04_VERIFY_AND_PUBLISH.md`](04_VERIFY_AND_PUBLISH.md). Don't delete a Ceph
   store until the source.coop copy has passed the check; the evidence trail is
   [`history/CEPH_DELETION_EVIDENCE.md`](history/CEPH_DELETION_EVIDENCE.md).
5. **Adding a channel later** (as `d2m` was) is done with `append_channels.py`
   into `-d2m` fragment stores. Next time, put every channel in the season
   from the start; 12 fragments (0.365 TB) are still unpublished.

## 4. Sizing the next run

| choice | effect |
|---|---|
| **48 readers** (what ran: 2 workers × 4 threads × 6 VMs) | 192 h for 887 dates. The cluster was **reader-starved** (flat 3.46 msg/s per reader) |
| 96 readers (2 × 8 × 6, what the runbook intended) | ~139–191 h for the 243 h corpus **[est]** |
| ~200 readers | ~135 h **[est]**. Throughput flattens past ~200 (179 MB/s at 400) |
| warm the read path (DAG v2, [`DAG_VERSIONS.md`](DAG_VERSIONS.md) §4) | up to 2.56× per process, measured on a warm cache **[est for the full run]** |
| more VMs | the wrong lever ([`REALIZATION_CONCURRENCY.md`](REALIZATION_CONCURRENCY.md) §5) |

**A backfill is not the daily run.** Historical archive dates are quiet, so
96 readers is plausible for a backfill. On 2026-10-05, the busy same-day
archive defeated 80 readers ([`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) §2).

Planning numbers for GCP, if this moves off EWC: ~$55 of Spot compute for
192 h on 6 × e2-standard-4, about $1,980 of egress for 18 TB, and $45–414 a
month to keep it, depending on storage class (all **[est]**;
[`REALIZATION_COST.md`](REALIZATION_COST.md) §A3). On that comparison, the
valuable part of the EWC allocation is the Ceph quota and the source.coop
hosting, not the VMs.

## 5. Open from the long run

- 12 `-d2m` fragment stores are unpublished, so older seasons lack public `d2m`.
- `0p4-mam2023` has 85 of 92 dates.
- 2024-02-29 is missing. It falls on the `0p4` → `49r1` seam, and it can be
  realized from the virtual store.
- `sond2026` publishes reserved (NaN) slots by design, until the season is complete.
