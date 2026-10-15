# Daily operations: one new forecast, from pars to source.coop

Every day, the new 00z ECMWF ENS forecast goes through five stages:

1. GIK pars are built for it.
2. The pars are appended to the virtual Icechunk store.
3. The virtual store is mirrored to source.coop.
4. The EA/SW-IO box is realized on EWC into Ceph.
5. The realized date is mirrored to source.coop.

Measured timings for 2026-10-01 → 10-15 are in
[`stats/`](stats/DAILY_RUNS_2026-10-01_to_10-15.md).

```
 ① pars            ② virtual store            mirror ②              ③ realize                mirror ③
 Lithops/CloudRun → gs://…/icechunk/       → source.coop          → frisky on 6 EWC VMs    → source.coop
 51 parquets       ecmwf-ens-v4 (50r1/00z)    …_icechunk_vd          → Ceph must-icechunk/     ecmwf-ifs-ea-swio-
 ~6 min            ~25 s                      ~10 min (listing)      ea-swio/v3-50r1-sond2026  realized-v3/…
                                                                     ~40–55 min @ 24 readers   ~5–6 min
```

## Where the driver lives

The driver is `daily_run.sh` in the **crma** repo
(`crma/medium-range-forecast/3-realize-frisky/daily_run.sh`). It also runs the
CRMA layers after the realization (Layer 2→3, Layer 4, briefs, API export),
which is why it stays there. Its ECMWF stages call the scripts in this folder,
as deployed in the `refill-crma-routine` checkout on the gateway. Every
command below is one the driver runs, in the order it runs them.

## 0. Before the driver: pars (stage ①)

Pars are started separately. They need no source.coop token, and they can
sit idle for hours if the GRIBs are late.

```bash
# are all 344 GRIB + index objects published? (00z normally lands by ~08 UTC)
curl -sI https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/<D>/00z/ifs/0p25/enfo/<D>000000-240h-enfo-ef.grib2

cd …/1-pars-lithops && . ./pars50r1.env          # see below -- never skip this
uv run era_check.py preflight --run 00 --date <D>
nohup uv run run_lithops_ecmwf.py --date <D> --run 00 --max-workers 10 -y > ~/d-<D>-pars.log 2>&1 &
```

`pars50r1.env` sets `UV_PYTHON=3.12`,
`GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf`, `AWS_NO_SIGN_REQUEST=YES`,
`ECMWF_REFERENCE_DATE=20260513`, `ECMWF_RESOLUTION=0p25`,
`ECMWF_CONTROL_STREAM=oper`, and the 50r1 `TEMPLATE_URL` (HuggingFace
`E4DRR/grib-index-kerchunk-templates`).

> **Without it the run "succeeds" with 50/51 members.** The script falls back
> to the 2024 template and the `enfo` stream, and in 50r1 `enfo` has no control
> member. This cost the whole of 2026-10-01.

The Lithops code is [`../lithops-cr-gik-ecmwf/`](../lithops-cr-gik-ecmwf/).
Output goes to `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{yyyy}/{mm}/{date}/00z/`.

## 1. Take the source.coop credential

Take it immediately before the run, because mirror ② and mirror ③ both need it.
It is merged into the gitignored `.env` next to the scripts. That file holds the Ceph
`AK`/`SK` and the five source.coop `AWS_*` lines side by side. The CLI emits
only three of those five, so merge rather than overwrite:

```bash
umask 077
source-coop creds --format env > /tmp/creds.env
{ grep -vE '^(export )?(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)=' .env
  sed -E 's/^(export )?/export /' /tmp/creds.env ; } > .env.new
mv .env.new .env && chmod 600 .env && shred -u /tmp/creds.env
cut -d= -f1 .env | sed 's/^export //'      # expect AK, SK and all five AWS_* names
```

Since 2026-10-13 this is refreshed silently from a 30-day refresh token
(source-coop CLI v0.3.0). Before that it was a 12 h token behind a browser
login, which cost four of the first nine days. See
[`SOURCE_COOP_CREDENTIALS.md`](SOURCE_COOP_CREDENTIALS.md).

## 2. Run the driver

```bash
cd /home/ubuntu/nishadhka-work/crma
bash medium-range-forecast/3-realize-frisky/daily_run.sh <YYYYMMDD>  > ~/crma-daily-<MMDD>.log 2>&1
```

Each stage is gated, and a failure stops the chain with `!! STOPPED: <reason>`.
The ECMWF stages are:

| # | stage | command (abridged) | gate | typical |
|---|---|---|---|---|
| 1 | verify pars | `era_check.py verify --run 00 --start D --end D` | `complete 51/51` | <15 s |
| 2 | extend axis + backfill | append D to `dates/dates-50r1-00z.txt`; `build_ecmwf_icechunk.py --era 50r1 --run 00 --extend-schema`; `backfill_all_eras.py` → `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4` | `built, 0 failed` | 1–1.5 min |
| 3 | mirror ② | `mirror_gcs_to_source_coop.py --dest forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd --threads 24` | `MIRROR COMPLETE` | 9–10 min |
| 4 | read gate | one `frisky_daily_dag.read_message("50r1","tp",…)` from the gateway, and 10 byte-range `curl`s on the 24 h GRIB | read OK and throttle ≥ 2/10 | ~15 s |
| 5 | **RULE 1**: restart frisky | `deploy_workers.py --scheduler-on 192.168.1.164 --start --workers-per-vm 1 --nthreads 4 --memory-limit 12GB` | 6 workers | ~10 s |
| 6 | **realize** | `build_corpus.py --sink ea-swio/v3-50r1-sond2026 --repair --era 50r1 --start 2026-09-01 --days 122 --vars <39> --members 51 --steps 65 --rounds 6 --pause 120 --scheduler 192.168.1.164:8796` | `D … repaired`, no `STILL INCOMPLETE` | 39–55 min |
| 7 | mirror ③ | `mirror_ceph_to_source_coop.py --src-prefix ea-swio/v3-50r1-sond2026 --dest forecasts/ecmwf-ifs-ea-swio-realized-v3/50r1-sond2026 --threads 48` | `pointer` | 5–6 min |

After stage 7 the driver restarts Dask (RULE 1 again) and continues with the
CRMA layers. Those read the realized store **from source.coop**, which is why
mirror ③ must complete first.

### Rules the driver enforces, and why

- **Dates run in order.** Two things are positional: the order of the dates
  file is the virtual store's write index, and the `--start 2026-09-01
  --days 122` window is the realized store's slot index. Running 10-10 before
  10-09 (both were waiting on 10-10) would have written into the wrong slot.
- **RULE 1: restart workers after every mirror.** `_open_era` caches the
  opened store per worker process. A worker started before mirror ② keeps the
  old snapshot, and every block fails with
  `KeyError: "not all values found in index 'time'"` (10-02, 25 min lost).
- **`unset AWS_*` before stage ③.** The source.coop credential exports
  `AWS_ENDPOINT_URL`. With it set, `sink_icechunk.py` aims the **Ceph** write
  at source.coop.
- **24 readers, not 96.** See [`05_BOTTLENECKS_AND_ISSUES.md`](05_BOTTLENECKS_AND_ISSUES.md) §2.
- **The throttle gate** stops the run on 0–1/10, because on 10-05 those levels
  meant four stalled passes. Hold and retry later; don't reduce `--rounds`.

## 3. When it stops

| stop message | cause | action |
|---|---|---|
| `pars not 51/51` | pars still running or failed; `pars50r1.env` not sourced | check `~/d-<D>-pars.log` |
| `mirror 2 incomplete` | source.coop 5xx or expired token | refresh `.env.sc`, re-run the driver; it resumes |
| `throttle N/10` | AWS busy with the same-day archive | wait 1–3 h, re-run |
| `stage 3 left holes` | throttle got worse mid-run | re-run: `--repair` builds only the missing blocks |
| `InvalidAccessKeyId` (Ceph) | the OpenStack **application credential** behind the key lapsed (the key itself is fine) | renew it in the EWC portal ([`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) §3) |
| stage ③ stalls at `0 B/s` | too many readers for a busy source | confirm 24 readers in `d-<D>-frisky.log` |

To restart a stuck cluster, see [`CLUSTER_RESTART_ROUTINE.md`](CLUSTER_RESTART_ROUTINE.md).
For the 10-01 → 10-15 failure record, see
[`stats/DAILY_RUNS_2026-10-01_to_10-15.md`](stats/DAILY_RUNS_2026-10-01_to_10-15.md).

## 4. Not yet automated

- Pars (§0) and the credential (§1) are started by hand. The GEOGLOWS capture
  is the only stage on a systemd timer (11:00 UTC).
- `daily_run.sh` hardcodes the SOND 2026 store and window. The JF season,
  starting 2027-01-01, needs a new `--sink`, `--start` and `--days`.
