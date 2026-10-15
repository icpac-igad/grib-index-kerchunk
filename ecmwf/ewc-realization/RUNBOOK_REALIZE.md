# Realizing a season — the runbook

How to take a window of dates from the published virtual store to a published
realized store. Written after the 2026-09-25 run (2026-08-24 → 09-24, 39 channels),
where four separate things went wrong before a single date committed. Each one is
a section here.

The upstream operational account of the *published* stores —
what is in them, how to check them against Herbie — is
[`grib-index-kerchunk/ecmwf/icechunk-par/REALIZED_EA_SWIO_STORE.md`](../../../grib-index-kerchunk/ecmwf/icechunk-par/REALIZED_EA_SWIO_STORE.md).
This file is the *producing* side; that one is the *reading* side. Keep them
consistent.

---

# ⚠️ STOP — the cluster is a prerequisite, and it lies about being up

**Nothing in stage ③ runs without the frisky cluster. Verify all four of the
following before you plan any work, because three of them can look fine while
being wrong.**

### 1. The documented IPs are stale. Ask the network, not the docs.

`deploy_workers.py`, `parallel_write.py` and every runbook example have carried
`192.168.1.74` for months. On 2026-09-25 that host did not exist: the VMs had been
rebuilt and **every IP had changed**. Probing the documented address and
concluding "the cluster is down" is wrong and costs hours.

```bash
# Who is actually alive? fa:16:3e:* MACs are the OpenStack VMs.
for i in $(seq 1 254); do (ping -c1 -W1 192.168.1.$i >/dev/null 2>&1 && echo "192.168.1.$i") & done; wait
ip neigh | grep -i reachable

# The scheduler is whichever host answers 8796 (frisky) and 8791 (dashboard).
for ip in <candidates>; do
  for p in 8796 8791; do timeout 4 bash -c "</dev/tcp/$ip/$p" 2>/dev/null && echo "$ip:$p OPEN"; done
done
```

`build_corpus.py --scheduler` must then be pointed at the live address. Note the
form: frisky ≥ 0.7 wants bare `host:port`. The `tcp://…` shown in
`frisky_daily_dag.py`'s docstring is stale and raises
`ConnectionError: failed to lookup address information`.

### 2. `deploy_workers.py` needs a *Dask* scheduler, which is a different thing

It uses Dask purely as a remote-exec channel (`client.run()`) because there is no
SSH key on the gateway. The frisky scheduler being up says nothing about it:

```bash
timeout 4 bash -c "</dev/tcp/127.0.0.1/8786" && echo "dask channel OK"
/opt/mamba/envs/dask/bin/python deploy_workers.py --status
```

`--status` also prints each VM's last log line, which is how you see a
cluster-wide fault at a glance.

### 3. Restart the workers whenever the source store has changed

`_open_era` caches the opened store **per worker process**. Workers started before
a re-mirror keep serving the old snapshot and raise
`KeyError: "not all values found in index 'time'"` on every block of a newly
published date — 2,535 of 2,535, with the client reporting `retried 0`.

```bash
/opt/mamba/envs/dask/bin/python deploy_workers.py \
    --scheduler-on <live-scheduler-ip> --start \
    --workers-per-vm 2 --nthreads 8 --memory-limit 6GB
```

> **A local smoke test does not prove the cluster is healthy.**
> `frisky_daily_dag.py` *without* `--scheduler` spins a LOCAL cluster in fresh
> processes, so it reads the new snapshot happily while every remote worker still
> fails. On 2026-09-25 that false positive sent a build at a cluster that could not
> read a single block. **Always pass `--scheduler` when testing:**
>
> ```bash
> .venv/bin/python frisky_daily_dag.py --scheduler <ip>:8796 \
>     --date <the newest date you need> --era 50r1 --vars d2m tp --members 4 --steps 2
> ```
>
> Two channels, four members, two steps. If that returns finite data, the workers
> can see the date. If it does not, no build will.

### 4. One client on the scheduler at a time

The scheduler is a single process on one VM and it is **not** safe to share.
Starting a second `build_corpus.py` while one is running killed the first with
`ConnectionError: Connection to scheduler closed … (scheduler recv ended: early
eof)`, losing the date in flight.

Different sinks and the 96-reader AWS ceiling are *not* the constraint — the
scheduler is. Run builds serially. Total throughput is unchanged either way,
because the worker pool is the bottleneck.

---

## 0. The other prerequisites

```bash
# Client venv (a REAL venv here; do NOT install into /opt/mamba/envs/dask --
# the workers' own .venv is a symlink to it)
python3 -m venv .venv
.venv/bin/pip install "frisky>=0.3.0" icechunk==2.1.1 zarr==3.2.1 \
    xarray==2026.7.0 gribberish==1.6.0 numpy==2.5.1 dask==2026.7.1 boto3

# Credentials: TWO files, and they are not interchangeable
#   .env      AK / SK            -> EWC Ceph, the build sink
#   ./.env    AWS_* (STS, 12 h)  -> source.coop, the mirror destination
```

Ceph TLS: boto3 validates against certifi, which lacks the ECMWF CA, so any
direct Ceph call needs `export AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`.
`curl` works without it, which makes the failure look like a credentials problem
when it is not.

Check Ceph headroom before starting — `must-icechunk` has a **5 TB hard limit**,
and a 24-date season is ~415 GB:

```bash
# see RUNBOOK §0 snippet, or: ~40 GB/date at 39ch x 51 members x 65 steps
```

---

## 1. The gate: publish the virtual store FIRST

**Stage ③ reads `SRC_PREFIX` on source.coop, not the GCS store.** A date that
exists in GCS but has not been mirrored is invisible to the realization, and the
symptom is the same `KeyError` as a stale worker cache — so the two are easy to
confuse.

```bash
cd ../2-icechunk-virtual
uv run mirror_gcs_to_source_coop.py --tip-only \
  --gcs-store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
  --dest forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd
```

Expect a few HTTP 520s on a run of any size. The script **withholds the `repo`
pointer if any object fails**, so the published store stays on its previous
consistent snapshot rather than pointing at manifests that are not there. Re-run
to finish the stragglers — it has done this on three separate runs and each time
turned an unattended partial failure into a non-event.

Then confirm the tip, and only then restart the workers (§3 above).

---

## 2. Choosing the shape

| | value | why |
|---|---|---|
| `--steps` | **65** | +240h, the 10-day lead. Every published season is 65. `build_corpus.py` still defaults to **53** (+168h, 7 days) — a bare invocation builds a season that matches nothing. |
| `--members` | 51 | control at `number=0` |
| channels | **39 by `--vars`** | `--channels` is pinned to 38 on purpose; pass the names instead of raising the pin |
| `--era` | 50r1 | from 2026-05-13 |

Get the channel list from the table rather than typing it:

```bash
.venv/bin/python -c "import frisky_daily_dag as d; print(' '.join(c[2] for c in d.CHANNELS))"
.venv/bin/python -c "import frisky_daily_dag as d; print(d.channels_absent_from('50r1', d.CHANNELS, '2026-09-01') or 'all available')"
```

`channels_absent_from` is the pre-flight for era/date availability: `0p4` carries
none of `w700`/`cape`/`d2m`, and `d2m` needs 49r1/50r1 **from 2024-03-06**
(`DEWPOINT_FROM`) — before that it decodes as all-NaN.

### Season boundaries and the axis

**A season's time axis is fixed at creation and cannot be grown.** `build_corpus.py`
preallocates it (`create_schema`) and writes regions into it; the `append_dim="time"`
path that *could* grow it is the documented cause of the non-monotonic axes this
pipeline has already been bitten by, and is never used here.

So **preallocate the whole season** and fill it incrementally:

```bash
# SOND 2026 = 122 dates. Build what exists; the rest are reserved and filled later.
--start 2026-09-01 --days 122
```

Dates absent from the source are skipped with their slots reserved. That is
deliberate — an unwritten slot has no chunks in the manifest, so `check_chunks.py`
names it precisely instead of zero-fill passing it off as data.

> **A reserved slot reads back as NaN, not as an error.** This is the hazard
> `REALIZED_EA_SWIO_STORE.md` §4 is written about. Publishing a part-filled season
> publishes those NaNs. Decide deliberately whether that is acceptable for the
> store you are making, and say so in its documentation.

Dates that belong to a *finished* season cannot be appended to it. Give them their
own store (`…-tail-aug`) — one store mirrors to exactly one published prefix, so
the layout you build is the layout readers get.

---

## 3. Build

```bash
VARS="$(.venv/bin/python -c 'import frisky_daily_dag as d; print(" ".join(c[2] for c in d.CHANNELS))')"

setsid nohup .venv/bin/python -u build_corpus.py \
  --sink ea-swio/v3-50r1-sond2026 \
  --era 50r1 --start 2026-09-01 --days 122 \
  --vars $VARS --members 51 --steps 65 --env .env \
  --scheduler <ip>:8796 \
  > build_sond2026.log 2>&1 < /dev/null &
```

**`python -u` is not optional.** Without it the log block-buffers and sits empty
for ten minutes while every block fails.

Expect ~2,535 blocks/date and **650–1,000 s/date** at 180–215 msg/s. Retries of a
few blocks are normal AWS throttling and recover; `--rounds 4 --pause 45` handles it.

### The anomaly that means data loss

> **A date that finishes much faster than its neighbours has written fewer blocks.**
> On 2026-09-25, `2026-09-05` reported `246.2s 525.1 msg/s retried 0` against a
> ~650 s norm, with merge time falling from ~3 s to 0.9 s. It had committed **13 of
> 39 channels** and reported success. `2026-09-04` silently lost `d2m` from step 44.
>
> Watch for: elapsed well below norm, msg/s well above, merge time collapsed.
> Anything matching that has lost blocks regardless of what the line says.

---

## 4. Verify — the success line is not evidence

Three checks, cheapest first. Run them **after the build exits**, not during: a
store being committed to gives a moving target.

```bash
# 1. Manifest. The only honest source -- a data read returns fill_value and tells
#    you nothing about whether a write was attempted.
.venv/bin/python check_chunks.py --prefix ea-swio/<sink> --env .env

# the shape of the answer matters more than the verdict:
.venv/bin/python check_chunks.py --prefix ea-swio/<sink> --env .env \
  | awk '/chunks/{print $2}' | sort | uniq -c
```

**Every channel must show the same count.** `39 × (dates × 65)` uniform means whole
dates. A *spread* of counts (260 / 283 / 304 / 325) means dates torn across
channels — that is what a silent partial commit looks like, and it is the single
most useful thing this check tells you.

`total missing` should equal `unwritten_dates × 39 × 65` exactly. Any other number
is a real gap.

```bash
# 2. Physics. A broadcast bug cannot fake spread growing with lead time.
.venv/bin/python check_members.py --prefix ea-swio/<sink> --env .env --var gh500
```

**Always pass `--var gh500`.** With no `--var` it takes the first data variable,
and a 2 m temperature field's flat tropical spread reads as a failure.

```bash
# 3. Read the data at both ends of the step axis for any date you had doubts about
#    -- step 0 AND step 64. A date can be whole at 0 and empty at 64.
```

### Repairing

```bash
.venv/bin/python build_corpus.py --sink ea-swio/<sink> --repair \
    --era 50r1 --start <SEASON START> --days <FULL SEASON LENGTH> ...
```

> **`--repair` must be given the store's whole window.** It scans the manifest by
> *position in the requested date list* (`missing_blocks(repo, names, len(dates), …)`),
> so a narrower window inspects the wrong slots and reports
> `repair nothing missing -- store is complete` while the store is demonstrably
> broken. Asking for `--start 2026-09-04 --days 2` examined indices 0–1 — which were
> 09-01 and 09-02, both fine.

Repair costs only the missing blocks: 21 blocks for a lost step range, 1,732 for a
half-written date, against 2,535 for a rebuild. It also builds *unbuilt* dates at
full cost, so one repair run over the season window finishes the job.

---

## 5. Mirror

```bash
set -a; . ./.env; set +a          # source.coop STS
export AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

setsid nohup .venv/bin/python -u mirror_ceph_to_source_coop.py \
  --src-prefix ea-swio/v3-50r1-sond2026 \
  --dest forecasts/ecmwf-ifs-ea-swio-realized-v3/50r1-sond2026 \
  --env .env > mirror_sond.log 2>&1 &
```

Measured **424–437 GB/h** (4 procs × 48 threads): ~148 GB / 21 min for 8 dates,
~415 GB / 57 min for 24. The estimate it prints (90 MB/s) is conservative.

**It prints nothing until it finishes.** Silence is not a stall — check the
destination object count if you want progress. The `repo` pointer goes **last**, so
until it lands the store is not openable and verifying is pointless:

```bash
# progress, and whether the pointer has landed
aws s3 ls --recursive ... | wc -l      # or list the prefix anonymously
```

Then verify as an external reader — anonymously, no credentials:

```python
st = icechunk.s3_storage(bucket="e4drr-project", prefix="forecasts/…/<sub>",
                         endpoint_url="https://data.source.coop", region="us-east-1",
                         anonymous=True, from_env=False, force_path_style=True)
```

Check dates, `vars`, `steps`, `members`, and read a field at step 0 and step 64 —
including any date you repaired, to confirm the repair carried through the mirror.

**Nothing is deleted from Ceph by any of this.** The documented lifecycle is
build → verify → mirror → verify → delete, and the delete is a human decision.

---

## 6. What this run left open

- **12 `-d2m` fragments** on Ceph (`ea-swio/v3-*-d2m`, 0.365 TB) are still
  unpublished, so the 12 older seasons lack `d2m` in public form. Upstream records
  the same gap (`REALIZED_EA_SWIO_STORE.md` §5). The two stores built on 2026-09-25
  carry `d2m` inside the season, which is the pattern to follow from here.
- **`0p4-mam2023` is short** — 85 of 92 dates (upstream §4).
- **`2024-02-29`** is missing from the realized corpus: `0p4-jf2024` ends 02-28 and
  `49r1-mam2024` starts 03-01, and the leap day falls on the era boundary. It is
  realizable from the virtual store.
- **`sond2026` publishes 98 NaN dates** by design, unlike every other published
  season. Say so wherever it is advertised.
