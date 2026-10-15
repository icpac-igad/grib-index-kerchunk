# Stage ③ — virtual store → realized dataset

Reads the virtual store from stage ② and writes **decoded arrays** for one
region to EWC Ceph, using a [Frisky](https://getfrisky.dev/) futures DAG across
six worker VMs. One task = one GRIB message.

This is the stage that produces the array CRMA's flood scenario actually uses:
ensemble rainfall over East Africa, ready to test against a return-period
threshold. Stages ①–② exist so this one can be cheap.

**Output:** `s3://must-icechunk/ea-cgan/...` — dims
`(time, step, number, latitude, longitude)` × 36 channels, **raw ensemble
members** (`number = 0..50`, control at 0). Nothing is averaged anywhere in
this stage.

## Domain and channels

Both live in `frisky_daily_dag.py` and both are read by the **workers**, which
hold their own copy of that module — `deploy_workers.py --install` after any
edit, or they keep cutting the old box.

| | `ea` (v3/v4 stores) | **`ea-sw-io`** (current) |
|---|---|---|
| latitude | 25.25 → −15.25 | **25.25 → −40.0** |
| longitude | 18.5 → 55.0 | **15.0 → 80.0** |
| grid points | 163 × 147 = 23,961 | **262 × 261 = 68,382** (2.85×) |
| channels | 30 | **36** |

Every edge lands exactly on the 0.25° grid, so no interpolation is implied.
The wider box takes in the south-west Indian Ocean — the basin the systems
that flood East Africa come out of.

The 36 channels draw on only **18 base variables**, and that set is exactly
the 18 common to `0p4`, `49r1` and `50r1`; the five levels used
(200/500/700/850/925) are all within the 9 levels common to the three. So this
channel set is **era-portable**, unlike the previous 30, which used `ssr`,
`ttr`, `tcw`, `mucape` and `w` — none of which exist in `0p4`.

What one date costs, before and after:

| | 30 ch / `ea` | 36 ch / `ea-sw-io` |
|---|---|---|
| messages | 81,090 | **97,308** |
| blocks (chunks) | 1,590 | **1,908** |
| chunk | 4.9 MB | **13.9 MB** |
| raw float32 | 7.8 GB | **26.6 GB** (~10.6 GB stored) |
| AWS egress | ~64 GB | **~77 GB** |

← [`../icechunk-par/`](../icechunk-par/) · [`README.md`](README.md)

---

## Environment

```bash
python -m venv .venv
.venv/bin/pip install "frisky>=0.3.0" icechunk==2.1.1 zarr==3.2.1 \
    xarray==2026.7.0 gribberish==1.6.0 numpy==2.5.1 dask==2026.7.1
P=.venv/bin/python
D=/opt/mamba/envs/dask/bin/python      # only for deploy_workers.py
```

`dask` is needed on the **client only**, by `create_schema`, which uses
`dask.array.zeros` to declare arrays without materialising them.
`deploy_workers.py` runs under `$D` because it talks to the **Dask** cluster,
used purely as a remote-exec channel (there is no SSH key on the gateway).

## Credentials — two sets, kept apart

| | store | credentials |
|---|---|---|
| **source** (read) | source.coop → virtual chunks on AWS | **anonymous**, `from_env=False` |
| **sink** (write) | EWC Ceph, `s3://must-icechunk` | `AK`/`SK` from `.env` |

`.env` alongside these scripts (gitignored):

```
AK=<ewc access key>
SK=<ewc secret key>
```

Workers are **never given the keys**. The client opens the sink with explicit
credentials, calls `session.fork()`, and the resulting `ForkSession` — carrying
its storage config — is pickled to each worker as a task argument.

> **The `AWS_*` trap.** The Dask workers carry `AWS_ENDPOINT_URL` and
> `AWS_DEFAULT_REGION` for Ceph. If the icechunk **virtual-chunk** client sees
> those it tries to resolve a host that does not exist, and the config is cached
> process-wide so unsetting it later does not help.
> `frisky_daily_dag.py` pops every `AWS_*` **at import**; `deploy_workers.py`
> launches each worker with `AWS_*` scrubbed from that child only. Do not "fix"
> this by exporting them.

## Bring up the cluster

```bash
$D deploy_workers.py --install                      # once — builds the venv on all six VMs
$D deploy_workers.py --scheduler-on 192.168.1.74 --start \
     --workers-per-vm 2 --nthreads 8 --memory-limit 6GB
$D deploy_workers.py --status
$D deploy_workers.py --stop                         # Dask workers untouched
```

| flag | why it is not arbitrary |
|---|---|
| `--scheduler-on <worker IP>` | **not** the gateway — its 8 GiB cgroup OOM-killed the scheduler mid-run and the client saw only `channel closed` |
| `--workers-per-vm 2` | icechunk caps in-flight connections **per process**: 1 proc ≈ 25–31 MB/s, 2 ≈ 47, flat beyond. Threads cannot substitute |
| `--nthreads 8` | 12 × 8 = 96 concurrent readers. 192 draws `SlowDown` from AWS |

**`--start` always kills existing workers first.** Skipping that once left 12
workers registered, six running a stale copy of the task module.

> **Restart the workers whenever the source store changes.** `_open_era` caches
> the opened store per worker **process**, so workers started before a re-mirror
> keep serving the old snapshot and will `KeyError` on newly published dates.

Optional dashboard (`jupyter-server-proxy` can only reach `127.0.0.1`, hence the
relay):

```bash
setsid nohup $P dashboard_forward.py --to 192.168.1.74:8791 --listen 8791 \
    > dashboard_forward.log 2>&1 < /dev/null &
# then https://<hub>/user/<user>/proxy/8791/   -- trailing slash, no /status
.venv/bin/frisky observe overview http://192.168.1.74:8791   # terminal, no proxy
```

## Build

One date:

```bash
$P parallel_write.py --sink ea-swio/one-date --date 2026-07-02 \
     --channels 36 --members 51 --steps 53 --fork-once --env .env
```

A month:

```bash
$P build_corpus.py --sink ea-cgan/v4-mar2026-49r1 --era 49r1 \
     --start 2026-03-01 --days 31 --env .env
```

The whole `time` axis is preallocated **sorted** up front, so every write is a
region write at a fixed index and **commit order cannot corrupt the axis** —
unlike stage ②. Dates absent from the source get their slots reserved but left
unwritten. One commit per date, ~6–10 min each.

| flag | when |
|---|---|
| *(none)* | first build only — creates the store |
| `--resume` | continue; skips dates already **complete** |
| `--repair` | fill **only** the chunks the manifest reports missing |
| `--recreate` | **destructive** — rewrites the schema over an existing store |

> **An existing store is never overwritten implicitly.** Running without
> `--resume`/`--repair` on a store that exists exits 1. That guard is there
> because its absence once wiped 30 committed dates in a single commit —
> `create_schema` writes `mode="w"`. Recovery was possible only because icechunk
> keeps history.
>
> **Before any writing run, note the rollback point:**
> ```python
> list(repo.ancestry(branch="main"))[0].id      # -> repo.reset_branch("main", id)
> ```

Blocks AWS throttles out are resubmitted over 4 rounds with a 45 s pause. The
pause is the point — resubmitting into an active throttling window just burns
the retries. Blocks that never land are committed as `PARTIAL` and filled later
by `--repair`, rather than discarding the whole date.

## Verify — do not trust "0 failed blocks"

```bash
$P check_chunks.py  --prefix ea-cgan/v4-mar2026-49r1 --env .env
$P check_members.py --prefix ea-cgan/v4-mar2026-49r1 --env .env --var gh500
$P check_complete.py --prefix ea-swio/one-date --env .env     # single date only
```

- **`check_chunks.py`** reads the **manifest** (~17 MB) and lists what was
  actually written. Right for a corpus. `check_complete.py` reads every block —
  right for one date (26.6 GB), hopeless for thirty-one (825 GB).
- **`check_members.py`** catches a `number` selection that silently broadcast, by
  checking ensemble **spread grows with lead time** — physics a broadcast bug
  cannot fake. **Always pass `--var gh500`.** With no `--var` it takes
  `list(ds.data_vars)[0]`, and if that lands on a 2 m temperature field the flat
  tropical spread reads as a failure. It also only ever inspects `time=0`.

Expected on a complete month:

```
59,148 / 59,148 chunks, every channel 1,643, none missing
gh500 spread 1.2939 -> 9.1447 from +0h to +168h
VERDICT: members are REAL
```

A hole is **loud**: unwritten chunks read back as NaN (`finite 0.000`), because
`create_schema` writes metadata only and `fill_value` is NaN. There is no
zero-fill.

## Throughput, measured

96 readers over six VMs: **~180 MB/s** from AWS, ~230 messages/s, ~6–10 min per
date on the old 30-channel `ea` shape; expect ~20% longer at 36 channels, since
the message count rises and the per-message pull does not. Only a fraction of
pulled bytes is kept — a GRIB message is atomic, so fetching the box still
pulls the global 721×1440 message. `ea-sw-io` is 6.6% of a global message
against `ea`'s 2.3%, so retention rises from the measured ~5% to ~14%. No
concurrency or chunk tuning changes that; only a differently-encoded source
would. Size the network for the pull rate, not the output rate. Details and the
scaling ceiling: `SCALING.md`.

## Files

| | |
|---|---|
| `frisky_daily_dag.py` | task functions + the domain box and 36-channel table. **Shipped to workers** — Frisky pickles task functions by reference |
| `build_corpus.py` | many dates; `--resume` / `--repair` / `--recreate` |
| `parallel_write.py` | a single date via fork/merge |
| `sink_icechunk.py` | schema creation, merge + commit. Client only |
| `deploy_workers.py` | install / scheduler / start / status / stop |
| `check_chunks.py` | completeness from the manifest |
| `check_complete.py` | completeness by reading every block (one date) |
| `check_members.py` | is the ensemble dimension real |
| `bandwidth_probe.py` / `procs_probe.py` | throughput vs threads / processes |
| `dashboard_forward.py` | TCP relay for the dashboard behind jupyter-server-proxy |
| `validate_domain.py` | **pre-flight**: does every channel decode on a given date, and what shape does the box cut. Anonymous, read-only |
| `history/verify_march2026.py` | are the ten March 2026 dates present **and** level-correct in the published store. Anonymous, read-only |

### Reference docs

| | |
|---|---|
| [`run_20260301.md`](history/run_20260301.md) | **A-to-Z record of one real run**: worker startup, dashboard proxy, pre-flight, build, verify, inspect |
| `SCALING.md` | what limits read throughput, measured — read before asking for more VMs |
| `OPENING_PUBLISHED_ECMWF_ICECHUNK.md` | minimal snippet to open the published store anonymously |
| `EWC_REALIZE_NOTES.md` | full notes: both stores, known defects, corrections |
| [`../ECMWF_VARIABLES_REFERENCE.md`](../ECMWF_VARIABLES_REFERENCE.md) | variables and message counts per era |
