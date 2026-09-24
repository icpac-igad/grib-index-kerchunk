# `audit_crosscheck/` — reproducing the GIK-vs-Herbie claims

A runnable harness for the claims this directory makes about Herbie, and for
the external audit of them (`dismantle-jh/GIK_VS_HERBIE_AUDIT.md`, 2026-09-22).

The rule here is the one the audit set and we accept: **every number is
produced by a script in this directory, or it is labelled as extrapolation.**
Nothing is quoted second-hand. Herbie is read and executed from a checkout,
not paraphrased.

```bash
git clone --depth 1 https://github.com/blaylockbk/Herbie.git /tmp/Herbie
HERBIE_SRC=/tmp/Herbie/src ./run_all.sh          # -> results/*.txt
```

Needs `eccodes`, `cfgrib`, `gribberish`, `xarray`, `icechunk`, `cloudpickle`,
and anonymous network access to `ecmwf-forecasts.s3` and `data.source.coop`.
No credentials. No writes to any store.

## What each script tests

| script | claim under test | source of the claim |
|---|---|---|
| `01_decode_single.py` | cfgrib ~2,000 ms vs gribberish ~25 ms ⇒ "80x" | `gefs/docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` §4.1 |
| `02_decode_threaded.py` | gribberish has "no GIL limitations" | same, §5.1 |
| `03_member_axis.py` | Herbie's `enfo` returns 50 of 51 members | `GIK_vs_Herbie_Evaluation.md` §1 |
| `04_dag_schedulability.py` | Herbie can stand in for stages ①–② "at ~2% overhead" | audit §5.4 |
| `05_head_to_head.py` | one date, both paths, phase-resolved | audit §8.2; `REALIZATION_COST.md` §5 "what would settle it" |
| `06_recompute_5_4.py` | the §5.4 arithmetic, with measured constants | audit §5.4 |
| `07_bytes_on_wire.py` | Herbie does a "full GRIB file download", "~1% useful bytes" | `GIK_vs_Herbie_Evaluation.md` §4.2, §6 |

## Results, 2026-09-23

Machine: AMD EPYC 7713, 8 cores, 31 GB. Herbie upstream `e6238f5` (2026-09-19)
— the same commit the audit used. eccodes 2.48.0, cfgrib 0.9.15.1,
gribberish 1.6.0, icechunk 2.1.1, xarray 2026.7.0.

**Claims of ours that did not survive** — these are now wrong in our docs and
must be corrected, not defended:

| | claimed | measured |
|---|---|---|
| cfgrib decode | ~2,000 ms | **42.4 ms** (eccodes-direct 16.0 ms, gribberish 22.2 ms; eccodes vs gribberish `max\|diff\| = 0`) |
| gribberish vs cfgrib | 80x | **1.9x** |
| gribberish threading | "no GIL limitations" | **flat**: 43.0 → 39.8 msg/s over 1→8 threads. eccodes: 63.9 → 152.0, **2.4x** |
| Herbie fetch per message | ~3 GB, "~1% useful bytes" | **601,380 B for a 601,379 B message — 100.0% useful**, 0.009% of the 6.20 GiB step file |
| Herbie parallelism | "no cloud parallelism" | `FastHerbie`, `ThreadPoolExecutor(max_threads=50)` |

**Claims of ours that did survive, and one the audit got wrong:**

| | result |
|---|---|
| member axis | `enfo` exposes `number` 1..50 in **all four eras**. Control is `type=cf` with **no `number`** in 0p4/49r1, and **absent from `enfo` entirely** on 50r1 (`type={'pf': 8500}`). `class ifs` in Herbie contains zero occurrences of `control`, `cf`, `get_control`. |
| store correctness | 9/9 messages **bit-identical** to Herbie (`max\|diff\| = 0.000000e+00`), including the 50r1 control hand-routed to `oper/fc` |
| audit §5.4's index cost | **~3x low** — 510 ms per index acquisition (45 ms parse + 314–465 ms fetch), not the ~187 ms implied. Still only 0.07% of requests, so its *conclusion on index cost* stands. ⚠ An earlier revision said 2,718 ms / 15x; that was Herbie's `index_as_dataframe`, not acquisition — corrected per audit §12.1 |
| what §5.4 never priced | `Herbie()` = 173 ms (network probe in `__init__`); warm object = 1.78 MB cloudpickled; `download() -> Path`, no in-memory read. Per-task construction = **6.1 h/date** → 224 days for the corpus. (Payload is *not* a blocker: an offset table is 0.29 MiB/file, ~50 MB/date) |

See `../AUDIT_REBUTTAL.md` for what follows from this.

## Caveats

One machine, one message type (CCSDS template 42), warm caches on repeated
decodes, and `05` is nine messages cold — not a scale run. It settles which
path is *possible*, and by how much per message on one box. It does not settle
throughput at 48 readers, where `REALIZATION_COST.md` §2 measures the source
rather than the client as the bound. That run has still not been done.
