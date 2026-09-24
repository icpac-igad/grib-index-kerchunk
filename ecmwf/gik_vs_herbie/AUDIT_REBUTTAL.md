# Response to the GIK-vs-Herbie audit

**2026-09-23.** A reply to `dismantle-jh/GIK_VS_HERBIE_AUDIT.md` (2026-09-22),
which re-derived the GIK-versus-Herbie position and found several of our
numbers wrong by one to two orders of magnitude.

Every claim below is produced by a script in `audit_crosscheck/`. Run
`HERBIE_SRC=<clone>/src ./audit_crosscheck/run_all.sh` and check it. Herbie
was executed, not paraphrased, from upstream `e6238f5` (2026-09-19) — the same
commit the audit used.

The audit has been annotated in place with a banner and a **§11 verification
record** carrying the same numbers, so either document can be read first and
neither states a finding the other cannot check. §1–§10 of the audit were not
altered.

The short version, in the order it matters:

1. **The audit is right about the decoder, the network claim, and the
   parallelism claim.** We concede all three without qualification. Three of
   our documents are wrong and are listed for correction in §5.
2. **The audit's §5.4 is the one load-bearing error in it.** It priced the
   replacement of stages ①–② as a bill for `.index` fetches. That bill is real
   and small — and it is not what stages ①–② supply. Measured, the substitution
   it proposes does not run at all at the realization's task granularity.
3. **Both documents were wrong about the member axis, in the same direction,
   for four years of archive.** §3 corrects ours and the audit's together.

---

## 1. What we concede

These are not defended. They are defects in our documents, found by the audit,
and reproduced here on our own hardware.

| our claim | where | measured (`01`, `02`, `07`) |
|---|---|---|
| cfgrib decodes a message in **~2,000 ms** | `gefs/docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` §4.1 | **42.4 ms**. eccodes direct-from-buffer: **16.0 ms**. gribberish: **22.2 ms**, bit-identical to eccodes. |
| gribberish is **80x** faster than cfgrib | same | **1.9x** — and it loses to eccodes-from-buffer |
| gribberish has **"no GIL limitations"** | same, §5.1 | **flat**: 43.0 → 39.8 msg/s across 1→8 threads. eccodes goes 63.9 → 152.0, **2.4x**. The stated reason for adopting it is the opposite of true. |
| Herbie downloads the **full 2–4 GB** step file | `GIK_vs_Herbie_Evaluation.md` §4.2 | **601,380 B fetched for a 601,379 B message** — 100.0% useful, **0.009%** of the 6.20 GiB step file. It is a byte-range GET, the same mechanism as ours. |
| Herbie achieves **"~1% useful bytes"** | same, §6 table | inverted: ~100% |
| Herbie has **"no cloud parallelism"** | same, §4.1 | `FastHerbie` with `ThreadPoolExecutor(max_threads=50)` |

The audit's §5.1 conclusion follows and we accept it: rebuilt on 42.4 ms, the
1,000-day table in `ecmwf/docs/GIK_Icechunk_vs_Herbie_Cost_Analysis.md`
collapses from "150x faster, 2.6x cheaper" to **roughly parity**. That table is
in a document written for outside readers and is the most consequential thing
the audit found. It must be rebuilt or withdrawn.

We also accept §5.3 without reservation: our own `cost-eval` measured chunked
ARCO Zarr beating GIK by 489x on a regional box and 7,800x on a pencil, and
`REPORT.md:105` says so. **The virtual store is an index and must never be
described as "ARCO analysis-ready."** Only the realized store is ARCO.

**The throughput case for GIK is dead. We are not reviving it.** What follows
is not an attempt to.

---

## 2. Where the audit is wrong: §5.4

> "Checked: a Herbie-driven realization would fetch 87 `.index` files per date,
> 887 dates = 77,169 fetches ≈ **4 hours of latency against a 192-hour build —
> about 2% overhead**, and parallelisable to minutes. **The $115 index earns its
> keep as a published artifact for other people, not as an internal dependency
> of your own realization.**"

### 2.1 The arithmetic is ~3x low, and that does not matter

> **Corrected 2026-09-24.** This section first claimed **2,718 ms** per index
> and "15x low". That figure was Herbie's `index_as_dataframe` — DataFrame
> construction, local caching and the search-string column — **not index
> acquisition**, which is what §5.4 priced. The auditor caught it (§12.1) and
> is right. Costing Herbie's DataFrame against a raw-fetch claim was not
> like-for-like, and the error was ours.

Re-measured (`06`), one `.index` acquisition — HTTP GET plus `json.loads` of an
8,500-row, 1.91 MiB sidecar — is **510 ms** on this box (45 ms parse, stable;
314–465 ms fetch, network-dependent), against the ~187 ms the audit's figures
imply. 77,169 of them is **10.9 hours** serial, **5.7%** of the build. §5.4 was
**~2–3x low, not 15x**.

We do not rest anything on this. Against 111.7 M message fetches it is **0.07%
more requests**, and it parallelises. **On index cost, §5.4's conclusion is
correct.** We say so plainly because the rest of this section is not a
throughput argument and should not be mistaken for one.

### 2.2 The error is that offsets are not what stage ② supplies

Here is the realization's entire unit of work, `frisky_daily_dag.py:340`,
executed 125,970 times per date and 111.7 million times over the corpus:

```python
ds[_resolve_var(var, date)] \
  .sel(isobaricInhPa=level) \
  .sel(time=date, number=member, step=step_h,
       latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX)) \
  .values
```

No URL. No stream. No era. No `.index`. No GRIB. The worker does not know it is
reading GRIB at all. **What stage ② provides to stage ③ is a coordinate space,
not a cheaper offset** — and a coordinate space cannot be priced in index
fetches, which is the whole of what §5.4 priced.

The measure of what was absorbed is what is *left*: the entire era-specific
residue in the realization client is `_GRID` (three entries), `_resolve_var`
(the 2024-11-13 `cape`→`mucape` rename), `NOT_IN_0P4`, and `has_dewpoint`.
Four small functions. Everything else — path construction, the four eras'
grids, level sets and member layouts, the 50r1 stream splice — is data in the
store, built once.

### 2.3 Measured: the proposed substitution does not run

§5.4 says "a Herbie-driven realization" as though the phrase costs nothing to
cash. `04_dag_schedulability.py` cashes it.

| | measured |
|---|---|
| `Herbie(...)` construction | **173 ms** — `__init__` probes `SOURCES` over the network. It is not a value object. |
| first `.index_as_dataframe` | **1,853 ms** (8,500 rows, 2.0 MB) |
| warm object, cloudpickled | **1,783,025 B** (cold: 5,185 B) |
| `Herbie.download` return type | **`pathlib.Path`** — no in-memory single-message read exists anywhere in the public API |
| frisky task payload | **46 B** of scalars; the function is pickled by reference |

The realization's graph is one task per message. Under Herbie that graph has
two possible shapes and both fail:

- **One Herbie object per task.** 125,970 × 173 ms = **6.1 hours per date of
  object construction alone**, before a single byte of data moves. Over 887
  dates: **224 days**. Dead.
- **One object per file, warmed and broadcast.** 174 objects per date at
  173 ms each is 30 s — fine. `download()` still returns a `Path`, so
  message-level work means bypassing Herbie's own fetch and issuing raw range
  reads.

> **Corrected 2026-09-24.** This bullet also claimed **310 MB/date** of graph
> payload, from cloudpickling *warm Herbie objects* at 1.78 MB each. The
> auditor's §12.1 is right that nobody would ship that: the object a graph
> needs is an offset table, measured at **0.29 MiB per file, ~50 MB/date** —
> about the size of stage ①'s own output. **Costing the Herbie object was the
> same methodological error we correctly identify in §5.4.** The payload is not
> a blocker. Object *construction* is, and that number stands.

At the end of the second path, Herbie contributes a URL template and a parsed
DataFrame. That is stages ①–② — rebuilt in RAM, per run, unpersisted, without
the era harmonisation. **§5.4 did not evaluate a substitute for stages ①–②. It
re-derived them and called the remainder overhead.**

### 2.4 The amortisation §5.5 asked for, measured

The audit's §5.5 is fair criticism — "$115 vs copying the archive" is a straw
man, and the honest framing is amortisation. Here it is, in bytes rather than
rhetoric (`07`):

| | |
|---|---|
| one `enfo` `.index` | **2,004,349 B** |
| × 87 files/date × 887 dates | **144.1 GiB re-fetched per pass** (bytes, not time — unaffected by §12.1) |
| the published virtual store | **76 GB**, 1,308 dates × 12 groups, fetched once, shared by everyone |

And §5.5's "worth nothing for one team doing one pass" is empirically false for
this corpus: `3-realize-frisky/` holds **22 `build_corpus_*.log` files across
v1, v2 and v3**, plus the wrong-longitude 3 TB rebuild and the `d2m` append.
The index was amortised over at least three full passes because realizations
get redone. That is the normal case, not the exception.

---

## 3. Both documents were wrong about the member axis

This corrects **our** §1 as much as the audit.

`GIK_vs_Herbie_Evaluation.md` §1 says GIK has 51 members "while Herbie's `enfo`
product returns 50 perturbed members (no control)", and treats it as a quirk
worth ~0.6–0.9% in the ensemble mean. `CLAUDE.md` says the control is "bundled
in `enfo` as `number=0`" for 0p4 and 49r1, and that only 50r1 is dual-stream.
Measured (`03`), reading the `.index` sidecars directly:

```
0p4-beta   20230601  enfo  type={'pf': 4150, 'cf':  83}   cf rows carry NO `number`
49r1-9lv   20240529  enfo  type={'pf': 5450, 'cf': 109}   cf rows carry NO `number`
49r1-13lv  20250515  enfo  type={'pf': 7850, 'cf': 157}   cf rows carry NO `number`
50r1       20260901  enfo  type={'pf': 8500}              no cf at all -> oper/fc only
```

So:

- **`enfo` exposes `number` 1..50 in every era.** The control is never
  `number=0`. In 0p4/49r1 it is present as `type=cf` with the `number` key
  **absent**; Herbie's own DataFrame carries `NaN` there, so any loop indexed on
  `number` drops it silently in all four eras.
- **On 50r1 it is not in the file at all.** `enfo` is 8,500 rows of `pf`. The
  control is `oper/fc`, a different stream, and recovering it is a second
  request plus a splice.
- **Herbie's `class ifs` contains zero occurrences of `control`, `cf` or
  `get_control`** (`03`, static check). Its only post-2026-05-12 branch remaps
  `scda`→`oper`. The splice is the caller's job, per era, forever.

Against that, the published virtual store (`03`, `05`):

```
0p4    number=51 [0..50]  time=401  step=85    2023-01-18 .. 2024-02-28
49r1   number=51 [0..50]  time=804  step=85    2024-02-29 .. 2026-05-12
50r1   number=51 [0..50]  time=128  step=85    2026-05-13 .. 2026-09-17
```

`ds.sel(number=0)` is the control in all three groups, whether it came from
`enfo/cf` or from `oper/fc`. **That is the harmonisation, and it is the thing
§5.4 costed at zero.**

This is not a 0.6% statistical footnote. Under a `number`-indexed Herbie
realization the control is absent from every era, and on the 103 realized 50r1
dates it is not recoverable from the requested file at all — silently, because
an unwritten Icechunk chunk reads as `NaN` rather than raising. That is the
same failure mode `frisky_daily_dag.py` documents for the `cape`/`mucape`
rename, which cost us seven weeks of wrongly-resolved SOND 2024 dates before it
was caught.

---

## 4. The benchmark nobody ran

`REALIZATION_COST.md` §5 proposed it, the audit's §8.2 repeated the request,
and neither ran it. `05_head_to_head.py` runs it: one date, both paths, same
box, same nine messages, same machine, phases timed separately.

| | path A — GIK store | path B — Herbie |
|---|---|---|
| per message, cold | **435 ms** | **838 ms** |
| setup | 11.0 s store open, once per process, then cached | ctor 334 ms + 2,718 ms per file for Herbie's `index_as_dataframe` — *its DataFrame build, not index acquisition, which is ~510 ms (§2.1)* |
| download | (included) | 785 ms |
| decode + subset | (included) | **51 ms** |

Two things to take from it, and one not to.

**Take:** the gap is entirely per-object overhead. Decode is 51 ms of the 838 —
about 6%. This is the same finding as the audit's §5.7, and it is why the gap
cannot be closed by swapping decoders, and why it *cannot be amortised* at
message granularity in a task graph (§2.3).

**Take:** all nine messages agree **bit-for-bit**, `max|diff| = 0.000000e+00`,
including the 50r1 control that had to be hand-routed to `oper/fc` to obtain at
all. That is an independent confirmation of the published store through a
completely separate code path, and it is the hand-routing that makes the point
in §3 concrete.

**Do not take:** this as a throughput verdict. Nine messages, cold, one box.
`REALIZATION_COST.md` §2 measures the **source** as the bound at 48 readers,
not the client, so both paths may converge on the same ceiling at scale.
Nothing measured suggests Herbie would be *faster*; nothing measured here
settles by how much it would be slower. **That run still has not been done**,
and we should stop citing per-message ratios as if it had.

---

## 5. Correction queue

Accepted from the audit's §7, plus what this response adds.

| document | section | change | status |
|---|---|---|---|
| `gefs/docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` | §4.1–4.3 | cfgrib is **42.4 ms**, not 2,000; 80x → **1.9x**; add eccodes-direct at 16 ms | accepted |
| " | §5.1 | remove "no GIL limitations" — gribberish is flat across threads, eccodes scales 2.2x | accepted |
| `ecmwf/docs/GIK_Icechunk_vs_Herbie_Cost_Analysis.md` | 1,000-day table | rebuild on 42 ms; both paths ≈ 10–11 h, ≈ $78; drop "150x faster / 2.6x cheaper" | accepted |
| `ecmwf/gik_vs_herbie/GIK_vs_Herbie_Evaluation.md` | §4.1 | Herbie **does** parallelise — `FastHerbie`, `ThreadPoolExecutor(max_threads=50)` | accepted |
| " | §4.2 | Herbie uses byte-ranges; **601,380 B for a 601,379 B message**. Egress is identical. | accepted |
| " | §6 | delete the "Network efficiency: byte-range vs full-file" row | accepted |
| " | §1 | `enfo` returns `number` 1..50 in **all four eras**; the control is `type=cf` unnumbered (0p4/49r1) or absent (50r1) — not a 0.6% quirk | **new, §3** |
| `CLAUDE.md` | ECMWF eras table | "control bundled in `enfo` as `number=0`" is wrong for every era; dual-stream is not 50r1-only in the way stated | **new, §3** |
| `crma/…/1-pars-lithops` cost note | "$115 vs copying" | replace with the amortisation framing — **144.1 GiB of index re-fetched per pass**, ≥3 passes run | accepted, §2.4 |
| anywhere | "ARCO" | never applied to ① or ②; only ③ is ARCO | accepted |
| `dismantle-jh/GIK_VS_HERBIE_AUDIT.md` | §5.4 | the substitution it proposes is not schedulable at message granularity | **rebutted §2; framing withdrawn by the auditor in §12.2** |
| this document | §2.1, §2.3 | 2,718 ms → **510 ms**; "15x" → **~3x**; 310 MB/date → **~50 MB/date** | **our error, corrected 2026-09-24 per audit §12.1** |

---

## 6. What this changes about the position

The audit's §10 stands and we adopt it: **GIK does not beat Herbie on speed, on
cost, or on bytes moved.** Four of our claims to the contrary failed. We should
never have made the throughput argument and should not make it again.

What the audit removed from the case, it removed correctly. What it did not
remove — and what §5.4 mistook for overhead — is this:

> The realized store exists because 111.7 million message reads could be
> addressed as `(time, number, step, level, lat, lon)` across four incompatible
> schema eras, by a worker that contains no GRIB, no S3 path, and no era logic.
> That addressability is what stage ② is. It is not a faster download, and it
> is not a cheaper offset. It is the reason the realization could be written as
> a task graph at all.

That is a correctness-and-schedulability claim, not a performance one, which is
why it survives everything else the audit demolished. It is also, unlike the
numbers in §1, one we can hand someone a script for.

## 7. Still open

1. **The scale benchmark.** One date, both paths, 48 readers, same VMs, from
   Europe and from Nairobi. §4 does not substitute for it.
2. **The GIL test** the audit proposed: one frisky worker at `--threads 1` vs
   `--threads 4`. Given `02`, this is likely to show the realization was
   decode-throttled, and that we spent cluster hours on it.
3. **Re-run `audit_crosscheck/` on an EWC VM.** These numbers are one 8-core
   EPYC; the cluster is what matters.
