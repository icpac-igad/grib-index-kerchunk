# GIK vs Herbie: an audit of the comparison, and what survived it

**Written 2026-09-22.** An independent re-derivation of the GIK-versus-Herbie
position, prompted by a simple question — *is there a like-for-like comparison
anywhere?* — that turned into an audit of the claims themselves.

The short version: **the throughput case for GIK does not survive measurement,
and never needed to.** What survives is narrower, quantitative, and better:
two artifacts nobody else holds, and a chunking decision worth 51x. Several
widely-quoted numbers in the repo are wrong by one to two orders of magnitude,
and three of them are load-bearing in documents intended for outside readers.

Read §1–2 for provenance, §3 for how the audit was conducted, §4 for what was
measured, §5 for the uncomfortable findings and how each was reached, §6 for
what stands up, §7 for the correction queue.

---

> **Responded to 2026-09-23.** The audited project replied in
> `grib-index-kerchunk/ecmwf/gik_vs_herbie/AUDIT_REBUTTAL.md` and built a
> runnable harness for the claims on both sides:
> `grib-index-kerchunk/ecmwf/gik_vs_herbie/audit_crosscheck/`
> (`HERBIE_SRC=<clone>/src ./run_all.sh`; committed output in `results/`).
> Herbie was executed from `e6238f5` — the same commit used here.
>
> **Independently reproduced, audit correct:** §4.2 decode (42.4 / 22.2 /
> 16.0 ms for cfgrib / gribberish / eccodes-direct), §4.3 threading
> (gribberish flat 43.0 → 39.8 msg/s; eccodes 63.9 → 152.0), §5.6 both claims
> — and §4.2's byte-range finding is now measured, not inferred: Herbie moved
> **601,380 B for a 601,379 B message**, 0.009% of the 6.20 GiB step file.
> §5.1, §5.3 and §10 stand as written and the project accepts them. §5.2's
> *finding* stands (gribberish does not thread); its inference that the
> realization was therefore decode-throttled is withdrawn in §12.7.
>
> **Corrected by the harness:** §5.4 — see §11, appended by the respondent,
> and then §12, the auditor's re-check of §11.
>
> **Read §12 before acting on §11.** The re-check found §11.2's headline
> numbers wrong in the respondent's favour and the respondent accepts it:
> per-index cost is **~510 ms**, not 2,718 ms (that was Herbie's
> `index_as_dataframe`, not index acquisition), so §5.4 was **~2–3x low, not
> 15x**; and the graph payload is an offset table at **~50 MB/date**, not
> 310 MB/date of warm Herbie objects. §12.2 nevertheless **withdraws §5.4's
> framing**, and §12.7 withdraws §5.2's throughput claim. §8.2 was run at
> reduced scale (§12.11) and at 48 readers from EWC (§12.12).
>
> This banner, §11, §12 and §13 are additive. Nothing in §1–§10 has been
> altered.

---

## 1. What was examined

### 1.1 Git repositories

| repo | location | state at audit |
|---|---|---|
| `grib-index-kerchunk` | `/var/lib/private/nishadhka/grib-index-kerchunk` | branch `main`, HEAD **`b6dc3ef`** (2026-09-18), **392 commits**, first `4014ba2` 2024-06-13 |
| `crma` | `/var/lib/private/nishadhka/crma` | contains `medium-range-forecast/{1-pars-lithops,2-icechunk-virtual,3-realize-frisky}`; cost note committed as `63c0a39` |
| `code-for-earth` | `/var/lib/private/nishadhka/code-for-earth` | demos 1–7 |
| **Herbie** (upstream) | `https://github.com/blaylockbk/Herbie.git` | cloned `--depth 50` during the audit; HEAD **`e6238f5`** (2026-09-19), *"Merge pull request #556 from blaylockbk/logo-redesign"* |

Herbie was cloned because every claim about it in this repo was second-hand.
It is current as of three days before this audit, so nothing here is about a
stale version.

### 1.2 Commit history reviewed

All 392 commits of `grib-index-kerchunk`, with close reading of the 262 since
2025-11-01. Commits per month were tabulated to locate the periods of dense
development; the phases in §3.2 come from that tabulation.

### 1.3 Session transcripts

| session id | project | size |
|---|---|---|
| `daf6eda3-111b-48af-944a-6d09dd2dc819` | `3-realize-frisky` | 17.9 MB |
| `5b0d0b97-09e6-48a2-9f83-6592a0e42da6` | `ecmwf/gik_vs_herbie` | 6.7 MB |
| `3e64e564-f07a-4603-a152-fb671b74c64f` | `1-pars-lithops` | 4.6 MB |
| `8b614750-3079-437f-bd22-a2065221d6e9` | `icechunk-par` | 2.1 MB |

### 1.4 Markdown documents consulted

| document | what was taken from it |
|---|---|
| `crma/…/3-realize-frisky/REALIZATION_COST.md` | 887 dates, **192.4 cluster-hours**, 111.7 M messages, 161 msg/s over 48 readers, **88.0 TB read → 14.9 TB written (5.9x)**, 127 MB/s sustained; §5 structural Herbie argument; §A1 resource profile; §A3 GCP pricing |
| `grib-index-kerchunk/ecmwf/gik_vs_herbie/GIK_vs_Herbie_Evaluation.md` | §3 per-date timing — **the only measured Herbie timing in the repo**; §4.1 "no cloud parallelism"; §4.2 "full-file download"; §7 pressure-level validation |
| `…/gik_vs_herbie/realized_ea_swio_eval/README.md` | 18 comparisons, **per-member `max|diff| = 0`**, coverage gaps, the preallocation hazard |
| `…/icechunk-par/REALIZED_EA_SWIO_STORE.md` | **chunk `(1, 1, 51, 321, 261)`** (line 76), 16 sub-stores, 1,264/1,271 date slots, credential-free read |
| `…/icechunk-par/CONVERSION_ESTIMATE.md` | **5.2 h** single-VM for 1,256 dates, 20.6 GB of pars, ~13 GB store, manifest splitting |
| `…/gefs/docs/gik_vs_herbie_comparison.md` | per-stage timing (Stage 2 = **89%** of GIK time), scenario table incl. **684 s vs 2,880 s** |
| `…/gefs/docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` | the **80x** decode claim, **cfgrib ~2,000 ms**, CCSDS template-42 failure, §5.1 "no GIL limitations" |
| `…/ecmwf/docs/GIK_Icechunk_vs_Herbie_Cost_Analysis.md` | the 1,000-day table — **GIK ~$78 / 10 h vs Herbie ~$205 / 64 days**, "150x faster" |
| `…/gefs/GIK_DASK_COST_PARADIGM.md` | three live Coiled bills, **$0.74–$2.20** per analysis |
| `…/cost-eval/results/REPORT.md` | ECMWF **1.109 PiB** / GEFS 0.303 PiB archive; GIK catalog **$8.96/yr**; ARCO duplicate **~$68,560/yr**; line 105 conclusion |
| `crma/…/1-pars-lithops/README.md` | stage ①: **"No GRIB data is downloaded… 51 parquets total ~22 MB and stand in for ~300 GB of GRIB"** |
| `crma/…/1-pars-lithops/cycles/README.md` | **$0.026/date**, relative cost 0.58 for 49-step cycles, §7 cost |

### 1.5 Data and results files

- `cost-eval/results/cost_matrix.csv`, `05_streaming_summary.json`, `04_herbie_summary.json` — the GIK-vs-ARCO-Zarr measurements
- `gik_vs_herbie/realized_ea_swio_eval/*.json` — 18 correctness comparisons; **checked for timing fields and found none**
- `crma/…/run_0p4_seasons.log`, `run_d2m_seasons.log` — the later builds that reconcile 14.9 TB to ~17.7 TB
- `code-for-earth/demo6-…/demo_open_virtual_icechunk_store_and_realize.ipynb` — executed outputs giving the virtual store's scale

---

## 2. The three artifacts, which must not be conflated

Most of the confusion in the existing documents comes from treating these as
one thing. They have different costs, different consumers, and different
comparisons to Herbie.

| | what it is | size | build cost | vs Herbie |
|---|---|---|---|---|
| **① parquet catalog** | `[url, offset, length]` per message, per member, per date | 97 GB | **$107–135** (Cloud Run) | Herbie re-derives the same from `.index` at runtime |
| **② virtual Icechunk store** | ① assembled into one queryable dataset | **76 GB** published | **<$1–8** (VM) | Herbie has no equivalent **at any price** |
| **③ realized store** | decoded, cut to EA-SWIO, written as real chunks | **~17.7 TB** | **$46 Spot / $155 on-demand** | not comparable — different kind of object |

Stage ③ is the only one that is ARCO. Stages ① and ② are an **index**.

### 2.1 The scale of ②, measured

From the executed notebook (`demo6`), 12 groups `{0p4,49r1,50r1}` × `{00z,06z,12z,18z}`,
1,308 dates, 2023-01-18 → 2026-08-23:

| | |
|---|---|
| date-cycles | **5,230** |
| virtual chunk references | **2,545,480,023** |
| store on disk (manifests) | **76 GB** |
| GRIB virtualized | **1.90 PB** — measured by anonymous S3 LIST, re-derived in cell 9 |
| addressable array space | **9.42 PB** float32 |
| **leverage** | **24,869x GRIB bytes addressed per byte stored** |

Total index build: **≈ $115** for one clean pass, plus $4.55/month storage.
The cost note flags itself as list-price arithmetic, not an invoice, and states
actual spend was materially higher because of era-mismatch re-runs, the
pre-per-level-fix regeneration, and the `-v4` rebuild.

---

## 3. Method

### 3.1 The discipline applied

Three rules, adopted because the existing documents mix the categories freely:

1. **Separate measurement from extrapolation.** Every number below is labelled
   as one or the other. `REALIZATION_COST.md` §5 does this well and says
   outright that no Herbie benchmark was run; most other documents do not.
2. **Trace every quoted number to a log, an object listing, or a run.** Where a
   number could not be traced, that is stated rather than repeated.
3. **When two documents disagree, go to the source.** Twice this meant reading
   Herbie's code; once it meant running a benchmark.

### 3.2 The chain of steps

**Step 1 — locate the comparison.** Three candidate sources. Only one contains
Herbie *timing*: `GIK_vs_Herbie_Evaluation.md` §3 (5 dates, `tp`, 9 steps).
The 2026-08 realized-store evaluation was checked directly —
`compare_realized_herbie.py` has no timer and the stats JSON has no duration
field. It is a **correctness** study, not a performance one.

**Step 2 — find the contradiction.** §4.2 claims Herbie downloads full 2–4 GB
GRIB files, ~1% useful bytes. `REALIZATION_COST.md` §5 says egress is
identical. Session `5b0d0b97` line 686 settles it: *"Herbie's byte range
matches the source exactly, so it reads the same message."* Commit `22b7fcb`
(2026-03-18) corrected the GEFS document accordingly — **the ECMWF document was
never updated.**

**Step 3 — normalise the rates.** §3's Herbie figures are 50 members × 9 steps
= 450 messages in 63–133 s, mean 99.4 s → **4.53 msg/s per process**.
Realization ran 161 msg/s over 48 readers → **3.35 msg/s per reader**. Herbie's
single stream was *faster*. This is the first sign the throughput argument is
backwards.

**Step 4 — read the history.** Commits per month, then the 262 since Nov 2025.
The arc: `scan_grib` replaced by template + `.idx` at 68x then 112x
(`e0a4167`, `d733ccf`); Herbie acknowledged as a byte-range reader
(`22b7fcb`); the cost-eval harness (`803ff9c`); then realization. The project
had already found most of this itself.

**Step 5 — read the cost-eval.** `cost_matrix.csv` compares GIK not to Herbie
but to a chunked ARCO Zarr. The result, four months before the realization run,
is that byte-range streaming is the wrong access pattern for regional and
point queries — by 489x and 7,800x. The realization is the response.

**Step 6 — measure the decoder.** The 80x claim is the last pillar. eccodes,
cfgrib and gribberish were all installed, so it was benchmarked rather than
argued. §4 below.

**Step 7 — read Herbie's source.** Cloned upstream and checked the five claims
that depend on its internals.

**Step 8 — read the notebook and the build session.** The scale of ② and the
$115, which reframe the whole question away from throughput.

---

## 4. What was measured

### 4.1 Setup

| | |
|---|---|
| machine | AMD EPYC 7713, **8 cores**, 31 GB RAM |
| libraries | eccodes **2.48.0**, cfgrib **0.9.15.1**, gribberish **1.6.0**, xarray 2026.7.0, numpy 2.5.1 |
| data | ECMWF `20260915` 00z `enfo` **T+24h**, index 8,500 lines |
| messages | `t850` 589 KiB (offset 1,937,718,789 / length 602,887); `t2m` 641 KiB (offset 544,932,015 / length 656,638) |
| method | median of 15 reps; bytes fetched once by HTTP Range, then decoded repeatedly |

Three paths compared: **eccodes direct from an in-memory buffer**
(`codes_new_from_message` → `codes_get_values`), **cfgrib** (temp file →
`xr.open_dataset`, which is Herbie's path), and **gribberish**
(`parse_grib_array` on the buffer).

### 4.2 Single-message decode

| decoder | `t850` | `t2m` | the repo claims |
|---|---|---|---|
| **eccodes, direct from buffer** | **16.6 ms** | **16.0 ms** | not measured anywhere |
| cfgrib (temp file → xarray) | 42.0 ms | 40.7 ms | **~2,000 ms** |
| gribberish, direct | 22.3 ms | 22.4 ms | ~25 ms ✓ |

Agreement eccodes vs gribberish: **`max|diff| = 0.000e+00`**, bit-identical.

The cfgrib index-sidecar hypothesis was tested and rejected: 42.4 ms with
`indexpath=""` vs 42.3 ms with the default. **The ~2,000 ms figure is off by
~48x** against this machine. The most likely origin is that the original
measurement opened a whole step file (8,500 messages, ~6.5 GB) rather than one
extracted message — cfgrib must scan every message to build its index, and
~2 s is about right for that. If so, it compared gribberish decoding *one
message* against cfgrib indexing *an entire file*.

### 4.3 Threaded scaling — the finding that reverses the argument

Same message, decoded concurrently:

| threads | eccodes | gribberish |
|---|---|---|
| 1 | 63.7 msg/s | 40.8 msg/s |
| 2 | 121.6 msg/s | 39.9 msg/s |
| 4 | **141.9 msg/s** | 39.4 msg/s |
| 8 | 122.0 msg/s | 39.0 msg/s |

**eccodes scales 2.2x across threads. gribberish is flat.** Its PyO3 binding
holds the GIL through the decode; the eccodes CFFI bindings release it. This is
the precise opposite of `GRIBBERISH_VS_CFGRIB_ANALYSIS.md` §5.1's *"Native Rust
performance: No GIL limitations"* — the stated reason for adopting it.

### 4.4 Herbie's source, checked directly

| claim | evidence | verdict |
|---|---|---|
| byte-range reads via `.idx` | `core.py:723-860` | **same mechanism as GIK** |
| cannot parallelise (§4.1) | `fast.py:90` `FastHerbie`; `ThreadPoolExecutor` at `:140`, `:237`, `:294`, `max_threads=50` | **false** |
| re-fetches `.idx` per request | `core.py:723` `@functools.cached_property` on `index_as_dataframe` | cached **per object** (per file), not per message — the "3,240 idx fetches" framing is wrong; it is per *file* |
| decodes via temp file + cfgrib | `core.py:25` `import cfgrib`; `core.py:1338` `cfgrib.open_datasets(local_file, …)` | **true — no in-memory path** |
| already disables the cfgrib sidecar | `core.py:1317` `backend_kwargs.setdefault("indexpath", "")` | true — so §4.2's 42 ms *is* Herbie's decode cost |
| can build a Zarr / reference / virtual store | `grep -i "to_zarr\|kerchunk\|icechunk\|virtualizarr\|arco"` over `src/herbie/` | **zero hits**; `hrrr_zarr.py` only *reads* an external store |

The "inherent impossibility" of using a Rust decoder with Herbie is not real:
`.download()` yields bytes, and only `.xarray()` is cfgrib-flavoured. But note
the irony — **eccodes-direct is both faster than gribberish and threads**, so
the better swap is not to Rust at all.

**On whether Herbie has *stated* a position on Rust decoders: nothing was
found.** No commit message in either repo records one. This is left open rather
than guessed.

---

## 5. The uncomfortable findings

Each with the derivation, so it can be argued with.

### 5.1 The decoder claim is wrong by ~48x, and it propagates

**Claim.** `GRIBBERISH_VS_CFGRIB_ANALYSIS.md` §4.1: cfgrib ~2,000 ms/message,
gribberish ~25 ms, **80x**.

**Measured.** 42 ms vs 22 ms — **1.9x**, and eccodes-direct at 16 ms beats both.

**Why it matters.** The figure is not confined to that document. It is the
engine of the 1,000-day table in `GIK_Icechunk_vs_Herbie_Cost_Analysis.md`:

| 1,000 days = 2.7 M messages | as documented | recomputed |
|---|---|---|
| decode, 1 thread | 2,000 ms → **62.6 days** | 42 ms → **31.5 hours** |
| decode, 50 workers | — | **~0.6 hours** |
| fetch 2.2 TB, 50 workers | ~10 h (the doc's own GIK figure) | same for both — the doc says *"both methods stream the same data volume"* |
| **total** | GIK 10 h / **Herbie 64 days, $205** | **both ≈ 10–11 h, ≈ $78** |

**"~150x faster and ~2.6x cheaper" collapses to roughly parity.** This is the
single most consequential error found, because that table is in a document
written for outside readers.

### 5.2 The decoder was chosen for a property it does not have

§5.1 of the analysis cites "no GIL limitations" as gribberish's advantage.
Measured (§4.3), gribberish does not scale across threads and eccodes does.

Frisky runs **4 threads per worker**. Commit `a5de8cc` records *"Two worker
processes per VM: the ~31 MB/s cap was per-process"* — processes helped,
threads did not. `REALIZATION_COST.md` §A1 measures load ~1.2 on 4 vCPU with 8
reader threads and reads it as "I/O-bound, idle CPU." **A GIL-serialised
decoder produces the same signature.** The two cannot be separated from
existing logs.

**The test is cheap:** run one frisky worker at `--threads 1` against
`--threads 4` and compare per-worker throughput. If it does not move, three of
the four threads are queued on the GIL, and the realization was slower than it
needed to be.

### 5.3 The project measured its own disadvantage and the document was never updated

`cost-eval/results/cost_matrix.csv`, May 2026, GIK vs chunked ARCO Zarr:

| ECMWF, 1 variable | GIK on wire | Zarr on wire | ratio |
|---|---|---|---|
| regional box | 20,671 MiB | **42 MiB** | **489x** |
| pencil (point, full archive) | 15.6 TiB — **$1,510** | 2.0 GiB — **$0.19** | **7,800x** |

`REPORT.md:105` states it: *"dynamical.org's chunked Zarr genuinely beats GIK
for regional and pencil workloads."* This is correct, it is the project's own
measurement, and it means **the virtual store must never be described as
"ARCO analysis-ready."** It is an index.

To the project's credit, this finding is what the realization answers.

### 5.4 Stage ① is not load-bearing for stage ③

The stage ① README says stages ①–② exist so stage ③ can run over a chosen box.
Checked: a Herbie-driven realization would fetch 87 `.index` files per date,
887 dates = 77,169 fetches ≈ **4 hours of latency against a 192-hour build —
about 2% overhead**, and parallelisable to minutes.

**The $115 index earns its keep as a published artifact for other people, not
as an internal dependency of your own realization.** That is a real value, but
it is a different value from the one claimed.

### 5.5 The headline economic comparison is against a straw man

The cost note says: *"indexing a 1.9 PB archive costs ~$115, where copying it
would not be attempted."*

Nobody proposed copying it — **and Herbie does not copy it either.** Herbie
reads the same `.index` files at runtime and downloads no GRIB to find an
offset. The honest contrast is:

| | upfront | per consumer, per run |
|---|---|---|
| GIK virtual store | **$115 once** | ~0 — the offset is already known |
| Herbie | $0 | ~200 ms × files touched, on every machine, forever |

Touching every index in the archive once is ~444,550 files ≈ **25 hours of
latency, ~$0 in dollars**. So $115 buys *not doing that again* — worth it if
the index is consumed many times by many people, worth nothing for one team
doing one pass.

### 5.6 Two further claims that do not hold

- **"No cloud parallelism"** (`GIK_vs_Herbie_Evaluation.md` §4.1) — `FastHerbie`
  exists and uses `ThreadPoolExecutor(max_threads=50)`.
- **"Full-file download, ~1% useful bytes"** (§4.2) — corrected in the GEFS
  document by `22b7fcb` on 2026-03-18; the ECMWF document has carried the wrong
  claim for **six months**, and it overstates the GIK advantage by ~100x.

### 5.7 How the 4.2x in the scenario table actually closes

The one row where GIK wins — *12 vars, 30 members, pre-built refs: 684 s vs
2,880 s* — is 3,240 chunks, so 211 ms/chunk (8 threads) vs 889 ms/chunk
(sequential). Decode is 16–42 ms of that, **under 5% either way**. The gap is:

| Herbie's 889 ms/chunk | fix | after |
|---|---|---|
| `.idx` re-fetch per Herbie object | cache per file | ~0 |
| temp file + `open_datasets` | `codes_new_from_message(buf)` | 42 → **16 ms** |
| sequential | `ThreadPoolExecutor(8)`, eccodes threads | ÷2.2 |

With those three, a Herbie-based fetcher converges on GIK's pre-built-ref path.
**None of them require GIK's machinery.**

---

## 6. What survives

Everything above is about throughput and cost claims. These are unaffected:

### 6.1 The realized store's chunking — the real result

`REALIZED_EA_SWIO_STORE.md:76` — chunk **`(1, 1, 51, 321, 261)`**, all 51
members of a `(date, channel, step)` in one object:

| | requests per date | decode on read |
|---|---|---|
| realized store | **2,470** | none |
| virtual store / Herbie | **125,970** | every read |

**51x fewer round trips**, on a workload `REALIZATION_COST.md` §A1 measures as
latency-bound. When latency dominates, request count *is* the cost. This holds
under every measurement in this audit, and it is what the 200 hours bought —
not a faster download, a different access pattern.

### 6.2 The leverage of the virtual store

**76 GB addressing 1.90 PB — 24,869x — for ~$115.** Herbie cannot produce this
artifact at any price, and that is a structural fact (§4.4), not a benchmark.

### 6.3 Correctness

18 comparisons against Herbie on the published realized store, **per-member
`max|diff| = 0`**; `r > 0.9999` across the surface and pressure-level studies;
`gh` at 300/500 hPa matching to `r = 1.000000` after the per-level fix. Herbie
was used well here — as ground truth, which is what it is good for.

### 6.4 Provenance

Icechunk snapshots pin which bytes were read. A Herbie run is not versioned.
Cheap, real, and not why Icechunk was chosen.

### 6.5 The corpus reconciliation

| build | dates | cluster hours | written |
|---|---|---|---|
| 0.25° v3, 12 seasons | 887 | 192.4 | 14.9 TB |
| 0.4° 2023, 4 seasons | 365 | ~25 | ~2.4 TB |
| d2m companions, 11 seasons | 882 | 6.5 | ~0.36 TB |
| **total** | | **~224 h** | **~17.7 TB** |

The informal "200 hours / 18 TB" is accurate.

---

## 7. Correction queue

| document | section | change |
|---|---|---|
| `ecmwf/gik_vs_herbie/GIK_vs_Herbie_Evaluation.md` | §4.1 | Herbie **does** parallelise — `FastHerbie`, `ThreadPoolExecutor(max_threads=50)` |
| " | §4.2 | Herbie uses byte-ranges; egress is identical. Already fixed in GEFS by `22b7fcb` |
| " | §6 | "Network efficiency: byte-range vs full-file" row is wrong |
| `gefs/docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` | §4.1, §4.2, §4.3 | cfgrib is **42 ms**, not 2,000 ms; 80x → **1.9x**; add eccodes-direct at 16 ms |
| " | §5.1 | remove "no GIL limitations" — gribberish does not scale across threads |
| `ecmwf/docs/GIK_Icechunk_vs_Herbie_Cost_Analysis.md` | 1,000-day table | rebuild on 42 ms; **both paths ≈ 10–11 h, ≈ $78**; drop "150x faster / 2.6x cheaper" |
| `crma/…/1-pars-lithops` cost note | "$115 vs copying" | replace with the amortisation framing of §5.5 |
| anywhere | "ARCO" | never applied to ① or ②; only ③ is ARCO |

---

## 8. What would settle the rest

1. **The GIL test.** One frisky worker, `--threads 1` vs `--threads 4`. An hour
   of work; decides whether the realization was decode-throttled.
2. **The benchmark §5 of `REALIZATION_COST.md` proposed and nobody ran.** One
   date, both paths, same box and channels, same VM, timing index acquisition /
   decode / wall-clock separately — **and run it twice, once from Europe and
   once from Nairobi**, so the RTT penalty becomes a measurement.
3. **Re-run §4 on an EWC VM.** The decode numbers here are from one 8-core
   EPYC; the cluster is what matters.

## 9. Caveats on this audit

- The decode benchmark reuses one message, so caches are warm; one message type
  (CCSDS template 42); one machine, 8 cores.
- Herbie's timing is extrapolated from the repo's own §3 measurements, not
  re-measured — Herbie is not installed in this environment.
- The $115 is the repo's list-price arithmetic, not an invoice, and the repo
  says actual spend was materially higher.
- Archive sizes come from two different extrapolations at two different scopes
  (1.109 PiB in `cost-eval`, 1.90 PB in the notebook); both are internally
  consistent, and the notebook's is the one that describes the published store.

---

## 10. The conclusion

The question that started this was whether GIK beats Herbie. It does not, on
speed, on cost, or on bytes moved — and the repo's own commits had already
found most of that before this audit began (`22b7fcb`, `e0a4167`, `803ff9c`).
The GIK-beats-Herbie framing has now failed four separate checks.

That framing was never needed. Two artifacts exist that nobody else holds:

- a **76 GB public index to 1.90 PB** of IFS GRIB, openable anonymously, at
  24,869x leverage, built for ~$115;
- an **ARCO regional store** answering a query in 2,470 requests per date
  instead of 125,970, built for $46 of Spot compute.

Neither claim depends on Herbie being slow. Both are measured. Those are the
numbers to take outside — and a rebuild today, knowing that Herbie does
byte-ranges, that eccodes threads and gribberish does not, and that the `.idx`
is cacheable, would cost a fraction of what the first one did.

---

## 11. Verification record — appended 2026-09-23 by the audited project

Added by the respondent, not by the auditor. §1–§10 above are untouched.
Every number here comes from `audit_crosscheck/results/`.

### 11.1 What the harness confirms

The audit's central findings reproduce on independent hardware, and two that
were reasoned are now measured:

| audit § | status |
|---|---|
| §4.2 decode, §5.1 "wrong by ~48x" | **confirmed** — 42.4 ms, not 2,000. `01` |
| §4.3 / §5.2 gribberish does not thread | **confirmed** — flat 43.0 → 39.8 msg/s; eccodes 2.4x. `02` |
| §5.6 "no cloud parallelism" is false | **confirmed** — `FastHerbie`, `ThreadPoolExecutor(max_threads=50)` |
| §5.6 "full-file download" is false | **confirmed and measured** — 601,380 B for a 601,379 B message, 100.0% useful. `07` |
| §4.4 "no in-memory path" | **confirmed** — `Herbie.download` return annotation is `pathlib.Path`; no public method returns a buffer. `04` |
| §4.4 no store writer | **confirmed** — zero hits for `to_zarr\|kerchunk\|icechunk\|virtualizarr` in `src/herbie/` |
| §6.3 correctness | **re-confirmed via a third path** — 9/9 messages `max\|diff\| = 0.000000e+00` between the published store and Herbie. `05` |
| §10 conclusion | **accepted by the project.** The throughput case for GIK is dead. |

### 11.2 What it corrects

**§5.4's index arithmetic is ~15x low.** One `.index` acquisition is a 2.0 MB
JSON-lines fetch plus parse: **2,718 ms measured**, not the ~187 ms implied by
"77,169 fetches ≈ 4 hours". Serial cost is 58.3 h, ~30% of the build, not 2%.
*This does not change §5.4's conclusion on index cost* — 0.07% more requests
against 111.7 M — and the respondent says so explicitly. `06`

**§5.4's inference does not follow.** The claim "stage ① is not load-bearing
for stage ③" priced offsets. What stage ② supplies to stage ③ is a coordinate
space. Measured cost of the proposed substitution, which §5.4 did not cost:

| | measured (`04`) |
|---|---|
| `Herbie(...)` construction | **173 ms** — `__init__` probes `SOURCES` over the network |
| first `.index_as_dataframe` | **1,853 ms** |
| warm object, cloudpickled | **1.78 MB** (cold 5.2 KB) |
| frisky task payload | **46 B**, function pickled by reference |

The realization's graph is one task per message, 125,970 per date. One Herbie
object per task is **6.1 h/date of construction alone — 224 days** for the
corpus. One object per file means broadcasting **310 MB/date** of parsed index,
or re-fetching per worker *process* (`cached_property` is per object per
process). Either way `download()` still returns a `Path`, so message-level work
bypasses Herbie's fetch entirely. §5.4 did not evaluate a substitute for
stages ①–②; it re-derived them and called the remainder overhead.

### 11.3 A finding that corrects both documents

Reading the `.index` sidecars directly across all four eras (`03`):

```
0p4-beta   enfo  type={'pf': 4150, 'cf':  83}   cf rows carry NO `number`
49r1-9lv   enfo  type={'pf': 5450, 'cf': 109}   cf rows carry NO `number`
49r1-13lv  enfo  type={'pf': 7850, 'cf': 157}   cf rows carry NO `number`
50r1       enfo  type={'pf': 8500}              no cf at all -> oper/fc only
```

`enfo` exposes `number` 1..50 in **every** era; the control is never
`number=0`. Herbie's `class ifs` contains zero occurrences of `control`, `cf`
or `get_control`. The published store exposes `number=[0..50]` uniformly for
`0p4`, `49r1` and `50r1` — that splice is the harmonisation §5.4 costed at
zero.

**This touches the audit's own method.** §3 Step 3 normalises Herbie's rate
from "50 members × 9 steps = 450 messages" and compares it to the
realization's 51-member throughput, concluding "Herbie's single stream was
*faster*." The 50 is not a neutral constant — it is the missing control, the
same defect §11.3 describes, used as a baseline. The direction of Step 3's
conclusion is unaffected at this magnitude (2%), but the comparison is not
like-for-like and should be re-normalised or annotated.

### 11.4 Against §8

- **§8.2** (the benchmark nobody ran) is now run *at small scale*: one date,
  both paths, same box, phases separated — 435 ms vs 838 ms per message cold,
  of which decode is 51 ms. `05`. **This does not discharge §8.2.** Nine cold
  messages on one 8-core box is not 48 readers against a source that
  `REALIZATION_COST.md` §2 measures as the binding constraint. The scale run,
  and the Europe/Nairobi split, remain undone.
- **§8.1** (the GIL test) is untouched. *(Superseded: §12.7 resolves it from
  existing logs — 12 processes x ~40 msg/s gave ~3x headroom over the 161
  msg/s achieved, so the GIL flaw cost this run nothing.)*
- **§8.3** (re-run on an EWC VM) is untouched. All numbers here are one EPYC.

---

## 12. Re-check of §11 — appended 2026-09-23 by the auditor

§11 was checked rather than accepted. The harness in `audit_crosscheck/` is
real, runnable and committed, and it reproduces the audit's central findings on
independent hardware. It also makes two corrections to §5.4. **One of them does
not survive measurement; the other does, and §5.4 is partly withdrawn because
of it.**

### 12.1 The 2,718 ms index figure is wrong by ~7.6x

§11.2 states: *"One `.index` acquisition is a 2.0 MB JSON-lines fetch plus
parse: 2,718 ms measured, not the ~187 ms implied."*

Re-measured on the same object (`20260915` 00z T+24h, 2.00 MB, 8,500 rows),
5 reps, median:

| operation | measured |
|---|---|
| HTTP fetch | **314 ms** |
| `json.loads` per line → dicts | **45 ms** |
| **fetch + parse** | **358 ms** |
| `pd.read_json(lines=True)` instead of `json.loads` | 75 ms |
| + datetime columns + search string (Herbie-shaped) | 206 ms → **~520 ms** total |

§11's own `04_dag_schedulability.txt` explains the discrepancy: *"first
`.index_as_dataframe`: 1808.8 ms"*. **The 2,718 ms is the cost of Herbie's
DataFrame construction, not of acquiring an index.** §5.4 priced obtaining byte
offsets, which costs 358 ms.

| | §5.4 assumed | §11.2 claimed | measured |
|---|---|---|---|
| per index | 187 ms | 2,718 ms | **358 ms** |
| serial, 77,169 fetches | 4.0 h | 58.3 h | **7.7 h** |
| as % of the 192.4 h build | 2% | 30.3% | **4.0%** |
| at 48 readers | — | 72.8 min | **~10 min** |

**§5.4 was ~2x low, not ~15x.**

The payload argument carries the same error. §11.2 prices *warm Herbie objects*
at 1.78 MB × 174 files = 310 MB/date. The object a task graph needs is an
offset table: **0.25 MB pickled per file**, built in 48 ms from the raw bytes —
**~22 MB/date**, which is precisely the size of stage ①'s output ("51 parquets
total ~22 MB"). §11.2 costed a substitute nobody would build, which is the same
methodological error it correctly identifies in §5.4.

*Caveat:* these fetches may hit a warm S3 edge for an object pulled repeatedly
during the audit. The parse half — 45 ms against an implied ~2,400 ms — cannot
be a caching artifact, and §11's own `04` accounts for the remainder.

### 12.2 §5.4's framing is withdrawn

Strip the arithmetic and §11.2's underlying argument is correct.

§5.4 says *"stage ① is not load-bearing for stage ③."* **That is withdrawn.**
The *published artifact* is not an internal dependency — it could be rebuilt at
runtime for ~4% of build time. But the *function* is load-bearing: a
125,970-task-per-date message-level DAG cannot run without something that turns
index files into a compact, broadcastable offset table. That is what stage ①
is. §5.4 priced offsets, found them cheap, and called the remainder overhead
without costing the substitute as a task graph.

**Corrected statement:** stage ①'s function is necessary, *and* it is cheap —
~7.7 h serial, ~10 min at 48 readers, ~22 MB/date of payload. Both documents
were wrong, in opposite directions.

§5.4's conclusion on *index cost* survives, and §11.2 concedes this explicitly.

### 12.3 A finding the audit missed — promoted into §6

§11.3 is the strongest item in the rebuttal and is independent of all timing.
Verified here directly against the `.index` sidecars:

```
49r1  20250315   types={'pf': 7850, 'cf': 157}   pf number 1..50   cf rows carry 'number': False
50r1  20260915   types={'pf': 8500}              pf number 1..50   NO cf ROWS AT ALL
```

**Confirmed.** The control member is not addressable as `number=0` in any era,
and on 50r1 it is absent from `enfo` entirely. The published store exposes
`number=[0..50]` uniformly across `0p4`, `49r1` and `50r1`. That harmonisation
is not derivable from the index, Herbie's `class ifs` does not perform it, and
**the audit costed it at zero.**

It belongs in §6 as **6.6 — the member axis**, alongside the chunking and the
leverage. It is semantic work, not plumbing, and it is the clearest example of
what stages ①–② contribute that no client library supplies.

§11.3's note on §3 Step 3 — that Herbie's 50-member baseline was compared
against a 51-member store — is accepted as an annotation. It does not move the
number: msg/s is normalised per message, so 450 versus 459 does not bias the
rate, and §11.3 concedes the direction is unaffected.

### 12.4 Status after the re-check

| audit finding | status |
|---|---|
| §5.1 decode wrong by ~48x, and propagates | **stands** — reproduced by §11 |
| §5.2 gribberish chosen for a property it lacks | **stands** — reproduced by §11 |
| §5.3 the cost-eval measured GIK's own disadvantage | **stands** — accepted by §11 |
| §5.4 index cost is small | **stands**, better supported: 4.0%, not 2% or 30.3% |
| §5.4 "stage ① is not load-bearing" | **withdrawn** — see §12.2 |
| §5.5 the "$115 vs copying" straw man | **stands** — undisputed |
| §5.6 both Herbie claims false | **stands** — reproduced and measured by §11 |
| §5.7 the 4.2x closes | **stands** — undisputed |
| §6.1–6.5 what survives | **stands**, plus **6.6** from §12.3 |
| §10 conclusion | **stands** — explicitly accepted by §11 |

Six of seven uncomfortable findings stand unchanged. One is split: its
conclusion survives, its framing is withdrawn. One finding is added that
strengthens the case *for* GIK.

### 12.5 What the whole exercise actually established

The audit, the rebuttal and this re-check agree on more than they dispute, and
the disagreements were all about magnitude, never direction. What both sides
now hold:

- **The throughput case for GIK is dead.** Herbie fetches the same bytes by the
  same mechanism, parallelises, and decodes at 42 ms where the repo claimed
  2,000. No document should claim otherwise.
- **GIK and Herbie are not the same category.** Herbie is a client: "give me
  this field now." GIK is a corpus-construction pipeline: "make 2023–2026 exist
  as one analysable object." The comparison was category-confused from the
  start, which is why it kept producing inflated numbers in both directions.
- **The outcome is the artifacts and the semantics inside them** — the uniform
  member axis, the per-level keys, the era splice, the chunking — not the
  byte-range streaming, which was never the differentiator.

### 12.6 The 200 hours are not a GIK result

> **Amended by §12.10.** The arithmetic below stands; the conclusion is
> over-stated. Read §12.10 before quoting this section.

The question "what would the 18 TB realization cost without the virtual
store?" is answerable from the measurements already in this document. Three
designs, 887 dates, 125,970 messages per date, 48 readers:

| design | index cost | per message | 887 dates at 48 readers | |
|---|---|---|---|---|
| **A** — one Herbie object per task | — | +173 ms construction | **+112 h of construction alone** | nobody would build this |
| **B** — one Herbie object per file, `.download()` per message | ~48 min | **838 ms** | **541.9 h** | works, costs **2.8x** |
| **C** — runtime `.idx` parse + own byte-range GET + eccodes-direct | **~10 min** | **240 ms** | **~192 h** | parity with the measured build |

**Design C is source-bound, exactly as the measured build was.** At 240 ms per
message, 48 readers deliver 200 msg/s against a source that sustained 161 —
so the ceiling is upstream, which is what `REALIZATION_COST.md` §2 concluded
independently (*"Adding workers past 48 does not help"*).

Design B is client-bound: 48 readers at 838 ms give only 57 msg/s. But that is
a concurrency problem, not a wall — **135 readers** at 838 ms saturate the same
161 msg/s and return to ~192 h at 2.8x the cluster. On Spot, 2.8x of $46 is
$129, which is the more damning number.

**The virtual store's contribution to the 192.4 hours is the index it
front-loaded: ~10 minutes.** The realization needed a message-level parallel
executor and byte offsets. It did not need the virtual store, and it did not
need Herbie, whose API is shaped for files rather than messages.

The 18 TB / 200 hours belongs to frisky and to the S3 ceiling. It is not
evidence for GIK and should not be presented as such.

### 12.7 Withdrawal of §5.2's throughput claim

§5.2 concludes: *"the realization was slower than it needed to be."* **That is
withdrawn.**

gribberish is flat at ~40 msg/s per *process* regardless of thread count
(§4.3). The cluster ran **12 worker processes** (6 VMs × 2 workers), giving a
decode ceiling of **~480 msg/s** against the **161 msg/s** actually achieved —
roughly **3x headroom**. The GIL serialisation was therefore not binding on
this cluster at this source rate, and cost the run nothing measurable.

What survives of §5.2 is the design finding, unchanged: gribberish was adopted
for a threading property it does not have, and eccodes-direct is both faster
(16 ms vs 22 ms) and actually threads. That matters for the next corpus. It did
not matter for this one.

§8.1's GIL test is correspondingly lower priority than the audit implied.

### 12.8 Addendum to §12.2 — the ergonomic argument, and where it breaks

A fair defence of stages ①–② is that they supply a *uniform routine* — open the
store with xarray, select, write — removing the need to hand-write byte-range
plumbing and task graphs. **This is true for consumption and false for
production**, and the repo's own history is the evidence.

Four commits, 2026-08-03:

```
f3fdbe9  Fix the hang: chunks={} built a 9M-task graph before any selection
a31f2f8  Add graph_size.py and DAG_METHOD.md: measure the graph before running
18feda8  Conclude: the Dask arrangement is the blocker, not AWS
e6bff28  Add a Frisky futures DAG: task count tracks the request, not the store
```

followed by `5d7d55c` *"Use icechunk fork/merge so workers write their own
chunks"*. The xarray-native path — `xr.open_zarr(store, chunks={})` — built a
nine-million-task graph before any selection and hung. The realization was
completed with a hand-written futures DAG, not with the store's own ergonomics.

**frisky is also not GIK's.** `icechunk-dask-frisky/deploy_workers.py:50` lists
`frisky>=0.3.0` as an external package; `frisky_daily_dag.py` is bespoke
integration code. The executor works on offsets, whatever produces them.

| | does the store offload it? |
|---|---|
| reading the corpus as a dataset | **yes** — decisively; this is the point of the exercise |
| byte-range mechanics | no — ~30 lines either way |
| the task graph | **no** — broke at 9M tasks; futures DAG written by hand |
| the Icechunk writer, fork/merge, block repair | no — custom |
| **the coordinate space** — era splice, member axis, per-level keys | **yes, and this is the actual answer** |

The strong form of the argument is therefore not about tooling: **GIK did not
offload writing byte ranges — it offloaded constructing a correct coordinate
space across three model eras.** The first is trivial; the second is the weeks
of domain work that §11.3 caught the audit pricing at zero.

### 12.9 What GIK is, and what it advances

*Revised after §12.10 (semantics as precondition), §12.12 (measured at 48
readers) and §13 (the Nairobi leg).*

Herbie is the state of the art in forecast-data **retrieval**, and this audit
confirms it on every axis it claims: it resolves a field from the archive's own
`.index` by byte range, decodes it correctly — 18 bit-exact comparisons here —
and from ECMWF's own network it moves the same bytes as fast as anything built
on top of it. **GIK uses the identical method. It is not a faster Herbie, and
no document should say that it is.**

Herbie's limitation is not speed. It is the **unit of delivery**: one GRIB
message — a global field — written to a local file. Because 92% of every
message falls outside the East Africa window, a regional query costs **5.9x
more bytes than the answer it returns**. Inside ECMWF's network that costs
nothing: the job is latency-bound and concurrency hides it, measured here at
**19.9 → 81.3 msg/s** from 6 to 48 readers. On a Nairobi retail link it is
fatal. At 20 Mbps the ceiling is **~3.2 messages per second regardless of how
many readers run**, because the wall is bandwidth rather than latency — and the
same training slice takes **31 days by message against 5 days by chunk**.

GIK is what closes that gap, in two layers worth naming separately. The first
is a **coordinate space**: that `type=cf` is the control member the index never
numbers; that 50r1's `enfo` stream carries no control at all; that the step axis
is 85 for 00z/12z and 49 for 06z/18z; that era boundaries fall per `(date, run)`
at 06z; that pressure levels need per-level keys or they silently collapse.
None of that is in the `.index` file, and Herbie's `class ifs` is a URL template
plus three date boundaries. The second is **chunking** — `(1, 1, 51, 321, 261)`,
all 51 members of a `(date, channel, step)` in one object — which turns an
ensemble query from **125,970 requests per date into 2,470**.

Published as a **76 GB index addressing 1.90 PB** at 24,869x leverage, and a
**17.7 TB regional store** readable anonymously with no GRIB decoder in the
reader's environment. The advance is from **retrieval to dataset construction**.
And the test of whether it was worth building is not a benchmark in Europe,
where Herbie and GIK are indistinguishable — it is whether a forecaster in
Nairobi can open it, which on these numbers is the only path that works.

### 12.10 Amendment to §12.6 — throughput is not feasibility

§12.6 concludes *"it did not need the virtual store."* **That is over-stated,
and is amended here.** The arithmetic in §12.6 is unaffected: Design C is
source-bound at ~192 h, the index costs ~10 minutes, and the 200 hours belong
to frisky and the S3 ceiling. What §12.6 gets wrong is the inference from a
*throughput* result to a *feasibility* one.

Three layers, which §12.2 separated correctly and §12.6 then re-merged:

| layer | needed at run time? | cost |
|---|---|---|
| the **published parquet artifact** | **no** — a runtime index parse substitutes | ~10 min |
| the **function** — index rows to a compact, broadcastable offset table | yes | ~30 lines; 22 MB/date |
| the **semantics** — the coordinate space | **yes, and not cheap** | weeks |

The semantics layer is what a DAG builder actually consumes, and none of it is
in the `.index` file or in Herbie:

- which `(param, levelist, number, step)` is channel `t850` at chunk
  coordinates `(time, step, number, lat, lon)`;
- that `type=cf` is the control and belongs at `number=0` — the index never
  numbers it (§12.3);
- that on **50r1 there is no `cf` in `enfo` at all**, so the control must come
  from the `oper/fc` stream;
- that the step axis is **85 for 00z/12z and 49 for 06z/18z**;
- that era boundaries fall **per (date, run) at 06z** (`c43cb66`), not at
  midnight;
- per-level keys, without which pressure-level channels silently collapse
  (`4ca1c21`).

**What Herbie supplies of this: none of it.** `src/herbie/models/ecmwf.py`
`class ifs` is a URL template plus three date boundaries (0p4→0p25 at
2024-02-01, the path-layout change at 2024-02-28 06:00, `scda`→`oper` at
2026-05-12) and six mirror URLs. That is real and useful, and it is not a
coordinate space.

So Design C is **not** "parse the index and go." It requires stage ①'s
derivation to have already happened, relocated from a parquet file into the DAG
builder. The relocation appears cheap only because the derivation was already
finished — and the repo's history shows it was not cheap when it was done:
the per-level collapse, the member axis, the era splice and the step-axis rule
each took weeks, and at least one produced a wrong answer that shipped.

**Corrected statement.** The realization did not need the *published artifact*
at run time; it did need the *coordinate space* stages ①–② encode. Design C
would have had to build that first, and would have arrived at stage ① by
another road. "Paralysed" overstates it — the code is writable — but
"the realization did not need the virtual store" understates it, and §12.6
should not be quoted without this amendment.

**A note on the pattern.** This is the third time in this record that the
artifact / function / semantics distinction has had to be applied — §12.2,
§12.8, and now here. Each time the audit under-applied it to a claim about
*time or cost*, and each time the correction went the same direction. Any
future throughput claim about this pipeline should be checked against these
three layers before it is written down.

### 12.11 §8.2 run, at reduced scale — 2026-09-23

§8.2 called for one date, both paths, same box and channels, phases separated.
Run here at reduced scale: **600 messages** (3 steps x 4 channels x 50 members)
against the EA-SWIO box, 6 threads, one 8-core EPYC. Script and output:
`audit_crosscheck/08_head_to_head_scoped.py`,
`audit_crosscheck/results/08_head_to_head_scoped.txt`.

Three paths, identical work, identical subset:

| path | wall | + index | fetch (med) | decode (med) | msg/s | per thread |
|---|---|---|---|---|---|---|
| **A** store-like — offsets known, gribberish | 37.5 s | **37.5 s** | 342.1 ms | 26.9 ms | 16.0 | 2.67 |
| **B** design C — runtime `.idx`, eccodes-direct | 35.7 s | **36.3 s** | 286.0 ms | 16.4 ms | **16.5** | **2.75** |
| **C** herbie-like — runtime `.idx`, tempfile + cfgrib | 43.2 s | **43.8 s** | 304.2 ms | 45.2 ms | 13.7 | 2.28 |

Index acquisition: **587 ms** for 3 files fetched in parallel — **1.6% of wall
clock**. 378 MB moved, identical on all three paths. 73 S3 retries absorbed.

**What it establishes.**

1. **Fetch dominates and decode does not.** ~300 ms per message against 16–45 ms
   of decode — **5.4% to 12.9%** of per-message cost. Every decoder argument in
   this repo was fought over a tenth of the budget.
2. **Design C is at parity with the store-like path, and marginally ahead** —
   0.97x, including the index it has to acquire and path A does not. The
   virtual store buys no throughput. §12.6's arithmetic is confirmed by
   measurement rather than extrapolation.
3. **The herbie-like decode path costs 17%, not 2.8x.** This **decomposes**
   §11's `06` figure (838 ms vs 240 ms) rather than contradicting it: that
   number measured Herbie's *API* — object construction plus
   `index_as_dataframe` per message. Path C isolates Herbie's *decode style*
   with a runtime index and no Herbie objects. Both are true. The 2.8x is
   object overhead; the 1.17x is the temp file and `cfgrib`. §12.6's Design B
   row should be read as "Herbie's API", not "Herbie's approach."

**What it does not establish, and §8.2 stays open.**

- 600 messages, not 125,970. 6 threads, not 48 readers.
- **73 retries in 600 fetches is a 12.2% retry rate**, against the realization's
  measured 5.85 *per million*. This box is being throttled far harder than the
  EWC cluster was, so the absolute rates here are not the cluster's. **The
  ratios are the trustworthy part; the absolutes are not.**
- Not run from Nairobi, so the RTT question in §8.2 remains entirely open.
- One date, one cycle, four channels.

At 48 readers the per-thread rates extrapolate to A 128 / B 132 / C 109 msg/s
against the realization's measured 161 — confirming this box is the slower
environment, and that the **ordering**, not the magnitude, is what transfers.

### 12.12 §8.2 at 48 readers, from EWC — 2026-09-23

The six worker VMs are down (all unreachable on :22), but **the box running
this audit is itself on the EWC network** — `192.168.1.129`, with routes to
`136.156.195.x`, ECMWF's range. So §12.11 and this run were both made from the
cluster's own network position. What changes here is concurrency: **48 readers
on one 8-core box** rather than 8 readers on each of six 4-vCPU VMs.
`audit_crosscheck/09_ewc_48readers.py`,
`audit_crosscheck/results/09_ewc_48readers.txt`.

**Concurrency sweep** (path B, eccodes-direct, 480 messages per point):

| readers | msg/s | MB/s | retries | per reader |
|---|---|---|---|---|
| 6 | 19.9 | 12.6 | 0 | **3.32** |
| 12 | 39.3 | 24.9 | 0 | **3.27** |
| 24 | 42.0 | 26.6 | 0 | 1.75 |
| 48 | **81.3** | **51.5** | 0 | 1.69 |
| 96 | 80.1 | 50.8 | 0 | 0.83 |

**Head-to-head at 48 readers** (600 messages):

| path | wall | msg/s | MB/s |
|---|---|---|---|
| A store-like (gribberish) | 17.6 s | **34.0** | 21.6 |
| B design C (eccodes-direct) | 8.3 s | **72.4** | 51.7 |
| C herbie-like (tempfile + cfgrib) | 32.6 s | **18.4** | 12.0 |

**1. `REALIZATION_COST.md`'s per-reader rate is independently confirmed.** At 6
and 12 readers this box sustains **3.32 and 3.27 msg/s per reader** against the
document's measured **3.35**. Same network, same source, three different runs,
one year apart in tooling. That figure is solid.

**2. §12.11's throttling caveat is withdrawn.** **Zero retries** across every
point here, including 96 readers. The 73 retries in §12.11 were a transient
episode, not this box's network position.

**3. The decoder becomes binding at high per-process concurrency.** At 6 threads
(§12.11) the three paths were within 17%. At 48 they are not: gribberish caps at
**34.0 msg/s** — almost exactly the **~40 msg/s flat ceiling** measured in §4.3
— while eccodes-direct reaches **72.4**. The GIL finding is not academic; it
bites precisely when you scale readers inside a process.

**4. §12.7 still stands, but narrowly.** The realization ran 2 processes × 4
threads per VM, so each process carried ~13.4 msg/s against gribberish's ~40
ceiling — not binding, as §12.7 concluded. What this run shows is *why*
`a5de8cc` ("two worker processes per VM") was necessary: processes escape the
GIL, threads do not. The decoder choice did not cost this run; it constrained
its shape.

**5. A finding against `REALIZATION_COST.md` §2.** That section states *"Adding
workers past 48 does not help."* The measured per-VM rate — 161 ÷ 6 = **26.8
msg/s** — is exactly **8 readers × 3.35 msg/s**, i.e. the cluster was bound by
*how many readers were running*, not by the source. One box here reaches **81
msg/s** with 48 readers and shows no source throttling at 96. If six boxes
sustained that, the corpus would take **~64 h rather than 192 h**.

This is a hypothesis, not a result: this box is 8-core where the workers were
4-vCPU, and six boxes × 48 readers = 288 concurrent against `ecmwf-forecasts` is
untested — the aggregate may throttle where one box does not. **But it is cheap
to test and worth £0 to try, and if it holds the realization was ~3x slower than
necessary.** Note also that gribberish would become the cap at that scale
(~34–40 msg/s per process), so the scale-up and the decoder swap go together.

**What §8.2 still lacks:** the Nairobi leg. Every number in this document comes
from inside ECMWF's network, ~10 ms from `eu-central-1`. The RTT question that
started this conversation is still unmeasured.

---

## 13. The Nairobi leg — 2026-09-23

§8.2 asked for the benchmark to be run from Nairobi as well as Europe. That has
**not** been done; nothing in this document is measured from an African link.
What follows is arithmetic on the measured corpus sizes against stated
bandwidths, and it is labelled as such.

### 13.1 A units correction that changes the answer 8x

Airtel and Safaricom advertise **Mbps** (megabits per second), not MB/s. A
promoted "10 Mbps" home plan delivers **1.25 MB/s**. "20 MB/s" is a 160 Mbps
business link — a different product at a different price. Both readings are
priced below; the distinction matters more than any other number here.

| link | 88 TB — realize locally | 17.7 TB — pull the finished store |
|---|---|---|
| 10 Mbps (1.25 MB/s) | 815 days | 164 days |
| **20 Mbps (2.5 MB/s)** | **407 days** | **82 days** |
| 100 Mbps (12.5 MB/s) | 81 days | 16 days |
| 20 **MB/s** (160 Mbps) | 51 days | 10 days |
| *EWC, measured §12.12* | *51.5 MB/s* | — |

At **100% utilisation, 24/7**. On a contended retail link assume 50–60%, so
multiply by ~1.8: at 20 Mbps, realizing locally is **~1.9 years** and pulling
the store is **~4.5 months**.

### 13.2 The bottleneck changes character, not just magnitude

At EWC the job is **latency-bound**, which is why §12.12's sweep scaled from
19.9 to 81.3 msg/s by adding readers — concurrency hides round trips.

On a 2.5 MB/s link, 789 KB messages give an aggregate ceiling of **~3.2
messages per second regardless of reader count**. The wall is bandwidth, and
concurrency cannot hide it. **Every tuning lever in this document — 48 readers,
thread counts, eccodes versus gribberish — becomes irrelevant.** An infinitely
fast decoder changes nothing.

### 13.3 Where the 5.9x amplification stops being a footnote

A realistic cGAN training slice — 10 channels × 1,264 dates × 8 steps:

| access path | bytes | at 2.5 MB/s, 60% duty |
|---|---|---|
| **realized store** — 101,120 chunks × 6.8 MB | **688 GB** | **~5 days** |
| virtual store or Herbie — the same query as messages (× 51 members) | 4.07 TB | **~31 days** |

Same query, same answer, **6x less data**. In Europe that ratio is a cost
optimisation. At Nairobi bandwidth it is the difference between a working week
and a month, and it is the clearest justification for the realization anywhere
in this document.

**Note what this does to the virtual store.** For all its 24,869x leverage, it
resolves to per-message reads of global fields, which puts a Nairobi consumer
on the 31-day row. The virtual store is the right artifact for addressing the
archive; it is the wrong artifact for consuming it over a constrained link.

### 13.4 Recommendations

1. **Never realize in Nairobi.** At any plausible retail bandwidth this is
   arithmetic, not preference.
2. **Do not pull the full 17.7 TB either** — ~4.5 months at 20 Mbps.
3. **Read lazily from source.coop**, fetching only the chunks a training run
   touches. Resumability matters more than throughput here: a link that drops
   makes an 82-day continuous transfer impossible, while chunk-level reads
   resume for free.
4. **If a local copy is genuinely needed, ship a disk.** At 20 Mbps a courier
   beats the link by weeks. It is the honest comparison and nobody makes it.

### 13.5 What would replace this section with a measurement

A single `iperf3` to `eu-central-1`, or a timed 1 GB `curl` from the actual
ICPAC machine, plus `ping` for the RTT. Ten minutes of work, and it replaces
every figure above with a measured one. **Until then §13 is arithmetic on
assumed bandwidth, and should be quoted as such.**
