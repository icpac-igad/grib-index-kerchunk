# 06z / 12z / 18z backfill — master plan & resource assessment

**Goal:** replicate the completed 00z corpus (1,256 dates × 3 eras) at 06z, 12z and 18z,
each driven by a **separate parallel Claude Code session**, Herbie-verified per era per
cycle before the full wave, with the whole backfill inside **2 hours**.

**Prerequisite reading:** [`../ECMWF_00Z_BACKFILL_SUMMARY.md`](../ECMWF_00Z_BACKFILL_SUMMARY.md)

| Cycle | Plan | Owner session | Blocking prerequisite |
|-------|------|---------------|-----------------------|
| 06z | [`06z/PLAN.md`](06z/PLAN.md) | session A | ✅ none — see §1a |
| 12z | [`12z/PLAN.md`](12z/PLAN.md) | session B | ✅ none |
| 18z | [`18z/PLAN.md`](18z/PLAN.md) | session C | ✅ none |

> ### ⚠️ Correction (2026-07-29) — supersedes the first version of this plan
>
> The first draft said 06z and 18z were **blocked** on three new `-s144` runtime
> images built from truncated templates. **That was wrong.** Inspecting an actual
> template artifact (§1a) shows the templates are **cycle-agnostic**: they carry
> only the per-variable Zarr *schema*, never step data. All four cycles run on the
> existing `:0p4` / `:49r1` / `:50r1` images.
>
> **No template work, no Cloud Build, no new images.** Phase 0 shrinks from
> hours/days to minutes. All three sessions can start immediately.

---

## 1. 06z and 18z are short runs

Probed directly against `s3://ecmwf-forecasts` (49r1 era, `20250115`, `enfo`):

| Cycle | +0h | +144h | +150h | +240h | +360h | Reach |
|-------|-----|-------|-------|-------|-------|-------|
| **00z** | 200 | 200 | 200 | 200 | 200 | **+360h** |
| **06z** | 200 | 200 | **404** | **404** | **404** | **+144h** |
| **12z** | 200 | 200 | 200 | 200 | 200 | **+360h** |
| **18z** | 200 | 200 | **404** | **404** | **404** | **+144h** |

Step cadence within 0–144h is 3-hourly for 06z (`+3h`, `+6h` both 200) — identical to
the first half of the 00z axis. All three cycles exist for all three eras, so coverage
is **1,256 dates at 06z** and **1,255 at 12z/18z** (see §1b for why they differ).

## 1b. ⚠️ Era cutovers are CYCLE-granular, not date-granular

Found while verifying the completed 06z run (2026-07-31). The era table treats both
cutovers as whole-day boundaries. **They are not** — the switch happens partway
through the day, so a cycle later than 00z can already be in the next era:

| Date | 00z | 06z | Evidence |
|------|-----|-----|----------|
| `20240228` | still **0p4** | already **49r1** | `0p4-beta/enfo` 06z → **404**; `ifs/0p25/enfo` 06z → **200**. The 0.4° stream ends after 20240228 00z. |
| `20260512` | still **49r1** | already **50r1** | 49r1 config wrote only **50/51** (control absent from `enfo`); `oper` control for 06z → **200**. |

Run under the era the table implies, `20240228` wrote **nothing** (0.4° paths 404) and
`20260512` wrote **50/51** (no control). Both fixed by running the single date under
the *next* era — each then produced 51/51.

> **Confirmed for all three cycles (2026-08-05).** 12z and 18z hit the identical two
> dates and were fixed the same way. A further cycle-dependence: the unpublished
> 0.4° window is `20230427..20230502` at 00z/06z but starts a day earlier
> (`20230426`, S3-404) at 12z/18z — so 00z/06z reach **1,256** dates and 12z/18z
> reach **1,255**. Fix the cutover dates with:
> ```bash
> bash fix_cycle_boundary_dates.sh <06|12|18>
> ```
> The 00z corpus never exposed this because at 00z both cutovers really do fall where
> the table says.

## 1a. Why this needs no new templates — what a template actually contains

Downloaded `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` (14 MB) and inspected it:

- **4,335 parquets = 51 members × 85 steps**, named `…-control-rt000.par`,
  `rt003`, `rt006` … `rt360` — `rtNNN` is the lead-time hour.
- **`merge_with_local_template()` reads only `rt000`.** The other 84 are never opened.
- **`rt000` holds 304 rows of pure Zarr metadata** — `.zarray`, `.zattrs`,
  `zarr_consolidated_format`, `metadata`. **Zero data references, zero `step_` keys.**
- Every `.zarray` is `"shape": [1, 721, 1440]` — **a single step**, not 85. The step
  dimension lives in the `step_NNN/` key *prefixes*, which come entirely from the S3
  index scan, not from the template.

So the merge is `template schema + whatever refs S3 actually yielded`:

```python
merged_refs = template_refs.copy()        # per-variable schema, unprefixed
for key, value in index_refs.items():
    merged_refs[key] = value              # step_000/…, step_003/… — from S3
```

**The number of steps is determined solely by what exists on S3.** A 06z run finds 49
steps and writes 49; nothing in the template forces 85. And because `rt000` carries no
data refs, a short cycle cannot silently inherit the template date's values — the
failure mode that motivated the original "truncate the templates" plan does not exist.

The hour loop already tolerates absence (`if not idx_entries: continue`), so 06z/18z
would have worked untouched — just wastefully.

### The one change that was needed

Asking for the 36 non-existent steps costs **36 × 51 = 1,836 futile ranged GETs per
date** against a bucket that demonstrably throttles (§6). So the hour list is now
scoped to the cycle:

```python
def forecast_hours_for_run(run: str) -> List[int]:
    return ALL_FORECAST_HOURS if str(run).zfill(2) in ('00', '12') else HOURS_3H
```

Correct by construction from `--run`, with no new flag to forget.

---

## 2. Do we have the Cloud Run resources? — the honest arithmetic

### Current capacity

Three services, one per era, each `maxScale = 35`, `containerConcurrency = 1`, 2 GB,
`europe-west3`. Ceiling today: **105 concurrent workers**.

**Service identity = runtime image + memory.** So:

- 12z reuses the 00z images at 2048 MB → it lands on the **same three services** the
  00z daily ops use. Concurrent 12z work and any 00z rolling job contend for one
  35-slot pool.
- 06z and 18z would both use the `-s144` images at 2048 MB → **they would collide
  with each other**. Force separation by deploying 18z at a different memory
  (e.g. 2560 MB), which yields a distinct service name.

### Work to be done

| | Steps/date | Rel. cost | Dates | Est. min/date |
|---|---|---|---|---|
| 12z | ~85 (0–360h) | 1.00 | 1,256 | ~12 |
| 06z | ~49 (0–144h) | ~0.58 | 1,256 | ~7 |
| 18z | ~49 (0–144h) | ~0.58 | 1,256 | ~7 |
| **Total** | | | **3,768** | **≈ 546 worker-hours** |

(Per-date minutes derived from §6 of the 00z summary: a 35-wide wave takes 8–20 min.)

### The floor

546 worker-hours ÷ 2 h wall = **273 workers at 100 % utilisation** — an arithmetic
floor no scheduling trick avoids. At a realistic ~70 % utilisation (cold starts,
retries, month-boundary barriers, the hang-at-exit tax) the requirement is:

> ### ≈ 385 concurrent instances — ~770 vCPU and ~770 GB RAM in `europe-west3`

Each worker is **2 vCPU + 2 GB** (`runtime_cpu: 2`, `runtime_memory: 2048`), so the
CPU ask is twice the instance count.

Against a present ceiling of **105 instances / 210 vCPU**. That is a **3.7× shortfall**.

### Verdict

**2 hours is not achievable as currently configured.** Three things must all hold:

1. **`maxScale` raised 35 → ~130** on each of the (now ~6) services. Self-service.
2. **Regional quota** for ~385 concurrent instances / **~770 vCPU** / ~770 GB in
   `europe-west3`. ⚠️ Not verifiable from here — the deployer SA is denied
   `serviceusage` quota reads. **Check with an account that can, and file any
   increase now**: Google quota increases take hours to days, so this is the
   long-pole item.
3. **ECMWF S3 must sustain ~385 concurrent byte-range readers.** ⚠️ **Highest risk,
   and the least controllable.** While probing for this plan, the public bucket
   returned **HTTP 503 throttling at a handful of requests per minute**. 385 workers
   each issuing thousands of ranged GETs is a materially different load. There is no
   published rate limit to design against, and no way to raise it.

**Realistic expectation at the current 105-wide ceiling: ~5–7 hours wall** for all
three cycles run in parallel (12z is the long pole at ~7 h).

**Recommendation:** treat 2 hours as the stretch goal contingent on (1)+(2)+(3), and
plan operationally for a **half-day window**. If the deadline is hard, the honest
lever is scope — e.g. ship 12z first (no new images, highest value: full 360h axis),
then 06z/18z.

---

## 3. Parallelisation design

All three cycles now use the **same three images**, so without separation all three
sessions would pile onto the same three 35-slot pools — and contend with 00z daily ops
too. Since a lithops service name is `image + memory + lithops version`, giving each
session a distinct `runtime_memory` yields an independent pool:

```
session A ── 06z ──► :0p4 / :49r1 / :50r1  @ 2048 MB ─┐
session B ── 12z ──► :0p4 / :49r1 / :50r1  @ 2560 MB ─┼─► GCS v20260623_run_par_ecmwf
session C ── 18z ──► :0p4 / :49r1 / :50r1  @ 3072 MB ─┘        (…/{date}/{HH}z/)
     (00z daily ops stays on 2048 MB — so A must coordinate with it)
```

This is the **whole** of Phase 0 now: six `lithops runtime deploy` calls (3 images ×
2 non-default memories). No image build — deploy against the existing images takes
about a minute each. The extra memory is incidental; separation is the point.

**Safe to parallelise** — the three sessions write to disjoint GCS prefixes
(`…/{date}/06z/`, `/12z/`, `/18z/`), so there is no write contention.

### Three rules the sessions must not break

1. **Never share `lithops_config.yaml`.** Each session flips the `runtime:` tag when
   it changes era; three sessions editing one file will corrupt each other's waves.
   **Each session works in its own checkout with its own copy** — see per-cycle plans.
2. **One Icechunk writer.** `backfill_all_eras.py` writes a single store; concurrent
   writers conflict. Icechunk ingest is a **serial step after all three cycles land**.
3. **Stagger the starts by ~5 min.** Three simultaneous cold-start storms against one
   region reproduce the HTTP 500 cold-start race that `max_workers 35` was chosen to
   avoid.

---

## 4. Phase plan

### Phase 0 — pre-flight (do once, before any session starts)

- [x] ~~Build 3 short templates + 3 images~~ — **not required**, see §1a.
- [x] `forecast_hours_for_run()` scopes the hour list per cycle.
- [ ] Confirm regional quota for the target concurrency; **file the increase now**.
- [ ] Deploy the 3 existing images at **2560 MB** (12z) and **3072 MB** (18z),
      under both lithops 3.6.3 and 3.6.4 — six `runtime deploy` calls, no build.
- [ ] Raise `maxScale` on all services to the agreed value.

### Starting the 12z and 18z sessions

**Run them as separate Claude Code sessions, in parallel — start both now.** Phase 0 +
Phase 1 are low-load (six `runtime deploy` calls, then 3 single-date builds at 4
workers each), so three sessions doing them concurrently is ~12 workers total. The
heavy part is Phase 2, which is gated on all three Herbie gates passing anyway.

What makes parallel safe here: each session gets **its own git worktree**, so each has
its own `lithops_config.yaml` and can flip the `runtime:` tag without touching the
others; each writes to a **disjoint** GCS prefix (`…/{date}/{HH}z/`); and each deploys
at a **distinct memory**, so the Cloud Run pools don't overlap.

Bootstrap prompt for a new session (substitute `12`/`2560` or `18`/`3072`):

> Read `devops/lithops_cr_ecmwf_gik/cycles/README.md` and `cycles/12z/PLAN.md`, plus
> `ECMWF_00Z_BACKFILL_SUMMARY.md` §4 (the two-switch era rule) and §5 (the gotchas).
> Then: create a worktree at `/tmp/gik-12z`, set `runtime_memory: 2560`, deploy the
> three existing images at that memory under both lithops 3.6.3 and 3.6.4, and run
> `bash run_cycle_herbie_gate.sh 12`. Report the correlations and stop — do not start
> the full waves.

The three rules below are what that prompt is protecting; state them explicitly if a
session looks like it might drift.

### Phase 1 — Herbie verification gate (~45 min, all sessions in parallel)

Use **`run_cycle_herbie_gate.sh <cycle>`** — it builds one validation date per era
(sequentially, flipping the tag) and then runs `compare_gik_herbie_pressure.py`, the
same tool the 00z corpus was validated with. It reuses the 00z eval's date picks
(`0p4:20230318`, `49r1:20240327`, `50r1:20260621`) so results are directly comparable
across cycles, and it checks **T+0 and the last step of the axis** — a T+0-only check
passes even when the axis is wrong.

**9 validations: 3 cycles × 3 eras, one date each.** Nothing else runs until these pass.

Reuse `ecmwf/compare_gik_herbie_pressure.py` exactly as the 00z work did. Pass criteria,
matching the 00z outcome:

| Era | Threshold |
|-----|-----------|
| 49r1 / 50r1 | **r ≥ 0.9999** |
| 0p4 | **r ≥ 0.9997** (grid-reindex residual, RMSE ~0.02–0.05 K) |

A failure here almost certainly means a template/axis mismatch — **stop, do not
release the wave.** This gate is what caught era mismatches cheaply in the 00z work.

### Phase 2 — full waves (the ~85 min budget)

Each session walks its eras oldest → newest, one month per wave, per its plan.

### Phase 3 — verification & downstream (serial)

- [ ] Per-date GCS file counts = 51/51 across all three cycles (**never** trust exit codes — see gotcha #1).
- [ ] Cross-era, cross-cycle Herbie spot check.
- [ ] HF mirror via `mirror_gcs_to_hf_v2.py`.
- [ ] Icechunk ingest — **single writer, after everything else**.

---

## 5. Common environment (every session, every era)

```bash
export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf   # NOT the legacy default
export AWS_NO_SIGN_REQUEST=YES
gcloud auth activate-service-account \
  --key-file=service_account/ecmwf-lithops-deployer-key.json --project=e4drr-crafd
```

Per-era env and the `--run {06,12,18}` flag are set in each cycle's plan.
`run_backfill_00z.sh` already takes `--era`; it needs a `--run` passthrough — noted
in each plan as a required small edit.

---

## 6. Risk register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **ECMWF S3 throttling at high concurrency** | 🔴 Could make 2 h impossible regardless of GCP capacity — 503s already seen at low rates | Ramp concurrency gradually; monitor 503 rate from wave 1; be ready to fall back to ~105-wide |
| Regional quota below requirement | 🔴 Hard stop | Verify + file increase in Phase 0 (long lead time) |
| Short template wrong | 🔴 Silent zero-file waves | Herbie gate (Phase 1) catches it before the wave |
| Sessions sharing `lithops_config.yaml` | 🟠 Cross-contaminated eras | Separate checkouts, rule 1 |
| Hang-at-exit read as failure | 🟠 Unnecessary re-runs | `timeout` wrapper; exit 124 = OK; verify via GCS counts |
| Icechunk concurrent writers | 🟠 Store corruption | Serial ingest, Phase 3 |
| Legacy GCS prefix | 🟠 Data in the wrong catalog | `GCS_PARQUET_PREFIX` exported in every session |

---

## 7. Cost

3,768 date-runs × ~$0.026 ≈ **$98** in Cloud Run, plus GCS storage (~21 GB per cycle,
~63 GB total) and egress for the HF mirror.
