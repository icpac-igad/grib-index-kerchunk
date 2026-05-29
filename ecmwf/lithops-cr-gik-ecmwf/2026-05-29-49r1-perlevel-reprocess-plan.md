# ECMWF 49r1 Per-Level-Keys Reprocess Plan

**Date:** 2026-05-29
**Branch:** `ecmwf-50r1-template`
**Sibling docs:**
  - [`2026-05-17-50r1-template-rebuild-plan.md`](2026-05-17-50r1-template-rebuild-plan.md) — 50r1 template rebuild (already authored, gated on Coiled run)
  - [`2026-05-29-per-level-keys-fix-and-template-implications.md`](2026-05-29-per-level-keys-fix-and-template-implications.md) — code fix + why template rebuild is mandatory

## 1. Scope

**In scope:** the 49r1 cycle, from **2024-03-01 onwards** (start of 49r1
on ECMWF Open Data) through **2026-05-12 00z** (last 49r run before the
50r1 cutover at 2026-05-12 06z).

**Out of scope:** the 28r1 cycle (pre-2024-03 archive). Treated separately
if/when needed.

**Goal:** rebuild the 49r1 template tar.gz against the **fixed**
preprocessing in `4ca1c21` so its `rt000.par` files carry a `level`
dimension for pl variables, then re-bake every 49r1-era HF parquet so
downstream consumers (cGAN training, EA flood-risk pipeline) get
physically coherent pl-channel data for MAM 2024, MAM 2025, and the
2024-03 → 2026-05-12 backfill window.

## 2. Verified facts

| Fact | Evidence |
|---|---|
| 49r1 data is retained on `s3://ecmwf-forecasts` from 2024-03 onwards | User-confirmed retention policy (2026-05-29) |
| The existing 49r1 template uses reference date **2024-05-29** | `CLAUDE.md`, `gik-fmrc-v2ecmwf_fmrc.tar.gz` on HF |
| 49r1 enfo/ef carried 51 members in a single file: 1 control (`number=0`) + 50 perturbed (`number=1..50`) | 49r1 `.index` JSON-line schema |
| 49r1 pl set: **13 levels** (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000) — no 10 hPa | 49r1 `.index` `levelist` distinct values |
| 49r1 msgs/file: ~8211 (vs 50r1's 8500 — the extra +289 in 50r1 is 50× new 10 hPa + reshuffled sfc/sol counts) | 49r1 `.index` line count |
| 49r1 cutover: last 49r run is 2026-05-12 **00z**; first 50r1 run is 2026-05-12 **06z** | Pinned in `2026-05-17-50r1-template-investigation.md` §1 |

## 3. 49r1 vs 50r1 — what changes in the chain

| Component | 49r1 | 50r1 (already done on this branch) |
|---|---|---|
| Streams scanned | `enfo/ef` only (51 members in one file) | `enfo/ef` (50 perturbed) **+** `oper/fc` (control) — dual-stream |
| Members | 51 (`-1` control + 1..50 perturbed; `-1` is the realigner's normalised form of `number=0`) | 51 = 50 + 1, but split across two source files |
| pl levels | 13 (no 10 hPa) | 14 (50r1 added 10 hPa) |
| Coiled preprocessing task count | 85 (one per forecast hour) | 170 (85 enfo + 85 oper) |
| Source URL pattern | `s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/enfo/{ts}-{h}h-enfo-ef.grib2` | enfo + oper paths |
| Output dump naming (post-fix) | `e_sg_mdt_{date}_enfo_{h}h.parquet` (stream-tagged for forward compat) | `e_sg_mdt_{date}_{enfo\|oper}_{h}h.parquet` |
| `pressure_levels` allowlist | 13 entries — but the 50r1 allowlist (14 entries incl `'10'`) is a strict superset that just won't match anything in 49r1 data → **no code change needed**. | 14 entries incl `'10'` |
| `ens_numbers` passed to `ecmwf_duplicate_dict_ens_mem` | `[-1, 1, 2, ..., 50]` (51 members) | `[1, 2, ..., 50]` (50 perturbed only) |
| `member_numbers` passed to `organize_ensemble_tree` | same 51-member list | 50-member list |
| Stage 2 `.index` URLs | one `enfo/ef` URL per hour | two URLs per hour (enfo + oper) |
| `fixed_ensemble_grib_tree` per-level fix from `4ca1c21` | **applies unchanged** (the typeOfLevel=isobaricInhPa branch is cycle-agnostic) | already applies |

The headline finding: **the 49r1 rebuild reuses essentially every piece
already on `ecmwf-50r1-template`**. The 50r1 work parameterised
`ecmwf_duplicate_dict_ens_mem(ens_numbers=...)` and
`organize_ensemble_tree(member_numbers=...)` in commit `c8d71e7` — those
parameters take the 49r1-shape inputs natively. The only NEW code is a
49r1-specific Coiled driver (single-stream, 51 members, 85 tasks).

## 4. Build & deploy steps  (corrected, validated chain)

Status legend: ✅ done & validated · ▶ run-later (gated/expensive) · ✋ user-only · 🆕 still to build

**🆕 Step 0 — write `ecmwf_49r1_coiled_preprocessing.py`.** Single-stream
cousin of `ecmwf_50r1_coiled_preprocessing.py`. ~85 lines. Same Coiled config
(`software=gik-coiled-v6`, `workspace=gcp-sewaa-nka`, `n2-standard-2`,
`us-east1`, `arm=False`, `idle_timeout=30m`, `cluster.adapt(1,9)`). Task
plan builder yields 85 enfo-only tasks for one date/run. `--dry-run`
prints the plan with no cost. Cheap to write (1 hour) and validate
(`uv run` + dry-run on date `20240529`).

**▶✋ Step 1 — Coiled preprocessing (85 tasks, ~1 h, PAID).** User-run.

```bash
cd ecmwf/dev-test
# free preview:
python3 ecmwf_49r1_coiled_preprocessing.py --date 20240529 --run 00 --dry-run
# real preprocessing run (~1 h, paid):
python3 ecmwf_49r1_coiled_preprocessing.py --date 20240529 --run 00 \
    --software gik-coiled-v6 --workspace gcp-sewaa-nka --max-workers 9
# -> GCS gs://gik-ecmwf-aws-tf/fmrc/scan_grib20240529/
#    e_sg_mdt_20240529_enfo_{h}h.parquet   (85 dumps)
```

Wall-clock is ~half the 50r1 preprocessing because there's only one stream.

**▶ Step 2 — per-member split.** Use the existing realigner — it
already handles the legacy-format case via its `parse_filename`
fallback (legacy `e_sg_mdt_{date}_{h}h.parquet` → `stream='enfo'`).
For the new stream-tagged 49r1 dumps it uses the 50r1 path. No code
change needed.

```bash
cd ecmwf/dev-test
python3 ecmwf_par_to_ensemble_members.py \
    --input-dir <dir-with-85-dumps> --output-dir ./ensemble_members_49r1 \
    --run 00
# enfo dumps -> control (-1) + ens01..ens50  (51 members)
# into {H}h/{m}.par
```

**GATE A (structural, no scan_grib, ~free):** open one `control/rt000.par`
and confirm:
- `u/instant/isobaricInhPa/.zarray` has a `level` axis of size **13**.
- `level` coordinate `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` (no 10 hPa).
- 51 members present incl. `-1`.
- sfc/sol shapes unchanged from the legacy template.
If gate A fails, do NOT proceed — diagnose first.

**▶ Step 3 — package & rename to the template layout.**

```
gik-fmrc/v2ecmwf_fmrc/{ens_control|ens_NN}/ecmwf-2024052900-{m}-rt{hhh}.par
tar czf gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz gik-fmrc/
```

The `-perlevel` suffix is intentional — distinguishes from the legacy
buggy `gik-fmrc-v2ecmwf_fmrc.tar.gz` so consumers can opt-in cleanly.

**▶✋ Step 4 — upload** to HF
`E4DRR/grib-index-kerchunk-templates`. New artifact name; do NOT
overwrite the legacy `gik-fmrc-v2ecmwf_fmrc.tar.gz` (keeps existing
deployed runtimes working until they cut over). Needs HF write token.

**▶ Step 5 — build a 49r1 Cloud Run runtime image.** Branch off a tag
of the current code and wire `run_lithops_ecmwf.py`:
- `REFERENCE_DATE = '20240529'` (49r1 template ref)
- `ECMWF_TEMPLATE_PATH = '/gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz'` (or the HF URL)
- Stage 1 member list = `['ens_control'] + [f'ens_{i:02d}' for i in range(1, 51)]`
  (51 members, same as the pre-50r1 line we restored conceptually for
  the dual-template strategy)
- Stage 2 reads `.index` from `enfo/ef` only (no `oper/fc` lookup).
- Per-level key emission is already in `ecmwf_index_processor.py` from
  `4ca1c21` — no further change.

Build via `cloudbuild.yaml`, tag the image distinctly
(`gik-cr-ecmwf-49r1-perlevel:YYYYMMDD`), deploy with `lithops runtime
deploy`. See `SETUP_NEW_MACHINE.md §5`.

**▶ Step 6 — re-bake HF parquets.** Run the gap-backfill driver
(`run_lithops_ecmwf.py` with the 49r1 runtime) over **every** 49r1
date in scope: 2024-03-01 → 2026-05-12 00z. ~800 dates × 4 runs ≈ 3200
backfill tasks. Use the **cleanup-hang watchdog**
([[ecmwf-lithops-cleanup-hang]]).

Order of attack (smallest-blast-radius first):
1. **MAM 2026** (2026-03-01 → 2026-05-12 00z) — unblocks the cGAN
   training that motivated this whole effort. ~73 dates.
2. **MAM 2025 + MAM 2024** — supports the multi-year training pass.
3. **Everything else** — rolling-forward only as needed.

**▶ Step 7 — re-mirror to HF**
`E4DRR/gik-ecmwf-par` (`upload_parquets_to_hf.py`). Needs HF token.
Overwrites existing-but-broken pl-channel parquets; sfc/sol keys are
identical so consumers that only used those see no change.

**▶ Step 8 — re-deploy the 50r1 runtime** so ongoing production stays
on the 50r1 image once the 49r1 re-bake completes. See §5 below for
why this matters.

## 5. Dual-template runtime strategy

Two production paths exist:

- **Historical backfill (49r1)** — `2024-03-01 → 2026-05-12 00z`. Uses
  the 49r1 Cloud Run image (Step 5 above).
- **Ongoing operations (50r1)** — `2026-05-12 06z → present`. Uses the
  50r1 Cloud Run image (the one built per the 50r1 plan).

Two distinct **images**, one **tag per cycle**. Switch at deploy time
by changing the Cloud Run service revision. No runtime logic needs to
choose between templates per-date — that would couple two cycles into
one image and complicate testing. The 50r1 plan §2.3 already
established this pattern.

The re-bake (Step 6) deploys the 49r1 image first, runs through
2024-03 → 2026-05-12 00z, then Step 8 re-deploys the 50r1 image for
ongoing operations. Total Cloud Run downtime during the cutover: a
few seconds per revision swap.

## 6. Cheap dry-run gates (do BEFORE the paid preprocessing)

Two gates that cost zero S3-scan time and can be run today on the
existing 50r1 dumps (which were produced by the FIXED scan_grib code
because we already ran the per-level fix into `c8d71e7`+`4ca1c21`):

**Gate B1 — `fixed_ensemble_grib_tree` aggregator structure check.**
Feed any existing 50r1 dump through the fixed aggregator and confirm:
- pl variables produce 14 chunk slots per (var, step, member).
- The `level` coord is correctly serialised.
- sfc/sol slots count is unchanged from the legacy template.

This is the 50r1 equivalent of GATE A above; if it passes for 50r1, it
will pass for 49r1 (same code path, just one less level value).

**Gate B2 — the 49r1 Coiled driver dry-run.** Once Step 0 is done,
`python3 ecmwf_49r1_coiled_preprocessing.py --date 20240529 --run 00 --dry-run`
should print:

```
49r1 preprocessing plan: date=20240529 run=00z  85 tasks (enfo only)
GCS dest: gs://gik-ecmwf-aws-tf/fmrc/scan_grib20240529/
          e_sg_mdt_20240529_enfo_<h>h.parquet
  [enfo]  0h  s3://ecmwf-forecasts/20240529/00z/ifs/0p25/enfo/20240529000000-0h-enfo-ef.grib2
  [enfo]  3h  s3://ecmwf-forecasts/20240529/00z/ifs/0p25/enfo/20240529000000-3h-enfo-ef.grib2
  ...
```

Both gates together exhaust the correctness questions that can be
answered without spending money. If both pass, the only risks left in
Step 1 (the paid preprocessing) are operational — Coiled quotas, S3 weather,
worker eviction — not algorithm correctness.

## 7. Risks & open items

- **Bulk re-bake cost** (Step 6). ~800 49r1 dates × 4 runs. Each backfill
  was historically a Lithops/Cloud Run run of a few minutes; at scale
  this is the biggest cost line. Mitigations: prioritise MAM 2026 first
  (cGAN need), throttle per day to avoid runaway, monitor cleanup-hang
  watchdog.
- **HF storage churn.** Overwriting existing parquets on
  `E4DRR/gik-ecmwf-par` will rev the dataset history but keep total
  bytes ~stable. No write-token / quota issues anticipated; flag if
  any.
- **Consumers of the legacy 49r1 template.** Anyone pinned to the old
  `gik-fmrc-v2ecmwf_fmrc.tar.gz` will keep the old (broken) behaviour
  — which is correct, because their key shape is the broken one. The
  new artifact has the `-perlevel` suffix so opt-in is explicit.
- **28r1 (pre-2024-03).** Out of scope per the 2026-05-29 decision. If
  pre-2024-03 pl-channel data is needed later, a parallel plan would
  apply.
- **What if 49r1 Coiled driver scaffold goes wrong.** The 49r1
  preprocessing is single-stream, so the most likely error is using
  the 50r1 dual-stream driver by accident on a 49r1 date — which would
  404 on every `oper/fc` URL because 49r1 didn't publish that stream.
  Hard to silently corrupt; easy to diagnose.

## 8. What this plan does NOT do

- **Does not** modify any code that is already on `ecmwf-50r1-template`.
  The 4ca1c21 per-level fix and the c8d71e7 parameterisation are
  exactly the pieces the 49r1 rebuild needs.
- **Does not** require a second copy of `utils_ecmwf_step1_scangrib.py`.
  Both copies (production + `dev-test/`) are byte-identical and carry
  the per-level fix.
- **Does not** introduce a dual-template runtime selector. Cycle
  selection happens at deploy time (Cloud Run revision), not runtime.

## 9. Summary

- **Code is ready** for 49r1 reprocessing on `ecmwf-50r1-template@4ca1c21`.
- **One new artifact to build**: `ecmwf_49r1_coiled_preprocessing.py` (~85
  lines, mirrors the 50r1 driver but single-stream / 51-member /
  enfo-only).
- **Same Coiled config, same workspace, same image stack** as the 50r1
  rebuild — institutional infra reused.
- **Two gates** (B1 aggregator structure, B2 driver dry-run) drain the
  correctness questions before any paid preprocessing.
- **Two paid steps** are user-gated: the ~1 h 49r1 preprocessing (Step 1) and
  the bulk MAM-first re-bake (Step 6).
- **Coordinates cleanly with the 50r1 plan** — the two cycles end up
  on separate Cloud Run images, swapped at the 2026-05-12 06z
  boundary by Cloud Run revisions, not runtime logic.

The next concrete action gated on the user, in order:
1. Write `ecmwf_49r1_coiled_preprocessing.py` (cheap, can be done in this branch).
2. Run gate B1 against an existing scan dump (free).
3. Approve the paid 49r1 preprocessing when ready (Step 1).
