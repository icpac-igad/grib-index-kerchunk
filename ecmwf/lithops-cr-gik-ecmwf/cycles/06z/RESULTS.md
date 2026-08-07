# 06z backfill — COMPLETE

**Finished:** 2026-07-31
**Catalog:** `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/06z/`
**Result:** ✅ **1,256 dates at 51/51**, `20230118 → 20260702` — identical coverage to the 00z corpus.

---

## Final verification (GCS object counts, not driver exit codes)

```
expected dates 20230118..20260702 : 1,262
delivered at 51/51                : 1,256
missing                           :     6   — all known S3-absent
  20230427 20230428 20230429 20230430 20230501 20230502   (0.4°, never published by ECMWF)
UNEXPECTED missing                :     0
```

Per era:

| Era | Runtime | Waves | Window |
|-----|---------|-------|--------|
| 0p4 | `:0p4` | 14 | 2023-01-18 → 2024-02-27 |
| 49r1 | `:49r1` | 28 | 2024-02-28 → 2026-05-11 |
| 50r1 | `:50r1` | 3 | 2026-05-12 → 2026-07-02 |
| | | **45** | |

Note the era windows differ from the 00z table at both ends — see the cutover finding below.

## Axis: 49 steps, not 85

06z is a short run. Verified on `20260601`:

| | Rows | Steps | Max step |
|---|---|---|---|
| 06z | 9,090 | **49** | **144h** |
| 00z | 15,498 | 85 | 360h |

No new templates or images were needed — templates are cycle-agnostic
(`cycles/README.md` §1a). `forecast_hours_for_run()` scopes the hour list from
`--run`, avoiding 1,836 futile ranged GETs per date.

## Herbie validation

Run via `run_cycle_herbie_gate.sh 06` before the waves, at T+0 **and** T+144h (the
last step of the truncated axis — a T+0-only check would pass even with a wrong axis):

| Era | Date | Step | t@500 | t@850 |
|-----|------|------|-------|-------|
| 0p4 | 20230318 | T+0h | 0.999974 | 0.999934 |
| 0p4 | 20230318 | T+144h | **0.999979** | **0.999966** |
| 49r1 | 20240327 | T+0h | **1.000000** | **1.000000** |

Herbie resolved `2023-Mar-18 06:00 UTC F00`, confirming the comparison is 06z-vs-06z.
"Herbie: 50 members" vs GIK's 51 is expected — Herbie's `enfo` returns the 50
perturbed members; GIK carries 51 including the bundled control.

> ⚠️ **Not completed:** the 50r1 arm of the gate. The gate was paused to free
> bandwidth for the waves. 50r1 at 06z is structurally verified (51/51, 49 steps)
> but **not** Herbie-verified. `20260512` in particular was re-run under 50r1's
> 14-level template without confirming the level count (S3 throttled the index
> fetch). Worth a targeted comparison on `20260512` and `20260621` before sign-off.

## ⚠️ The main finding: era cutovers are cycle-granular

Two dates failed under the era table's day-granular boundaries:

| Date | Run as | Result | Actual era at 06z |
|------|--------|--------|-------------------|
| `20240228` | 0p4 | **0 files** (0.4° paths 404) | **49r1** — `ifs/0p25/enfo` 06z → 200 |
| `20260512` | 49r1 | **50/51** (no control) | **50r1** — `oper` control 06z → 200 |

Both gave 51/51 when re-run under the next era (`fix_cycle_boundary_dates.sh 06`).
The 00z corpus never exposed this because at 00z both cutovers do fall where the
table says. **12z and 18z are affected at least as much** — the transition is
monotonic through the day. See `cycles/README.md` §1b.

## Run history — 4 driver runs

| # | Why it ended | Outcome |
|---|--------------|---------|
| 1 | DNS outage at ~13:02 | ~23 waves; `49r1 2024-09` logged `PARTIAL(0/30)` but was in fact **complete** — Cloud Run workers finish regardless of the driver, only the local monitor went blind |
| 2 | stopped (duplicate-driver incident) | resumed via the range-aware skip |
| 3 | stopped deliberately after a wave | to apply the shorter timeout |
| 4 | **ran to completion** | 45/45 waves |

Two fixes came out of this and are committed:

- **`require_net()`** — blocks on DNS/GCS loss instead of marching through the
  remaining waves logging `PARTIAL(0/n)` for work that may have succeeded.
- **`WAVE_TIMEOUT` 2400s → 1500s** — the driver finishes work in 5–16 min then
  hangs at interpreter exit, so the timeout is what actually ends a wave. Six
  consecutive waves burned the full 2400s having completed in ~15 min. Cutting it
  is safe: success comes from GCS counts, and a truncated wave is redone next pass.

**Operational lesson:** log silence ≠ a dead driver — a wave can run 40 minutes
without writing. Verify with `ps`, not by tailing the log.

## Cost

List-price estimate (no billing access from the deployer SA): ~1,262 activations
× 2 vCPU × ~350–550 s at 2 GiB ≈ **$23–37**. One activation per **date**, not per
member — the 51 members are processed inside a single worker.

## Files

| File | Role |
|------|------|
| `../../run_cycle_waves.sh` | wave driver (`06`/`12`/`18`) |
| `../../run_cycle_herbie_gate.sh` | pre-wave Herbie gate |
| `../../fix_cycle_boundary_dates.sh` | the two cycle-granular cutover fixes (any cycle) |
| `summary.log` | wave-by-wave record (all 4 runs, cumulative) |
| `driver_final.log` | run 4 — the one that completed |
| `fix_boundary.log` | boundary-date fix output |

## Remaining before sign-off

- [ ] Herbie-verify 50r1 at 06z (`20260512`, `20260621`)
- [ ] HF mirror — `mirror_gcs_to_hf_v2.py` currently covers 00z only
- [ ] Icechunk ingest — **single writer**, after 12z/18z land
