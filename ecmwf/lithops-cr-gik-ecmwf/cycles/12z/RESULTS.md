# 12z backfill — COMPLETE

**Finished:** 2026-08-05 (waves 2026-07-30/31; boundary dates fixed 2026-08-05)
**Catalog:** `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/12z/`
**Result:** ✅ **1,255 dates at 51/51**, `20230118 → 20260702`

---

## Coverage

```
expected dates 20230118..20260702 : 1,262
S3-absent at 12z (0.4°, 404)      :     7   — 20230426..20230502
delivered at 51/51                : 1,255
```

⚠️ **The unpublished window is cycle-dependent.** At 00z/06z it is
`20230427..20230502` (6 dates); at 12z/18z it starts a day earlier and includes
**`20230426`** (7 dates). Verified on S3: `20230426` is 200 at 00z/06z, **404 at
12z/18z**. So 12z/18z top out at **1,255**, not the 1,256 of 00z/06z.

Full-length axis (0–360h, ~85 steps), same as 00z — 12z is not a short run.

## Run

Waves ran in a separate session from a `/tmp/gik-12z` worktree at
`runtime_memory: 2560`, giving it Cloud Run service pools independent of 06z
(2048) and 18z (3072). Wave record: `summary.log` — 27 COMPLETE, 4 PARTIAL, 3 SKIP.

12z is the heaviest cycle: ~85 steps/date vs 49 for 06z/18z, and several waves hit
the then-2400s timeout (one ran 54m29s).

## Boundary dates — the cycle-granular cutover

The run finished with three unexpected gaps. One was unfixable, two were real:

| Date | Ran as | Result | Correct era at 12z | After fix |
|------|--------|--------|--------------------|-----------|
| `20230426` | 0p4 | 0 files | — **404 on S3** | not fixable |
| `20240228` | 0p4 | 0 files | **49r1** (`ifs/0p25` 200) | ✅ 51/51 |
| `20260512` | 49r1 | 50/51 (no control) | **50r1** (`oper` control) | ✅ 51/51 |

Fixed with `../../fix_cycle_boundary_dates.sh 12`. This is the same finding as 06z
(`cycles/README.md` §1b) — the era cutovers are cycle-granular, and the transition
is monotonic through the day, so 12z is affected exactly as 06z was.

## Not done

- [ ] **Herbie validation — 1 of 6 checks done, and it passed.** The 12z gate
      (`run_cycle_herbie_gate.sh 12`) was paused mid-run to free local bandwidth
      for the waves. It completed the 0p4 T+0 pair before stopping
      (`herbie_gate_partial.log`), against Herbie `2023-Mar-18 12:00 UTC F00`:

      | Era / date | Level | GIK | Herbie | mean r | RMSE | max\|diff\| | vs r ≥ 0.9997 |
      |---|---|---|---|---|---|---|---|
      | 0p4 `20230318` T+0h | t@500 | 51 | 50 | **0.999978** | 0.0159 K | 0.504 K | ✅ |
      | 0p4 `20230318` T+0h | t@850 | 51 | 50 | **0.999841** | 0.0503 K | 2.28 K | ✅ |

      So 0p4 is confirmed at T+0; **49r1, 50r1 and every late-step (T+240h) check
      remain unvalidated.** The late-step check is the one that matters most — a
      T+0-only pass cannot detect a wrong time axis. Worth running before
      sign-off, especially the re-run boundary dates.

      ⚠️ `compare_gik_herbie_pressure.py:315` hardcodes `00Z` in the plot
      suptitle, so 12z plots are mislabelled `00Z`. The comparison itself is
      correct — line 221 passes `--run` into Herbie — but the PNGs read wrong.
- [ ] HF mirror (`mirror_gcs_to_hf_v2.py` covers 00z only)
- [ ] Icechunk ingest — single writer, after all cycles land

## Files

| File | Role |
|------|------|
| `PLAN.md` | the runbook this session followed |
| `summary.log` | wave-by-wave record |
| `herbie_gate_partial.log` | the one gate check that completed (0p4 T+0) |
| `../../run_cycle_waves.sh` | wave driver |
| `../../fix_cycle_boundary_dates.sh` | the two cutover fixes |
