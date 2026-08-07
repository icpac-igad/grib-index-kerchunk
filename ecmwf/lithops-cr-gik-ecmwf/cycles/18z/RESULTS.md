# 18z backfill — COMPLETE

**Finished:** 2026-08-05 (waves 2026-07-30/31; boundary dates fixed 2026-08-05)
**Catalog:** `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/18z/`
**Result:** ✅ **1,255 dates at 51/51**, `20230118 → 20260702`

---

## Coverage

```
expected dates 20230118..20260702 : 1,262
S3-absent at 18z (0.4°, 404)      :     7   — 20230426..20230502
delivered at 51/51                : 1,255
```

⚠️ **The unpublished window is cycle-dependent.** At 00z/06z it is
`20230427..20230502` (6 dates); at 12z/18z it starts a day earlier and includes
**`20230426`** (7 dates) — S3-verified 200 at 00z/06z, **404 at 12z/18z**. So
12z/18z top out at **1,255**, not 1,256.

Short axis (0–144h, 49 steps), same as 06z — `+150h` and beyond are 404 for this cycle.

## Run

Waves ran in a separate session from a `/tmp/gik-18z` worktree at
`runtime_memory: 3072`, giving it Cloud Run pools independent of 06z (2048) and
12z (2560). Wave record: `summary.log` — 39 COMPLETE, 11 PARTIAL, 14 SKIP
(cumulative across restarts, so PARTIAL/SKIP double-count re-runs).

## Boundary dates — the cycle-granular cutover

| Date | Ran as | Result | Correct era at 18z | After fix |
|------|--------|--------|--------------------|-----------|
| `20230426` | 0p4 | 0 files | — **404 on S3** | not fixable |
| `20240228` | 0p4 | 0 files | **49r1** (`ifs/0p25` 200) | ✅ 51/51 |
| `20260512` | 49r1 | 50/51 (no control) | **50r1** (`oper` control) | ✅ 51/51 |

Fixed with `../../fix_cycle_boundary_dates.sh 18`.

### Note on `refill_18z_gaps.sh.orig`

This session wrote its own refill script (preserved here for the record). It
correctly identified the two gap dates, but assigned them the **wrong eras** —
`20240228` as `0p4` and `20260512` as `49r1`, i.e. the eras the day-granular table
implies. Those are precisely the assignments that produced the gaps in the first
place, so the refill could not have closed them. Superseded by
`fix_cycle_boundary_dates.sh`, which uses the *next* era for both.

That mistake is worth keeping visible: the era table reads as authoritative, and
nothing in the tooling flags a date sitting on the wrong side of a cutover — the
run simply writes 0 or 50 files and reports `PARTIAL`.

## Not done

- [ ] **Herbie validation.** The 18z gate was paused mid-run to free bandwidth and
      never completed — it produced no correlations. **18z has no Herbie
      confirmation at all.** Worth running before sign-off, especially the two
      re-run boundary dates.
- [ ] HF mirror (`mirror_gcs_to_hf_v2.py` covers 00z only)
- [ ] Icechunk ingest — single writer, after all cycles land

## Files

| File | Role |
|------|------|
| `PLAN.md` | the runbook this session followed |
| `summary.log` | wave-by-wave record |
| `refill_18z_gaps.sh.orig` | this session's refill attempt (wrong eras — see above) |
| `../../run_cycle_waves.sh` | wave driver |
| `../../fix_cycle_boundary_dates.sh` | the two cutover fixes |
