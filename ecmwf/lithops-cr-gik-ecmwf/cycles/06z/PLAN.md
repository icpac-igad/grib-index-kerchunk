# 06z backfill — session A runbook

**Status:** ✅ **Ready to run — no new images required.**
**Why:** 06z is a **short run reaching only +144h** (S3-verified: `+144h` → 200,
`+150h`/`+240h`/`+360h` → 404), but the baked templates are **cycle-agnostic** —
they carry only per-variable Zarr schema, never step data, so the existing
`:0p4` / `:49r1` / `:50r1` images serve every cycle. See [`../README.md`](../README.md) §1a.

**Read first:** [`../README.md`](../README.md) · [`../../ECMWF_00Z_BACKFILL_SUMMARY.md`](../../ECMWF_00Z_BACKFILL_SUMMARY.md)

---

## Scope

| Era | Runtime tag | Dates | Window |
|-----|-------------|-------|--------|
| 0p4 | `:0p4` | ~401 | 2023-01-18 → 2024-02-28 |
| 49r1 | `:49r1` | ~804 | 2024-02-29 → 2026-05-12 |
| 50r1 | `:50r1` | ~51 | 2026-05-13 → 2026-07-02 |
| | | **~1,256** | |

Writes to `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/06z/` —
**49 steps per date** (0–144h, 3-hourly), against 85 for a 00z/12z date.

⚠️ **6 permanently absent dates:** `20230427`–`20230502` (0p4, 404 on S3).

⚠️ 06z runs at the **default 2048 MB**, which is also where 00z daily ops live.
Coordinate so a rolling 00z job is not competing for the same 35-slot pools.
(12z uses 2560 MB, 18z 3072 MB — see [`../README.md`](../README.md) §3.)

---

## Phase 0 — already done

- [x] `forecast_hours_for_run()` scopes the hour list to the cycle, so 06z requests
      49 steps instead of 85 — avoiding 1,836 futile ranged GETs per date.
- [x] No template or image work needed ([`../README.md`](../README.md) §1a).
- [ ] Raise `maxScale` per [`../README.md`](../README.md) §2 (shared with all sessions).

---

## Step 1 — isolated checkout & environment

```bash
git worktree add /tmp/gik-06z HEAD
cd /tmp/gik-06z/devops/lithops_cr_ecmwf_gik

export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
gcloud auth activate-service-account \
  --key-file=service_account/ecmwf-lithops-deployer-key.json --project=e4drr-crafd
```

## Step 2 — `--run` passthrough

`run_backfill_00z.sh` hardcodes `--run "00"`. Add a `--run` flag (default `00`),
pass it to `run_lithops_ecmwf.py`, and log to `logs/backfill_06z/`.

## Step 3 — Herbie verification gate (3 dates, ~25 min)

**Highest-value step in this plan** — it is what proves the new short templates are
correct. Do not start Step 4 until all three pass.

| Era | Suggested date | Runtime tag | Pass threshold |
|-----|----------------|-------------|----------------|
| 0p4 | `20230601` | `:0p4` | r ≥ 0.9997 |
| 49r1 | `20250115` | `:49r1` | r ≥ 0.9999 |
| 50r1 | `20260601` | `:50r1` | r ≥ 0.9999 |

Run one date at `--run 06`, confirm 51/51 files **and** that the step axis stops at
+144h (49 steps, not 85), then compare against Herbie with
`ecmwf/compare_gik_herbie_pressure.py`.

Validate at a **late lead time** (e.g. +120h) as well as +0h. A short-cycle bug shows
up at the tail of the axis, and a t+0 check alone would pass regardless.

## Step 4 — full waves

```bash
# 0p4
export ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4  ECMWF_CONTROL_STREAM=enfo
bash run_backfill_00z.sh --era 0p4  --run 06 --from 2023-01 --to 2024-02

# 49r1
export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
bash run_backfill_00z.sh --era 49r1 --run 06 --from 2024-03 --to 2026-05

# 50r1
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
bash run_backfill_00z.sh --era 50r1 --run 06 --from 2026-05 --to 2026-07
```

Remember the two switches: `--era` sets only the env; the `runtime:` tag in
`lithops_config.yaml` must be flipped to match **between** waves, never during one
(`ECMWF_00Z_BACKFILL_SUMMARY.md` §4).

⚠️ Era boundaries are **mid-month** (`20240229`, `20260512/13`) — handle those
boundary dates with explicit `--start-date`/`--end-date` calls, as
`run_gaps_49r1-20240229_50r1-20260701-20260702.sh` does.

## Step 5 — verify

**Never judge by exit code** — a fully successful wave hangs at exit and `timeout`
returns **124**. Count GCS objects: 51/51 per date, contiguous over the era windows
apart from the 6 known 0p4 holes.

## Step 6 — hand off

Report dates written, any `<51` dates, and Herbie stats. **Do not** run the HF mirror
or Icechunk ingest — serial Phase 3, after all three cycles land.

---

## Estimated wall time

~1,256 dates × ~7 min/date (≈58 % of a 00z date — 49 steps vs ~85). At 35-wide
≈ **4 h**; ~85 min only at ~103 concurrent.
