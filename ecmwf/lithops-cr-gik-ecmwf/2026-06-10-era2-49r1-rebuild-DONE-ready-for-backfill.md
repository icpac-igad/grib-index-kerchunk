# Era-2 (49r1) Lithops runtime — REBUILD DONE, ready for backfill

**Date:** 2026-06-10
**Scope:** the VM-session handoff task from `grib-index-kerchunk` commit `2adff1b`
(era-2 49r1 backfill plan, §3) — *rebuild the era-2 Lithops runtime image and deploy it.*
**Status:** ✅ **COMPLETE. Cleared to start the era-2 backfill.**

---

## The ask (what the handoff requested)

1. Rebuild the `:49r1` Lithops runtime image (bake the 49r1 13-level template + era env).
2. Point `lithops_config.yaml` runtime at the `:49r1` tag.
3. Deploy the runtime and confirm.
4. Stop after deploy — do **not** run the backfill from the VM.

## What was done

| Step | Result |
|------|--------|
| 0. Auth | `ecmwf-lithops-deployer@e4drr-crafd` activated, project `e4drr-crafd` |
| 1. Rebuild | Cloud Build `9341bff5` → **SUCCESS** (2m3s) |
| 2. Config | `lithops_config.yaml:32` → `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1` ✅ |
| 3. Deploy | `lithops runtime deploy …:49r1` → service `lithops-worker-363-e004c694d6` up ✅ |
| 4. Confirm | `lithops runtime list` shows `…:49r1` (2048 MB, lithops 3.6.3) ✅ |

## The image (verified in GCR)

- **`gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1`** — digest `dddfa7504b5c`, built 2026-06-10 15:03
- Baked template: **`gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz`** (13-level, 14 MB)
- Era env: `ECMWF_REFERENCE_DATE=20250515`, `ECMWF_RESOLUTION=0p25`, `ECMWF_CONTROL_STREAM=enfo`

## One deviation (a real bug, fixed)

The first build **failed**: `namegenerator` was removed from PyPI
(`No matching distribution found`), breaking every Cloud Build of the runtime.
It is a vestigial pin copied from lithops' own default cloudrun requirements —
lithops 3.6.x neither requires nor imports it. Removed the one line; rebuild
succeeded. Fix committed to both repos (applies to all future 50r1 / 0p4
rebuilds too):

- `cno-e4drr` (private) — `0714d3a`
- `grib-index-kerchunk` (mirror) — `8efff5c`

> **Not pushed** — this session has no SSH key. Push manually when ready:
> `git -C <repo> push origin <branch>`

## Scope note — this run is 49r1 only

The plan builds **only `:49r1`**. The other two eras (0.4°-beta `:0p4`, 50r1 `:50r1`)
use the same per-era build mechanism (`--substitutions=_ERA=…`) but are separate
future runs. 50r1 is already deployed as the default image and needs no rebuild here.

---

## ➡️ Cleared for backfill

The `:49r1` runtime is live and correct. Drive the backfill from the **sandbox
session** (not this VM), per the handoff:

```
run_backfill_00z.sh --era 49r1 --from 2024-02 --to 2026-05
```

Pre-flight reminder (plan §, "Step 0"): GCS-audit which era-2 00z dates already
have 51 parquets, so the backfill only fills gaps (idempotent).
