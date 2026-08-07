# ecFlow daily GIK operations — current configuration, the 49r1 era defect, and the 50r1 fix

**Subject:** branch `operationalize-ecflow` on `icpac-team-branch-cno-e4drr`, commits of **2026-07-27**
**Server:** `crafd-gpu` · `/home/hkoros/e4drr-ecflow` · venv `~/gikvenv` (lithops 3.6.4)
**Verdict:** ⚠️ **The daily suite is wired to the 49r1 era. It cannot produce today's dates.**
**Companion:** [`ECMWF_00Z_BACKFILL_SUMMARY.md`](ECMWF_00Z_BACKFILL_SUMMARY.md) §4 (era selection) and §11 (50r1 ops)

---

## 1. What was built on 2026-07-27

Thirteen commits took the suite from nothing to "operational". Phase 0 (S1–S8) and
Phase 1–2 (C1, C4) are ticked in the branch `README.md`:

| Task | Claim |
|------|-------|
| S2 | `uv` + orchestrator venv `~/gikvenv` built; lithops **3.6.4** verified |
| S3 | GCP SA key `ecmwf-lithops-deployer@e4drr-crafd` installed, `chmod 600`, `GOOGLE_APPLICATION_CREDENTIALS` set |
| S4 | GCS + Cloud Run auth verified — *"50r1 runtime already deployed"* |
| S6 | ecFlow suite `ecmwf_ifs_gik` authored, layers L1–L4, cron **06:00 UTC** |
| S7 | Scheduling validated — auto-schedule, L1–L4 trigger chain, `meter=100`, `event=committed` |
| S8 | Container→host executor bridge proven; `discover` does real read-only GCS + S3 |
| **C1** | **"Real run proven — 49r1 `20260315` → 51 parquet files, via ecFlow"** |
| **C4** | Daily suite assembled, 06:00 UTC cron, *"era-aware"*, failure→abort, dashboard telemetry. Note: ***"50r1 template blocks current dates"*** |

Architecture as designed: **lithops = the worker** (fans out to Cloud Run);
**ecFlow = the foreman** (fires 06:00 UTC, sequences L1–L4, handles retry, shows status).
L3 (Icechunk append) and L4 (source.coop mirror) are **stubs**, deferred to Phase 3.

So: the scheduling engine is real and working. **The era wiring is not.**

---

## 2. The defect, in the team's own words

Two entries in the branch `README.md` state it plainly:

> **C1** · Real run proven — **49r1** `20260315` → 51 parquet files, via ecFlow
>
> **C4** · … *`50r1` template blocks current dates*

`20260315` sits inside the **49r1** window (`2024-02-29 → 2026-05-12`). Every date
from `20260513` onward — including every date the 06:00 UTC cron will ever ask for —
is **50r1**. The pipeline has never successfully produced a current date, and the
tracker records this as a known blocker rather than a defect.

**The daily cron is therefore firing every morning against an era that ended on
2026-05-12.**

---

## 3. Root cause — the two-switch problem

Era selection is **two independent switches that nothing links** (see
[`ECMWF_00Z_BACKFILL_SUMMARY.md`](ECMWF_00Z_BACKFILL_SUMMARY.md) §4):

| # | Switch | Set where | Controls |
|---|--------|-----------|----------|
| 1 | `ECMWF_CONTROL_STREAM`, `ECMWF_REFERENCE_DATE`, `ECMWF_RESOLUTION` | **exported env** | which S3 bytes are **read** |
| 2 | `runtime:` tag | **`lithops_config.yaml:32`** | which image/template **decodes** them |

### Switch 2 is correct on the branch

`16d0141` ("deploy :50r1 runtime + point config at it", 2026-06-27) is an ancestor of
`operationalize-ecflow`, and the branch file reads:

```yaml
runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1   # lithops_config.yaml:32
```

### Switch 1 is never set — and its defaults are 49r1

Nothing in the ecFlow work exports the era env. `run_lithops_ecmwf.py` therefore
falls back to its module-level defaults:

```python
REFERENCE_DATE       = os.environ.get('ECMWF_REFERENCE_DATE',  '20240529')  # ← a 49r1-era date
ECMWF_RESOLUTION     = os.environ.get('ECMWF_RESOLUTION',      '0p25')      # ← fine for both
CONTROL_STREAM       = os.environ.get('ECMWF_CONTROL_STREAM',  'enfo')      # ← 49r1 stream, NOT oper
```

**The defaults are a complete 49r1 configuration.** An orchestrator that sets no era
env silently runs 49r1 — no warning, no error. That is exactly what C1 recorded.

50r1 needs `ECMWF_CONTROL_STREAM=oper` (at the 49r1→50r1 cutover, `enfo` drops
51→50 members and the control moves to the `oper` stream) and
`ECMWF_REFERENCE_DATE=20260513`.

### Why the symptom reads as "50r1 template blocks current dates"

With switch 1 on 49r1 and switch 2 on `:50r1`, the driver reads 49r1 `enfo` paths and
hands them to workers holding the 50r1 14-level template → `51/51 "Template not
found"`, **zero files**, for any date. C1 nevertheless wrote 51 files.

**Both cannot be true of the same configuration.** The server's *live* config must
differ from the branch's committed one — most likely `runtime:` is still `:49r1`
there, which would make C1 succeed (consistent 49r1 on both switches) and make every
current date fail (50r1 data, 49r1 template). The team read that failure as "the 50r1
template is blocking us" rather than "we are pinned to the wrong era".

> ⚠️ **Unverified — I have no access to `crafd-gpu`.** §7 lists the three commands
> that settle it. The conclusion that the suite runs 49r1 rests on C1 + C4, which is
> already sufficient; the open question is only *which* switch is wrong on the server.

### It was never designed in

The original plan (`devops/ecflow-ifs-operational/OPERATIONALIZE_ECFLOW_PLAN.md`,
added in `5b07242`, deleted in `d3604ad`) contains **no mention of era handling** —
no `ECMWF_CONTROL_STREAM`, no `ECMWF_REFERENCE_DATE`, no runtime tag. Its only "50r1"
reference is an unrelated Icechunk OOM concern. C4's "era-aware" claim is not
supported by any era logic in the plan or the repo.

---

## 4. The second structural problem: the suite is not in version control

`devops/ecflow-ifs-operational/` was added (`5b07242`) and then removed
(`d3604ad`, `e0c0e4d` — "tracker now lives in README.md"). The **suite definition,
the L1–L4 task wrappers, and whatever env they export exist only on `crafd-gpu`.**

Consequences:

- The era wiring — the thing that is wrong — is **unreviewable**. This defect could
  not have been caught by reading the repo.
- A server rebuild loses the operational configuration.
- The repo's `lithops_config.yaml` and the server's cannot be diffed.

**Committing the suite is a prerequisite for the fix being durable**, not a nicety.

---

## 5. The fix — an era-pinned run directory

You suggested duplicating the lithops env for 50r1. That is the right instinct and it
works, provided the duplicate pins **both** switches. Duplicating the venv alone does
not help: the era comes from env vars + the config tag, not from which Python is used.

Recommended layout — one self-contained directory per era, so the era cannot be
inherited from an ambient shell:

```
devops/lithops_cr_ecmwf_gik/
  era_profiles/
    50r1.env                 # switch 1, committed
    49r1.env                 # for historical re-runs
  lithops_config.50r1.yaml   # switch 2, committed, tag pinned to :50r1
  run_era_daily.sh           # sets BOTH, asserts they agree, then runs
```

### `era_profiles/50r1.env`

```bash
# ECMWF 50r1 era — 2026-05-13 onward. Must match the :50r1 image.
export ECMWF_REFERENCE_DATE=20260513
export ECMWF_RESOLUTION=0p25
export ECMWF_CONTROL_STREAM=oper
export ECMWF_ERA_TAG=50r1                 # asserted against lithops_config below
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
export UV_PYTHON=3.12
```

### `run_era_daily.sh` — the preflight guard is the point

```bash
#!/usr/bin/env bash
set -euo pipefail
ERA="${1:?usage: run_era_daily.sh <era> <YYYYMMDD> [run]}"
DATE="${2:?}"; RUN="${3:-00}"
cd "$(dirname "${BASH_SOURCE[0]}")"

source "era_profiles/${ERA}.env"
CFG="lithops_config.${ERA}.yaml"

# --- Preflight: switch 1 and switch 2 MUST agree, or refuse to run --------------
TAG=$(grep -oP '(?<=ecmwf-lithops-runtime:)[A-Za-z0-9._-]+' "$CFG")
[[ "$TAG" == "$ECMWF_ERA_TAG" ]] || {
  echo "FATAL: config runtime :$TAG != era profile $ECMWF_ERA_TAG" >&2; exit 2; }

# --- Preflight: the date must belong to this era --------------------------------
case "$ERA" in
  50r1) [[ "$DATE" -ge 20260513 ]] ;;
  49r1) [[ "$DATE" -ge 20240229 && "$DATE" -le 20260512 ]] ;;
  0p4)  [[ "$DATE" -le 20240228 ]] ;;
esac || { echo "FATAL: $DATE is not in the $ERA era" >&2; exit 3; }

LITHOPS_CONFIG_FILE="$CFG" timeout 1500 \
  uv run run_lithops_ecmwf.py --start-date "$DATE" --end-date "$DATE" \
      --run "$RUN" --max-workers 4 --yes
RC=$?
[[ $RC -eq 0 || $RC -eq 124 ]] || exit $RC   # 124 = hang-at-exit = SUCCESS

# --- Verify by GCS object count, never by exit code ------------------------------
N=$(gsutil ls "gs://gik-ecmwf-aws-tf/${GCS_PARQUET_PREFIX}/${DATE:0:4}/${DATE:4:2}/${DATE}/${RUN}z/**" 2>/dev/null | wc -l)
[[ "$N" -eq 51 ]] || { echo "FATAL: $DATE wrote $N/51 files" >&2; exit 4; }
echo "OK $ERA $DATE ${RUN}z 51/51"
```

Two properties matter for ecFlow:

1. **A wrong era aborts before spending anything** — the era mismatch that caused
   this defect becomes a hard exit 2 instead of a silent 49r1 run.
2. **Exit 124 is treated as success, and success is confirmed by counting 51 objects.**
   The driver hangs at interpreter exit (`ECMWF_00Z_BACKFILL_SUMMARY.md` §5, gotcha 1),
   so an ecFlow task that gates on the exit code will report failure on good runs and
   — worse — success on runs that wrote nothing.

### Service account

No new SA is needed. `ecmwf-lithops-deployer@e4drr-crafd` (already installed under S3)
carries GCS write + Cloud Run invoke for **all** eras — era is a property of the image
and the env, not of the identity. Keep `GOOGLE_APPLICATION_CREDENTIALS` pointing at
the existing `chmod 600` key.

### Which Cloud Run the daily suite must use

| | |
|---|---|
| Image | `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:50r1` |
| Service (lithops **3.6.4** — the server's venv) | `lithops-worker-364-5a680638f5` |
| Service (lithops 3.6.3) | `lithops-worker-363-d1cea16f95` |
| Region / size | `europe-west3` · 2 GB · 2 vCPU · concurrency 1 |

You never name the service — lithops derives it from `runtime` + `runtime_memory` +
**its own version**. `~/gikvenv` has 3.6.4, so it will bind `…-364-5a680638f5`. If that
registration is ever missing, lithops silently auto-deploys a fresh service mid-run,
costing minutes on a time-boxed daily window.

---

## 6. Migration steps

1. **Commit the ecFlow suite** — `ecmwf_ifs_gik.def` and the L1–L4 wrappers from
   `/home/hkoros/e4drr-ecflow` into `devops/ecflow-ifs-operational/`. Do this **first**:
   until the wrappers are readable, no one can confirm what era the suite runs.
2. Add `era_profiles/`, `lithops_config.50r1.yaml`, `run_era_daily.sh` (§5).
3. Repoint the L2 wrapper at `run_era_daily.sh 50r1 $ECF_DATE 00`, dropping any
   ambient env it currently relies on.
4. **Re-run C1 against a 50r1 date** (e.g. `20260703`) and require 51/51 in GCS.
   C1 as it stands proves only that the 49r1 path works.
5. Herbie-validate that date before declaring the suite operational
   ([`ECMWF_00Z_BACKFILL_SUMMARY.md`](ECMWF_00Z_BACKFILL_SUMMARY.md) §7, expect r ≥ 0.9999).
6. Backfill the tail the cron never produced — `20260703` → today — then let the cron
   take over.
7. Only then resume C5 (tracksuite deploy + soak test).

---

## 7. Confirm on the server

```bash
ssh crafd-gpu
grep -E 'runtime: gcr' ~/e4drr-ecflow/**/lithops_config.yaml     # expect :50r1 — likely :49r1
grep -rnE 'ECMWF_(CONTROL_STREAM|REFERENCE_DATE|RESOLUTION)' ~/e4drr-ecflow/   # expect: no hits
gsutil ls 'gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/2026/07/**' | wc -l   # dates since 0703?
```

Predictions, from the evidence above:

- The runtime tag is **`:49r1`**, not the `:50r1` the branch commits.
- **No** `ECMWF_*` era env is exported anywhere in the suite.
- **No** parquets exist for any date after `20260702` — the daily cron has produced
  nothing since the catalog's last backfilled date.

---

## 8. Summary

| | |
|---|---|
| **What works** | ecFlow engine, 06:00 UTC cron, L1–L4 trigger chain, telemetry, auth, container→host bridge — all genuinely proven |
| **What is broken** | Era wiring. The suite runs **49r1**; every date it is asked for is **50r1** |
| **Root cause** | `run_lithops_ecmwf.py` defaults to `ECMWF_CONTROL_STREAM=enfo` + `ECMWF_REFERENCE_DATE=20240529` (a complete 49r1 config), and nothing in the suite overrides them |
| **Why it went unnoticed** | An era mismatch is silent — it writes zero files rather than raising; and the suite is not in version control, so it could not be reviewed |
| **Fix** | Era-pinned profile + config + a wrapper that asserts both switches agree and verifies 51/51 in GCS (§5) |
| **Also required** | Commit the suite; re-prove C1 on a 50r1 date; backfill `20260703`→today |
| **Not required** | A new service account — `ecmwf-lithops-deployer@e4drr-crafd` already covers every era |
