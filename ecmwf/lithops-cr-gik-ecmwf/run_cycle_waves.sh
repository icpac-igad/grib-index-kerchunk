#!/usr/bin/env bash
# ==============================================================================
# Full 3-era backfill for one cycle (06z / 12z / 18z). Self-contained.
# ==============================================================================
# Walks eras oldest -> newest, one calendar month per wave, flipping the lithops
# runtime tag between eras (the two-switch rule -- ECMWF_00Z_BACKFILL_SUMMARY.md
# section 4). Era windows are clipped to the S3-proven cutovers, so the
# mid-month boundaries (20240229, 20260512/13) are handled correctly.
#
#   0p4   20230118 -> 20240228   (:0p4  0p4 /enfo)
#   49r1  20240229 -> 20260512   (:49r1 0p25/enfo)
#   50r1  20260513 -> 20260702   (:50r1 0p25/oper)
#
# Idempotent: a month whose every date already has 51 objects is skipped, so the
# script is safe to re-run after an interruption.
#
# Usage:  bash run_cycle_waves.sh 06 [--from YYYY-MM] [--dry-run]
#
# NOTE exit 124 from a wave means the driver finished the work then hung at
# interpreter exit -- that is SUCCESS. Truth is the per-date GCS object count.
# ==============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

RUN="${1:?usage: run_cycle_waves.sh <06|12|18> [--from YYYY-MM] [--dry-run]}"; shift || true
FROM_MONTH=""; DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM_MONTH="$2"; shift ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac; shift
done
[[ "$RUN" =~ ^(06|12|18)$ ]] || { echo "FATAL: run must be 06|12|18" >&2; exit 1; }

CFG=lithops_config.yaml
BUCKET=gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf
LOG_DIR="$SCRIPT_DIR/logs/waves_${RUN}z"; mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"
MAX_WORKERS="${MAX_WORKERS:-35}"
# Wall-clock cap per wave. The driver reliably finishes the work in ~5-16 min and
# then HANGS at interpreter exit, so the timeout is what actually ends a wave --
# on 2026-07-30/31 six consecutive waves burned the full 2400s having completed
# in ~15 min, ~25 min wasted each. 1500s still clears the slowest observed real
# work with margin, and cutting a wave short is safe: success is judged from GCS
# date counts, and anything truncated is simply redone by the next pass.
WAVE_TIMEOUT="${WAVE_TIMEOUT:-1500}"

export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"

set_rt () {
  sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$1|" "$CFG"
  echo "  runtime -> :$1  ($(grep -E 'runtime: gcr' "$CFG" | tr -d ' '))"
}

# ---------------------------------------------------------------------------
# Network guard. A local outage makes every GCS call fail, which would (a) make
# the skip-check report 0 done dates and (b) let the driver march through every
# remaining wave failing in seconds -- logging PARTIAL(0/n) for work that may
# actually have SUCCEEDED on Cloud Run (the workers write GCS themselves and do
# not depend on this host). Observed for real on 2026-07-30: a DNS outage logged
# 49r1 2024-09 as PARTIAL(0/30) when all 30 dates were in fact complete.
# So: block until DNS+GCS come back, and abort rather than mislabel.
require_net () {
  local tries=0
  until getent hosts storage.googleapis.com >/dev/null 2>&1 \
        && gsutil ls "$BUCKET/" >/dev/null 2>&1; do
    tries=$(( tries + 1 ))
    if [[ "$tries" -gt 60 ]]; then      # ~30 min
      echo "FATAL: no GCS connectivity after 30 min -- aborting so waves are not" >&2
      echo "       mislabelled. Re-run this script when the network is back." >&2
      exit 3
    fi
    echo "  [net] GCS unreachable (try $tries) -- waiting 30s ..."
    sleep 30
  done
}
era_env () {
  case "$1" in
    0p4)  export ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4  ECMWF_CONTROL_STREAM=enfo
          export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz" ;;
    49r1) export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
          export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz" ;;
    50r1) export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
          export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz" ;;
  esac
}

# era:first:last  -- S3-proven cutovers
ERAS="0p4:20230118:20240228 49r1:20240229:20260512 50r1:20260513:20260702"

echo "======================================================================"
echo "  ECMWF ${RUN}z backfill — 3 eras -> $BUCKET/{Y}/{M}/{date}/${RUN}z/"
echo "  workers/wave: $MAX_WORKERS   logs: $LOG_DIR"
[[ -n "$DRY" ]] && echo "  MODE: DRY RUN (no dispatch)"
echo "======================================================================"

for spec in $ERAS; do
  era="${spec%%:*}"; rest="${spec#*:}"; first="${rest%%:*}"; last="${rest##*:}"

  # month chunks clipped to the era window
  mapfile -t CHUNKS < <(python3 -c "
import datetime as dt, calendar
f=dt.date(int('$first'[:4]),int('$first'[4:6]),int('$first'[6:]))
l=dt.date(int('$last'[:4]),int('$last'[4:6]),int('$last'[6:]))
y,m=f.year,f.month
while (y,m)<=(l.year,l.month):
    a=max(f,dt.date(y,m,1)); b=min(l,dt.date(y,m,calendar.monthrange(y,m)[1]))
    if a<=b: print(f'{y:04d}-{m:02d} {a:%Y%m%d} {b:%Y%m%d}')
    m+=1
    if m>12: m=1; y+=1
")

  echo; echo "############ era $era  ($first -> $last, ${#CHUNKS[@]} waves) ############"
  TAG_SET=0
  for chunk in "${CHUNKS[@]}"; do
    read -r ym s e <<<"$chunk"
    [[ -n "$FROM_MONTH" && "$ym" < "$FROM_MONTH" ]] && continue

    require_net    # never judge a wave while the network is down

    # ---- idempotency: count only dates INSIDE this chunk --------------------
    # A boundary month belongs to two eras (2024-02 is 0p4 01..28 AND 49r1 29
    # alone; 2026-05 is 49r1 01..12 AND 50r1 13..31), so a whole-month object
    # count would let one era's data mask the other era's missing wave.
    want=$(python3 -c "
import datetime as dt
a=dt.date(int('$s'[:4]),int('$s'[4:6]),int('$s'[6:])); b=dt.date(int('$e'[:4]),int('$e'[4:6]),int('$e'[6:]))
print((b-a).days+1)")
    done_dates=$(gsutil ls "$BUCKET/${ym%-*}/${ym#*-}/*/${RUN}z/*.parquet" 2>/dev/null \
      | awk -v s="$s" -v e="$e" -F/ '{d=$(NF-2)} d>=s && d<=e {c[d]++} END{n=0; for(k in c) if(c[k]>=51) n++; print n+0}')
    if [[ "$done_dates" -ge "$want" ]]; then
      echo "  SKIP $era $ym  ($done_dates/$want dates already at 51/51)"
      echo "SKIP  $era $ym" >> "$SUMMARY"; continue
    fi

    if [[ -n "$DRY" ]]; then
      echo "  DRY  $era $ym  $s -> $e  ($want dates, $done_dates already done)"; continue
    fi

    # flip the tag once per era, immediately before its first real wave
    if [[ "$TAG_SET" -eq 0 ]]; then set_rt "$era"; era_env "$era"; TAG_SET=1; fi

    echo "----------------------------------------------------------------------"
    echo "  $era  $ym  ($s -> $e, $want dates)  ${RUN}z"
    echo "----------------------------------------------------------------------"
    T0=$(date +%s)
    timeout "$WAVE_TIMEOUT" uv run run_lithops_ecmwf.py \
        --start-date "$s" --end-date "$e" --run "$RUN" \
        --max-workers "$MAX_WORKERS" --yes \
        > "$LOG_DIR/${era}_${ym}.log" 2>&1
    RC=$?
    T1=$(date +%s); EL=$(( T1-T0 ))

    require_net    # the post-wave count must not be taken during an outage
    done_dates=$(gsutil ls "$BUCKET/${ym%-*}/${ym#*-}/*/${RUN}z/*.parquet" 2>/dev/null \
      | awk -v s="$s" -v e="$e" -F/ '{d=$(NF-2)} d>=s && d<=e {c[d]++} END{n=0; for(k in c) if(c[k]>=51) n++; print n+0}')
    case "$RC" in
      0)   note=OK ;;
      124) note="OK(hung-at-exit)" ;;   # success -- see header
      *)   note="rc=$RC" ;;
    esac
    if [[ "$done_dates" -ge "$want" ]]; then verdict="COMPLETE"
    else verdict="PARTIAL($done_dates/$want dates)"; fi
    line="$verdict  $era $ym  $note  $(( EL/60 ))m$(( EL%60 ))s"
    echo "  $line"; echo "$line" >> "$SUMMARY"
  done
done

echo
echo "======================================================================"
echo "  ${RUN}z WAVES DONE — summary: $SUMMARY"
grep -c COMPLETE "$SUMMARY" 2>/dev/null | sed 's/^/  months COMPLETE: /'
grep -c PARTIAL  "$SUMMARY" 2>/dev/null | sed 's/^/  months PARTIAL : /'
echo "  verify: gsutil ls '$BUCKET/*/*/*/${RUN}z/*.parquet' | wc -l"
echo "======================================================================"
