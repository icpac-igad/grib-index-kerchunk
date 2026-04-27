#!/usr/bin/env bash
# ==============================================================================
# GEFS Backfill — single run-hour, date range configurable
# ==============================================================================
# Processes one run-hour at a time in batches of MAX_WORKERS.
# Designed to be launched as 4 parallel nohup processes (one per run-hour).
#
# Required env var:
#   GCS_BUCKET    Name of your GCS bucket (no gs:// prefix). Output prefix is
#                 gs://${GCS_BUCKET}/run_par_gefs/{Y}/{M}/{D}/{run}z/
#
# Usage:
#   export GCS_BUCKET=your-bucket-name
#   bash run_backfill_single_run.sh --run 00 --start 20200925 --end 20201231
#   bash run_backfill_single_run.sh --run 06 --start 20210101 --end 20211231
#
# Logs: logs/backfill/{run}z_YYYYMMDDTHHMMSSZ.log
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${GCS_BUCKET:?ERROR: set GCS_BUCKET env var to your bucket name}"

MAX_WORKERS=20
DRY_RUN=false
RUN=""
START_DATE=""
END_DATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)       RUN="$2"; shift ;;
        --start)     START_DATE="$2"; shift ;;
        --end)       END_DATE="$2"; shift ;;
        --workers)   MAX_WORKERS="$2"; shift ;;
        --dry-run)   DRY_RUN=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

[[ -z "$RUN" ]]        && { echo "ERROR: --run required (00|06|12|18)"; exit 1; }
[[ -z "$START_DATE" ]] && { echo "ERROR: --start required (YYYYMMDD)"; exit 1; }
[[ -z "$END_DATE" ]]   && { echo "ERROR: --end required (YYYYMMDD)"; exit 1; }

LOG_DIR="$SCRIPT_DIR/logs/backfill"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG="$LOG_DIR/${RUN}z_${TIMESTAMP}.log"

echo "======================================================================"
echo "  GEFS Backfill — ${RUN}z  ($START_DATE to $END_DATE)"
echo "======================================================================"
echo "  Bucket  : gs://${GCS_BUCKET}"
echo "  Workers : $MAX_WORKERS per batch"
echo "  Log     : $RUN_LOG"
[[ "$DRY_RUN" == "true" ]] && echo "  Mode    : DRY RUN"
echo "======================================================================"

ALL_DATES=$(python3 -c "
from datetime import date, timedelta
s = '$START_DATE'; e = '$END_DATE'
d = date(int(s[:4]),int(s[4:6]),int(s[6:]))
end = date(int(e[:4]),int(e[4:6]),int(e[6:]))
dates = []
while d <= end:
    dates.append(d.strftime('%Y%m%d'))
    d += timedelta(days=1)
print(' '.join(dates))
")

echo "Checking GCS for ${RUN}z..."
MISSING=()
for D in $ALL_DATES; do
    Y=${D:0:4}; M=${D:4:2}
    N=$(gsutil ls "gs://${GCS_BUCKET}/run_par_gefs/${Y}/${M}/${D}/${RUN}z/" 2>/dev/null | grep -c parquet || true)
    if [[ "$N" -ge 30 ]]; then
        echo "  SKIP $D ${RUN}z ($N parquets)"
    else
        echo "  QUEUE $D ${RUN}z ($N parquets)"
        MISSING+=("$D")
    fi
done

echo ""
echo "  ${RUN}z: ${#MISSING[@]} missing"
echo ""

[[ ${#MISSING[@]} -eq 0 ]] && { echo "All ${RUN}z dates complete."; exit 0; }
[[ "$DRY_RUN" == "true" ]] && { echo "DRY RUN — would process: ${MISSING[*]}"; exit 0; }

# Split into batches of MAX_WORKERS
BATCH=(); BATCH_NUM=0; GRAND_OK=0; GRAND_FAIL=0

flush_batch() {
    [[ ${#BATCH[@]} -eq 0 ]] && return
    (( BATCH_NUM++ )) || true
    DATES_ARG="${BATCH[*]}"
    echo "  Dispatching batch $BATCH_NUM (${#BATCH[@]} dates, ${RUN}z)..."
    BATCH_START=$(date +%s)
    uv run _run_batch_dates.py "$DATES_ARG" "$MAX_WORKERS" "$RUN" 2>&1 | tee -a "$RUN_LOG"
    BATCH_END=$(date +%s)
    OK=$(grep -c "^OK " "$RUN_LOG" 2>/dev/null || true)
    FAIL=$(grep -c "^FAIL " "$RUN_LOG" 2>/dev/null || true)
    echo "  Batch $BATCH_NUM done in $(( (BATCH_END-BATCH_START)/60 ))m$(( (BATCH_END-BATCH_START)%60 ))s"
    BATCH=()
}

for D in "${MISSING[@]}"; do
    BATCH+=("$D")
    [[ ${#BATCH[@]} -eq $MAX_WORKERS ]] && flush_batch
done
flush_batch

OK=$(grep -c "^OK " "$RUN_LOG" 2>/dev/null || true)
FAIL=$(grep -c "^FAIL " "$RUN_LOG" 2>/dev/null || true)
echo ""
echo "======================================================================"
echo "  ${RUN}z COMPLETE — ok=$OK fail=$FAIL"
echo "======================================================================"
