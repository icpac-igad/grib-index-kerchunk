#!/usr/bin/env bash
# ================================================================
# run_backfill_18z.sh — Month-by-month ECMWF 18Z parquet backfill
# ================================================================
#
# Processes one month at a time, sequentially, calling
# run_lithops_ecmwf.py with --run 06 for each month.
#
# Usage:
#   bash run_backfill_18z.sh                          # Full: 2024-03 → 2026-01
#   bash run_backfill_18z.sh --from 2025-06            # Resume from 2025-06
#   bash run_backfill_18z.sh --from 2024-10 --to 2024-12
#   bash run_backfill_18z.sh --dry-run                 # Preview only
#   bash run_backfill_18z.sh --workers 20              # Custom worker count
#
# Background run (recommended for long backfills):
#   nohup bash run_backfill_18z.sh > logs/backfill_18z/run_full.log 2>&1 &
#   echo "PID: $!"
#
# Monitor:
#   tail -f logs/backfill_18z/summary.log
#   grep -E "^(OK|FAIL)" logs/backfill_18z/summary.log
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────
FROM_MONTH="2024-03"
TO_MONTH="2026-01"
MAX_WORKERS=35
DRY_RUN=false
RUN_HOUR="18"

# ── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)   FROM_MONTH="$2"; shift 2 ;;
        --to)     TO_MONTH="$2"; shift 2 ;;
        --workers) MAX_WORKERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: bash run_backfill_18z.sh [--from YYYY-MM] [--to YYYY-MM] [--workers N] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Set ADC for gcsfs on orchestrator ───────────────────────────────
export GOOGLE_APPLICATION_CREDENTIALS="$SCRIPT_DIR/service_account/ecmwf-lithops-deployer-key.json"

# ── Logging setup ───────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs/backfill_18z"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.log"

# ── Generate month list ─────────────────────────────────────────────
generate_months() {
    local from_y=${1%-*}
    local from_m=${1#*-}
    local to_y=${2%-*}
    local to_m=${2#*-}

    # Remove leading zeros for arithmetic
    from_m=$((10#$from_m))
    to_m=$((10#$to_m))

    local y=$from_y
    local m=$from_m

    while [[ $y -lt $to_y || ($y -eq $to_y && $m -le $to_m) ]]; do
        printf "%04d-%02d\n" "$y" "$m"
        m=$((m + 1))
        if [[ $m -gt 12 ]]; then
            m=1
            y=$((y + 1))
        fi
    done
}

MONTHS=$(generate_months "$FROM_MONTH" "$TO_MONTH")
MONTH_COUNT=$(echo "$MONTHS" | wc -l)

echo "================================================================"
echo "ECMWF 18Z BACKFILL"
echo "================================================================"
echo "Range:   $FROM_MONTH → $TO_MONTH ($MONTH_COUNT months)"
echo "Workers: $MAX_WORKERS"
echo "Run:     ${RUN_HOUR}Z"
echo "Logs:    $LOG_DIR"
echo ""
echo "Months:"
echo "$MONTHS" | sed 's/^/  /'
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would process $MONTH_COUNT months. Exiting."
    exit 0
fi

# ── Process month by month ──────────────────────────────────────────
TOTAL_OK=0
TOTAL_FAIL=0
BACKFILL_START=$(date +%s)

for MONTH in $MONTHS; do
    YEAR=${MONTH%-*}
    MM=${MONTH#*-}

    # Compute start/end dates for the month
    START_DATE="${YEAR}${MM}01"

    # Last day of month
    if [[ $MM == "02" ]]; then
        # Leap year check
        if (( YEAR % 4 == 0 && (YEAR % 100 != 0 || YEAR % 400 == 0) )); then
            LAST_DAY=29
        else
            LAST_DAY=28
        fi
    elif [[ $MM =~ ^(04|06|09|11)$ ]]; then
        LAST_DAY=30
    else
        LAST_DAY=31
    fi
    END_DATE="${YEAR}${MM}$(printf '%02d' $LAST_DAY)"

    MONTH_LOG="$LOG_DIR/backfill_18z_${MONTH}.log"
    MONTH_START=$(date +%s)

    echo ""
    echo "================================================================"
    echo "Month: $MONTH  ($START_DATE → $END_DATE)  [$(date '+%Y-%m-%d %H:%M:%S')]"
    echo "================================================================"

    # Run the pipeline for this month
    if uv run run_lithops_ecmwf.py \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --run "$RUN_HOUR" \
        --max-workers "$MAX_WORKERS" \
        --yes \
        > "$MONTH_LOG" 2>&1; then

        MONTH_END=$(date +%s)
        MONTH_ELAPSED=$((MONTH_END - MONTH_START))
        MONTH_MIN=$((MONTH_ELAPSED / 60))

        echo "OK  $MONTH  ($LAST_DAY dates)  ${MONTH_MIN}m"
        echo "OK  $MONTH  ($LAST_DAY dates)  ${MONTH_MIN}m  [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$SUMMARY_LOG"
        TOTAL_OK=$((TOTAL_OK + 1))
    else
        MONTH_END=$(date +%s)
        MONTH_ELAPSED=$((MONTH_END - MONTH_START))
        MONTH_MIN=$((MONTH_ELAPSED / 60))

        # Check if it actually succeeded via GCS rescue (exit code 1 but data in GCS)
        if grep -q "GCS verification rescued" "$MONTH_LOG" 2>/dev/null; then
            # Count how many succeeded
            SUCCEEDED=$(grep -oP '\d+/\d+ successful' "$MONTH_LOG" | tail -1 || echo "?/?")
            echo "OK  $MONTH  ($LAST_DAY dates)  ${MONTH_MIN}m  [GCS-rescued: $SUCCEEDED]"
            echo "OK  $MONTH  ($LAST_DAY dates)  ${MONTH_MIN}m  [GCS-rescued: $SUCCEEDED]  [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$SUMMARY_LOG"
            TOTAL_OK=$((TOTAL_OK + 1))
        else
            echo "FAIL  $MONTH  ${MONTH_MIN}m  — check $MONTH_LOG"
            echo "FAIL  $MONTH  ${MONTH_MIN}m  [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$SUMMARY_LOG"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    fi

    # Brief pause between months to let Cloud Run scale down
    sleep 5
done

BACKFILL_END=$(date +%s)
BACKFILL_ELAPSED=$((BACKFILL_END - BACKFILL_START))
BACKFILL_HOURS=$((BACKFILL_ELAPSED / 3600))

echo ""
echo "================================================================"
echo "BACKFILL COMPLETE"
echo "================================================================"
echo "OK:     $TOTAL_OK months"
echo "FAIL:   $TOTAL_FAIL months"
echo "Total:  ${BACKFILL_HOURS}h"
echo "Summary: $SUMMARY_LOG"
echo ""
echo "DONE  $FROM_MONTH → $TO_MONTH  OK=$TOTAL_OK FAIL=$TOTAL_FAIL  ${BACKFILL_HOURS}h  [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$SUMMARY_LOG"
