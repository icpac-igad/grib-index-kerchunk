#!/usr/bin/env bash
# ==============================================================================
# ECMWF 00Z Backfill — 2024-03-01 to 2025-12-31
# ==============================================================================
#
# Processes one calendar month at a time (28-31 dates per batch).
# With max_workers=35 in lithops_config.yaml every month fits in one Cloud Run
# wave (~8.5 min), avoiding the cold-start HTTP 500 race.
#
# Retry logic is handled inside run_lithops_ecmwf.py (up to 3 attempts, 30s
# delay between retries).
#
# Usage:
#   bash run_backfill_00z.sh                   # full 2024-03 → 2025-12
#   bash run_backfill_00z.sh --dry-run         # preview months/dates only
#   bash run_backfill_00z.sh --from 2024-09    # resume from a specific month
#   bash run_backfill_00z.sh --from 2025-06 --to 2025-08   # sub-range
#
# Logs:
#   logs/backfill_00z_YYYY-MM.log   per-month log
#   logs/backfill_00z_summary.log   one-line per month (OK / FAIL)
#
# Cost:  671 dates × $0.026 ≈ $17.45 USD
# Wall:  22 months × ~8.5 min ≈ 3.1 hrs  (sequential)
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/backfill_00z"
mkdir -p "$LOG_DIR"

# --- Defaults ---
FROM_MONTH="2024-03"
TO_MONTH="2025-12"
DRY_RUN_FLAG=""
MAX_WORKERS=35
ERA="49r1"

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN_FLAG="--dry-run" ;;
        --from)      FROM_MONTH="$2";  shift ;;
        --to)        TO_MONTH="$2";    shift ;;
        --workers)   MAX_WORKERS="$2"; shift ;;
        --era)       ERA="$2";         shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done


# --- Per-era env (MUST match the DEPLOYED Cloud Run image's era) -------------
# Era selects template + source path + control stream. Default 49r1 (the
# 13-level era covers the 2024-2025 backfill range, incl. MAM 2025/2026).
ERA="${ERA:-49r1}"
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"
case "$ERA" in
  49r1) export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
        export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz" ;;
  50r1) export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
        export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz" ;;
  0p4)  export ECMWF_REFERENCE_DATE=20230601 ECMWF_RESOLUTION=0p4  ECMWF_CONTROL_STREAM=enfo
        export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz" ;;
  *) echo "Unknown --era: $ERA (use 49r1|50r1|0p4)"; exit 1 ;;
esac
echo "  Era        : $ERA  (ref $ECMWF_REFERENCE_DATE, res $ECMWF_RESOLUTION, control $ECMWF_CONTROL_STREAM)"

# Build ordered list of YYYY-MM months between FROM and TO (inclusive)
mapfile -t MONTHS < <(python3 -c "
from datetime import date
import sys
start_y, start_m = map(int, '${FROM_MONTH}'.split('-'))
end_y,   end_m   = map(int, '${TO_MONTH}'.split('-'))
y, m = start_y, start_m
while (y, m) <= (end_y, end_m):
    print(f'{y:04d}-{m:02d}')
    m += 1
    if m > 12:
        m = 1; y += 1
")

# --- Summary header ---
echo "======================================================================"
echo "  ECMWF 00Z Backfill"
echo "======================================================================"
echo "  Range      : $FROM_MONTH → $TO_MONTH"
echo "  Months     : ${#MONTHS[@]}"
echo "  Max workers: $MAX_WORKERS  (maxScale:35 on Cloud Run)"
echo "  Logs       : $LOG_DIR/"
if [[ -n "$DRY_RUN_FLAG" ]]; then
    echo "  Mode       : DRY RUN"
else
    echo "  Mode       : LIVE"
fi
echo "======================================================================"
echo ""

SUMMARY_LOG="$LOG_DIR/summary.log"
OVERALL_START=$(date +%s)
MONTH_OK=0
MONTH_FAIL=0

for YM in "${MONTHS[@]}"; do
    YEAR="${YM%-*}"
    MONTH="${YM#*-}"

    # Compute first and last day of the month
    START_DATE="${YEAR}${MONTH}01"
    END_DATE=$(python3 -c "
import calendar
print(f'{${YEAR}}{${MONTH}:02d}{calendar.monthrange(${YEAR}, ${MONTH})[1]:02d}')
" 2>/dev/null || date -d "${YEAR}-${MONTH}-01 +1 month -1 day" +%Y%m%d)

    N_DAYS=$(( ( $(date -d "${YEAR}-${MONTH}-01 +1 month" +%s) - $(date -d "${YEAR}-${MONTH}-01" +%s) ) / 86400 ))

    echo "----------------------------------------------------------------------"
    echo "  Month: $YM  ($START_DATE → $END_DATE, $N_DAYS dates)"
    echo "----------------------------------------------------------------------"

    MONTH_LOG="$LOG_DIR/backfill_00z_${YM}.log"
    MONTH_START=$(date +%s)

    if uv run run_lithops_ecmwf.py \
        --start-date "$START_DATE" \
        --end-date   "$END_DATE" \
        --run        "00" \
        --max-workers "$MAX_WORKERS" \
        --yes \
        $DRY_RUN_FLAG \
        2>&1 | tee "$MONTH_LOG"; then

        MONTH_END=$(date +%s)
        ELAPSED=$(( MONTH_END - MONTH_START ))
        STATUS="OK"
        MONTH_OK=$(( MONTH_OK + 1 ))
    else
        MONTH_END=$(date +%s)
        ELAPSED=$(( MONTH_END - MONTH_START ))
        STATUS="FAIL"
        MONTH_FAIL=$(( MONTH_FAIL + 1 ))
        echo "" >&2
        echo "  ERROR: $YM failed — check $MONTH_LOG" >&2
    fi

    ELAPSED_FMT="$(( ELAPSED / 60 ))m$(( ELAPSED % 60 ))s"
    echo "$STATUS  $YM  ($N_DAYS dates)  ${ELAPSED_FMT}" | tee -a "$SUMMARY_LOG"
    echo ""
done

OVERALL_END=$(date +%s)
TOTAL=$(( OVERALL_END - OVERALL_START ))

echo "======================================================================"
echo "  BACKFILL COMPLETE"
echo "======================================================================"
echo "  Months OK   : $MONTH_OK / ${#MONTHS[@]}"
echo "  Months FAIL : $MONTH_FAIL"
echo "  Total time  : $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
echo ""
echo "  Summary log : $SUMMARY_LOG"
echo "  Verify GCS  :"
echo "    gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/2024/"
echo "    gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/2025/"
echo "======================================================================"

[[ "$MONTH_FAIL" -eq 0 ]]
