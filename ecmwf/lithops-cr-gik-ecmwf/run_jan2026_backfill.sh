#!/usr/bin/env bash
# ==============================================================================
# January 2026 ECMWF Backfill - All four model runs
# ==============================================================================
#
# Processes all dates in January 2026 (2026-01-01 to 2026-01-31) for each of
# the four ECMWF ensemble forecast runs: 00Z, 06Z, 12Z, 18Z.
#
# GCS output layout (new format):
#   gs://gik-ecmwf-aws-tf/run_par_ecmwf/2026/01/YYYYMMDD/{run}z/
#
# Usage:
#   bash run_jan2026_backfill.sh
#   bash run_jan2026_backfill.sh --dry-run        # show what would run, no execution
#   bash run_jan2026_backfill.sh --runs "00 12"   # subset of runs
#
# Requires:
#   - uv (https://docs.astral.sh/uv/)
#   - lithops_config.yaml in same directory
#   - service_account/ecmwf-lithops-deployer-key.json
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration ---
START_DATE="20260101"
END_DATE="20260131"
RUNS=("00" "06" "12" "18")   # all four model runs
MAX_WORKERS=20                 # Cloud Run parallel workers (max per lithops_config.yaml)

# --- Parse optional args ---
DRY_RUN_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN_FLAG="--dry-run" ;;
        --runs)    shift; IFS=' ' read -r -a RUNS <<< "$1" ;;
    esac
done

# --- Summary ---
echo "======================================================================"
echo "  ECMWF January 2026 Backfill"
echo "======================================================================"
echo "  Start date   : $START_DATE"
echo "  End date     : $END_DATE"
echo "  Runs (UTC)   : ${RUNS[*]}"
echo "  Max workers  : $MAX_WORKERS"
echo "  GCS prefix   : gs://gik-ecmwf-aws-tf/run_par_ecmwf/2026/01/"
if [ -n "$DRY_RUN_FLAG" ]; then
    echo "  Mode         : DRY RUN (no processing)"
else
    echo "  Mode         : LIVE"
fi
echo "======================================================================"
echo ""

# Calculate total jobs: 31 dates × 4 runs = 124 jobs
N_DATES=$(( ( $(date -d "$END_DATE" +%s) - $(date -d "$START_DATE" +%s) ) / 86400 + 1 ))
N_RUNS=${#RUNS[@]}
echo "  Total jobs   : ${N_DATES} dates × ${N_RUNS} runs = $(( N_DATES * N_RUNS )) Lithops invocations"
echo "  Est. cost    : ~\$$(echo "scale=2; ${N_DATES} * ${N_RUNS} * 0.026" | bc) USD"
echo "  Est. wall    : ~$(( (N_DATES + MAX_WORKERS - 1) / MAX_WORKERS * 8 * N_RUNS )) min (sequential runs)"
echo ""

OVERALL_START=$(date +%s)
SUCCESS_RUNS=0
FAILED_RUNS=0

for run in "${RUNS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "  Processing run: ${run}Z  ($START_DATE → $END_DATE)"
    echo "----------------------------------------------------------------------"

    RUN_START=$(date +%s)

    if uv run run_lithops_ecmwf.py \
        --start-date "$START_DATE" \
        --end-date   "$END_DATE" \
        --run        "$run" \
        --max-workers "$MAX_WORKERS" \
        --yes \
        $DRY_RUN_FLAG; then

        RUN_END=$(date +%s)
        ELAPSED=$(( RUN_END - RUN_START ))
        echo ""
        echo "  Run ${run}Z complete in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        SUCCESS_RUNS=$(( SUCCESS_RUNS + 1 ))
    else
        echo ""
        echo "  ERROR: Run ${run}Z failed (exit code $?)" >&2
        FAILED_RUNS=$(( FAILED_RUNS + 1 ))
    fi

    echo ""
done

OVERALL_END=$(date +%s)
TOTAL=$(( OVERALL_END - OVERALL_START ))

echo "======================================================================"
echo "  BACKFILL COMPLETE"
echo "======================================================================"
echo "  Successful runs : $SUCCESS_RUNS / $N_RUNS"
echo "  Failed runs     : $FAILED_RUNS"
echo "  Total wall time : $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
echo ""
echo "  Verify output:"
echo "    gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/2026/01/"
echo "======================================================================"

if [ "$FAILED_RUNS" -gt 0 ]; then
    exit 1
fi
