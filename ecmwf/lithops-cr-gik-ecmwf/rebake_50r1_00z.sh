#!/usr/bin/env bash
# ==============================================================================
# 50r1 per-level 00z run — 2026-05-13 -> 2026-06-30  (oper control, 14-level)
# ==============================================================================
# 50r1 starts at 2026-05-13, immediately after the 49r1 cutover (..05-12 stays
# 49r1/enfo, already in the catalog). Writes into the SAME versioned catalog
# v20260623_run_par_ecmwf/ (default GCS_PARQUET_PREFIX), extending it by date —
# no overlap with 49r1's <=05-12.
#
# Two waves (day-granular so May starts at the 13th, not the 1st):
#   A) 20260513 -> 20260531   (19 dates)
#   B) 20260601 -> 20260624   (24 dates)
#   C) 20260625 -> 20260630   (6 dates)
#
# PREREQUISITE: the :50r1 runtime image is built + deployed and
# lithops_config.yaml runtime: points at :50r1. (Done 2026-06-26.)
#
# This host runs Python 3.11; the runtime is 3.12 + lithops==3.6.4, and lithops
# enforces a host<->runtime version match — so UV_PYTHON=3.12 is mandatory.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/backfill_00z"
mkdir -p "$LOG_DIR"

export UV_PYTHON=3.12

# --- 50r1 era env (must match the deployed :50r1 image) ---
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz"

MAX_WORKERS=35
SUMMARY_LOG="$LOG_DIR/summary_50r1.log"
# On this host the driver completes the work + prints its final output, then
# HANGS at interpreter exit (a lingering lithops thread). Wrap each wave in
# `timeout` so the next wave still runs; exit 124 means "work done, hung at exit"
# — NOT a failure. Always confirm via per-date GCS file counts (51/date).
WAVE_TIMEOUT=1500

run_wave () {
    local LABEL="$1" START="$2" END="$3"
    local WAVE_LOG="$LOG_DIR/backfill_00z_50r1_${LABEL}.log"
    echo "----------------------------------------------------------------------"
    echo "  50r1 wave $LABEL : $START -> $END"
    echo "----------------------------------------------------------------------"
    local T0; T0=$(date +%s)
    set +e
    timeout "$WAVE_TIMEOUT" uv run run_lithops_ecmwf.py \
        --start-date "$START" --end-date "$END" \
        --run 00 --max-workers "$MAX_WORKERS" --yes \
        2>&1 | tee "$WAVE_LOG"
    local RC=${PIPESTATUS[0]}
    set -e
    local STATUS
    case "$RC" in
        0)   STATUS=OK ;;
        124) STATUS="OK(hung-at-exit,killed)" ;;
        *)   STATUS="FAIL(rc=$RC)" ;;
    esac
    local T1; T1=$(date +%s); local EL=$(( T1 - T0 ))
    echo "$STATUS  50r1 $LABEL  ($START->$END)  $(( EL/60 ))m$(( EL%60 ))s" | tee -a "$SUMMARY_LOG"
}

echo "======================================================================"
echo "  ECMWF 50r1 00Z run  (ref $ECMWF_REFERENCE_DATE, control $ECMWF_CONTROL_STREAM)"
echo "  Catalog: gs://gik-ecmwf-aws-tf/${GCS_PARQUET_PREFIX:-v20260623_run_par_ecmwf}/"
echo "======================================================================"

run_wave "2026-05b" 20260513 20260531
run_wave "2026-06"  20260601 20260624
run_wave "2026-06b" 20260625 20260630

echo "======================================================================"
echo "  50r1 00Z run complete — summary: $SUMMARY_LOG"
echo "======================================================================"
