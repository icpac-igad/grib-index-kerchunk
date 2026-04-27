#!/usr/bin/env bash
# ==============================================================================
# GEFS Multi-Year Backfill Chain
# Runs date ranges sequentially. Edit the run_range calls at the bottom to
# configure the sequence. Usage:
#   export GCS_BUCKET=your-bucket-name
#   nohup bash run_backfill_chain.sh [WAIT_PID] > logs/backfill/chain.out 2>&1 &
#
# NOAA GEFS reforecast/realtime archive starts 2020-09-25; earlier dates
# return no data.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${GCS_BUCKET:?ERROR: set GCS_BUCKET env var to your bucket name}"

WAIT_PID="${1:-}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Wait for an existing PID to finish, ignoring zombies
if [[ -n "$WAIT_PID" ]]; then
    log "Waiting for PID $WAIT_PID to finish..."
    while [[ -f "/proc/$WAIT_PID/status" ]] && ! grep -q "^State:.*Z" "/proc/$WAIT_PID/status" 2>/dev/null; do
        sleep 60
    done
    log "PID $WAIT_PID finished."
fi

run_range() {
    local START="$1"
    local END="$2"
    local OUTFILE="logs/backfill/00z_${START}_${END}.out"
    log "========================================================"
    log "  Running $START → $END"
    log "========================================================"
    bash run_backfill_single_run.sh --run 00 --start "$START" --end "$END" --workers 20 \
        > "$OUTFILE" 2>&1
    log "Done: $START → $END. Log: $OUTFILE"
}

# Edit these to your desired backfill range. The example below covers the
# full GEFS archive (2020-09-25 onwards) at 00z; copy / adapt for other
# run-hours by changing --run inside run_range and the OUTFILE prefix.
run_range 20200925 20201231   # NOAA archive starts 2020-09-25
run_range 20210101 20211231
run_range 20220101 20221231
run_range 20230101 20231231
run_range 20240101 20241231
run_range 20250101 20251231

log "All ranges complete."
