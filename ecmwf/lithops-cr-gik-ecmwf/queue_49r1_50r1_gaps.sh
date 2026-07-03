#!/usr/bin/env bash
# ==============================================================================
# Queue the two small remaining gaps AFTER the 0p4 wave finishes.
# ==============================================================================
# The 0p4 backfill wave re-reads lithops_config.yaml (runtime: tag) fresh every
# month, so we MUST NOT flip the runtime while it runs. This script blocks on the
# wave PID, then fills each gap sequentially with the correct per-era runtime +
# driver env:
#
#   49r1  -> 20240229           (:49r1 image, enfo/0p25, ref 20250515)
#            the single 0p25 date on the 0p4->0p25 boundary that both catalogs miss
#   50r1  -> 20260701,20260702  (:50r1 image, oper/0p25, ref 20260513)
#            the rolling daily tail
#
# All writes go to the FIXED catalog (GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf).
# Final config state is left at :50r1 (the current era for daily ops).
#
# Usage:  WAVE_PID=<pid> bash queue_49r1_50r1_gaps.sh    (WAVE_PID optional)
# ==============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CFG=lithops_config.yaml
LOG_DIR="$SCRIPT_DIR/logs/gaps"; mkdir -p "$LOG_DIR"
export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"

set_runtime () {  # $1 = tag (49r1|50r1|0p4)
  sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$1|" "$CFG"
  echo "  lithops_config.yaml runtime -> :$1  ($(grep -E 'runtime: gcr' "$CFG" | tr -d ' '))"
}

# proc_live PID -> true only if the PID exists AND is not a zombie.
# NOTE: `kill -0` succeeds on a <defunct> (zombie) process, so a bare
# `while kill -0 PID` loop hangs forever once the wave becomes a zombie that
# init hasn't reaped. Gate on the process STATE instead (Z* = zombie).
proc_live () {
  local pid="$1" st
  [[ -n "$pid" ]] || return 1
  st="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')" || return 1
  [[ -n "$st" && "$st" != Z* ]]
}

# 1) Wait for the 0p4 wave to finish.
# Resolve the wave PID ONCE (explicit WAVE_PID hint, else discover the live
# run_backfill via a one-shot pgrep), then poll with proc_live — which is
# zombie-safe, so it works whether the hint is the run_backfill itself or its
# parent wrapper (which may end up <defunct>). We deliberately do NOT poll with
# `pgrep -f` in the loop: -f matches on argv and can self-match any shell whose
# command line happens to contain the pattern.
WAVE_PID="${WAVE_PID:-$(pgrep -f 'run_backfill_00z.sh --era 0p4' | head -1)}"
if [[ -n "${WAVE_PID:-}" ]] && proc_live "$WAVE_PID"; then
  echo "[queue] waiting for 0p4 wave PID $WAVE_PID (zombie-safe proc_live) ..."
  while proc_live "$WAVE_PID"; do sleep 30; done
else
  echo "[queue] no live 0p4 wave found; assuming it is already done."
fi
echo "[queue] 0p4 wave finished; draining any child run_lithops ..."
while pgrep -f 'run_lithops_ecmwf.py' >/dev/null 2>&1; do sleep 20; done

echo "=============================================================="
echo "[queue] filling 49r1 gap: 20240229"
echo "=============================================================="
set_runtime 49r1
export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz"
uv run run_lithops_ecmwf.py --start-date 20240229 --end-date 20240229 \
    --run 00 --max-workers 4 --yes 2>&1 | tee "$LOG_DIR/49r1_20240229.log"

echo "=============================================================="
echo "[queue] filling 50r1 tail: 20260701 -> 20260702"
echo "=============================================================="
set_runtime 50r1
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz"
uv run run_lithops_ecmwf.py --start-date 20260701 --end-date 20260702 \
    --run 00 --max-workers 4 --yes 2>&1 | tee "$LOG_DIR/50r1_20260701-02.log"

echo "=============================================================="
echo "[queue] done. Final runtime tag:"
grep -E 'runtime: gcr' "$CFG"
echo "[queue] gap logs in $LOG_DIR/"
