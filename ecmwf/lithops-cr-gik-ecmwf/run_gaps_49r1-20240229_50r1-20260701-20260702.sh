#!/usr/bin/env bash
# Direct gap fills — run only after the 0p4 wave is confirmed DONE (no wait loop).
# 49r1 20240229 (:49r1, enfo) then 50r1 20260701-20260702 (:50r1, oper).
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$D"
CFG=lithops_config.yaml; LOG="$D/logs/gaps"; mkdir -p "$LOG"
export UV_PYTHON=3.12 GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf AWS_NO_SIGN_REQUEST=YES
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"
set_rt(){ sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$1|" "$CFG"; echo "runtime -> :$1"; }

echo "=== 49r1 gap: 20240229 ==="
set_rt 49r1
export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz"
uv run run_lithops_ecmwf.py --start-date 20240229 --end-date 20240229 --run 00 --max-workers 4 --yes 2>&1 | tee "$LOG/49r1_20240229.log"

echo "=== 50r1 tail: 20260701 -> 20260702 ==="
set_rt 50r1
export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz"
uv run run_lithops_ecmwf.py --start-date 20260701 --end-date 20260702 --run 00 --max-workers 4 --yes 2>&1 | tee "$LOG/50r1_20260701-02.log"

echo "=== done. final runtime: $(grep -E 'runtime: gcr' "$CFG" | tr -d ' ') ==="
