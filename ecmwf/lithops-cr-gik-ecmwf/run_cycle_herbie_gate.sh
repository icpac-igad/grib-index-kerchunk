#!/usr/bin/env bash
# ==============================================================================
# Per-cycle Herbie verification gate — 3 eras x 1 date, for 06z / 12z / 18z.
# ==============================================================================
# Mirrors ecmwf/run_random_3era_herbie_eval.sh (which covers 00z) for the other
# three cycles. Two stages:
#
#   Stage 1  build one validation date per era at the target cycle, flipping the
#            lithops runtime tag between eras (the two-switch rule -- see
#            ECMWF_00Z_BACKFILL_SUMMARY.md section 4). STRICTLY SEQUENTIAL: all
#            eras share one lithops_config.yaml, so they cannot overlap.
#   Stage 2  GIK-vs-Herbie intercomparison on each, at T+0 and at a late step.
#
# Dates match run_random_3era_herbie_eval.sh's 00z picks so results are directly
# comparable across cycles.
#
# Usage:  bash run_cycle_herbie_gate.sh 06 [--skip-build]
#
# PASS: 49r1/50r1 r >= 0.9999 ; 0p4 r >= 0.9997 (grid-reindex residual).
# A miss means STOP -- do not release the wave.
# ==============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

RUN="${1:?usage: run_cycle_herbie_gate.sh <00|06|12|18> [--skip-build]}"
SKIP_BUILD="${2:-}"
CFG=lithops_config.yaml
GIK=/data/08-2023/working_notes_jupyter/ignore_nka_gitrepos/pam_team/grib-index-kerchunk/ecmwf
BUCKET=gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf
LOG_DIR="$SCRIPT_DIR/logs/herbie_gate_${RUN}z"; mkdir -p "$LOG_DIR"

# 06z/18z are short runs (0-144h); 00z/12z reach 360h. Validate at T+0 AND at the
# last step of the axis -- a T+0-only check passes even when the axis is wrong.
case "$RUN" in
  00|12) LATE_STEP=240 ;;
  06|18) LATE_STEP=144 ;;
  *) echo "FATAL: run must be 00|06|12|18" >&2; exit 1 ;;
esac

export UV_PYTHON=3.12
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
export AWS_NO_SIGN_REQUEST=YES
export GIK_GCS_KEY="$SCRIPT_DIR/service_account/ecmwf-lithops-deployer-key.json"
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"

# era:date:grid  -- same picks as the 00z eval
PICKS="0p4:20230318:0p4 49r1:20240327:0p25 50r1:20260621:0p25"

set_rt () {
  sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$1|" "$CFG"
  echo "  runtime -> :$1  ($(grep -E 'runtime: gcr' "$CFG" | tr -d ' '))"
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

# ---------------------------------------------------------------- stage 1 ----
if [[ "$SKIP_BUILD" != "--skip-build" ]]; then
for p in $PICKS; do
  era="${p%%:*}"; rest="${p#*:}"; d="${rest%%:*}"
  y="${d:0:4}"; m="${d:4:2}"
  echo "=============================================================="
  echo "  build  $era  $d  ${RUN}z"
  echo "=============================================================="
  set_rt "$era"; era_env "$era"

  # exit 124 = hang-at-exit = success (gotcha 1); verify via GCS below.
  timeout 1500 uv run run_lithops_ecmwf.py \
      --start-date "$d" --end-date "$d" --run "$RUN" --max-workers 4 --yes \
      2>&1 | tee "$LOG_DIR/build_${era}_${d}.log" | tail -5

  n=$(gsutil ls "$BUCKET/$y/$m/$d/${RUN}z/**" 2>/dev/null | wc -l)
  if [[ "$n" -ne 51 ]]; then
      echo "FATAL: $era $d ${RUN}z wrote $n/51 files -- STOP, do not release the wave." >&2
      exit 2
  fi
  echo "  OK  $era $d ${RUN}z  51/51"
done
fi

# ---------------------------------------------------------------- stage 2 ----
echo
echo "=============================================================="
echo "  GIK vs Herbie — ${RUN}z, T+0 and T+${LATE_STEP}h"
echo "=============================================================="
cd "$GIK"
OUT="gik_vs_herbie/cycle_${RUN}z_eval"
for p in $PICKS; do
  era="${p%%:*}"; rest="${p#*:}"; d="${rest%%:*}"; grid="${rest##*:}"
  y="${d:0:4}"; m="${d:4:2}"
  for step in 0 "$LATE_STEP"; do
    echo "######## $era $d ${RUN}z  T+${step}h  (grid $grid) ########"
    uv run compare_gik_herbie_pressure.py \
        --grid "$grid" \
        --gcs-path "$BUCKET/$y/$m/$d/${RUN}z" \
        --date "$d" --run "$RUN" --step "$step" \
        --var t --levels 500,850 \
        --output-dir "$OUT" 2>&1 \
      | grep -vE "Downloading|Installed|Resolved|Prepared|Building|Built|Audited|Bytecode|Downloaded|Created a default config|view/edit|config.toml|^ *╭|^ *│|^ *╰"
  done
done
echo "######## GATE COMPLETE — ${RUN}z ########"
echo "PASS thresholds: 49r1/50r1 r >= 0.9999 | 0p4 r >= 0.9997"
ls -1 "$OUT"/*.png 2>/dev/null | sed 's/^/plot: /'
