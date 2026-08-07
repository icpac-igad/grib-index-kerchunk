#!/usr/bin/env bash
# The two era-boundary dates that the daily-granular era table gets wrong, for
# any of 06z / 12z / 18z.   Usage:  bash fix_cycle_boundary_dates.sh <06|12|18>
# The 0p4->0p25 and 49r1->50r1 cutovers are CYCLE-granular, not date-granular:
#
#   20240228 : 0p4-beta/enfo 404 but ifs/0p25/enfo 200 -- the 0.4 deg stream
#              ends after 20240228 00z, so every later cycle is already 49r1.
#              (At 00z it is still 0p4, which is why the 00z corpus is fine.)
#   20260512 : only 50/51 written under 49r1 (control absent from enfo); the
#              oper control IS published, i.e. the cycle has already moved to
#              50r1 while 00z had not.
#
# Verified identical for 06z, 12z and 18z -- the transition is monotonic through
# the day, so if 06z has switched, 12z and 18z have too.
#
# NOTE also cycle-dependent: the unpublished 0.4 deg window is 20230427..20230502
# at 00z/06z but starts a day earlier (20230426) at 12z/18z. Those are 404 on S3
# and are NOT fixable -- expect 1256 dates at 00z/06z and 1255 at 12z/18z.
set -uo pipefail
RUN="${1:?usage: fix_cycle_boundary_dates.sh <06|12|18>}"
[[ "$RUN" =~ ^(06|12|18)$ ]] || { echo "run must be 06|12|18" >&2; exit 1; }
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$D"
CFG=lithops_config.yaml; LOG="$D/logs/fix_${RUN}z"; mkdir -p "$LOG"
B=gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf
export UV_PYTHON=3.12 GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf AWS_NO_SIGN_REQUEST=YES
_HF="https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main"
set_rt(){ sed -i -E "s|(runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime:)[A-Za-z0-9._-]+|\1$1|" "$CFG"; echo "  runtime -> :$1"; }

run_one(){ # era date
  local era="$1" d="$2" y="${2:0:4}" m="${2:4:2}"
  echo "=== $d ${RUN}z as $era ==="
  set_rt "$era"
  case "$era" in
    49r1) export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
          export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz" ;;
    50r1) export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=oper
          export TEMPLATE_URL="$_HF/gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz" ;;
  esac
  timeout 1500 uv run run_lithops_ecmwf.py --start-date "$d" --end-date "$d" \
      --run "$RUN" --max-workers 4 --yes > "$LOG/${era}_${d}.log" 2>&1
  local n; n=$(gsutil ls "$B/$y/$m/$d/${RUN}z/*.parquet" 2>/dev/null | wc -l)
  echo "  -> $d ${RUN}z now has $n/51 files"
}

run_one 49r1 20240228
run_one 50r1 20260512
echo "=== FIX DONE ==="
