#!/usr/bin/env bash
# Random cross-era GIK-vs-Herbie intercomparison: 2 dates x 3 eras (seed=42).
# var=t at 500 & 850 hPa, analysis step T+0h. Writes plots + stats JSON to
# gik_vs_herbie/random_3era_eval/.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"        # ecmwf/
export GIK_GCS_KEY=/scratch/notebook/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json
export AWS_NO_SIGN_REQUEST=YES
OUT=gik_vs_herbie/random_3era_eval
BUCKET=gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf

# era:date:grid
PICKS="0p4:20230318:0p4 0p4:20231112:0p4 49r1:20240327:0p25 49r1:20251125:0p25 50r1:20260621:0p25 50r1:20260701:0p25"

for p in $PICKS; do
  era="${p%%:*}"; rest="${p#*:}"; d="${rest%%:*}"; grid="${rest##*:}"
  y="${d:0:4}"; m="${d:4:2}"
  echo "############ $era  $d  (grid $grid) ############"
  uv run compare_gik_herbie_pressure.py \
      --grid "$grid" \
      --gcs-path "$BUCKET/$y/$m/$d/00z" \
      --date "$d" --step 0 --var t --levels 500,850 \
      --output-dir "$OUT" 2>&1 | grep -vE "Downloading|Installed|Resolved|Prepared|Building|Built|Audited|Bytecode|Downloaded|Created a default config|view/edit|config.toml|^ *╭|^ *│|^ *╰"
done
echo "############ ALL DONE ############"
ls -1 "$OUT"/*.png 2>/dev/null | sed 's/^/plot: /'
