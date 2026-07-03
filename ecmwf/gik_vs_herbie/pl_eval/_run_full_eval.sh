#!/usr/bin/env bash
# Full pressure-level GIK-vs-Herbie validation for the cGAN-critical fields.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> grib-index-kerchunk/ecmwf
OUT=gik_vs_herbie/pl_eval/out
PAR=gik_vs_herbie/pl_eval/par_local
for VAR in u v gh; do
  echo "############## VAR=$VAR ##############"
  uv run compare_gik_herbie_pressure.py \
      --par-dir "$PAR" --date 20260301 --step 48 \
      --var "$VAR" --levels 300,500,700,850 \
      --output-dir "$OUT" 2>&1 | grep -vE 'File "|^ {4}|\^\^|Downloaded|Installed|Bytecode'
done
echo "############## FULL EVAL DONE ##############"
