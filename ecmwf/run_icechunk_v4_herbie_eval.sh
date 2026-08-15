#!/usr/bin/env bash
# Icechunk-v4 (00z) vs Herbie intercomparison, two dates per era.
#
# The par-vs-Herbie study (run_random_3era_herbie_eval.sh) validates the pipeline
# INPUT. This validates the store: the axes the conversion labelled, the member it
# filed each field under, and the index it wrote each date at. Same dates as that
# study wherever they exist in the v4 00z groups, so the numbers line up.
#
# Cases 1-4 are clean dates. Case 5 is deliberate: verify_store_completeness
# reports 20240327 ens_24 missing at step 36, so that run must show exactly one
# all-NaN member -- the two tools agreeing on the same gap.
#
#   PY=/path/to/python ./run_icechunk_v4_herbie_eval.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"          # ecmwf/

PY=${PY:-uv run}
STORE=${STORE:-gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4}
SA_KEY=${SA_KEY:-/tmp/frisky-ea/gcs-key.json}
OUT=${OUT:-gik_vs_herbie/icechunk_v4_eval}
export AWS_NO_SIGN_REQUEST=YES

run() {                                       # run <era> <date> <step> <var> <levels>
  echo "############ $1 $2 00z  T+$3h  $4 ${5:-sfc} ############"
  $PY compare_icechunk_herbie.py --store "$STORE" --era "$1" --run 00 \
      --date "$2" --step "$3" --var "$4" ${5:+--levels "$5"} \
      --sa-key "$SA_KEY" --output-dir "$OUT" 2>&1 \
    | grep -vE "Downloading|obstore|external_backend|Found ┊|Installed|Resolved|Prepared|Audited"
  echo "    rc=$?"
}

# 1. pressure levels, analysis step -- same dates/vars as the par-based study
run 0p4  20230318 0  t 500,850
run 0p4  20231112 0  t 500,850
run 49r1 20240327 0  t 500,850
run 49r1 20251125 0  t 500,850
run 50r1 20260621 0  t 500,850
run 50r1 20260701 0  t 500,850

# 2. a forecast step and a different pl variable -- exercises the step axis
run 0p4  20231112 48 u 700
run 49r1 20251125 48 u 700
run 50r1 20260701 48 u 700

# 3. surface, instantaneous
run 0p4  20231112 48 t2m ""
run 49r1 20251125 48 t2m ""
run 50r1 20260701 48 t2m ""

# 4. surface, accumulated (tp is zero at T+0, so only a forecast step is a test)
run 0p4  20231112 48 tp ""
run 49r1 20251125 48 tp ""
run 50r1 20260701 48 tp ""

# 5. known gap: expect exactly one all-NaN member (ens_24)
run 49r1 20240327 36 t 500

echo "############ ALL DONE ############"
