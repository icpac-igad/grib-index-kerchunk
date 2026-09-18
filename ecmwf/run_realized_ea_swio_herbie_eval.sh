#!/usr/bin/env bash
# Published realized EA-SWIO store vs Herbie — every sub-store, anonymous read.
#
#   https://source.coop/e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3
#
# Four cases per sub-store, chosen to vary the things that can go wrong
# independently of each other: pressure-level vs surface vs accumulated channel,
# analysis step vs short vs long lead, and dates spread across each MAM window.
#
# IMPORTANT -- 49r1-mam2024 is only PARTLY published: its time axis is
# preallocated to all 92 MAM dates but only 33 are written (2024-03-01 ..
# 2024-04-02). An unwritten date is on the axis and reads as all-NaN, so it
# does not raise -- it silently compares against nothing. The dates below are
# all confirmed written (`--list`, and coverage.json for the per-date map).
# The final section covers two 2024 dates the realized store never wrote by
# comparing the VIRTUAL store over the same EA-SWIO box instead.
#
#   PY=/path/to/python ./run_realized_ea_swio_herbie_eval.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"          # ecmwf/

PY=${PY:-uv run}
OUT=${OUT:-gik_vs_herbie/realized_ea_swio_eval}
export AWS_NO_SIGN_REQUEST=YES

run() {                                       # run <sub> <date> <step> <var>
  echo "############ $1  $2  T+$3h  $4 ############"
  $PY compare_realized_herbie.py --sub "$1" --date "$2" --step "$3" --var "$4" \
      --output-dir "$OUT" 2>&1 \
    | grep -vE "Downloading|obstore|external_backend|Found ┊|Installed|Resolved|Prepared|Audited"
}

# MAM 2024 — pressure level, accumulated, analysis step, surface.
# Confined to the written window (2024-03-01 .. 2024-04-02).
run 49r1-mam2024      20240327   48  t850
run 49r1-mam2024      20240315  120  tp
run 49r1-mam2024      20240320    0  u700
run 49r1-mam2024      20240402   24  t2m

# MAM 2025 — geopotential, long lead, humidity, mean sea-level pressure
run 49r1-mam2025      20250315   48  gh500
run 49r1-mam2025      20250420  240  tp
run 49r1-mam2025      20250510   72  q925
run 49r1-mam2025      20250501    0  msl

# MAM 2026 (49r1, up to the 05-12 era boundary)
run 49r1-mam2026      20260310   48  t850
run 49r1-mam2026      20260401   96  tcwv
run 49r1-mam2026      20260415   24  v850
run 49r1-mam2026      20260512    0  u10

# MAM 2026 tail (50r1, the dual-stream era)
run 50r1-mam2026-tail 20260515   48  t850
run 50r1-mam2026-tail 20260520  120  tp
run 50r1-mam2026-tail 20260525   72  vo850
run 50r1-mam2026-tail 20260531    0  skt

# Dates the realized store never wrote, covered from the VIRTUAL store over the
# same EA-SWIO box. Same domain, same 321x261 grid, same per-member test -- so a
# gap in the published subset does not become a gap in the validation.
compensate() {                                # compensate <date> <step> <var> <levels>
  echo "############ COMPENSATE (virtual v4)  $1  T+$2h  $3 ${4:-} ############"
  $PY compare_icechunk_herbie.py \
      --store "${STORE:-gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4}" \
      --era 49r1 --run 00 --date "$1" --step "$2" --var "$3" ${4:+--levels "$4"} \
      --lon-min 15 --lon-max 80 --lat-min -40 --lat-max 40 --tag _easwio \
      --sa-key "${SA_KEY:-/tmp/frisky-ea/gcs-key.json}" --output-dir "$OUT" 2>&1 \
    | grep -vE "Downloading|obstore|external_backend|Found ┊"
}

compensate 20240501    0  u  700
compensate 20240520   24  t2m

echo "############ ALL DONE ############"
