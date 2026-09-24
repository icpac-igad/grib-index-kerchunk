#!/usr/bin/env bash
# Cross-check harness for GIK_VS_HERBIE_AUDIT.md.
#
#   HERBIE_SRC=<path-to-Herbie-clone>/src ./run_all.sh [outdir]
#
# Needs: eccodes, cfgrib, gribberish, xarray, icechunk, cloudpickle, and a
# Herbie checkout (not pip-installed -- 03/04/05 read its source as evidence).
# Network: ecmwf-forecasts.s3 (anonymous) and data.source.coop (anonymous).
set -u
PY="${PY:-python3}"
OUT="${1:-results}"
: "${HERBIE_SRC:?set HERBIE_SRC to <Herbie-clone>/src}"
mkdir -p "$OUT"
for s in 01_decode_single 02_decode_threaded 03_member_axis \
         04_dag_schedulability 05_head_to_head 06_recompute_5_4 \
         07_bytes_on_wire; do
  [ -f "$s.py" ] || { echo "skip $s (absent)"; continue; }
  echo "=== $s ==="
  HERBIE_SRC="$HERBIE_SRC" "$PY" "$s.py" >"$OUT/$s.txt" 2>&1
  echo "    exit=$? -> $OUT/$s.txt"
done
