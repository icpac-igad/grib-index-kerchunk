#!/bin/bash
# Mirror every realized store on Ceph to source.coop, in one credential window.
#
#   source <this folder>/.env
#   tmux new-session -d -s mirror ./mirror_all_to_source_coop.sh
#   tail -f mirror_all.log
#
# The constraint this is built around: source.coop credentials are STS tokens
# with a fixed lifetime (12 h when obtained with `source-coop login --duration
# 12h`). Running past expiry is not fatal -- the mirror resumes on size
# comparison -- but it means an unattended run stops half-done. So before each
# store this checks the credential has enough life left for it, at the MEASURED
# rate, and refuses to start one it cannot finish.
#
# Measured 2026-08-19 on the 50r1 tail: 302.59 GB in 42.6 min across 6 VMs at
# 48 threads = 118.3 MB/s = 426 GB/h. That is the planning number. The earlier
# 599 GB/h benchmark timed only the upload loop against pre-loaded blobs; a
# real run reads every object from Ceph first.
#
# Each store gets up to 3 passes. A pass that leaves errors does NOT upload the
# `repo` pointer, so the store stays visibly incomplete rather than
# readable-but-wrong; the next pass retries just the failures and then writes
# `repo`. On the 50r1 tail that was 25 of 45,229 objects, cleared in 5 s.
#
# Nothing is deleted from Ceph. Deletion is a separate, deliberate step.
set -u

cd "$(dirname "$(readlink -f "$0")")" || exit 1   # run from ecmwf/ewc-realization
P=${P:-.venv/bin/python}   # override: P=/path/to/venv/bin/python
SC=~/.local/bin/source-coop
LOG=mirror_all.log
DASK=tcp://127.0.0.1:8786
RATE_GB_H=426            # measured, not benchmarked
MARGIN_H=0.5             # never start a store this close to expiry
REPO=forecasts/ecmwf-ifs-ea-swio-realized-v3

# sink|approx GB   -- the v3- prefix is dropped at the destination, since the
# repository name already carries it.
STORES=(
  "ea-swio/v3-49r1-mam2026|49r1-mam2026|1162"
  "ea-swio/v3-49r1-mam2025|49r1-mam2025|1490"
  "ea-swio/v3-49r1-mam2024|49r1-mam2024|522"
)

say() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')  $*" | tee -a "$LOG"; }

creds_hours_left() {
    "$SC" creds --format credential-process 2>/dev/null | "$P" -c "
import sys,json,datetime
try:
    d=json.load(sys.stdin)
    e=datetime.datetime.fromisoformat(d['Expiration'].replace('Z','+00:00'))
    print(f\"{(e-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/3600:.2f}\")
except Exception: print('0')"
}

say "=== mirror chain: ${#STORES[@]} stores, ${RATE_GB_H} GB/h assumed ==="
say "credential has $(creds_hours_left) h left"

for entry in "${STORES[@]}"; do
    IFS='|' read -r SRC DSTNAME GB <<< "$entry"
    NEED=$(echo "$GB $RATE_GB_H" | awk '{printf "%.2f", $1/$2}')
    LEFT=$(creds_hours_left)
    say "--- $SRC : ${GB} GB, needs ~${NEED} h, credential has ${LEFT} h ---"

    if awk "BEGIN{exit !($LEFT < $NEED + $MARGIN_H)}"; then
        say "STOP: not enough credential life for $SRC."
        say "      Refresh:  $SC login --duration 12h --format env"
        say "      then re-run this script -- completed stores are skipped by"
        say "      the size comparison, so nothing is re-transferred."
        exit 2
    fi

    ok=0
    for pass in 1 2 3; do
        say "pass $pass/3 -> $REPO/$DSTNAME"
        "$P" -u mirror_ceph_to_source_coop.py --via-dask "$DASK" --threads 48 \
            --src-prefix "$SRC" --dest "$REPO/$DSTNAME" --env .env \
            >> "$LOG" 2>&1
        rc=$?
        if [ $rc -eq 0 ] && tail -40 "$LOG" | grep -qE "repo. pointer"; then
            say "DONE $SRC on pass $pass"; ok=1; break
        fi
        if [ $rc -eq 2 ]; then
            say "STOP: credential expired mid-store. Refresh and re-run."; exit 2
        fi
        say "pass $pass left errors; retrying"
    done
    [ $ok -eq 1 ] || { say "ABORT: $SRC unfinished after 3 passes"; exit 1; }
done

say "=== all stores mirrored ==="
say "NOT deleted from Ceph -- that is a separate, deliberate step."
say "verify one with: $P - <<'X'  (see the 50r1 tail verification)"
