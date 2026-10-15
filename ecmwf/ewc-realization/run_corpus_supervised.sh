#!/bin/bash
# Supervisor for a multi-date corpus build.
#
#   ./run_corpus_supervised.sh <sink> <start> <days> [era] [steps] [channels]
#
#   tmux new-session -d -s mam2025 \
#       './run_corpus_supervised.sh ea-swio/v1-49r1-mam2025 2025-03-01 92'
#   tmux attach -t mam2025
#
# Two things this adds over running build_corpus.py directly:
#
#   1. It is meant to be run under tmux, so the build is not a child of the
#      shell that launched it. A plain `setsid nohup` was NOT enough -- the
#      2026-08-08 07:25 run died silently at date 11/73 when the launching
#      session ended, with no traceback, just a log that stops.
#   2. It retries. Every attempt passes --resume, which skips dates already in
#      the commit log, so a kill costs the in-flight date and nothing else.
#
# --resume is safe on attempt 1 even for a brand-new store: build_corpus.py
# still creates the schema when it created the repo, and the commit log is
# simply empty. Without --resume it would refuse an EXISTING store rather than
# lay a fresh schema over it, and that guard is why this loop cannot destroy
# anything.
set -u

SINK=${1:?usage: $0 <sink> <start> <days> [era] [steps] [channels]}
START=${2:?missing <start> (YYYY-MM-DD)}
DAYS=${3:?missing <days>}
ERA=${4:-49r1}
STEPS=${5:-65}
CHANNELS=${6:-36}

cd "$(dirname "$(readlink -f "$0")")" || exit 1   # run from ecmwf/ewc-realization

P=${P:-.venv/bin/python}   # override: P=/path/to/venv/bin/python
LOG="build_corpus_$(echo "$SINK" | tr '/' '_').log"
MAX_ATTEMPTS=10

echo "supervisor: $SINK  $START +${DAYS}d  era=$ERA steps=$STEPS ch=$CHANNELS" \
    >> "$LOG"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "" >> "$LOG"
    echo "=== supervisor attempt $attempt/$MAX_ATTEMPTS  $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" >> "$LOG"

    "$P" -u build_corpus.py \
        --sink "$SINK" --era "$ERA" \
        --start "$START" --days "$DAYS" \
        --channels "$CHANNELS" --members 51 --steps "$STEPS" \
        --env .env --resume >> "$LOG" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        echo "=== supervisor: build_corpus exited 0 on attempt $attempt ===" >> "$LOG"
        exit 0
    fi

    echo "=== supervisor: exit $rc on attempt $attempt; 120s pause, then --resume ===" >> "$LOG"
    # The pause matters when the cause was AWS throttling: resuming straight
    # into an active SlowDown window just burns the next attempt too.
    sleep 120
done

echo "=== supervisor: gave up after $MAX_ATTEMPTS attempts ===" >> "$LOG"
exit 1
