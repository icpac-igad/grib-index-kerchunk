#!/bin/bash
# Restart the Frisky cluster every N hours while a realization run is going.
#
#   tmux new-session -d -s watchdog './cluster_watchdog.sh 16'
#   tail -f cluster_watchdog.log
#   tmux kill-session -t watchdog          # ALWAYS stop it before manual work
#
# Why this exists: a cluster left up for ~24 h drops to about a third of its
# fresh throughput, silently -- each worker ends up running ONE task instead of
# eight, and most of its RSS is allocator arena that was freed but never
# returned to the OS. Nothing errors; the run just crawls. Measured on
# 2026-08-18: 12 tasks processing vs 197 fresh, 34.4 GiB vs 3.1 GiB, 67 msg/s
# vs 180+. See CLUSTER_RESTART_ROUTINE.md.
#
# The procedure is exactly the manual one:
#   kill build_corpus  ->  restart workers  ->  the supervisor resumes itself
#
# It kills ONLY build_corpus.py. The supervisor (run_corpus_supervised.sh) and
# the season driver (run_all_seasons.sh) must survive -- they are what brings
# the build back, with --resume, after a 120 s pause. Killing the wrong one
# stops the run for good.
set -u

HOURS=${1:-16}
cd "$(dirname "$(readlink -f "$0")")" || exit 1   # run from ecmwf/ewc-realization

D=/opt/mamba/envs/dask/bin/python
SCHED_VM=192.168.1.74
LOG=cluster_watchdog.log

say() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')  $*" >> "$LOG"; }

# Last `msg/s` printed by whichever build log is newest -- the cheapest health
# signal there is, and the one worth recording either side of a restart.
last_rate() {
    local f
    f=$(ls -t build_corpus_*.log 2>/dev/null | head -1) || return
    [ -n "${f:-}" ] || return
    # Match on "msg/s" itself, not on the [n/m] prefix: a --resume run emits
    # "[ 4/92] 2025-03-04  skipped (committed)" lines that carry no rate, and
    # tailing those returns nothing.
    grep -oE "[0-9]+\.[0-9]+ msg/s" "$f" 2>/dev/null | tail -1
}

say "watchdog started, interval ${HOURS}h"

while true; do
    sleep "$((HOURS * 3600))"

    if ! pgrep -f "build_corpus.py --sink" > /dev/null; then
        say "no build running -- nothing to do (a cluster sitting idle does "\
"not degrade; it degrades by working)"
        continue
    fi

    say "=== ${HOURS}h elapsed; rate before restart: $(last_rate) ==="

    # Kill only the build. Never the supervisor, never the season driver.
    pkill -f "build_corpus.py --sink"
    say "killed build_corpus; supervisor will resume with --resume after 120 s"

    if "$D" deploy_workers.py --scheduler-on "$SCHED_VM" --start \
            --workers-per-vm 2 --nthreads 8 --memory-limit 6GB \
            >> "$LOG" 2>&1; then
        say "workers restarted"
    else
        say "WARNING deploy_workers exited non-zero -- check the cluster by "\
"hand. The supervisor will keep retrying, but against workers that may not "\
"be there."
    fi

    # Give the supervisor its pause plus the first date, then record the gain.
    sleep 1200
    say "rate after restart: $(last_rate)"
done
