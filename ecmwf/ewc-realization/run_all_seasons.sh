#!/bin/bash
# Rebuild all three MAM seasons from the CORRECTED v4 source store, in order,
# on the WIDENED domain: lat 40N -> 40S, lon 15E -> 80E, 321 x 261.
#
# Widening from 25.25N cost nothing to read -- a GRIB message is global, so the
# message count is fixed by channels x members x steps and the box only decides
# what is kept. It costs on the write side: 12.0 -> 14.8 GB stored per date.
# For the same reason, realizing the northern strip as a SEPARATE store and
# concatenating would have been correct but slower: a 59-row strip costs a full
# read pass, so two passes cost twice, for 18% more data.
#
#   tmux new-session -d -s seasons ./run_all_seasons.sh
#   tmux attach -t seasons              # Ctrl-b d to detach
#   tail -f run_all_seasons.log
#
# Order is deliberate: 2026 first. Its 2026-03-01 was realized and verified
# byte-exact against the source at the narrower 25.25N extent, so the pipeline
# itself is trusted and anything new shows up in the first hours rather than
# after two days. The first date is also the de-facto test of the widened
# write path -- a 321-row chunk is 17.09 MB against 13.95 -- so watch date 1.
#
# MAM 2026 needs two stores. The 49r1 era ends 2026-05-12 and 50r1 begins
# 2026-05-13, and a store's `time` axis is fixed at creation, so the seam
# cannot live inside one store. Read them together with
# xr.concat([a, b], dim="time") -- grid, steps and channels are identical.
#
# Each season runs under run_corpus_supervised.sh, which retries with --resume
# up to 10 times. This driver additionally refuses to start a season if the
# previous one did not leave its success marker, so a failure stops the line
# instead of quietly building the rest on a broken foundation.
set -u

cd "$(dirname "$(readlink -f "$0")")" || exit 1   # run from ecmwf/ewc-realization
LOG=run_all_seasons.log
MARKER="supervisor: build_corpus exited 0"

say() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')  $*" | tee -a "$LOG"; }

# sink                              start        days  era
SEASONS=(
  "ea-swio/v3-49r1-mam2026|2026-03-01|73|49r1"
  "ea-swio/v3-50r1-mam2026-tail|2026-05-13|19|50r1"
  "ea-swio/v3-49r1-mam2025|2025-03-01|92|49r1"
  "ea-swio/v3-49r1-mam2024|2024-03-01|92|49r1"
)

say "=== rebuild from the corrected v4 source: ${#SEASONS[@]} seasons, 276 dates ==="

for entry in "${SEASONS[@]}"; do
    IFS='|' read -r SINK START DAYS ERA <<< "$entry"
    SLOG="build_corpus_$(echo "$SINK" | tr '/' '_').log"

    if [ -f "$SLOG" ] && grep -q "$MARKER" "$SLOG"; then
        say "SKIP  $SINK -- already complete"
        continue
    fi

    say "START $SINK  $START +${DAYS}d  era=$ERA"
    ./run_corpus_supervised.sh "$SINK" "$START" "$DAYS" "$ERA" 65 36
    rc=$?

    if [ $rc -ne 0 ] || ! grep -q "$MARKER" "$SLOG"; then
        say "ABORT $SINK exited $rc without a success marker."
        say "      Remaining seasons NOT started. Fix, then re-run this script:"
        say "      it skips seasons that already carry the marker and resumes"
        say "      the one that failed from its commit log."
        exit 1
    fi
    say "DONE  $SINK"
done

say "=== all seasons complete ==="
say "verify:  for p in v3-49r1-mam2026 v3-50r1-mam2026-tail v3-49r1-mam2025 v3-49r1-mam2024; do"
say "           \$P check_chunks.py      --prefix ea-swio/\$p --env .env"
say "           \$P check_geolocation.py --prefix ea-swio/\$p --env .env"
say "         done"
