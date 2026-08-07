"""Build the multi-date corpus: one schema, one commit per date.

    build_corpus.py --sink ea-cgan/v3-june2026 --start 2026-06-01 --days 30

Shape
-----
The `time` axis is preallocated to the full date range up front, so every
write is a plain region write into an existing array.  Nothing appends and
nothing mutates the schema, which means dates are independent and a failure
costs exactly one date.

Per date: 1,590 blocks (30 channels x 53 steps), all sharing ONE forked
session, merged and committed together.

    read_message x 51 -> write_block(fork, channel, date_idx, step_idx)
    ... 1,590 per date ...
    merge -> commit "2026-06-01: 30 channels ..."

Why one fork per date and not per block: `session.fork()` costs ~0.30 s, so
forking per block cost 476 s of the 494 s a single date took, and would be
~4 hours of pure overhead across 30 dates.  One fork pickled to every task
gives each worker an independent copy that records its own changes, verified
identical at 1,590 blocks (`check_complete.py` on two independently written
stores).

Resume
------
Each date is its own commit, so `--resume` reads the commit log and skips
dates already done.  A date costs 81,090 messages and ~64 GB of egress; none
of that should ever be repeated because a later date failed.

Holes in the source
-------------------
The published virtual store is not gap-free -- `49r1/00z` is missing ten March
2026 dates that ECMWF did publish.  Requested dates absent from the source axis
are SKIPPED, but their `time` slots are still reserved by the schema.  An
unwritten slot has no chunks in the manifest, so `check_chunks.py` names it
precisely instead of `create_schema`'s zero-fill passing it off as data.  When
Stage 1 catches up, `--resume` writes those slots in place: they are region
writes into a coordinate that was laid down sorted, so filling March 19th after
March 31st cannot put the axis out of order.  That ordering guarantee is a
property of preallocation, NOT of the commit sequence -- the source store's own
non-monotonic axes came from `append_dim="time"`, a path this script never
takes.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date as _date, timedelta

import frisky_daily_dag as dag
import sink_icechunk as sink

DONE_PREFIX = "date "          # commit message marker, parsed by --resume


def _rss_gb():
    """Client resident memory. The gateway cgroup is 8 GiB and the client
    gathers one ForkSession per block, so this is worth watching."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("nan")


def date_range(start, days):
    y, m, d = (int(x) for x in start.split("-"))
    d0 = _date(y, m, d)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(days)]


def committed_dates(repo):
    """(complete, partial) date sets, read back off the commit log.

    A commit no longer implies a COMPLETE date -- `run_one_date` now commits
    the blocks that landed rather than discarding a whole date for one bad
    block, and labels such a commit PARTIAL. So `--resume` must key off
    completeness, not mere presence, or it would skip a date full of holes.

    Ancestry is newest-first, so the first mention of a date is its current
    state: a repaired date's later complete commit correctly overrides the
    earlier PARTIAL one.
    """
    complete, partial = set(), set()
    try:
        for s in repo.ancestry(branch="main"):
            msg = s.message or ""
            if not msg.startswith(DONE_PREFIX):
                continue
            d = msg[len(DONE_PREFIX):].split(":")[0].strip()
            if d in complete or d in partial:
                continue                      # already saw a newer commit
            (partial if "PARTIAL" in msg else complete).add(d)
    except Exception:
        pass
    return complete, partial


def missing_blocks(repo, channel_names, n_dates, n_steps):
    """Blocks in the schema but absent from the MANIFEST -> [(t_idx, name, s_idx)].

    The manifest is the only honest source here. A chunk that was never
    written simply is not in it; reading the array instead would return
    `fill_value` (nan for this store) which tells you nothing about whether a
    write was attempted.

    Costs one manifest read, not a data read -- the same trick `check_chunks.py`
    uses to audit a corpus without pulling 233 GB.
    """
    import asyncio

    sess = repo.readonly_session("main")

    async def go():
        miss = []
        for name in channel_names:
            have = {(c[0], c[1]) async for c in sess.chunk_coordinates(f"/{name}")}
            for ti in range(n_dates):
                for si in range(n_steps):
                    if (ti, si) not in have:
                        miss.append((ti, name, si))
        return miss

    return asyncio.run(go())


def run_one_date(client, repo, channels, members, steps, era, date, date_idx,
                 rounds=4, pause=45, pending=None):
    """1,590 blocks for one date, one shared fork, one commit.

    Blocks that fail are RESUBMITTED rather than failing the date.  AWS
    throttles this workload even at 96 concurrent readers: the per-read
    backoff in `read_message` absorbs most of it, but a few blocks exhaust
    all 8 attempts. Discarding the date for 6 bad blocks out of 1,590 throws
    away 81,090 messages and ~64 GB of egress to avoid redoing ~300.

    Between rounds we pause: the whole point is to stop asking for a while,
    and immediately resubmitting into a throttling window just burns the
    retries again.
    """
    session = repo.writable_session("main")
    fork = session.fork()

    if pending is None:
        pending = [(var, level, name, si, step_h)
                   for var, level, name in channels
                   for si, step_h in enumerate(steps)]
    n_total = len(pending)

    t0 = time.time()
    forks, submit_s, retried = [], 0.0, 0

    for rnd in range(rounds):
        ts = time.time()
        futures = []
        for var, level, name, si, step_h in pending:
            reads = [client.submit(dag.read_message, era, var, level, date,
                                   m, step_h) for m in members]
            futures.append(client.submit(dag.write_block, fork, name,
                                         date_idx, si, *reads))
        if rnd == 0:
            submit_s = time.time() - ts

        still, errs = [], []
        for spec, f in zip(pending, futures):
            try:
                forks.append(f.result())
            except Exception as exc:
                still.append(spec)
                errs.append(f"{type(exc).__name__}: {exc}")

        if not still:
            break
        retried += len(still)
        pending = still
        if rnd < rounds - 1:
            # Print WHY, not just how many.  This path used to report a bare
            # count, which is useless for telling AWS throttling (retrying is
            # the right response) apart from the client's channel dropping
            # (retrying cannot possibly work, and the pause just delays the
            # inevitable).  Group by exception type and show one example each.
            tally = {}
            for e in errs:
                tally.setdefault(e.split(":")[0], []).append(e)
            print(f"          retry {len(still)} block(s) of {len(futures)} "
                  f"(round {rnd + 2}/{rounds}) after {pause}s pause",
                  flush=True)
            for kind, group in sorted(tally.items(),
                                      key=lambda kv: -len(kv[1])):
                print(f"            {len(group):5d}x {kind}: "
                      f"{group[0][:150]}", flush=True)
            print(f"            client rss {_rss_gb():.2f} GB, "
                  f"{len(forks)} forks held", flush=True)
            time.sleep(pause)
    else:
        # Rounds exhausted.  COMMIT WHAT LANDED rather than discarding it.
        #
        # This path used to `return None` before merging, so one unrecovered
        # block out of 1,590 threw away the other 1,589 -- ~81,000 GRIB
        # messages and ~64 GB of egress already paid for.  That happened to
        # 2026-03-23 on the March fill.
        #
        # Committing the partial set is safe because every write is a region
        # write into an array `create_schema` already laid out: a missing chunk
        # is simply absent from the manifest, and reads return the array's
        # fill_value (nan here), so a hole is loud rather than plausible.
        # The commit is labelled PARTIAL so `--resume` will not mistake it for
        # a finished date, and `--repair` can fill just the gaps.
        read_s = time.time() - t0
        t2 = time.time()
        snap = None
        if forks:
            snap = sink.merge_and_commit(
                repo, forks,
                f"{DONE_PREFIX}{date}: PARTIAL {n_total - len(pending)}/"
                f"{n_total} blocks, {len(members)} members")
        return snap, submit_s, read_s, time.time() - t2, errs, retried, pending

    read_s = time.time() - t0
    t2 = time.time()
    snap = sink.merge_and_commit(
        repo, forks,
        f"{DONE_PREFIX}{date}: {len(channels)} channels, {len(members)} "
        f"members, {len(steps)} steps")
    return snap, submit_s, read_s, time.time() - t2, [], retried, []


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheduler", default="192.168.1.74:8796")
    p.add_argument("--sink", required=True)
    p.add_argument("--start", default="2026-06-01")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--era", default="50r1")
    p.add_argument("--channels", type=int, default=30)
    p.add_argument("--members", type=int, default=51)
    p.add_argument("--steps", type=int, default=53)
    p.add_argument("--env", default=".env")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--recreate", action="store_true",
                   help="DESTRUCTIVE: rewrite the schema over an existing "
                        "store, making every committed date unreachable from "
                        "`main`. Without this, an existing store is never "
                        "overwritten")
    p.add_argument("--repair", action="store_true",
                   help="fill ONLY the chunks the manifest reports missing, "
                        "instead of redoing whole dates. A date that lost two "
                        "blocks costs two blocks, not 1,590")
    p.add_argument("--rounds", type=int, default=4,
                   help="resubmit rounds for blocks AWS throttled out")
    p.add_argument("--pause", type=int, default=45,
                   help="seconds between retry rounds -- the point is to "
                        "stop asking for a while")
    args = p.parse_args()

    import frisky

    dates = date_range(args.start, args.days)
    channels = dag.select_channels(args)
    members = list(range(args.members))
    steps = dag.STEPS_ALL_H[:args.steps]
    names = [c[2] for c in channels]
    per_date = len(channels) * len(steps)

    n_msg = len(dates) * len(channels) * len(members) * len(steps)
    tb = n_msg * 0.788 / 1e6          # 0.788 MB per message -> TB
    print(f"corpus   {dates[0]} .. {dates[-1]}  ({len(dates)} dates)")
    print(f"shape    {len(channels)} channels x {len(members)} members x "
          f"{len(steps)} steps")
    print(f"work     {per_date} blocks/date, "
          f"{len(dates) * per_date} total, {n_msg:,} messages, "
          f"~{tb:.2f} TB of AWS egress\n")

    ak, sk = sink.load_env(args.env)
    repo, created = sink.open_sink(args.sink, ak, sk)
    print(f"sink     must-icechunk/{args.sink} "
          f"({'created' if created else 'existing'})")

    done, partial = (committed_dates(repo) if (args.resume or args.repair)
                     else (set(), set()))
    if done:
        print(f"resume   {len(done)} date(s) complete, skipping")
    if partial:
        print(f"partial  {len(partial)} date(s) committed WITH HOLES: "
              f"{', '.join(sorted(partial))}")
        print(f"         run --repair to fill just the missing blocks")

    client = frisky.Client(args.scheduler)

    # Only `create_schema` consumes `coords`, so a --resume or --repair run has
    # no use for it -- and fetching it opens the SOURCE store, which is one more
    # thing that can fail before any work starts. source.coop returns sporadic
    # 5xx ("error parsing XML: no root element"), and one of those killed a
    # repair run at this line before it had written a single block. Don't pay
    # for what we are not going to use.
    coords = None
    if created or args.recreate:
        coords = dag.subset_coords(args.era, dates[0], members, steps)

    # Which dates does the SOURCE actually have?  The schema still reserves a
    # slot for every requested date -- an absent one is left unwritten, so it
    # is absent from the manifest too and `check_chunks.py` names it exactly,
    # rather than being zero-filled into something that looks like data.
    # A later `--resume` fills those reserved slots in place as region writes.
    present = set(dag.available_dates(args.era, dates))
    absent = [d for d in dates if d not in present]
    if absent:
        print(f"source   {len(present)}/{len(dates)} dates on the {args.era} "
              f"axis; {len(absent)} absent, slots reserved but unwritten:")
        print(f"         {', '.join(absent)}")

    # `create_schema` writes with mode="w", which WIPES an existing store.
    # This used to fire on `created or not args.resume`, so any invocation that
    # merely forgot --resume silently destroyed the corpus. On 2026-08-07 a
    # --repair run (which does not set --resume) did exactly that: 30 committed
    # dates, 47,700 chunks, gone in one commit. Recovered only because icechunk
    # keeps history -- repo.reset_branch("main", <pre-wipe snapshot>).
    #
    # Now: lay down the schema ONLY for a store this run created, and refuse to
    # overwrite an existing one unless asked in so many words.
    if created:
        t = time.time()
        snap = sink.create_schema(repo, names, coords, dates)
        print(f"schema   {len(names)} arrays x {len(dates)} dates, "
              f"metadata only, {str(snap)[:12]} in {time.time() - t:.1f}s")
    elif args.recreate:
        print(f"WARNING  --recreate: overwriting the existing schema at "
              f"must-icechunk/{args.sink}; every committed date becomes "
              f"unreachable from `main`")
        t = time.time()
        snap = sink.create_schema(repo, names, coords, dates)
        print(f"schema   rewritten, {str(snap)[:12]} in {time.time() - t:.1f}s")
    elif not (args.resume or args.repair):
        raise SystemExit(
            f"must-icechunk/{args.sink} already exists.\n"
            f"Writing a fresh schema over it would make every committed date "
            f"unreachable.\n"
            f"  --resume   continue, skipping complete dates\n"
            f"  --repair   fill only the chunks the manifest reports missing\n"
            f"  --recreate deliberately start the store over")

    if args.repair:
        # Fill ONLY the chunks the manifest says are absent.  A date that lost
        # two blocks costs two blocks to fix, not 1,590 -- the whole point of
        # writing into a preallocated schema.
        t_r = time.time()
        miss = missing_blocks(repo, names, len(dates), len(steps))
        if not miss:
            print("repair   nothing missing -- store is complete")
            client.close()
            return 0
        by_date = {}
        for ti, name, si in miss:
            by_date.setdefault(ti, []).append((name, si))
        print(f"repair   {len(miss):,} missing chunk(s) across "
              f"{len(by_date)} date(s)  (manifest read {time.time()-t_r:.1f}s)")
        by_name = {c[2]: c for c in channels}
        bad = []
        for ti in sorted(by_date):
            date = dates[ti]
            if date not in present:
                print(f"  {date}  {len(by_date[ti]):5d} missing -- but absent "
                      f"from the {args.era} source, cannot fill")
                continue
            spec = [(by_name[n][0], by_name[n][1], n, si, steps[si])
                    for n, si in by_date[ti]]
            t0 = time.time()
            snap, sub, rd, mg, failed, retried, still = run_one_date(
                client, repo, channels, members, steps, args.era, date, ti,
                rounds=args.rounds, pause=args.pause, pending=spec)
            state = ("STILL INCOMPLETE" if still else "repaired")
            if still:
                bad.append(date)
            print(f"  {date}  {len(spec):5d} block(s)  {time.time()-t0:6.1f}s  "
                  f"retried {retried:3d}  -> {str(snap)[:12]}  {state}"
                  + (f"  ({len(still)} left)" if still else ""), flush=True)
        client.close()
        print(f"\nrepair done in {(time.time()-t_r)/60:.1f} min")
        if bad:
            print(f"still incomplete: {bad} -- rerun --repair")
        return 1 if bad else 0

    print()
    t_all = time.time()
    todo = [d for d in dates if d in present and d not in done]
    ok, bad = [], []
    for i, date in enumerate(dates):
        if date in done:
            print(f"[{i + 1:2d}/{len(dates)}] {date}  skipped (committed)")
            continue
        if date not in present:
            print(f"[{i + 1:2d}/{len(dates)}] {date}  skipped "
                  f"(absent from {args.era} source)")
            continue
        snap, sub, rd, mg, failed, retried, still = run_one_date(
            client, repo, channels, members, steps, args.era, date, i,
            rounds=args.rounds, pause=args.pause)
        if still:
            # No longer a total loss: the blocks that landed are committed and
            # labelled PARTIAL. Only `still` needs redoing, via --repair.
            bad.append((date, failed))
            n_blk = len(channels) * len(steps)
            print(f"[{i + 1:2d}/{len(dates)}] {date}  PARTIAL "
                  f"{n_blk - len(still)}/{n_blk} blocks committed -> "
                  f"{str(snap)[:12]}   {len(still)} missing: "
                  f"{failed[0][:70] if failed else ''}", flush=True)
            continue
        ok.append(date)
        rate = len(channels) * len(members) * len(steps) / rd
        # ETA over the dates still to DO, not the calendar days left -- with
        # ten absent dates those differ by a third of the month.
        eta = (time.time() - t_all) / len(ok) * max(len(todo) - len(ok), 0)
        print(f"[{i + 1:2d}/{len(dates)}] {date}  {rd:6.1f}s  "
              f"{rate:6.1f} msg/s  submit {sub:4.1f}s  merge {mg:4.1f}s  "
              f"retried {retried:4d}  -> {str(snap)[:12]}   "
              f"eta {eta / 60:.0f}m", flush=True)

    client.close()
    mins = (time.time() - t_all) / 60
    print(f"\n{len(ok)}/{len(todo)} available dates in {mins:.1f} min "
          f"({len(dates)} slots reserved)")
    if absent:
        print(f"absent from source ({len(absent)}): {', '.join(absent)}")
        print("  slots reserved and unwritten -- rerun with --resume once "
              "Stage 1 publishes them")
    if bad:
        print(f"PARTIAL dates: {[d for d, _ in bad]}")
        print("rerun with --repair to fill only the missing blocks "
              "(--resume would redo the whole date)")
    print(f"\nverify: check_complete.py --prefix {args.sink}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
