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
    """Dates already committed, read back off the commit log."""
    done = set()
    try:
        for s in repo.ancestry(branch="main"):
            if s.message.startswith(DONE_PREFIX):
                done.add(s.message[len(DONE_PREFIX):].split(":")[0].strip())
    except Exception:
        pass
    return done


def run_one_date(client, repo, channels, members, steps, era, date, date_idx,
                 rounds=4, pause=45):
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

    pending = [(var, level, name, si, step_h)
               for var, level, name in channels
               for si, step_h in enumerate(steps)]

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
        return None, submit_s, time.time() - t0, 0.0, errs, retried

    read_s = time.time() - t0
    t2 = time.time()
    snap = sink.merge_and_commit(
        repo, forks,
        f"{DONE_PREFIX}{date}: {len(channels)} channels, {len(members)} "
        f"members, {len(steps)} steps")
    return snap, submit_s, read_s, time.time() - t2, [], retried


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

    done = committed_dates(repo) if args.resume else set()
    if done:
        print(f"resume   {len(done)} date(s) already committed, skipping")

    client = frisky.Client(args.scheduler)
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

    if created or not args.resume:
        t = time.time()
        snap = sink.create_schema(repo, names, coords, dates)
        print(f"schema   {len(names)} arrays x {len(dates)} dates, "
              f"metadata only, {str(snap)[:12]} in {time.time() - t:.1f}s")

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
        snap, sub, rd, mg, failed, retried = run_one_date(
            client, repo, channels, members, steps, args.era, date, i,
            rounds=args.rounds, pause=args.pause)
        if failed:
            bad.append((date, failed))
            print(f"[{i + 1:2d}/{len(dates)}] {date}  FAILED "
                  f"{len(failed)} blocks: {failed[0][:90]}", flush=True)
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
        print(f"FAILED dates: {[d for d, _ in bad]}")
        print("rerun with --resume to retry only those")
    print(f"\nverify: check_complete.py --prefix {args.sink}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
