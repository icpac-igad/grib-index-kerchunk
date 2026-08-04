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
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date as _date, timedelta

import frisky_daily_dag as dag
import sink_icechunk as sink

DONE_PREFIX = "date "          # commit message marker, parsed by --resume


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


def run_one_date(client, repo, channels, members, steps, era, date, date_idx):
    """1,590 blocks for one date, one shared fork, one commit."""
    session = repo.writable_session("main")
    fork = session.fork()

    t0 = time.time()
    futures = []
    for var, level, name in channels:
        for si, step_h in enumerate(steps):
            reads = [client.submit(dag.read_message, era, var, level, date,
                                   m, step_h) for m in members]
            futures.append(client.submit(dag.write_block, fork, name,
                                         date_idx, si, *reads))
    submit_s = time.time() - t0

    forks, failed = [], []
    for f in futures:
        try:
            forks.append(f.result())
        except Exception as exc:
            failed.append(f"{type(exc).__name__}: {exc}")
    read_s = time.time() - t0

    if failed:
        return None, submit_s, read_s, 0.0, failed

    t2 = time.time()
    snap = sink.merge_and_commit(
        repo, forks,
        f"{DONE_PREFIX}{date}: {len(channels)} channels, {len(members)} "
        f"members, {len(steps)} steps")
    return snap, submit_s, read_s, time.time() - t2, []


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
    p.add_argument("--env", default="../.env")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import frisky

    dates = date_range(args.start, args.days)
    channels = dag.select_channels(args)
    members = list(range(args.members))
    steps = dag.STEPS_ALL_H[:args.steps]
    names = [c[2] for c in channels]
    per_date = len(channels) * len(steps)

    gb = len(dates) * len(channels) * len(members) * len(steps) * 0.788 / 1000
    print(f"corpus   {dates[0]} .. {dates[-1]}  ({len(dates)} dates)")
    print(f"shape    {len(channels)} channels x {len(members)} members x "
          f"{len(steps)} steps")
    print(f"work     {per_date} blocks/date, "
          f"{len(dates) * per_date} total, ~{gb:.2f} TB of AWS egress\n")

    ak, sk = sink.load_env(args.env)
    repo, created = sink.open_sink(args.sink, ak, sk)
    print(f"sink     must-icechunk/{args.sink} "
          f"({'created' if created else 'existing'})")

    done = committed_dates(repo) if args.resume else set()
    if done:
        print(f"resume   {len(done)} date(s) already committed, skipping")

    client = frisky.Client(args.scheduler)
    coords = dag.subset_coords(args.era, dates[0], members, steps)

    if created or not args.resume:
        t = time.time()
        snap = sink.create_schema(repo, names, coords, dates)
        print(f"schema   {len(names)} arrays x {len(dates)} dates, "
              f"metadata only, {str(snap)[:12]} in {time.time() - t:.1f}s")

    print()
    t_all = time.time()
    ok, bad = [], []
    for i, date in enumerate(dates):
        if date in done:
            print(f"[{i + 1:2d}/{len(dates)}] {date}  skipped (committed)")
            continue
        snap, sub, rd, mg, failed = run_one_date(
            client, repo, channels, members, steps, args.era, date, i)
        if failed:
            bad.append((date, failed))
            print(f"[{i + 1:2d}/{len(dates)}] {date}  FAILED "
                  f"{len(failed)} blocks: {failed[0][:90]}", flush=True)
            continue
        ok.append(date)
        rate = len(channels) * len(members) * len(steps) / rd
        eta = (time.time() - t_all) / len(ok) * (len(dates) - i - 1)
        print(f"[{i + 1:2d}/{len(dates)}] {date}  {rd:6.1f}s  "
              f"{rate:6.1f} msg/s  submit {sub:4.1f}s  merge {mg:4.1f}s  "
              f"-> {str(snap)[:12]}   eta {eta / 60:.0f}m", flush=True)

    client.close()
    mins = (time.time() - t_all) / 60
    print(f"\n{len(ok)}/{len(dates)} dates in {mins:.1f} min")
    if bad:
        print(f"FAILED dates: {[d for d, _ in bad]}")
        print("rerun with --resume to retry only those")
    print(f"\nverify: check_complete.py --prefix {args.sink}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
