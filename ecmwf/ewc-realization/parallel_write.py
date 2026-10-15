"""Realize a date using icechunk's fork/merge distributed writes.

The difference from the client-side path in frisky_daily_dag.py:

    client-side   worker reads -> CLIENT gathers 259 MB/channel -> client
                  writes -> 30 commits.  Client OOM-killed at channel 21 on
                  the gateway's 8 GiB cgroup; write is serial.

    fork/merge    worker reads -> WORKER writes its own chunk -> returns a
                  changeset.  Client merges and commits once.  No bulk data
                  reaches the client at all.

Shape, per date:

    read_message x 51 members ---> write_block(fork, channel, step) -> ForkSession
    ... 1,908 of those (36 channels x 53 steps) ...
    client: session.merge(*1908 forks); session.commit()

Chunk alignment is what makes it legal: the store chunk is
(1, 1, number, lat, lon), so one (channel, step) block is exactly one chunk.
No two workers write the same chunk, which is the constraint icechunk puts on
the caller.

Usage
-----
    P=.venv/bin/python
    $P parallel_write.py --sink ea-swio/smoke --channels 2 \
        --members 4 --steps 3                      # smoke test
    $P parallel_write.py --sink ea-swio/one-date --channels 36 \
        --members 51 --steps 53                    # a full date
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import frisky_daily_dag as dag
import sink_icechunk as sink


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheduler", default="192.168.1.74:8796")
    p.add_argument("--date", default="2026-07-02")
    p.add_argument("--era", default="50r1")
    p.add_argument("--sink", required=True, help="prefix under must-icechunk")
    p.add_argument("--channels", type=int, default=dag.N_CHANNELS)
    p.add_argument("--vars", nargs="+", default=None)
    p.add_argument("--members", type=int, default=51)
    p.add_argument("--steps", type=int, default=53)
    p.add_argument("--env", default=".env")
    p.add_argument("--recreate", action="store_true",
                   help="DESTRUCTIVE: write the schema over an existing store, "
                        "making every committed channel unreachable from "
                        "`main`. Without this an existing store is refused")
    p.add_argument("--fork-once", action="store_true",
                   help="fork ONE session and pickle it to every task, "
                        "instead of calling session.fork() per block. "
                        "Each unpickled copy records its own changes, so the "
                        "merge is unchanged -- but 1,908 fork() calls cost "
                        "476s of the 494s a full date took")
    args = p.parse_args()

    import frisky

    channels = dag.select_channels(args)
    members = list(range(args.members))
    steps = dag.STEPS_ALL_H[:args.steps]
    n_leaf = len(channels) * len(members) * len(steps)
    n_block = len(channels) * len(steps)

    print(f"date     {args.date}   era {args.era}")
    print(f"channels {len(channels)}  members {len(members)}  "
          f"steps {len(steps)}")
    print(f"tasks    {n_leaf} reads + {n_block} chunk writes")

    ak, sk = sink.load_env(args.env)
    repo, created = sink.open_sink(args.sink, ak, sk)
    print(f"sink     must-icechunk/{args.sink} "
          f"({'created' if created else 'existing'})")

    # `create_schema` below writes with mode="w".  Against a store that already
    # holds data that is not a no-op -- it lays a fresh schema over the live
    # one and makes every committed chunk unreachable from `main` in a single
    # commit.  That is exactly what cost 30 committed dates on 2026-08-07
    # (EWC_REALIZE_NOTES.md, Corrections §6); `build_corpus.py` was guarded
    # afterwards and this script was not.
    if not created and not args.recreate:
        already = sink.existing_channels(repo)
        if already:
            sys.exit(
                f"REFUSING: must-icechunk/{args.sink} already holds "
                f"{len(already)} channel(s) and writing the schema would make "
                f"them unreachable.\n"
                f"  pick a new --sink, or pass --recreate if wiping it is "
                f"genuinely what you want.")

    # The rollback path, printed before anything can write.  Cheap, and the
    # entire recovery route if a run goes wrong.
    try:
        head = list(repo.ancestry(branch="main"))[0].id
        print(f"rollback repo.reset_branch(\"main\", \"{head}\")")
    except Exception:
        pass

    client = frisky.Client(args.scheduler)
    print(f"cluster  {args.scheduler}")

    t0 = time.time()
    coords = dag.subset_coords(args.era, args.date, members, steps)
    names = [c[2] for c in channels]
    snap = sink.create_schema(repo, names, coords, [args.date])
    print(f"schema   {len(names)} arrays, metadata only, "
          f"{str(snap)[:12]} in {time.time() - t0:.1f}s")

    # Fresh coordinator session: fork() refuses a session with uncommitted
    # changes, and the schema commit above closed the previous one.
    session = repo.writable_session("main")

    t1 = time.time()
    shared_fork = session.fork() if args.fork_once else None
    futures = []
    for var, level, name in channels:
        for si, step_h in enumerate(steps):
            reads = [client.submit(dag.read_message, args.era, var, level,
                                   args.date, m, step_h) for m in members]
            futures.append(client.submit(dag.write_block,
                                         shared_fork or session.fork(),
                                         name, 0, si, *reads))
    print(f"submitted {len(futures)} write blocks in {time.time() - t1:.2f}s"
          f" -- now waiting\n")

    forks, failed = [], []
    for i, f in enumerate(futures):
        try:
            forks.append(f.result())
        except Exception as exc:
            failed.append(f"{type(exc).__name__}: {exc}")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(futures)} blocks written "
                  f"({time.time() - t1:.0f}s)", flush=True)
    read_write = time.time() - t1
    client.close()

    if failed:
        print(f"\n{len(failed)} block(s) FAILED, first few:")
        for e in failed[:5]:
            print("   ", e)
        if not forks:
            return 1

    t2 = time.time()
    final = sink.merge_and_commit(
        repo, forks, f"{args.date}: {len(names)} channels, "
        f"{len(members)} members, {len(steps)} steps")
    merge_s = time.time() - t2

    total_bytes = n_block * len(members) * dag.NY * dag.NX * 4
    print(f"\nread+write {read_write:.1f}s  "
          f"({n_leaf / read_write:.1f} messages/s)")
    print(f"merge      {len(forks)} changesets in {merge_s:.1f}s "
          f"-> {str(final)[:12]}")
    print(f"total      {time.time() - t0:.1f}s for "
          f"{total_bytes / 1e9:.2f} GB raw float32")
    print(f"\nmust-icechunk/{args.sink}  ONE commit, "
          f"{len(failed)} failed blocks")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
