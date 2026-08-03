"""Stop work on the EWC Dask workers, from the client side.

Needed because `client.restart()` is not reliable here. With tasks still
registered on the scheduler it raises, inside Dask itself:

    File ".../distributed/scheduler.py", line 6455, in restart
        assert not self.tasks
    AssertionError

which leaves the cluster running whatever it was running. This script does the
steps in an order that works: cancel first, then bounce the workers via their
nannies (`restart_workers`), which does not go through that assertion.

Commands
--------
    status   what is executing/queued right now, and which client owns it
    cancel   cancel every task the scheduler knows about (workers keep running,
             memory is NOT reclaimed)
    restart  cancel, then restart worker processes via the nanny -- this is the
             one that also reclaims the leaked ~7 GB/worker (BLOCKERS.md s5)
    kill     restart without cancelling first; last resort if `restart` hangs

Usage
-----
    P=/opt/mamba/envs/dask/bin/python
    $P stop_work.py status
    $P stop_work.py cancel
    $P stop_work.py restart          # the usual one

Note: a client process that was waiting on the cancelled futures will raise
`FutureCancelledError` / `CancelledError` and exit. That is expected -- this
stops the *cluster* side; a detached client script may need killing separately.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def scheduler_state(client):
    """What the scheduler thinks is going on. Runs on the scheduler itself so
    it sees tasks from OTHER clients too -- including dead ones whose futures
    are orphaned, which is the usual cause of a stuck cluster here."""
    def _state(dask_scheduler=None):
        s = dask_scheduler
        by_state = {}
        for ts in s.tasks.values():
            by_state[ts.state] = by_state.get(ts.state, 0) + 1
        return {
            "n_tasks": len(s.tasks),
            "by_state": by_state,
            "n_clients": len([c for c in s.clients if c != "fire-and-forget"]),
            "clients": [c for c in s.clients if c != "fire-and-forget"][:8],
        }
    return client.run_on_scheduler(_state)


def worker_state(client):
    info = client.scheduler_info()["workers"]
    rows = []
    for addr in sorted(info):
        w = info[addr]
        tc = w["metrics"].get("task_counts", {}) or {}
        rows.append({
            "id": w["id"], "addr": addr,
            "rss_gb": w["metrics"].get("memory", 0) / 1e9,
            "limit_gb": w["memory_limit"] / 1e9,
            "executing": tc.get("executing", 0),
            "counts": {k: v for k, v in tc.items() if v},
        })
    return rows


def show(client):
    st = scheduler_state(client)
    print(f"scheduler tasks : {st['n_tasks']}   {st['by_state'] or '{}'}")
    print(f"clients         : {st['n_clients']}  {st['clients']}")
    print(f"\n{'worker':16s} {'addr':22s} {'rss':>8s} {'exec':>5s}  task counts")
    print("-" * 78)
    busy = 0
    for r in worker_state(client):
        busy += r["executing"]
        print(f"{r['id']:16s} {r['addr'].split('/')[-1]:22s} "
              f"{r['rss_gb']:7.2f}G {r['executing']:5d}  {r['counts'] or ''}")
    print("-" * 78)
    print(f"executing across cluster: {busy}")
    return st, busy


def cmd_status(client, args):
    show(client)


def cmd_cancel(client, args):
    st, busy = show(client)
    if not st["n_tasks"]:
        print("\nnothing to cancel")
        return
    print(f"\ncancelling {st['n_tasks']} task(s) on the scheduler ...")

    def _cancel(dask_scheduler=None):
        keys = list(dask_scheduler.tasks)
        dask_scheduler.client_releases_keys(keys=keys, client="stop_work")
        return len(keys)

    try:
        n = client.run_on_scheduler(_cancel)
        print(f"  released {n} key(s)")
    except Exception as e:                                      # noqa: BLE001
        print(f"  scheduler-side release failed ({type(e).__name__}); "
              f"falling back to client.cancel on own futures")
        try:
            client.cancel(list(client.futures.values()), force=True)
        except Exception as e2:                                 # noqa: BLE001
            print(f"  that failed too: {type(e2).__name__}")
    time.sleep(2)
    print()
    show(client)
    print("\nNOTE: cancelling stops the work but does NOT reclaim worker "
          "memory.\n      Use `restart` for that.")


def cmd_restart(client, args, skip_cancel=False):
    if not skip_cancel:
        cmd_cancel(client, args)
        print()
    info = client.scheduler_info()["workers"]
    addrs = sorted(info)
    print(f"restarting {len(addrs)} worker(s) via their nannies ...")
    # restart_workers() goes to the nannies directly and avoids the
    # `assert not self.tasks` path that makes client.restart() unusable here.
    try:
        client.restart_workers(workers=addrs, timeout=args.timeout,
                               raise_for_error=False)
    except Exception as e:                                      # noqa: BLE001
        print(f"  restart_workers failed ({type(e).__name__}: "
              f"{str(e)[:120]})")
        print(f"  falling back to client.restart() ...")
        try:
            client.restart(timeout=args.timeout)
        except Exception as e2:                                 # noqa: BLE001
            print(f"  client.restart() also failed: {type(e2).__name__}: "
                  f"{str(e2)[:160]}")
            print(f"  -> the scheduler is wedged. Restart the scheduler "
                  f"service itself.")
            return 1
    try:
        client.wait_for_workers(len(addrs), timeout=args.timeout)
    except Exception:                                           # noqa: BLE001
        print("  (not all workers came back within the timeout)")
    time.sleep(2)
    print()
    st, busy = show(client)
    tot = sum(r["rss_gb"] for r in worker_state(client))
    print(f"\ntotal worker RSS now: {tot:.2f} GB")
    if tot < 5:
        print("  . memory reclaimed")
    else:
        print("  ! still high -- some workers may not have actually bounced")
    return 0


def cmd_kill(client, args):
    return cmd_restart(client, args, skip_cancel=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "cancel", "restart", "kill"])
    ap.add_argument("--scheduler", default=os.environ.get(
        "DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"))
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    from dask.distributed import Client
    client = Client(args.scheduler, timeout=60)
    try:
        rc = {"status": cmd_status, "cancel": cmd_cancel,
              "restart": cmd_restart, "kill": cmd_kill}[args.command](client, args)
    finally:
        client.close()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
