"""Is the ~31 MB/s ceiling per PROCESS or per VM?

bandwidth_probe.py raised threads inside one process and flattened at ~31 MB/s.
Two very different explanations fit that curve equally well:

  per-VM / network   the path to AWS is saturated.  Nothing helps except more
                     VMs or moving in-region.  CPU and RAM stay low forever
                     and that is correct, not wasteful.

  per-PROCESS        icechunk's Rust object-store client caps in-flight
                     connections per process.  Then N processes on one VM
                     should give ~N x 31 MB/s, and running more, smaller
                     Frisky workers per VM is the whole answer.

Threads cannot tell these apart.  Independent processes can: this launches N
of them on one machine, each doing the same read workload at fixed
concurrency, and sums the throughput.

    1 proc -> ~31 MB/s and 4 proc -> ~31 MB/s   =>  network. stop tuning.
    1 proc -> ~31 MB/s and 4 proc -> ~120 MB/s  =>  per-process cap. add
                                                    workers per VM.

Usage
-----
    .venv/bin/python procs_probe.py --procs 1 2 4 --conc 32 --messages 120
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

MSG_MB = 0.788


def _worker(n_msgs, conc, era, var, level, date, out_q):
    """One independent process: its own store session, its own S3 client."""
    import os
    for k in [k for k in list(os.environ) if k.startswith("AWS_")]:
        os.environ.pop(k)
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from frisky_daily_dag import (_open_era, LAT_MAX, LAT_MIN, LON_MIN,
                                  LON_MAX, STEPS_ALL_H)

    _open_era(era)                       # warm, excluded from the timing

    def one(i):
        try:
            ds = _open_era(era)
            da = ds[var]
            if level is not None:
                da = da.sel(isobaricInhPa=level)
            da = da.sel(time=np.datetime64(date), number=i % 51,
                        step=np.timedelta64(
                            STEPS_ALL_H[(i // 51) % len(STEPS_ALL_H)], "h"),
                        latitude=slice(LAT_MAX, LAT_MIN),
                        longitude=slice(LON_MIN, LON_MAX))
            np.asarray(da.values, dtype=np.float32)
            return True
        except Exception:
            return False

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        ok = sum(ex.map(one, range(n_msgs)))
    out_q.put((ok, time.time() - t0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--conc", type=int, default=32, help="threads per process")
    p.add_argument("--messages", type=int, default=120, help="per process")
    p.add_argument("--era", default="50r1")
    p.add_argument("--var", default="t2m")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--date", default="2026-07-02")
    args = p.parse_args()

    ctx = mp.get_context("spawn")
    print(f"{args.var}  {args.conc} threads/process, "
          f"{args.messages} messages/process\n")
    print(f"{'procs':>6} {'in flight':>10} {'read':>8} {'MB/s':>9} "
          f"{'Gbps':>7} {'per-proc':>9}")
    print("-" * 56)

    rows = []
    for n in args.procs:
        q = ctx.Queue()
        ps = [ctx.Process(target=_worker,
                          args=(args.messages, args.conc, args.era, args.var,
                                args.level, args.date, q))
              for _ in range(n)]
        t0 = time.time()
        for pr in ps:
            pr.start()
        res = [q.get() for _ in ps]
        for pr in ps:
            pr.join()
        # Time the READS, using the children's own clocks.  The parent's
        # wall clock also covers process spawn, module import and the ~15 s
        # store open -- which on a 4 s read workload is most of the number,
        # and made a first run report 4.9 MB/s where the true rate was ~21.
        spawn_s = time.time() - t0
        total_ok = sum(o for o, _ in res)
        wall = max(e for _, e in res)          # concurrent window
        mbs = total_ok * MSG_MB / wall
        rows.append((n, mbs))
        print(f"{n:6d} {n * args.conc:10d} {wall:7.1f}s {mbs:9.1f} "
              f"{mbs * 8 / 1000:7.2f} {mbs / n:9.1f}   "
              f"(+{spawn_s - wall:.0f}s spawn/open)", flush=True)

    print("-" * 56)
    base = rows[0][1]
    best = max(rows, key=lambda r: r[1])
    gain = best[1] / base if base else 0
    print(f"1 process {base:.1f} MB/s  ->  {best[0]} processes "
          f"{best[1]:.1f} MB/s   ({gain:.2f}x)")
    if gain >= 1.5:
        print("PER-PROCESS cap: more worker processes per VM will help.")
    else:
        print("Scaling is flat across processes: the limit is the VM's path "
              "to AWS.\nMore workers, more threads or more RAM cannot move "
              "it -- only more VMs,\nor reading in-region.")


if __name__ == "__main__":
    main()
