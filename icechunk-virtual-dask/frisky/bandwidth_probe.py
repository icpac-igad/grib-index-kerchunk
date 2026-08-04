"""Where does read throughput stop scaling with concurrency?

The daily run sat at ~100 MB/s with 14% CPU and 4% RAM. Neither is the limit,
so the limit is one of: in-flight request count, the per-VM NIC, or the
cross-cloud path to AWS. This tells them apart before anything is retuned.

Method: read the SAME set of GRIB messages at rising concurrency, in one
process, and watch MB/s. The curve says which:

    rises to N=128 and flattens        -> we were request-starved; raise it
    flattens early at a round number   -> NIC or path ceiling; concurrency
                                          will not help and only AWS-region
                                          co-location will
    rises then FALLS                   -> throttling; the 503 path is being hit

Each read is one whole global message (~0.788 MB packed), which is the unit
the DAG uses, so the numbers transfer directly.

Usage
-----
    .venv/bin/python bandwidth_probe.py                    # 1,4,16,32,64,128
    .venv/bin/python bandwidth_probe.py --levels 64 256 --messages 512
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Same trap as the DAG: the virtual-chunk client must be built without AWS_*.
_STRIPPED = {k: os.environ.pop(k) for k in list(os.environ)
             if k.startswith("AWS_")}

import numpy as np  # noqa: E402

from frisky_daily_dag import (_open_era, LAT_MAX, LAT_MIN, LON_MIN,  # noqa: E402
                              LON_MAX, STEPS_ALL_H)

MSG_MB = 0.788          # measured packed size, materialize_ea_icechunk_ewc.py


def one_read(args):
    """One message, timed. Returns (seconds, ok, throttled)."""
    era, var, level, date, member, step_h = args
    t0 = time.time()
    try:
        ds = _open_era(era)
        da = ds[var]
        if level is not None:
            da = da.sel(isobaricInhPa=level)
        da = da.sel(time=np.datetime64(date), number=member,
                    step=np.timedelta64(step_h, "h"),
                    latitude=slice(LAT_MAX, LAT_MIN),
                    longitude=slice(LON_MIN, LON_MAX))
        np.asarray(da.values, dtype=np.float32)
        return time.time() - t0, True, False
    except Exception as exc:
        msg = str(exc)
        return (time.time() - t0, False,
                "SlowDown" in msg or "503" in msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-07-02")
    p.add_argument("--era", default="50r1")
    p.add_argument("--var", default="t2m")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--messages", type=int, default=256,
                   help="messages per concurrency level")
    p.add_argument("--levels", type=int, nargs="+",
                   default=[1, 4, 16, 32, 64, 128])
    args = p.parse_args()

    print(f"warming the store open (~5 s)...", flush=True)
    _open_era(args.era)

    # Distinct (member, step) pairs so nothing is served from a warm cache.
    work = []
    for i in range(args.messages):
        work.append((args.era, args.var, args.level, args.date,
                     i % 51, STEPS_ALL_H[(i // 51) % len(STEPS_ALL_H)]))

    print(f"\n{args.var}{args.level or ''}  {args.messages} messages "
          f"per level, {MSG_MB} MB each\n")
    print(f"{'conc':>5} {'wall':>8} {'msg/s':>8} {'MB/s':>8} {'Gbps':>7} "
          f"{'p50 s':>7} {'p95 s':>7} {'fail':>5} {'503':>4}")
    print("-" * 70)

    rows = []
    for conc in args.levels:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            out = list(ex.map(one_read, work))
        wall = time.time() - t0
        lat = sorted(t for t, ok, _ in out if ok)
        ok = sum(1 for _, o, _ in out if o)
        thr = sum(1 for _, _, t in out if t)
        mbs = ok * MSG_MB / wall
        rows.append((conc, mbs))
        print(f"{conc:5d} {wall:7.1f}s {ok / wall:8.1f} {mbs:8.1f} "
              f"{mbs * 8 / 1000:7.2f} "
              f"{(lat[len(lat) // 2] if lat else 0):7.2f} "
              f"{(lat[int(len(lat) * .95)] if lat else 0):7.2f} "
              f"{len(out) - ok:5d} {thr:4d}", flush=True)

    print("-" * 70)
    best = max(rows, key=lambda r: r[1])
    print(f"peak {best[1]:.1f} MB/s ({best[1] * 8 / 1000:.2f} Gbps) "
          f"at concurrency {best[0]}")
    if len(rows) < 2:
        print("single level -- says nothing about where it saturates")
    elif best[0] == args.levels[-1] and best[1] > rows[-2][1] * 1.1:
        print("still climbing at the top level -- raise --levels and retest")
    else:
        print(f"flat past ~{best[0]} concurrent reads; the limit is bandwidth, "
              f"not in-flight count")


if __name__ == "__main__":
    main()
