#!/usr/bin/env python
"""Read-only check of a proposed domain + channel set against the source store.

Anonymous: touches source.coop and the public AWS GRIB bytes only.  Writes
nothing.  Answers the three questions that must be true before a realization
run is worth starting:

  1. is the date on the era's `time` axis at all
  2. does every (var, level) in the channel set decode for THAT date -- the
     group's variable list is the union across the whole era, so schema
     membership is not enough (ECMWF_VARIABLES_REFERENCE.md, "the single most
     important caveat")
  3. what shape does the box actually cut, and what does that cost

    $P validate_domain.py --date 2026-03-01 --era 49r1
"""
import argparse
import sys
import time

import numpy as np

import frisky_daily_dag as dag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-03-01")
    p.add_argument("--era", default="49r1")
    p.add_argument("--member", type=int, default=1)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--channels", type=int, default=len(dag.CHANNELS))
    args = p.parse_args()

    chans = dag.CHANNELS[:args.channels]
    print(f"domain   lat {dag.LAT_MAX} -> {dag.LAT_MIN}  "
          f"lon {dag.LON_MIN} -> {dag.LON_MAX}")
    print(f"channels {len(chans)}")
    print(f"date     {args.date}  era {args.era}  "
          f"member {args.member}  step +{args.step}h")
    print()

    t0 = time.time()
    ds = dag._open_era(args.era)
    print(f"store open: {time.time() - t0:.1f}s")

    # --- 1. date on the axis --------------------------------------------
    have = set(np.datetime_as_string(ds.time.values, "D").tolist())
    on_axis = args.date in have
    print(f"date on axis: {'YES' if on_axis else 'NO'}  "
          f"({len(have)} dates in {args.era}/00z)")
    if not on_axis:
        print("ABORT: source store does not carry this date")
        return 1

    # --- 2. box shape ----------------------------------------------------
    box = ds["t2m"].sel(latitude=slice(dag.LAT_MAX, dag.LAT_MIN),
                        longitude=slice(dag.LON_MIN, dag.LON_MAX))
    ny, nx = box.sizes["latitude"], box.sizes["longitude"]
    print(f"box: {ny} lat x {nx} lon = {ny * nx:,} points")
    print(f"     lat {box.latitude.values[0]} -> {box.latitude.values[-1]}")
    print(f"     lon {box.longitude.values[0]} -> {box.longitude.values[-1]}")
    print()

    # --- 3. the member axis, control included ----------------------------
    num = ds.number.values
    print(f"number axis: {len(num)} members, "
          f"{num.min()} .. {num.max()}")
    c0 = dag.read_message(args.era, "msl", None, args.date, int(num.min()),
                          args.step)
    c1 = dag.read_message(args.era, "msl", None, args.date, int(num.min()) + 1,
                          args.step)
    same = bool(np.allclose(c0, c1))
    print(f"  member {num.min()} vs {num.min() + 1} on msl: "
          f"{'IDENTICAL -- broadcast?' if same else 'differ (control is real)'}"
          f"  rms {float(np.sqrt(((c0 - c1) ** 2).mean())):.4g} Pa")
    print()

    # --- 4. schema membership, then decode -------------------------------
    levels = set(int(x) for x in ds.isobaricInhPa.values)
    absent = [(v, lv, n) for v, lv, n in chans
              if v not in ds.data_vars or (lv is not None and lv not in levels)]
    if absent:
        print("NOT IN SCHEMA (whole era):")
        for v, lv, n in absent:
            why = "var absent" if v not in ds.data_vars else f"level {lv} absent"
            print(f"  {n:<8} {v}@{lv}  -- {why}")
        print()

    print(f"{'channel':<8} {'var':<6} {'lev':>5} {'shape':>10} "
          f"{'finite':>7} {'min':>12} {'max':>12} {'s':>6}")
    bad = []
    for var, lev, name in chans:
        if (var, lev, name) in absent:
            bad.append((name, "not in schema"))
            continue
        t = time.time()
        try:
            a = dag.read_message(args.era, var, lev, args.date,
                                 args.member, args.step)
        except Exception as exc:
            print(f"{name:<8} {var:<6} {str(lev):>5}   FAILED: "
                  f"{str(exc)[:60]}")
            bad.append((name, str(exc)[:80]))
            continue
        fin = float(np.isfinite(a).mean())
        lo = float(np.nanmin(a)) if fin else float("nan")
        hi = float(np.nanmax(a)) if fin else float("nan")
        print(f"{name:<8} {var:<6} {str(lev):>5} {str(a.shape):>10} "
              f"{fin:>7.3f} {lo:>12.4g} {hi:>12.4g} {time.time() - t:>6.2f}")
        if fin == 0.0:
            bad.append((name, "all-NaN for this date"))
        elif a.shape != (ny, nx):
            bad.append((name, f"shape {a.shape} != ({ny}, {nx})"))

    # --- 5. what a full date would cost ----------------------------------
    n_mem, n_step = len(num), len(dag.STEPS_ALL_H[:53])
    n_msg = len(chans) * n_mem * n_step
    chunk_mb = n_mem * ny * nx * 4 / 1e6
    print()
    print(f"one date at {len(chans)} ch x {n_mem} mem x {n_step} steps:")
    print(f"  messages        {n_msg:,}")
    print(f"  blocks (chunks) {len(chans) * n_step:,}")
    print(f"  chunk size      {chunk_mb:.1f} MB  (51, {ny}, {nx}) float32")
    print(f"  raw output      {len(chans) * n_step * chunk_mb / 1000:.1f} GB "
          f"(~{len(chans) * n_step * chunk_mb / 1000 * 0.40:.1f} GB stored "
          f"at 40% compression)")
    print(f"  AWS egress      ~{n_msg * 0.788 / 1000:.0f} GB "
          f"(whole global message per read)")

    print()
    if bad:
        print(f"VERDICT: {len(bad)} channel(s) unusable on {args.date}")
        for n, why in bad:
            print(f"  {n:<8} {why}")
        return 1
    print(f"VERDICT: all {len(chans)} channels decode on {args.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
