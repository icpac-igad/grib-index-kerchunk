#!/usr/bin/env python
"""Add channels to a FINISHED realized store, without rebuilding it.

The 36-channel table was chosen era-portable -- 18 base variables common to
`0p4`, `49r1` and `50r1` -- and that deliberately dropped `w` (absent from
0p4) and `mucape`. Right for archive portability, wrong for this model: all
three MAM seasons built are 0.25 deg eras, `0p4` never enters, and the
convective pair is the one channel the PyTorch-EP cGAN set needs and the store
does not have (CGAN_VARIABLE_GAP.md).

Rebuilding at 38 channels is ~56 h. Appending two is ~2.5 h, because channels
scale the message count proportionally:

    2 ch x 51 members x 65 steps  =   6,630 messages/date
    36 ch                         = 119,340   -> 5.6% of a full run

Icechunk commits are additive, so this is safe and cheap: `to_zarr(mode="a")`
writes only the NEW arrays' metadata -- a few KB -- and the resulting snapshot
references every existing chunk unchanged. Verified on a scratch store: after
appending two channels to a finished two-channel store, the originals read
back bit-identical and the coordinates are unchanged.

    THE THING THIS MUST NEVER DO IS CALL create_schema().

That writes mode="w", and on 2026-08-07 an invocation that merely forgot a
flag laid a fresh schema over a live store and made 30 committed dates
unreachable in one commit (EWC_REALIZE_NOTES.md, Corrections §6). This module
does not import it. The guard is structural, not a check.

Era-aware CAPE
--------------
CAPE is not one variable across the archive. Measured by decode, not read off
the era's variable union:

    season             cape    mucape   w700
    MAM 2024 (49r1)    ok      NaN      ok
    MAM 2025 (49r1)    NaN     ok       ok
    MAM 2026 (49r1)    NaN     ok       ok
    MAM 2026 (50r1)    absent  ok       ok

So `w` is unconditional, and one output channel `cape` is fed by `cape` before
2025-01-01 and `mucape` after. Which source fed which date is recorded in the
array's attributes, so nobody has to rediscover the swap.

    $P append_channels.py --sink ea-swio/v3-49r1-mam2024 --era 49r1 \
        --start 2024-03-01 --days 92 --env .env --dry-run
    $P append_channels.py --sink ea-swio/v3-49r1-mam2024 --era 49r1 \
        --start 2024-03-01 --days 92 --env .env
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

import build_corpus as bc          # date_range, run_one_date, committed_dates
import frisky_daily_dag as dag
import sink_icechunk as sink       # load_env, open_sink, merge_and_commit ONLY

# (out_name, base_var_or_None, level) -- base_var None => resolved per date.
NEW_CHANNELS = [
    ("w700", "w", 700),
    ("cape", None, None),          # `cape` <= 2024-12-31, `mucape` after
]
CAPE_SWITCH = "2025-01-01"         # first date on which mucape replaces cape


def cape_source(date):
    return "cape" if date < CAPE_SWITCH else "mucape"


def channels_for(date, names):
    """The (var, level, out_name) triples run_one_date expects, for one date."""
    out = []
    for name, var, lev in NEW_CHANNELS:
        if name not in names:
            continue
        out.append((cape_source(date) if var is None else var, lev, name))
    return out


def add_arrays(repo, names, coords, dates):
    """Create the new arrays, metadata only, with mode='a'.

    mode='a' adds variables to the existing group. It does not touch the 36
    already there: no read, no rewrite, and the new snapshot references their
    chunks unchanged. This is the whole reason an append is minutes of
    metadata rather than a rebuild.
    """
    import dask.array as dsa
    import xarray as xr

    shape = (len(dates), len(coords["step"]), len(coords["number"]),
             len(coords["latitude"]), len(coords["longitude"]))
    chunks = (1, 1, shape[2], shape[3], shape[4])
    dims = ("time", "step", "number", "latitude", "longitude")
    ds = xr.Dataset(
        {n: (dims, dsa.zeros(shape, chunks=chunks, dtype="float32"))
         for n in names},
        coords={"time": np.array([np.datetime64(d, "ns") for d in dates]),
                "step": coords["step"], "number": coords["number"],
                "latitude": coords["latitude"],
                "longitude": coords["longitude"]},
    )
    for n in names:
        if n == "cape":
            ds[n].attrs["source_variable"] = (
                f"cape for dates < {CAPE_SWITCH}, mucape on/after; "
                f"the ECMWF field changed mid-49r1")
        else:
            src = next(v for nm, v, _ in NEW_CHANNELS if nm == n)
            ds[n].attrs["source_variable"] = src
    session = repo.writable_session("main")
    ds.to_zarr(session.store, compute=False, zarr_format=3,
               consolidated=False, mode="a")          # NEVER "w"
    return session.commit(f"append schema: {', '.join(sorted(names))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sink", required=True)
    p.add_argument("--era", required=True, choices=["0p4", "49r1", "50r1"])
    p.add_argument("--start", required=True)
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--channels", nargs="+",
                   default=[n for n, _, _ in NEW_CHANNELS])
    p.add_argument("--members", type=int, default=51)
    p.add_argument("--steps", type=int, default=65)
    p.add_argument("--scheduler", default="192.168.1.74:8796")
    p.add_argument("--env", default=".env")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--pause", type=int, default=45)
    p.add_argument("--resume", action="store_true",
                   help="skip dates this append already committed, and allow "
                        "the channels to exist. Without it an interrupted run "
                        "cannot be restarted: the clash guard blocks it")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import frisky

    dates = bc.date_range(args.start, args.days)
    members = list(range(args.members))
    steps = dag.STEPS_ALL_H[:args.steps]
    names = list(args.channels)

    ak, sk = sink.load_env(args.env)
    repo, created = sink.open_sink(args.sink, ak, sk, create=False)
    if created:
        sys.exit(f"{args.sink} does not exist -- append is for a FINISHED "
                 f"store; build it with build_corpus.py first")

    import xarray as xr
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False,
                      zarr_format=3, decode_timedelta=True)
    have = set(ds.data_vars)
    t_axis = np.datetime_as_string(ds.time.values, "D").tolist()

    print(f"sink     must-icechunk/{args.sink}")
    print(f"existing {len(have)} channels, {len(t_axis)} dates "
          f"{t_axis[0]} .. {t_axis[-1]}")

    # The new arrays must land on the store's OWN time axis, not a recomputed
    # one -- a mismatch would write into the wrong date index silently.
    if t_axis != dates:
        sys.exit(f"ABORT: --start/--days give {len(dates)} dates "
                 f"({dates[0]}..{dates[-1]}) but the store's axis is "
                 f"{len(t_axis)} ({t_axis[0]}..{t_axis[-1]}). They must match.")

    clash = [n for n in names if n in have]
    if clash and not args.resume:
        sys.exit(f"ABORT: {clash} already in the store. Appending would "
                 f"rewrite them; drop them from --channels, or pass --resume "
                 f"if this is restarting an interrupted append.")

    # Resume by READING THE DATA, not by trusting the commit log.
    #
    # The first version matched append commits by channel count ("date X: 2
    # channels" vs the build's 36) and skipped anything committed. That was
    # wrong and it cost five dates: a run killed mid-store left five commits
    # whose data never landed, resume skipped them on the strength of the
    # commit, and 2024-03-01..05 stayed all-NaN in both new channels while
    # every check reported success.
    #
    # A commit is not evidence of content. build_corpus.py can lean on the
    # commit log because its dates are all-or-nothing; this cannot. So probe
    # the array: one chunk per date, all-NaN means unwritten. That is ~17 MB
    # per date and a minute per store -- the right price for a check whose
    # whole job is to be true.
    done = set()
    if args.resume:
        probe = ds[names[0]]
        t_probe = time.time()
        for i, d in enumerate(dates):
            v = probe.isel(time=i, number=0).isel(step=0).values
            if np.isfinite(v).any():
                done.add(d)
        print(f"resume   {len(done)}/{len(dates)} date(s) already hold data "
              f"in `{names[0]}`, skipping ({time.time()-t_probe:.0f}s to check)")
        missing = [d for d in dates if d not in done]
        if missing and len(missing) < len(dates):
            print(f"         redoing {len(missing)}: {missing[0]} .. {missing[-1]}")

    for d in (dates[0], dates[-1]):
        print(f"  {d}: cape <- {cape_source(d)}")
    n_msg = len(dates) * len(names) * len(members) * len(steps)
    print(f"append   {names}  ->  {len(dates)} dates x {len(names)} ch x "
          f"{len(members)} mem x {len(steps)} steps = {n_msg:,} messages")
    print(f"         ~{n_msg/200/3600:.1f} h at 200 msg/s, "
          f"~{len(dates)*len(names)*len(steps)*len(members)*dag.NY*dag.NX*4*0.369/1e9:.0f} GB stored")

    if args.dry_run:
        print("\nDRY RUN -- nothing written")
        return 0

    # subset_coords opens the SOURCE for the box coordinates; reuse the store's
    # own step/number so the new arrays align exactly with the existing ones.
    coords = dag.subset_coords(args.era, dates[0], members, steps)
    coords["step"] = ds.step.values
    coords["number"] = ds.number.values
    coords["latitude"] = ds.latitude.values
    coords["longitude"] = ds.longitude.values

    if clash and args.resume:
        print(f"\nschema   arrays already present, not re-created")
    else:
        snap = add_arrays(repo, names, coords, dates)
        print(f"\nschema   {len(names)} arrays appended (metadata only), "
              f"{str(snap)[:12]}")

    client = frisky.Client(args.scheduler)
    t0 = time.time()
    ok = []
    for i, date in enumerate(dates):
        if date in done:
            continue
        chans = channels_for(date, names)
        t = time.time()
        good = bc.run_one_date(client, repo, chans, members, steps,
                               args.era, date, i, args.rounds, args.pause)
        rate = len(chans) * len(members) * len(steps) / (time.time() - t)
        print(f"[{i+1:2d}/{len(dates)}] {date}  {time.time()-t:6.1f}s  "
              f"{rate:6.1f} msg/s  {'ok' if good else 'PARTIAL'}", flush=True)
        ok.append(bool(good))
    client.close()
    print(f"\n{sum(ok)}/{len(dates)-len(done)} dates in {(time.time()-t0)/60:.1f} min")
    print(f"re-mirror with mirror_ceph_to_source_coop.py -- it transfers only "
          f"the new objects, so ~{n_msg*dag.NY*dag.NX*4*0.369/len(members)/1e9:.0f} GB, not the whole store")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
