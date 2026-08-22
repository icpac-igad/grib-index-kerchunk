#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "numpy"]
# ///
"""Which (date, step) does the published realized EA-SWIO store actually hold?

    https://source.coop/e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3

The time axis is PREALLOCATED to a whole MAM season. A date that was never
written is therefore not absent -- it is on the axis, `.sel(time=...)` finds it,
and it reads back as all-NaN. Nothing raises. A consumer asking for 2024-05-01
gets a field of NaN and no indication that it means "not published" rather than
"no rain".

So coverage cannot be read off `ds.sizes["time"]`, and it is not a data
question either: reading every field to look for NaN would move terabytes. The
manifest knows exactly which chunks exist, and `Session.chunk_coordinates`
enumerates them without fetching any of them.

Two independent methods, because a manifest scan alone only proves the manifest
agrees with itself:
  manifest -- chunk_coordinates per channel, cheap, exhaustive
  data     -- with --verify-data, actually read one field per verdict and
              confirm a "written" date is finite and a "missing" one is NaN

    uv run check_realized_coverage.py
    uv run check_realized_coverage.py --sub 49r1-mam2024 --all-channels --verify-data
    uv run check_realized_coverage.py --json coverage.json

Exit status is 1 if any sub-store is incompletely written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import icechunk
import numpy as np
import xarray as xr

BUCKET = "us-west-2.opendata.source.coop"
BASE = "e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3"
SUBS = ["49r1-mam2024", "49r1-mam2025", "49r1-mam2026", "50r1-mam2026-tail"]
PROBE = "t850"          # any channel present on every date


def open_sub(sub: str):
    """-> (session, dataset, tip_snapshot_id). Anonymous: no credentials at all."""
    st = icechunk.s3_storage(bucket=BUCKET, prefix=f"{BASE}/{sub}",
                             region="us-west-2", anonymous=True, from_env=False)
    # source.coop returns sporadic 500s on a few percent of GETs, and the default
    # open prefetches thousands of manifests in parallel -- one of them tends to
    # draw a 500 and the whole open raises. Disable the preload.
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))
    repo = icechunk.Repository.open(st, config=cfg)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False, chunks=None)
    return session, ds, session.snapshot_id


async def written_pairs(session, channel: str) -> set:
    """Every (time_index, step_index) this channel has a chunk for.

    The chunk grid is (1, 1, 51, lat, lon) -- one chunk per (date, step) holding
    all 51 members -- so the leading two coordinates identify the pair and the
    rest are always zero.
    """
    out = set()
    async for c in session.chunk_coordinates(f"/{channel}"):
        out.add(tuple(c)[:2])
    return out


def verify_data(ds, times, ti_written, ti_missing, channel):
    """Read one real field from each verdict. The manifest says these differ;
    this checks the bytes agree with it."""
    out = {}
    for label, ti in (("written", ti_written), ("missing", ti_missing)):
        if ti is None:
            continue
        f = ds[channel].isel(time=ti, step=1, number=0).values
        out[label] = {"date": times[ti], "finite_fraction": float(np.isfinite(f).mean()),
                      "min": None if not np.isfinite(f).any() else float(np.nanmin(f)),
                      "max": None if not np.isfinite(f).any() else float(np.nanmax(f))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", action="append", choices=SUBS,
                    help="repeatable; default all four")
    ap.add_argument("--channel", default=PROBE)
    ap.add_argument("--all-channels", action="store_true",
                    help="scan every channel, to prove gaps are whole dates")
    ap.add_argument("--verify-data", action="store_true",
                    help="also read one real field per verdict")
    ap.add_argument("--json", default=None, help="write the per-date map here")
    a = ap.parse_args()

    subs = a.sub or SUBS
    report, incomplete = {}, False
    print(f"s3://{BUCKET}/{BASE}\n")

    for sub in subs:
        session, ds, snap = open_sub(sub)
        times = [str(t)[:10] for t in ds.time.values]
        n_step = ds.sizes["step"]
        channels = sorted(ds.data_vars) if a.all_channels else [a.channel]

        per_channel, union_written = {}, None
        for ch in channels:
            if ds[ch].ndim < 5:          # lsm and friends carry no time axis
                continue
            pairs = asyncio.run(written_pairs(session, ch))
            by_time = {}
            for ti, si in pairs:
                by_time[ti] = by_time.get(ti, 0) + 1
            full = {ti for ti, n in by_time.items() if n == n_step}
            per_channel[ch] = {"n_full": len(full),
                               "n_partial": len(by_time) - len(full)}
            union_written = full if union_written is None else (union_written | full)

        written = sorted(union_written or set())
        missing = [i for i in range(len(times)) if i not in set(written)]
        pct = 100 * len(written) / len(times)
        flag = "COMPLETE" if not missing else "INCOMPLETE"
        incomplete |= bool(missing)
        print(f"{sub:20s} {len(written):>3d}/{len(times)} dates  ({pct:5.1f}%)  {flag}")
        print(f"    snapshot {snap}")
        if written:
            print(f"    written {times[written[0]]} .. {times[written[-1]]}")
        if missing:
            print(f"    missing {times[missing[0]]} .. {times[missing[-1]]}  "
                  f"({len(missing)} dates)")
        if a.all_channels:
            spread = {(v["n_full"], v["n_partial"]) for v in per_channel.values()}
            print(f"    {len(per_channel)} channels scanned; "
                  + ("all agree -- gaps are whole dates"
                     if len(spread) == 1 else f"DISAGREE: {spread}"))

        rec = {"n_axis": len(times), "n_written": len(written), "snapshot": snap,
               "written": [times[i] for i in written],
               "missing": [times[i] for i in missing],
               "channels_scanned": len(per_channel)}
        if a.verify_data:
            v = verify_data(ds, times, written[0] if written else None,
                            missing[0] if missing else None, a.channel)
            rec["data_check"] = v
            for label, d in v.items():
                print(f"    data {label:8s} {d['date']}  finite={d['finite_fraction']:.3f}"
                      + (f"  range [{d['min']:.1f}, {d['max']:.1f}]"
                         if d["min"] is not None else "  (all NaN)"))
        report[sub] = rec
        print()

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"-> {a.json}")
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
