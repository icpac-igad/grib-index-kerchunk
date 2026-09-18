#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "icechunk>=2.1",
#     "zarr>=3.2",
#     "xarray>=2025.1",
#     "numpy",
#     "herbie-data",
#     "cfgrib",
#     "eccodes",
#     "matplotlib",
#     "cartopy",
# ]
# ///
"""Compare the PUBLISHED, REALIZED EA-SWIO Icechunk store against Herbie.

    https://source.coop/e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3

`compare_icechunk_herbie.py` validates the **virtual** store, whose chunks are
`[url, offset, length]` references decoded from ECMWF's GRIB on every read. This
one validates the **realized** store, where the values were decoded once, cut to
the EA-SWIO box, and written as real chunks. That is a different set of things
that can go wrong, and none of them are covered by the virtual-store check:

  * the subset could be taken on the wrong window, or off by a row;
  * the flattening of `(var, level)` into channel names (`t` @ 850 -> `t850`)
    could mis-assign a level -- the exact failure the per-level-keys fix
    addressed upstream, re-introduced at a different layer;
  * the dim order is `(time, step, number, ...)` here but `(time, number, step,
    ...)` in the virtual store, so a transposed write would be silent;
  * `step` is stored in SECONDS here and hours there.

Read anonymously -- no credentials, that is the point of publishing it. Compared
per member against Herbie, which fetches the same GRIB messages straight from
ECMWF's open-data mirror and knows nothing about any of the above.

    uv run compare_realized_herbie.py --sub 49r1-mam2024 --date 20240327 \
        --step 48 --var t850
    uv run compare_realized_herbie.py --list          # what dates exist

Exit status is 1 if any case falls outside tolerance.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import icechunk

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the Herbie fetch and the statistics rather than re-deriving them. The
# fetch is domain-agnostic -- it selects whatever coordinate labels it is
# handed -- and re-typing it here is how a constant ends up wrong in one of
# twelve copies (grids.py, HANDOVER_LONGITUDE_FIX.md).
from compare_icechunk_herbie import (  # noqa: E402
    herbie_field, field_stats, per_member_stats, VAR_LABELS, VAR_UNITS)

warnings.filterwarnings("ignore")

BUCKET = "us-west-2.opendata.source.coop"
BASE = "e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3"
SUBS = ["49r1-mam2024", "49r1-mam2025", "49r1-mam2026", "50r1-mam2026-tail"]

# Surface channels keep their own names; everything else is <param><level>.
# This set has to be consulted FIRST: 't2m', 'u10' and 'v10' both match the
# <param><level> pattern and would parse as t@2 and u@10 hPa.
SURFACE = {"t2m", "u10", "v10", "msl", "sp", "skt", "tp", "ro", "tcwv",
           "lsm", "d2m", "tcc", "ssr", "ssrd", "sf"}
PL_BASES = {"t", "u", "v", "q", "r", "d", "vo", "gh", "w"}


def split_channel(name: str) -> tuple[str, int | None]:
    """'t850' -> ('t', 850); 't2m' -> ('t2m', None). Raises on anything else."""
    if name in SURFACE:
        return name, None
    m = re.fullmatch(r"([a-z]+)(\d{2,4})", name)
    if m and m.group(1) in PL_BASES:
        return m.group(1), int(m.group(2))
    raise SystemExit(f"cannot map channel {name!r} to a (param, level)")


def open_sub(sub: str) -> xr.Dataset:
    st = icechunk.s3_storage(bucket=BUCKET, prefix=f"{BASE}/{sub}",
                             region="us-west-2", anonymous=True, from_env=False)
    # source.coop returns sporadic 500s on a few percent of GETs; the default
    # open prefetches thousands of manifests at once and one of them draws a
    # 500. See v20260819-icechunk-store-source-coop.md.
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))
    repo = icechunk.Repository.open(st, config=cfg)
    return xr.open_zarr(repo.readonly_session("main").store, zarr_format=3,
                        consolidated=False, chunks=None)


def store_field(ds: xr.Dataset, var: str, date: str, step_h: int):
    """-> (values[number, lat, lon], lats, lons, numbers), selected by label."""
    t = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00", "ns")
    if t not in ds.time.values:
        raise SystemExit(f"{date} not on this sub-store's time axis")
    da = ds[var].sel(time=t, step=np.timedelta64(step_h * 3600, "s"))
    # the realized store is (time, step, number, lat, lon); make member first
    da = da.transpose("number", "latitude", "longitude")
    return (da.values.astype(np.float32), da.latitude.values,
            da.longitude.values, da.number.values)


def plot_compare(date, sub, var, step_h, lats, lons,
                 s_mean, s_std, h_mean, h_std, n_s, n_h, out_dir):
    param, level = split_channel(var)
    units = VAR_UNITS.get(param if level else var, "")
    label = VAR_LABELS.get(param if level else var, var)
    lat_hi, lat_lo = float(lats.max()), float(lats.min())
    lon_lo, lon_hi = float(lons.min()), float(lons.max())

    # Explicit margins, and NO bbox_inches="tight" on save. The domain is taller
    # than it is wide, and cartopy's gridline labels are not counted in the tight
    # bbox -- the combination cropped the whole first column off the left edge,
    # title and all, while leaving the other two panels looking correct.
    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.04],
                           left=0.05, right=0.94, top=0.89, bottom=0.05,
                           hspace=0.12, wspace=0.12)
    for row, (s, h, rlabel, cmap) in enumerate([
            (s_mean, h_mean, "Ensemble mean", "RdBu_r"),
            (s_std, h_std, "Ensemble spread", "viridis")]):
        vext = np.nanmax(np.abs([np.nanmin([s, h]), np.nanmax([s, h])]))
        vmin, vmax = (-vext, vext) if row == 0 else (0, np.nanmax([s, h]))
        diff = s - h
        dext = max(float(np.nanmax(np.abs(diff))), 1e-12)
        panels = [(s, f"Realized store ({n_s}m) — {rlabel}", cmap, vmin, vmax),
                  (h, f"Herbie ({n_h}m) — {rlabel}", cmap, vmin, vmax),
                  (diff, "Difference (store − Herbie)", "RdBu_r", -dext, dext)]
        last = None
        for col, (data, title, cm, vmn, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            im = ax.pcolormesh(lons, lats, data, cmap=cm, vmin=vmn, vmax=vmx,
                               transform=ccrs.PlateCarree(), shading="auto")
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
            ax.set_extent([lon_lo, lon_hi, lat_lo, lat_hi], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=(col == 0), linewidth=0.3, alpha=0.4)
            gl.top_labels = gl.right_labels = False
            # NOT ax.set_title: cartopy's Gridliner suppresses the axes title
            # whenever draw_labels=True, silently and only on the labelled
            # panel -- matplotlib still reports the title present and visible.
            # A plain text artist is independent of the gridliner.
            ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
            if col == 2:
                last = im
        cb = fig.colorbar(last, cax=fig.add_subplot(gs[row, 3]))
        cb.set_label(units, fontsize=9)

    lev = f" @ {level} hPa" if level else ""
    fig.suptitle(f"Realized EA-SWIO store vs Herbie — {label}{lev} — {sub} — "
                 f"{date[:4]}-{date[4:6]}-{date[6:]} 00Z  T+{step_h}h\n"
                 f"{lon_lo:g}–{lon_hi:g}°E, {abs(lat_lo):g}°S–{lat_hi:g}°N",
                 fontsize=13, fontweight="bold", y=0.97)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"realized_{sub}_{var}_{date}_T{step_h}h.png"
    plt.savefig(out, dpi=130, facecolor="white")
    plt.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", default="49r1-mam2024", choices=SUBS)
    ap.add_argument("--date", help="YYYYMMDD (must exist in --sub)")
    ap.add_argument("--step", type=int, default=48, help="forecast hour")
    ap.add_argument("--var", default="t850", help="store channel, e.g. t850, tp, t2m")
    ap.add_argument("--list", action="store_true", help="print each sub-store's dates")
    ap.add_argument("--output-dir", default="gik_vs_herbie/realized_ea_swio_eval")
    a = ap.parse_args()

    if a.list:
        for sub in SUBS:
            ds = open_sub(sub)
            t = [str(x)[:10] for x in ds.time.values]
            print(f"{sub:20s} {len(t):>3d} dates  {t[0]} .. {t[-1]}  "
                  f"steps {ds.sizes['step']}  members {ds.sizes['number']}")
        return 0
    if not a.date:
        ap.error("--date is required (or use --list)")

    ds = open_sub(a.sub)
    if a.var not in ds:
        print(f"{a.var!r} not in {a.sub}; have: {sorted(ds.data_vars)}")
        return 2
    param, level = split_channel(a.var)

    s_v, lats, lons, s_num = store_field(ds, a.var, a.date, a.step)
    print(f"\n[{a.sub} {a.date} T+{a.step}h {a.var}]")
    print(f"  store : {s_v.shape[0]} members, grid {s_v.shape[1]}x{s_v.shape[2]}, "
          f"lat {lats[0]:.2f}..{lats[-1]:.2f}, lon {lons[0]:.2f}..{lons[-1]:.2f}")

    # The time axis is PREALLOCATED to the whole season, so a date that was
    # never written is on the axis and reads as all-NaN rather than raising.
    # Say so and stop: downloading 50 GRIB messages to compare against nothing
    # wastes minutes, and the ensemble stats would go on to die on an empty
    # field with a KeyError that names the wrong problem.
    if np.all(np.isnan(s_v)):
        print(f"  store : NOT WRITTEN -- every member is NaN at this (date, step).")
        print(f"          {a.sub} has unwritten dates; use --list and coverage.json,")
        print(f"          or compare this date against the virtual store instead.")
        print("RESULT: UNWRITTEN")
        return 3

    # Herbie takes the GRIB name; for a pl channel that is the base + level.
    h_v, h_num = herbie_field(a.date, "00", a.step, param if level else a.var,
                              level, lats, lons)
    pm = per_member_stats(s_v, s_num, h_v, h_num)
    ok = pm["n_compared"] > 0 and pm["max_rel_diff"] < 1e-4 and pm["n_store_all_nan"] == 0
    print(f"  per-member: {pm['n_compared']} matched | min r={pm['min_corr']:.8f} "
          f"| max|diff|={pm['max_abs_diff']:.3g} ({pm['max_rel_diff']:.2e} of range)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    s_mean, s_std = np.nanmean(s_v, axis=0), np.nanstd(s_v, axis=0)
    h_mean, h_std = np.nanmean(h_v, axis=0), np.nanstd(h_v, axis=0)
    rec = {"store": f"s3://{BUCKET}/{BASE}/{a.sub}", "sub": a.sub, "date": a.date,
           "step": a.step, "var": a.var, "param": param, "level": level,
           "n_store_members": int(s_v.shape[0]), "n_herbie_members": int(h_v.shape[0]),
           "lat_range": [float(lats[0]), float(lats[-1])],
           "lon_range": [float(lons[0]), float(lons[-1])],
           "per_member": pm,
           "ensemble_mean": field_stats(s_mean, h_mean),
           "ensemble_spread": field_stats(s_std, h_std)}
    m = rec["ensemble_mean"]
    if "corr" in m:
        print(f"  ensemble  : {s_v.shape[0]}m vs {h_v.shape[0]}m | mean r={m['corr']:.8f} "
              f"RMSE={m['rmse']:.3g} max|diff|={m['max_abs_diff']:.3g}")
    else:
        print(f"  ensemble  : {m.get('error', 'no statistics')}")

    out_dir = Path(a.output_dir)
    p = plot_compare(a.date, a.sub, a.var, a.step, lats, lons,
                     s_mean, s_std, h_mean, h_std, s_v.shape[0], h_v.shape[0], out_dir)
    print(f"  plot -> {p}")
    sf = out_dir / f"realized_{a.sub}_{a.var}_{a.date}_T{a.step}h.json"
    sf.write_text(json.dumps(rec, indent=2, default=float))
    print(f"  stats -> {sf}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
