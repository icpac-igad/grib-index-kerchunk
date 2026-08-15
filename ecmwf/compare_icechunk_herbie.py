#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "icechunk>=2.1",
#     "zarr>=3.2",
#     "xarray>=2025.1",
#     "gribberish>=1.4",
#     "numpy",
#     "herbie-data",
#     "cfgrib",
#     "eccodes",
#     "matplotlib",
#     "cartopy",
# ]
# ///
"""Compare the ECMWF **Icechunk** store against Herbie (East Africa / ICPAC).

This is the Icechunk companion to ``compare_gik_herbie_pressure.py``, which
compares the *par* files. The par files are the input to the store, so a par
comparison says nothing about what the conversion did to them -- the axes it
labelled, the member it filed a field under, the index it wrote a date at.
Those are exactly the things that have gone wrong before:

  * every store built before ``HANDOVER_LONGITUDE_FIX.md`` was labelled 180 deg
    out, so a subset over East Africa silently returned the eastern Pacific.
    The pars were fine throughout;
  * the step axis is a UNION across members, so a member missing a step lands as
    a silent NaN (``verify_store_completeness.py``).

So this script reads the store the way a consumer does -- ``xr.open_zarr`` then
``.sel(latitude=..., longitude=...)`` **by label**, never by integer index -- and
holds it against the same GRIB messages fetched independently by Herbie.

Two comparisons, because they fail differently:

  per-member   store ``number=n`` vs Herbie ``number=n``, n = 1..50, element-wise.
               Same source bytes, so this must be ~exact. A permuted member axis
               shows up here as r ~ 0 and is INVISIBLE to an ensemble-mean check.
  ensemble     mean and spread, all 51 store members vs Herbie's 50 perturbed.
               Comparable to the published par-vs-Herbie table; the ~0.6-0.9%
               offset is the control member, which Herbie's enfo does not return.

Usage:
    uv run compare_icechunk_herbie.py --store gs://bucket/icechunk/ecmwf-ens-v4 \
        --era 49r1 --run 00 --date 20240327 --step 0 --var t --levels 500,850 \
        --sa-key /tmp/frisky-ea/gcs-key.json

    # surface variable -- --levels is ignored
    uv run compare_icechunk_herbie.py ... --var t2m --step 0

Exit status is 1 if any level fails the tolerance, so this can gate a publish.
"""
from __future__ import annotations

import argparse
import json
import os
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
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

warnings.filterwarnings("ignore")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Store name -> GRIB shortName. The builder renames three surface fields
# (build_ecmwf_icechunk.SFC_RENAME); Herbie searches the raw .index, which
# carries the GRIB name.
TO_GRIB = {"t2m": "2t", "u10": "10u", "v10": "10v", "d2m": "2d"}

VAR_LABELS = {"u": "U-wind", "v": "V-wind", "gh": "Geopotential height",
              "t": "Temperature", "q": "Specific humidity", "r": "Relative humidity",
              "t2m": "2 m temperature", "tp": "Total precipitation",
              "u10": "10 m U-wind", "v10": "10 m V-wind", "msl": "Mean sea-level pressure"}
VAR_UNITS = {"u": "m/s", "v": "m/s", "gh": "gpm", "t": "K", "q": "kg/kg", "r": "%",
             "t2m": "K", "tp": "m", "u10": "m/s", "v10": "m/s", "msl": "Pa"}

# Per-member agreement is the same bytes decoded twice, so the only permitted
# difference is float32 packing noise. Relative to the field's own range.
PER_MEMBER_RTOL = 1e-4


# -- store side --------------------------------------------------------------

def open_store(store: str, era: str, run: str, sa_key: str | None) -> xr.Dataset:
    if store.startswith("gs://"):
        bucket, _, prefix = store[5:].partition("/")
        st = icechunk.gcs_storage(
            bucket=bucket, prefix=prefix.rstrip("/"),
            service_account_file=sa_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    else:
        st = icechunk.local_filesystem_storage(store)
    auth = icechunk.containers_credentials(
        {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(st, authorize_virtual_chunk_access=auth)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, group=f"{era}/{run}z", zarr_format=3,
                        consolidated=False, chunks=None)


def store_field(ds: xr.Dataset, var: str, date: str, step: int, level: int | None):
    """-> (values[number, lat, lon], lats, lons, numbers).

    Selection is by LABEL on every axis. That is the point: an integer index
    would read the same bytes whatever the coordinate said, which is how a
    180 deg longitude error survived every previous check.
    """
    da = ds[var]
    t = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00", "ns")
    if t not in ds.time.values:
        raise SystemExit(f"{date} not on the {var} time axis of this group")
    da = da.sel(time=t, step=step)
    if "isobaricInhPa" in da.dims:
        if level is None:
            raise SystemExit(f"{var} is a pressure-level field; pass --levels")
        da = da.sel(isobaricInhPa=float(level))
    da = da.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    return (da.values.astype(np.float32), da.latitude.values, da.longitude.values,
            da.number.values)


# -- Herbie side -------------------------------------------------------------

def herbie_field(date: str, run: str, step: int, var: str, level: int | None,
                 lats, lons):
    """-> (values[number, lat, lon], numbers) on the store's own grid.

    Reindexed onto the store's coordinate arrays, so if the store's axes are
    wrong the diff is huge -- which is the intended failure.
    """
    from herbie import Herbie
    param = TO_GRIB.get(var, var)
    search = f":{param}:{level}:pl:" if level is not None else f":{param}:sfc:"
    H = Herbie(f"{date[:4]}-{date[4:6]}-{date[6:]} {run}:00",
               model="ifs", product="enfo", fxx=step)
    ds = H.xarray(search, verbose=False)
    if isinstance(ds, list):
        cand = [d for d in ds if "number" in d.dims]
        if not cand:
            raise RuntimeError(f"no perturbed-member dataset for {search}")
        ds = cand[0]
    name = param if param in ds.data_vars else list(ds.data_vars)[0]
    da = ds[name]
    if da.longitude.max() > 180:
        da = da.assign_coords(
            longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
    if "number" not in da.dims:
        da = da.expand_dims("number")
    # Select the store's own labels directly, with a tolerance far below half a
    # cell (0.125 deg on the coarsest grid). Do NOT slice-then-reindex: eccodes
    # reports the 0.4 deg axis as -14.000000000000057, so slice(25, -14) drops
    # the last row, and a bare method="nearest" reindex then back-fills it from
    # -13.6 -- a duplicated row that reads as a real 0.56 K disagreement at the
    # domain edge. With a tolerance, a genuinely absent label raises instead.
    da = da.sel(latitude=lats, longitude=lons, method="nearest", tolerance=0.01)
    da = da.sortby("number")
    out = da.values.astype(np.float32)
    if out.shape[-2:] != (len(lats), len(lons)):
        raise RuntimeError(f"herbie grid {out.shape[-2:]} != store "
                           f"{(len(lats), len(lons))}")
    return out, da.number.values


# -- stats -------------------------------------------------------------------

def field_stats(a, b) -> dict:
    diff = a - b
    ok = ~(np.isnan(a) | np.isnan(b))
    av, bv = a[ok], b[ok]
    if av.size == 0:
        return {"error": "no overlapping valid pixels"}
    r = (float(np.corrcoef(av, bv)[0, 1])
         if av.std() > 0 and bv.std() > 0 else float("nan"))
    return {"corr": r,
            "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
            "mae": float(np.nanmean(np.abs(diff))),
            "max_abs_diff": float(np.nanmax(np.abs(diff))),
            "n_valid": int(av.size),
            "store_range": [float(np.nanmin(a)), float(np.nanmax(a))],
            "herbie_range": [float(np.nanmin(b)), float(np.nanmax(b))]}


def per_member_stats(store_v, store_num, herb_v, herb_num) -> dict:
    """Element-wise store-vs-Herbie for every member both sides carry."""
    si = {int(n): i for i, n in enumerate(store_num)}
    worst = {"max_abs_diff": -1.0}
    rows, missing = [], []
    for j, n in enumerate(herb_num):
        n = int(n)
        if n not in si:
            missing.append(n)
            continue
        a, b = store_v[si[n]], herb_v[j]
        if np.all(np.isnan(a)):
            rows.append({"number": n, "store_all_nan": True})
            continue
        s = field_stats(a, b)
        s["number"] = n
        rows.append(s)
        if s.get("max_abs_diff", -1) > worst["max_abs_diff"]:
            worst = s
    graded = [r for r in rows if "max_abs_diff" in r]
    span = float(max(abs(np.nanmax(store_v)), abs(np.nanmin(store_v)), 1e-12))
    worst_diff = max((r["max_abs_diff"] for r in graded), default=float("nan"))
    return {"n_compared": len(graded),
            "n_store_all_nan": sum(1 for r in rows if r.get("store_all_nan")),
            "n_herbie_only": len(missing),
            "min_corr": min((r["corr"] for r in graded), default=float("nan")),
            "max_abs_diff": worst_diff,
            "max_rel_diff": worst_diff / span,
            "worst_member": worst.get("number"),
            "members": rows}


# -- plot --------------------------------------------------------------------

def _map(ax, data, lons, lats, title, cmap, vmin, vmax):
    im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="auto")
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def plot_compare(date, era, var, level, step, lats, lons,
                 s_mean, s_std, h_mean, h_std, n_s, n_h, out_dir):
    units = VAR_UNITS.get(var, "")
    label = VAR_LABELS.get(var, var)
    fig = plt.figure(figsize=(22, 9))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05],
                           hspace=0.22, wspace=0.12)
    for row, (s, h, rlabel, cmap) in enumerate([
            (s_mean, h_mean, "Ensemble mean", "RdBu_r"),
            (s_std, h_std, "Ensemble spread", "viridis")]):
        vext = np.nanmax(np.abs([np.nanmin([s, h]), np.nanmax([s, h])]))
        vmin, vmax = (-vext, vext) if row == 0 else (0, np.nanmax([s, h]))
        diff = s - h
        dext = max(float(np.nanmax(np.abs(diff))), 1e-12)
        panels = [(s, f"Icechunk ({n_s}m) — {rlabel}", cmap, vmin, vmax),
                  (h, f"Herbie ({n_h}m) — {rlabel}", cmap, vmin, vmax),
                  (diff, "Difference (Icechunk − Herbie)", "RdBu_r", -dext, dext)]
        last = None
        for col, (data, title, cm, vmn, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            im = _map(ax, data, lons, lats, title, cm, vmn, vmx)
            if col == 2:
                last = im
        cb = fig.colorbar(last, cax=fig.add_subplot(gs[row, 3]))
        cb.set_label(units, fontsize=9)
    lev = f" @ {level} hPa" if level is not None else ""
    fig.suptitle(f"Icechunk v4 vs Herbie — {label}{lev} — {era} "
                 f"{date[:4]}-{date[4:6]}-{date[6:]} 00Z  T+{step}h  (East Africa)",
                 fontsize=13, fontweight="bold", y=0.99)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{var}{level}" if level is not None else var
    out = out_dir / f"icechunk_v4_{era}_{tag}_{date}_T{step}h.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# -- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--era", required=True, choices=["0p4", "49r1", "50r1"])
    ap.add_argument("--run", default="00")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--var", default="t", help="STORE variable name (t, u, t2m, ...)")
    ap.add_argument("--levels", default="500,850",
                    help="comma-separated hPa; ignored for surface variables")
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--output-dir", default="gik_vs_herbie/icechunk_v4_eval")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    ds = open_store(a.store, a.era, a.run, a.sa_key)
    if a.var not in ds:
        print(f"{a.var!r} not in {a.era}/{a.run}z")
        return 2
    is_pl = "isobaricInhPa" in ds[a.var].dims
    levels = [int(x) for x in a.levels.split(",")] if is_pl else [None]
    out_dir = Path(a.output_dir)

    print("=" * 78)
    print(f"Icechunk v4 vs Herbie — {a.store}  {a.era}/{a.run}z")
    print(f"date={a.date} step=T+{a.step}h var={a.var} "
          f"{'levels=' + str(levels) if is_pl else '(surface)'}")
    print("=" * 78)

    results, failed = [], False
    for lev in levels:
        s_v, lats, lons, s_num = store_field(ds, a.var, a.date, a.step, lev)
        print(f"\n[{a.var}{'@' + str(lev) + 'hPa' if lev else ''} T+{a.step}h]")
        print(f"  store  : {s_v.shape[0]} members, grid {s_v.shape[1]}x{s_v.shape[2]}, "
              f"lat {lats[0]:.2f}..{lats[-1]:.2f}, lon {lons[0]:.2f}..{lons[-1]:.2f}")
        nan_members = [int(n) for n, f in zip(s_num, s_v) if np.all(np.isnan(f))]
        if nan_members:
            print(f"  store  : {len(nan_members)} all-NaN members {nan_members[:8]}"
                  f"{' ...' if len(nan_members) > 8 else ''}")

        rec = {"store": a.store, "era": a.era, "run": a.run, "date": a.date,
               "var": a.var, "level": lev, "step": a.step,
               "n_store_members": int(s_v.shape[0]),
               "store_all_nan_members": nan_members,
               "lat_range": [float(lats[0]), float(lats[-1])],
               "lon_range": [float(lons[0]), float(lons[-1])]}
        try:
            h_v, h_num = herbie_field(a.date, a.run, a.step, a.var, lev, lats, lons)
        except Exception as e:
            print(f"  Herbie FAILED: {e}")
            rec["herbie_error"] = str(e)
            results.append(rec)
            failed = True
            continue

        pm = per_member_stats(s_v, s_num, h_v, h_num)
        rec["n_herbie_members"] = int(h_v.shape[0])
        rec["per_member"] = pm
        ok = (pm["n_compared"] > 0
              and pm["max_rel_diff"] < PER_MEMBER_RTOL
              and pm["n_store_all_nan"] == 0)
        failed |= not ok
        print(f"  per-member: {pm['n_compared']} matched | min r={pm['min_corr']:.8f} "
              f"| max|diff|={pm['max_abs_diff']:.3g} "
              f"({pm['max_rel_diff']:.2e} of range)  -> {'PASS' if ok else 'FAIL'}")

        s_mean, s_std = np.nanmean(s_v, axis=0), np.nanstd(s_v, axis=0)
        h_mean, h_std = np.nanmean(h_v, axis=0), np.nanstd(h_v, axis=0)
        rec["ensemble_mean"] = field_stats(s_mean, h_mean)
        rec["ensemble_spread"] = field_stats(s_std, h_std)
        m = rec["ensemble_mean"]
        print(f"  ensemble  : {s_v.shape[0]}m vs {h_v.shape[0]}m | mean r={m['corr']:.8f} "
              f"RMSE={m['rmse']:.3g} max|diff|={m['max_abs_diff']:.3g}")

        if not a.no_plot:
            p = plot_compare(a.date, a.era, a.var, lev, a.step, lats, lons,
                             s_mean, s_std, h_mean, h_std,
                             s_v.shape[0], h_v.shape[0], out_dir)
            print(f"  plot -> {p}")
        results.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.var if is_pl else f"{a.var}_sfc"
    sf = out_dir / f"icechunk_v4_{a.era}_{tag}_{a.date}_T{a.step}h.json"
    # numpy scalars leak in from coordinate values; float() them at the boundary
    sf.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nstats -> {sf}")
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
