#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "cartopy",
#     "xarray",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "gribberish",
#     "herbie-data",
#     "cfgrib",
#     "eccodes",
#     "netcdf4",
#     "scipy",
# ]
# ///
"""
Compare GIK vs Herbie for ECMWF PRESSURE-LEVEL variables (East Africa / ICPAC).

This is the pressure-level companion to ``compare_gik_herbie.py`` (which only
covered surface total precipitation). It exists to *validate the per-level-keys
fix* (grib-index-kerchunk commit 4ca1c21): the re-baked par files now key
isobaric variables as

    step_{NNN}/{var}/pl/{level_hPa}/{member}/0.0.0

so a consumer can pull the correct pressure level for every (var, step, member).
Before the fix all 13 levels collapsed onto one reference, so this comparison
would have shown gross level-mixing errors (e.g. u-wind at "700 hPa" actually
returning 250/1000 hPa depending on step).

What it does, for one date / forecast step / variable, across pressure levels:

  GIK side   : read per-member par files (local dir OR gs:// path), resolve each
               step/var/pl/level/member byte-range ref, stream that exact GRIB
               message from the ECMWF S3 archive (anon), decode with gribberish,
               subset to the ICPAC domain, stack members -> ensemble mean & std.
  Herbie side: Herbie(model=ifs, product=enfo).xarray(":{var}:{level}:pl:")
               -> 50 perturbed members, subset to ICPAC, ensemble mean & std.
  Compare    : Pearson r, RMSE, MAE, max|diff| for mean and spread; and a
               3-column (GIK | Herbie | Diff) x 2-row (mean | spread) map per level.

Usage:
    # GIK par files already downloaded locally (see _download_par.py):
    uv run compare_gik_herbie_pressure.py \
        --par-dir gik_vs_herbie/pl_eval/par_local \
        --date 20260301 --step 48 --var u --levels 300,500,700,850

    # Read par straight from GCS instead of a local dir:
    uv run compare_gik_herbie_pressure.py \
        --gcs-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/2026/03/20260301/00z \
        --date 20260301 --step 48 --var v --levels 500,850

    # GIK-only (no Herbie download) — just plot the par fields:
    uv run compare_gik_herbie_pressure.py --par-dir ... --date 20260301 --no-herbie
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from scipy import stats

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# ── Domain / grid (must match the GIK streaming pipeline) ─────────────
# Grid is era-dependent: 0p25 (49r1/50r1, 721x1440) vs 0p4 (0.4-beta, 451x900).
# Defaults below are 0p25; set_grid() reassigns them for --grid 0p4.
#
# Derived from grids.py -- the single source of truth for the ECMWF grid origin.
# This file used to spell the axes out by hand. They were correct here, but the
# same fact hand-written in twelve places is how build_ecmwf_icechunk.py came to
# assume a 0-start longitude and displace every store by 180 deg. Import, do not
# retype. See HANDOVER_LONGITUDE_FIX.md.
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import grids as _grids  # noqa: E402

GRIDS = {name: {"shape": _grids.field_shape(name),
                "lats": _grids.latitudes(name),
                "lons": _grids.longitudes(name)}
         for name in ("0p25", "0p4")}
ECMWF_GRID_SHAPE = GRIDS["0p25"]["shape"]
ECMWF_LATS = GRIDS["0p25"]["lats"]
ECMWF_LONS = GRIDS["0p25"]["lons"]
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

DEFAULT_LEVELS = [300, 500, 700, 850]

# Friendly names for plot titles
VAR_LABELS = {
    "u": "U-wind", "v": "V-wind", "gh": "Geopotential height",
    "t": "Temperature", "q": "Specific humidity", "r": "Relative humidity",
    "d": "Divergence", "w": "Vertical velocity",
}
VAR_UNITS = {
    "u": "m/s", "v": "m/s", "gh": "gpm", "t": "K",
    "q": "kg/kg", "r": "%", "d": "1/s", "w": "Pa/s",
}


def icpac_grid():
    lat_idx = np.where((ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX))[0]
    lon_idx = np.where((ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX))[0]
    lats = ECMWF_LATS[lat_idx[0]:lat_idx[-1] + 1]
    lons = ECMWF_LONS[lon_idx[0]:lon_idx[-1] + 1]
    return (lat_idx[0], lat_idx[-1] + 1), (lon_idx[0], lon_idx[-1] + 1), lats, lons


LAT_SLICE, LON_SLICE, ICPAC_LATS, ICPAC_LONS = icpac_grid()


def set_grid(name):
    """Reassign the module-level grid globals for the chosen era (0p25|0p4)."""
    global ECMWF_GRID_SHAPE, ECMWF_LATS, ECMWF_LONS
    global LAT_SLICE, LON_SLICE, ICPAC_LATS, ICPAC_LONS
    g = GRIDS[name]
    ECMWF_GRID_SHAPE, ECMWF_LATS, ECMWF_LONS = g["shape"], g["lats"], g["lons"]
    LAT_SLICE, LON_SLICE, ICPAC_LATS, ICPAC_LONS = icpac_grid()


# ── GIK: read par files -> ensemble field ────────────────────────────

def list_member_pars(par_dir=None, gcs_path=None):
    """Return [(member_key_token, par_path)], member_key_token is e.g. 'control','ens01'."""
    out = []
    if gcs_path:
        import gcsfs
        key = os.environ.get("GIK_GCS_KEY")
        fs = gcsfs.GCSFileSystem(token=key) if key else gcsfs.GCSFileSystem()
        clean = gcs_path[5:] if gcs_path.startswith("gs://") else gcs_path
        files = sorted(fs.glob(f"{clean}/*.parquet"))
        for f in files:
            tok = f.split("/")[-1].split("z-")[-1].replace(".parquet", "")
            out.append((tok.replace("_", ""), f"gs://{f}"))
    else:
        for f in sorted(Path(par_dir).glob("*.parquet")):
            tok = f.name.split("z-")[-1].replace(".parquet", "")
            out.append((tok.replace("_", ""), str(f)))
    return out


def load_zstore(par_path):
    if par_path.startswith("gs://"):
        import gcsfs
        key = os.environ.get("GIK_GCS_KEY")
        fs = gcsfs.GCSFileSystem(token=key) if key else gcsfs.GCSFileSystem()
        df = pd.read_parquet(par_path, filesystem=fs)
    else:
        df = pd.read_parquet(par_path)
    z = {}
    for _, r in df.iterrows():
        k, v = r["key"], r["value"]
        if isinstance(v, bytes):
            try:
                d = v.decode("utf-8")
                v = json.loads(d) if d[:1] in "[{" else d
            except Exception:
                pass
        elif isinstance(v, str) and v[:1] in "[{":
            try:
                v = json.loads(v)
            except Exception:
                pass
        z[k] = v
    return z


def fetch_field(fs_s3, ref):
    """ref = [s3_url, offset, length] -> 721x1440 float32 (ICPAC subset)."""
    import gribberish
    url, off, length = ref[0], ref[1], ref[2]
    if not url.endswith(".grib2"):
        url += ".grib2"
    with fs_s3.open(url, "rb") as f:
        f.seek(off)
        raw = f.read(length)
    arr = gribberish.parse_grib_array(raw, 0).reshape(ECMWF_GRID_SHAPE)
    return arr[LAT_SLICE[0]:LAT_SLICE[1], LON_SLICE[0]:LON_SLICE[1]].astype(np.float32)


def gik_ensemble(members, var, level, step):
    """Stream every member's (var, level, step) pl field -> (n_members, lat, lon)."""
    fs_s3 = fsspec.filesystem("s3", anon=True)
    fields, used = [], []
    nan = np.full((LAT_SLICE[1] - LAT_SLICE[0], LON_SLICE[1] - LON_SLICE[0]),
                  np.nan, dtype=np.float32)
    for tok, path in members:
        z = load_zstore(path)
        key = f"step_{step:03d}/{var}/pl/{level}/{tok}/0.0.0"
        ref = z.get(key)
        if not (isinstance(ref, list) and len(ref) >= 3):
            fields.append(nan.copy())
            continue
        try:
            fields.append(fetch_field(fs_s3, ref))
            used.append(tok)
        except Exception:
            fields.append(nan.copy())
    return np.stack(fields, axis=0), used


# ── Herbie: pressure-level ensemble ──────────────────────────────────

def herbie_ensemble(date_str, run, var, level, step):
    """Herbie enfo pl ensemble -> (mean, std) reindexed onto the GIK ICPAC grid."""
    from herbie import Herbie
    H = Herbie(f"{date_str} {run}:00", model="ifs", product="enfo", fxx=step)
    ds = H.xarray(f":{var}:{level}:pl:", verbose=False)
    if isinstance(ds, list):
        cand = [d for d in ds if "number" in d.dims]
        ds = cand[0] if cand else ds[0]
    # variable name in the returned dataset
    vn = var if var in ds.data_vars else list(ds.data_vars)[0]
    # longitude 0..360 -> -180..180
    lons = ds.longitude.values
    if lons.max() > 180:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
    da = ds[vn]
    if "number" not in da.dims:
        da = da.expand_dims("number")
    # Select the GIK ICPAC labels directly, with a tolerance far below half a
    # cell (0.125 deg on the coarsest grid). Do NOT slice-then-reindex: eccodes
    # reports the 0.4 deg axis edge as -14.000000000000057, so
    # slice(LAT_MAX, LAT_MIN) drops that row -- 97 of 98 -- and a bare
    # method="nearest" reindex then back-fills it from -13.6. That duplicated
    # edge row is what produced the 0.57 K "difference" this file once reported
    # for every 0p4 date; with the tolerance it is 9e-05, and per-member the
    # fields are bit-identical. 0p25 was never affected -- its edges land on
    # exactly representable floats. See gik_vs_herbie/icechunk_v4_eval/README.md.
    da = da.sel(latitude=ICPAC_LATS, longitude=ICPAC_LONS,
                method="nearest", tolerance=0.01)
    vals = da.values  # (number, lat, lon)
    if vals.shape[-2:] != (len(ICPAC_LATS), len(ICPAC_LONS)):
        raise RuntimeError(f"herbie grid {vals.shape[-2:]} != GIK "
                           f"{(len(ICPAC_LATS), len(ICPAC_LONS))}")
    mean = np.nanmean(vals, axis=0)
    std = np.nanstd(vals, axis=0)
    return mean, std, vals.shape[0]


# ── Stats + plotting ─────────────────────────────────────────────────

def field_stats(g, h):
    diff = g - h
    valid = ~(np.isnan(g) | np.isnan(h))
    gv, hv = g[valid], h[valid]
    if gv.size == 0:
        return {"error": "no overlapping valid pixels"}
    r = float(stats.pearsonr(gv, hv)[0]) if gv.std() > 0 and hv.std() > 0 else float("nan")
    return {
        "corr": r,
        "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
        "mae": float(np.nanmean(np.abs(diff))),
        "max_abs_diff": float(np.nanmax(np.abs(diff))),
        "gik_range": [float(np.nanmin(g)), float(np.nanmax(g))],
        "herbie_range": [float(np.nanmin(h)), float(np.nanmax(h))],
    }


def _map(ax, data, title, cmap, vmin, vmax, units):
    im = ax.pcolormesh(ICPAC_LONS, ICPAC_LATS, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="auto")
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def plot_level(date_str, var, level, step, gik_mean, gik_std, herb_mean, herb_std,
               n_gik, n_herb, out_dir):
    units = VAR_UNITS.get(var, "")
    label = VAR_LABELS.get(var, var)
    have_h = herb_mean is not None
    ncol = 3 if have_h else 1
    fig = plt.figure(figsize=(7 * ncol + 1, 9))
    gs = gridspec.GridSpec(2, ncol + 1,
                           width_ratios=[1] * ncol + [0.05], hspace=0.22, wspace=0.12)

    for row, (gik, herb, rlabel, cmap) in enumerate([
        (gik_mean, herb_mean, "Ensemble mean", "RdBu_r"),
        (gik_std, herb_std, "Ensemble spread", "viridis"),
    ]):
        if have_h:
            vext = np.nanmax(np.abs([np.nanmin([gik, herb]), np.nanmax([gik, herb])]))
            vmin, vmax = (-vext, vext) if row == 0 else (0, np.nanmax([gik, herb]))
            diff = gik - herb
            dext = max(np.nanmax(np.abs(diff)), 1e-9)
            panels = [
                (gik, f"GIK ({n_gik}m) — {rlabel}", cmap, vmin, vmax),
                (herb, f"Herbie ({n_herb}m) — {rlabel}", cmap, vmin, vmax),
                (diff, "Difference (GIK − Herbie)", "RdBu_r", -dext, dext),
            ]
        else:
            vext = np.nanmax(np.abs(gik))
            vmin, vmax = (-vext, vext) if row == 0 else (0, np.nanmax(gik))
            panels = [(gik, f"GIK ({n_gik}m) — {rlabel}", cmap, vmin, vmax)]

        last_im = None
        for col, (data, title, cm, vmn, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            im = _map(ax, data, title, cm, vmn, vmx, units)
            if col == ncol - 1:
                last_im = im
        cax = fig.add_subplot(gs[row, ncol])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label(units, fontsize=9)

    fig.suptitle(
        f"GIK vs Herbie — {label} @ {level} hPa — "
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} 00Z  T+{step}h  (East Africa)",
        fontsize=13, fontweight="bold", y=0.99)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"compare_pl_{var}{level}_{date_str}_T{step}h.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="GIK vs Herbie — pressure-level comparison (ICPAC)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--par-dir", help="local dir of per-member par files")
    src.add_argument("--gcs-path", help="gs://.../{date}/{run}z dir of par files")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--run", default="00")
    ap.add_argument("--step", type=int, default=48, help="forecast hour (default T+48h)")
    ap.add_argument("--var", default="u", help="pl variable short name (u,v,gh,t,q,r,...)")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)),
                    help="comma-separated hPa levels")
    ap.add_argument("--max-members", type=int, default=None)
    ap.add_argument("--grid", choices=["0p25", "0p4"], default="0p25",
                    help="source grid/era: 0p25 (49r1/50r1) or 0p4 (0.4-beta)")
    ap.add_argument("--no-herbie", action="store_true", help="GIK-only (skip Herbie)")
    ap.add_argument("--output-dir", default="gik_vs_herbie")
    args = ap.parse_args()

    set_grid(args.grid)
    levels = [int(x) for x in args.levels.split(",")]
    out_dir = Path(args.output_dir)
    members = list_member_pars(par_dir=args.par_dir, gcs_path=args.gcs_path)
    if args.max_members:
        members = members[:args.max_members]

    print("=" * 70)
    print("GIK vs Herbie — PRESSURE LEVELS")
    print("=" * 70)
    print(f"date={args.date} run={args.run}Z step=T+{args.step}h var={args.var} "
          f"levels={levels}")
    print(f"GIK members found: {len(members)}  source: "
          f"{args.gcs_path or args.par_dir}")
    print(f"Herbie: {'DISABLED' if args.no_herbie else 'enabled'}")
    print()

    results = []
    for lev in levels:
        print(f"[{args.var}@{lev}hPa T+{args.step}h]")
        gik, used = gik_ensemble(members, args.var, lev, args.step)
        gmean, gstd = np.nanmean(gik, axis=0), np.nanstd(gik, axis=0)
        print(f"  GIK: {len(used)}/{len(members)} members resolved, "
              f"mean range [{np.nanmin(gmean):.3g}, {np.nanmax(gmean):.3g}] {VAR_UNITS.get(args.var,'')}")

        hmean = hstd = None
        n_herb = 0
        rec = {"date": args.date, "var": args.var, "level": lev, "step": args.step,
               "n_gik_members": len(used)}
        if not args.no_herbie:
            try:
                hmean, hstd, n_herb = herbie_ensemble(args.date, args.run, args.var, lev, args.step)
                rec["n_herbie_members"] = n_herb
                rec["mean"] = field_stats(gmean, hmean)
                rec["spread"] = field_stats(gstd, hstd)
                m = rec["mean"]
                print(f"  Herbie: {n_herb} members | mean r={m.get('corr', float('nan')):.6f} "
                      f"RMSE={m.get('rmse', float('nan')):.3g} "
                      f"MaxDiff={m.get('max_abs_diff', float('nan')):.3g}")
            except Exception as e:
                print(f"  Herbie FAILED: {e}")
                rec["herbie_error"] = str(e)
        results.append(rec)
        plot_level(args.date, args.var, lev, args.step, gmean, gstd, hmean, hstd,
                   len(used), n_herb, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    sf = out_dir / f"pl_comparison_stats_{args.var}_{args.date}_T{args.step}h.json"
    sf.write_text(json.dumps(results, indent=2))
    print(f"\nstats -> {sf}")

    # summary table
    if not args.no_herbie:
        print("\n" + "=" * 70)
        print(f"{'level':>6}  {'GIK':>4} {'Herb':>4}  {'mean r':>10}  {'mean RMSE':>10}  {'max|diff|':>10}")
        print("-" * 60)
        for r in results:
            m = r.get("mean", {})
            print(f"{r['level']:>6}  {r['n_gik_members']:>4} {r.get('n_herbie_members','-'):>4}  "
                  f"{m.get('corr', float('nan')):>10.6f}  {m.get('rmse', float('nan')):>10.3g}  "
                  f"{m.get('max_abs_diff', float('nan')):>10.3g}")
        print("=" * 70)


if __name__ == "__main__":
    main()
