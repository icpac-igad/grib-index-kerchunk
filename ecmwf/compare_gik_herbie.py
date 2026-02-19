#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "cartopy",
#     "xarray",
#     "numpy",
#     "netcdf4",
#     "scipy",
# ]
# ///
"""
Compare GIK vs Herbie total-precipitation NetCDF outputs.

For each date, loads both the GIK and Herbie NC files, computes
numerical statistics (correlation, RMSE, max absolute diff), and
creates side-by-side map plots.

Usage:
    uv run compare_gik_herbie.py
    uv run compare_gik_herbie.py --dates 20240301,20240509
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from scipy import stats


# ── Configuration ─────────────────────────────────────────────────────
GIK_DIR = Path("random_run_test")
HERBIE_DIR = Path("herbie_tp_output")
OUTPUT_DIR = Path("gik_vs_herbie")

DEFAULT_DATES = ["20240301", "20240509", "20240708", "20240924", "20241122"]
STEP_INDEX = 4  # T+48h (middle of the 9-step window)


# ── Numerical comparison ─────────────────────────────────────────────

def compare_one_date(date_str: str, step_idx: int = STEP_INDEX) -> dict:
    """Load both NC files for a date and compute comparison stats."""

    gik_file = GIK_DIR / f"IFS_{date_str}_00Z_cgan.nc"
    herbie_file = HERBIE_DIR / f"IFS_{date_str}_00Z_herbie_tp.nc"

    if not gik_file.exists():
        return {"date": date_str, "error": f"GIK file missing: {gik_file}"}
    if not herbie_file.exists():
        return {"date": date_str, "error": f"Herbie file missing: {herbie_file}"}

    ds_g = xr.open_dataset(gik_file)
    ds_h = xr.open_dataset(herbie_file)

    result = {"date": date_str}

    for var_suffix, label in [
        ("tp_ensemble_mean", "mean"),
        ("tp_ensemble_standard_deviation", "spread"),
    ]:
        g = ds_g[var_suffix].isel(time=0, valid_time=step_idx).values
        h = ds_h[var_suffix].isel(time=0, valid_time=step_idx).values

        # Align grids — they should match but verify
        if g.shape != h.shape:
            result[f"{label}_error"] = f"Shape mismatch: GIK {g.shape} vs Herbie {h.shape}"
            continue

        diff = g - h
        valid = ~(np.isnan(g) | np.isnan(h))
        g_v, h_v = g[valid], h[valid]

        if len(g_v) == 0:
            result[f"{label}_error"] = "No valid overlapping pixels"
            continue

        # Pearson correlation
        if np.std(g_v) > 0 and np.std(h_v) > 0:
            r, p = stats.pearsonr(g_v, h_v)
        else:
            r, p = float('nan'), float('nan')

        rmse = np.sqrt(np.nanmean(diff ** 2))
        mae = np.nanmean(np.abs(diff))
        max_abs = np.nanmax(np.abs(diff))
        rel_diff_pct = (mae / max(np.nanmean(np.abs(g_v)), 1e-12)) * 100

        result[f"{label}_corr"] = float(r)
        result[f"{label}_corr_p"] = float(p)
        result[f"{label}_rmse"] = float(rmse)
        result[f"{label}_mae"] = float(mae)
        result[f"{label}_max_abs_diff"] = float(max_abs)
        result[f"{label}_rel_diff_pct"] = float(rel_diff_pct)
        result[f"{label}_gik_range"] = [float(np.nanmin(g)), float(np.nanmax(g))]
        result[f"{label}_herbie_range"] = [float(np.nanmin(h)), float(np.nanmax(h))]

    ds_g.close()
    ds_h.close()
    return result


# ── Plot comparison ──────────────────────────────────────────────────

def plot_comparison(date_str: str, step_idx: int = STEP_INDEX,
                    output_dir: Path = OUTPUT_DIR):
    """Create a 3-column (GIK | Herbie | Diff) × 2-row (mean | spread) plot."""

    gik_file = GIK_DIR / f"IFS_{date_str}_00Z_cgan.nc"
    herbie_file = HERBIE_DIR / f"IFS_{date_str}_00Z_herbie_tp.nc"

    ds_g = xr.open_dataset(gik_file)
    ds_h = xr.open_dataset(herbie_file)

    lons_g = ds_g.longitude.values
    lats_g = ds_g.latitude.values
    lons_h = ds_h.longitude.values
    lats_h = ds_h.latitude.values

    vt = str(ds_g.valid_time.values[step_idx])[:16]
    fh_list = ds_g.attrs.get("forecast_hours", "")
    try:
        fh = eval(fh_list)[step_idx]
        fh_label = f"T+{fh}h"
    except Exception:
        fh_label = f"step {step_idx}"

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05],
                           hspace=0.25, wspace=0.15)

    rows = [
        ("tp_ensemble_mean", "Ensemble Mean (mm)", 1000, "YlGnBu"),
        ("tp_ensemble_standard_deviation", "Ensemble Spread (mm)", 1000, "Oranges"),
    ]

    for row_idx, (var, title, scale, cmap) in enumerate(rows):
        g = ds_g[var].isel(time=0, valid_time=step_idx).values * scale
        h = ds_h[var].isel(time=0, valid_time=step_idx).values * scale
        diff = g - h

        vmax = max(np.nanmax(np.abs(g)), np.nanmax(np.abs(h)), 1e-6)
        diff_vmax = max(np.nanmax(np.abs(diff)), 1e-6)

        panels = [
            (g, lons_g, lats_g, f"GIK — {title}", cmap, 0, vmax),
            (h, lons_h, lats_h, f"Herbie — {title}", cmap, 0, vmax),
            (diff, lons_g, lats_g, f"Difference (GIK - Herbie)", "RdBu_r",
             -diff_vmax, diff_vmax),
        ]

        for col_idx, (data, lons, lats, panel_title, cm, vmin, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 projection=ccrs.PlateCarree())
            im = ax.pcolormesh(lons, lats, data, cmap=cm, vmin=vmin, vmax=vmx,
                               transform=ccrs.PlateCarree(), shading='auto')
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle='--')
            ax.add_feature(cfeature.LAKES, alpha=0.3)
            ax.set_extent([19, 55, -14, 25], crs=ccrs.PlateCarree())
            ax.set_title(panel_title, fontsize=10, fontweight='bold')

            # Colorbars in the 4th column
            if col_idx == 2:
                cbar_ax = fig.add_subplot(gs[row_idx, 3])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label("mm", fontsize=9)
                cbar.ax.tick_params(labelsize=8)

    n_members_g = ds_g.attrs.get("n_ensemble_members", "?")
    n_members_h = ds_h.attrs.get("n_ensemble_members", "?")

    fig.suptitle(
        f"GIK vs Herbie — {date_str[:4]}-{date_str[4:6]}-{date_str[6:]} 00Z "
        f"— {fh_label} (valid {vt})\n"
        f"GIK: {n_members_g} members | Herbie: {n_members_h} members",
        fontsize=13, fontweight='bold', y=0.98,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"compare_{date_str}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    ds_g.close()
    ds_h.close()
    print(f"Saved: {out_path}")
    return out_path


# ── Scatter plot across all dates ────────────────────────────────────

def plot_scatter_all(results: list[dict], output_dir: Path = OUTPUT_DIR):
    """Create scatter plots of GIK vs Herbie values across all dates."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label, title in [
        (axes[0], "mean", "Ensemble Mean"),
        (axes[1], "spread", "Ensemble Spread"),
    ]:
        all_g, all_h = [], []
        for r in results:
            if "error" in r or f"{label}_error" in r:
                continue
            date_str = r["date"]
            gik_file = GIK_DIR / f"IFS_{date_str}_00Z_cgan.nc"
            herbie_file = HERBIE_DIR / f"IFS_{date_str}_00Z_herbie_tp.nc"
            ds_g = xr.open_dataset(gik_file)
            ds_h = xr.open_dataset(herbie_file)
            var = f"tp_ensemble_{label}" if label == "mean" else "tp_ensemble_standard_deviation"
            g = ds_g[var].isel(time=0, valid_time=STEP_INDEX).values.flatten()
            h = ds_h[var].isel(time=0, valid_time=STEP_INDEX).values.flatten()
            valid = ~(np.isnan(g) | np.isnan(h))
            all_g.extend(g[valid])
            all_h.extend(h[valid])
            ds_g.close()
            ds_h.close()

        all_g = np.array(all_g)
        all_h = np.array(all_h)

        ax.scatter(all_h, all_g, s=1, alpha=0.3, c='steelblue')
        lim = max(np.nanmax(all_g), np.nanmax(all_h)) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='1:1 line')
        r_val, _ = stats.pearsonr(all_g, all_h)
        ax.set_xlabel("Herbie (m)", fontsize=10)
        ax.set_ylabel("GIK (m)", fontsize=10)
        ax.set_title(f"TP {title}\nr = {r_val:.6f}  |  N = {len(all_g):,}",
                      fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    plt.tight_layout()
    out_path = output_dir / "scatter_gik_vs_herbie.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare GIK vs Herbie TP NetCDF outputs")
    parser.add_argument("--dates", type=str,
                        default=",".join(DEFAULT_DATES),
                        help="Comma-separated dates (YYYYMMDD)")
    parser.add_argument("--step-index", type=int, default=STEP_INDEX,
                        help="Forecast step index to compare (default: 4 = T+48h)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GIK vs Herbie — Total Precipitation Comparison")
    print("=" * 70)
    print(f"Dates: {dates}")
    print(f"Step index: {args.step_index}")
    print(f"GIK source: {GIK_DIR}")
    print(f"Herbie source: {HERBIE_DIR}")
    print(f"Output: {out_dir}")
    print()

    # Numerical comparison
    results = []
    for d in dates:
        print(f"\n[{d}] Computing statistics...")
        r = compare_one_date(d, step_idx=args.step_index)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            for label in ["mean", "spread"]:
                if f"{label}_corr" in r:
                    print(f"  {label}: r={r[f'{label}_corr']:.8f}  "
                          f"RMSE={r[f'{label}_rmse']:.2e}  "
                          f"MAE={r[f'{label}_mae']:.2e}  "
                          f"MaxDiff={r[f'{label}_max_abs_diff']:.2e}  "
                          f"RelDiff={r[f'{label}_rel_diff_pct']:.4f}%")

    # Save stats as JSON
    stats_file = out_dir / "comparison_stats.json"
    with open(stats_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStats saved: {stats_file}")

    # Create per-date comparison plots
    print("\n--- Creating comparison maps ---")
    for d in dates:
        gik_file = GIK_DIR / f"IFS_{d}_00Z_cgan.nc"
        herbie_file = HERBIE_DIR / f"IFS_{d}_00Z_herbie_tp.nc"
        if gik_file.exists() and herbie_file.exists():
            plot_comparison(d, step_idx=args.step_index, output_dir=out_dir)

    # Scatter plot
    print("\n--- Creating scatter plot ---")
    plot_scatter_all(results, output_dir=out_dir)

    # Print summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Date':<12} {'Mean r':<14} {'Mean RMSE':<14} {'Spread r':<14} {'Spread RMSE':<14}")
    print("-" * 68)
    for r in results:
        if "error" in r:
            print(f"{r['date']:<12} {'ERROR':<14}")
            continue
        mc = r.get("mean_corr", float('nan'))
        mr = r.get("mean_rmse", float('nan'))
        sc = r.get("spread_corr", float('nan'))
        sr = r.get("spread_rmse", float('nan'))
        print(f"{r['date']:<12} {mc:<14.8f} {mr:<14.2e} {sc:<14.8f} {sr:<14.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
