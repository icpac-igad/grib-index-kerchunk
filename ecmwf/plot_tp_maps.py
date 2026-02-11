#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "cartopy",
#     "xarray",
#     "numpy",
#     "netcdf4",
# ]
# ///
"""
Plot total precipitation ensemble maps from tp-only cGAN NetCDF files.

For each NC file, creates a 6x3 panel figure:
  - Top 3x3: ensemble mean at 9 forecast steps (T+36h to T+60h)
  - Bottom 3x3: ensemble spread at the same steps

Usage:
    uv run plot_tp_maps.py
    uv run plot_tp_maps.py --input-dir random_run_test --output-dir random_run_test_plots
    uv run plot_tp_maps.py --input-file random_run_test/IFS_20240301_00Z_cgan.nc
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np


def plot_tp_single(input_file: str, output_file: str = None):
    """Create 6x3 panel (mean + spread) precipitation map from one NC file."""

    ds = xr.open_dataset(input_file)
    n_steps = ds.sizes['valid_time']
    lons = ds.longitude.values
    lats = ds.latitude.values

    model_date = ds.attrs.get('model_date', str(ds.time.values[0])[:10])
    run_hour = ds.attrs.get('model_run', '00Z')
    n_members = ds.attrs.get('n_ensemble_members', '?')
    forecast_hours = ds.attrs.get('forecast_hours', '')

    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.png'))

    fig, axes = plt.subplots(
        2, n_steps, figsize=(3.2 * n_steps, 8),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    # Compute consistent colour limits across all steps
    mean_data = ds['tp_ensemble_mean'].isel(time=0).values * 1000  # → mm
    std_data = ds['tp_ensemble_standard_deviation'].isel(time=0).values * 1000

    mean_vmax = max(np.nanmax(mean_data), 1e-6)
    std_vmax = max(np.nanmax(std_data), 1e-6)

    for si in range(n_steps):
        vt = str(ds.valid_time.values[si])
        vt_short = vt[5:16].replace('T', ' ')  # "MM-DD HH:MM"

        # Try to recover forecast hour
        try:
            fh = eval(forecast_hours)[si] if forecast_hours else ''
            hour_label = f'T+{fh}h'
        except Exception:
            hour_label = f'step {si}'

        for row, (data_all, vmax, cmap, label) in enumerate([
            (mean_data, mean_vmax, 'YlGnBu', 'Mean'),
            (std_data, std_vmax, 'Oranges', 'Spread'),
        ]):
            ax = axes[row, si]
            data = data_all[si]

            im = ax.pcolormesh(
                lons, lats, data, cmap=cmap, vmin=0, vmax=vmax,
                transform=ccrs.PlateCarree(), shading='auto',
            )
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle='--')
            ax.add_feature(cfeature.LAKES, alpha=0.3)
            ax.set_extent([19, 55, -14, 25], crs=ccrs.PlateCarree())

            if row == 0:
                ax.set_title(f'{hour_label}\n{vt_short}', fontsize=8, fontweight='bold')
            if si == 0:
                ax.text(
                    -0.08, 0.5, f'TP {label} (mm)', transform=ax.transAxes,
                    fontsize=9, fontweight='bold', rotation=90,
                    va='center', ha='center',
                )

    # One colorbar per row, anchored to the right
    for row, (vmax, cmap, label) in enumerate([
        (mean_vmax, 'YlGnBu', 'Ensemble Mean (mm)'),
        (std_vmax, 'Oranges', 'Ensemble Spread (mm)'),
    ]):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[row, :].tolist(), shrink=0.8,
                            pad=0.02, aspect=20)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'Total Precipitation — {model_date} {run_hour} — '
        f'{n_members} members | ICPAC Region',
        fontsize=12, fontweight='bold', y=1.02,
    )

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    ds.close()

    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot TP ensemble maps from cGAN NetCDF files')
    parser.add_argument('--input-dir', type=str, default='random_run_test',
                        help='Directory containing tp-only NC files')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Plot a single NC file instead of a directory')
    parser.add_argument('--output-dir', type=str, default='random_run_test_plots',
                        help='Output directory for PNG files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        nc_files = [Path(args.input_file)]
    else:
        nc_files = sorted(Path(args.input_dir).glob('IFS_*.nc'))

    if not nc_files:
        print(f"No NC files found in {args.input_dir}")
        return

    print(f"Plotting {len(nc_files)} files → {output_dir}/")

    for nc in nc_files:
        out_png = output_dir / nc.with_suffix('.png').name
        try:
            plot_tp_single(str(nc), str(out_png))
        except Exception as e:
            print(f"ERROR plotting {nc.name}: {e}")

    print(f"\nDone. {len(list(output_dir.glob('*.png')))} plots in {output_dir}/")


if __name__ == '__main__':
    main()
