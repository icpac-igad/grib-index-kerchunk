#!/usr/bin/env python3
"""
Plot ECMWF cGAN ensemble forecast maps over the ICPAC region.

Reads the NetCDF output from stream_cgan_variables*.py and creates
a 4x3 panel of map plots showing ensemble mean and spread for key
meteorological variables.

Usage:
    python plot_cgan_maps.py
    python plot_cgan_maps.py --input cgan_output/IFS_20260206_00Z_cgan_simple.nc
    python plot_cgan_maps.py --input cgan_output/IFS_20260206_00Z_cgan_simple.nc --step-index 4
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


def plot_cgan_maps(input_file: str, step_index: int = 4, output_file: str = None):
    """Create 4x3 panel map plots from cGAN NetCDF output."""

    ds = xr.open_dataset(input_file)

    # Clamp step_index to valid range
    n_steps = ds.dims['valid_time']
    if step_index >= n_steps:
        step_index = n_steps - 1
    vt = str(ds.valid_time.values[step_index])[:16]

    # Default output path next to input
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.png'))

    # (variable, title, colormap, units, optional transform)
    plots = [
        ('tp_ensemble_mean',   'Total Precipitation (mean)',   'YlGnBu',  'mm',    lambda x: x * 1000),
        ('t2m_ensemble_mean',  '2m Temperature (mean)',        'RdYlBu_r','K',     None),
        ('sp_ensemble_mean',   'Surface Pressure (mean)',      'viridis',  'hPa',  lambda x: x / 100),
        ('tcc_ensemble_mean',  'Total Cloud Cover (mean)',     'Greys',    '',      None),
        ('u700_ensemble_mean', 'U-wind 700hPa (mean)',         'RdBu_r',  'm/s',   None),
        ('v700_ensemble_mean', 'V-wind 700hPa (mean)',         'RdBu_r',  'm/s',   None),
        ('tcwv_ensemble_mean', 'Total Column Water Vapour',    'Blues',    'kg/m2', None),
        ('ssr_ensemble_mean',  'Surface Solar Radiation',      'YlOrRd',  'MJ/m2', lambda x: x / 1e6),
        ('tp_ensemble_standard_deviation',  'Precipitation (ensemble spread)', 'Oranges', 'mm',  lambda x: x * 1000),
        ('t2m_ensemble_standard_deviation', 'Temperature (ensemble spread)',   'Reds',    'K',   None),
        ('u700_ensemble_standard_deviation','U-wind 700hPa (ensemble spread)', 'Purples', 'm/s', None),
        ('tcc_ensemble_standard_deviation', 'Cloud Cover (ensemble spread)',   'Oranges', '',    None),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(20, 22),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    lons = ds.longitude.values
    lats = ds.latitude.values

    for ax, (var, title, cmap, units, transform) in zip(axes.flat, plots):
        if var not in ds:
            ax.set_visible(False)
            continue

        data = ds[var].isel(time=0, valid_time=step_index).values
        if transform is not None:
            data = transform(data)

        # Symmetric colormap for wind mean fields
        if 'RdBu' in cmap and 'standard_deviation' not in var:
            vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
            im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=-vmax, vmax=vmax,
                               transform=ccrs.PlateCarree(), shading='auto')
        else:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                               transform=ccrs.PlateCarree(), shading='auto')

        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.add_feature(cfeature.LAKES, alpha=0.3)
        ax.set_extent([19, 55, -14, 25], crs=ccrs.PlateCarree())

        unit_str = f' ({units})' if units else ''
        ax.set_title(f'{title}{unit_str}', fontsize=11, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    n_members = ds.attrs.get('n_ensemble_members', '?')
    fig.suptitle(
        f'ECMWF IFS Ensemble Forecast — {str(ds.time.values[0])[:10]} '
        f'{str(ds.time.values[0])[11:16]}Z — Valid: {vt}\n'
        f'{n_members} ensemble members | ICPAC Region',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    ds.close()

    print(f"Saved: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ECMWF cGAN ensemble maps')
    parser.add_argument('--input', type=str, default='cgan_output/IFS_20260206_00Z_cgan_simple.nc',
                        help='Input NetCDF file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG file (default: same name as input with .png)')
    parser.add_argument('--step-index', type=int, default=4,
                        help='Forecast step index to plot (default: 4 = T+48h)')
    args = parser.parse_args()

    plot_cgan_maps(args.input, args.step_index, args.output)
