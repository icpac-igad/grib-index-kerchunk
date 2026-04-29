#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "icechunk",
#     "netcdf4",
#     "matplotlib",
#     "cartopy",
#     "shapely",
# ]
# ///
"""
Plot 7-day total precipitation exceedance probability from source.coop Icechunk store.

Reads the ECMWF IFS ensemble tp from source.coop, computes 7-day accumulated
precipitation, and overlays empirical exceedance probabilities against CMORPH
return period thresholds. Uses ea_ghcf_simple.geojson for country boundaries.

Usage:
    uv run plot_source_coop_tp_probability.py
    uv run plot_source_coop_tp_probability.py --date 20260315
"""

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import icechunk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from shapely.geometry import shape

# ─── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
RETURN_PERIODS_PATH = "/scratch/notebook/cmorph_ea_return_periods.nc"
GEOJSON_PATH = SCRIPT_DIR.parent / "gefs" / "ea_ghcf_simple.geojson"

# ─── source.coop Icechunk store ─────────────────────────────────────────────

S3_BUCKET = "us-west-2.opendata.source.coop"
S3_PREFIX = "e4drr-project/forecasts/ecmwf_ea_tp_icechunk"
S3_REGION = "us-west-2"

# East Africa bounding box
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Lead times for 7-day accumulation (0h to 168h at 3h intervals, then 6h)
LEAD_HOURS_7DAY = list(range(0, 145, 3)) + [150, 156, 162, 168]


def open_icechunk_store():
    """Open the source.coop Icechunk store (anonymous read)."""
    storage = icechunk.s3_storage(
        bucket=S3_BUCKET,
        prefix=S3_PREFIX,
        region=S3_REGION,
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def load_return_period_thresholds(duration="24hr"):
    """Load CMORPH return period thresholds for given duration."""
    ds_rp = xr.open_dataset(RETURN_PERIODS_PATH)
    rp = ds_rp["return_period_precip"].sel(duration=duration)
    return rp, ds_rp.coords["return_period"].values


def load_geojson_geometries():
    """Load East Africa country boundaries from GeoJSON."""
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)
    geometries = []
    labels = []
    for feature in gj["features"]:
        geom = shape(feature["geometry"])
        geometries.append(geom)
        labels.append(feature["properties"].get("GID_0", ""))
    return geometries, labels


def compute_daily_tp(ds, init_date_str):
    """Compute daily accumulated tp for all 51 members, 7 days.

    ECMWF tp is cumulative from forecast init. Daily accumulation for day N
    is tp(end_of_day_N) - tp(end_of_day_N-1). For day 1: tp(24h) - tp(0h).

    Returns: list of 7 xr.DataArrays, each (member, lat, lon) in mm.
    """
    import pandas as pd

    init_date = pd.Timestamp(init_date_str)
    tp = ds["tp"].sel(init_date=init_date)
    tp_loaded = tp.load()  # (member, lead_time, lat, lon)

    lead_hours = tp_loaded.lead_time.values / np.timedelta64(1, "h")

    # Day boundaries in hours
    day_bounds = [0, 24, 48, 72, 96, 120, 144, 168]

    daily_tp = []
    for d in range(7):
        h_start = day_bounds[d]
        h_end = day_bounds[d + 1]

        # Find nearest lead time to start and end of this day
        idx_start = np.argmin(np.abs(lead_hours - h_start))
        idx_end = np.argmin(np.abs(lead_hours - h_end))

        tp_end = tp_loaded.isel(lead_time=idx_end)
        tp_start = tp_loaded.isel(lead_time=idx_start)

        day_accum = (tp_end - tp_start).clip(min=0) * 1000.0  # m → mm
        daily_tp.append(day_accum)

    return daily_tp  # list of 7 x (member, lat, lon)


def regrid_thresholds(rp_thresholds, rp_year, ecmwf_lats, ecmwf_lons):
    """Regrid a single return period threshold to the ECMWF grid."""
    cmorph_lats = rp_thresholds.lat.values
    cmorph_lons = rp_thresholds.lon.values
    lat_idx = np.array([np.argmin(np.abs(cmorph_lats - lat)) for lat in ecmwf_lats])
    lon_idx = np.array([np.argmin(np.abs(cmorph_lons - lon)) for lon in ecmwf_lons])
    thresh = rp_thresholds.sel(return_period=rp_year).values
    return thresh[np.ix_(lat_idx, lon_idx)]


def compute_exceedance_probability(tp_daily_mm, thresh_ecmwf):
    """Compute fraction of ensemble members exceeding threshold.

    Args:
        tp_daily_mm: (member, lat, lon) in mm — one day's accumulation
        thresh_ecmwf: (lat, lon) — regridded threshold in mm

    Returns:
        np.ndarray (lat, lon) — exceedance probability [0, 1]
    """
    tp_vals = tp_daily_mm.values
    n_members = tp_vals.shape[0]
    exceed_count = np.sum(tp_vals > thresh_ecmwf[np.newaxis, :, :], axis=0)
    return exceed_count / n_members


def plot_daily_results(daily_tp, daily_probs, init_date_str, rp_year,
                       geometries, ecmwf_lats, ecmwf_lons):
    """Create a 7x3 panel figure: ensemble mean, ensemble max, exceedance probability."""
    import pandas as pd

    init_date = pd.Timestamp(init_date_str)

    fig, axes = plt.subplots(
        7, 3, figsize=(20, 36),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(
        f"ECMWF IFS Ensemble — Daily Precipitation Diagnostics\n"
        f"Init: {init_date_str} 00Z | 51 members | {rp_year}-yr 24h CMORPH threshold | "
        f"Source: source.coop Icechunk",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Shared colorbar settings for probability
    prob_cmap = mcolors.ListedColormap([
        "#ffffff", "#c6efce", "#a8d08d", "#ffd966",
        "#f4b183", "#ff6f61", "#c00000",
    ])
    prob_bounds = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    prob_norm = mcolors.BoundaryNorm(prob_bounds, prob_cmap.N)

    # Compute shared vmax for mean and max TP
    all_means = [d.mean(dim="member").values for d in daily_tp]
    all_maxes = [d.max(dim="member").values for d in daily_tp]
    mean_vmax = max(float(np.nanpercentile(np.concatenate([m.ravel() for m in all_means]), 98)), 1)
    max_vmax = max(float(np.nanpercentile(np.concatenate([m.ravel() for m in all_maxes]), 98)), 1)

    day_bounds = [0, 24, 48, 72, 96, 120, 144, 168]

    # Nairobi marker
    nrb_lat, nrb_lon = -1.3, 36.8

    for day in range(7):
        valid_start = init_date + pd.Timedelta(hours=day_bounds[day])
        valid_end = init_date + pd.Timedelta(hours=day_bounds[day + 1])
        day_label = (f"Day {day+1}: {valid_start.strftime('%b %d')}–"
                     f"{valid_end.strftime('%b %d')} ({day_bounds[day]}–{day_bounds[day+1]}h)")

        # ── Col 1: Ensemble mean TP ──
        ax = axes[day, 0]
        ens_mean = daily_tp[day].mean(dim="member")
        im = ax.pcolormesh(
            ecmwf_lons, ecmwf_lats, ens_mean.values,
            cmap="YlGnBu", vmin=0, vmax=mean_vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(nrb_lon, nrb_lat, "r*", markersize=10, transform=ccrs.PlateCarree())
        ax.set_title(f"{day_label}\nEns. Mean (mm) | max={float(ens_mean.max()):.1f}", fontsize=9)
        plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, label="mm", shrink=0.8)
        _add_boundaries(ax, geometries)

        # ── Col 2: Ensemble MAX TP (worst-case member at each pixel) ──
        ax = axes[day, 1]
        ens_max = daily_tp[day].max(dim="member")
        im = ax.pcolormesh(
            ecmwf_lons, ecmwf_lats, ens_max.values,
            cmap="YlOrRd", vmin=0, vmax=max_vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(nrb_lon, nrb_lat, "k*", markersize=10, transform=ccrs.PlateCarree())
        ax.set_title(f"{day_label}\nEns. MAX (mm) | max={float(ens_max.max()):.1f}", fontsize=9)
        plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, label="mm", shrink=0.8)
        _add_boundaries(ax, geometries)

        # ── Col 3: Exceedance probability ──
        ax = axes[day, 2]
        prob = daily_probs[day]
        im = ax.pcolormesh(
            ecmwf_lons, ecmwf_lats, prob,
            cmap=prob_cmap, norm=prob_norm,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(nrb_lon, nrb_lat, "r*", markersize=10, transform=ccrs.PlateCarree())
        pmax = float(prob.max())
        pct = float((prob > 0).mean()) * 100
        ax.set_title(
            f"{day_label}\nP(TP > {rp_year}-yr) | max={pmax:.2f}, area>0: {pct:.1f}%",
            fontsize=9,
        )
        plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                      label="Probability", shrink=0.8, ticks=prob_bounds)
        _add_boundaries(ax, geometries)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = SCRIPT_DIR / f"ecmwf_tp_daily_probability_{init_date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def _add_boundaries(ax, geometries):
    """Add country boundaries and map features to an axis."""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")

    for geom in geometries:
        ax.add_geometries(
            [geom], ccrs.PlateCarree(),
            facecolor="none", edgecolor="black", linewidth=0.8,
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def main():
    parser = argparse.ArgumentParser(
        description="Plot daily TP exceedance probability from source.coop Icechunk"
    )
    parser.add_argument(
        "--date", type=str, default="20260304",
        help="Init date YYYYMMDD (default: 20260304)"
    )
    parser.add_argument(
        "--rp", type=int, default=5,
        help="Return period in years for exceedance (default: 5)"
    )
    args = parser.parse_args()

    print(f"Opening source.coop Icechunk store (anonymous)...")
    ds = open_icechunk_store()
    print(f"  Dataset: {dict(ds.sizes)}")
    print(f"  Init dates: {ds.init_date.values[0]} → {ds.init_date.values[-1]}")

    print(f"\nLoading CMORPH 24h return period thresholds...")
    rp_thresholds, rp_values = load_return_period_thresholds(duration="24hr")
    print(f"  Return periods: {rp_values} years")
    print(f"  Using {args.rp}-yr threshold for exceedance")

    print(f"\nLoading East Africa boundaries...")
    geometries, labels = load_geojson_geometries()
    print(f"  Countries: {len(geometries)} ({', '.join(labels)})")

    print(f"\nComputing daily accumulated TP for {args.date}...")
    daily_tp = compute_daily_tp(ds, args.date)
    for d in range(7):
        ens_mean = daily_tp[d].mean(dim="member")
        print(f"  Day {d+1}: max={float(ens_mean.max()):.1f} mm")

    print(f"\nRegridding {args.rp}-yr threshold and computing exceedance...")
    ecmwf_lats = daily_tp[0].lat.values
    ecmwf_lons = daily_tp[0].lon.values
    thresh_ecmwf = regrid_thresholds(rp_thresholds, args.rp, ecmwf_lats, ecmwf_lons)

    daily_probs = []
    for d in range(7):
        prob = compute_exceedance_probability(daily_tp[d], thresh_ecmwf)
        pmax = float(prob.max())
        pct = float((prob > 0).mean()) * 100
        print(f"  Day {d+1}: max prob={pmax:.2f}, area>0: {pct:.1f}%")
        daily_probs.append(prob)

    print(f"\nPlotting...")
    plot_daily_results(
        daily_tp, daily_probs, args.date, args.rp,
        geometries, ecmwf_lats, ecmwf_lons,
    )
    print("Done.")


if __name__ == "__main__":
    main()
