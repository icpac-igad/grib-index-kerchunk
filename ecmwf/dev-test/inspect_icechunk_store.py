#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "icechunk",
#     "matplotlib",
#     "cartopy",
#     "Pillow",
# ]
# ///
"""
Inspect the ECMWF EA tp Icechunk store.

- Reports dimensions, number of filled dates, commit history
- For each filled date: produces 7 daily forecast maps (ensemble mean TP)
  with East Africa country boundaries from ea_ghcf_simple.geojson
- Organises PNGs into date-wise folders and creates animated GIFs

Output structure:
    icechunk_inspect_plots/
    ├── 20240301_00z/
    │   ├── 20240301_00z_day1_000-024h.png
    │   ├── 20240301_00z_day2_024-048h.png
    │   ├── ...
    │   ├── 20240301_00z_day7_144-168h.png
    │   └── 20240301_00z_forecast.gif
    ├── 20240302_00z/
    │   └── ...

Usage:
    uv run inspect_icechunk_store.py [--max-dates 5]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_ACCOUNT_FILE = str(SCRIPT_DIR / "coiled-data.json")
GEOJSON_FILE = str(SCRIPT_DIR / "ea_ghcf_simple.geojson")
GCS_BUCKET = "gik-ecmwf-aws-tf"
GCS_PREFIX = "ecmwf-ea-ic-store"

# Lead times: 0-144h @3h + 150-168h @6h = 53 steps
LEAD_TIME_HOURS = list(range(0, 145, 3)) + [150, 156, 162, 168]

# Day boundaries in hours: day1=[0,24), day2=[24,48), ... day7=[144,168]
DAY_RANGES = [
    (1, 0, 24),
    (2, 24, 48),
    (3, 48, 72),
    (4, 72, 96),
    (5, 96, 120),
    (6, 120, 144),
    (7, 144, 168),
]


def load_geojson_geometries(geojson_path):
    """Load country polygons from GeoJSON for cartopy overlay."""
    from shapely.geometry import shape

    with open(geojson_path) as f:
        gj = json.load(f)

    geometries = []
    labels = []
    for feature in gj["features"]:
        geom = shape(feature["geometry"])
        geometries.append(geom)
        props = feature.get("properties", {})
        name = props.get("ADM0_NAME", props.get("name", props.get("GID_0", "")))
        labels.append(name)

    return geometries, labels


def open_store(args):
    """Open the Icechunk repository and return (repo, dataset)."""
    import icechunk
    import xarray as xr

    if args.local:
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )

    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default(),
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return repo, ds


def report_store_info(repo, ds):
    """Print store dimensions, coordinates, and commit history."""
    logger.info("=" * 60)
    logger.info("STORE INFO")
    logger.info("=" * 60)
    logger.info(f"\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    for dim in ["init_date", "member", "lead_time", "lat", "lon"]:
        if dim in ds.dims:
            vals = ds[dim].values
            logger.info(f"  {dim}: {ds.sizes[dim]}  [{vals[0]} .. {vals[-1]}]")

    # Commit history
    try:
        commits = list(repo.ancestry(branch="main"))
        logger.info(f"\nCommit history ({len(commits)} commits):")
        for c in commits[:15]:
            logger.info(f"  {c.message}")
        if len(commits) > 15:
            logger.info(f"  ... and {len(commits) - 15} more")
    except Exception:
        pass


def find_filled_dates(ds, sample_members=3):
    """Return list of date indices that have non-NaN data."""
    filled = []
    n_dates = ds.sizes["init_date"]
    logger.info(f"\nScanning {n_dates} dates for filled data...")

    for i in range(n_dates):
        try:
            sample = ds["tp"].isel(
                init_date=i, member=slice(0, sample_members),
                lead_time=1, lat=80, lon=70,
            ).values
            if np.any(~np.isnan(sample)):
                filled.append(i)
        except Exception:
            continue

    logger.info(f"  Filled dates: {len(filled)} / {n_dates}")
    return filled


def get_lead_time_indices_for_day(lead_times_hours, h_start, h_end):
    """Return indices into the lead_time array for hours in [h_start, h_end]."""
    indices = []
    for i, h in enumerate(lead_times_hours):
        if h_start <= h <= h_end:
            indices.append(i)
    return indices


def make_daily_pngs(ds, date_idx, out_dir, geometries, labels):
    """Create 7 daily forecast PNGs for one init date, plus an animated GIF."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from PIL import Image
    from shapely.geometry import MultiPolygon, Polygon

    init_date = ds["init_date"].values[date_idx]
    date_str = np.datetime_as_string(init_date, unit="D").replace("-", "")
    date_display = np.datetime_as_string(init_date, unit="D")
    folder_name = f"{date_str}_00z"
    date_dir = out_dir / folder_name
    date_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Processing date {date_idx}: {date_display}")

    # Load all members, all lead times for this date
    # Shape: (51, 53, 157, 145) — (member, lead_time, lat, lon)
    tp = ds["tp"].isel(init_date=date_idx).values

    lats = ds["lat"].values
    lons = ds["lon"].values
    # Convert timedelta64 lead_times to integer hours
    lt_raw = ds["lead_time"].values
    lead_times_hours = (lt_raw / np.timedelta64(1, "h")).astype(int).tolist()

    png_paths = []

    for day_num, h_start, h_end in DAY_RANGES:
        lt_indices = get_lead_time_indices_for_day(lead_times_hours, h_start, h_end)
        if not lt_indices:
            continue

        # Sum tp over lead times in this day range, per member: (51, 157, 145)
        tp_day = np.nansum(tp[:, lt_indices, :, :], axis=1)

        # Ensemble mean across members: (157, 145)
        ens_mean = np.nanmean(tp_day, axis=0)

        # Convert from metres to mm
        ens_mean_mm = ens_mean * 1000.0

        # Create figure
        fig, ax = plt.subplots(
            figsize=(8, 9),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

        # Plot data
        vmax = max(np.nanpercentile(ens_mean_mm, 98), 1.0)
        im = ax.pcolormesh(
            lons, lats, ens_mean_mm,
            cmap="Blues", vmin=0, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )

        # Overlay country boundaries from geojson
        for geom in geometries:
            polys = []
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)

            for poly in polys:
                exterior = np.array(poly.exterior.coords)
                mpl_poly = MplPolygon(
                    exterior, closed=True,
                    facecolor="none", edgecolor="black", linewidth=0.8,
                )
                ax.add_patch(mpl_poly)

        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="mm", pad=0.02)

        valid_date_start = np.datetime64(init_date) + np.timedelta64(h_start, "h")
        valid_date_end = np.datetime64(init_date) + np.timedelta64(h_end, "h")
        vd_start_str = np.datetime_as_string(valid_date_start, unit="D")
        vd_end_str = np.datetime_as_string(valid_date_end, unit="D")

        ax.set_title(
            f"ECMWF IFS Ensemble Mean TP (mm) — East Africa\n"
            f"Init: {date_display} 00Z | Day {day_num} ({h_start}-{h_end}h)\n"
            f"Valid: {vd_start_str} to {vd_end_str} | "
            f"Max: {np.nanmax(ens_mean_mm):.1f} mm",
            fontsize=11,
        )

        plt.tight_layout()
        png_name = f"{date_str}_00z_day{day_num}_{h_start:03d}-{h_end:03d}h.png"
        png_path = date_dir / png_name
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(png_path)
        logger.info(
            f"    Day {day_num} ({h_start}-{h_end}h): "
            f"mean={np.nanmean(ens_mean_mm):.2f} mm, "
            f"max={np.nanmax(ens_mean_mm):.1f} mm → {png_name}"
        )

    # Create animated GIF
    if png_paths:
        gif_path = date_dir / f"{date_str}_00z_forecast.gif"
        frames = [Image.open(p) for p in png_paths]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=800,  # ms per frame
            loop=0,
        )
        for f in frames:
            f.close()
        logger.info(f"    GIF: {gif_path}")

    return png_paths


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ECMWF EA tp Icechunk store and generate daily forecast maps",
    )
    parser.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    parser.add_argument("--gcs-prefix", type=str, default=GCS_PREFIX)
    parser.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    parser.add_argument("--local", type=str, default=None,
                        help="Local filesystem path (overrides GCS)")
    parser.add_argument("--max-dates", type=int, default=5,
                        help="Max filled dates to generate plots for (0=all)")
    parser.add_argument("--out-dir", type=str, default="icechunk_inspect_plots",
                        help="Output directory for plots")
    parser.add_argument("--geojson", type=str, default=GEOJSON_FILE,
                        help="Path to EA country boundary GeoJSON")
    args = parser.parse_args()

    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load geojson boundaries
    logger.info(f"Loading country boundaries from {args.geojson}")
    geometries, labels = load_geojson_geometries(args.geojson)
    logger.info(f"  Loaded {len(geometries)} country polygons: {', '.join(labels)}")

    # Open store
    logger.info("Opening Icechunk store...")
    try:
        repo, ds = open_store(args)
    except Exception as e:
        logger.error(f"Failed to open store: {e}")
        sys.exit(1)

    # Report basic info
    report_store_info(repo, ds)

    # Find dates with actual data
    filled = find_filled_dates(ds)
    if not filled:
        logger.warning("No filled dates found in the store.")
        return

    # Show which dates are filled
    for idx in filled:
        date_val = ds["init_date"].values[idx]
        logger.info(f"  [{idx}] {np.datetime_as_string(date_val, unit='D')}")

    # Generate daily forecast PNGs + GIFs
    dates_to_plot = filled
    if args.max_dates > 0:
        dates_to_plot = filled[:args.max_dates]

    logger.info(f"\nGenerating daily forecast maps for {len(dates_to_plot)} dates...")
    total_pngs = 0
    for date_idx in dates_to_plot:
        try:
            pngs = make_daily_pngs(ds, date_idx, out_dir, geometries, labels)
            total_pngs += len(pngs)
        except Exception as e:
            logger.error(f"  Date index {date_idx} failed: {e}")

    elapsed = time.time() - start
    logger.info(
        f"\nDone in {elapsed:.1f}s. "
        f"Generated {total_pngs} PNGs + {len(dates_to_plot)} GIFs in: {out_dir}/"
    )


if __name__ == "__main__":
    main()
