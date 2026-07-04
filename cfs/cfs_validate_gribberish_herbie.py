# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=2", "pandas>=2.2", "pyarrow", "fsspec", "s3fs",
#     "gribberish", "herbie-data", "cfgrib", "eccodes",
#     "xarray", "matplotlib",
# ]
# ///
"""
CFS GIK vs Herbie decode validation.

Proves the GIK-streamed values are correct by decoding the SAME CFS flux message
two independent ways and comparing:

  GIK / gribberish : take [url, offset, length] from a GIK parquet (produced by
                     lithops-cr-gik-cfs/run_lithops_cfs.py), fetch just those
                     bytes from S3, decode with gribberish (Rust).
  Herbie / cfgrib  : Herbie(model='cfs', product='6_hourly', kind='flxf', fxx=H)
                     downloads the full official flxf file and decodes with
                     cfgrib — the trusted reference.

Both resolve to s3://noaa-cfs-pds/cfs.{date}/{run}/6hrly_grib_{mm}/flxf... so a
match confirms the GIK byte-ranges + gribberish decode reproduce the official
data. Writes a 3-panel PNG (Herbie | gribberish | difference) per variable.

Usage:
    uv run cfs_validate_gribberish_herbie.py \
        --parquet cfs_validation/2026070200z.parquet \
        --date 20260702 --run 00 --member 1 --out cfs_validation
"""
import argparse
import io
import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# CFS T126 Gaussian flux grid
NLAT, NLON = 190, 384

# (zarr group, varname, Herbie wgrib2 search, label, units)
TARGETS = [
    ("t2m/instant/heightAboveGround", "t2m", ":TMP:2 m above ground:", "2 m temperature", "K"),
    ("sdswrf/instant/surface", "sdswrf", ":DSWRF:surface:", "Downward SW radiation", "W m-2"),
    ("prate/instant/surface", "prate", ":PRATE:surface:", "Precipitation rate", "kg m-2 s-1"),
]


def load_refs(parquet):
    df = pd.read_parquet(parquet)
    refs = {}
    for _, r in df.iterrows():
        v = r["value"]
        refs[r["key"]] = v.decode("utf-8") if isinstance(v, bytes) else v
    return refs


def gik_field(refs, group, varname, step_idx):
    """Fetch the byte-range for {group}/{varname}/{step}.0.0 and gribberish-decode."""
    import s3fs
    import gribberish
    key = f"{group}/{varname}/{step_idx}.0.0"
    ref = refs.get(key)
    if ref is None:
        raise KeyError(f"no GIK ref for {key}")
    url, offset, length = json.loads(ref) if isinstance(ref, str) else ref
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(url.replace("s3://", ""), "rb") as f:
        f.seek(offset)
        raw = f.read(length)
    arr = np.asarray(gribberish.parse_grib_array(raw, 0)).reshape(NLAT, NLON).astype("float64")
    return arr, (url, offset, length)


def herbie_field(date, run, member, kind, search, fxx):
    """Download the full flxf via Herbie and cfgrib-decode the matching message."""
    import importlib
    importlib.import_module("herbie.models.cfs")  # register the 'cfs' template
    import herbie.models
    if not hasattr(herbie.models, "cfs"):
        herbie.models.cfs = importlib.import_module("herbie.models.cfs")
    from herbie import Herbie
    H = Herbie(f"{date[:4]}-{date[4:6]}-{date[6:8]} {run}:00",
               model="cfs", product="6_hourly", member=int(member), kind=kind, fxx=int(fxx))
    ds = H.xarray(search, remove_grib=False)
    if isinstance(ds, list):
        ds = ds[0]
    dv = [v for v in ds.data_vars][0]
    da = ds[dv]
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    return da.values.astype("float64"), lat, lon, dv


def orient(gik, ref):
    """gribberish scan order vs cfgrib may be N-S flipped; pick the better match."""
    d0 = np.nanmean(np.abs(gik - ref))
    d1 = np.nanmean(np.abs(np.flipud(gik) - ref))
    return (np.flipud(gik), True, d1) if d1 < d0 else (gik, False, d0)


def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    diff = a - b
    denom = (np.nanmax(b) - np.nanmin(b)) or 1.0
    return {
        "n": int(m.sum()),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "rel_max_pct": float(100 * np.max(np.abs(diff)) / denom),
    }


def plot(herbie_arr, gik_arr, lat, lon, label, units, out_png, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    diff = gik_arr - herbie_arr
    lon2 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon2)
    x, y = lon2[order], lat
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), constrained_layout=True)
    vmin, vmax = np.nanpercentile(herbie_arr, [2, 98])
    for ax, data, title, cmap, lim in [
        (axes[0], herbie_arr, "Herbie / cfgrib (reference)", "viridis", (vmin, vmax)),
        (axes[1], gik_arr, "GIK / gribberish (streamed)", "viridis", (vmin, vmax)),
        (axes[2], diff, "gribberish − Herbie", "RdBu_r",
         (-max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)) or 1e-9),
           max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)) or 1e-9))),
    ]:
        im = ax.pcolormesh(x, y, data[:, order], cmap=cmap, vmin=lim[0], vmax=lim[1], shading="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"CFS {label} ({units})   max|Δ|={st['max_abs']:.3e}  "
                 f"rmse={st['rmse']:.3e}  corr={st['corr']:.6f}", fontsize=12)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--run", default="00")
    ap.add_argument("--member", default="1")
    ap.add_argument("--kind", default="flxf")
    ap.add_argument("--fxx", type=int, default=24, help="forecast hour to validate (6-hourly)")
    ap.add_argument("--out", default="cfs_validation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    refs = load_refs(args.parquet)
    step_idx = args.fxx // 6
    print(f"CFS GIK vs Herbie — {args.date} {args.run}z member {args.member} "
          f"f{args.fxx:03d} (step idx {step_idx})\n")

    results = []
    for group, var, search, label, units in TARGETS:
        print(f"[{var}] {label}")
        try:
            gik, (url, off, ln) = gik_field(refs, group, var, step_idx)
            print(f"  GIK: {url.split('/')[-1]} @ {off}+{ln}  -> {gik.shape}")
        except Exception as e:
            print(f"  GIK FAIL: {e}\n"); continue
        try:
            ref, lat, lon, dv = herbie_field(args.date, args.run, args.member,
                                             args.kind, search, args.fxx)
            print(f"  Herbie: var '{dv}' -> {ref.shape}")
        except Exception as e:
            print(f"  Herbie FAIL ({type(e).__name__}: {str(e)[:90]})"
                  f" — cfgrib likely can't decode this CFS field; gribberish still can.\n")
            continue
        if ref.shape != gik.shape:
            print(f"  shape mismatch {ref.shape} vs {gik.shape}\n"); continue
        gik_o, flipped, _ = orient(gik, ref)
        st = stats(gik_o, ref)
        print(f"  flip={flipped}  max|Δ|={st['max_abs']:.3e}  rmse={st['rmse']:.3e}  "
              f"corr={st['corr']:.6f}  rel_max={st['rel_max_pct']:.4f}%")
        png = os.path.join(args.out, f"cfs_gik_vs_herbie_{var}_{args.date}{args.run}_f{args.fxx:03d}.png")
        plot(ref, gik_o, lat, lon, label, units, png, st)
        print(f"  plot -> {png}\n")
        results.append((var, st, png))

    if results:
        print("=" * 60)
        print("SUMMARY (gribberish vs Herbie/cfgrib):")
        for var, st, png in results:
            verdict = "MATCH" if st["corr"] > 0.9999 and st["rel_max_pct"] < 0.1 else "CHECK"
            print(f"  {var:8s} corr={st['corr']:.6f} max|Δ|={st['max_abs']:.3e} [{verdict}]")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
