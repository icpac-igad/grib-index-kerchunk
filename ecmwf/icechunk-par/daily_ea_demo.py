# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "dask[distributed]", "gribberish>=1.4",
# ]
# ///
"""Daily ops pattern on the unified ECMWF Icechunk store: for one forecast
init (latest by default, or a specific date), compute the next-7-days daily
rainfall exceedance probabilities over East Africa from all 51 members.

Everything after open_zarr is stock xarray/dask -- no GIK-specific code.
The declarative selection (one date, EA box, 8 daily-mark steps) prunes the
graph to ~408 chunks out of millions, and manifest splitting (1 date/shard)
means only that date's ~100 KB manifest shard is fetched regardless of how
large the archive has grown.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
  uv run daily_ea_demo.py                    # latest init in the 49r1 group
  uv run daily_ea_demo.py --date 20240506    # a specific init date
  uv run daily_ea_demo.py --date 20230601 --group 0p4/00z
"""
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

LAT_MAX, LAT_MIN = 25.0, -14.0   # East Africa / ICPAC box
LON_MIN, LON_MAX = 19.0, 55.0
THRESHOLDS_MM = (10, 25, 50)


def era_group(date: str | None) -> str:
    if date is None or date >= "20260513":
        return "50r1/00z" if date else "49r1/00z"
    return "0p4/00z" if date < "20240229" else "49r1/00z"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens")
    ap.add_argument("--date", default=None, help="YYYYMMDD init date; default latest")
    ap.add_argument("--group", default=None, help="era group; default from --date")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    group = args.group or era_group(args.date)

    t0 = time.time()
    bucket, _, prefix = args.store[5:].partition("/")
    storage = icechunk.gcs_storage(
        bucket=bucket, prefix=prefix.rstrip("/"),
        service_account_file=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    auth = icechunk.containers_credentials(
        {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    ds = xr.open_zarr(repo.readonly_session("main").store, group=group,
                      consolidated=False, zarr_format=3, chunks={})
    print(f"opened {group} ({ds.sizes['time']} dates, "
          f"{ds.nbytes/2**40:.0f} TB virtual) in {time.time()-t0:.1f}s")

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=3,
                           processes=True, memory_limit="1.2GB",
                           dashboard_address=None)
    client = Client(cluster)

    t0 = time.time()
    tp = ds.tp.sel(time=args.date) if args.date else ds.tp.isel(time=-1)
    tp = (tp.sel(latitude=slice(LAT_MAX, LAT_MIN),
                 longitude=slice(LON_MIN, LON_MAX))
            .sel(step=[24 * d for d in range(8)]))   # daily accumulation marks
    n_chunks = tp.sizes["number"] * tp.sizes["step"]
    print(f"selection {dict(tp.sizes)} -> {n_chunks} chunks "
          f"(tp array holds {ds.tp.size // (ds.sizes['latitude']*ds.sizes['longitude']):,})")

    daily = tp.diff("step") * 1000.0                 # 7 daily accumulations, mm
    # one shared graph: each chunk fetched/decoded once for all thresholds
    probs = xr.concat([(daily > mm).mean("number") for mm in THRESHOLDS_MM],
                      dim="threshold").assign_coords(threshold=list(THRESHOLDS_MM))
    probs = probs.compute()
    init = args.date or str(ds.time.values[-1])[:10]
    print(f"P(daily rain > {THRESHOLDS_MM} mm), init {init}, EA, "
          f"{tp.sizes['number']} members: {time.time()-t0:.0f}s")
    for d in range(daily.sizes["step"]):
        p10, p25, p50 = (probs.sel(threshold=mm).isel(step=d) for mm in THRESHOLDS_MM)
        print(f"  day {d+1}: EA area with P>10mm >=50%: {(p10 >= .5).mean().item()*100:5.1f}% | "
              f"P>25mm >=50%: {(p25 >= .5).mean().item()*100:4.1f}% | "
              f"max P>50mm: {p50.max().item()*100:3.0f}%")
    client.close(); cluster.close()


if __name__ == "__main__":
    main()
