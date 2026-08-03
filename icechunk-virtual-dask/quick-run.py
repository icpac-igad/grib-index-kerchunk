import time

import numpy as np
import matplotlib.pyplot as plt
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

print("icechunk", icechunk.__version__)
print("xarray  ", xr.__version__)


ENDPOINT = "https://data.source.coop"
BUCKET = "e4drr-project"
PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
CONTAINER = "s3://ecmwf-forecasts/"     # virtual chunks (public AWS, anon)
ERAS = ["0p4", "49r1", "50r1"]          # each has a 00z group holding the arrays

t0 = time.time()

storage = icechunk.s3_storage(
    bucket=BUCKET,
    prefix=PREFIX,
    endpoint_url=ENDPOINT,
    region="us-east-1",
    anonymous=True,         # public read of the store metadata
    from_env=False,         # ignore any AWS_* env vars
    force_path_style=True,  # source.coop needs path-style addressing
)

# authorize anonymous byte-range reads of the virtual chunks on AWS
auth = icechunk.containers_credentials(
    {CONTAINER: icechunk.s3_anonymous_credentials()})

# Disable eager manifest preload -- see "source.coop sporadic 500s" in Gotchas.
cfg = icechunk.RepositoryConfig.default()
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))

repo = icechunk.Repository.open(
    storage, config=cfg, authorize_virtual_chunk_access=auth)
sess = repo.readonly_session("main")

print(f"repo opened in {time.time() - t0:.1f}s")

def open_era(era):
    """Open one schema era. Arrays live under `{era}/00z`, never the root."""
    return xr.open_zarr(
        sess.store,
        group=f"{era}/00z",
        consolidated=False,
        zarr_format=3,
        decode_timedelta=True,   # `step` carries timedelta-like units -- say so
    )


eras = {era: open_era(era) for era in ERAS}

for era, d in eras.items():
    ny, nx = d.sizes["latitude"], d.sizes["longitude"]
    print(f"{era:5s} {len(d.data_vars):2d} vars  grid {ny}x{nx}  "
          f"time={d.sizes['time']:4d}  number={d.sizes['number']}  "
          f"step={d.sizes['step']}  levels={d.sizes.get('isobaricInhPa')}  "
          f"| {np.datetime_as_string(d.time.values.min(), unit='D')} .. "
          f"{np.datetime_as_string(d.time.values.max(), unit='D')}")


ds = eras["50r1"]
print(ds)


total = 0
for era, d in eras.items():
    total += d.nbytes
    print(f"{era:5s} dense float32 (ds.nbytes): {d.nbytes / 1e12:8.1f} TB  "
          f"({d.nbytes:,} bytes)")
print(f"{'all':5s} dense float32, materialized: {total / 1e15:8.2f} PB")
print()
print("store objects on source.coop : ~15 GB   (what is actually hosted)")
print("referenced GRIB (packed)     : ~620 TB  (97-100% of enfo published)")
print("enfo published, same dates   : ~627 TB")