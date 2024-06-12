import datetime
import copy
import xarray as xr
import numpy as np
import pandas as pd
import fsspec
import kerchunk
from kerchunk.grib2 import scan_grib, grib_tree
import gcsfs
import datatree
from joblib import parallel_config

# This could be generalized to any gridded FMRC dataset but right now it works with NOAA's Grib2 files
import dynamic_zarr_store


# Pick two files to build a grib_tree with the correct dimensions
gfs_files = [
    "gs://global-forecast-system/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f000",
    "gs://global-forecast-system/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f001",
]

# This operation reads two of the large Grib2 files from GCS
# scan_grib extracts the zarr kerchunk metadata for each individual grib message
# grib_tree builds a zarr/xarray compatible hierarchical view of the dataset
gfs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
# it is slow even in parallel because it requires a huge amount of IO

# The grib_tree can be opened directly using either zarr or xarray datatree
# But this is too slow to build big aggregations
gfs_dt = datatree.open_datatree(
    fsspec.filesystem("reference", fo=gfs_grib_tree_store).get_mapper(""),
    engine="zarr",
    consolidated=False,
)

gfs_kind = dynamic_zarr_store.extract_datatree_chunk_index(
    gfs_dt, gfs_grib_tree_store, grib=True
)

# While the static zarr metadata associated with the dataset can be seperated - created once.
deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
dynamic_zarr_store.strip_datavar_chunks(deflated_gfs_grib_tree_store)


print("Original references: ", len(gfs_grib_tree_store["refs"]))
print("Stripped references: ", len(deflated_gfs_grib_tree_store["refs"]))

# We can pull this out into a dataframe, that starts to look a bit like what we got above extracted from the actual grib files
# But this method runs in under a second reading a file that is less than 100k
idxdf = dynamic_zarr_store.parse_grib_idx(
    fs=fsspec.filesystem("gcs"),
    basename="gs://global-forecast-system/gfs.20230901/00/atmos/gfs.t00z.pgrb2.0p25.f006",
)
idxdf

# Unfortunately, some accumulation variables have duplicate attributes making them
# indesinguishable from the IDX file
idxdf.loc[idxdf["attrs"].duplicated(keep=False), :]

# What we need is a mapping from our grib/zarr metadata to the attributes in the idx files
# They are unique for each time horizon e.g. you need to build a unique mapping for the 1 hour
# forecast, the 2 hour forecast... the 48 hour forecast.

# let's make one for the 6 hour horizon. This requires reading both the grib and the idx file,
# mapping the data for each grib message in order
# took 2 minutes for one

mapping = dynamic_zarr_store.build_idx_grib_mapping(
    fs=fsspec.filesystem("gcs"),
    basename="gs://global-forecast-system/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f006",
)
"""# Now if we parse the RunTime from the idx file name `gfs.20230901/00/`
# We can build a fully compatible k_index"""

mapped_index = dynamic_zarr_store.map_from_index(
    pd.Timestamp("2023-09-01T00"),
    mapping.loc[~mapping["attrs"].duplicated(keep="first"), :],
    idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :],
)


mapped_index_list = []

deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
for date in pd.date_range("2023-09-01", "2023-09-30"):
    for runtime in range(0, 24, 6):
        horizon = 6
        fname = f"gs://global-forecast-system/gfs.{date.strftime('%Y%m%d')}/{runtime:02}/atmos/gfs.t{runtime:02}z.pgrb2.0p25.f{horizon:03}"

        idxdf = dynamic_zarr_store.parse_grib_idx(
            fs=fsspec.filesystem("gcs"), basename=fname
        )

        mapped_index = dynamic_zarr_store.map_from_index(
            pd.Timestamp(date + datetime.timedelta(hours=runtime)),
            deduped_mapping,
            idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :],
        )
        mapped_index_list.append(mapped_index)

gfs_kind = pd.concat(mapped_index_list)

"""We just aggregated a 120 GFS grib files in 18 seconds!

Lets build it back into a data tree!

The reinflate_grib_store interface is pretty opaque but allows building any slice of an FMRC. A good area for future improvement, but for now, since we have just a single 6 hour horizon slice let's build that..."""
axes = [
    pd.Index(
        [
            pd.timedelta_range(
                start="0 hours", end="6 hours", freq="6h", closed="right", name="6 hour"
            ),
        ],
        name="step",
    ),
    pd.date_range(
        "2023-09-01T06:00", "2023-10T00:00", freq="360min", name="valid_time"
    ),
]


# It is fast to rebuild the datatree - but lets pull out two varables to look at...
gfs_store = dynamic_zarr_store.reinflate_grib_store(
    axes=axes,
    aggregation_type=dynamic_zarr_store.AggregationType.HORIZON,
    chunk_index=gfs_kind.loc[gfs_kind.varname.isin(["dswrf", "t2m"])],
    zarr_ref_store=deflated_gfs_grib_tree_store,
)
