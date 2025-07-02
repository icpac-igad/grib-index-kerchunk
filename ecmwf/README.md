## To run the step1 of ecmwf gik 

```
coiled notebook start --name dask-thresholds --vm-type n2-standard-2 --software itt-jupyter-env-v20250318 --workspace=geosfm
```

In notebook bash 
```
pip install "kerchunk<=0.2.7"
pip install "zarr<=3.0"
```
upload the coiled data service account, and the file

```ecmwf/test_run_ecmwf_step1_scangrib.py```

The output of the run is 

```
python test_run_ecmwf_step1_scangrib.py 
[2025-07-01 10:03:21] Starting ECMWF ensemble processing script
[2025-07-01 10:03:21] Processing 2 ECMWF files for date 20250701
[2025-07-01 10:03:21] Starting file processing
[2025-07-01 10:03:21] Processing file 1/2: 20250701000000-0h-enfo-ef.grib2
Completed scan_grib for s3://ecmwf-forecasts/20250701/00z/ifs/0p25/enfo/20250701000000-0h-enfo-ef.grib2, found 8007 messages
Completed index files and found 8007 entries in it
Found 2091 matching indices
[2025-07-01 10:10:11] File 1 completed, found 2091 groups (Elapsed: 409.71s)
[2025-07-01 10:10:11] Processing file 2/2: 20250701000000-3h-enfo-ef.grib2
Completed scan_grib for s3://ecmwf-forecasts/20250701/00z/ifs/0p25/enfo/20250701000000-3h-enfo-ef.grib2, found 8007 messages
Completed index files and found 8007 entries in it
Found 2091 matching indices
[2025-07-01 10:17:27] File 2 completed, found 2091 groups (Elapsed: 436.43s)
[2025-07-01 10:17:27] File processing completed. Total groups: 4182 (Elapsed: 846.14s)
[2025-07-01 10:17:27] Skipping standard grib_tree (credentials issue) - proceeding with fixed_ensemble_grib_tree
[2025-07-01 10:17:27] Created output directory: e_20250701_00
[2025-07-01 10:17:27] Starting fixed_ensemble_grib_tree processing
Found 41 unique paths from 4182 messages
  str/accum/surface: 102 groups, 51 ensemble members, 1 levels
  ro/accum/surface: 102 groups, 51 ensemble members, 1 levels
  u100/instant/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  q/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  tprate/instant/surface: 102 groups, 51 ensemble members, 1 levels
  u10/instant/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  tp/accum/surface: 102 groups, 51 ensemble members, 1 levels
  ttr/accum/nominalTop: 102 groups, 51 ensemble members, 1 levels
  fg10/max/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  lsm/instant/surface: 102 groups, 51 ensemble members, 1 levels
  v100/instant/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  asn/instant/surface: 102 groups, 51 ensemble members, 1 levels
  mx2t3/max/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  ewss/accum/surface: 102 groups, 51 ensemble members, 1 levels
  v10/instant/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  sp/instant/surface: 102 groups, 51 ensemble members, 1 levels
  mn2t3/min/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  msl/instant/meanSea: 102 groups, 51 ensemble members, 1 levels
  nsss/accum/surface: 102 groups, 51 ensemble members, 1 levels
  vo/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  skt/instant/surface: 102 groups, 51 ensemble members, 1 levels
  sithick/instant/surface: 102 groups, 51 ensemble members, 1 levels
  sve/instant/surface: 102 groups, 51 ensemble members, 1 levels
  gh/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  tcw/instant/entireAtmosphere: 102 groups, 51 ensemble members, 0 levels
  t2m/instant/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  strd/accum/surface: 102 groups, 51 ensemble members, 1 levels
  v/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  mucape/instant/mostUnstableParcel: 102 groups, 51 ensemble members, 1 levels
  u/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  svn/instant/surface: 102 groups, 51 ensemble members, 1 levels
  ssr/accum/surface: 102 groups, 51 ensemble members, 1 levels
  w/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  d/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  d2m/instant/heightAboveGround: 102 groups, 51 ensemble members, 1 levels
  tcwv/instant/entireAtmosphere: 102 groups, 51 ensemble members, 0 levels
  t/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
  ptype/instant/surface: 102 groups, 51 ensemble members, 1 levels
  zos/instant/surface: 102 groups, 51 ensemble members, 1 levels
  ssrd/accum/surface: 102 groups, 51 ensemble members, 1 levels
  r/instant/isobaricInhPa: 102 groups, 51 ensemble members, 1 levels
Processing str/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
/opt/conda/envs/itt/lib/python3.11/site-packages/kerchunk/combine.py:376: UserWarning: Concatenated coordinate 'time' contains less than expectednumber of values across the datasets: [1751328000]
  warnings.warn(
Processing ro/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing u100/instant/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing q/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing tprate/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing u10/instant/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing tp/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing ttr/accum/nominalTop with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'nominalTop']
Processing fg10/max/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing lsm/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing v100/instant/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing asn/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing mx2t3/max/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
/opt/conda/envs/itt/lib/python3.11/site-packages/kerchunk/combine.py:376: UserWarning: Concatenated coordinate 'step' contains less than expectednumber of values across the datasets: [3]
  warnings.warn(
Processing ewss/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing v10/instant/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing sp/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing mn2t3/min/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing msl/instant/meanSea with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'meanSea']
Processing nsss/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing vo/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing skt/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing sithick/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing sve/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing gh/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing tcw/instant/entireAtmosphere with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude']
Processing t2m/instant/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing strd/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing v/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing mucape/instant/mostUnstableParcel with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'mostUnstableParcel']
Processing u/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing svn/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing ssr/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing w/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing d/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing d2m/instant/heightAboveGround with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'heightAboveGround']
Processing tcwv/instant/entireAtmosphere with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude']
Processing t/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
Processing ptype/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing zos/instant/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing ssrd/accum/surface with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'surface']
Processing r/instant/isobaricInhPa with concat_dims=['time', 'step', 'number'], identical_dims=['longitude', 'latitude', 'isobaricInhPa']
[2025-07-01 10:19:30] Ensemble tree built with 9303 references (Elapsed: 123.17s)
[2025-07-01 10:19:30] Saving raw ensemble tree to JSON
[2025-07-01 10:19:30] Saved raw ensemble tree to e_20250701_00/ensemble_tree_raw.json (Elapsed: 0.03s)
[2025-07-01 10:19:30] Creating deflated store for parquet
[2025-07-01 10:19:30] Deflated store created (Elapsed: 0.02s)
[2025-07-01 10:19:30] Saving deflated store as parquet file
Parquet file saved to e_20250701_00/ecmwf_20250701_00z_ensemble.parquet
[2025-07-01 10:19:30] Parquet file saved: e_20250701_00/ecmwf_20250701_00z_ensemble.parquet (Elapsed: 0.02s)

Total refs in ensemble tree: 9303
Total refs in deflated store: 981

Sample keys from ensemble tree:
['.zgroup', 'str/.zgroup', 'str/.zattrs', 'ro/.zgroup', 'ro/.zattrs', 'u100/.zgroup', 'u100/.zattrs', 'q/.zgroup', 'q/.zattrs', 'tprate/.zgroup']
[2025-07-01 10:19:30] Opening with xarray datatree
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
/tmp/test_run_ecmwf_step1_scangrib.py:940: FutureWarning: In a future version of xarray decode_timedelta will default to False rather than None. To silence this warning, set decode_timedelta to True, False, or a 'CFTimedeltaCoder' instance.
  egfs_dt = xr.open_datatree(fsspec.filesystem(
[2025-07-01 10:19:32] DataTree opened successfully (Elapsed: 1.17s)
[2025-07-01 10:19:32] Saving DataTree structure analysis
DataTree structure saved to e_20250701_00/datatree_structure.json
[2025-07-01 10:19:32] DataTree structure saved (Elapsed: 0.00s)

DataTree keys: ['asn', 'd', 'd2m', 'ewss', 'fg10', 'gh', 'lsm', 'mn2t3', 'msl', 'mucape', 'mx2t3', 'nsss', 'ptype', 'q', 'r', 'ro', 'sithick', 'skt', 'sp', 'ssr', 'ssrd', 'str', 'strd', 'sve', 'svn', 't', 't2m', 'tcw', 'tcwv', 'tp', 'tprate', 'ttr', 'u', 'u10', 'u100', 'v', 'v10', 'v100', 'vo', 'w', 'zos']
[2025-07-01 10:19:32] Analyzing t2m variable

t2m dimensions: Frozen(ChainMap({}, {}))
[2025-07-01 10:19:32] Variable analysis completed (Elapsed: 0.00s)
[2025-07-01 10:19:32] Saving DataTree object as pickle
[2025-07-01 10:19:32] DataTree object saved to e_20250701_00/egfs_dt.pkl (Elapsed: 0.03s)
[2025-07-01 10:19:32] Creating processing summary
[2025-07-01 10:19:32] Processing summary saved (Elapsed: 0.00s)
[2025-07-01 10:19:32] === Processing complete === (Total time: 970.58s)
All results saved to: e_20250701_00/

Key files created:
  - e_20250701_00/ecmwf_20250701_00z_ensemble.parquet (main output for further processing)
  - e_20250701_00/ensemble_tree_raw.json (raw tree structure)
  - e_20250701_00/datatree_structure.json (DataTree structure)
  - e_20250701_00/egfs_dt.pkl (pickled DataTree object)
  - e_20250701_00/processing_summary.json (processing summary)

Total processing time: 970.58 seconds
```





