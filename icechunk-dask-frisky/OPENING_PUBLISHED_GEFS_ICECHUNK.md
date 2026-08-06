# Opening the published GEFS Icechunk store (source.coop)

The NOAA GEFS ensemble GIK store is published on **source.coop** as a virtual
Icechunk repo. Its metadata (repo pointer + snapshots + manifests + transactions)
lives on source.coop; the actual data chunks are **virtual references** into the
public `s3://noaa-gefs-pds/` GRIB archive on AWS. A reader therefore needs only:

- **anonymous** GET/LIST on source.coop (the store metadata), and
- **anonymous** GET on `noaa-gefs-pds` (the GRIB bytes the virtual chunks point at).

No credentials of any kind are required.

| | value |
|---|---|
| Endpoint | `https://data.source.coop` |
| Bucket | `e4drr-project` |
| Prefix | `forecasts/noaa_gefs_aws_s3_icechunk_vd` |
| Group | `0p25/00z`  ← **arrays are here, not at the root** |
| Virtual chunk source | `s3://noaa-gefs-pds/` (anonymous) |
| Chunk codec | `gribberish` (Zarr v3) |

## Dependencies

```
icechunk>=2.1  zarr>=3.2  xarray>=2025.1  gribberish>=1.4  s3fs  numpy
```

## Minimal open (xarray)

```python
import icechunk
import xarray as xr
import gribberish.zarr  # noqa: F401 -- registers the "gribberish" Zarr v3 codec

storage = icechunk.s3_storage(
    bucket="e4drr-project",
    prefix="forecasts/noaa_gefs_aws_s3_icechunk_vd",
    endpoint_url="https://data.source.coop",
    region="us-east-1",
    anonymous=True,        # public read of the store metadata
    from_env=False,        # ignore any AWS_* env vars
    force_path_style=True, # source.coop needs path-style addressing
)

# authorize anonymous byte-range reads of the virtual chunks on AWS
auth = icechunk.containers_credentials(
    {"s3://noaa-gefs-pds/": icechunk.s3_anonymous_credentials()})

# Disable eager manifest preload -- see "source.coop sporadic 500s" below.
cfg = icechunk.RepositoryConfig.default()
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))

repo = icechunk.Repository.open(
    storage, config=cfg, authorize_virtual_chunk_access=auth)

ds = xr.open_zarr(
    repo.readonly_session("main").store,
    group="0p25/00z",      # <-- the data lives under this group
    consolidated=False,
    zarr_format=3,
)
print(ds)   # xarray reports "Size: 697TB" -- the virtual (unrealized) size

# read one field -- resolves a virtual chunk from noaa-gefs-pds and decodes it
t2m = ds["t2m"].isel(time=-1, number=0, step=0).values   # (721, 1440), Kelvin
```

## Same thing with zarr directly (no xarray)

```python
import zarr
ro = repo.readonly_session("main").store
g = zarr.open_group(store=ro, path="0p25/00z", mode="r", zarr_format=3)
t2m = g["t2m"][-1, 0, 0]     # (time, number, step, lat, lon) -> (lat, lon)
```

## What you get

```
Dimensions: (time, number, step, latitude, longitude)
  time      2031  datetime64   2020-09-23 .. 2026-04-15   (00z runs)
  number    30    gep01..gep30
  step      81    0..240 h at 3 h
  latitude  721   90 .. -90
  longitude 1440  0 .. 359.75
  ~34 data vars: t2m, tp, u10, v10, cape, sp, gh, r2, ...

```

### Size: three numbers, don't conflate them

| measure | size | meaning |
|---|---|---|
| store objects on source.coop | 5.4 GB | what is actually hosted |
| **referenced GRIB** (packed) | ~92 TB | bytes pulled if you read the whole store once |
| dense float32 (`ds.nbytes`) | 697 TB | xarray's "Size: 697TB" -- **misleading** |

`ds.nbytes` assumes every `time x number x step x lat x lon` cell exists and is
stored uncompressed. Real GRIB2 is packed, and many cells have no message at all
(accumulated vars at `step=0`). Nothing is materialized on open -- you only fetch
the byte ranges you actually index.

## Gotchas

- **Arrays are under group `0p25/00z`.** Opening the root group returns an empty
  dataset (no data vars) — this is expected, not a broken store.
- **Disable eager manifest preload (source.coop sporadic 500s).** source.coop
  returns transient `HTTP 500` ("service error") on a small % of GETs under
  concurrency. This is source.coop, *not* AWS S3 (manifests live on source.coop;
  the GRIB byte-range reads to `noaa-gefs-pds` are never even reached at open
  time), and it is *not* a rate limit (128-way concurrent GET bursts succeed).
  But icechunk's default open **prefetches thousands of manifests in parallel**,
  so at a few-% error rate at least one *critical* fetch tends to draw a 500 and
  the whole open raises `IcechunkError: object store error service error`.
  Passing the `ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0)`
  above turns preload off: manifests then load lazily on demand (with retry).
  Open is slower (~60 s vs a few s) but robust and silent. Retrying the default
  open a few times also usually succeeds, since the failure is probabilistic.
- **`gribberish.zarr` must be imported** before reading any chunk, or the decode
  fails with an unknown-codec error. The import registers the Zarr v3 codec.
- **`force_path_style=True`** is required for source.coop; without it the client
  builds a virtual-host URL that source.coop does not serve.
- **`from_env=False`** — pass it so stray `AWS_*` env vars (e.g. a leftover
  `source .env` for publishing) don't turn the anonymous read into a signed one.
- **Accumulated vars have no f000 message.** `tp`/`tmax`/`tmin` etc. are all-NaN
  at `step=0` by construction; use `step>=1`. Instant vars (`t2m`, `u10`, ...)
  decode at `step=0`.
- **`time` may not be globally sorted** if the store was gap-filled out of order;
  call `ds.sortby("time")` if you need monotonic time.

## Smoke test

`smoke_test_published_gefs.py` runs exactly the anonymous path above and decodes
one `t2m` field, asserting a physically sane mean (200–330 K). Run it with **no**
AWS credentials in the environment:

```
uv run smoke_test_published_gefs.py
```

Expected tail:

```
== decoding t2m @ 2026-04-15, member gep01, f000 ==
  shape (721, 1440) decoded in ~1.4s
  finite fraction 1.000
  min/mean/max = 212.4 / 278.1 / 311 K
RESULT: PASS -- anonymous open + virtual decode works
```
