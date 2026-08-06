# Opening the published ECMWF IFS Icechunk store (source.coop)

The ECMWF IFS ensemble GIK store is published on **source.coop** as a virtual
Icechunk repo. Its metadata (repo pointer + snapshots + manifests + transactions)
lives on source.coop; the actual data chunks are **virtual references** into the
public `s3://ecmwf-forecasts/` GRIB archive on AWS. A reader needs only:

- **anonymous** GET/LIST on source.coop (the store metadata), and
- **anonymous** GET on `ecmwf-forecasts` (the GRIB bytes the virtual chunks point at).

No credentials of any kind are required.

| | value |
|---|---|
| Endpoint | `https://data.source.coop` |
| Bucket | `e4drr-project` |
| Prefix | `forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd` |
| Groups | `0p4/00z`, `49r1/00z`, `50r1/00z`  ← **one group per schema era** |
| Virtual chunk source | `s3://ecmwf-forecasts/` (anonymous) |
| Chunk codec | `gribberish` (Zarr v3) |

## The store is multi-era (three groups, not one)

Unlike GEFS (a single `0p25/00z` group), the ECMWF store keeps each schema era
in its own group because grid, level count, and variable set differ:

| Group | Grid | pl levels | Data vars | Window (00z) |
|---|---|---|---|---|
| `0p4/00z`  | 451 × 900  | 9  | 19 | 2023-01-18 .. 2024-02-28 |
| `49r1/00z` | 721 × 1440 | 13 | 59 | 2024-02-29 .. 2026-05-12 |
| `50r1/00z` | 721 × 1440 | 14 | 54 | 2026-05-12 .. present |

All three share `number=51` (control + 50 perturbed) and `step=85`.

Coverage is **complete**: for every era the store references 100% of the pl params
and levels ECMWF publishes in `enfo` (0p4 has only 8 pl params -- no `w` -- because
0p4-beta never published it), plus all sfc/sol params. Referenced bytes are
97-100% of published `enfo` bytes, date for date.

### Size: three numbers, don't conflate them

| measure | size | meaning |
|---|---|---|
| store objects on source.coop | 15 GB | what is actually hosted |
| **referenced GRIB** (packed) | **~620 TB** | bytes pulled if you read the whole store once |
| enfo published, same 1246 dates | ~627 TB | what ECMWF put on S3 |
| dense float32 (`ds.nbytes`) | 2.79 PB | 234 TB + 2.39 PB + 169 TB -- **misleading** |

`ds.nbytes` assumes every cell exists, uncompressed. Use the referenced number
for "how much data is this".

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
    prefix="forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd",
    endpoint_url="https://data.source.coop",
    region="us-east-1",
    anonymous=True,        # public read of the store metadata
    from_env=False,        # ignore any AWS_* env vars
    force_path_style=True, # source.coop needs path-style addressing
)

# authorize anonymous byte-range reads of the virtual chunks on AWS
auth = icechunk.containers_credentials(
    {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})

# Disable eager manifest preload -- see "source.coop sporadic 500s" below.
cfg = icechunk.RepositoryConfig.default()
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))

repo = icechunk.Repository.open(
    storage, config=cfg, authorize_virtual_chunk_access=auth)
sess = repo.readonly_session("main")

# pick the era you want -- e.g. the current 50r1 stream
ds = xr.open_zarr(sess.store, group="50r1/00z",
                  consolidated=False, zarr_format=3)
print(ds)   # xarray reports the virtual (unrealized) size, e.g. "Size: 168TB"

# read one field -- resolves a virtual chunk from ecmwf-forecasts and decodes it
t2m = ds["t2m"].isel(time=-1, number=0, step=0).values   # (721, 1440), Kelvin
```

## Gotchas

- **Arrays are under `{era}/00z`, never the root.** Opening the root group is
  empty; opening a bare `era` group is empty too -- you need `50r1/00z` etc.
- **Variable names differ by era.** 2 m temperature is `t2m` in some eras and the
  raw ECMWF short name `2t` in others; likewise `10u`/`u10`. Probe with
  `next(v for v in ("t2m","2t") if v in ds)` rather than hard-coding.
- **`gribberish.zarr` must be imported** before reading any chunk (registers the
  Zarr v3 codec) -- otherwise even opening a group fails with
  `UnknownCodecError: 'gribberish'` while reading array metadata.
- **Reading a 49r1 pressure-level variable needs >6 GB RAM.** Resolving *any* chunk
  makes icechunk load that array's **entire manifest**. Cost is ~200 bytes per
  chunk-ref, and refs = `dates x members x steps x levels`:

  | array | chunk-refs | peak RSS |
  |---|---|---|
  | `50r1/t2m` | 221 K | 0.25 GB |
  | `50r1/t` (14 levels, 51 dates) | 3.10 M | 0.81 GB |
  | `49r1/t2m` (794 dates, 2D) | 3.44 M | 0.80 GB |
  | `49r1/t` (794 dates x 13 levels) | **44.7 M** | **~9.1 GB -> OOM** |

  This is *not* caused by turning preload off (preload only controls eager prefetch
  at open, and actually lowers memory). It is purely array size -- `50r1` will hit
  the same wall once it accumulates ~800 dates. The fix is
  `icechunk.ManifestSplittingConfig` (split along `time`) **at write time**, so a
  read loads one shard instead of the whole array manifest.
- **`force_path_style=True`** is required for source.coop.
- **`from_env=False`** so stray `AWS_*` env vars don't turn the anonymous read
  into a signed one.
- **Disable eager manifest preload (source.coop sporadic 500s).** source.coop
  returns transient `HTTP 500` ("service error") on a small % of GETs under
  concurrency. This is source.coop, *not* AWS S3 (manifests live on source.coop;
  the GRIB byte-range reads to `ecmwf-forecasts` are never even reached at open
  time), and it is *not* a rate limit (128-way concurrent GET bursts succeed).
  icechunk's default open **prefetches thousands of manifests in parallel**, so
  at a few-% error rate at least one *critical* fetch tends to draw a 500 and the
  open raises `IcechunkError: object store error service error` -- with hundreds
  of "Failure pre-loading manifest ... service error" lines on stderr. Passing
  the `ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0)` above turns
  preload off: manifests load lazily on demand (with retry), the stderr noise
  disappears, and open is robust (slower, but reliable). The store itself is
  complete -- a full object-count check shows all 46,154 manifests present, so
  the errors are transport-side, not missing data.

## Smoke test

`smoke_test_published_ecmwf.py` runs the anonymous path above, opens **all three
eras**, and decodes a 2 m temperature field from each, asserting a physically
sane mean (200-330 K). Run it with **no** AWS credentials in the environment:

```
uv run smoke_test_published_ecmwf.py
```

Expected tail:

```
  [PASS] 0p4  t2m @ 2024-02-28 decodes -- shape (451, 900),  mean 276.5 K, finite 1.000
  [PASS] 49r1 t2m @ 2026-05-12 decodes -- shape (721, 1440), mean 279.7 K, finite 1.000
  [PASS] 50r1 t2m @ 2026-07-02 decodes -- shape (721, 1440), mean 281.5 K, finite 1.000
RESULT: PASS -- anonymous open + virtual decode works for all eras
```

See `OPENING_PUBLISHED_GEFS_ICECHUNK.md` for the GEFS counterpart (single group,
`noaa-gefs-pds` container).
