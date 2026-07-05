# GIK par → Icechunk virtual store (pilot, 0p4-beta era)

Converts the GIK ECMWF reference parquets from
[`E4DRR/gik-ecmwf-par-v2`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par-v2)
into [Icechunk](https://icechunk.io/en/latest/guides/virtual/) stores holding
**virtual chunk refs** into `s3://ecmwf-forecasts/` — no GRIB bytes are copied.
The store opens **natively** with `xr.open_zarr()`: the
[gribberish](https://github.com/mpiannucci/gribberish) Zarr v3 codec decodes
raw GRIB2 message bytes on read (registered automatically via the
`zarr.codecs:gribberish` entry point — readers just need `pip install gribberish`).

Context: mentoring input for the Code for Earth *GIK IceChain* team; follows
VirtualiZarr discussions
[#884](https://github.com/zarr-developers/VirtualiZarr/discussions/884)
(avoid the direct-write OOM: batch, commit per date) and
[#964](https://github.com/zarr-developers/VirtualiZarr/discussions/964)
(parquet is the wrong target for a living dataset; Icechunk gives
transactional append).

## What a GIK par actually contains (step 0 audit)

One par per member per forecast run, `(7224, 2)` DataFrame `[key, value]`:

| category | rows | example key |
|---|---|---|
| chunk ref `[url, offset, length]` | 7055 | `step_000/2t/sfc/control/0.0.0` |
| `.zarray` / `.zattrs` / `.zgroup` | 83 / 83 / 1 | `t2m/heightAboveGround/0.0/.zarray` |

7055 = 85 steps × (11 sfc vars + 8 pl vars × 9 levels). Two facts force the
converter to **rebuild the zarr model from the chunk keys** instead of reusing
the stored metadata:

1. The `.zarray` entries declare raw `<f4`, `compressor: null` — but the bytes
   at the refs are **GRIB2 messages**, so that metadata cannot decode anything.
2. The chunk keys (`step_SSS/{var}/{sfc|pl}/[{level}/]{member}/0.0.0`) live in
   a different tree than the metadata, so no kerchunk/VirtualiZarr parser can
   open the par as-is.

Two more gotchas found the hard way:

- par URLs **omit the `.grib2` suffix** the real S3 objects carry — append it.
- gribberish 1.4.0 bug: `GribberishCodec(var=None)` writes codec metadata its
  own `from_dict` cannot parse back — always pass `var=<array name>`
  (upstream fix candidate).
- anonymous S3 needs
  `icechunk.containers_credentials({prefix: icechunk.s3_anonymous_credentials()})`;
  container region must be exact (`eu-central-1` for `ecmwf-forecasts`).

## Store model

FMRC layout, one chunk per GRIB message, era 0p4-beta (grid 451×900):

```
surface var   (time, number, step, latitude, longitude)                 chunks (1,1,1,451,900)
pressure var  (time, number, step, isobaricInhPa, latitude, longitude)  chunks (1,1,1,1,451,900)
```

`time` = forecast init (append dimension, one commit per date), `number` =
0 control, 1–50 = ens_01…ens_50. Coordinates are materialized (not virtual).
Per the three-era plan: separate stores for 0p4 / 49r1 (13-level superset) /
50r1; this folder implements the 0p4 pilot.

## Scripts

| script | purpose |
|---|---|
| `audit_par.py` | step 0: dump a par's schema, key trees, ref categories |
| `pilot_0p4_to_icechunk.py` | single member → store, inline bit-exact proof |
| `build_0p4_ensemble_icechunk.py` | 51 members; creates store or **appends** a date along `time` |
| `test_icechunk_read.py` | test routine T1 structure / T2 decode / T3 bit-exact |

## Commands

Fetch pars for a date (51 files, ~10 MB):

```bash
D=20230627; mkdir -p pars/$D
curl -s "https://huggingface.co/api/datasets/E4DRR/gik-ecmwf-par-v2/tree/main/par/2023/06/$D/00z" |
  python3 -c "import json,sys;[print(x['path'].split('/')[-1]) for x in json.load(sys.stdin)]" |
  xargs -P8 -I{} curl -sL -o pars/$D/{} \
    "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par-v2/resolve/main/par/2023/06/$D/00z/{}"
```

Audit, build (create then append), and test:

```bash
uv run audit_par.py --par pars/20230627/2023062700z-control.parquet

uv run pilot_0p4_to_icechunk.py --par pars/20230627/2023062700z-control.parquet \
    --store stores/ecmwf-0p4-pilot

uv run build_0p4_ensemble_icechunk.py --pars-dir pars/20230627 --date 20230627 \
    --store stores/ecmwf-0p4-ens
uv run build_0p4_ensemble_icechunk.py --pars-dir pars/20230628 --date 20230628 \
    --store stores/ecmwf-0p4-ens   # appends at time index 1

# test routine (exit 0 = all pass); T3 checks random chunks bit-exact vs
# a direct S3 byte-range + gribberish decode of the same par refs
uv run test_icechunk_read.py --store stores/ecmwf-0p4-ens \
    --par pars/20230627/2023062700z-ens_07.parquet --member 7 --time-index 0
uv run test_icechunk_read.py --store stores/ecmwf-0p4-ens \
    --par pars/20230628/2023062800z-ens_33.parquet --member 33 --time-index 1
```

Read the result anywhere (only icechunk + xarray + gribberish needed):

```python
import icechunk, xarray as xr
storage = icechunk.local_filesystem_storage("stores/ecmwf-0p4-ens")
auth = icechunk.containers_credentials(
    {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
ds.t2m.sel(time="2023-06-28", number=7).isel(step=0).plot()
```

## Measured results (2026-07-05, icechunk 2.1.0 / zarr 3.2.1 / gribberish 1.4.0)

| run | refs | wall time | store size |
|---|---|---|---|
| pilot: 1 member, 1 date | 7,055 | 5 s (incl. read proof) | 232 KB |
| ensemble create: 20230627, 51 members | 359,805 | 10 s | — |
| ensemble append: 20230628 | 359,805 | 11 s | 19 MB total (2 dates) |

The 2-date ensemble store presents as an **11 GB-per-member / 1 TB virtual
dataset** while storing only 19 MB of manifests. Test routine: **all PASS** on
both time indices — structure, codec decode (~0.5 s/field cold), and 4/4
random chunks bit-exact against direct S3 decodes (surface + pressure vars,
both dates, members 7 and 33). Extrapolating ~10 s/date, the full 0p4 era
(401 dates, ~144 M refs) converts in ~1–2 h single-threaded.

## Next steps

- 49r1 and 50r1 era builders (13/14-level superset, 50r1 dual-stream control
  paths already absolute in the pars so no special handling expected).
- `last_updated_at` checksums on refs; cloud (GCS/source.coop) instead of
  local storage; manifest-split config before the 804-date 49r1 era.
- Upstream: report the `GribberishCodec(var=None)` round-trip bug.
