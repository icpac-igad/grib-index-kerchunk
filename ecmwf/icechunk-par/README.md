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
| `build_0p4_ensemble_icechunk.py` | 51 members; creates store or **appends** a date along `time`; `--store` accepts a local path or `s3://bucket/prefix` |
| `test_icechunk_read.py` | test routine T1 structure / T2 decode / T3 bit-exact (local or S3 store; anonymous read when no AWS creds in env) |
| `test_dask_read.py` | dask-cluster read test: multi-process workers, ensemble mean/std, per-worker peak RSS |
| `materialize_ea_from_icechunk.py` | dask-cluster **realize** test: virtual store → East Africa subset → plain zarr on another source.coop path |
| `publish_to_source_coop.py` | resumable file-sync of a local store to source.coop (survives 1-hour STS expiry) |

See `CONVERSION_ESTIMATE.md` for the full four-era time/size estimate (~5 h
build for all 1,256 dates, ~13 GB published) and how the credential-free
build + resumable publish keep the 1-hour STS window a non-issue. The build now
uses **manifest splitting** along `time` (mandatory: without it store size is
O(n²) in dates).

## One store, era/run groups (current direction)

`build_ecmwf_icechunk.py` supersedes the per-era script: a **single Icechunk
repo** holding zarr groups `{era}/{run}z` — `0p4/00z`, `49r1/00z` (13-level
superset), `50r1/00z`, with room for `06z/12z/18z` groups (each run keeps its
own time and step axes, since 06z/18z forecasts are shorter). Verified: two
different-grid eras coexist in one store, each opens with
`xr.open_zarr(store, group="49r1/00z")`, the whole archive opens with
`xr.open_datatree(...)`, and reads decode bit-exact. 49r1+ pars carry two soil
fields (`sot`, `vsw`, levtype `sol`, no level segment) — treated as surface
vars; the builder now refuses to drop unknown levtypes silently.

It also writes **natively to GCS** (`--store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens`,
service-account json via `GOOGLE_APPLICATION_CREDENTIALS` or `--sa-key`):
Icechunk commits work directly against GCS — no local staging, no disk ceiling,
no 1-hour STS. Measured per date on GCS: 0p4 ~26 s, 49r1-13L ~48 s (vs 13/24 s
local). source.coop then gets a plain GCS→S3 object copy of the finished store
(same mirror pattern as the pars).

Single-store trade-offs (evaluated): one URL + one publish pipeline + datatree
view, at the cost of a single commit stream — era backfills serialize (or need
commit-retry/rebase if run concurrently), and repo history/GC spans all eras.
For a sequential backfill and 4 appends/day in operations, the serialization
cost is irrelevant.

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

## Dask-cluster evaluation (the #884 OOM question)

The historical CMORPH OOM (VirtualiZarr
[#884](https://github.com/zarr-developers/VirtualiZarr/discussions/884)) was on
the **write** side: the coordinator accumulated ~85 KB kerchunk JSON × 236K
files, and `append_dim` re-read O(n) metadata at ~87K timesteps. This pipeline
avoids both by construction — refs stream straight into icechunk manifests via
`set_virtual_refs` (no JSON accumulation; builder RSS is flat per date), one
commit per date, and the time axis grows by 1/day (401 dates for the whole 0p4
era, not 473K timesteps).

`test_dask_read.py` evaluates the complementary **read** side on a
multi-process cluster (same pickling path as Coiled workers):

```bash
uv run test_dask_read.py --store stores/ecmwf-0p4-ens --steps 3
# or against the published source.coop store, no credentials needed:
AWS_DEFAULT_REGION=us-west-2 uv run test_dask_read.py \
    --store s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test
```

Result (4 worker processes × 2 GB): ensemble mean+std of `t2m` over 51 members
× 2 dates × 3 steps = 306 virtual chunks (497 MB of GRIB decoded from
`ecmwf-forecasts`) in ~37 s, per-worker **peak RSS 355–394 MB**, no worker
restarts — identical numbers for the local and source.coop-hosted store. The
icechunk readonly session and the gribberish codec pickle to workers cleanly.
For Coiled, swap `LocalCluster` for `coiled.Cluster(...)` and follow the DAG
pattern of `../dev-test/ecmwf_ea_tp_source_coop_zarr.py` (one task per
(date, member), `as_completed`, batching + STS credential-timeout guard);
workers need `icechunk zarr xarray gribberish` installed.

## Materializing East Africa from the virtual store (the #884 answer, both planes)

Where does the #884 OOM actually live? It is a **virtual-reference write**
(metadata plane) problem — the CMORPH coordinator accumulated kerchunk JSON
for 236K files and `append_dim` re-read O(n) metadata. It is *not* a problem
of writing realized data. Measured on both planes here:

| plane | operation | peak RSS | notes |
|---|---|---|---|
| metadata (where #884 OOM'd) | builder: 359,805 virtual refs, one date | **468 MB**, flat | streamed `set_virtual_refs`, commit per date — nothing accumulates |
| data (this test) | dask cluster: read virtual store → EA subset → write realized zarr | coordinator **308 MB**, workers **~280 MB** | workers stream chunk → decode → subset → PUT |

The full run (`materialize_ea_from_icechunk.py`): 2 worker processes read the
**source.coop icechunk virtual store**, decode all 8,670 GRIB messages of `tp`
(2 dates × 51 members × 85 steps, ~5.2 GB streamed from `ecmwf-forecasts`),
subset to the ICPAC box (lat 25…-14, lon 19…55 → 98×90), and write a 306 MB
realized zarr (chunks `(1,1,85,98,90)`, one object per date+member — the
layout of `../dev-test/ecmwf_ea_tp_source_coop_zarr.py`) to a second
source.coop path in **470 s (18.4 msgs/s)**. Verified: anonymous public read,
bit-exact against the virtual store, zero NaNs.

```bash
source .env && uv run materialize_ea_from_icechunk.py \
    --source s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test \
    --dest   s3://e4drr-project/forecasts/ecmwf_ea_tp_0p4_zarr_test
```

Realized output (public):
`s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ea_tp_0p4_zarr_test`

Three gotchas encountered, relevant for the Coiled scale-up:

- **S3 `SlowDown` throttling** from `ecmwf-forecasts` under burst range-GETs.
  Fix: `StorageRetriesSettings(max_tries=10, ...)` +
  `config.max_concurrent_requests = 8` on the icechunk repo, `ZARR_ASYNC__CONCURRENCY=6`,
  2 threads/worker. On Coiled, bound per-worker fan-out before adding workers.
- **Strip inherited encodings** before `to_zarr`: the source arrays carry the
  read-only gribberish serializer, which cannot encode.
- **`AWS_ENDPOINT_URL` env leakage**: icechunk's Rust client auto-reads it, so
  proxy creds in the env silently reroute the (anonymous, direct-AWS) source
  reads — capture the endpoint for the dest writer and pop it from the env.

## Publishing to source.coop

Published test store (public, anonymous read):
`s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test`
(region `us-west-2`, no endpoint override).

Empirical findings with source.coop credentials (`.env` with
`AWS_ENDPOINT_URL=https://data.source.coop`):

- The `data.source.coop` proxy supports plain PUT/GET/LIST/DELETE and even
  conditional PUT (`If-None-Match: *`), but **not** the full S3 API icechunk
  commits need — `Repository.create()`'s commit fails in `update_repo_info`
  (service error) and batch `DeleteObjects` returns `NoSuchBucket`.
- The proxy credentials are **not** AWS STS tokens: direct writes to
  `us-west-2.opendata.source.coop` fail with `InvalidAccessKeyId` (the older
  `ecmwf_ea_tp_source_coop_zarr.py` used real STS creds and wrote directly).
- **Working pattern: build + commit locally, publish as a plain file sync**
  through the proxy — icechunk's local layout is identical to its S3 layout
  (59 objects, 19 MB, ~5 s to upload), and *reading* an icechunk repo needs
  only GET/LIST, which works anonymously on the public opendata bucket. This
  mirrors the reference script's own conclusion: keep transactions off the
  source.coop write path.

```python
# publish: sync the local store tree via the proxy (see git history for the
# exact snippet) -- boto3 upload_file of every file under stores/ecmwf-0p4-ens
# to e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test/

# consume from anywhere, no credentials:
import icechunk, xarray as xr
storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="e4drr-project/forecasts/ecmwf_ifs_0p4_icechunk_virtualdataset_test",
    region="us-west-2", anonymous=True, force_path_style=True)
auth = icechunk.containers_credentials(
    {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
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
