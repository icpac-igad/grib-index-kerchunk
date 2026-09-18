# ECMWF IFS ENS — Icechunk store complete and published

**v20260819.** The par → Icechunk conversion is finished for all four forecast
runs, and the store is mirrored to Source Coop where it reads with no
credentials at all. This document records what exists, how to open it, and how
big it actually is — because the number that matters (**~1.5 PB of GRIB
addressed by a 35 GB store**) is not visible from any file listing.

---

## 1. Status

| | |
|---|---|
| store (source of truth) | `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4` |
| published mirror | `s3://e4drr-project/forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd` via `https://data.source.coop` |
| tip snapshot | `1JF2A0G2XAP2GN4R6JS0` (identical on both) |
| groups | 12 — `{0p4, 49r1, 50r1} × {00z, 06z, 12z, 18z}` |
| date-runs | **5,023**, zero unwritten |
| chunk references | **2,054,129,675** |
| store size | 35.27 GB / 179,382 objects |
| verification | all 12 groups open and decode anonymously, `finite=1.000` |

Conversion of the 12z/18z remainder ran 2026-08-17 → 08-18: **2,406 dates,
0 failures, 33.4 h** on the sequential driver. The mirror moved the 16.19 GB
delta on 2026-08-18 and was pruned and re-verified on 2026-08-19.

---

## 2. Opening the store

No credentials are needed. The store metadata comes from Source Coop; the data
itself is fetched by byte-range from the public ECMWF bucket on AWS, so both
containers are declared anonymous.

```python
# /// script
# dependencies = ["icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "gribberish>=1.4"]
# ///
import icechunk, xarray as xr
import gribberish.zarr          # registers the "gribberish" Zarr v3 codec

storage = icechunk.s3_storage(
    bucket="e4drr-project",
    prefix="forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd",
    endpoint_url="https://data.source.coop",
    region="us-east-1", anonymous=True, from_env=False, force_path_style=True,
)
auth = icechunk.containers_credentials(
    {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})

# Disable eager manifest preload. source.coop returns sporadic HTTP 500s on a
# few percent of GETs; the default open prefetches thousands of manifests in
# parallel, so one of them tends to draw a 500 and the whole open raises.
cfg = icechunk.RepositoryConfig.default()
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))

repo = icechunk.Repository.open(storage, config=cfg,
                                authorize_virtual_chunk_access=auth)
session = repo.readonly_session("main")

ds = xr.open_zarr(session.store, group="49r1/00z",
                  consolidated=False, zarr_format=3)
print(ds.sizes)          # time=804, number=51, step=85, latitude=721, longitude=1440
```

Same store from GCS, for anything running inside the project:

```python
storage = icechunk.gcs_storage(bucket="gik-ecmwf-aws-tf",
                               prefix="icechunk/ecmwf-ens-v4",
                               service_account_file="/path/sa-key.json")
```

Reading one field is one HTTP range request against the original GRIB:

```python
t2m = ds.t2m.sel(time="2026-05-12").isel(number=0, step=0)   # control, analysis
print(float(t2m.mean()))     # ~280 K -- decoded through gribberish, no download
```

**Retry your reads.** A single-attempt read against Source Coop will
occasionally fail on a perfectly healthy store — three attempts with a short
sleep is enough:

```python
for attempt in range(3):
    try:
        field = ds.t2m.isel(time=-1, number=0, step=0).values
        break
    except Exception:
        if attempt == 2: raise
        time.sleep(3)
```

---

## 3. What is in it

Every group is `(time, number, step[, isobaricInhPa], latitude, longitude)`
with one 2-D field per chunk. 51 members throughout (control = `number 0`).

| group | vars | dates | steps | levels | grid | span |
|---|---|---|---|---|---|---|
| `0p4/00z` | 19 | 401 | 85 | 9 | 451×900 | 2023-01-18 → 2024-02-28 |
| `0p4/06z` | 19 | 400 | 49 | 9 | 451×900 | 2023-01-18 → 2024-02-27 |
| `0p4/12z` | 19 | 399 | 85 | 9 | 451×900 | 2023-01-18 → 2024-02-27 |
| `0p4/18z` | 19 | 399 | 49 | 9 | 451×900 | 2023-01-18 → 2024-02-27 |
| `49r1/00z` | 59 | 804 | 85 | 13 | 721×1440 | 2024-02-29 → 2026-05-12 |
| `49r1/06z` | 59 | 804 | 49 | 13 | 721×1440 | 2024-02-28 → 2026-05-11 |
| `49r1/12z` | 59 | 804 | 85 | 13 | 721×1440 | 2024-02-28 → 2026-05-11 |
| `49r1/18z` | 57 | 804 | 49 | 13 | 721×1440 | 2024-02-28 → 2026-05-11 |
| `50r1/00z` | 54 | 52 | 85 | 14 | 721×1440 | 2026-05-13 → 2026-07-03 |
| `50r1/06z` | 54 | 52 | 49 | 14 | 721×1440 | 2026-05-12 → 2026-07-02 |
| `50r1/12z` | 54 | 52 | 85 | 14 | 721×1440 | 2026-05-12 → 2026-07-02 |
| `50r1/18z` | 54 | 52 | 49 | 14 | 721×1440 | 2026-05-12 → 2026-07-02 |

The step axis is **run-dependent, not era-dependent**: 00z/12z run to 360h
(85 steps, 3-hourly then 6-hourly from 150h); 06z/18z are short-range and stop
at 144h (49 steps, 3-hourly). The short axis is an exact prefix of the long one.

---

## 4. The real size of the dataset

A reference is one GRIB message: a single 2-D field for one
(date, member, step, variable[, level]). The store holds nothing but those
pointers, so its own size says nothing about the data it addresses.

| quantity | value |
|---|---|
| chunk references (exact, from the manifests) | **2,054,129,675** |
| store on Source Coop | **35.27 GB** in 179,382 objects → **17.2 bytes per reference** |
| **GRIB bytes addressed** (the source data these references point at) | **≈ 1.5 PB** |
| **float32 materialised** (what an equivalent dense Zarr/NetCDF copy would cost) | **≈ 7.6 PB** |
| ratio, store : GRIB addressed | **1 : 42,700** |
| ratio, store : materialised float32 | **1 : 214,000** |

Per era, at full slot capacity:

| era | reference slots | mean GRIB message | GRIB addressed | as float32 | per-chunk float32 |
|---|---|---|---|---|---|
| 0p4 | 453,570,183 | 0.40 MB | 181.5 TB | 736.4 TB | 1.62 MB |
| 49r1 | 1,831,156,632 | 0.81 MB | 1,482.9 TB | 7,604.7 TB | 4.15 MB |
| 50r1 | 130,775,424 | 0.80 MB | 104.8 TB | 543.1 TB | 4.15 MB |
| **total** | **2,415,502,239** | | **1.77 PB** | **8.88 PB** | |

**Capacity vs actual: 2.42 billion slots exist, 2.05 billion (85.0%) carry a
reference.** The 15% gap is not data loss — it is variables that do not exist
on every date of an era (the ENS variable set drifts across 2024→2026, and an
array is created at full time length the first time a variable appears, so
earlier dates read NaN). The petabyte figures above are quoted at capacity;
scaled to the 85% actually populated they are **1.5 PB** of GRIB and
**7.6 PB** of float32, which is what the summary table reports.

So: **a 35 GB store, streamed on demand, stands in for one and a half petabytes
of ECMWF ensemble GRIB — and for seven and a half petabytes if anyone were to
decode and store it densely.** Nothing was copied to build it.

---

## 5. Reproducing these numbers

The reference count is exact — it comes from the manifest index, not a sample:

```python
tip  = repo.lookup_branch("main")
mans = repo.list_manifest_files(tip)
refs  = sum(m.num_chunk_refs for m in mans)
bytes_ = sum(m.size_bytes    for m in mans)
print(f"{len(mans):,} manifests, {refs:,} refs, {bytes_/1e9:.2f} GB "
      f"({bytes_/refs:.1f} bytes/ref)")
# 179,352 manifests, 2,054,129,675 refs, 35.27 GB (17.2 bytes/ref)
```

Slot capacity and materialised size come from the array shapes — every chunk is
one 2-D field, so the count is the product of the non-spatial dimensions:

```python
import numpy as np, zarr
COORD = {"time","number","step","isobaricInhPa","latitude","longitude"}
g = zarr.open_group(store=session.store, path="49r1/00z", mode="r", zarr_format=3)
cap = sum(int(np.prod(g[n].shape[:-2])) for n in g.array_keys() if n not in COORD)
print(f"{cap:,} slots, {cap * 721*1440*4 / 1e12:.1f} TB as float32")
# 582,051,780 slots, 2417.2 TB as float32
```

The GRIB side is measured from the pars' own `[url, offset, length]` triplets —
sampled, since reading every length would mean reading every par:

```python
df = parse_par(one_member_par)          # build_ecmwf_icechunk.parse_par
print(df.length.mean(), df.length.median())
# 49r1: mean 810 KB, median 718 KB   0p4: mean 400 KB (0.33-0.63 MB by regime)
```

Eleven pars across all three eras, both stream lengths and control plus
perturbed members. The 0p4 mean is the softest number here: early-2023 dates
pack at ~630 KB per message against ~330 KB later, so treat 0p4's 181 TB as
±50%. It is 10% of the total, so the petabyte headline is unaffected.

---

## 6. Known limitations

- **5,335 missing (member, step) chunks**, 0.031% of the corpus, scattered as
  isolated singletons. Verified upstream: for `20240302 06z ens_03` step 0 the
  ECMWF `.index` publishes 83 records but the par contains no step-0 rows at
  all. These are par-generation gaps in `run_lithops_ecmwf.py`, not conversion
  losses, and they are ~20× more common in 06/12/18z than in 00z. They read as
  NaN. Fixing them means regenerating those dates' pars.
- **`mn2t6` / `mx2t6` in `49r1/06z` are empty arrays** — 0 chunks written.
  ECMWF publishes 6-hour extrema only in the ≥150h step regime, which the
  144h-capped 06z/18z runs never reach; `49r1/18z` correctly omits them. A
  consumer would see two variables that silently read NaN.
- **History is not published.** The mirror carries only what the `main` tip
  references (35 GB of the 126 GB in GCS, the rest being superseded manifests
  and snapshots). The published repo reads, but not *as of* an earlier commit.
- **Source Coop returns sporadic HTTP 500s.** Retry reads; disable manifest
  preload on open. Neither indicates a problem with the store.

---

## 7. Provenance

| | |
|---|---|
| conversion driver | `ecmwf/icechunk-par/backfill_all_eras.py` (sequential, one date per subprocess) — commit `9dfebc9` |
| builder | `ecmwf/icechunk-par/build_ecmwf_icechunk.py` |
| mirror | `icechunk-dask-frisky/mirror_gcs_to_source_coop.py --live-only [--prune]` — commit `c49f6ca` |
| completeness check | `ecmwf/icechunk-par/verify_store_completeness.py` (manifest-level, per group) |
| health check | `ecmwf/icechunk-par/check_store_health.py` |
| logs | `~/ecmwf-rebuild/logs/{serial-12z-18z, verify-sweep, mirror-v4-full, mirror-v4-prune}.log` |
| pars | `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf` (authoritative; **not** the HuggingFace copy, which still holds the defective March-2026 pars) |

Rates measured on this run, for planning the next one: sequential conversion
33–74 s/date, rising roughly linearly as a group fills because each per-date
commit rewrites manifests carrying all prior references. The mirror moved
16.19 GB in 39 min (35 objects/s) — request-rate bound through the proxy at
~196 KB average object size, not bandwidth bound.
