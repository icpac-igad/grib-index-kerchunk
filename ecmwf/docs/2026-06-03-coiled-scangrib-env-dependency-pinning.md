# The scan_grib Coiled env: why it's pinned to old kerchunk, and how to recreate it

**Date:** 2026-06-03
**Scope:** the dependency fault line that forces the ECMWF `scan_grib`
preprocessing to run on an **old, frozen** software stack, the exact pins,
and reproducible recreation (pinned `environment.yml` + `Dockerfile` +
Coiled software env + local client).

---

## 1. The breakage, in one sentence

The kerchunk version that the scan path needs (`_map_grib_file_by_group`
→ `_extract_single_group`) only works with **zarr 2**; every kerchunk new
enough to be "current" requires **zarr 3**, and zarr 3 breaks the
reference-filesystem datatree open that `_extract_single_group` performs.
So the scan is bound to a 2024-era stack and cannot simply be upgraded.

## 2. What actually fails on a modern stack (observed)

Reproduced in this work while trying to run the scan locally / rebuild a
fresh env:

1. **kerchunk import** — modern kerchunk does `from zarr.core.array_spec
   import ArraySpec` → requires **zarr 3**. With zarr 2 it fails:
   `ModuleNotFoundError: No module named 'zarr.core.array_spec'`.
2. **The scan itself under zarr 3** — `_extract_single_group` builds a
   grib_tree and opens it as a datatree
   (`xr.open_datatree("reference://", engine="zarr")`). On zarr 3 / recent
   fsspec this raises, in sequence:
   - `ValueError: Reference-FS's target filesystem must have same value of asynchronous`
   - `ValueError: 'path' was provided but is not used for FSMap store_like objects`
3. **Client ↔ scheduler skew** — a newer local `distributed` can't talk to
   the 2024 scheduler: `TypeError: Scheduler.identity() got an unexpected
   keyword argument 'n_workers'`.

Net: you can have a kerchunk that *imports* (zarr 3) but then *fails at
scan time*, or a kerchunk that *can't import* (zarr 2). The only stable
point is the **2024-10 combination** below.

## 3. Why the old kerchunk (0.2.6) codebase is required

- **kerchunk 0.2.6** ships `kerchunk.grib2.scan_grib` + a vendored
  `_extract_single_group` / `_map_grib_file_by_group` that target the
  `datatree`/zarr-2 API. It has **no** `kerchunk._grib_idx` module.
- The repo's newer `ecmwf/dev-test/fmrc_utils.py` was refactored to
  `from kerchunk._grib_idx import _extract_single_group` — which **does not
  exist** in 0.2.6, so it can't run on the pinned image. The drivers
  therefore call the image's **baked-in** `dynamic_zarr_store`
  (`gfs/dynamic_zarr_store.py`: `_map_grib_file_by_group(fname, mapper=None)`,
  `_extract_single_group(grib_group, idx)`, `s3_parse_ecmwf_grib_idx`,
  importing only from `kerchunk.grib2`) — the version-matched primitives.
- Important: the per-level correctness does **not** depend on this old
  scan. The dump's `level` is a sequential index for multi-member files;
  the **realigner recovers true hPa from the `.index` `levelist`**
  downstream (see `2026-06-03-three-era-scan-run-verification-and-perlevel-fix.md`).
  So staying on kerchunk 0.2.6 for the scan is safe.

## 4. The exact pinned stack (captured from a live `gik-coiled-v6` worker, 2026-06-03)

Container `gik-coiled:v4` (built 2024-10-30 from the unpinned
`devops/environment.yml` = conda-forge latest at that date):

| package | version | | package | version |
|---|---|---|---|---|
| python | 3.12.7 | | fsspec | 2024.10.0 |
| **kerchunk** | **0.2.6** | | s3fs | 2024.10.0 |
| **zarr** | **2.18.3** | | gcsfs | 2024.10.0 |
| numcodecs | 0.13.1 | | aiohttp | 3.10.10 |
| cfgrib | 0.9.14.1 | | botocore | 1.35.23 |
| eccodes | 2.37.0 | | google-cloud-storage | 2.18.2 |
| xarray | 2024.9.0 | | ujson | 5.10.0 |
| **dask** | **2024.10.0** | | numpy | 2.1.2 |
| **distributed** | **2024.10.0** | | pandas | 2.2.3 |
| cloudpickle | 3.1.0 | | pyarrow | 17.0.0 |
| msgpack | 1.1.0 | | lz4 | 4.3.3 |
| tornado | 6.4.1 | | toolz | 1.0.0 |

(`gribberish`, `virtualizarr`, `ecmwflibs` are NOT in this image.)

## 5. Client ↔ cluster version match (mandatory)

`coiled.function` submits from a local **client** to the cluster. The
client's `distributed`/`dask` must match the senv or the connection fails
(§2.3). Build a matched client:

```bash
micromamba create -p /scratch/gikclient -c conda-forge \
    python=3.12 dask=2024.10.0 distributed=2024.10.0 coiled fsspec s3fs
```

`coiled` itself is version-tolerant (1.134.1 worked as client against the
2024.10.0 scheduler); only `distributed`/`dask` must match. Mismatches in
cloudpickle/msgpack/etc. produce **warnings**, not errors.

## 6. Reproducible recreation (if the prebuilt image is ever lost)

Artifacts added in `devops/`:

- **`environment-gik-coiled-pinned.yml`** — the exact pins from §4.
- **`Dockerfile.gik-coiled-pinned`** — micromamba base → installs the
  pinned env → bakes `gfs/dynamic_zarr_store.py` at `/app` (PYTHONPATH) so
  the worker's `import dynamic_zarr_store` resolves. Does **not** bake AWS
  or GCS creds (driver forces anon s3 + uploads the GCS key at runtime).

```bash
# build + push
docker build -f devops/Dockerfile.gik-coiled-pinned \
  -t us-east1-docker.pkg.dev/sewaa-416306/coiled/gik-coiled:pinned .
docker push us-east1-docker.pkg.dev/sewaa-416306/coiled/gik-coiled:pinned

# register as a Coiled software env
coiled software create gik-coiled-pinned \
  --container us-east1-docker.pkg.dev/sewaa-416306/coiled/gik-coiled:pinned \
  --workspace gcp-sewaa-nka

# run the drivers against it (instead of gik-coiled-v6)
python ecmwf_0p4_coiled_preprocessing.py  --date 20230601 --run 00 --software gik-coiled-pinned ...
python ecmwf_49r1_coiled_preprocessing.py --date 20250515 --run 00 --software gik-coiled-pinned ...
python ecmwf_50r1_coiled_preprocessing.py --date 20260513 --run 00 --software gik-coiled-pinned ...
```

The three drivers now **default to `--software gik-coiled-pinned`** — so you
must build+register that image (above) before running with defaults. Until
then, pass `--software gik-coiled-v6` (the existing prebuilt image with
identical pins). Because the pins are identical, behaviour matches exactly.

## 7. Why we did NOT modernize instead

A fresh "recent kerchunk" senv was considered and rejected for now: recent
kerchunk needs zarr 3, which reintroduces the §2.2 datatree failure. Going
modern would require finding a kerchunk+zarr-3+fsspec+xarray combo where
the reference-FS datatree open works **and** re-validating the scan output
end-to-end — a separate, riskier effort. The pinned 2024-10 stack is the
known-good, lowest-risk path and is now fully reproducible via §6.

## 8. Maintenance triggers

Rebuild/repin only when the **source GRIB schema** changes (new era), not
on a routine cadence. The pinned env is intentionally frozen; upgrading any
of kerchunk/zarr/dask/distributed/fsspec risks the §2 failures and must be
validated against a one-file scan + the realigner gate before adoption.

## References
- Run/verification + per-level fix: `2026-06-03-three-era-scan-run-verification-and-perlevel-fix.md`
- Realigner scope/fix: `../lithops-cr-gik-ecmwf/2026-06-01-step2-realigner-fix-scope.md`
- Pinned artifacts: `devops/environment-gik-coiled-pinned.yml`, `devops/Dockerfile.gik-coiled-pinned`
