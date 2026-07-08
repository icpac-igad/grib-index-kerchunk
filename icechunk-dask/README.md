# icechunk-dask

Publishing the GIK Icechunk stores to **source.coop** and reading them back
anonymously. The stores hold *virtual* references: metadata lives on source.coop,
the actual GRIB bytes stay in the public AWS buckets (`noaa-gefs-pds`,
`ecmwf-forecasts`). A reader needs **no credentials**.

Published stores (bucket `e4drr-project`, endpoint `https://data.source.coop`):

| product | prefix | group(s) | virtual size |
|---|---|---|---|
| GEFS  | `forecasts/noaa_gefs_aws_s3_icechunk_vd`     | `0p25/00z`                     | ~697 TB |
| ECMWF | `forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd` | `0p4/00z`, `49r1/00z`, `50r1/00z` | 234 TB / 2.39 PB / 169 TB |

## Open a store (anonymous smoke test)

Each script opens the store with zero credentials and decodes one field to prove
it resolves end to end. Deps are declared inline (PEP 723), so `uv run` handles
everything — run with **no** `AWS_*` env vars set:

```bash
uv run smoke_test_published_gefs.py     # GEFS  (group 0p25/00z)
uv run smoke_test_published_ecmwf.py    # ECMWF (all three eras)
```

Open takes ~60 s (eager manifest preload is disabled so it survives source.coop's
sporadic 500s — see the docs below). Both print the virtual dataset size and end
with `RESULT: PASS`.

See **`OPENING_PUBLISHED_GEFS_ICECHUNK.md`** and
**`OPENING_PUBLISHED_ECMWF_ICECHUNK.md`** for the minimal open snippet to drop
into your own code, plus gotchas (group paths, the `gribberish.zarr` codec import,
`force_path_style`, per-era variable names).

## Publish / mirror a store to source.coop

`mirror_gcs_to_source_coop.py` copies a GCS-hosted store to source.coop by plain
object copy (resumable; skips objects already in sync). Needs GCS read creds and a
source.coop STS token (`.env`, ~1 h lifetime):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./coiled-data.json   # GCS read
source .env                                                # source.coop STS
uv run mirror_gcs_to_source_coop.py \
    --gcs-store gs://gik-gefs-aws-tf/icechunk/gefs-ens \
    --dest      forecasts/noaa_gefs_aws_s3_icechunk_vd \
    --sa-key    ./coiled-data.json --threads 32
```

If the STS token expires mid-run, refresh `.env` and re-run — it resumes.
