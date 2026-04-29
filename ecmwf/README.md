# ECMWF IFS Ensemble — GIK Pipeline

Lightweight parquet reference files that turn ECMWF's 3–4 GB GRIB files
into ~140 KB virtual datasets, enabling Dask-based parallel analysis
without downloading the raw data. Mirrors the same 4-step pipeline used
for GEFS (`../gefs/`), adapted for the 51-member IFS ensemble.

| | Value |
|---|---|
| Source | `s3://ecmwf-forecasts/` (anonymous) |
| Members | 51 (1 control + ens01–ens50) |
| Timesteps | 85 (0–144 h at 3 h, 150–360 h at 6 h) |
| Grid | 0.25° global (721 × 1440) |
| Daily volume per run hour | ~340 GB GRIB → ~7 MB of GIK refs |
| Public dataset | <https://huggingface.co/datasets/E4DRR/gik-ecmwf-par> |

---

## The 4-step pipeline applied to ECMWF

| Step | Script(s) | Output |
|------|-----------|--------|
| **1. Template** | `utils_ecmwf_step1_scangrib.py`, `ecmwf_index_preprocessing.py` | `gik-fmrc-ecmwf-YYYYMMDD.tar.gz` (~30 MB) |
| **2. Index** | `ecmwf_index_processor.py` | per-member kerchunk dict from each `.idx` |
| **3. Build parquet & save** | `ecmwf_three_stage_multidate.py` (local) or `lithops-cr-gik-ecmwf/` (Cloud Run) | per-member parquets in GCS / on HF |
| **4. Stream → xarray** | `run_single_ecmwf_to_zarr_gribberish.py` (single member), `stream_cgan_variables_coiled_simple.py` (Coiled Dask cluster) | xarray Dataset / `.zarr` |

---

## Quick start — single date, single machine

End-to-end on one date with the local three-stage runner:

```bash
cd ecmwf
cp .env.example .env             # edit GCS bucket / SA + HF_TOKEN
uv run ecmwf_three_stage_multidate.py --date 20250101 --run 00z
```

Outputs 51 per-member parquets to `gs://${GCS_BUCKET}/run_par_ecmwf/2025/01/20250101/00z/`.

For a multi-date backfill at scale, use the Cloud Run path —
see [`lithops-cr-gik-ecmwf/README.md`](lithops-cr-gik-ecmwf/README.md).

---

## Quick start — read one date as xarray (consumer side)

```python
import ast, fsspec, gribberish
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

# Per-member parquet — no aggregation step needed for ECMWF (each parquet
# is ~140 KB and lives at a stable path on HF)
ref = hf_hub_download(
    repo_id="E4DRR/gik-ecmwf-par", repo_type="dataset",
    filename="run_par_ecmwf/2025/01/20250101/00z/2025010100z-ens01.parquet",
)
df = pd.read_parquet(ref)
zstore = {row["key"]: row["value"] for _, row in df.iterrows()}
# (decode JSON-encoded list values as needed; see hf_README.md)
```

For the lazy-Dask ensemble pattern over all 51 members — see
[`hf_README.md`](hf_README.md) "Open Parquets as Lazy xarray Dataset".

---

## Scripts on the main path (v1.0)

| File | Step | Purpose |
|------|------|---------|
| `ecmwf_util.py` | shared | Variable definitions, grid utilities, common helpers |
| `utils_ecmwf_step1_scangrib.py` | 1 | scan_grib helpers used to build the template |
| `ecmwf_index_preprocessing.py` | 1 | Build the deflated zarr store + mapping parquets |
| `ecmwf_index_processor.py` | 2 | Index lookup against an existing template |
| `ecmwf_three_stage_multidate.py` | 1+2+3 | Local runner — useful for single-date / dev work |
| `lithops-cr-gik-ecmwf/` | 3 | Cloud Run / Lithops orchestration for production |
| `run_single_ecmwf_to_zarr_gribberish.py` | 4 | Read one parquet → zarr |
| `stream_cgan_variables_coiled_simple.py` | 4 | Coiled Dask cluster — distributed ensemble streaming |
| `compare_gik_herbie.py` | validation | Side-by-side maps + metrics vs Herbie |
| `fetch_tp_herbie.py` | validation | Herbie fetch helper for TP |
| `validate_gik_vs_herbie_2025.py` | validation | Year-long random-date Herbie comparison |
| `upload_parquets_to_hf.py` | publish | Mirror per-member parquets + monthly aggregates + catalog to HF |
| `hf_README.md` | docs | Dataset card on `E4DRR/gik-ecmwf-par` |

Plus folders:
- `lithops-cr-gik-ecmwf/` — Cloud Run + Lithops setup (Dockerfile, lithops_config, backfill scripts)
- `docs/` — method documentation including `gik_vs_herbie_comparison.md`
- `gik_vs_herbie/` — validation artefacts (PNGs, JSON metrics)
- `plots/` — reference plots
- `dev-test/` — older variants, experiments, and superseded scripts
  ([`RELEASE_CLEANUP_PLAN.md`](../RELEASE_CLEANUP_PLAN.md) §3 has the full inventory)

---

## Validation

GIK reads the **same source bytes** as
[Herbie](https://herbie.readthedocs.io/) — only the path to those bytes
differs. Validation across 21 dates spanning 2024–2025 shows:

- **Pearson r > 0.9999** for both ensemble mean and spread
- **RMSE < 2e-04 m** for total precipitation
- Tiny residuals trace to gribberish vs cfgrib floating-point representation

Re-run the comparison:

```bash
uv run validate_gik_vs_herbie_2025.py --max-members 5 --dates 20250101,20250615
```

Output PNGs land in `gik_vs_herbie/`, JSON metrics in `validation_log.txt`.

---

## Publishing to HuggingFace

```bash
cp .env.example .env             # set HF_TOKEN, HF_REPO, GCS_BUCKET, GCS_SA_FILE
uv run upload_parquets_to_hf.py --catalog        # build root catalog.parquet
uv run upload_parquets_to_hf.py --aggregate      # per-month aggregates with filter pushdown
uv run upload_parquets_to_hf.py --sync           # incremental per-member upload (skip dates already on HF)
```

The aggregate mode produces one `run_par_ecmwf_agg/monthly_agg/{Y}/{MM}_{run}z.parquet`
per month-run, sorted by `(date, member)` with `row_group_size=102` and
zstd compression — so a single-date filter pushdown reads ~few MB instead
of the full ~80 MB monthly file.

---

## References

- [NOAA ECMWF on AWS](https://registry.opendata.aws/ecmwf-forecasts/)
- [Kerchunk](https://fsspec.github.io/kerchunk/)
- [gribberish](https://github.com/mpiannucci/gribberish) — fast GRIB decoding
- [Herbie](https://herbie.readthedocs.io/) — reference implementation we validate against
- [Coiled](https://coiled.io/) — managed Dask clusters used by `stream_cgan_variables_coiled_simple.py`
- [Lithops](https://lithops-cloud.github.io/) — serverless framework used for the Cloud Run backfill

---

## License

Apache 2.0 — see top-level [`LICENSE`](../LICENSE).
Developed by [ICPAC](https://www.icpac.net/) under the **E4DRR** (UN CRAF'd) and **SEWAA** projects.
