# GIK v1.0.0 — Release Notes

First tagged release of `grib-index-kerchunk`. This release marks the
point at which the GEFS and ECMWF pipelines are **production-ready**,
**validated bit-exactly against Herbie**, and **publicly mirrored to
HuggingFace**.

---

## Highlights

### Production pipelines

- **GEFS** (NOAA, 30-member ensemble, 0.25°)
  - Full 4-step pipeline: template → index → parquet → xarray.
  - Cloud Run / Lithops orchestration in `gefs/lithops-cr-gik-gefs/`
    handles multi-year backfills (~$0.092 / date blended cost; full
    6-year backfill validated end-to-end).
  - Public dataset: <https://huggingface.co/datasets/E4DRR/gik-gefs-par>
    — 1,924 dates from 2020-09-25 → 2025-12-31 at 00z, mirrored as
    per-member parquets + 64 monthly aggregates + a catalog index at
    the repo root.

- **ECMWF IFS** (51-member ensemble, 0.25°)
  - Same 4-step pipeline adapted for 51 members and all four run hours.
  - Cloud Run / Lithops orchestration in `ecmwf/lithops-cr-gik-ecmwf/`.
  - Public dataset: <https://huggingface.co/datasets/E4DRR/gik-ecmwf-par>
    — 144,228 per-member parquets + catalog (~17.8 GB total).

### Validation

GIK references produce **bit-identical** APCP / TP fields to
[Herbie](https://herbie.readthedocs.io/) because both paths read the
same source bytes from S3:

| Pipeline | Validator | Result |
|---|---|---|
| GEFS local | `gefs/validate_gik_vs_herbie_2022_2026.py` | r=1.0, RMSE=0 across 50+ random dates |
| GEFS HF aggregate (consumer) | `gefs/validate_hf_gik_vs_herbie.py` | r=1.0, RMSE=0 |
| ECMWF | `ecmwf/validate_gik_vs_herbie_2025.py` | r > 0.9999 |

### Filter-pushdown HuggingFace aggregates

Both `gefs/upload_parquets_to_hf.py` and `ecmwf/upload_parquets_to_hf.py`
produce **monthly aggregate parquets** sorted by `(date, member)` with
small row groups (60 for GEFS, 102 for ECMWF) + zstd compression. A
single-date filter pushdown reads only ~1.5–5 MB of an otherwise
44–80 MB monthly file via HuggingFace range requests.

End-user pattern (canonical):

```python
import pandas as pd
df = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet",
    filters=[("date", "=", "20240615"), ("member", "=", "gep01")],
)
```

See [`gefs/hf_README.md`](gefs/hf_README.md) for the full lazy-Dask
ensemble pattern; same shape works for ECMWF.

---

## Repository cleanup

`v1.0.0` archives every experimental, superseded, or
application-specific script under `<datasource>/dev-test/` so the main
path of each datasource folder contains *only* the canonical 4-step
pipeline + Herbie / Coiled / HF-aggregate validators.

| Datasource | Main-path scripts before / after | Archived to dev-test |
|---|---|---|
| ECMWF | 38 → **14** | 22 files (legacy creators, AIFS, Icechunk + source.coop variants, older validators, plot/test one-offs) |
| GEFS | 17 → **12** | 8 files + bundled template parquet |

Per-file rationale and decisions are recorded in
[`RELEASE_CLEANUP_PLAN.md`](RELEASE_CLEANUP_PLAN.md). Each move shipped
as a separate atomic commit with a clear "moved (no functional change)"
list — see commits `5ae884e`..`f1db8eb`.

---

## Documentation

- [`README.md`](README.md) — top-level overview + 4-step diagram + supported products + quickstart
- [`ecmwf/README.md`](ecmwf/README.md) — ECMWF step-by-step usage, scripts table, HF publishing recipe
- [`gefs/README.md`](gefs/README.md) — GEFS step-by-step usage, lazy-Dask consumer pattern, HF dataset description
- [`ecmwf/hf_README.md`](ecmwf/hf_README.md) and [`gefs/hf_README.md`](gefs/hf_README.md) — dataset cards published on HuggingFace

---

## Known limitations

- **GEFS coverage**: only the 00z run-hour is published in v1.0. Adding
  06/12/18z is straightforward (re-run `lithops-cr-gik-gefs/` chain
  with `--run 06` etc.) but not in scope for this release.
- **GEFS pre-2020-09-25**: NOAA's reforecast/realtime archive starts on
  Sep 25, 2020. Earlier dates have no upstream data.
- **ECMWF coverage**: 2024-03 → 2026-02. Earlier dates predate the
  ECMWF open-data S3 bucket's continuous archive.
- **AWS / GCP credentials**: production pipelines need a service
  account on the bucket-owning project. See `lithops-cr-gik-*/SETUP_NEW_MACHINE.md`.

---

## What's next (post-v1.0)

- ECMWF monthly aggregates with the optimised filter-pushdown layout
  (`upload_parquets_to_hf.py --aggregate`) — script ported in commit
  `8fa1284`; the ~7 h re-aggregate run is a follow-up task.
- `validate_hf_ecmwf_gik_vs_herbie.py` — ECMWF analogue of the GEFS
  HF-aggregate validator.
- GFS pipeline modernisation — currently legacy in `gfs/`.
- CFS pipeline — experimental in `cfs/`.

---

## Acknowledgements

GIK is developed by [ICPAC](https://www.icpac.net/) (IGAD Climate
Prediction and Applications Centre) for continuous climate-risk
monitoring over East Africa, funded by:

- **E4DRR** — UN Complex Risk Analytics Fund (CRAF'd)
- **SEWAA** — Strengthening Early Warning Systems for Anticipatory Action

Method foundation: [dynamic-Grib-chunking](https://github.com/asascience-open/nextgen-dmac/commit/6b3286627070c36127ec97b7dbbb88b0ab481f06).

License: Apache 2.0 — see [`LICENSE`](LICENSE).
