# cost-eval — reproducible cost/compute evaluation of the GIK method

This directory contains a self-contained, primary-measurement harness for
verifying the cost claims made about Grib-Index-Kerchunk (GIK) in
[`gik-data-streaming.tex`][tex]. The tex paper *states* the headline numbers;
this harness *re-derives them from object-store reality*, so any reader can
re-run, change pricing knobs, and audit the maths.

[tex]: ../../../69f087edf4021d2f2371dd2a/gik-data-streaming.tex

## What this harness asserts (vs what the tex claims)

| Question | Tex answer (cited) | What `cost-eval/` does |
|---|---|---|
| How much source GRIB does the catalog index? | ~242 TB ECMWF, ~47 TB GEFS | HEAD-probe a sample → extrapolate per-product |
| How small are the parquet references vs the GRIB? | ~10,000× compression | Measures both bytes; computes ratio |
| Is dynamical.org's ARCO Zarr really that big? | ~289 TB combined | Lists the public Zarr stores → real bytes |
| What does each access path cost per analysis slice? | $0.05–$0.09 cited | Measures egress, GETs, vCPU directly |
| Does "tens of thousands saved" hold up? | Yes, ~$80k/yr | Re-derives from measured TB × current price list |
| At what consumer workload does GIK lose to ARCO? | Not addressed | `07_cost_model.py` plots the break-even curve |

The deliberate non-goals: we do **not** re-run the Pearson-correlation validation
(already in `gefs/validate_*` and `ecmwf/validate_*`) and we do **not** restate
the pipeline architecture (already in `CLAUDE.md`).

## Files

```
config.yaml                    # pricing, sample dates, workload spec
01_measure_grib_sizes.py       # S3 HEAD → bytes per (date, member, step)
02_measure_parquet_sizes.py    # GCS list → parquet ref bytes              (todo)
03_measure_dynamical_zarr.py   # zarr store listing → bytes                (todo)
04_benchmark_herbie.py         # live: bytes, wall, vCPU for one slice     (todo)
05_benchmark_gik_stream.py     # live: bytes, wall, vCPU, GET count        (todo)
06_benchmark_dynamical_open.py # live: bytes, wall, vCPU via xr.open_zarr  (todo)
07_cost_model.py               # combine measurements + pricing → matrix   (todo)
08_report.py                   # render results/REPORT.md                  (todo)
results/                       # CSVs + JSON summaries (created on first run)
```

## Quick start

Each script is a self-describing `uv run` PEP-723 module:

```bash
# 1) Inventory: measure source GRIB volume for the 7-date sample
uv run cost-eval/01_measure_grib_sizes.py

# Override knobs without editing config.yaml:
uv run cost-eval/01_measure_grib_sizes.py \
    --products gefs \
    --dates 20251201 20251202 \
    --concurrency 16
```

Sample run cost / wall-clock:

| Step | What runs | S3 cost | Wall (defaults) |
|---|---|---|---|
| 01 | 17,605 HEAD requests (anonymous S3 — $0) | $0 | ~5 min |
| 02 | GCS LIST on `gs://gik-{ecmwf,gefs}-aws-tf/` | $0 | <1 min |
| 03 | Anonymous reads of `.zmetadata` from dynamical.org | $0 | <1 min |
| 04 | 1 date × 4 members × 9 steps Herbie download (~few GB) | a few cents egress | minutes |
| 05 | Same slice via GIK byte-range (~tens of MB) | sub-cent | seconds |
| 06 | Same slice via `xr.open_zarr` on dynamical.org | sub-cent | seconds |

## Design notes

- **All anonymous.** No AWS credentials, no GCS service accounts — every probe
  hits the public-data path the tex describes.
- **Sample-then-extrapolate.** Default sample is 7 dates × 00z (see
  `config.yaml → sample.dates`). The cost model multiplies per-date sums by
  the published-archive date count in `config.yaml → archive.*`. Override
  with `--dates`/`--days`.
- **Pricing is a knob, not a constant.** Every dollar figure in the final
  report flows from `config.yaml → pricing.*`. Update the `as_of` field and
  the dollar columns refresh on the next `07_cost_model.py` run.
- **Stakeholder ledger.** The final report breaks costs by *who pays*:
  provider (NOAA / ECMWF) → publisher (regional centre) → analyst.
  This is what makes the "win–win–win" framing checkable rather than
  rhetorical.

## Outputs

`results/grib_sizes_<product>.csv` — one row per probed object:

| product | date | run | member | step | url | size_bytes |
|---|---|---|---|---|---|---|
| gefs | 20251201 | 00 | gep01 | 0 | s3://noaa-gefs-pds/… | 14,428,476 |
| ecmwf | 20251201 | 00 | packed | 0 | s3://ecmwf-forecasts/… | 6,172,399,416 |

`results/01_grib_sizes_summary.json` — per-product totals, means, and the
extrapolation to the full published-archive window using
`archive.<product>.n_dates_published`.

Later scripts append more CSVs (`parquet_sizes_*.csv`, `dynamical_sizes_*.csv`,
`benchmark_*.csv`) and a single `REPORT.md` that ties everything together.

## Reproducibility checklist

- ✅ Pinned dependencies via PEP-723 inline metadata (each script declares its own)
- ✅ Anonymous public-data access — no environment-specific credentials
- ✅ Pricing dated in `config.yaml → pricing.as_of`
- ✅ Sample window pinned to specific YYYYMMDD strings (not "last 7 days")
- ✅ Outputs are diff-friendly CSVs + JSON, safe to commit
