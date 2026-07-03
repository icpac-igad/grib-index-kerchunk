# Pressure-Level Validation — GIK (per-level-keys fix) vs Herbie

**Date:** 2026-06-23 · **Domain:** East Africa / ICPAC (19–55°E, 14°S–25°N)
· **Branch:** `ecmwf-50r1-template`

This folder validates that the **re-baked 49r1 par files** carry *physically
correct pressure-level data*, after the per-level-keys fix
(`grib-index-kerchunk` commit `4ca1c21`). It is the pressure-level companion
to the surface-precipitation study in
[`../GIK_vs_Herbie_Evaluation.md`](../GIK_vs_Herbie_Evaluation.md) (see its
**Section 7**).

---

## 1. The bug this proves is fixed

Before the fix, the par store collapsed **all 13 pressure levels** of an
isobaric variable onto **one** zarr reference per `(var, step, member)`. The
single stored byte-range pointed at whichever GRIB message came first, so a
consumer asking for `u` at 700 hPa could silently receive 250 hPa or 1000 hPa
data depending on the forecast step. Geopotential height `gh` collapsed onto
~1000 hPa (surface, ~0–100 gpm) where 500 hPa expects ~5700 gpm — a >9000 gpm
error. Every upper-air channel cGAN needs (Somali/Turkana jets, 500/300 hPa
trough-ridge, vertical wind shear) was unusable.

The fix emits **one reference per level**:

```
step_{NNN}/{var}/pl/{level_hPa}/{member}/0.0.0
```

This evaluation confirms each per-level reference resolves to the correct
pressure level, end-to-end.

---

## 2. How the evaluation was carried out

Three stages — get the references, *realize* the data, then plot/compare.

### Stage 1 — get the par files from GCS  → `_download_par.py`

The re-baked per-member par files live at
`gs://gik-ecmwf-aws-tf/run_par_ecmwf/2026/03/20260301/00z/` (51 files:
`...-control.parquet`, `...-ens_01.parquet` … `...-ens_50.parquet`). The
helper downloads them to `par_local/`:

```bash
uv run _download_par.py all      # all 51 members  (≈22 MB)
uv run _download_par.py sample   # control + ens_01 + ens_02 only
```

A par file is **not data** — it is a zarr *reference store*: a table of
`key → value` rows where each chunk key maps to a `[s3_url, offset, length]`
byte-range pointer into the ECMWF GRIB archive on S3.

### Stage 2 — *realize* the data from the references (streaming + **gribberish**)

This is the step that turns a reference into an actual field, performed inside
[`../../compare_gik_herbie_pressure.py`](../../compare_gik_herbie_pressure.py):

1. Load the par → dict `{key: value}`.
2. For a `(var, level, step, member)`, look up
   `step_{step:03d}/{var}/pl/{level}/{member}/0.0.0` → `[s3_url, offset, length]`.
3. **Stream only those bytes** from the ECMWF S3 archive (anonymous),
   `f.seek(offset); raw = f.read(length)` — one GRIB2 message, ~0.7 MB.
4. **Decode with `gribberish`:**
   `gribberish.parse_grib_array(raw, 0).reshape(721, 1440)`.
5. Subset to the ICPAC grid; stack all 51 members → ensemble mean & spread.

#### Why `gribberish`?

The par only points *at* GRIB messages; something has to decode the raw GRIB2
bytes into a numpy array — that decode is what "realizes" the data. We use
**`gribberish`** (a pure-Rust GRIB2 decoder) rather than eccodes/cfgrib
because:

- **It decodes an in-memory byte string directly** (`parse_grib_array(raw, 0)`).
  cfgrib/eccodes need an on-disk file, forcing a temp-file write per message —
  slow and I/O-heavy across 51 members × 13 levels × many steps.
- **No GRIB index rebuild.** cfgrib builds a `.idx` and opens the message
  through the eccodes C library; for single isolated byte-range messages that
  overhead dominates. gribberish goes straight bytes → array.
- **It is exactly what the production GIK streamer uses**
  (`stream_cgan_variables_coiled_simple.py`), so this validation exercises the
  *same* realization path consumers use, not a different decoder.

(The script keeps a cfgrib fallback only for environments without gribberish;
the validation runs reported here all used gribberish.)

### Stage 3 — Herbie ground truth + plots  → `compare_gik_herbie_pressure.py`

For the same date/step/level, Herbie fetches the ECMWF `enfo` ensemble
(`Herbie(model="ifs", product="enfo").xarray(":{var}:{level}:pl:")`, 50
perturbed members), reindexed onto the GIK grid. The script then computes
Pearson r / RMSE / MAE / max|diff| for ensemble mean and spread and renders a
GIK │ Herbie │ Difference map (mean on top, spread below).

```bash
# from grib-index-kerchunk/ecmwf
uv run compare_gik_herbie_pressure.py \
    --par-dir gik_vs_herbie/pl_eval/par_local \
    --date 20260301 --step 48 --var u --levels 300,500,700,850 \
    --output-dir gik_vs_herbie/pl_eval/out

# the full u/v/gh sweep used for this report:
bash gik_vs_herbie/pl_eval/_run_full_eval.sh
```

`compare_gik_herbie_pressure.py` can also read par straight from GCS
(`--gcs-path gs://…/2026/03/20260301/00z`) and run GIK-only (`--no-herbie`).

---

## 3. Result (2026-03-01 00Z, T+48h)

Ensemble-mean Pearson correlation, GIK (51 members) vs Herbie (50):

| Variable | 300 hPa | 500 hPa | 700 hPa | 850 hPa |
|----------|---------|---------|---------|---------|
| **u** | 0.999999 | 0.999999 | 0.999997 | 0.999993 |
| **v** | 0.999991 | 0.999996 | 0.999997 | 0.999995 |
| **gh** | 1.000000 | 1.000000 | 0.999999 | 0.999999 |

Max absolute difference across all 12 fields: **< 0.18 m/s** (winds),
**< 0.15 gpm** (height) — rounding-level noise. The entire residual is the
51-vs-50 member offset (GIK includes the control member; Herbie's `enfo` does
not), identical to the surface-`tp` finding. **`gh` — which previously
collapsed to the surface — now matches Herbie to r = 1.000000.**

Per-variable numbers: `out/pl_comparison_stats_{u,v,gh}_20260301_T48h.json`.

---

## 4. Plots

All 12 maps are retained on HuggingFace (dataset `E4DRR/gik-ecmwf-par`,
folder **`validation_gik_vs_herbie_pl_correction/`**):

<https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/tree/main/validation_gik_vs_herbie_pl_correction>

Each figure: **GIK (left) │ Herbie (center) │ Difference (right)**, with
ensemble **mean** on the top row and ensemble **spread** on the bottom row,
over the East Africa domain.

| Field | Plot |
|-------|------|
| U-wind @ 700 hPa | ![u700](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/validation_gik_vs_herbie_pl_correction/compare_pl_u700_20260301_T48h.png) |
| V-wind @ 850 hPa | ![v850](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/validation_gik_vs_herbie_pl_correction/compare_pl_v850_20260301_T48h.png) |
| Geopotential @ 500 hPa | ![gh500](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/validation_gik_vs_herbie_pl_correction/compare_pl_gh500_20260301_T48h.png) |

In every map the GIK and Herbie panels are visually indistinguishable and the
difference panel shows only noise — confirming the per-level keys resolve to
the correct pressure level for each level, step, and member.

---

## 5. Files in this folder

| File | Purpose |
|------|---------|
| `README.md` | this document |
| `_download_par.py` | download the 51 re-baked par files from GCS → `par_local/` |
| `_run_full_eval.sh` | run the full u/v/gh × 4-level sweep |
| `out/compare_pl_*.png` | 12 GIK│Herbie│Diff maps (also on HuggingFace) |
| `out/pl_comparison_stats_*.json` | per-variable correlation / RMSE / max-diff |
| `par_local/` | downloaded par files (git-ignored; re-fetch with `_download_par.py`) |

The realization + plotting routine itself lives one level up:
[`../../compare_gik_herbie_pressure.py`](../../compare_gik_herbie_pressure.py).

To upload the plots to HuggingFace:
[`../../upload_pl_validation_to_hf.py`](../../upload_pl_validation_to_hf.py).
