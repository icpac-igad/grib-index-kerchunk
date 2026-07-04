# CFS Corpus: Duplicate-to-Zarr vs GIK-Reference — Cost Comparison

**Scope:** NOAA CFS `flxf` (6-hourly flux), **member 01**, all 4 cycles
(00/06/12/18z), **2018-10-31 → 2026-07-03**. Prepared 2026-07-04 from measured
values (see `CFS_MEMBER_TIMING_AND_PIPELINE.md`).

---

## 1. Dataset facts

| Property | Value |
|---|---|
| Grid | **T126 Gaussian, 384 × 190** (72,960 pts/field) |
| Resolution | **0.9375° ≈ ~104 km** at equator (constant 2018→2026, no grid change) |
| Cadence | 6-hourly |
| Forecast length (member 01) | ~793–857 steps (~198–214 days ≈ 7 months), calendar-horizon dependent |
| Variables | ~103 GRIB messages/file · ~68 distinct variables · 86 (var×stepType×level) groups (surface/flux 2-D only, **no pressure levels**) |
| flxf file size | ~4.15 MB |

**Why the whole thing is only ~38 TB** (vs ECMWF IFS at 0.25°/25 km reaching
petabyte scale for 2023→2026): CFS `flxf` is **~1° (14× fewer points/field than
0.25°) × 1 member × surface-only (no pressure levels)**. ECMWF is
14× resolution × 51 members × 13–14 pressure levels — those multipliers, not the
forecast length, are what push IFS into PB territory.

## 2. Referenced-corpus size

| | Value |
|---|---|
| flxf files referenced (member 01, ×4 cycles, full period) | **~9.22 million** |
| **Total GRIB data referenced** | **≈ 38 TB** |
| GIK parquet references | **29 GB** |
| **Leverage** | **~1,320×** — references are **0.076 %** of the data |
| Per member | 3.4 GB GRIB → 2.6 MB parquet |

## 3. The two approaches

- **A — Duplicate to Zarr:** decode the corpus once and store an analysis-ready
  Zarr (whole corpus, or just the ~10 SPI/SPEI/PET variables). Fast repeat
  reads; pay perpetual storage + a big one-time conversion.
- **B — GIK references + on-demand Dask:** keep only the 29 GB references; when
  an analysis runs, a Dask/Coiled cluster streams *only* the needed
  variables/steps/members via byte-range reads, materializes a small realized
  Zarr, runs the analysis, and **discards it** — the corpus is never duplicated.

**Physics caveat (matters for "East Africa only"):** a GRIB2 message is one
compressed blob for the *whole global field* — you cannot byte-subset within it.
A regional analysis still **fetches full global messages** for the variables/
steps it needs, then spatially subsets after decode. Savings come from fetching
only the needed **variables** (~10 of ~68), members, and steps — not from the
region.

**Streamed volume for the SPI/SPEI/PET set** (~10 vars × ~50 KB/msg):

| Analysis scope | Data streamed |
|---|---|
| One seasonal forecast (1 target month = 112 members × ~822 steps) | **~46 GB** |
| Full 8-year reanalysis (11,212 members) | **~4.6 TB** |

## 4. Cost comparison (GCS Standard $0.02/GB-month; Cloud Run/Coiled compute)

| | **A — Duplicate → Zarr** | **B — GIK refs + on-demand** |
|---|---|---|
| Stored permanently | 4 TB (target vars) – 38 TB (all vars) | **29 GB** |
| **Storage $/yr** | **$960 (4 TB) – $9,200 (38 TB)** | **~$7** |
| One-time build | stream+decode 4.6–38 TB → **$100–2,000** | already done (~$70 Lithops) |
| Per EA seasonal run (1 month) | ~free (already materialized) | stream ~46 GB → **~$1–5** |
| Per full-corpus reanalysis | ~free | stream ~4.6 TB → **~$50–150** |
| S3 reads (NOAA Open Data) | n/a | **free** |
| Analysis-ready corpus access | 38 TB | 38 TB (via 29 GB refs) |

## 5. Break-even & verdict

- **Storage alone**: B is **~140–1,300× cheaper** ($7/yr vs $960–9,200/yr).
- **Break-even for A2** (cheapest Zarr, target-vars-only, ~$960/yr) vs B's
  transient streaming: a full-corpus pass streams ~4.6 TB (~$50–150), so A2 only
  pays off if you run **> ~6–15 full-corpus passes per year**. Regional or
  operational-monthly workloads never reach that.
- **For East Africa seasonal monitoring** (regional, incremental, monthly — the
  ICPAC / E4DRR / SEWAA pattern): **B wins by ~100–1,000×**. You get
  analysis-ready access to 38 TB for **~$7/yr storage + a few $ per analysis**,
  instead of **$1k–9k/yr** to keep a Zarr duplicate you only read a slice of.

**Recommendation:** keep the 29 GB GIK references; stream-and-discard on a Dask
cluster per analysis. Duplicate to Zarr only for a specific, hot, repeatedly
re-scanned subset if one emerges — and even then, only that subset, not the
corpus.

### Sensitivity notes
- Numbers are member-01 `flxf` only. Adding members 02–04 (~0.58× steps each)
  roughly triples reference count/size (still tiny) and streamed volume.
- Adding CFS pressure-level (`pgbf`) fields would multiply the *referenced* GRIB
  size several-fold (more messages/levels), making B's advantage even larger.
- GCS Nearline/Coldline cut A's storage 2–5× but "analysis-ready" implies
  frequent reads → Standard is the fair comparison.
