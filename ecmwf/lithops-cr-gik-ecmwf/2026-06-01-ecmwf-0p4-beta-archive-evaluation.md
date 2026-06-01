# Evaluation: GIK for the pre-2024-03 ECMWF 0.4° (`0p4-beta`) archive

**Date:** 2026-06-01
**Branch:** `ecmwf-50r1-template`
**Question:** can GIK cover the period before the 0.25° era (resolution
0.4°), which on AWS S3 goes back to early 2023? Earlier docs declared the
"28r1 / pre-2024-03" era out of scope
([`2026-05-29-49r1-perlevel-reprocess-plan.md`](2026-05-29-49r1-perlevel-reprocess-plan.md) §1/§7).
This is the empirical evaluation of that era.

## Verdict

**Yes — GIK applies to the 0.4° archive with only two changes**, both
already implemented and validated:

1. **Path prefix.** The 0.4° era lives at `{date}/{run}z/0p4-beta/{stream}/…`
   (no `ifs` segment), vs `{date}/{run}z/ifs/0p25/{stream}/…` for 0.25°.
   Only the Coiled scan driver's URL builder needs this prefix.
2. **Grid shape.** 0.4° fields are **451 × 900**, not 721 × 1440. The
   realigner is now resolution-aware (`field_shape_for_uri`,
   `ECMWF_0P4_FIELD_SHAPE = [1, 451, 900]`) and picks the shape from the
   resolution token in the GRIB URI — which the real scan dump carries in
   its `uri` column.

Everything else GIK relies on is identical: `.index` JSON-lines sidecars,
`enfo`/`oper` streams, `-{h}h-enfo-ef.grib2` naming, `number`-based member
encoding, `levtype`/`levelist` fields.

## Empirical findings (anonymous S3, 2026-06-01)

| Property | 0.4° `0p4-beta` |
|---|---|
| Archive start | **2023-01-18** (2023-01-17 and earlier: absent) |
| Era end | **2024-02-28**; `ifs/0p25` begins **2024-02-29** (pinned) |
| Path | `s3://ecmwf-forecasts/{date}/{run}z/0p4-beta/{enfo,oper,waef,wave}/` |
| Resolution / grid | 0.4° → **latitude 451 × longitude 900** (confirmed via scan_grib coord arrays) |
| ENS file naming | `{ts}-{h}h-enfo-ef.grib2` + `.index` (~1.0 MB), GRIB ~2.6 GB |
| Members | **51** (control bundled in `enfo` as `number`→-1, perturbed 1..50); `oper/fc` carries the control separately too |
| Pressure levels | **9**, homogeneous across the whole era: `[50,200,250,300,500,700,850,925,1000]` (no 100/150/400/600, no 10) |
| Forecast hours | **85**: 0–144 every 3 h, 150–360 every 6 h (same cadence as 0.25°) |
| levtypes / params | `sfc` + `pl`; ~18 params; ~4 182–4 233 msgs/file |
| `.index` present | Yes — GIK byte-range method works unchanged |

The 0.4° era is **homogeneous** in member count and level set throughout
(checked 2023-01-18, 2023-06, 2023-11, 2024-01, 2024-02-28) — unlike 49r1,
which split 9→13 levels mid-era. So a single 0.4° template reference date
is sufficient for the whole `0p4-beta` period.

## Proof: the realigner produces a correct 0.4° template skeleton

`ecmwf/dev-test/gate_step2_realigner.py --res 0p4` runs the real patched
realigner against the live 0.4° `.index`:

```
python gate_step2_realigner.py --date 20230201 --run 00 --hour 12 --stream enfo --res 0p4
→ GATE PASS: 8/8
  - 51 members incl. control
  - per-level pl keys u/isobaricInhPa/{50..1000}, full 9-level coverage
  - .zarray shape [1, 451, 900] (0.4°, not the 1° stub, not 0.25°)
  - no replicate-to-all (control groups=82 from its own 82 messages)
```

The 0.25° gates (49r1 13-level, 50r1 14-level enfo+oper) still pass 8/8 —
the resolution-awareness is additive.

## The complete four-era picture (for template planning)

| Era | Window | Path | Grid | Members | pl levels |
|---|---|---|---|---|---|
| **0.4° beta** | 2023-01-18 → 2024-02-28 | `…/0p4-beta/` | 451×900 | 51 (ctrl bundled) | **9** |
| 49r1 0.25° (9-lvl) | 2024-02-29 → 2025-01-14 00z | `…/ifs/0p25/` | 721×1440 | 51 (ctrl bundled) | 9 |
| 49r1 0.25° (13-lvl) | 2025-01-14 06z → 2026-05-12 00z | `…/ifs/0p25/` | 721×1440 | 51 (ctrl bundled) | 13 |
| 50r1 0.25° (14-lvl) | 2026-05-12 06z → present | enfo(50)+oper(ctrl) | 721×1440 | 50 + 1 dual-stream | 14 |

So the project actually spans **four** schema regimes, not the three the
user sketched — the 0.4° era plus the 49r1 sub-split found on 2026-06-01.
Each distinct (grid, members, levels, stream-layout) needs its own
template; within an era a single reference date suffices.

Note: 0.4° and the early 0.25° era share the **same 9-level pl set and
same 51-member single-stream layout** — they differ ONLY in grid
resolution. They cannot share a template (different `.zarray` shape), but
they share the entire pipeline path (single-stream, control bundled,
9-level), so a 0.4° rebuild is the cheapest of all four to add.

## What remains to actually build a 0.4° template

1. **A 0.4° Coiled scan driver** — a one-line-prefix variant of
   `ecmwf_49r1_coiled_preprocessing.py` (single-stream, 85 tasks), with
   the URL built as `…/0p4-beta/enfo/…` instead of `…/ifs/0p25/enfo/…`.
   ~5-line delta. (Not written yet.)
2. **Reference date** — any 0.4° date works (era is homogeneous); pick a
   clean mid-era 00z, e.g. `20230601`.
3. **Realigner** — no change needed; it is already resolution-aware and
   the GATE passes on real 0.4° data.
4. **Runtime (`run_lithops_ecmwf.py`)** — Stage 2 `.index` URL builder
   needs a `0p4-beta` path branch, and `REFERENCE_DATE` / template name
   set to the 0.4° artifact, same dual-image cutover strategy as 49r1/50r1
   (separate Cloud Run image per era, swapped by revision at the
   2024-02-29 boundary).
5. **Cost** — identical shape to the 49r1 single-stream preprocessing
   (~1 h Coiled, 85 tasks), since 0.4° files are smaller (~2.6 GB) the
   scan is if anything slightly faster.

## Whether it is worth doing

Driver decision, not technical. The 0.4° era is 2023-01-18 → 2024-02-28
(~13 months). If cGAN / flood-risk training wants pre-2024 ground (e.g.
MAM 2023), this era is the only ECMWF ENS source and GIK can serve it for
the same low per-date cost as the others. If training stays 2024+, it can
remain unbuilt with no loss. Either way the **method is proven** — the
blocker that made it "out of scope" (unknown structure) is removed.
