# ECMWF open-data variable & pressure-level inventory, per schema era

**Date:** 2026-06-02
**Purpose:** a reference of *which variables and pressure levels ECMWF
open-data publishes in each schema era* of the GIK ECMWF pipeline, so a
template/consumer can be scoped without re-scanning the archive. The raw
`.index` sidecars these tables were derived from are preserved on
HuggingFace (links below) rather than committed here (they are ~1–2 MB
each).

## Where the sample `.index` files live (HuggingFace, not git)

Dataset: **`E4DRR/gik-ecmwf-par`** → folder **`sample_index/`**
Browse: <https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/tree/main/sample_index>

One set per era, `0h` + `3h` (3h carries the accumulation/step-type
variants that 0h lacks). Resolve-URL base:
`https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/`

| Era | 0h `.index` | 3h `.index` |
|---|---|---|
| 0p4-beta | [`0p4-beta/20230601000000-0h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/0p4-beta/20230601000000-0h-enfo-ef.index) | [`…-3h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/0p4-beta/20230601000000-3h-enfo-ef.index) |
| 49r1-9level | [`49r1-9level/20240529000000-0h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/49r1-9level/20240529000000-0h-enfo-ef.index) | [`…-3h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/49r1-9level/20240529000000-3h-enfo-ef.index) |
| 49r1-13level | [`49r1-13level/20250515000000-0h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/49r1-13level/20250515000000-0h-enfo-ef.index) | [`…-3h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/49r1-13level/20250515000000-3h-enfo-ef.index) |
| 50r1-enfo | [`50r1-enfo/20260513000000-0h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/50r1-enfo/20260513000000-0h-enfo-ef.index) | [`…-3h-enfo-ef.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/50r1-enfo/20260513000000-3h-enfo-ef.index) |
| 50r1-oper | [`50r1-oper/20260513000000-0h-oper-fc.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/50r1-oper/20260513000000-0h-oper-fc.index) | [`…-3h-oper-fc.index`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/sample_index/50r1-oper/20260513000000-3h-oper-fc.index) |

## Era summary

| Era | Date window | Path prefix | Grid | Members | sfc | sol | pl vars | pl levels | total params |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **0p4-beta** | 2023-01-18 → 2024-02-28 | `0p4-beta/` | 0.4° (451×900) | 51 (control bundled) | 11 | 0 | 8 | 9 | 19 |
| **49r1 9-level** | 2024-02-29 → 2025-01-14 00z | `ifs/0p25/` | 0.25° (721×1440) | 51 (control bundled) | 28 | 0 | 9 | 9 | 37 |
| **49r1 13-level** | 2025-01-14 06z → 2026-05-12 00z | `ifs/0p25/` | 0.25° (721×1440) | 51 (control bundled) | 34 | 2 | 9 | 13 | 45 |
| **50r1 enfo** | 2026-05-12 06z → present | `ifs/0p25/` enfo/ef | 0.25° (721×1440) | 50 (perturbed) | 38 | 2 | 9 | 14 | 49 |
| **50r1 oper** | 2026-05-12 06z → present | `ifs/0p25/` oper/fc | 0.25° (721×1440) | 1 (control) | 41 | 2 | 10 | 14 | 53 |

Within every era, **all pl variables share the same level set** (verified).
Inventory unioned over forecast hours {0, 3, 24, 150} per era to capture
step-0-only, 3-hourly accumulation, and 6-hourly-range variables.

`.index` line schema: `_offset`, `_length`, `param`, `number` (member;
absent ⇒ control), `levtype` (`pl`/`sfc`/`sol`), `levelist` (hPa, pl only),
`step`, `class`, `stream`, `type`, `date`, `time`.

---

## Per-era variables

### 0p4-beta (0.4°, 9 pl levels)
- **pl levels (9):** `50, 200, 250, 300, 500, 700, 850, 925, 1000`
- **pl vars (8):** `d, gh, q, r, t, u, v, vo`  ⚠ **no `w`** (vertical velocity)
- **sfc (11):** `10u, 10v, 2t, lsm, msl, ro, skt, sp, st, tcwv, tp`
- **sol:** none

### 49r1 9-level (0.25°, 9 pl levels)
- **pl levels (9):** `50, 200, 250, 300, 500, 700, 850, 925, 1000`
- **pl vars (9):** `d, gh, q, r, t, u, v, vo, w`
- **sfc (28):** `100u, 100v, 10u, 10v, 2d, 2t, asn, cape, lsm, msl, ro, skt, sp, ssr, ssrd, st, stl2, stl3, stl4, str, strd, swvl1, swvl2, swvl3, swvl4, tcwv, tp, ttr`
- **sol:** none

### 49r1 13-level (0.25°, 13 pl levels) — cGAN MAM 2025/2026 window
- **pl levels (13):** `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`
- **pl vars (9):** `d, gh, q, r, t, u, v, vo, w`
- **sfc (34):** `100u, 100v, 10fg, 10u, 10v, 2d, 2t, asn, ewss, lsm, mn2t3, mn2t6, msl, mucape, mx2t3, mx2t6, nsss, ptype, ro, sithick, skt, sp, ssr, ssrd, str, strd, sve, svn, tcw, tcwv, tp, tprate, ttr, zos`
- **sol (2):** `sot, vsw`

### 50r1 enfo (0.25°, 14 pl levels, 50 perturbed members)
- **pl levels (14):** `10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`
- **pl vars (9):** `d, gh, q, r, t, u, v, vo, w`
- **sfc (38):** `100u, 100v, 10fg, 10u, 10v, 2d, 2t, asn, ewss, lsm, mn2t3, mn2t6, msl, mucape, mx2t3, mx2t6, nsss, ptype, ro, rsn, sd, sf, sithick, skt, sp, ssr, ssrd, str, strd, sve, svn, tcc, tcw, tcwv, tp, tprate, ttr, zos`
- **sol (2):** `sot, vsw`

### 50r1 oper (0.25°, 14 pl levels, control — superset)
- **pl levels (14):** `10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`
- **pl vars (10):** `d, gh, q, r, t, u, v, vo, w, z` ← extra **`z`** (geopotential on pl)
- **sfc (41):** the 50r1-enfo set **plus** static fields `sdor, slor, z`
- **sol (2):** `sot, vsw`

---

## What changes across eras (the schema-break deltas)

- **0.4° → 0.25° (2024-02-29):** grid `451×900 → 721×1440`. The 0.4° set is
  also the leanest — only 11 sfc vars and **no `w`** on pl.
- **9-level → 13-level (2025-01-14 06z):** pl set gains `100, 150, 400, 600`
  hPa; **soil vars `sot, vsw` appear**; sfc set broadens (adds `10fg, ewss,
  mn2t*/mx2t*, mucape, nsss, ptype, sithick, tcw, tprate, sve, svn, zos`,
  drops `cape, stl*, swvl*` in favour of the `sot/vsw` soil encoding).
- **49r1 → 50r1 (2026-05-12 06z):** pl set gains **`10` hPa** (→14 levels);
  **control moves out of `enfo/ef` into `oper/fc`** (enfo becomes 50
  perturbed only); sfc adds `rsn, sd, sf, tcc`.
- **enfo vs oper within 50r1:** the control (`oper/fc`) is a strict superset
  — extra static fields `sdor, slor, z` on sfc and `z` on pl. This is why
  the 50r1 template build scans both streams.

## Notes & caveats

- These are the variables **published in ECMWF open data** (`enfo/ef`,
  `oper/fc`), not the full IFS field set.
- `param` values are ECMWF shortNames (paramId lookup:
  <https://codes.ecmwf.int/grib/param-db/>).
- A variable's presence at a given step varies: instantaneous fields exist
  at all steps; accumulation/flux fields (`tp, ssr, ssrd, str, strd, ro,
  ewss, nsss, ttr`, the `mn/mx` extrema, `tprate`) start at step > 0.
- Reproduce / refresh: `ecmwf/dev-test/gate_step2_realigner.py` reads these
  same live `.index` files; the inventory was built by unioning `param` per
  `levtype` across hours {0,3,24,150}.
