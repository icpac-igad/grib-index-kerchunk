# Per-era Cloud Run deploy prep — push the per-level fix into operations

**Date:** 2026-06-03
**Scope:** wiring `run_lithops_ecmwf.py` + Dockerfile to deploy one Cloud Run
image per ECMWF template era, against the rebuilt per-level HF templates, so
the per-level fix reaches operational parquets (the cGAN consumer).

## Runtime readiness (what's already done in this branch)

- **Per-level keys** — `create_references_from_index` emits
  `{var}/pl/{hPa}/{member}/0.0.0` for pl (the `59b89dd` fix). ✅
- **Resolution-aware source path** — `ECMWF_RESOLUTION` (`0p25`|`0p4`) →
  `ifs/0p25` vs `0p4-beta`. ✅
- **Dual-stream control (NEW)** — `ECMWF_CONTROL_STREAM` (`enfo`|`oper`):
  perturbed always from `enfo/ef`; control from this stream. 50r1 set
  `oper`; 49r1/0p4 keep `enfo` (control bundled as `number=0`). Without this
  a 50r1 deploy produced parquets **missing the control member**. ✅ (fixed)
- **Dockerfile** — template now `--build-arg`-selected from
  `E4DRR/grib-index-kerchunk-templates` (was hardcoded to the legacy buggy
  tar in the wrong HF repo); bakes the matching era env. ✅

## Per-era build + deploy matrix

| Era | TEMPLATE_ARTIFACT | ECMWF_REFERENCE_DATE | ECMWF_RESOLUTION | ECMWF_CONTROL_STREAM | Covers dates |
|---|---|---|---|---|---|
| **49r1 13-level** | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` | `20250515` | `0p25` | `enfo` | 2025-01-14 06z → 2026-05-12 00z |
| **50r1 14-level** | `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` (default) | `20260513` | `0p25` | `oper` | 2026-05-12 06z → present |
| **0.4-beta 9-level** | `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` | `20230601` | `0p4` | `enfo` | 2023-01-18 → 2024-02-28 |
| (49r1 9-level) | — covered by the 13-level template (superset) | (use 49r1 image) | `0p25` | `enfo` | 2024-02-29 → 2025-01-14 00z |

Build one image per era (example, 49r1):
```bash
docker build -f Dockerfile \
  --build-arg TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz \
  --build-arg ECMWF_REFERENCE_DATE=20250515 \
  --build-arg ECMWF_RESOLUTION=0p25 \
  --build-arg ECMWF_CONTROL_STREAM=enfo \
  -t <registry>/gik-cr-ecmwf-49r1:YYYYMMDD .
# (50r1 = defaults; 0p4 = TEMPLATE_ARTIFACT=...-0p4-beta.tar.gz, REFERENCE_DATE=20230601, RESOLUTION=0p4)
```
Then `lithops runtime deploy` / Cloud Run revision per `SETUP_NEW_MACHINE.md §5`.
Env can also be overridden at deploy without rebuilding (they're plain
Cloud Run env vars), but baking keeps the image self-consistent.

## Re-bake order (cGAN first)

**MAM 2026 (Mar 1 – May 11 2026) is in the 49r1 13-level era**, not 50r1
(50r1 starts 2026-05-12 06z). So the **49r1 13-level image unblocks both
MAM 2025 and MAM 2026** for the cGAN port — build/deploy it first and
re-bake those windows, then 50r1 for ongoing ops, 0p4 only if a consumer
needs the pre-2024 archive.

1. Deploy **49r1 13-level** → re-bake MAM 2026 (2026-03-01 → 2026-05-12 00z),
   then MAM 2025, then the rest of 2025-01-14 06z → 2026-05-12.
2. Deploy **50r1** → ongoing 2026-05-12 06z → present.
3. (optional) **0p4** → pre-2024 archive.

Each re-bake = `run_lithops_ecmwf.py` over the era's dates (cleanup-hang
watchdog), then re-mirror parquets to HF `E4DRR/gik-ecmwf-par`.

## Validation before bulk re-bake (cheap)

For one date in the target era, run a single date and assert the output
parquet has per-level pl keys `step_NNN/{var}/pl/{hPa}/{member}/0.0.0`
across the full era level set, and that **control is present** (esp. 50r1,
which exercises the new `oper` path). Then bulk-bake.

## Not done here (needs the user / infra)

- Building + pushing the 3 Cloud Run images (Docker + registry) and the
  `lithops runtime deploy` (paid, your GCP). The runtime + Dockerfile are
  ready; this is an infra action.
- The bulk re-bake itself (paid Cloud Run) + HF re-mirror (HF token).
