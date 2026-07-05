# Full par → Icechunk conversion: time estimate & the 1-hour STS story

Measured on this 2-core VM (icechunk 2.1 / zarr 3.2 / gribberish 1.4), 2026-07-05.
Corpus from the HF catalog (`E4DRR/gik-ecmwf-par-v2/catalog.parquet`): **1,256
dates, 20.6 GB of pars** across four schema regimes.

## Corpus (from the catalog, not assumed)

| era (store) | grid | pl levels | dates | refs/date | avg par MB/date |
|---|---|---|---|---|---|
| 0p4 | 451×900 | 9 | 401 | 359,805 | 10.8 |
| 49r1 (9L) | 721×1440 | 9 | 321 | ~503k | 15.9 |
| 49r1 (13L) | 721×1440 | 13 | 483 | 654,585 | 21.5 |
| 50r1 | 721×1440 | 14 | 51 | ~700k | 23.4 |
| **total** | | | **1,256** | | **20.6 GB** |

Plan: **one store per era** (the 49r1 store built on the 13-level superset, so
its 321 nine-level dates simply leave the 4 extra levels empty — same decision
as the templates). 0p4 and 50r1 can't share (different grid / control stream).

## Per-date cost (measured, single-threaded)

| phase | 0p4 (9L) | 49r1 (13L) | notes |
|---|---|---|---|
| HF download (51 pars, ‖8) | 2.5 s | 2.5 s | ~21 MB, this VM's bandwidth |
| parse 51 pars | ~2 s | 6.9 s | pandas, scales with refs |
| `set_virtual_refs` | ~5 s | 6.5 s | scales with refs |
| commit | ~1 s | 1.9 s | with manifest splitting |
| **per date** | **~11 s** | **~18 s** | download + build |

Peak coordinator RSS 468 MB (0p4) / 723 MB (13L) — comfortable; nothing
accumulates across dates.

## Full build time (local, credential-free)

| era | dates | s/date | build time |
|---|---|---|---|
| 0p4 | 401 | 11 | ~74 min |
| 49r1 (9L) | 321 | 14 | ~75 min |
| 49r1 (13L) | 483 | 18 | ~145 min |
| 50r1 | 51 | 18 | ~16 min |
| **all four** | **1,256** | | **≈ 5.2 hours** |

Single 2-core VM, one date at a time. Trivially shortened by running the four
era stores in parallel (they're independent) or by download-ahead; the wall is
CPU-bound parse+set, so more cores ≈ linear speedup down to ~1–1.5 h.

## Store size — why manifest splitting is mandatory

Without splitting, each append rewrites one ever-growing manifest → store size
is **O(n²)** in dates (measured: +12, +18, +24 MB for dates 2, 3, 4; 401 dates
would be ~0.5 TB). `build_0p4_ensemble_icechunk.py` now sets
`ManifestSplittingConfig` along `time` (1 date/shard), giving **linear ~7 MB/date**
and flat ~8.5 s appends. Projected published sizes:

| store | dates | ~MB/date | store size |
|---|---|---|---|
| 0p4 | 401 | 7 | ~2.8 GB |
| 49r1 (9L+13L) | 804 | ~12 | ~9.5 GB |
| 50r1 | 51 | ~14 | ~0.7 GB |
| **total published** | | | **~13 GB** |

## The 1-hour STS window — how it's handled

The heavy work is **not** credential-bound. Build runs against anonymous ECMWF
S3 and writes a **local** store, so the ~5 h build has zero STS exposure — token
expiry cannot interrupt it. (It bit us live once here: an expired token failed a
publish while every build had already succeeded locally — exactly this split.)

Only the **publish** (upload the local store to source.coop) needs the 1-hour
STS token, and `publish_to_source_coop.py` is **resumable**: it skips
already-uploaded objects, so if the token expires mid-upload you `source .env`
with fresh creds and re-run — it continues from where it stopped. Three ways to
stay inside the window:

1. **Per-era publish.** Build an era locally, then publish it (0.7–9.5 GB). At
   the proxy's ~4 MB/s that's up to ~40 min for the big 49r1 store — under an
   hour, and resumable if not.
2. **Incremental sync during the build.** Because each date is one sealed
   manifest shard (~7–14 MB), re-running the publisher after every N dates
   uploads only the new shards — a few seconds each, always far inside the
   window. This keeps source.coop continuously current with no large final push.
3. **Refresh-and-resume.** Worst case, let it run; on `ExpiredToken` refresh
   `.env` and re-run the same command.

Recommended: option 2 — a cron/loop that builds a batch of dates then calls the
publisher — so the STS window is never a bottleneck.

## Direct-commit is still not an option

Icechunk commits cannot go through `data.source.coop` (missing S3 ops) and the
proxy creds are not valid on the direct AWS bucket — see the "Publishing to
source.coop" section of the README. Build-local + file-sync is the only working
path, and it's also what makes the STS window a non-issue for the expensive
part.

## Runbook per era

```bash
# 1. build locally (credential-free; ~1–2.5 h/era)
for D in <era dates>; do
  mkdir -p pars/$D && <download 51 pars for $D>
  uv run build_0p4_ensemble_icechunk.py --pars-dir pars/$D --date $D \
      --store /tmp/ecmwf-<era>-real
done

# 2. publish (needs fresh 1-hour STS creds; resumable)
source .env
uv run publish_to_source_coop.py \
    --local /tmp/ecmwf-<era>-real \
    --dest  forecasts/ecmwf_ifs_<era>_icechunk
```

Real (non-test) target paths on source.coop:
`e4drr-project/forecasts/ecmwf_ifs_{0p4,49r1,50r1}_icechunk`.
