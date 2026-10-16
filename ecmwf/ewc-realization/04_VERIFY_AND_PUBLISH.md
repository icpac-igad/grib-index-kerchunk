# Verify and publish

Two stores get published to source.coop:

- the **virtual** store, mirror ②, from GCS;
- the **realized** store, mirror ③, from EWC Ceph.

The lifecycle is **build → verify → mirror → verify as an anonymous reader →
(human decision) delete from Ceph**. The exact commands and their failure
modes are in [`RUNBOOK_REALIZE.md`](RUNBOOK_REALIZE.md) §4–5. This page lists
the tools and the order to use them in.

## 1. Verify the realized store on Ceph (after the build exits)

Run the checks in order, cheapest first:

| # | check | command | pass |
|---|---|---|---|
| 1 | manifest completeness | `check_chunks.py --prefix ea-swio/<sink> --env .env` | **every channel the same count**, and `total missing = unwritten_dates × 39 × 65` exactly. Unequal counts mean dates torn across channels, which is a silent partial commit |
| 2 | chunk writes | `check_complete.py --prefix ea-swio/<sink> --env .env` | no missing chunks. It reads blocks, so it is slow: on a 33-date store it produced nothing in 30 min. For a whole season use the manifest (row 1, or `Session.chunk_coordinates`, ~3 min) |
| 3 | members are real | `check_members.py --prefix ea-swio/<sink> --env .env --var gh500` | spread grows with lead time. **Always pass `--var gh500`**: the default variable has a flat tropical spread and looks like a failure |
| 4 | geolocation | `check_geolocation.py --prefix ea-swio/<sink> --env .env` | `CONSISTENT`. This check exists because of an earlier longitude bug ([`figures/globe_longitude_bug.png`](figures/globe_longitude_bug.png)) |
| 5 | spot read | `inspect_store.py --prefix ea-swio/<sink> --var tp --step 0`, then `--step 240` (hours) | finite at both ends of the step axis. A date can be whole at step 0 (index 0) and empty at 240 h (index 64) |

**The build's success line is not evidence. Only the manifest is.** A read
of an unwritten chunk returns the fill value without error.

Repair with `build_corpus.py --repair`, and **give it the store's whole
window** (`--start <season start> --days <season length>`). It indexes by
position, so a narrower window inspects the wrong slots and reports "store is
complete".

## 2. Publish

| mirror | command | rate measured | notes |
|---|---|---|---|
| ② virtual, GCS → source.coop | `mirror_gcs_to_source_coop.py --gcs-store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 --dest forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd --threads 24 [--live-only] [--prune]` | upload 7–18 s/day; **listing ~425 k objects dominates (9–10 min)** | prints `MIRROR COMPLETE` |
| ③ realized, Ceph → source.coop | `mirror_ceph_to_source_coop.py --src-prefix ea-swio/<sink> --dest forecasts/ecmwf-ifs-ea-swio-realized-v3/<era>-<season> --threads 48` | **300–437 GB/h** from one host | silent until done. The `repo` pointer goes **last**, so the store can't be opened before it lands |
| ③, many seasons | `mirror_all_to_source_coop.sh` | ~430 GB/h | the pattern is in [`MIRROR_RUN_ARCHITECTURE.md`](MIRROR_RUN_ARCHITECTURE.md) |

Both mirrors need the source.coop STS in the environment
(`set -a; . ./.env; set +a`), and mirror ③ also needs
`AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` for Ceph. Keep these
shells separate from the build shell (see the `AWS_*` trap,
[`01_SETUP_FROM_SCRATCH.md`](01_SETUP_FROM_SCRATCH.md) §3).

## 3. Verify as an external reader

Use no credentials:

```python
st = icechunk.s3_storage(bucket="e4drr-project", prefix="forecasts/ecmwf-ifs-ea-swio-realized-v3/<sub>",
                         endpoint_url="https://data.source.coop", region="us-east-1",
                         anonymous=True, from_env=False, force_path_style=True)
```

- Check the dates, vars, steps and members, and read a field at step 0 and
  step 64. Include every date you repaired.
- For the virtual store, run `smoke_test_published_ecmwf.py` and
  `../icechunk-par/verify_store_completeness.py`.
- For users, see [`OPENING_PUBLISHED_ECMWF_ICECHUNK.md`](OPENING_PUBLISHED_ECMWF_ICECHUNK.md)
  and [`USING_REALIZED_EA_SWIO_STORE.ipynb`](USING_REALIZED_EA_SWIO_STORE.ipynb).

## 4. Delete from Ceph only on evidence

Delete only after step 3 passes on the published copy, and record why.
`purge_wrong_stores.py` exists for stores built on the wrong region.

**Two independent checks, because one is not enough.** The time axis is
preallocated, so an unwritten date reads back as NaN, and a mirror faithfully
copies the hole. A matching listing therefore proves the copy, not the content.

1. **Copy check.** Do a full listing of both sides, excluding `overwritten/`:
   object count, bytes, snapshot count and HEAD snapshot id must be identical.
2. **Content check.** Season completeness from the manifest
   (`Session.chunk_coordinates`, via `../check_realized_coverage.py`, `bc00287`).
   Read through the **`us-west-2.opendata.source.coop`** alias, not the write
   endpoint.

**The deletion done, 2026-08-23:** four MAM stores, 4,897.97 GB.

| store | objects | GB | snapshots | HEAD |
|---|---|---|---|---|
| `v3-49r1-mam2024` | 243,982 | 1,651.75 | 291 | `7MB4YH8J09DSKS` |
| `v3-49r1-mam2025` | 235,487 | 1,592.77 | 187 | `9VVSNV9KKSPS9V` |
| `v3-49r1-mam2026` | 193,610 | 1,329.83 | 224 | `4QBREPA14W6D6V` |
| `v3-50r1-mam2026-tail` | 47,821 | 323.62 | 41 | `9X2Z1WP5KH33S9` |

- Both sides listed 720,900 objects, with a size difference of 0.000 GB.
- The content check caught `mam2024` at **33/92** dates. A delta re-mirror
  brought it to 92/92 before anything was deleted.
- `w700` and `cape` are 87/92 in mam2024 (2024-03-01…05). That is expected:
  49r1 has only 83 fields/member/step until 2024-03-06.
- The `purge_wrong_stores.py` dry-run reported 722,370 objects / 4,898.00 GB.
  The extra 1,470 objects are the `overwritten/` set.
