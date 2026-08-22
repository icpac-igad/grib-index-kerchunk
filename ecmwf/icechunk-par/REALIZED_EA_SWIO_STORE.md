# Reading the realized EA-SWIO store, and checking it against Herbie

**https://source.coop/e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3**

Written 2026-08-22. The sibling of `v20260819-icechunk-store-source-coop.md`,
which documents the **virtual** ENS store. This one is **realized**: the values
were decoded once, cut to the EA-SWIO box and written as real chunks, so a read
needs nothing from `s3://ecmwf-forecasts` and no GRIB decoder at all.

Read it with **no credentials**.

---

## 1. What is in it

Four independent Icechunk repos under one prefix, one per season:

| sub-store | dates on axis | written | window |
|---|---|---|---|
| `49r1-mam2024` | 92 | **33** | 2024-03-01 … **2024-04-02** — see §4 |
| `49r1-mam2025` | 92 | 92 | 2025-03-01 … 2025-05-31 |
| `49r1-mam2026` | 73 | 73 | 2026-03-01 … 2026-05-12 |
| `50r1-mam2026-tail` | 19 | 19 | 2026-05-13 … 2026-05-31 |

Each carries, per date:

- **36 channels**, flattened — `t500 t700 t850`, `u200 u500 u700 u850 u925`,
  `v*`, `q500 q700 q850 q925`, `r700 r850`, `d200 d700 d850`,
  `vo500 vo700 vo850`, `gh500`, and surface `t2m u10 v10 msl sp skt tp ro tcwv lsm`.
  There is **no level dimension**: the level is part of the name.
- **51 members** (`number` 0–50, 0 = control), **65 steps** (0–240 h at 3 h).
- Domain **15–80°E, 40°S–40°N** — 321 × 261 at 0.25°. This is the *extended*
  EA-SWIO box, wider than the ICPAC box (19–55°E, 14°S–25°N) used elsewhere in
  this repo.

Three things differ from the virtual store and will bite if you assume otherwise:

| | realized (this) | virtual (`ecmwf_ifs_ens_aws_s3_icechunk_vd`) |
|---|---|---|
| dim order | `(time, step, number, lat, lon)` | `(time, number, step, …)` |
| `step` units | **seconds** (`0, 10800, …, 864000`) | hours (`0, 3, …, 360`) |
| level | in the channel name (`t850`) | `isobaricInhPa` dimension |
| chunk | `(1, 1, 51, 321, 261)` — one per (date, step) | one per (date, member, step) |

---

## 2. Opening it

```python
# /// script
# dependencies = ["icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1"]
# ///
import icechunk, xarray as xr

BUCKET = "us-west-2.opendata.source.coop"
BASE = "e4drr-project/forecasts/ecmwf-ifs-ea-swio-realized-v3"

storage = icechunk.s3_storage(
    bucket=BUCKET, prefix=f"{BASE}/49r1-mam2025",
    region="us-west-2", anonymous=True, from_env=False,
)

# source.coop returns sporadic HTTP 500s on a few percent of GETs. The default
# open prefetches thousands of manifests in parallel, so one of them tends to
# draw a 500 and the whole open raises. Disabling the preload makes the open
# reliable; chunks are still fetched normally on demand.
cfg = icechunk.RepositoryConfig.default()
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))

repo = icechunk.Repository.open(storage, config=cfg)
ds = xr.open_zarr(repo.readonly_session("main").store,
                  zarr_format=3, consolidated=False, chunks=None)
```

No `gribberish` import, no `authorize_virtual_chunk_access`, no AWS credentials
— all three are needed for the *virtual* store and none of them here.

### Reading a field

Select by **label** on every axis, never by integer index. An integer index
reads the same bytes whatever the coordinate says; that is how a 180° longitude
error survived every check in an earlier store while silently returning the
eastern Pacific for an East Africa subset (`HANDOVER_LONGITUDE_FIX.md`).

```python
import numpy as np

da = (ds["t850"]
      .sel(time=np.datetime64("2025-04-20T00:00:00", "ns"))
      .sel(step=np.timedelta64(48 * 3600, "s"))     # SECONDS, not hours
      .sel(latitude=slice(25, -14), longitude=slice(19, 55)))   # N->S, W->E
mean = da.mean("number")
```

`latitude` descends (40 → −40), so its slice reads `slice(north, south)`.

---

## 3. Comparing against Herbie

`ecmwf/compare_realized_herbie.py` does this; the method is worth stating
because the obvious version of it does not test very much.

**Compare per member.** Store `number=n` against Herbie `number=n`,
element-wise, for n = 1…50. Both sides are reading the *same GRIB messages*, so
the only permitted difference is float32 packing noise — and in practice the
answer is exactly zero.

**Do not settle for ensemble mean and spread.** They are invariant under a
permutation of the member axis, so a store that filed member 7's field under
member 12 passes an ensemble-level check and fails a per-member one. Ensemble
statistics are still worth reporting, but as context, not as the test: the store
has 51 members and Herbie's `enfo` returns only the 50 perturbed, so `r < 1`
there is the missing control member rather than a defect.

**Pick channels that cannot hide a level mix-up.** `vo850`, `gh500` and `q925`
would be grossly wrong if the `(var, level) -> name` flattening mis-assigned a
level. `t500` vs `t850` differ by ~30 K and would also show, but vorticity
differs by orders of magnitude.

### Which script does what

| script | role |
|---|---|
| `ecmwf/compare_realized_herbie.py` | the comparison — one (sub-store, date, step, channel) per run |
| `ecmwf/run_realized_ea_swio_herbie_eval.sh` | driver for the whole published set: 16 realized cases + 2 compensation |
| `ecmwf/compare_icechunk_herbie.py` | supplies `herbie_field`, `field_stats`, `per_member_stats`; also runs the compensation cases against the virtual store |
| `ecmwf/check_realized_coverage.py` | which dates the store actually holds — run this first (§4) |

The Herbie fetch is defined **once**, in `compare_icechunk_herbie.py`, and
`compare_realized_herbie.py` imports it rather than re-deriving it. A
domain-dependent constant copied into a second file is exactly how the longitude
bug happened.

### The exact commands

`uv` is not installed on the EWC gateway, and `herbie` is not in the shared
conda env (`/opt/mamba/envs/dask` — **do not install into it**, the frisky
workers' `.venv` is a symlink to it). Everything in
`../gik_vs_herbie/realized_ea_swio_eval/` was produced with a throwaway venv
that inherits that env:

```bash
python3 -m venv --system-site-packages /tmp/gik-herbie-venv
/tmp/gik-herbie-venv/bin/pip install herbie-data
export PY=/tmp/gik-herbie-venv/bin/python     # or PY="uv run" where uv exists
```

Then, verbatim, from the `ecmwf/` directory:

```bash
# 1. coverage first -- exit 1 means the sub-store is short (§4)
$PY check_realized_coverage.py --verify-data \
    --json gik_vs_herbie/realized_ea_swio_eval/coverage.json

# 2. what dates each sub-store holds
$PY compare_realized_herbie.py --list

# 3. one case
$PY compare_realized_herbie.py --sub 49r1-mam2025 --date 20250420 \
    --step 240 --var tp --output-dir gik_vs_herbie/realized_ea_swio_eval

# 4. the whole published set (16 realized + 2 compensation)
PY=$PY ./run_realized_ea_swio_herbie_eval.sh

# 5. a date the realized store never wrote, from the virtual store (§4)
$PY compare_icechunk_herbie.py \
    --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
    --era 49r1 --run 00 --date 20240501 --step 0 --var u --levels 700 \
    --lon-min 15 --lon-max 80 --lat-min -40 --lat-max 40 --tag _easwio \
    --sa-key /tmp/frisky-ea/gcs-key.json \
    --output-dir gik_vs_herbie/realized_ea_swio_eval
```

Step 4 takes a couple of hours — one Herbie download of 50 GRIB messages per
case. Run it detached and follow the log:

```bash
PY=/tmp/gik-herbie-venv/bin/python setsid nohup ./run_realized_ea_swio_herbie_eval.sh \
    > gik_vs_herbie/realized_ea_swio_eval/sweep.log 2>&1 < /dev/null &
```

`setsid` is not optional — a plain `nohup ... &` from a short-lived shell gets
reaped when the parent exits. To check whether it is still alive use the log's
mtime, **not** `pgrep -f`: the pattern matches the checking command's own
command line and will report a long-dead run as running. That misreads a
finished sweep as in-progress, and it happened here.

Herbie is given the GRIB name, which for a pl channel is base + level:
`t850 -> ":t:850:pl:"`, and for surface the renamed ones map back
(`t2m -> ":2t:sfc:"`, `u10 -> ":10u:"`, `v10 -> ":10v:"`).

Exit status: `0` pass, `1` outside tolerance, `3` the date is on the axis but
was never written (§4).

### What the answer should look like

18 comparisons across all four sub-stores, both eras, pressure-level / surface /
accumulated channels, T+0h to T+240h — per-member `max|diff| = 0` on every one.
Plots and per-member statistics:
`ecmwf/gik_vs_herbie/realized_ea_swio_eval/`, published at
[huggingface.co/datasets/E4DRR/gik-ecmwf-par](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/tree/main/herbie-vs-realized-ea-swio-v3).

### Two traps in the comparison itself, both silent

- **Never `.sel(slice(...))` then `.reindex(method="nearest")`.** eccodes reports
  the 0.4° axis edge as `-14.000000000000057`, so `slice(25, -14)` drops that row
  and the reindex back-fills it from `-13.6`. One duplicated row at the boundary
  read as a real 0.56 K disagreement for a year. Select the target labels
  directly with a tolerance — `sel(latitude=lats, method="nearest",
  tolerance=0.01)` — so an absent label raises instead.
- **Cartopy suppresses the axes title when `draw_labels=True`.** Only on the
  labelled panel, and `matplotlib` still reports the title present and visible.
  Use a text artist.

---

## 4. The `49r1-mam2024` coverage gap

**`49r1-mam2024` holds 33 of its 92 dates: 2024-03-01 to 2024-04-02. The other
59 dates, 2024-04-03 to 2024-05-31, were never written.**

### Why it is dangerous rather than merely incomplete

The time axis is **preallocated to the whole MAM season** when the store is
created, and dates are then filled in. A date that was never filled is
therefore **not absent**:

- it is on the `time` coordinate, so `ds.sizes["time"]` says 92;
- `.sel(time="2024-05-01")` finds it and raises nothing;
- it reads back as **all-NaN**.

A consumer asking for 2024-05-01 gets a field of NaN with no indication that it
means *not published* rather than *no rain*. Verified both ways:

| | 2024-03-01 (written) | 2024-04-03 (missing) |
|---|---|---|
| manifest | chunks present | no chunks |
| data read | `finite = 1.000`, 263.2–302.0 K | `finite = 0.000`, all NaN |

All **36 channels agree** on exactly the same 59 dates, so this is whole dates
missing from the materialization, not individual fields lost.

This is the same hazard as the union step axis in
`verify_store_completeness.py` and as the longitude bug: **preallocation makes
absence look like data**, and only an explicit check tells them apart. It cost a
sweep here — two comparisons ran against a field of NaN, reported "0 members
matched", and then died with `KeyError: 'corr'`, an error naming the wrong
problem entirely.

### Checking coverage

Do not infer it from `ds.sizes["time"]`, and do not read every field looking for
NaN — that moves terabytes. The manifest knows which chunks exist and
`Session.chunk_coordinates` enumerates them without fetching any:

```bash
uv run ecmwf/check_realized_coverage.py                       # all four
uv run ecmwf/check_realized_coverage.py --sub 49r1-mam2024 \
    --all-channels --verify-data                              # prove it two ways
uv run ecmwf/check_realized_coverage.py --json coverage.json  # per-date map
```

Exit status is 1 if any sub-store is incompletely written, so it can gate a
publish. The current per-date map is
`ecmwf/gik_vs_herbie/realized_ea_swio_eval/coverage.json`.

Tip snapshots at the time of writing:

| sub-store | snapshot |
|---|---|
| 49r1-mam2024 | `JDNA5PW5Z9EGN08W4PF0` |
| 49r1-mam2025 | `WVG5M4VQVXQ6NJABY5W0` |
| 49r1-mam2026 | `0HVA7ZZJ55YBPF63Z9ZG` |
| 50r1-mam2026-tail | `ZJK2N4HNC5QGM12P6FHG` |

### Working around it

Two options, and the first is better if you can afford it.

1. **Finish the materialization** for 2024-04-03 … 2024-05-31, or trim the axis
   to what is written. Right now the store advertises a full MAM 2024 season it
   does not have.
2. **Fall back to the virtual store**, which carries every date of every era.
   Subset it to the same box and you get the identical 321 × 261 grid:

   ```bash
   uv run ecmwf/compare_icechunk_herbie.py \
       --store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
       --era 49r1 --run 00 --date 20240501 --step 0 --var u --levels 700 \
       --lon-min 15 --lon-max 80 --lat-min -40 --lat-max 40
   ```

   That path is verified: the two 2024 dates missing here were checked against
   the virtual store over this box and are bit-exact against Herbie, so a gap in
   the published subset need not become a gap in the validation.

Note the virtual store is a different kind of object — its chunks are byte-range
references into `s3://ecmwf-forecasts`, so reading it needs `gribberish`, the
anonymous virtual-chunk container credentials, and a live ECMWF archive. See
`v20260819-icechunk-store-source-coop.md`.
