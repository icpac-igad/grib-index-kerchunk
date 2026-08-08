# ECMWF ENS variables and message counts, by era

Reference for the three era groups in the published virtual Icechunk store
(`e4drr-project/forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd` on source.coop,
mirrored from `gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens`).

**All figures measured 2026-08-07** from the store itself — variable lists from
each group's schema, message counts from the 1,256 per-date commit messages,
which record `<era>/00z <YYYYMMDD>: 51 members, N refs`.
**00z cycle only.** See [§Reproducing](#reproducing-these-numbers).

> **The single most important caveat:** a group's variable list is the **union
> across its whole era**. No individual date carries all of them. A variable
> that appeared mid-era reads NaN on earlier dates, and one that was withdrawn
> reads NaN on later ones. Use the measured *fields per member per step*
> (§3) for sizing, never the arithmetic maximum (§2).

---

## 1. The three eras

| | `0p4` | `49r1` | `50r1` |
|---|---|---|---|
| window (00z) | 2023-01-18 → 2024-02-28 | 2024-02-29 → 2026-05-12 | 2026-05-13 → present |
| dates in store | 401 | 804 | 51 |
| grid | 451 × 900 (0.4°) | 721 × 1440 (0.25°) | 721 × 1440 (0.25°) |
| members | 51 (control bundled in `enfo`) | 51 (control bundled in `enfo`) | 50 perturbed + 1 control (**dual-stream**, control in `oper/fc`) |
| steps | 85 (0–360h) | 85 | 85 |
| **variables** | **19** | **59** | **54** |
| pressure levels | 9 | 13 | 14 |
| S3 path segment | `0p4-beta/` | `ifs/0p25/` | `ifs/0p25/` + `oper/fc` |

Era windows are **cycle-granular** — 06z/12z/18z switch a day earlier than 00z.
See `lithops-cr-gik-ecmwf/era_check.py` §E.

---

## 2. Variable lists

### `0p4` — 19 variables

**Pressure-level (8)** on 9 levels `[50, 200, 250, 300, 500, 700, 850, 925, 1000]`:

```
d, gh, q, r, t, u, v, vo
```

**Surface (11):**

```
lsm, msl, ro, skt, sp, st, t2m, tcwv, tp, u10, v10
```

Expanded: 8 × 9 = 72 + 11 = **83 fields**

### `49r1` — 59 variables

**Pressure-level (9)** on 13 levels
`[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]`:

```
d, gh, q, r, t, u, v, vo, w
```

**Surface (50):**

```
100u, 100v, 10fg, 10fg3, 2d, asn, cape, ewss, lsm, mn2t3, mn2t6, msl,
mucape, mx2t3, mx2t6, nsss, ptype, ro, rsn, sd, sf, sithick, skt, sot,
sp, ssr, ssrd, st, stl2, stl3, stl4, str, strd, sve, svn, swvl1, swvl2,
swvl3, swvl4, t2m, tcc, tcw, tcwv, tp, tprate, ttr, u10, v10, vsw, zos
```

Expanded: 9 × 13 = 117 + 50 = **167 fields** (ceiling; real dates carry 83–155)

### `50r1` — 54 variables

**Pressure-level (10)** on 14 levels
`[10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]`:

```
d, gh, q, r, t, u, v, vo, w, z
```

**Surface (44):**

```
100u, 100v, 10fg, 10fg3, 2d, asn, ewss, lsm, mn2t3, mn2t6, msl, mucape,
mx2t3, mx2t6, nsss, ptype, ro, rsn, sd, sdor, sf, sithick, skt, slor,
sot, sp, ssr, ssrd, str, strd, sve, svn, t2m, tcc, tcw, tcwv, tp,
tprate, ttr, u10, v10, vsw, z_sfc, zos
```

Expanded: 10 × 14 = 140 + 44 = **184 fields** (ceiling; real dates carry 164)

### How the eras relate

- **`0p4` ⊂ `49r1` exactly.** Zero variables exist in `0p4` but not `49r1`.
- **49r1 → 50r1 dropped 9:** `cape`, `st`, `stl2`–`stl4`, `swvl1`–`swvl4`
  (`mucape` supersedes `cape`; the soil temperature/moisture layers went away).
- **49r1 → 50r1 added 4:** `sdor`, `slor`, `z`, `z_sfc`.

**Only 18 variables exist in all three eras** — the safe set for any corpus
crossing era boundaries:

```
d, gh, lsm, msl, q, r, ro, skt, sp, t, t2m, tcwv, tp, u, u10, v, v10, vo
```

Note `w` (vertical velocity) is **absent from `0p4`**, and only 9 levels are
common to all three.

### Naming: two renames applied by the builder

`build_ecmwf_icechunk.py` normalises GRIB shortNames:

| GRIB | store |
|---|---|
| `2t`, `10u`, `10v` | `t2m`, `u10`, `v10` |
| `z` at surface (50r1 control carries both) | `z_sfc` — the pressure-level `z` keeps its name |

---

## 3. Message counts — what a date actually costs

Measured per date, then divided by 51 members and 85 steps.

| era | modal refs/date | min | max | fields/member/step | arithmetic ceiling |
|---|---|---|---|---|---|
| `0p4` | 359,805 | 355,470 | 359,805 | **83** | 83 |
| `49r1` | 654,585 | 359,473 | 671,925 | **83 → 155** (see below) | 167 |
| `50r1` | 712,133 | 712,133 | 712,133 | **164** | 184 |

**`0p4` hits its ceiling exactly** (83.00). Every field, every step, every
member. The only era where the arithmetic is the answer.

**`49r1` drifted through eight schema regimes** — this is the one that catches
people out. A corpus spanning it is not homogeneous:

| window | fields/mem/step | dates |
|---|---|---|
| 2024-02-29 → 2024-03-05 | 83 | 6 |
| 2024-03-06 → 2024-03-18 | 102 | 13 |
| 2024-03-19 → 2024-10-24 | 109 | 220 |
| 2024-10-25 → 2024-11-12 | 110 | 19 |
| 2024-11-13 → 2025-01-14 | 115 | 63 |
| **2025-01-15 → 2025-11-20** | **151** | 310 |
| 2025-11-21 → 2026-02-11 | 154 | 83 |
| 2026-02-12 → 2026-05-12 | 155 | 90 |

The 2025-01-15 jump (115 → 151) is the **9-level → 13-level** change: four
levels added (100, 150, 400, 600) × 9 pl variables = +36 fields. This is why a
49r1 template must be built from a 13-level reference date (`20250515`) — a
9-level template orphans those levels for every later date.

`0p4` has a smaller equivalent: 82 fields/mem/step until 2023-03-29, 83 after.

**`50r1` refs are not divisible by 51** — `712,133 / 51 = 13,963.39`. This is
expected, not corruption: 50r1 is dual-stream, so the control (from `oper/fc`)
carries a different field count from the 50 perturbed members (from `enfo/ef`).
The population is not uniform, so the arithmetic cannot factor evenly.

**Single-date dips** of 100–300 refs appear throughout (e.g. `654,434` and
`654,283` against a modal `654,585`). These are individual messages missing
upstream on that date, not a schema change — they do not shift the
fields/mem/step figure.

### Sizing rule of thumb

```
messages per date = fields/mem/step  x  51 members  x  85 steps
```

e.g. a 2026 49r1 date: 155 × 51 × 85 = **671,925** — exactly what each
March 2026 commit recorded. At ~0.788 MB/message that is ~530 GB of AWS
egress per date if every message is read.

---

## 4. Reproducing these numbers

Variable lists, straight from a group's schema:

```python
import frisky_daily_dag as dag            # icechunk-dask-frisky/
ds = dag._open_era("49r1")                # "0p4" | "49r1" | "50r1"
pl  = sorted(v for v in ds.data_vars if "isobaricInhPa" in ds[v].dims)
sfc = sorted(v for v in ds.data_vars if "isobaricInhPa" not in ds[v].dims)
levels = [int(x) for x in ds.isobaricInhPa.values]
```

Message counts, from the commit log (no data read):

```python
import re, collections
PAT = re.compile(r"^(\S+)/00z\s+(\d{8}):\s+(\d+) members,\s+(\d+) refs")
per = collections.defaultdict(list)
for s in repo.ancestry(branch="main"):
    m = PAT.match((s.message or "").strip())
    if m:
        per[m.group(1)].append((m.group(2), int(m.group(4))))
# fields per member per step = refs / 51 / 85
```

> Reading the axis of `0p4` or `49r1` requires care: both have **non-monotonic
> `time`** axes (three backfill commits on 2026-07-07 landed out of order), so
> `.sel(time=slice(...))` raises `KeyError`. Use `ds.sortby("time")` first, or
> exact-label `.sel(time=np.datetime64(d))`, which is unaffected.
> See `icechunk-dask-frisky/README.md` "Known defects in the published store".

---

## See also

| document | |
|---|---|
| `lithops-cr-gik-ecmwf/era_check.py` | era windows, cycle reach, the two era switches, GCS layout |
| `lithops-cr-gik-ecmwf/README.md` | how the per-era runtime image selects the template |
| `icechunk-par/build_ecmwf_icechunk.py` | `ERAS` table: grid + canonical level superset per era |
| `icechunk-dask-frisky/README.md` | known defects in the published store; date coverage audit |
