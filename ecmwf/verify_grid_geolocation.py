# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "s3fs", "eccodes", "gribberish>=1.4"]
# ///
"""Prove an ECMWF grid/store actually sits where it claims to sit.

This is the check that did not exist, and whose absence let a 180 deg longitude
error ship into 3.5 TB of Icechunk stores. Every pre-existing check compared the
store to *itself* -- internally self-consistent, uniformly displaced, invisible.
These checks compare against something outside the store: the GRIB header, known
geography, and the raw decoded message.

See HANDOVER_LONGITUDE_FIX.md (rungs 1, 2 and 6) and grids.py.

Four subcommands, cheapest first
-------------------------------
    header   GRIB Section 3 vs grids.py               ~5 s, no decode  (rung 1)
    land     land-sea mask geography, both labellings ~20 s            (rung 1b)
    codec    gribberish byte-equality vs eccodes      ~20 s
    store    realized Icechunk store vs source GRIB   ~1 min           (rungs 2/6)

    uv run verify_grid_geolocation.py header
    uv run verify_grid_geolocation.py land   --era 49r1 --date 20250515
    uv run verify_grid_geolocation.py codec  --era 49r1 --date 20250515
    uv run verify_grid_geolocation.py store  --prefix ea-swio/v1-49r1-mar-may2026 \
                                             --era 49r1 --env ../.env --var lsm

Exit status is 0 only if every check passed, so this drops straight into CI or a
pre-write gate (`build_corpus.py`, `parallel_write.py`).

Split environments
------------------
`store` needs icechunk+xarray; `header`/`land`/`codec` need eccodes+s3fs. If no
single interpreter has both (common), bridge them with an npz:

    <icechunk-python> verify_grid_geolocation.py store --prefix ... --dump-npz /tmp/r.npz
    <eccodes-python>  verify_grid_geolocation.py store --from-npz /tmp/r.npz --era 49r1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grids import ERAS, grid_of, latitudes, longitudes  # noqa: E402

# Geography that does not move. Land probes must be land, ocean probes ocean --
# on the TRUE axis. Under a 180 deg displacement every one of these flips.
PROBES = [
    ("Congo basin",       0.0,   25.0, True),
    ("Ethiopia highland", 9.0,   39.0, True),
    ("Kalahari",        -22.0,   22.0, True),
    ("Amazon",           -3.0,  -60.0, True),
    ("Tibet",            30.0,   90.0, True),
    ("Indian Ocean",    -20.0,   65.0, False),
    ("Mozambique Ch.",  -18.0,   42.0, False),
    ("mid-Pacific",       0.0, -160.0, False),
    ("S Atlantic",      -30.0,  -20.0, False),
]
EA_BOX = dict(lat=(25.25, -40.0), lon=(15.0, 80.0))  # the CRMA ea-swio domain
EA_LAND_RANGE = (0.30, 0.55)  # East Africa + SWIO is ~41% land; Pacific is ~1%

# Realized stores flatten (param, level) into one channel name -- ("u", 925,
# "u925") in frisky_daily_dag.CHANNELS. To fetch the matching GRIB message we
# have to undo that. Surface names are listed explicitly rather than inferred,
# because "u10"/"v10" are surface fields whose trailing digits would otherwise
# parse as the 10 hPa level that 50r1 added.
SFC_ALIASES = {"t2m": "2t", "u10": "10u", "v10": "10v", "d2m": "2d",
               "u100": "100u", "v100": "100v"}
SURFACE_NAMES = set(SFC_ALIASES) | {
    "lsm", "msl", "sp", "tp", "ro", "skt", "tcwv", "tcw", "tcc", "ssr", "ssrd",
    "str", "strd", "ttr", "mucape", "sf", "sd", "asn", "rsn", "ptype", "zos",
    "sithick", "ewss", "nsss", "10fg", "sot", "vsw", "sve", "svn", "tprate",
    "mn2t3", "mx2t3",
}
_PL_NAME = __import__("re").compile(r"^([a-z]+)(\d{2,4})$")


def split_channel(name: str) -> tuple[str, int | None]:
    """'u925' -> ('u', 925);  't2m' -> ('2t', None);  'tp' -> ('tp', None)."""
    if name in SURFACE_NAMES:
        return SFC_ALIASES.get(name, name), None
    if name.endswith("_sfc"):                      # builder's both-levtypes suffix
        base = name[:-4]
        return SFC_ALIASES.get(base, base), None
    m = _PL_NAME.match(name)
    if m:
        return m.group(1), int(m.group(2))
    return SFC_ALIASES.get(name, name), None


def _s3():
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    import s3fs
    return s3fs.S3FileSystem(anon=True)


def grib_path(era: str, date: str, run: str, step: int = 0) -> str:
    seg = "0p4-beta" if era == "0p4" else "ifs/0p25"
    return (f"ecmwf-forecasts/{date}/{run}z/{seg}/enfo/"
            f"{date}{run}0000-{step}h-enfo-ef.grib2")


def fetch_message(era: str, date: str, run: str, var: str | None = None,
                  number: str = "1", step: int = 0, level: int | None = None):
    """Byte range of one GRIB message, located via the companion .index.

    `var` is a GRIB shortName (not a flattened channel name -- run it through
    `split_channel` first). `level` selects a pressure level; None means a
    surface/soil field.
    """
    fs, p = _s3(), grib_path(era, date, run, step)
    idx = p.replace(".grib2", ".index")
    if not fs.exists(idx):
        raise SystemExit(f"no .index at s3://{idx} -- wrong era for this date?")
    recs = [json.loads(L) for L in fs.cat(idx).decode().splitlines()]
    if var is None:
        return fs.read_block(p, int(recs[0]["_offset"]),
                             int(recs[0]["_length"])), recs[0]

    def match(r, want_number):
        if r.get("param") != var:
            return False
        if level is None:
            if r.get("levtype") not in ("sfc", "sol"):
                return False
        elif r.get("levtype") != "pl" or int(r.get("levelist", -1)) != level:
            return False
        return (r.get("number") == want_number) if want_number else True

    hits = [r for r in recs if match(r, number)] or \
           [r for r in recs if match(r, None)]
    if not hits:
        avail = sorted({r.get("param") for r in recs})
        where = f"{var}" + (f"@{level}hPa" if level else " (surface)")
        raise SystemExit(f"{where} not in {Path(p).name}; params available: "
                         f"{', '.join(avail)}")
    rec = hits[0]
    return fs.read_block(p, int(rec["_offset"]), int(rec["_length"])), rec


def decode(buf: bytes, era: str) -> tuple[np.ndarray, dict]:
    """eccodes decode -> (ny, nx) array + Section 3 header."""
    import eccodes
    h = eccodes.codes_new_from_message(buf)
    try:
        hdr = {k: eccodes.codes_get(h, k) for k in (
            "gridType", "Ni", "Nj",
            "longitudeOfFirstGridPointInDegrees",
            "longitudeOfLastGridPointInDegrees",
            "latitudeOfFirstGridPointInDegrees",
            "latitudeOfLastGridPointInDegrees",
            "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees",
            "iScansNegatively", "jScansPositively")}
        vals = eccodes.codes_get_values(h)
    finally:
        eccodes.codes_release(h)
    return vals.reshape(grid_of(era)["ny"], grid_of(era)["nx"]), hdr


# --------------------------------------------------------------------------
# rung 1 -- the GRIB header is the ground truth for the origin
# --------------------------------------------------------------------------
def cmd_header(a) -> bool:
    cases = ([(a.era, a.date)] if a.era else
             [("0p4", "20230601"), ("49r1", "20250515"), ("50r1", "20260621")])
    ok = True
    for era, date in cases:
        try:
            buf, rec = fetch_message(era, date, a.run)
            _, hdr = decode(buf, era)
        except SystemExit as e:
            print(f"{era:6s} SKIP  {e}")
            continue
        g = grid_of(era)
        first = hdr["longitudeOfFirstGridPointInDegrees"]
        checks = [
            ("gridType regular_ll", hdr["gridType"] == "regular_ll"),
            (f"Ni={g['nx']} Nj={g['ny']}",
             hdr["Ni"] == g["nx"] and hdr["Nj"] == g["ny"]),
            # eccodes normalises Section 3 to 0-360, so -180 reads back as 180.0.
            ("first lon = +/-180 (NOT 0)",
             min(abs(first - 180.0), abs(first + 180.0)) < 1e-6),
            ("lat 90 -> -90",
             hdr["latitudeOfFirstGridPointInDegrees"] == 90.0
             and hdr["latitudeOfLastGridPointInDegrees"] == -90.0),
            ("scan order +i, -j",
             hdr["iScansNegatively"] == 0 and hdr["jScansPositively"] == 0),
            (f"increment {g['dlon']}",
             abs(hdr["iDirectionIncrementInDegrees"] - g["dlon"]) < 1e-9),
        ]
        good = all(c for _, c in checks)
        ok &= good
        print(f"\n{era} {date} {a.run}z  ({rec.get('param')})  "
              f"{'PASS' if good else 'FAIL'}")
        print(f"  header: first lon {first}  last lon "
              f"{hdr['longitudeOfLastGridPointInDegrees']}  "
              f"lat {hdr['latitudeOfFirstGridPointInDegrees']} -> "
              f"{hdr['latitudeOfLastGridPointInDegrees']}")
        for label, c in checks:
            print(f"    [{'ok' if c else 'XX'}] {label}")
        print(f"  grids.py: lon {longitudes(era)[0]:+.2f} .. "
              f"{longitudes(era)[-1]:+.2f}   lat {latitudes(era)[0]:+.1f} .. "
              f"{latitudes(era)[-1]:+.1f}")
    return ok


# --------------------------------------------------------------------------
# rung 1b -- geography. Decisive without touching any store.
# --------------------------------------------------------------------------
def _box(a2d, era, lon_axis):
    lat = latitudes(era)
    la = np.where((lat <= EA_BOX["lat"][0]) & (lat >= EA_BOX["lat"][1]))[0]
    li = np.where((lon_axis >= EA_BOX["lon"][0])
                  & (lon_axis <= EA_BOX["lon"][1]))[0]
    return a2d[np.ix_(la, li)]


def cmd_land(a) -> bool:
    buf, rec = fetch_message(a.era, a.date, a.run, var="lsm")
    lsm, _ = decode(buf, a.era)
    lat, lon = latitudes(a.era), longitudes(a.era)
    g = grid_of(a.era)
    wrong = np.arange(g["nx"]) * g["dlon"]          # the old 0-start assumption

    frac_ok = _box(lsm, a.era, lon).mean()
    frac_bad = _box(lsm, a.era, wrong).mean()
    lo, hi = EA_LAND_RANGE
    box_pass = lo < frac_ok < hi
    print(f"\n{a.era} {a.date} lsm -- land fraction over "
          f"{EA_BOX['lon'][0]}..{EA_BOX['lon'][1]}E / "
          f"{EA_BOX['lat'][0]}..{EA_BOX['lat'][1]}N")
    print(f"  grids.py axis (-180 start) : {frac_ok:.4f}   "
          f"{'PASS' if box_pass else 'FAIL'}  (expect {lo}-{hi})")
    print(f"  old 0-start axis           : {frac_bad:.4f}   "
          f"<- what the broken builder selected (eastern Pacific)")

    print("\n  point probes on the grids.py axis:")
    probe_pass = True
    for name, plat, plon, is_land in PROBES:
        i = int(np.abs(lat - plat).argmin())
        k = int(np.abs(lon - plon).argmin())
        v = float(lsm[i, k])
        good = (v > 0.5) if is_land else (v < 0.5)
        probe_pass &= good
        print(f"    [{'ok' if good else 'XX'}] {name:18s} "
              f"({plat:+6.1f},{plon:+7.1f})  lsm={v:.3f}  "
              f"expect {'land' if is_land else 'ocean'}")
    return box_pass and probe_pass


# --------------------------------------------------------------------------
# gribberish must preserve the GRIB scan order, or the origin argument is moot
# --------------------------------------------------------------------------
def cmd_codec(a) -> bool:
    buf, rec = fetch_message(a.era, a.date, a.run, var=a.var)
    ec, _ = decode(buf, a.era)
    import gribberish
    gb = np.asarray(gribberish.parse_grib_message(buf, 0).data()).reshape(ec.shape)
    d = float(np.nanmax(np.abs(gb - ec)))
    ok = np.allclose(gb, ec, equal_nan=True)
    print(f"\n{a.era} {a.date} {rec.get('param')}: gribberish vs eccodes")
    print(f"  max |delta| = {d:.6g}   ordering preserved: "
          f"{'PASS' if ok else 'FAIL'}")
    print("  (if this fails the fault is in the codec, not the grid origin)")
    return ok


# --------------------------------------------------------------------------
# rungs 2 & 6 -- realized store vs source GRIB, selected BY VALUE on both sides
# --------------------------------------------------------------------------
# Preference order when --var auto. `lsm` first: time-invariant, and its land
# fraction is a hard number over any known box. Then thermal fields, which have
# the largest land/ocean contrast. Every one of these is byte-comparable, so the
# choice only affects how readable the output is.
AUTO_VARS = ["lsm", "skt", "t2m", "sp", "msl", "t850", "t500", "tcwv", "tcw",
             "gh500", "u850", "v850", "r850", "q850", "u200", "w500"]


def _open_realized(a):
    """Open a realized Icechunk store on EWC Ceph, read-only."""
    import icechunk as ic
    import xarray as xr
    ak, sk = os.environ.get("AK"), os.environ.get("SK")
    if a.env and not (ak and sk):
        for line in Path(a.env).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        ak, sk = os.environ.get("AK"), os.environ.get("SK")
    if not (ak and sk):
        raise SystemExit("AK/SK not set and --env did not supply them")
    storage = ic.s3_storage(bucket=a.bucket, prefix=a.prefix, region="RegionOne",
                            endpoint_url=a.endpoint, access_key_id=ak,
                            secret_access_key=sk, force_path_style=True,
                            from_env=False)
    try:
        repo = ic.Repository.open(storage)
    except Exception as e:
        raise SystemExit(f"cannot open {a.bucket}/{a.prefix}: "
                         f"{str(e).splitlines()[0].strip()}")
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False,
                        zarr_format=3, decode_timedelta=True)


def _dump_realized(a) -> dict:
    """Read one readable 2-D slice out of a realized store.

    Stores differ in which channels they carry (the ea-cgan corpora hold 30
    flattened channels and no `lsm`; ea-swio held 36 with it) and some have
    genuinely missing chunks, so this walks candidates until one reads rather
    than insisting on a single variable.
    """
    ds = _open_realized(a)
    have = sorted(ds.data_vars)
    if a.var == "auto":
        cands = [v for v in AUTO_VARS if v in have] + \
                [v for v in have if v not in AUTO_VARS]
    elif a.var in have:
        cands = [a.var]
    else:
        raise SystemExit(f"{a.var!r} not in store; have: {', '.join(have)}")

    times = [a.time] if a.time is not None else \
            list(dict.fromkeys([0, ds.sizes["time"] // 2, ds.sizes["time"] - 1]))
    skipped = []
    for var in cands:
        for ti in times:
            da = ds[var].isel(time=ti, number=a.number, step=a.step)
            if "isobaricInhPa" in da.dims:
                da = da.isel(isobaricInhPa=0)
            try:
                arr = np.asarray(da.values, dtype="float64")
            except Exception as e:                 # missing chunk, decode failure
                skipped.append(f"{var}@t{ti}: {str(e).splitlines()[0].strip()[:44]}")
                continue
            if np.all(np.isnan(arr)):
                skipped.append(f"{var}@t{ti}: all-NaN")
                continue
            if skipped:
                print(f"  skipped {len(skipped)} unreadable slice(s): "
                      f"{'; '.join(skipped[:3])}"
                      f"{' ...' if len(skipped) > 3 else ''}")
            print(f"  using channel {var!r} at time index {ti} "
                  f"(number={a.number}, step={a.step})")
            return dict(a=arr, lat=ds.latitude.values, lon=ds.longitude.values,
                        var=np.array(var),
                        date=np.array(str(ds.time.values[ti])[:10].replace("-", "")))
    raise SystemExit("no readable slice found. tried:\n  "
                     + "\n  ".join(skipped[:20]))


def cmd_store(a) -> bool:
    if a.from_npz:
        z = np.load(a.from_npz, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    else:
        d = _dump_realized(a)
        if a.dump_npz:
            np.savez(a.dump_npz, **d)
            print(f"wrote {a.dump_npz} -- now re-run with --from-npz in an "
                  f"interpreter that has eccodes")
            return None          # a dump is not a verdict; main() prints DUMPED
    real = d["a"]
    rlat, rlon = d["lat"], d["lon"]
    var, date = str(d["var"]), str(d["date"])

    param, level = split_channel(var)
    buf, rec = fetch_message(a.era, date, a.run, var=param, level=level,
                             number=str(a.number))
    src, _ = decode(buf, a.era)
    lat_t, lon_t = latitudes(a.era), longitudes(a.era)

    li = int(np.abs(lat_t - rlat[0]).argmin())
    la = slice(li, li + len(rlat))
    if not np.allclose(lat_t[la], rlat, atol=1e-6):
        print(f"  latitude axes do not line up -- store {rlat[0]}..{rlat[-1]} "
              f"vs source {lat_t[la][0]}..{lat_t[la][-1]}")
        return False

    def sel(lon_lo):
        k = int(np.abs(lon_t - lon_lo).argmin())
        return src[la, k:k + len(rlon)], lon_t[k:k + len(rlon)]

    true_lo = float(rlon[0])
    disp_lo = true_lo - 180.0 if true_lo >= 0 else true_lo + 180.0
    src_name = param + (f"@{level}hPa" if level else " (surface)")
    print(f"\n{a.bucket}/{a.prefix}  channel {var} -> GRIB {src_name}  {date}")
    print(f"  shape {real.shape}, store mean {np.nanmean(real):.4f}")
    print(f"  store labels itself lon {rlon[0]:+.2f}..{rlon[-1]:+.2f}  "
          f"lat {rlat[0]:+.2f}..{rlat[-1]:+.2f}")

    verdict = {}
    for label, lo in (("as labelled (correct)", true_lo),
                      ("displaced 180 deg", disp_lo)):
        sub, ax = sel(lo)
        if sub.shape != real.shape:
            print(f"  {label}: shape {sub.shape} != {real.shape}, skipped")
            continue
        dmax = float(np.nanmax(np.abs(real - sub)))
        # Stores hold float32; eccodes hands back float64. Compare at float32,
        # or a match shows up as ~1.5e-5 at 300 K (2^-16) and reads as a
        # near-miss. A genuine 180 deg displacement differs by tens of kelvin,
        # so nothing subtle hides in this tolerance.
        same = bool(np.array_equal(
            np.nan_to_num(real.astype("float32")),
            np.nan_to_num(sub.astype("float32"))))
        verdict[label] = same
        print(f"\n  {label}: source lon {ax[0]:+.2f}..{ax[-1]:+.2f}")
        print(f"    source mean {np.nanmean(sub):.4f}   "
              f"max |delta| {dmax:.6g} (float64)   identical at float32: {same}")

    ok = verdict.get("as labelled (correct)") and not verdict.get("displaced 180 deg")
    if ok:
        print("\n  VERDICT: CORRECT -- the store holds the region it claims")
    elif verdict.get("displaced 180 deg"):
        print(f"\n  VERDICT: WRONG REGION -- displaced 180 deg in longitude.\n"
              f"           Labelled {rlon[0]:+.2f}..{rlon[-1]:+.2f}E but holds "
              f"{disp_lo:+.2f}..{disp_lo + (rlon[-1] - rlon[0]):+.2f}E.\n"
              f"           The wrong bytes were read, so relabelling cannot fix "
              f"it -- rebuild is required.")
    else:
        print("\n  VERDICT: INCONCLUSIVE -- matches neither candidate region. "
              "Wrong --era, wrong date, or a different defect.")

    # Land fraction, calibrated against this store's own box rather than a
    # hardcoded constant, so it works for any domain (ea-swio, ea-cgan, ...).
    if var == "lsm":
        lf = float(np.nanmean(real))
        exp_true = float(np.nanmean(sel(true_lo)[0]))
        exp_disp = float(np.nanmean(sel(disp_lo)[0]))
        print(f"  lsm land fraction: store {lf:.4f} | source at true coords "
              f"{exp_true:.4f} | source displaced {exp_disp:.4f}")
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, era_required=False):
        p.add_argument("--era", choices=sorted(ERAS), required=era_required)
        p.add_argument("--date", default="20250515", help="YYYYMMDD")
        p.add_argument("--run", default="00", choices=["00", "06", "12", "18"])

    p = sub.add_parser("header", help="rung 1: GRIB Section 3 vs grids.py")
    common(p)
    p.set_defaults(fn=cmd_header)

    p = sub.add_parser("land", help="rung 1b: land-sea mask geography")
    common(p, era_required=True)
    p.add_argument("--var", default="lsm")
    p.set_defaults(fn=cmd_land)

    p = sub.add_parser("codec", help="gribberish vs eccodes byte equality")
    common(p, era_required=True)
    p.add_argument("--var", default="lsm")
    p.set_defaults(fn=cmd_codec)

    p = sub.add_parser("store", help="rungs 2/6: realized store vs source GRIB")
    common(p, era_required=True)
    p.add_argument("--prefix", required=True, help="e.g. ea-swio/v1-49r1-mar-may2026")
    p.add_argument("--bucket", default="must-icechunk")
    p.add_argument("--endpoint",
                   default="https://object-store.os-api.cci1.ecmwf.int")
    p.add_argument("--env", default=None, help="file holding AK=/SK=")
    p.add_argument("--var", default="auto",
                   help="channel name, or 'auto' to pick the first readable one "
                        "(lsm preferred; the ea-cgan stores do not carry it)")
    p.add_argument("--time", type=int, default=None,
                   help="time index; default tries first/middle/last so a "
                        "missing chunk does not abort the check")
    p.add_argument("--number", type=int, default=1)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--dump-npz", default=None,
                   help="split-env bridge: write the store slice and stop")
    p.add_argument("--from-npz", default=None,
                   help="split-env bridge: compare a previously dumped slice")
    p.set_defaults(fn=cmd_store)

    a = ap.parse_args()
    ok = a.fn(a)
    if ok is None:                       # --dump-npz half of the split-env bridge
        print(f"\n{'=' * 62}\n{a.cmd}: DUMPED (no verdict yet)")
        return 0
    print(f"\n{'=' * 62}\n{a.cmd}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
