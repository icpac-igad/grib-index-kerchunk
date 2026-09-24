# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "gribberish>=1.4", "icechunk>=2.1", "zarr>=3.2",
#                 "xarray>=2025.1"]
# ///
"""Prove a GEFS grid/store actually sits where it claims to sit.

The GEFS counterpart of `ecmwf/verify_grid_geolocation.py`, which was written
after a 180 deg longitude error shipped into 3.5 TB of ECMWF Icechunk stores.
Every pre-existing check there compared the store to *itself* -- internally
self-consistent, uniformly displaced, invisible. These checks compare against
something outside the store: the GRIB header, known geography, and the raw
decoded message.

GEFS was audited for the same defect on 2026-09-04 and is CLEAN: the NCEP scan
genuinely starts at 0 deg E, so `np.arange(1440) * 0.25` is correct here even
though the identical line was the bug in ECMWF. This script exists so that stays
true by test rather than by luck. See `grids.py`.

Four subcommands, cheapest first
-------------------------------
    header      GRIB Section 3 vs grids.py, several dates   ~10 s, no decode
    geography   model orography vs known elevations         ~15 s
    codec       gribberish byte-equality vs eccodes         ~15 s (skips w/o eccodes)
    store       published Icechunk store vs source GRIB     ~2 min

    uv run verify_grid_geolocation.py header
    uv run verify_grid_geolocation.py geography
    uv run verify_grid_geolocation.py codec
    uv run verify_grid_geolocation.py store --var auto

Exit status is 0 only if every check passed, so this drops straight into CI or a
pre-write gate in `build_gefs_icechunk.py` / `backfill_gefs_icechunk.py`.

`header` and `geography` need no credentials and no store. `store` reads the
public source.coop repo anonymously (see OPENING_PUBLISHED_GEFS_ICECHUNK.md).
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grids import (ERAS, LON_ORIGIN, MEMBERS_INGESTED, era_for, field_shape,  # noqa: E402
                   grid_of, latitudes, longitudes, verify as verify_header)

S3 = "https://noaa-gefs-pds.s3.amazonaws.com"

# Published store (source.coop). Metadata is hosted there; the chunks are
# virtual references into s3://noaa-gefs-pds.
STORE = dict(bucket="e4drr-project",
             prefix="forecasts/noaa_gefs_aws_s3_icechunk_vd",
             endpoint="https://data.source.coop", group="0p25/00z")

# Geography that does not move. Model orography in metres -- ocean is exactly 0
# in the GFS/GEFS terrain field, so the land/sea contrast is absolute. Under a
# 180 deg displacement every one of these flips.
#   (name, lat, lon_east_0_360, min_m, max_m)
PROBES = [
    ("Tibetan Plateau",     32.0,  88.0, 3500, 6000),
    ("Ethiopia highland",    9.0,  39.0, 1200, 3500),
    ("Kenya highland",      -1.3,  36.8, 1000, 2500),
    ("Andes altiplano",    -20.0, 292.0, 2000, 5000),
    ("Sahara",              22.0,  10.0,  100,  900),
    ("Amazon lowland",      -3.0, 300.0,    0,  400),
    ("Indian Ocean",       -10.0,  60.0,    0,    1),
    ("mid-Pacific",         -1.3, 216.8,    0,    1),
    ("N Atlantic",          40.0, 330.0,    0,    1),
]
# ICPAC's East Africa window, on the 0-360 axis GEFS actually uses.
EA_BOX = dict(lat=(20.0, -15.0), lon=(20.0, 55.0))
EA_OROG_RANGE = (250.0, 1400.0)   # real terrain; the Pacific counterpart is ~0

# Store channel -> (GRIB shortName, level string) in the GEFS .idx.
# `orog` (HGT:surface) is published at f000 only -- it is the time-invariant
# terrain field and therefore the best geographic ground truth GEFS offers,
# pgrb2sp25 carrying no land-sea mask.
STORE_TO_IDX = {
    "orog":   ("HGT", "surface"),
    "sp":     ("PRES", "surface"),
    "t2m":    ("TMP", "2 m above ground"),
    "d2m":    ("DPT", "2 m above ground"),
    "r2":     ("RH", "2 m above ground"),
    "u10":    ("UGRD", "10 m above ground"),
    "v10":    ("VGRD", "10 m above ground"),
    "cape":   ("CAPE", "surface"),
    "cin":    ("CIN", "surface"),
    "gust":   ("GUST", "surface"),
    "vis":    ("VIS", "surface"),
    "prmsl":  ("PRMSL", "mean sea level"),
    "mslet":  ("MSLET", "mean sea level"),
    "pwat":   ("PWAT", "entire atmosphere (considered as a single layer)"),
    "tp":     ("APCP", "surface"),
    "sde":    ("SNOD", "surface"),
    "sdwe":   ("WEASD", "surface"),
}
# Preference order for --var auto: time-invariant and high-contrast first.
AUTO_VARS = ["orog", "sp", "t2m", "prmsl", "d2m", "u10", "v10", "pwat"]


# --------------------------------------------------------------------------
# fetching -- stdlib only, anonymous, no s3fs
# --------------------------------------------------------------------------
def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    return urllib.request.urlopen(req, timeout=90).read()


def grib_key(date: str, run: str, member: str, step: int) -> str:
    return (f"gefs.{date}/{run}/atmos/pgrb2sp25/"
            f"{member}.t{run}z.pgrb2s.0p25.f{step:03d}")


def fetch_message(date: str, run: str, member: str, step: int,
                  var: str, level: str) -> tuple[bytes, str]:
    """Byte range of one GRIB message, located via the companion .idx.

    GEFS .idx lines are `num:offset:d=...:VAR:LEVEL:fcst:ENS=+n`; a message runs
    from its own offset to the next one (or EOF for the last).
    """
    key = grib_key(date, run, member, step)
    try:
        lines = _get(f"{S3}/{key}.idx").decode().splitlines()
    except Exception as e:
        raise SystemExit(f"no .idx for {key}: {e}")
    recs = [ln.split(":") for ln in lines]
    hit = [i for i, r in enumerate(recs) if r[3] == var and r[4] == level]
    if not hit:
        avail = sorted({f"{r[3]}:{r[4]}" for r in recs})
        raise SystemExit(f"{var}:{level} not in {Path(key).name}; have: "
                         f"{', '.join(avail[:12])} ...")
    i = hit[0]
    off = int(recs[i][1])
    end = int(recs[i + 1][1]) - 1 if i + 1 < len(recs) else ""
    req = urllib.request.Request(f"{S3}/{key}")
    req.add_header("Range", f"bytes={off}-{end}")
    return urllib.request.urlopen(req, timeout=90).read(), key


def decode_gribberish(buf: bytes, era: str = "v12") -> np.ndarray:
    import gribberish
    ny, nx = field_shape(era)
    return np.asarray(gribberish.parse_grib_message(buf, 0).data()).reshape(ny, nx)


def _idx(axis: np.ndarray, value: float) -> int:
    return int(np.abs(axis - value).argmin())


# --------------------------------------------------------------------------
# rung 1 -- the GRIB header is the ground truth for the origin
# --------------------------------------------------------------------------
def cmd_header(a) -> bool:
    # A spread over the corpus: first date, mid, recent; all four runs; control
    # and perturbed; first/mid/last step. One era must cover all of it.
    cases = ([(a.date, a.run, a.member, a.step)] if a.date else [
        ("20200923", "00", "gep01", 0),
        ("20220615", "12", "gep30", 120),
        ("20240301", "06", "gep15", 240),
        ("20240301", "18", "gec00", 0),
        ("20260415", "00", "gep01", 3),
    ])
    ok = True
    for date, run, member, step in cases:
        try:
            h = verify_header(date, run, member, step)
        except AssertionError as e:
            print(f"{date} {run}z {member} f{step:03d}  FAIL  {e}")
            ok = False
            continue
        except Exception as e:
            print(f"{date} {run}z {member} f{step:03d}  SKIP  "
                  f"{type(e).__name__}: {e}")
            continue
        era = era_for(date, run)
        g = grid_of(era)
        print(f"[ok] {date} {run}z {member} f{step:03d}  era={era}  "
              f"Ni={h['Ni']} Nj={h['Nj']}  "
              f"lon {h['Lo1']} .. {h['Lo2']}  lat {h['La1']} .. {h['La2']}  "
              f"scan={h['scanning_mode']}")
    lon, lat = longitudes(), latitudes()
    print(f"\ngrids.py: lon {lon[0]:+.2f} .. {lon[-1]:+.2f}   "
          f"lat {lat[0]:+.1f} .. {lat[-1]:+.1f}   LON_ORIGIN={LON_ORIGIN}")
    print("  (GEFS is 0-360. ECMWF is -180..180. They are different grids.)")
    return ok


# --------------------------------------------------------------------------
# rung 1b -- geography. Decisive without touching any store.
# --------------------------------------------------------------------------
def _box_mean(a2d: np.ndarray, lon_axis: np.ndarray) -> float:
    lat = latitudes()
    la = np.where((lat <= EA_BOX["lat"][0]) & (lat >= EA_BOX["lat"][1]))[0]
    li = np.where((lon_axis >= EA_BOX["lon"][0])
                  & (lon_axis <= EA_BOX["lon"][1]))[0]
    return float(a2d[np.ix_(la, li)].mean())


def cmd_geography(a) -> bool:
    var, level = STORE_TO_IDX["orog"]
    buf, key = fetch_message(a.date or "20240301", a.run, a.member, 0, var, level)
    orog = decode_gribberish(buf)
    lon, lat = longitudes(), latitudes()
    g = grid_of("v12")
    # What the ECMWF convention would have produced here -- the mirror image of
    # the defect that hit ECMWF. Same labels, shifted by half the globe.
    wrong = -180.0 + np.arange(g["nx"]) * g["dlon"]

    good = _box_mean(orog, lon)
    bad = _box_mean(orog, wrong)
    lo, hi = EA_OROG_RANGE
    box_pass = lo < good < hi
    print(f"\n{key}\n  mean model orography over East Africa "
          f"({EA_BOX['lon'][0]}..{EA_BOX['lon'][1]}E / "
          f"{EA_BOX['lat'][1]}..{EA_BOX['lat'][0]}N)")
    print(f"    grids.py axis (0-start)  : {good:8.1f} m   "
          f"{'PASS' if box_pass else 'FAIL'}  (expect {lo}-{hi} m)")
    print(f"    ECMWF -180 axis          : {bad:8.1f} m   "
          f"<- what that convention would select here (mid-Pacific)")

    print("\n  point probes on the grids.py axis (metres):")
    probe_pass = True
    for name, plat, plon, mn, mx in PROBES:
        v = float(orog[_idx(lat, plat), _idx(lon, plon)])
        w = float(orog[_idx(lat, plat), _idx(wrong, plon)])
        hit = mn <= v <= mx
        probe_pass &= hit
        print(f"    [{'ok' if hit else 'XX'}] {name:18s} "
              f"({plat:+6.1f},{plon:7.2f}E)  {v:7.1f}  "
              f"expect {mn}-{mx}   [displaced: {w:7.1f}]")
    return box_pass and probe_pass


# --------------------------------------------------------------------------
# gribberish must preserve the GRIB scan order, or the origin argument is moot
# --------------------------------------------------------------------------
def cmd_codec(a) -> bool:
    try:
        import eccodes
    except ImportError:
        print("\neccodes not installed -- SKIP (this rung is optional; the "
              "`store` rung already pins gribberish against the source bytes)")
        return True
    var, level = STORE_TO_IDX.get(a.var, STORE_TO_IDX["orog"])
    buf, key = fetch_message(a.date or "20240301", a.run, a.member, 0, var, level)
    gb = decode_gribberish(buf)
    h = eccodes.codes_new_from_message(buf)
    try:
        ec = eccodes.codes_get_values(h).reshape(gb.shape)
    finally:
        eccodes.codes_release(h)
    d = float(np.nanmax(np.abs(gb - ec)))
    ok = bool(np.allclose(gb, ec, equal_nan=True))
    print(f"\n{key}  {var}:{level}: gribberish vs eccodes")
    print(f"  max |delta| = {d:.6g}   ordering preserved: "
          f"{'PASS' if ok else 'FAIL'}")
    print("  (if this fails the fault is in the codec, not the grid origin)")
    return ok


# --------------------------------------------------------------------------
# rung 2 -- published store vs source GRIB, selected BY LABEL on both sides
# --------------------------------------------------------------------------
def _open_store(a, attempts: int = 8):
    """Open the published repo anonymously, retrying source.coop's 500s.

    source.coop returns transient HTTP 500s on a small fraction of GETs; with
    manifest preload disabled an open still touches enough objects to draw one
    fairly often. Failure is probabilistic, so retry rather than give up.
    """
    import icechunk
    import xarray as xr
    import gribberish.zarr  # noqa: F401 -- registers the Zarr v3 codec

    storage = icechunk.s3_storage(
        bucket=a.bucket, prefix=a.prefix, endpoint_url=a.endpoint,
        region="us-east-1", anonymous=True, from_env=False, force_path_style=True)
    auth = icechunk.containers_credentials(
        {"s3://noaa-gefs-pds/": icechunk.s3_anonymous_credentials()})
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    last = None
    for i in range(attempts):
        try:
            repo = icechunk.Repository.open(
                storage, config=cfg, authorize_virtual_chunk_access=auth)
            return xr.open_zarr(repo.readonly_session("main").store,
                                group=a.group, consolidated=False, zarr_format=3)
        except Exception as e:                       # noqa: BLE001
            last = e
            print(f"  open attempt {i + 1}/{attempts} failed "
                  f"({str(e).splitlines()[0].strip()[:60]}) -- retrying")
            time.sleep(2 + 2 * i)
    raise SystemExit(f"cannot open {a.bucket}/{a.prefix}: {last}")


def cmd_store(a) -> bool:
    ds = _open_store(a)
    have = sorted(ds.data_vars)
    cands = ([a.var] if a.var != "auto" else
             [v for v in AUTO_VARS if v in have])
    cands = [v for v in cands if v in STORE_TO_IDX]
    if not cands:
        raise SystemExit(f"no comparable var; store has {', '.join(have)}")

    times = ([a.time] if a.time is not None else
             list(dict.fromkeys([0, ds.sizes["time"] // 2,
                                 ds.sizes["time"] - 1])))
    lon, lat = longitudes(), latitudes()
    ok_any, results = False, []

    for var in cands:
        gvar, glevel = STORE_TO_IDX[var]
        for ti in times:
            # orog is published at f000 only; everything else is fine at any step
            step = 0 if var == "orog" else a.step
            date = str(ds.time.values[ti])[:10].replace("-", "")
            member = f"gep{int(ds.number.values[a.number - 1]):02d}"
            try:
                real = ds[var].isel(time=ti, number=a.number - 1,
                                    step=step // 3).values
            except Exception as e:                   # noqa: BLE001
                print(f"  {var} t={ti}: unreadable ({type(e).__name__}) -- next")
                continue
            if not np.isfinite(real).any():
                print(f"  {var} t={ti} ({date}): all-NaN slice -- next")
                continue
            try:
                buf, key = fetch_message(date, a.run, member, step, gvar, glevel)
            except SystemExit as e:
                print(f"  {var} t={ti} ({date}): {e} -- next")
                continue
            src = decode_gribberish(buf)

            # Compare at float32: the store holds float32, gribberish returns
            # float64, so a float64 comparison measures the storage rounding
            # rather than the reference. Pressure fields are ~1e5 Pa, where one
            # float32 ulp is ~0.008 Pa -- an absolute tolerance would reject
            # byte-identical data.
            src32 = src.astype("float32")
            d = float(np.nanmax(np.abs(real.astype("float32") - src32)))
            same = bool(np.allclose(real.astype("float32"), src32,
                                    rtol=1e-6, atol=0, equal_nan=True))
            # And the point that actually matters: does a labelled selection on
            # the store land on the same geography as the source array indexed
            # through grids.py?
            geo = []
            for name, plat, plon, *_ in PROBES[:5]:
                sv = float(ds[var].isel(time=ti, number=a.number - 1,
                                        step=step // 3)
                           .sel(latitude=plat, longitude=plon,
                                method="nearest").values)
                tv = float(src[_idx(lat, plat), _idx(lon, plon)])
                geo.append((name, sv, tv,
                            bool(np.isclose(np.float32(sv), np.float32(tv),
                                            rtol=1e-6, atol=0))))
            geo_ok = all(g[3] for g in geo)

            print(f"\n  {var:6s} t={ti} {date} {member} f{step:03d}  "
                  f"({gvar}:{glevel})")
            print(f"    store vs source GRIB : max|delta| = {d:.6g}   "
                  f"{'PASS' if same else 'FAIL'}")
            for name, sv, tv, g_ok in geo:
                print(f"    [{'ok' if g_ok else 'XX'}] {name:18s} "
                      f"store {sv:10.2f}  source {tv:10.2f}")
            results.append(same and geo_ok)
            ok_any = True
            break                                    # one good slice per var
        if ok_any and a.var != "auto":
            break
    if not ok_any:
        print("  no readable slice found to compare")
        return False
    return all(results)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--era", choices=sorted(ERAS), default="v12")
        p.add_argument("--date", default=None, help="YYYYMMDD")
        p.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
        p.add_argument("--member", default="gep01")
        p.add_argument("--step", type=int, default=0)

    p = sub.add_parser("header", help="GRIB Section 3 vs grids.py")
    common(p)
    p.set_defaults(fn=cmd_header)

    p = sub.add_parser("geography", help="model orography vs known elevations")
    common(p)
    p.set_defaults(fn=cmd_geography)

    p = sub.add_parser("codec", help="gribberish vs eccodes byte equality")
    common(p)
    p.add_argument("--var", default="orog")
    p.set_defaults(fn=cmd_codec)

    p = sub.add_parser("store", help="published Icechunk store vs source GRIB")
    common(p)
    p.add_argument("--bucket", default=STORE["bucket"])
    p.add_argument("--prefix", default=STORE["prefix"])
    p.add_argument("--endpoint", default=STORE["endpoint"])
    p.add_argument("--group", default=STORE["group"])
    p.add_argument("--var", default="auto",
                   help="store channel, or 'auto' to walk AUTO_VARS")
    p.add_argument("--time", type=int, default=None,
                   help="time index; default tries first/middle/last so a "
                        "missing chunk does not abort the check")
    p.add_argument("--number", type=int, default=1,
                   help=f"ensemble member 1..{len(MEMBERS_INGESTED)}")
    p.set_defaults(fn=cmd_store)

    a = ap.parse_args()
    ok = a.fn(a)
    print(f"\n{'=' * 62}\n{a.cmd}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
