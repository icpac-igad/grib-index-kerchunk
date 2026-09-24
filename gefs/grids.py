"""Canonical NOAA GEFS grid definitions -- the single source of truth.

Import this. Do not re-derive a coordinate axis anywhere else.

Why this file exists
--------------------
`ecmwf/grids.py` was written after a 180 deg longitude error shipped into 3.5 TB
of ECMWF Icechunk stores: the builder assumed a 0-start axis when the ECMWF GRIB
scan begins at -180, so every East Africa subset silently read the eastern
Pacific. The defect was not numerical -- the same physical constant was
hand-written in twelve places with nothing binding the copies.

GEFS had the identical exposure: eight files spell the axis out by hand
(`np.linspace(0, 359.75, 1440)` in seven, `np.arange(NX) * 0.25` in the
builder). They all agree and they are all correct -- but nothing enforced that,
and nothing could have caught it if one drifted.

**GEFS IS NOT ECMWF.** The NCEP GRIB scan starts at 0 deg E, not -180. Decoded
straight from GRIB2 Section 3 (template 3.0) on `s3://noaa-gefs-pds`:

    Ni 1440   Nj 721   Di 0.25   Dj 0.25
    La1  90.0    La2 -90.0
    Lo1   0.0    Lo2  359.75      scanning_mode 0  (i scans W->E, j scans N->S)

Verified 2026-09-04 to be invariant across the whole published corpus -- dates
2020-09-23 / 2022-06-15 / 2024-03-01 / 2026-04-15, runs 00/06/12/18, control and
perturbed members, steps f000 / f120 / f240. One grid, one era.

So `LON_ORIGIN` here is **0.0**, and importing `ecmwf/grids.py` into GEFS code
would introduce the very error it was written to prevent, in reverse. The two
modules are deliberately separate and each self-checks its own origin at import.

Usage
-----
    from grids import ERAS, latitudes, longitudes, field_shape, normalize_lon

    lat = latitudes("v12")           # 90.0 ... -90.0    (721,)
    lon = longitudes("v12")          # 0.0  ... 359.75   (1440,)
    ny, nx = field_shape("v12")      # (721, 1440)

    # GEFS is 0-360. A +/-180 longitude must be normalised before .sel():
    ds.sel(longitude=normalize_lon(-20.0))    # -> 340.0, not a KeyError

Deliberately dependency-light: numpy only at import (and `verify()` needs
nothing beyond the standard library), so PEP 723 `uv run` scripts can
`sys.path`-insert this directory and import it without adding a dependency.
"""
from __future__ import annotations

import numpy as np

__all__ = ["GRIDS", "ERAS", "LON_ORIGIN", "LAT_ORIGIN", "STEPS", "RUNS",
           "MEMBERS_PUBLISHED", "MEMBERS_INGESTED", "ERA_BOUNDS",
           "grid_of", "era_for", "latitudes", "longitudes", "field_shape",
           "normalize_lon", "verify"]

# The origin. This is the fact that was wrong on the ECMWF side. It belongs here
# and nowhere else -- if you find yourself typing 359.75 into another file,
# import instead.
#
# NOTE THE SIGN: GEFS starts at 0 (0-360 convention), ECMWF starts at -180
# (+/-180 convention). Copying either module's origin into the other is a
# 180 deg error. See the module docstring.
LON_ORIGIN = 0.0      # longitudeOfFirstGridPointInDegrees, 0-360 convention
LAT_ORIGIN = 90.0     # latitudeOfFirstGridPointInDegrees (grid scans N -> S)

# Physical grids. One so far: the 0.25 deg pgrb2sp25 product.
GRIDS = {
    "0p25": dict(ny=721, nx=1440, dlat=0.25, dlon=0.25),
}

# Era table. GEFS v12 introduced pgrb2sp25 and is the only era in the archive:
# `gefs.20200922/00/atmos/` has no pgrb2sp25 directory at all, `gefs.20200923`
# does (checked 2026-09-04). Earlier GEFS is a different product at a different
# resolution and is not part of this corpus.
ERAS = {
    "v12": dict(grid="0p25"),
}

# ny/nx/dlon/dlat are mirrored onto each era entry so callers that do `era["nx"]`
# keep working. The mirror is derived, never hand-written.
for _era in ERAS.values():
    _era.update(GRIDS[_era["grid"]])
del _era

# Era boundaries as inclusive (YYYYMMDD, run_hour) pairs. `None` means open.
# Single entry today; the shape matches ecmwf/grids.py so a future v13 with a
# changed grid is a one-line addition rather than a new convention.
ERA_BOUNDS = {
    "v12": (("20200923", 0), None),
}

# Forecast axis. 0..240 h at 3 h = 81 steps, identical for all four runs --
# f240 is published, f243 is not (checked 2026-09-04). Unlike ECMWF, the GEFS
# step axis does NOT vary by run.
STEPS = np.arange(0, 241, 3, dtype="int32")

RUNS = ("00", "06", "12", "18")

# Members. NOAA publishes the control (`gec00`) alongside the 30 perturbed
# members in pgrb2sp25: 31 members x 81 steps = 2,511 GRIB files per (date,
# run), plus the derived `geavg`/`gespr` products which are NOT members.
# Verified at 2020-09-23, 2024-03-01 and 2026-04-15.
#
# CAUTION -- these two differ on purpose:
#   MEMBERS_PUBLISHED  what actually exists on S3                       (31)
#   MEMBERS_INGESTED   what build_gefs_icechunk.py writes to `number`   (30)
# The Icechunk store's `number` axis is 1..30 (gep01..gep30) and omits the
# control. CLAUDE.md and build_gefs_icechunk.py both assert "no control
# published in this path", which is not true of the archive. Recorded rather
# than silently corrected: widening `number` to include gec00 changes the store
# schema and needs a rebuild, not a constant edit.
MEMBERS_PUBLISHED = ("gec00",) + tuple(f"gep{i:02d}" for i in range(1, 31))
MEMBERS_INGESTED = tuple(f"gep{i:02d}" for i in range(1, 31))


def era_for(date: str, run: int | str = 0) -> str:
    """Which era a (YYYYMMDD, run) initialisation belongs to."""
    k = (str(date), int(str(run).rstrip("zZ")))
    for name, (lo, hi) in ERA_BOUNDS.items():
        if (lo is None or k >= lo) and (hi is None or k <= hi):
            return name
    raise KeyError(f"{date} {run}z is outside every known GEFS era window "
                   f"(pgrb2sp25 begins 2020-09-23)")


def grid_of(era: str) -> dict:
    """Grid dict for an era name ('v12') or a grid name ('0p25')."""
    if era in ERAS:
        return GRIDS[ERAS[era]["grid"]]
    if era in GRIDS:
        return GRIDS[era]
    raise KeyError(f"unknown era or grid {era!r}; "
                   f"eras={sorted(ERAS)} grids={sorted(GRIDS)}")


def latitudes(era: str = "v12") -> np.ndarray:
    """Latitude axis, north to south: 90.0 ... -90.0."""
    g = grid_of(era)
    return np.linspace(LAT_ORIGIN, LAT_ORIGIN - (g["ny"] - 1) * g["dlat"],
                       g["ny"], dtype="float64")


def longitudes(era: str = "v12") -> np.ndarray:
    """Longitude axis in the GRIB's own scan order: 0.0 ... 359.75.

    Monotonic ascending, so `.sel(longitude=slice(15, 80))` works directly for
    eastern-hemisphere domains and no `sortby`/`roll` is ever needed. Western
    longitudes must be normalised first -- see `normalize_lon`.
    """
    g = grid_of(era)
    return LON_ORIGIN + np.arange(g["nx"], dtype="float64") * g["dlon"]


def field_shape(era: str = "v12") -> tuple[int, int]:
    """(ny, nx) of one global field for this era."""
    g = grid_of(era)
    return g["ny"], g["nx"]


def normalize_lon(lon):
    """Map a +/-180 longitude onto this grid's 0-360 convention.

    The GEFS axis carries no negative labels, so `.sel(longitude=-20)` raises
    (or, with method="nearest", silently snaps to 0 deg E). Anyone arriving from
    the ECMWF side of this repo -- where the axis IS +/-180 -- will write that
    by reflex. Normalise instead:

        normalize_lon(-20.0)  -> 340.0
        normalize_lon(36.8)   ->  36.8

    Accepts scalars or arrays. Note that a +/-180 slice spanning the prime
    meridian (e.g. -20..55E) is NOT contiguous on a 0-360 axis and cannot be
    expressed as one `slice()`; select the two runs and concatenate, or roll the
    axis for plotting.
    """
    return np.mod(np.asarray(lon, dtype="float64"), 360.0)[()]


# ---------------------------------------------------------------------------
# Cheap invariants, checked at import. These cost microseconds and are the thing
# that would have caught the ECMWF defect the moment the module was loaded.
# ---------------------------------------------------------------------------
def _self_check() -> None:
    for name in ERAS:
        g, lat, lon = grid_of(name), latitudes(name), longitudes(name)
        assert lat.shape == (g["ny"],) and lon.shape == (g["nx"],)
        assert lat[0] == 90.0 and lat[-1] == -90.0, f"{name}: latitude span"
        # A GEFS global longitude axis starts at 0 and stops one increment short
        # of 360 (the 360 column is the same meridian as 0).
        assert lon[0] == 0.0, \
            f"{name}: GEFS longitude must start at 0, not {lon[0]} -- if this " \
            f"now reads -180 someone has copied ecmwf/grids.py in here"
        assert abs(lon[-1] - (360.0 - g["dlon"])) < 1e-9, \
            f"{name}: longitude must end at 360 - dlon, got {lon[-1]}"
        assert np.all(np.diff(lon) > 0), f"{name}: longitude not ascending"
        assert np.all(lon >= 0.0), f"{name}: 0-360 axis carries a negative label"
        # Nairobi (36.8E) must land in the first eighth of a 0-start axis. Under
        # the ECMWF -180 convention the same label lands past the midpoint.
        assert int(np.abs(lon - 36.8).argmin()) < g["nx"] // 8, \
            f"{name}: 36.8E is in the wrong place -- origin is not 0"
        assert normalize_lon(-20.0) == 340.0

    assert STEPS.size == 81 and STEPS[0] == 0 and STEPS[-1] == 240
    assert len(MEMBERS_PUBLISHED) == 31 and MEMBERS_PUBLISHED[0] == "gec00"
    assert len(MEMBERS_INGESTED) == 30 and "gec00" not in MEMBERS_INGESTED


_self_check()


def verify(date: str = "20240301", run: str = "00", member: str = "gep01",
           step: int = 0) -> dict:
    """Assert this table against a real GRIB message header (needs network).

    Returns the Section 3 fields read. Raises AssertionError on any mismatch.

    Unlike `ecmwf/grids.py.verify()`, this needs neither eccodes nor s3fs: it
    range-GETs the first message over public HTTPS and decodes GRIB2 Section 3
    (grid definition template 3.0) directly, so the module stays importable
    anywhere. `verify_grid_geolocation.py` is the CLI wrapper.
    """
    import struct
    import urllib.request

    base = "https://noaa-gefs-pds.s3.amazonaws.com"
    key = (f"gefs.{date}/{run}/atmos/pgrb2sp25/"
           f"{member}.t{run}z.pgrb2s.0p25.f{step:03d}")

    def _get(url, start=None, end=None):
        req = urllib.request.Request(url)
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end}")
        return urllib.request.urlopen(req, timeout=60).read()

    # The .idx gives the second message's offset, so message 1 is [0, that-1).
    idx = _get(f"{base}/{key}.idx").decode().splitlines()
    msg = _get(f"{base}/{key}", 0, int(idx[1].split(":")[1]) - 1)

    assert msg[:4] == b"GRIB", f"not a GRIB message: {msg[:4]!r}"
    p, sec3 = 16, None                       # 16 = past Section 0
    while p < len(msg) and msg[p:p + 4] != b"7777":
        ln = struct.unpack(">I", msg[p:p + 4])[0]
        if msg[p + 4] == 3:
            sec3 = msg[p:p + ln]
            break
        p += ln
    assert sec3 is not None, "no Section 3 in message"

    def u4(o):                               # octets are 1-indexed in the spec
        return struct.unpack(">I", sec3[o - 1:o + 3])[0]

    def s4(o):                               # sign-and-magnitude, not two's comp
        v = u4(o)
        return -(v & 0x7FFFFFFF) if v & 0x80000000 else v

    hdr = dict(template=struct.unpack(">H", sec3[12:14])[0],
               Ni=u4(31), Nj=u4(35),
               La1=s4(47) / 1e6, Lo1=s4(51) / 1e6,
               La2=s4(56) / 1e6, Lo2=s4(60) / 1e6,
               Di=u4(64) / 1e6, Dj=u4(68) / 1e6,
               scanning_mode=sec3[71], key=key)

    g = grid_of(era_for(date, run))
    assert hdr["template"] == 0, f"not a regular_ll grid: template {hdr['template']}"
    assert hdr["Ni"] == g["nx"] and hdr["Nj"] == g["ny"], hdr
    assert not hdr["scanning_mode"] & 0x80, f"i scans negatively: {hdr}"
    assert not hdr["scanning_mode"] & 0x40, f"j scans positively: {hdr}"
    assert abs(hdr["Di"] - g["dlon"]) < 1e-9, hdr
    assert abs(hdr["Dj"] - g["dlat"]) < 1e-9, hdr
    assert abs(hdr["Lo1"] - LON_ORIGIN) < 1e-6, \
        f"GEFS first longitude is {hdr['Lo1']}, expected {LON_ORIGIN} -- the " \
        f"grid origin changed and LON_ORIGIN must be revisited"
    assert abs(hdr["Lo2"] - (360.0 - g["dlon"])) < 1e-6, hdr
    assert hdr["La1"] == 90.0 and hdr["La2"] == -90.0, hdr
    return hdr
