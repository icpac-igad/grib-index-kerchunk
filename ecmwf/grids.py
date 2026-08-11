"""Canonical ECMWF ENS open-data grid definitions -- the single source of truth.

Import this. Do not re-derive a coordinate axis anywhere else.

Why this file exists
--------------------
The ECMWF ENS GRIB2 messages are `regular_ll` with the scan starting at
**-180 degrees east**, not 0. eccodes on a real message reports:

    longitudeOfFirstGridPointInDegrees   180.0     (i.e. -180 in +/-180 terms)
    longitudeOfLastGridPointInDegrees    179.75
    iScansNegatively                     0
    jScansPositively                     0
    latitudeOfFirstGridPointInDegrees    90.0
    latitudeOfLastGridPointInDegrees     -90.0

Verified 2026-08-11 against all three eras (`verify_grid_geolocation.py grib`).

That fact had been written out by hand in twelve places across two repos. Eleven
were right; `build_ecmwf_icechunk.py` assumed a 0-start origin and wrote
`0, 0.25, ... 359.75`. Every Icechunk store built from it was labelled 180 deg
out, and every realized subset over East Africa silently read the eastern
Pacific instead. See `crma/medium-range-forecast/HANDOVER_LONGITUDE_FIX.md`.

The defect was not numerical -- it was a duplicated constant with nothing binding
the copies. So: one table, one pair of axis constructors, and `verify()` to
prove them against the GRIB whenever you care to.

Usage
-----
    from grids import ERAS, latitudes, longitudes, field_shape

    era = ERAS["49r1"]
    lat = latitudes("49r1")          # 90.0 ... -90.0   (721,)
    lon = longitudes("49r1")         # -180.0 ... 179.75 (1440,)

Deliberately dependency-light (numpy only) so PEP 723 `uv run` scripts can
`sys.path`-insert this directory and import it without adding a dependency.
"""
from __future__ import annotations

import numpy as np

__all__ = ["GRIDS", "ERAS", "LON_ORIGIN", "LAT_ORIGIN",
           "grid_of", "latitudes", "longitudes", "field_shape", "verify"]

# The origin. This is the fact that was wrong in the builder. It belongs here and
# nowhere else -- if you find yourself typing -180.0 into another file, import
# instead.
LON_ORIGIN = -180.0   # longitudeOfFirstGridPointInDegrees, +/-180 convention
LAT_ORIGIN = 90.0     # latitudeOfFirstGridPointInDegrees (grid scans N -> S)

# Physical grids. Two distinct grids across the whole archive; three eras map
# onto them.
GRIDS = {
    "0p25": dict(ny=721, nx=1440, dlat=0.25, dlon=0.25),
    "0p4":  dict(ny=451, nx=900,  dlat=0.4,  dlon=0.4),
}

# Era table: grid + canonical pressure-level SUPERSET. Dates carrying fewer
# levels simply leave those slots empty -> NaN on read, matching the template
# decision documented in CLAUDE.md ("ECMWF spans FOUR template eras").
ERAS = {
    "0p4":  dict(grid="0p4",
                 levels=[50, 200, 250, 300, 500, 700, 850, 925, 1000]),
    "49r1": dict(grid="0p25",
                 levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
                         850, 925, 1000]),
    "50r1": dict(grid="0p25",
                 levels=[10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
                         850, 925, 1000]),
}

# ny/nx/dlon/dlat are mirrored onto each era entry so existing callers that do
# `era["nx"]` keep working. The mirror is derived, never hand-written.
for _era in ERAS.values():
    _era.update(GRIDS[_era["grid"]])
del _era


# Era boundaries as inclusive (YYYYMMDD, run_hour) pairs. `None` means open.
#
# EVERY transition happens at 06z, mid-date -- verified 2026-08-11 by reading the
# pars themselves (grid from the referenced GRIB URL, era from the pl-level
# count) at each of the four runs across the three boundary dates:
#
#   20240228  00z 0p4/9L  |  06z 0p25/9L  12z 0p25/9L  18z 0p25/9L
#   20250114  00z 9L      |  06z 13L      12z 13L      18z 13L
#   20260512  00z 13L     |  06z 14L      12z 14L      18z 14L
#
# CLAUDE.md's era table is date-granular and so is wrong at the first edge: it
# puts 0p4's end at 2024-02-28 and 49r1's start at 2024-02-29, which is true only
# of 00z. 2024-02-28's 06z/12z/18z runs are already 0p25/49r1. That never mattered
# while only 00z was ingested; with all four runs it decides where three days of
# data belong.
#
# The 9L -> 13L change inside 49r1 is NOT a boundary here: both live in one 49r1
# group built on the 13-level superset, so 9-level dates simply leave the four
# extra levels empty.
ERA_BOUNDS = {
    "0p4":  (("20230118", 0), ("20240228", 0)),
    "49r1": (("20240228", 6), ("20260512", 0)),
    "50r1": (("20260512", 6), None),
}

# 00z/12z run to 360h, 06z/18z stop at 144h. Mirrors STEPS_BY_RUN in the builder.
LONG_RUNS = (0, 12)


def era_for(date: str, run: int | str) -> str:
    """Which era a (YYYYMMDD, run) initialisation belongs to."""
    k = (date, int(str(run).rstrip("zZ")))
    for name, (lo, hi) in ERA_BOUNDS.items():
        if (lo is None or k >= lo) and (hi is None or k <= hi):
            return name
    raise KeyError(f"{date} {run}z is outside every known era window")


def grid_of(era: str) -> dict:
    """Grid dict for an era name ('49r1') or a grid name ('0p25')."""
    if era in ERAS:
        return GRIDS[ERAS[era]["grid"]]
    if era in GRIDS:
        return GRIDS[era]
    raise KeyError(f"unknown era or grid {era!r}; "
                   f"eras={sorted(ERAS)} grids={sorted(GRIDS)}")


def latitudes(era: str) -> np.ndarray:
    """Latitude axis, north to south: 90.0 ... -90.0."""
    g = grid_of(era)
    return np.linspace(LAT_ORIGIN, LAT_ORIGIN - (g["ny"] - 1) * g["dlat"],
                       g["ny"], dtype="float64")


def longitudes(era: str) -> np.ndarray:
    """Longitude axis in the GRIB's own scan order: -180.0 ... +179.75.

    Monotonic ascending, so `.sel(longitude=slice(15, 80))` works directly and
    no `sortby`/`roll` is ever needed. THIS is the line that was wrong.
    """
    g = grid_of(era)
    return LON_ORIGIN + np.arange(g["nx"], dtype="float64") * g["dlon"]


def field_shape(era: str) -> tuple[int, int]:
    """(ny, nx) of one global field for this era."""
    g = grid_of(era)
    return g["ny"], g["nx"]


# ---------------------------------------------------------------------------
# Cheap invariants, checked at import. These cost microseconds and would have
# caught the original defect the moment the module was loaded.
# ---------------------------------------------------------------------------
def _self_check() -> None:
    for name in ERAS:
        g, lat, lon = grid_of(name), latitudes(name), longitudes(name)
        assert lat.shape == (g["ny"],) and lon.shape == (g["nx"],)
        assert lat[0] == 90.0 and lat[-1] == -90.0, f"{name}: latitude span"
        # A global longitude axis must start at -180 and stop one increment
        # short of +180 (the +180 column is the same meridian as -180).
        assert lon[0] == -180.0, f"{name}: longitude must start at -180"
        assert abs(lon[-1] - (180.0 - g["dlon"])) < 1e-9, \
            f"{name}: longitude must end at 180 - dlon, got {lon[-1]}"
        assert np.all(np.diff(lon) > 0), f"{name}: longitude not ascending"
        # Nairobi must be in the eastern half of the axis, not the western.
        assert lon[int(np.abs(lon - 36.8).argmin())] > 0, f"{name}: sign error"


_self_check()


def verify(era: str = "49r1", date: str = "20250515", run: str = "00") -> dict:
    """Assert this table against a real GRIB message header (needs network).

    Returns the header fields read. Raises AssertionError on any mismatch.
    Requires `eccodes` and `s3fs`; kept out of module import so the builders
    never depend on them. `verify_grid_geolocation.py` is the CLI wrapper.
    """
    import json
    import os

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    import eccodes
    import s3fs

    seg = "0p4-beta" if era == "0p4" else "ifs/0p25"
    stem = f"{date}{run}0000"
    path = f"ecmwf-forecasts/{date}/{run}z/{seg}/enfo/{stem}-0h-enfo-ef.grib2"

    fs = s3fs.S3FileSystem(anon=True)
    rec = json.loads(fs.cat(path.replace(".grib2", ".index")).decode()
                     .splitlines()[0])
    buf = fs.read_block(path, int(rec["_offset"]), int(rec["_length"]))
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
    finally:
        eccodes.codes_release(h)

    g = grid_of(era)
    assert hdr["gridType"] == "regular_ll", hdr["gridType"]
    assert hdr["Ni"] == g["nx"] and hdr["Nj"] == g["ny"], hdr
    assert hdr["iScansNegatively"] == 0 and hdr["jScansPositively"] == 0, hdr
    assert abs(hdr["iDirectionIncrementInDegrees"] - g["dlon"]) < 1e-9, hdr
    assert abs(hdr["jDirectionIncrementInDegrees"] - g["dlat"]) < 1e-9, hdr
    # eccodes normalises the header to 0-360, so -180 is reported as 180.0.
    first = hdr["longitudeOfFirstGridPointInDegrees"]
    assert min(abs(first - 180.0), abs(first + 180.0)) < 1e-6, \
        f"{era}: GRIB first longitude is {first}, expected +/-180 -- the grid " \
        f"origin changed and LON_ORIGIN must be revisited"
    assert hdr["latitudeOfFirstGridPointInDegrees"] == 90.0, hdr
    assert hdr["latitudeOfLastGridPointInDegrees"] == -90.0, hdr
    return hdr
