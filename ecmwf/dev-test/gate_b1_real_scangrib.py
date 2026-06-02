#!/usr/bin/env python3
"""Gate B1 — fix-validation against REAL scan_grib output.

Validates the per-pressure-level fix in `4ca1c21` by:
  1. scan_grib-ing one real ECMWF 49r1 GRIB from s3://ecmwf-forecasts,
     filtered to u-wind only to bound the runtime (~663 messages = 51
     members × 13 isobaric levels, instead of all 8500 for the full
     file).
  2. Feeding the result through the FIXED `fixed_ensemble_grib_tree()`
     in `utils_ecmwf_step1_scangrib.py`.
  3. Asserting the new path shape carries one zarr group per
     (var, stepType, isobaricInhPa, level) for u-wind, i.e. 13 distinct
     per-level groups under u/instant/isobaricInhPa/.

NB: this is fix-validation on real-shaped scan_grib data. The output
zarr-store shape is INTENTIONALLY different from the legacy template's
{var}/{typeOfLevel}/{N.0}/... shape (per Gate A0). The legacy shape was
broken; the new shape is what the synthesis MD §4 says consumers must
update to.

Run (allow ~10 min for the scan; cfgrib + eccodes are heavy):
    cd ecmwf/dev-test
    uv run --quiet \\
      --with "kerchunk==0.2.7" --with "zarr==2.18.7" \\
      --with "numcodecs==0.12.1" --with cfgrib --with eccodes \\
      --with s3fs --with fsspec --with pandas --with numpy \\
      gate_b1_real_scangrib.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# 49r1 reference date / hour the existing template was built on.
# ECMWF Open Data retains 2024-03 onwards per the user (2026-05-29).
DATE, RUN, HOUR = "20240529", "00", 0
SOURCE_GRIB = (f"s3://ecmwf-forecasts/{DATE}/{RUN}z/ifs/0p25/enfo/"
               f"{DATE}{RUN}0000-{HOUR}h-enfo-ef.grib2")

# 49r1 isobaric set (no 10 hPa — 50r1 added that as the 14th).
EXPECTED_LEVELS_49R1 = {50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
                        850, 925, 1000}


def chk(label, ok, *details):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        for d in details:
            print(f"         {d}")
    return ok


def main() -> int:
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

    # Make the `ecmwf` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ecmwf.utils_ecmwf_step1_scangrib import fixed_ensemble_grib_tree  # noqa: E402

    from kerchunk.grib2 import scan_grib

    print(f"Gate B1 — scan_grib on {SOURCE_GRIB}\n")
    print("  filter: shortName=u (u-wind only)  ~663 expected messages "
          "(51 members × 13 levels)")
    print("  this can take 5-15 min (one S3 byte-range read per message,"
          " latency-dominated)")

    t0 = time.time()
    groups = scan_grib(
        SOURCE_GRIB,
        storage_options={"anon": True},
        filter={"shortName": "u"},
    )
    elapsed = time.time() - t0
    print(f"\n  scan_grib done in {elapsed:.1f}s — {len(groups)} message groups\n")

    # Sanity: each group should have version=1, GRIB_typeOfLevel attr,
    # and an isobaricInhPa/0 BINARY BLOB carrying the level (f8).
    # Discovery 2026-05-29: cfgrib's scan_grib does NOT put GRIB_level
    # in dattrs; level lives in group["refs"]["isobaricInhPa/0"].
    import numpy as np
    levels_seen = Counter()
    members_seen = set()
    bad = 0
    for g in groups:
        if g.get("version") != 1:
            bad += 1
            continue
        try:
            zattrs = json.loads(g["refs"][".zattrs"])
            coords = zattrs["coordinates"].split(" ")
            vname = None
            for key in g["refs"]:
                name = key.split("/")[0]
                if name not in (".zattrs", ".zgroup") and name not in coords:
                    vname = name
                    break
            if vname is None:
                bad += 1
                continue
            dattrs = json.loads(g["refs"][f"{vname}/.zattrs"])
            if dattrs.get("GRIB_typeOfLevel") == "isobaricInhPa":
                # Decode level from binary blob (f8).
                level_raw = g["refs"].get("isobaricInhPa/0")
                if isinstance(level_raw, str):
                    arr = np.frombuffer(level_raw.encode("latin1"),
                                        dtype=np.float64)
                    if len(arr) == 1:
                        levels_seen[int(arr[0])] += 1
            # ensemble member from number/0 binary blob.
            num_raw = g["refs"].get("number/0")
            if isinstance(num_raw, str):
                arr = np.frombuffer(num_raw.encode("latin1"),
                                    dtype=np.int64)
                if len(arr) == 1:
                    members_seen.add(int(arr[0]))
        except Exception:
            bad += 1

    print(f"Sanity on scan_grib output:")
    print(f"  bad-shape groups: {bad}")
    print(f"  distinct isobaric levels seen: {sorted(levels_seen)}")
    print(f"  per-level counts (should be 51 for each): "
          f"{dict(sorted(levels_seen.items()))}")
    print(f"  distinct ensemble_member values from .zattrs: "
          f"{sorted(members_seen)[:5]}... ({len(members_seen)} total)")

    all_ok = True
    all_ok &= chk(
        "scan_grib output covers all 13 49r1 isobaric levels",
        set(levels_seen) == EXPECTED_LEVELS_49R1,
        f"missing: {EXPECTED_LEVELS_49R1 - set(levels_seen)}",
        f"extra:   {set(levels_seen) - EXPECTED_LEVELS_49R1}",
    )
    all_ok &= chk(
        "scan_grib output has ~51 messages per level (one per member)",
        all(c >= 50 for c in levels_seen.values()),
        f"per-level counts: {dict(sorted(levels_seen.items()))}",
    )

    # Drive through the FIXED aggregator.
    print("\nDriving scan_grib output through fixed_ensemble_grib_tree...")
    t0 = time.time()
    try:
        zarr_store = fixed_ensemble_grib_tree(groups, debug_output=False)
    except Exception as e:
        print(f"  FAIL  fixed_ensemble_grib_tree raised: "
              f"{type(e).__name__}: {e}")
        return 1
    elapsed = time.time() - t0
    print(f"  fixed_ensemble_grib_tree done in {elapsed:.1f}s")
    print(f"  zarr_store has {len(zarr_store)} keys")

    # Per-level paths under u/instant/isobaricInhPa/
    pl_zgroups = sorted({
        k.rsplit("/.zgroup", 1)[0] for k in zarr_store
        if k.startswith("u/instant/isobaricInhPa/") and k.endswith("/.zgroup")
    })
    pl_leaves = [p for p in pl_zgroups if p != "u/instant/isobaricInhPa"]
    seen_levels = sorted({int(p.split("/")[-1]) for p in pl_leaves
                          if p.split("/")[-1].isdigit()})

    print(f"\nFixed-aggregator output for u-wind:")
    print(f"  per-level zarr groups under u/instant/isobaricInhPa/: "
          f"{len(pl_leaves)}")
    print(f"  per-level path levels: {seen_levels}")

    all_ok &= chk(
        "13 per-level zarr groups under u/instant/isobaricInhPa/",
        len(pl_leaves) == 13,
        f"observed leaf paths ({len(pl_leaves)}): {pl_leaves}",
    )
    all_ok &= chk(
        "per-level levels match the 49r1 isobaric set",
        set(seen_levels) == EXPECTED_LEVELS_49R1,
        f"observed: {sorted(seen_levels)}",
        f"expected: {sorted(EXPECTED_LEVELS_49R1)}",
    )
    all_ok &= chk(
        "per-level count > 1 (pre-fix bug would yield exactly 1)",
        len(pl_leaves) > 1,
        f"pre-fix collapse symptom: {len(pl_leaves)}",
    )

    print()
    if all_ok:
        print("Gate B1 PASS — `fixed_ensemble_grib_tree` from 4ca1c21 emits "
              "13 distinct per-isobaric-level zarr groups when fed real "
              "scan_grib output from a 49r1 GRIB. The per-level fix is "
              "structurally sound end-to-end. Safe to proceed to the paid "
              "Step 1 Coiled preprocessing when ready.")
        return 0
    print("Gate B1 FAIL — investigate before paying for the ~1 h scan.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
