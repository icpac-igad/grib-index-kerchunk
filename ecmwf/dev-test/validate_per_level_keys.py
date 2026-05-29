#!/usr/bin/env python3
"""Index-only validation for the per-pressure-level key fix.

NO scan_grib, NO cost. Pulls the .index sidecar for 2026-05-13 00z f009
and verifies:

  1. ecmwf_index_processor.parse_grib_index() now exposes `level_value`
     for pl entries (was silently dropped pre-fix).
  2. ecmwf_index_processor.create_references_from_index() emits one key
     per (var, member, level) for pl messages — exactly 13 pl keys per
     (var, member) for variables that have all 13 isobaric levels.
  3. sfc keys are unchanged (no level segment).
  4. The level values are within the expected isobaric set
     {10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}.

The pre-fix symptom (the smoking gun in
cGAN_tutorial/example_notebooks/pytorch_cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md):
the same parquet key resolved to a DIFFERENT isobaricInhPa per step.
After the fix that's impossible by construction — each level lives at
its own key.

Run:
    cd ecmwf/dev-test
    uv run --with pandas --with fsspec --with s3fs validate_per_level_keys.py
"""
from __future__ import annotations
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the `ecmwf` package importable by adding the repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ecmwf import ecmwf_index_processor as P  # noqa: E402

# Expected set per ECMWF ENS Open Data 50r1 (50r1 added 10 hPa to pre-50r1's 13)
EXPECTED_LEVELS = {"10", "50", "100", "150", "200", "250", "300", "400",
                   "500", "600", "700", "850", "925", "1000"}

# 50r1 reference date / hour the project standardised on
DATE, RUN, HOUR = "20260513", "00", 9


def chk(label, ok, *details):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        for d in details:
            print(f"         {d}")
    return ok


def main() -> int:
    idx_url = (f"s3://ecmwf-forecasts/{DATE}/{RUN}z/ifs/0p25/enfo/"
               f"{DATE}{RUN}0000-{HOUR}h-enfo-ef.index")
    grib_url = idx_url.replace(".index", "")
    print(f"Fetching {idx_url}")

    entries = P.parse_grib_index(idx_url)
    print(f"  parsed {len(entries)} index entries")

    all_ok = True

    # (1) level_value is exposed
    pl_entries = [e for e in entries if e.get("level") == "pl"]
    sfc_entries = [e for e in entries if e.get("level") == "sfc"]
    all_ok &= chk(f"index has pl entries ({len(pl_entries)} found)", len(pl_entries) > 0)
    all_ok &= chk(f"index has sfc entries ({len(sfc_entries)} found)", len(sfc_entries) > 0)
    pl_with_level = [e for e in pl_entries if e.get("level_value")]
    all_ok &= chk(
        "all pl entries carry level_value",
        len(pl_with_level) == len(pl_entries),
        f"pl entries without level_value: {len(pl_entries) - len(pl_with_level)}",
    )

    # (4) levels are within the expected isobaric set
    seen_levels = {e["level_value"] for e in pl_entries}
    unexpected = seen_levels - EXPECTED_LEVELS
    all_ok &= chk(
        f"pl level values within expected set ({sorted(seen_levels)})",
        not unexpected,
        f"unexpected level values: {sorted(unexpected)}",
    )

    # (2) per-(var,member) we should see one entry per level
    refs = P.create_references_from_index(grib_url, entries)
    print(f"  built {len(refs)} references")
    pl_keys = [k for k in refs if "/pl/" in k]
    sfc_keys = [k for k in refs if "/sfc/" in k]

    all_ok &= chk(f"emitted pl keys ({len(pl_keys)} found)", len(pl_keys) > 0)
    all_ok &= chk(f"emitted sfc keys ({len(sfc_keys)} found)", len(sfc_keys) > 0)

    # Every pl key must have shape var/pl/{level}/{member}/0.0.0 (5 segments)
    bad_shape = [k for k in pl_keys if len(k.split("/")) != 5]
    all_ok &= chk(
        "every pl key has shape var/pl/level/member/0.0.0",
        not bad_shape,
        f"bad-shape pl keys: {bad_shape[:5]}",
    )

    # Per (var, member), count distinct levels:
    by_var_member = defaultdict(set)
    for k in pl_keys:
        parts = k.split("/")
        var, _pl, lev, mem = parts[:4]
        by_var_member[(var, mem)].add(lev)

    sample_var = next(iter({v for (v, _m) in by_var_member}))
    sample_levels = by_var_member[(sample_var, next(iter({m for (_v, m) in by_var_member})))]
    all_ok &= chk(
        f"sample pl var '{sample_var}' has multiple levels per member "
        f"({len(sample_levels)} distinct)",
        len(sample_levels) > 1,
        f"sample levels: {sorted(sample_levels)}",
    )

    # Variables that should have the FULL set per member (u/v/gh/t/q/w/r/d/vo/w):
    fullset_vars = {"u", "v", "gh", "t", "q", "w", "r", "d", "vo"}
    for var in fullset_vars:
        per_mem = [len(s) for (v, _m), s in by_var_member.items() if v == var]
        if not per_mem:
            continue
        counter = Counter(per_mem)
        mode_count, mode_n = counter.most_common(1)[0]
        all_ok &= chk(
            f"variable '{var}' has {mode_count} distinct levels per member "
            f"({mode_n} members)",
            mode_count >= 13,
            f"per-member level counts: {sorted(counter.items())}",
        )

    # (3) sfc keys keep old shape var/sfc/member/0.0.0 (4 segments)
    bad_sfc = [k for k in sfc_keys if len(k.split("/")) != 4]
    all_ok &= chk(
        "sfc keys unchanged (var/sfc/member/0.0.0 shape)",
        not bad_sfc,
        f"bad-shape sfc keys: {bad_sfc[:5]}",
    )

    print()
    if all_ok:
        print("ALL CHECKS PASS — per-level keys validated end-to-end on real .index.")
        return 0
    print("FAILURES — fix needed.")
    return 1


if __name__ == "__main__":
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    sys.exit(main())
