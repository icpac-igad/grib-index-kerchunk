#!/usr/bin/env python3
"""
GATE: Step 2 realigner per-level / per-member fix
(see ../lithops-cr-gik-ecmwf/2026-06-01-step2-realigner-fix-scope.md).

Runs the REAL patched ECMWFParquetProcessor against a faithful scan-dump
parquet + the REAL live .index for the 49r1 reference file. The dump is
built so that pl rows carry level=0 (simulating an old-kerchunk dump that
never surfaced the level) — so a PASS proves the realigner recovers the
true level from the .index `levelist`, reconciles the 1-based-dump vs
0-based-index off-by-one, assigns each message to exactly one member
(no replicate-to-all), and stamps the real 0.25 deg [1,721,1440] shape.

No scan_grib, no Coiled, no cost: only an anonymous S3 read of the small
.index file. Free to run.

Usage:
    python gate_step2_realigner.py
"""
import os, sys, json, tempfile, shutil
from pathlib import Path

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import fsspec
from ecmwf_par_to_ensemble_members import (
    ECMWFParquetProcessor, ECMWF_0P25_FIELD_SHAPE, ECMWF_0P4_FIELD_SHAPE,
)

import argparse
_ap = argparse.ArgumentParser(description="Step 2 realigner structural gate")
_ap.add_argument("--date", default="20240529", help="YYYYMMDD (default 49r1 ref 20240529)")
_ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
_ap.add_argument("--hour", default="0", help="forecast hour (default 0; use a non-0h to exercise full level coverage)")
_ap.add_argument("--stream", default="enfo", choices=["enfo", "oper"],
                 help="enfo=perturbed (49r1 also bundles control); oper=50r1 control")
_ap.add_argument("--res", default="0p25", choices=["0p25", "0p4"],
                 help="0p25=ifs/0p25 (>=2024-02-29); 0p4=0p4-beta archive (2023-01-18..2024-02-28)")
_args = _ap.parse_args()
DATE, RUN, HOUR, STREAM, RES = _args.date, _args.run, _args.hour, _args.stream, _args.res

# Resolution -> path prefix and expected per-message field shape.
_PREFIX = "0p4-beta" if RES == "0p4" else "ifs/0p25"
EXPECTED_SHAPE = ECMWF_0P4_FIELD_SHAPE if RES == "0p4" else ECMWF_0P25_FIELD_SHAPE
_suffix = "oper-fc" if STREAM == "oper" else "enfo-ef"
GRIB = (f"s3://ecmwf-forecasts/{DATE}/{RUN}z/{_PREFIX}/{STREAM}/"
        f"{DATE}{RUN}0000-{HOUR}h-{_suffix}.grib2")
INDEX = GRIB.replace(".grib2", ".index")

# levtype -> the typeOfLevel name scan_grib would put in the dump row
LEVTYPE_TO_TOL = {"pl": "isobaricInhPa", "sfc": "surface", "sol": "depthBelowLandLayer"}

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def build_faithful_dump(index_lines):
    """One row per .index message, 1-based `idx` index (mirrors
    enumerate(scan_grib, start=1)). pl rows get level=0 on purpose."""
    rows = []
    for i, ln in enumerate(index_lines):
        d = json.loads(ln.strip().rstrip(","))
        lt = d.get("levtype")
        rows.append({
            "idx": i + 1,                                   # 1-based, like the real dump
            "varname": d.get("param", "unknown"),
            "typeOfLevel": LEVTYPE_TO_TOL.get(lt, lt or "unknown"),
            "stepType": "instant",
            "name": d.get("param", ""),
            "paramId": 0,
            "level": 0,                                     # FORCE recovery from .index
            "uri": GRIB,                                    # real dumps carry this; drives resolution + .index
        })
    return pd.DataFrame(rows).set_index("idx")


def expected_from_index(index_lines):
    """Ground truth: per-member message counts and the pl level set."""
    from collections import defaultdict
    counts = defaultdict(int)
    pl_levels = set()
    for ln in index_lines:
        d = json.loads(ln.strip().rstrip(","))
        num = d.get("number")
        num = -1 if num is None else int(num)
        counts[num] += 1
        if d.get("levtype") == "pl" and d.get("levelist") is not None:
            pl_levels.add(int(d["levelist"]))
    return dict(counts), pl_levels


def main():
    print(f"GATE: Step 2 realigner fix\n  index: {INDEX}")
    fs = fsspec.filesystem("s3", anon=True)
    index_lines = fs.cat(INDEX).decode().splitlines()
    print(f"  .index messages: {len(index_lines)}")

    exp_counts, exp_levels = expected_from_index(index_lines)
    print(f"  expected members: {len(exp_counts)} (control={-1 in exp_counts}); "
          f"pl levels: {sorted(exp_levels)}")

    workdir = Path(tempfile.mkdtemp(prefix="gate_step2_"))
    try:
        in_dir = workdir / "in"; in_dir.mkdir()
        out_dir = workdir / "out"
        fname = f"e_sg_mdt_{DATE}_{STREAM}_{HOUR}h.parquet"
        dump = build_faithful_dump(index_lines)
        dump.to_parquet(in_dir / fname, engine="pyarrow")

        proc = ECMWFParquetProcessor(str(in_dir), str(out_dir), run=RUN)
        proc.process_forecast_hour(in_dir / fname)

        hour_dir = out_dir / f"{HOUR}h"
        produced = sorted(p.stem for p in hour_dir.glob("*.par"))
        print(f"\n  produced {len(produced)} member files: "
              f"{produced[:3]} ... {produced[-2:]}")

        # ---- assertions ----
        check("member count == .index member count",
              len(produced) == len(exp_counts),
              f"{len(produced)} vs {len(exp_counts)}")
        has_control = -1 in exp_counts
        check("control present iff .index has it",
              ("control" in produced) == has_control,
              f"control in output={'control' in produced}, .index has control={has_control}")

        # Load the key/value parquet for a member.
        def load_store(member):
            df = pd.read_parquet(hour_dir / f"{member}.par")
            return dict(zip(df["key"], df["value"]))

        # Primary member to inspect: control if this stream carries it, else ens01.
        primary = "control" if has_control else "ens01"
        store = load_store(primary)

        def levels_for(s, var):
            out = set()
            for k in s:
                parts = k.split("/")
                if parts[0] == var and len(parts) >= 3 and parts[1] == "isobaricInhPa":
                    try: out.add(int(float(parts[2])))
                    except ValueError: pass
            return out

        u_levels = levels_for(store, "u")
        check("pl keys are PER-LEVEL (u has multiple isobaric levels)",
              len(u_levels) > 1, f"u levels in {primary}: {sorted(u_levels)}")
        check(f"FULL level coverage: u carries ALL {len(exp_levels)} published levels",
              u_levels == exp_levels,
              f"u={sorted(u_levels)} vs published={sorted(exp_levels)}")
        check("recovered pl levels are a subset of .index levelist set",
              u_levels.issubset(exp_levels) and len(u_levels) >= 1,
              f"recovered {sorted(u_levels)} ⊆ {sorted(exp_levels)}")
        check("NO collapse to level 0 for pl",
              0 not in u_levels, f"u levels: {sorted(u_levels)}")

        # resolution-correct field shape, not the legacy 1 deg placeholder
        zarray_keys = [k for k in store if k.endswith(".zarray")]
        shapes = {tuple(json.loads(store[k])["shape"]) for k in zarray_keys}
        check(f".zarray shape is {RES} {EXPECTED_SHAPE} (not [1,181,360])",
              shapes == {tuple(EXPECTED_SHAPE)},
              f"distinct shapes: {shapes}")

        # member separation: a member's (var,level) group count must derive
        # only from ITS OWN messages — never the all-members union (the old
        # replicate-to-all bug gave every member the full set).
        prim_vars = sum(1 for k in store if k.endswith("/.zarray"))
        total_msgs = len(index_lines)
        prim_member = -1 if has_control else 1
        own_msgs = exp_counts[prim_member]
        # groups can't exceed this member's own message count...
        ok_own = prim_vars <= own_msgs
        # ...and with >1 member, must be strictly below the all-members total
        # (single-member streams like oper/fc legitimately have own_msgs≈total).
        ok_sep = (prim_vars < total_msgs) if len(exp_counts) > 1 else True
        check("per-member key set derives from own messages only (no replicate-to-all)",
              ok_own and ok_sep,
              f"{primary} groups={prim_vars}, own msgs={own_msgs}, "
              f"total msgs={total_msgs}, members={len(exp_counts)}")

        print()
        if all(results):
            print(f"GATE {PASS}: {sum(results)}/{len(results)} checks passed")
            return 0
        print(f"GATE {FAIL}: {sum(results)}/{len(results)} checks passed")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
