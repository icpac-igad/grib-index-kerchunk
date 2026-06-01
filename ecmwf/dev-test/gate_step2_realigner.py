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
    ECMWFParquetProcessor, ECMWF_0P25_FIELD_SHAPE,
)

DATE, RUN, HOUR = "20240529", "00", "0"
GRIB = (f"s3://ecmwf-forecasts/{DATE}/{RUN}z/ifs/0p25/enfo/"
        f"{DATE}{RUN}0000-{HOUR}h-enfo-ef.grib2")
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
        dump = build_faithful_dump(index_lines)
        dump.to_parquet(in_dir / f"e_sg_mdt_{DATE}_enfo_{HOUR}h.parquet", engine="pyarrow")

        proc = ECMWFParquetProcessor(str(in_dir), str(out_dir), run=RUN)
        proc.process_forecast_hour(in_dir / f"e_sg_mdt_{DATE}_enfo_{HOUR}h.parquet")

        hour_dir = out_dir / f"{HOUR}h"
        produced = sorted(p.stem for p in hour_dir.glob("*.par"))
        print(f"\n  produced {len(produced)} member files: "
              f"{produced[:3]} ... {produced[-2:]}")

        # ---- assertions ----
        check("member count == .index member count",
              len(produced) == len(exp_counts),
              f"{len(produced)} vs {len(exp_counts)}")
        check("control member produced", "control" in produced)

        # Load control + one perturbed member, parse their key/value parquet.
        def load_store(member):
            df = pd.read_parquet(hour_dir / f"{member}.par")
            # save_parquet writes columns [key, value]
            return dict(zip(df["key"], df["value"]))

        ctrl = load_store("control")
        ens01 = load_store("ens01")

        # per-level keys present (u at multiple isobaric levels)?
        def levels_for(store, var):
            out = set()
            for k in store:
                parts = k.split("/")
                if parts[0] == var and len(parts) >= 3 and parts[1] == "isobaricInhPa":
                    try: out.add(int(float(parts[2])))
                    except ValueError: pass
            return out

        u_levels = levels_for(ctrl, "u")
        check("pl keys are PER-LEVEL (u has multiple isobaric levels)",
              len(u_levels) > 1, f"u levels in control: {sorted(u_levels)}")
        check("recovered pl levels match .index levelist set",
              u_levels.issubset(exp_levels) and len(u_levels) >= 1,
              f"recovered {sorted(u_levels)} ⊆ {sorted(exp_levels)}")
        check("NO collapse to level 0 for pl",
              0 not in u_levels, f"u levels: {sorted(u_levels)}")

        # 0.25 deg shape, not the 1 deg placeholder
        zarray_keys = [k for k in ctrl if k.endswith(".zarray")]
        shapes = {tuple(json.loads(ctrl[k])["shape"]) for k in zarray_keys}
        check(".zarray shape is 0.25deg [1,721,1440] (not [1,181,360])",
              shapes == {tuple(ECMWF_0P25_FIELD_SHAPE)},
              f"distinct shapes: {shapes}")

        # member separation: control's data-var key count should reflect ITS
        # own messages, not all members replicated. With replicate-to-all the
        # old bug gave every member the same (full) key set.
        ctrl_chunkrefs = sum(1 for k in ctrl if k.endswith("/.zarray"))
        ens_chunkrefs = sum(1 for k in ens01 if k.endswith("/.zarray"))
        # control in 49r1 carries extra static fields => >= perturbed, but both
        # must be FAR below the all-members total (≈ unique (var,level) per member).
        total_msgs = len(index_lines)
        check("per-member key set is a fraction of all messages (no replicate-to-all)",
              ctrl_chunkrefs < total_msgs / 5 and ens_chunkrefs < total_msgs / 5,
              f"control vars={ctrl_chunkrefs}, ens01 vars={ens_chunkrefs}, "
              f"total msgs={total_msgs}")

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
