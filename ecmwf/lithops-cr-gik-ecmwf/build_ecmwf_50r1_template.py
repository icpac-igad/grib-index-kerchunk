#!/usr/bin/env python3
"""
ECMWF 50r1 template builder — orchestrator.   [WIP — WRONG PACKAGING FORMAT]

!! 2026-05-17 hour-0 validation finding: the fixed_ensemble_grib_tree +
   extract_member_refs path used below produces a bare grib_tree skeleton
   (~375 rows/member, key style `asn/instant/surface/.zattrs`, NOT
   consolidated) — structurally DIFFERENT from the live production template
   (~2774 rows/member, consolidated zarr, key style
   `cape/entireAtmosphere/0.0/.zarray`). The real producer is the notebook
   path (fmrc_utils.s3_ecmwf_scan_grib_storing) + a consolidation step.
   The 50r1 scan/idx-join layer (ecmwf_scan_grib_50r1_one_pass) IS correct;
   only this packaging layer must be reworked onto the fmrc_utils format.
   Kept as scaffold + the validated one-pass wiring. DO NOT run the full
   85-hour build with this until the packaging layer is corrected.

Wires the validated one-pass dual-stream scanner
(`utils_ecmwf_step1_scangrib.ecmwf_scan_grib_50r1_one_pass`) into the exact
production template layout consumed by run_lithops_ecmwf.py:

    gik-fmrc/v2ecmwf_fmrc/{ens_control|ens_NN}/ecmwf-{REF}{run}-{m}-rt{HHH}.par

Per forecast hour (one pass over BOTH 50r1 streams):
  enfo/ef -> 50 perturbed (members 1..50)
  oper/fc -> control (member -1, superset schema)
    -> fixed_ensemble_grib_tree -> strip_datavar_chunks (deflated)
    -> extract_member_refs per member -> one rt{HHH}.par per (member, hour)

85 forecast hours x 51 members = 4,335 parquets (matches the live template).

Usage:
    # cheap end-to-end check: just hour 0 (1 enfo + 1 oper/fc scan ~15-20 min)
    uv run build_ecmwf_50r1_template.py --date 20260513 --run 00 --hours 0 \
        --out /tmp/tmpl50r1

    # full template (the ~2 h job; parallelise externally / Coiled)
    uv run build_ecmwf_50r1_template.py --date 20260513 --run 00 --out /tmp/tmpl50r1
"""
import os, sys, copy, time, argparse
from pathlib import Path

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

ECMWF_DIR = "/scratch/notebook/grib-index-kerchunk/ecmwf"
sys.path.insert(0, ECMWF_DIR)
sys.path.insert(0, os.path.join(ECMWF_DIR, "dev-test"))

from utils_ecmwf_step1_scangrib import (          # noqa: E402
    ecmwf_scan_grib_50r1_one_pass,
    fixed_ensemble_grib_tree,
)
from kerchunk._grib_idx import strip_datavar_chunks   # noqa: E402
from ecmwf_ensemble_par_creator_efficient import (  # noqa: E402  (import-safe: __main__ guarded)
    extract_member_refs,
    create_parquet_file_fixed,
)

HOURS_3H = list(range(0, 145, 3))      # 0..144 @3h  (49)
HOURS_6H = list(range(150, 361, 6))    # 150..360 @6h (36)
ALL_HOURS = HOURS_3H + HOURS_6H        # 85
MEMBERS = [-1] + list(range(1, 51))    # control + 50 perturbed


def _dirname(m):  return "ens_control" if m == -1 else f"ens_{m:02d}"
def _fname(m):    return "control" if m == -1 else f"ens{m:02d}"


def build_hour(date_str, run, hour, out_root, ref_date):
    """One pass for one hour -> 51 per-member rt{HHH}.par files."""
    t0 = time.time()
    groups, idx_maps = ecmwf_scan_grib_50r1_one_pass(date_str, run, hour)
    if not groups:
        print(f"[{hour:03d}h] no groups — skipped", flush=True)
        return []
    tree = fixed_ensemble_grib_tree(groups, remote_options={"anon": True})
    deflated = copy.deepcopy(tree)
    strip_datavar_chunks(deflated)
    refs = deflated["refs"]

    written = []
    for m in MEMBERS:
        mref = extract_member_refs(refs, m)
        if not mref:
            print(f"[{hour:03d}h] WARN no refs for member {m} ({_fname(m)})", flush=True)
            continue
        d = Path(out_root) / "gik-fmrc" / "v2ecmwf_fmrc" / _dirname(m)
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"ecmwf-{ref_date}{run}-{_fname(m)}-rt{hour:03d}.par"
        create_parquet_file_fixed(mref, str(out))
        written.append(str(out))
    print(f"[{hour:03d}h] {len(written)}/51 member parquets in "
          f"{time.time()-t0:.1f}s", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser(description="ECMWF 50r1 template builder")
    ap.add_argument("--date", required=True, help="scan date YYYYMMDD (>=20260513 for 50r1)")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--ref-date", default=None,
                    help="reference date embedded in filenames (default: --date)")
    ap.add_argument("--hours", default=None,
                    help="comma list of forecast hours (default: all 85)")
    ap.add_argument("--out", required=True, help="output root dir")
    args = ap.parse_args()

    ref_date = args.ref_date or args.date
    hours = ([int(h) for h in args.hours.split(",")] if args.hours else ALL_HOURS)

    print(f"50r1 template build: date={args.date} run={args.run}z "
          f"ref={ref_date} hours={len(hours)} -> {args.out}", flush=True)
    overall = time.time()
    total = 0
    for i, h in enumerate(hours, 1):
        try:
            total += len(build_hour(args.date, args.run, h, args.out, ref_date))
        except Exception as e:
            print(f"[{h:03d}h] ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
        print(f"  progress {i}/{len(hours)} hours, {total} parquets, "
              f"{(time.time()-overall)/60:.1f} min", flush=True)
    print(f"DONE: {total} parquets ({len(hours)} hours x 51) in "
          f"{(time.time()-overall)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
