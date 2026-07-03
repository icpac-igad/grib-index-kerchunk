#!/usr/bin/env python3
"""
Dry-run: does the production scan_grib -> grib_tree -> strip_datavar_chunks
Stage-1 path (ecmwf_util.py logic) survive the 50r1 structure?

Faithful replica of ecmwf_util.py:
  ECMWF_CONTROL_NUMBER=-1; ECMWF_ALL_NUMBERS=[-1,1..50]
  identify_ensemble_members(): number from attrs, normalize 0->-1
  fixed_ensemble_grib_tree(): iterate ECMWF_ALL_NUMBERS, grib_tree(found)
  prepare_ecmwf_zarr_store(): strip_datavar_chunks(deepcopy)
Target: 2026-05-13 00z 0h (first full-85-step 50r1 date).
"""
import sys, time, copy, collections, json
from kerchunk.grib2 import grib_tree, scan_grib
from kerchunk._grib_idx import strip_datavar_chunks

URL = "s3://ecmwf-forecasts/20260513/00z/ifs/0p25/enfo/20260513000000-0h-enfo-ef.grib2"
ECMWF_ALL_NUMBERS = [-1] + list(range(1, 51))

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

t0 = time.time()
log(f"scan_grib START {URL}")
groups = scan_grib(URL, storage_options={"anon": True})
log(f"scan_grib DONE: {len(groups)} groups in {time.time()-t0:.1f}s")

# identify_ensemble_members (exact production logic)
member_groups = {}
no_number = 0
for g in groups:
    attrs = g.get("attrs", {})
    number = attrs.get("number", None)
    if number is None:
        number = attrs.get("perturbationNumber", None)
    if number is None:
        no_number += 1
        continue
    if number == 0:
        number = -1
    member_groups.setdefault(number, []).append(g)

found = sorted(member_groups)
log(f"members found: {len(found)}  range=({found[0] if found else None}..{found[-1] if found else None})")
log(f"  control(-1) present: {-1 in member_groups}   groups w/o number: {no_number}")
log(f"  ECMWF_ALL_NUMBERS expects 51 incl -1; MISSING: {[n for n in ECMWF_ALL_NUMBERS if n not in member_groups]}")

# fixed_ensemble_grib_tree
all_member_groups = []
for num in ECMWF_ALL_NUMBERS:
    if num in member_groups:
        all_member_groups.extend(member_groups[num])
log(f"building grib_tree from {len(all_member_groups)} member-groups...")
ts = time.time()
tree = grib_tree(all_member_groups)
log(f"grib_tree DONE: {len(tree.get('refs',{}))} refs in {time.time()-ts:.1f}s")

# prepare_ecmwf_zarr_store
deflated = copy.deepcopy(tree)
strip_datavar_chunks(deflated)
refs = deflated["refs"]
log(f"deflated store: {len(refs)} refs")

# Does the new 10 hPa pressure level flow through?
lvls = collections.Counter()
for k, v in refs.items():
    if k.endswith("/isobaricInhPa/0") or k.endswith("/isobaricInhPa"):
        pass
isob = sorted({k.split("/")[ -2] for k in refs if "isobaricInhPa" in k}) if any("isobaricInhPa" in k for k in refs) else []
# pull actual level values if present
lvl_vals = None
for k in refs:
    if k.endswith("isobaricInhPa/0"):
        try:
            import numpy as np
            lvl_vals = np.frombuffer(refs[k].encode("latin1") if isinstance(refs[k], str) else refs[k], dtype="<f8")
        except Exception as e:
            lvl_vals = f"decode-fail {e}"
        break
top_vars = sorted({k.split("/")[0] for k in refs if not k.startswith(".")})[:40]
log(f"isobaricInhPa level values: {lvl_vals}")
log(f"top-level vars ({len(set(k.split('/')[0] for k in refs if not k.startswith('.')))}): {top_vars}")
log(f"TOTAL dry-run time {time.time()-t0:.1f}s")
log("RESULT: members=%d control=%s tenhPa=%s" % (
    len(found), (-1 in member_groups),
    (lvl_vals is not None and not isinstance(lvl_vals, str) and 10.0 in list(lvl_vals)) if lvl_vals is not None else "n/a"))
