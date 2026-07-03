#!/usr/bin/env python3
"""Cheap index-only validation of the 50r1 builder restructure.

NO scan_grib. Exercises the pure idx-join logic of
ecmwf/utils_ecmwf_step1_scangrib.py against the real 2026-05-13 00z 0h
.index files for both 50r1 streams.
"""
import sys, numpy as np
sys.path.insert(0, "/scratch/notebook/grib-index-kerchunk/ecmwf")
import utils_ecmwf_step1_scangrib as U

OK = True
def check(name, cond, detail=""):
    global OK
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <-- {detail}"))
    OK = OK and cond

# 1) member-expansion seed
d_def = U.ecmwf_duplicate_dict_ens_mem({"x": {"param":"t","levtype":"sfc","ens_number":1,"levelist":"null"}})
mem_def = sorted({v["ens_number"] for k,v in d_def.items() if k.startswith("x_ens")})
check("perturbed default = 1..50, no control", mem_def == list(range(1,51)), mem_def[:3])
d_ctl = U.ecmwf_duplicate_dict_ens_mem({"x": {"param":"t","levtype":"sfc","ens_number":-1,"levelist":"null"}},
                                       ens_numbers=np.array([-1]))
mem_ctl = sorted({v["ens_number"] for k,v in d_ctl.items() if k.startswith("x_ens")})
check("control seed = [-1] only", mem_ctl == [-1], mem_ctl)

B = "s3://ecmwf-forecasts/20260513/00z/ifs/0p25"
ENFO = f"{B}/enfo/20260513000000-0h-enfo-ef.grib2"
OPER = f"{B}/oper/20260513000000-0h-oper-fc.grib2"

# 2) perturbed enfo: idx-join -> members 1..50, includes 10 hPa
imap_p, cdict_p = U.ecmwf_idx_df_create_with_keys(ENFO, is_control=False)
pmembers = sorted({int(k.split("ens")[-1]) for k in imap_p.values() if "ens" in k})
check("enfo idx-join members == 1..50", pmembers == list(range(1,51)), f"got {len(pmembers)}: {pmembers[:5]}..")
has10_p = any(v.get("levtype")=="pl" and str(v.get("levelist"))=="10" for v in cdict_p.values())
check("enfo combined_dict includes 10 hPa", has10_p)

# 3) control oper/fc: idx-join -> single control (-1), includes 10 hPa + z/sdor/slor
imap_c, cdict_c = U.ecmwf_idx_df_create_with_keys(OPER, is_control=True)
cmembers = sorted({int(k.split("ens")[-1]) for k in imap_c.values() if "ens" in k})
check("oper/fc idx-join member == [-1] only", cmembers == [-1], cmembers)
has10_c = any(v.get("levtype")=="pl" and str(v.get("levelist"))=="10" for v in cdict_c.values())
check("oper/fc combined_dict includes 10 hPa", has10_c)
params_c = {v["param"] for v in cdict_c.values()}
check("oper/fc control carries superset z/sdor/slor",
      {"z","sdor","slor"}.issubset(params_c), sorted(params_c & {"z","sdor","slor"}))

# 4) organize_ensemble_tree number-coord sizing (no scan needed: minimal tree)
tree = {"refs": {".zgroup": '{"zarr_format":2}'}}
pt = U.organize_ensemble_tree(tree)                                   # default perturbed
ct = U.organize_ensemble_tree(tree, member_numbers=np.array([-1]))    # control
import json
pz = json.loads(pt["refs"]["number/.zarray"]); cz = json.loads(ct["refs"]["number/.zarray"])
pnum = np.frombuffer(pt["refs"]["number/0"].encode("latin1"), dtype="<i8")
cnum = np.frombuffer(ct["refs"]["number/0"].encode("latin1"), dtype="<i8")
check("perturbed number coord shape==[50] & values 1..50",
      pz["shape"]==[50] and list(pnum)==list(range(1,51)), pz["shape"])
check("control number coord shape==[1] & value [-1]",
      cz["shape"]==[1] and list(cnum)==[-1], (cz["shape"], list(cnum)))

print("\n" + ("ALL PASS" if OK else "FAILURES PRESENT"))
sys.exit(0 if OK else 1)
