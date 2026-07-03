#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gcsfs"]
# ///
"""Download sample re-baked par files for 20260301 00z to par_local/."""
import os, sys
import gcsfs

KEY = '/scratch/notebook/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json'
BASE = 'gik-ecmwf-aws-tf/run_par_ecmwf/2026/03/20260301/00z/'
DST = os.path.join(os.path.dirname(__file__), 'par_local')
os.makedirs(DST, exist_ok=True)

fs = gcsfs.GCSFileSystem(token=KEY)
files = sorted(fs.ls(BASE))
print('available members:', len(files))

want = sys.argv[1] if len(sys.argv) > 1 else 'sample'
if want == 'all':
    sel = files
else:
    sel = [f for f in files if any(m in f for m in ['control', 'ens_01.', 'ens_02.'])]

for f in sel:
    dst = os.path.join(DST, f.split('/')[-1])
    fs.get(f, dst)
    print('downloaded', dst, os.path.getsize(dst), 'bytes')
print('DONE', len(sel), 'files ->', DST)
