#!/usr/bin/env python3
"""C3 -- the benchmark REALIZATION_COST.md §5 proposed and nobody ran.

One date, both paths, same box, same messages, same machine.  Times index
acquisition, fetch, decode and subset SEPARATELY, and cross-checks the values
so this doubles as a correctness test of the published store.

  path A (GIK) : icechunk virtual store -> ds[v].sel(...).values
  path B (Herbie): Herbie(...).download(search=...) -> cfgrib -> subset

Frisky's unit of work is exactly path A, 125,970 times per date.
"""
import os, sys, time, io, shutil, tempfile, statistics as st
for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)
sys.path.insert(0, os.environ.get("HERBIE_SRC", "herbie/src"))

import numpy as np, xarray as xr
from datetime import datetime

LAT_MAX, LAT_MIN, LON_MIN, LON_MAX = 40.0, -40.0, 15.0, 80.0
DATE_S, ERA = "2026-09-01", "50r1"
DATE = datetime(2026, 9, 1)

# (store_var, level, herbie_param, member, step_h)
# number 0 is the CONTROL: on 50r1 it is not in `enfo` at all (see 03_member_axis)
WORK = [
    ("t",   850, "t",  1, 24), ("t",   850, "t",  7, 24),
    ("t",   850, "t",  1, 48), ("q",   850, "q",  7, 24),
    ("t2m", None, "2t", 1, 24), ("t2m", None, "2t", 7, 48),
    ("msl", None, "msl", 3, 24), ("u10", None, "10u", 3, 24),
    ("t",   850, "t",  0, 24),               # <- control, the interesting one
]

def banner(s): print("\n" + "=" * 78); print(s); print("=" * 78)

# ----------------------------------------------------------------- path A ---
banner("path A -- GIK virtual icechunk store")
import icechunk, gribberish.zarr  # noqa: F401
t0 = time.perf_counter()
storage = icechunk.s3_storage(
    bucket="e4drr-project", prefix="forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd",
    endpoint_url="https://data.source.coop", region="us-east-1",
    anonymous=True, from_env=False, force_path_style=True)
auth = icechunk.containers_credentials(
    {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials()})
cfg = icechunk.RepositoryConfig.default()
cfg.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))
repo = icechunk.Repository.open(storage, config=cfg,
                                authorize_virtual_chunk_access=auth)
ds = xr.open_zarr(repo.readonly_session("main").store, group=f"{ERA}/00z",
                  consolidated=False, zarr_format=3, decode_timedelta=True,
                  chunks=None)
setup_a = time.perf_counter() - t0
print(f"  store open (once per worker process, then cached): {setup_a:6.2f} s")

A, ta = {}, []
for var, lvl, _p, mem, step in WORK:
    t0 = time.perf_counter()
    da = ds[var]
    if lvl is not None:
        da = da.sel(isobaricInhPa=lvl)
    da = da.sel(time=np.datetime64(DATE_S), number=mem,
                step=np.timedelta64(step, "h"),
                latitude=slice(LAT_MAX, LAT_MIN),
                longitude=slice(LON_MIN, LON_MAX))
    arr = np.asarray(da.values, dtype=np.float32)
    dt = time.perf_counter() - t0
    A[(var, lvl, mem, step)] = arr
    ta.append(dt)
    print(f"  {var:4s} lvl={str(lvl):4s} n={mem:2d} f{step:03d}  "
          f"{dt*1000:7.0f} ms   shape={arr.shape}")
print(f"  ---- median {st.median(ta)*1000:.0f} ms/message, "
      f"total {sum(ta):.1f} s for {len(ta)} messages")

# ----------------------------------------------------------------- path B ---
banner("path B -- Herbie")
import warnings; warnings.filterwarnings("ignore")
from herbie import Herbie
tmp = tempfile.mkdtemp(prefix="herbie-bench-")

def quiet(fn, *a, **k):
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        return fn(*a, **k)
    finally:
        sys.stdout = old

B, tb, t_ctor, t_idx, t_dl, t_dec = {}, [], [], [], [], []
hcache = {}
for var, lvl, param, mem, step in WORK:
    # 50r1: the control is NOT in enfo -- it must come from oper/fc.
    product = "enfo" if mem > 0 else "oper"
    key = (product, step)
    t0 = time.perf_counter()
    if key not in hcache:
        H = quiet(Herbie, DATE, model="ifs", product=product, fxx=step,
                  save_dir=tmp, overwrite=True, verbose=False)
        t_ctor.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        _ = quiet(lambda: H.index_as_dataframe)
        t_idx.append(time.perf_counter() - t1)
        hcache[key] = H
    H = hcache[key]

    lv = str(lvl) if lvl is not None else "sfc"
    srch = (rf":{param}:{lvl}:pl:{mem}:" if lvl is not None
            else rf":{param}:sfc:{mem}:")
    if mem == 0:                       # oper/fc rows carry no `number`
        srch = (rf":{param}:{lvl}:pl:" if lvl is not None else rf":{param}:sfc:")

    t2 = time.perf_counter()
    try:
        p = quiet(H.download, search=srch, overwrite=True, verbose=False)
        t_dl.append(time.perf_counter() - t2)
        t3 = time.perf_counter()
        d = xr.open_dataset(p, engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
        name = [n for n in d.data_vars][0]
        sub = d[name].sel(latitude=slice(LAT_MAX, LAT_MIN),
                          longitude=slice(LON_MIN, LON_MAX))
        arr = np.asarray(sub.values, dtype=np.float32)
        t_dec.append(time.perf_counter() - t3)
        dt = time.perf_counter() - t0
        B[(var, lvl, mem, step)] = arr
        tb.append(dt)
        print(f"  {var:4s} lvl={lv:4s} n={mem:2d} f{step:03d}  "
              f"{dt*1000:7.0f} ms   shape={arr.shape}  search={srch}")
    except Exception as exc:
        print(f"  {var:4s} lvl={lv:4s} n={mem:2d} f{step:03d}  "
              f"FAILED  {type(exc).__name__}: {str(exc)[:90]}  search={srch}")

if tb:
    print(f"  ---- median {st.median(tb)*1000:.0f} ms/message, "
          f"total {sum(tb):.1f} s for {len(tb)} messages")
print(f"  phase medians: ctor {st.median(t_ctor)*1000 if t_ctor else 0:.0f} ms | "
      f"index {st.median(t_idx)*1000 if t_idx else 0:.0f} ms | "
      f"download {st.median(t_dl)*1000 if t_dl else 0:.0f} ms | "
      f"decode+subset {st.median(t_dec)*1000 if t_dec else 0:.0f} ms")

# ------------------------------------------------------------ correctness ---
banner("C3b -- do the two paths agree?  (independent check of the store)")
for k in A:
    if k not in B:
        print(f"  {str(k):28s}  NO HERBIE RESULT -- path B could not fetch it")
        continue
    a, b = A[k], B[k]
    if a.shape != b.shape:
        print(f"  {str(k):28s}  shape {a.shape} vs {b.shape}")
        continue
    d = float(np.nanmax(np.abs(a - b)))
    print(f"  {str(k):28s}  max|diff| = {d:.6e}  {'OK' if d == 0 else ''}")

shutil.rmtree(tmp, ignore_errors=True)
