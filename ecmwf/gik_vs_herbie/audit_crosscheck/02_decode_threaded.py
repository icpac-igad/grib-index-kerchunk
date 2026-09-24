import json,time,os,tempfile,urllib.request,statistics as st
from concurrent.futures import ThreadPoolExecutor
BASE="https://ecmwf-forecasts.s3.amazonaws.com"
def get(url,rng=None):
    r=urllib.request.Request(url)
    if rng: r.add_header("Range",f"bytes={rng[0]}-{rng[1]}")
    return urllib.request.urlopen(r,timeout=60).read()
date="20260915";key=f"{date}/00z/ifs/0p25/enfo/{date}000000-24h-enfo-ef"
recs=[json.loads(l) for l in get(f"{BASE}/{key}.index").decode().strip().split("\n")]
r=next(x for x in recs if x.get("param")=="t" and str(x.get("levelist"))=="850" and str(x.get("number"))=="1")
buf=get(f"{BASE}/{key}.grib2",(r["_offset"],r["_offset"]+r["_length"]-1))
import numpy as np,eccodes,xarray as xr,gribberish

print("A) where did 2000 ms come from? cfgrib variants (median of 10)")
def cf(indexpath):
    ts=[]
    for _ in range(10):
        t0=time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as tf: tf.write(buf); p=tf.name
        ds=xr.open_dataset(p,engine="cfgrib",backend_kwargs={"indexpath":indexpath})
        a=ds[list(ds.data_vars)[0]].values; ds.close()
        for f in (p,p+".*"): 
            try: os.unlink(f)
            except: pass
        ts.append((time.perf_counter()-t0)*1000)
    return st.median(ts)
print(f"   cfgrib indexpath='' (disabled) : {cf(''):7.1f} ms")
try: print(f"   cfgrib default index sidecar   : {cf(None):7.1f} ms")
except Exception as e: print("   default idx failed:",e)

print("\nB) threaded scaling -- decode the SAME message N times concurrently")
def ecc(_):
    h=eccodes.codes_new_from_message(buf); v=eccodes.codes_get_values(h); eccodes.codes_release(h); return v.size
def grb(_): return gribberish.parse_grib_array(buf,0).size
for nthreads in (1,2,4,8):
    row=[]
    for name,fn in (("eccodes",ecc),("gribberish",grb)):
        work=nthreads*8
        best=1e9
        for _ in range(3):
            t0=time.perf_counter()
            with ThreadPoolExecutor(nthreads) as ex: list(ex.map(fn,range(work)))
            best=min(best,time.perf_counter()-t0)
        row.append((name, work/best))
    print(f"   threads={nthreads}: " + "  ".join(f"{n} {r:6.1f} msg/s" for n,r in row))
