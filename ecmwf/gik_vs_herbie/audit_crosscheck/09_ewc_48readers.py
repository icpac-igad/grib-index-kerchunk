"""8.2 at 48 readers from the EWC network. Concurrency sweep + 3-path head-to-head."""
import json, os, time, tempfile, random, urllib.request, statistics as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np, eccodes, gribberish, xarray as xr

BASE="https://ecmwf-forecasts.s3.amazonaws.com"; DATE,RUN="20260915","00"
CHANNELS=[("t","850"),("t","700"),("2t",None),("u","700")]
LAT0,LAT1,LON0,LON1=200,521,780,1041
RET=[0]
def key(s): return f"{DATE}/{RUN}z/ifs/0p25/enfo/{DATE}000000-{s}h-enfo-ef"
def get(url,rng=None):
    last=None
    for a in range(6):
        try:
            r=urllib.request.Request(url)
            if rng: r.add_header("Range",f"bytes={rng[0]}-{rng[1]}")
            return urllib.request.urlopen(r,timeout=120).read()
        except Exception as e:
            last=e; RET[0]+=1; time.sleep(0.4*2**a+random.random()*0.3)
    raise last
def index(steps):
    t0=time.perf_counter(); tbl={}; nb=0
    def one(s):
        raw=get(f"{BASE}/{key(s)}.index")
        return s,[json.loads(l) for l in raw.decode().strip().split("\n")],len(raw)
    with ThreadPoolExecutor(min(len(steps),12)) as ex:
        for s,recs,n in ex.map(one,steps):
            nb+=n
            for r in recs: tbl[(s,r.get("param"),r.get("levelist"),r.get("number"))]=(r["_offset"],r["_length"])
    return tbl,(time.perf_counter()-t0)*1000,nb
def work(tbl,steps,nmem):
    w=[]
    for s in steps:
        for p,l in CHANNELS:
            for n in range(1,nmem+1):
                k=(s,p,l,str(n))
                if k in tbl: w.append((s,)+tbl[k])
    return w
def task(item,path):
    s,off,ln=item
    buf=get(f"{BASE}/{key(s)}.grib2",(off,off+ln-1))
    if path=="A": a=np.asarray(gribberish.parse_grib_array(buf,0)).reshape(721,1440)
    elif path=="B":
        h=eccodes.codes_new_from_message(buf); a=eccodes.codes_get_values(h).reshape(721,1440); eccodes.codes_release(h)
    else:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as tf: tf.write(buf); p=tf.name
        ds=xr.open_dataset(p,engine="cfgrib",backend_kwargs={"indexpath":""}); a=ds[list(ds.data_vars)[0]].values; ds.close(); os.unlink(p)
    _=a[LAT0:LAT1,LON0:LON1]; return len(buf)
def run(w,path,nthread):
    RET[0]=0; t0=time.perf_counter(); nb=0
    with ThreadPoolExecutor(nthread) as ex:
        for b in ex.map(lambda i: task(i,path), w): nb+=b
    el=time.perf_counter()-t0
    return el, len(w)/el, nb, RET[0]

print(f"EWC box 192.168.1.129 | 8 cores | date {DATE} {RUN}z\n")
STEPS=["6","12","24","48","72","96"]
tbl,idx_ms,idx_nb=index(STEPS)
print(f"index: {len(STEPS)} files, {idx_nb/1e6:.1f} MB, {idx_ms:.0f} ms parallel\n")
w=work(tbl,STEPS,20)   # 6 steps x 4 ch x 20 mem = 480
print(f"CONCURRENCY SWEEP  (path B, {len(w)} messages each)")
print(f"{'readers':>8} {'wall':>8} {'msg/s':>9} {'MB/s':>8} {'retries':>8}")
for nt in (6,12,24,48):
    el,rate,nb,ret=run(w,"B",nt)
    print(f"{nt:>8} {el:>7.1f}s {rate:>8.1f} {nb/1e6/el:>7.1f} {ret:>8}")
w2=work(tbl,STEPS,25)  # 600
print(f"\nHEAD-TO-HEAD AT 48 READERS  ({len(w2)} messages)")
print(f"{'path':<34} {'wall':>8} {'msg/s':>9} {'MB/s':>8} {'retries':>8}")
for p,name in (("A","A store-like (gribberish)"),("B","B design C (eccodes-direct)"),("C","C herbie-like (tempfile+cfgrib)")):
    el,rate,nb,ret=run(w2,p,48)
    tot=el+(0 if p=="A" else idx_ms/1000)
    print(f"{name:<34} {tot:>7.1f}s {len(w2)/tot:>8.1f} {nb/1e6/el:>7.1f} {ret:>8}")
