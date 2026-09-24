import json, time, io, os, tempfile, urllib.request, statistics as st
BASE="https://ecmwf-forecasts.s3.amazonaws.com"
def get(url, rng=None):
    r=urllib.request.Request(url)
    if rng: r.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    return urllib.request.urlopen(r, timeout=60).read()

date=os.environ.get("D","20260915"); run="00"; step="24"
stem=f"{date}000000-{step}h-enfo-ef"
key=f"{date}/{run}z/ifs/0p25/enfo/{stem}"
idx=get(f"{BASE}/{key}.index").decode().strip().split("\n")
print("index lines:", len(idx))
recs=[json.loads(l) for l in idx]
# pick a pressure-level temperature message and a surface one
picks=[]
for r in recs:
    if r.get("param")=="t" and r.get("levtype")=="pl" and str(r.get("levelist"))=="850" and str(r.get("number"))=="1":
        picks.append(("t850", r)); break
for r in recs:
    if r.get("param")=="2t" and str(r.get("number"))=="1":
        picks.append(("t2m", r)); break
print("picked:", [(n, r["_offset"], r["_length"]) for n,r in picks])

import numpy as np, eccodes, xarray as xr, gribberish
N=int(os.environ.get("N","15"))
for name, r in picks:
    off,ln=r["_offset"], r["_length"]
    buf=get(f"{BASE}/{key}.grib2",(off,off+ln-1))
    print(f"\n=== {name}  msg={len(buf)/1024:.0f} KiB ===")

    # 1. eccodes DIRECT from in-memory buffer (no temp file, no xarray)
    ts=[]
    for _ in range(N):
        t0=time.perf_counter()
        h=eccodes.codes_new_from_message(buf)
        v=eccodes.codes_get_values(h)
        eccodes.codes_release(h)
        ts.append((time.perf_counter()-t0)*1000)
    print(f"eccodes  direct-from-buffer : {st.median(ts):7.1f} ms  (min {min(ts):.1f})  n={v.size}")

    # 2. cfgrib via temp file + xarray Dataset  (what the doc measured)
    ts=[]
    for _ in range(N):
        t0=time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
            tf.write(buf); p=tf.name
        ds=xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath":""})
        a=ds[list(ds.data_vars)[0]].values
        ds.close(); os.unlink(p)
        ts.append((time.perf_counter()-t0)*1000)
    print(f"cfgrib   tempfile + xarray  : {st.median(ts):7.1f} ms  (min {min(ts):.1f})  shape={a.shape}")

    # 3. gribberish direct
    ts=[]
    for _ in range(N):
        t0=time.perf_counter()
        g=gribberish.parse_grib_array(buf,0)
        ts.append((time.perf_counter()-t0)*1000)
    print(f"gribberish direct           : {st.median(ts):7.1f} ms  (min {min(ts):.1f})  n={g.size}")
    print(f"   agreement eccodes vs gribberish: max|diff| = {np.nanmax(np.abs(v-np.asarray(g,dtype=float))):.3e}")
