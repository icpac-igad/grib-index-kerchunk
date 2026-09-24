"""8.2 benchmark, scoped: one date, one box, three paths, phases separated.

Path A  store-like : offsets already known (what the virtual/realized store gives)
                     -> byte-range GET -> gribberish -> subset
Path B  design C   : parse .index at run time -> GET -> eccodes-direct -> subset
Path C  herbie-like: parse .index at run time -> GET -> temp file -> cfgrib -> subset
"""
import json, os, time, tempfile, urllib.request, statistics as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BASE = "https://ecmwf-forecasts.s3.amazonaws.com"
DATE, RUN = "20260915", "00"
STEPS = ["24", "48", "72"]
CHANNELS = [("t", "850"), ("t", "700"), ("2t", None), ("u", "700")]
NTHREAD = 6
# EA-SWIO 15-80E, 40S-40N on the 721x1440 0.25deg grid
LAT0, LAT1, LON0, LON1 = 200, 521, 780, 1041

def key(step): return f"{DATE}/{RUN}z/ifs/0p25/enfo/{DATE}000000-{step}h-enfo-ef"
RETRIES = [0]
def get(url, rng=None):
    import random
    last = None
    for attempt in range(6):
        try:
            r = urllib.request.Request(url)
            if rng: r.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            return urllib.request.urlopen(r, timeout=120).read()
        except Exception as e:
            last = e; RETRIES[0] += 1
            time.sleep((0.5 * 2**attempt) + random.random()*0.3)
    raise last

def acquire_index():
    """Phase 1 -- what paths B and C pay and path A does not."""
    t0 = time.perf_counter(); table = {}
    def one(step):
        raw = get(f"{BASE}/{key(step)}.index")
        return step, [json.loads(l) for l in raw.decode().strip().split("\n")], len(raw)
    nbytes = 0
    with ThreadPoolExecutor(len(STEPS)) as ex:
        for step, recs, nb in ex.map(one, STEPS):
            nbytes += nb
            for r in recs:
                table[(step, r.get("param"), r.get("levelist"), r.get("number"))] = (r["_offset"], r["_length"])
    return table, (time.perf_counter()-t0)*1000, nbytes

def worklist(table):
    w = []
    for step in STEPS:
        for param, lev in CHANNELS:
            for n in range(1, 51):
                k = (step, param, lev, str(n))
                if k in table: w.append((step,)+table[k])
    return w

def run(path, work):
    """Returns wall_s, per-message fetch/decode medians, bytes."""
    fetch, decode, nbytes = [], [], [0]
    import eccodes, gribberish, xarray as xr
    def task(item):
        step, off, ln = item
        t0 = time.perf_counter()
        buf = get(f"{BASE}/{key(step)}.grib2", (off, off+ln-1))
        t1 = time.perf_counter()
        if path == "A":
            a = np.asarray(gribberish.parse_grib_array(buf, 0)).reshape(721, 1440)
        elif path == "B":
            h = eccodes.codes_new_from_message(buf)
            a = eccodes.codes_get_values(h).reshape(721, 1440); eccodes.codes_release(h)
        else:
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
                tf.write(buf); p = tf.name
            ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
            a = ds[list(ds.data_vars)[0]].values; ds.close(); os.unlink(p)
        sub = a[LAT0:LAT1, LON0:LON1]
        t2 = time.perf_counter()
        return (t1-t0)*1000, (t2-t1)*1000, len(buf), sub.shape
    t0 = time.perf_counter()
    with ThreadPoolExecutor(NTHREAD) as ex:
        for f, d, nb, shp in ex.map(task, work):
            fetch.append(f); decode.append(d); nbytes[0] += nb
    return time.perf_counter()-t0, st.median(fetch), st.median(decode), nbytes[0], shp

table, idx_ms, idx_bytes = acquire_index()
work = worklist(table)
print(f"date {DATE} {RUN}z | steps {STEPS} | channels {[c[0]+(c[1] or '') for c in CHANNELS]}")
print(f"messages: {len(work)}   threads: {NTHREAD}   box: {LAT1-LAT0}x{LON1-LON0}")
print(f"\nPHASE 1  index acquisition ({len(STEPS)} files, {idx_bytes/1e6:.1f} MB): {idx_ms:.0f} ms")
print(f"         path A pays 0 ms here (offsets come from the manifest)\n")
rows = []
for p, name in (("A","A store-like (gribberish)"), ("B","B design C (eccodes-direct)"), ("C","C herbie-like (tempfile+cfgrib)")):
    wall, f, d, nb, shp = run(p, work)
    total = wall + (0 if p == "A" else idx_ms/1000)
    rows.append((name, wall, total, f, d, nb, shp))
    print(f"{name:34} wall {wall:6.1f}s  +index -> {total:6.1f}s | fetch {f:6.1f} ms  decode {d:6.1f} ms  {nb/1e6:.0f} MB  subset {shp}")
print(f"\nS3 retries absorbed: {RETRIES[0]}")
print(f"\nper-message end-to-end (wall/msg):")
for name, wall, total, f, d, nb, shp in rows:
    print(f"  {name:34} {total/len(work)*1000:7.1f} ms   ({len(work)/total:5.1f} msg/s at {NTHREAD} threads)")
