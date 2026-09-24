#!/usr/bin/env python3
"""C5 -- how many bytes does Herbie actually move for one message?

`GIK_vs_Herbie_Evaluation.md` §4.2 and its §6 summary row claim Herbie
performs a "full GRIB file download" of 2-4 GB and achieves "~1% useful
bytes".  This measures it.
"""
import os, sys, tempfile, shutil, json, urllib.request
sys.path.insert(0, os.environ.get("HERBIE_SRC", "herbie/src"))
import io, warnings; warnings.filterwarnings("ignore")
from datetime import datetime
from herbie import Herbie

DATE, FXX, SEARCH = datetime(2026, 9, 1), 24, r":t:850:pl:7:"
BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
KEY = f"20260901/00z/ifs/0p25/enfo/20260901000000-{FXX}h-enfo-ef"

def quiet(fn, *a, **k):
    buf, old = io.StringIO(), sys.stdout; sys.stdout = buf
    try: return fn(*a, **k)
    finally: sys.stdout = old

# size of the whole step file, per S3
req = urllib.request.Request(f"{BASE}/{KEY}.grib2", method="HEAD")
full = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

tmp = tempfile.mkdtemp(prefix="bow-")
H = quiet(Herbie, DATE, model="ifs", product="enfo", fxx=FXX,
          save_dir=tmp, overwrite=True, verbose=False)
df = H.index_as_dataframe
row = df[df.search_this.str.contains(SEARCH, regex=True)]
msg_len = int(row.end_byte.iloc[0]) - int(row.start_byte.iloc[0])

p = quiet(H.download, search=SEARCH, overwrite=True, verbose=False)
on_disk = os.path.getsize(p)

print("=" * 78)
print("C5  bytes on the wire for ONE message, via Herbie")
print("=" * 78)
print(f"  full step file (S3 Content-Length) : {full:>14,d} B  "
      f"({full/2**30:.2f} GiB)")
print(f"  the message, per .index `_length`  : {msg_len:>14,d} B")
print(f"  what Herbie wrote to disk          : {on_disk:>14,d} B")
print(f"  ratio written / full file          : {on_disk/full*100:>14.4f} %")
print()
print(f"  §4.2 claims ~3 GB fetched and '~1% useful bytes'.")
print(f"  measured: Herbie fetched {on_disk:,} B for a {msg_len:,} B message")
print(f"  -> useful bytes = {msg_len/on_disk*100:.1f} %, "
      f"NOT 1%.  The download is a byte-range GET.")
print()
print(f"  index sidecar also fetched, once per file: "
      f"{len(urllib.request.urlopen(f'{BASE}/{KEY}.index', timeout=60).read()):,} B")
shutil.rmtree(tmp, ignore_errors=True)

# ---- what that index sidecar costs at corpus scale -------------------------
idx_b = len(urllib.request.urlopen(f"{BASE}/{KEY}.index", timeout=60).read())
FILES_PER_DATE, DATES = 87, 887          # enfo, 00z; REALIZATION_COST.md §1
tot = idx_b * FILES_PER_DATE * DATES
print()
print("=" * 78)
print("C5b  index traffic at corpus scale -- the thing the store buys once")
print("=" * 78)
print(f"  one enfo `.index`                  : {idx_b:>14,d} B")
print(f"  x {FILES_PER_DATE} files/date                     : "
      f"{idx_b*FILES_PER_DATE:>14,d} B/date")
print(f"  x {DATES} dates (one full pass)      : {tot:>14,d} B  "
      f"({tot/2**40:.2f} TiB)" if tot > 2**40 else
      f"  x {DATES} dates (one full pass)      : {tot:>14,d} B  "
      f"({tot/2**30:.1f} GiB)")
print(f"  re-paid on EVERY pass.  The published virtual store is 76 GB")
print(f"  covering 1,308 dates x 12 groups, fetched once and shared.")
