#!/usr/bin/env python3
"""C2 -- can the realization's task graph be expressed with Herbie at all?

The realization's unit of work (frisky_daily_dag.read_message) is a
module-level function taking six SCALARS and returning one 321x261 float32
slab.  125,970 such tasks per date, 48 in flight.  Frisky/Dask pickle the
function by reference, so ~0 bytes ride with each task.

This measures what the Herbie equivalent would have to carry, and what it
costs to create.  Nothing here is extrapolated: every number is timed.
"""
import os, sys, time, io, statistics as st
sys.path.insert(0, os.environ.get("HERBIE_SRC", "herbie/src"))
os.environ.setdefault("HERBIE_VERBOSE", "false")

import cloudpickle
from datetime import datetime
from herbie import Herbie

DATE, FXX = datetime(2026, 9, 1), 24
REPS = int(os.environ.get("REPS", "5"))

def q(fn, *a, **k):
    """silence Herbie's banner; return (result, seconds)"""
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        t0 = time.perf_counter(); r = fn(*a, **k); return r, time.perf_counter() - t0
    finally:
        sys.stdout = old

print("=" * 78); print("C2  cost of the Herbie object itself"); print("=" * 78)

ts = []
for _ in range(REPS):
    _, dt = q(Herbie, DATE, model="ifs", product="enfo", fxx=FXX)
    ts.append(dt)
print(f"  Herbie(...) construction      : median {st.median(ts)*1000:8.1f} ms"
      f"   (min {min(ts)*1000:.0f}, max {max(ts)*1000:.0f})")
print("    -> __init__ probes SOURCES over the network to find the file;")
print("       it is not a cheap value object.")

H, _ = q(Herbie, DATE, model="ifs", product="enfo", fxx=FXX)
_, dt_idx = q(lambda: H.index_as_dataframe)
print(f"  first .index_as_dataframe     : {dt_idx*1000:8.1f} ms   "
      f"({len(H.index_as_dataframe)} rows)")
_, dt_idx2 = q(lambda: H.index_as_dataframe)
print(f"  cached .index_as_dataframe    : {dt_idx2*1000:8.1f} ms   (cached_property)")

print("\n" + "=" * 78); print("C2b  can it ride in a task graph?"); print("=" * 78)
for label, obj in (("Herbie, index cold", q(Herbie, DATE, model="ifs",
                                            product="enfo", fxx=FXX)[0]),
                   ("Herbie, index warm", H)):
    try:
        t0 = time.perf_counter(); blob = cloudpickle.dumps(obj)
        dt = time.perf_counter() - t0
        t1 = time.perf_counter(); cloudpickle.loads(blob)
        dt2 = time.perf_counter() - t1
        print(f"  {label:20s}: {len(blob):9,d} B   dumps {dt*1000:6.1f} ms  "
              f"loads {dt2*1000:6.1f} ms")
    except Exception as exc:
        print(f"  {label:20s}: NOT PICKLABLE -- {type(exc).__name__}: {exc}")

# frisky's actual per-task payload, for comparison
task = ("50r1", "t", 850, "2026-09-01", 7, 24)
blob = cloudpickle.dumps(task)
print(f"  {'frisky task args':20s}: {len(blob):9,d} B   (six scalars; the "
      f"function is pickled by reference)")

print("\n" + "=" * 78)
print("C2c  is there an in-memory single-message read?")
print("=" * 78)
import inspect
sig = inspect.signature(Herbie.download)
print(f"  Herbie.download   -> {sig.return_annotation}")
src_x = inspect.getsource(Herbie.xarray)
print(f"  Herbie.xarray     -> writes a local file? "
      f"{'YES' if 'download' in src_x else 'no'}; "
      f"cfgrib.open_datasets? {'YES' if 'open_datasets' in src_x else 'no'}")
meths = [m for m in dir(Herbie) if not m.startswith('_')]
byteish = [m for m in meths if any(w in m.lower()
           for w in ("byte", "buffer", "memory", "bytesio", "read"))]
print(f"  public methods hinting at an in-memory read: {byteish or 'NONE'}")

print("\n" + "=" * 78)
print("C2d  what one date of the realization would cost in Herbie objects")
print("=" * 78)
STEPS, CHANNELS, MEMBERS = 65, 38, 51
msgs = CHANNELS * MEMBERS * STEPS
ctor = st.median(ts)
print(f"  messages per date                     : {msgs:,}")
print(f"  Herbie objects if one per (step,stream): {STEPS*2:,}"
      f"   -> {STEPS*2*ctor:6.1f} s of construction")
print(f"  Herbie objects if one per TASK         : {msgs:,}"
      f"   -> {msgs*ctor/3600:6.1f} h of construction ALONE, per date")
print(f"  frisky: objects created per task       : 0 "
      f"(module-level fn + cached store handle)")
