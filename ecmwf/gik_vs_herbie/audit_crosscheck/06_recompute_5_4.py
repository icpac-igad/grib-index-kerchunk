#!/usr/bin/env python3
"""C4 -- recompute GIK_VS_HERBIE_AUDIT.md §5.4 with MEASURED constants.

§5.4 says:
    "a Herbie-driven realization would fetch 87 `.index` files per date,
     887 dates = 77,169 fetches ~ 4 hours of latency against a 192-hour
     build - about 2% overhead, and parallelisable to minutes."

That implies ~187 ms per index acquisition.  04/05 measured the real cost.
Constants below are overridable from the environment so the numbers can be
re-derived on another machine rather than trusted.
"""
import os

def c(name, default):
    return float(os.environ.get(name, default))

# ---- measured (04_dag_schedulability.py, 05_head_to_head.py) --------------
#
# CORRECTED 2026-09-24 after the auditor's re-check (§12.1).  This value was
# 2.718 s, which is Herbie's `index_as_dataframe` -- DataFrame construction,
# local caching and the search-string column, NOT index acquisition.  §5.4
# priced *obtaining byte offsets*, which is an HTTP GET plus json.loads:
# 45 ms parse (stable) + 314-465 ms fetch (network-dependent).  Costing
# Herbie's DataFrame against a raw-fetch claim was not like-for-like, and
# the error was mine.
IDX_S     = c("IDX_S",     0.510)  # fetch + json.loads one .index, median
HERBIE_DF = c("HERBIE_DF", 2.718)  # Herbie's index_as_dataframe, for contrast
CTOR_S  = c("CTOR_S",  0.1735)  # Herbie(...) construction, median
IDX_MB  = c("IDX_MB",  1.783)   # warm Herbie object, cloudpickled
OFF_MB  = c("OFF_MB",  0.290)   # the offset table a graph would ACTUALLY ship
MSG_A   = c("MSG_A",   0.240)   # frisky's measured warm per-message read
MSG_B   = c("MSG_B",   0.838)   # path B per message, this machine

# ---- corpus (REALIZATION_COST.md §1-2) -----------------------------------
DATES, HOURS, READERS = 887, 192.4, 48
CH, MEM, STEP = 38, 51, 65
MSGS_DATE = CH * MEM * STEP
MSGS = DATES * MSGS_DATE
STREAMS_DATE = 87 * 2          # enfo + oper/fc; the control needs the second

def hr(s): return s / 3600.0

print("=" * 78)
print("C4  §5.4 recomputed")
print("=" * 78)
print(f"  corpus: {DATES} dates x {MSGS_DATE:,} messages = {MSGS:,}")
print(f"  build as measured: {HOURS} h wall-clock at {READERS} readers\n")

print("  -- index acquisition ------------------------------------------------")
audit_fetches, audit_hours = 77_169, 4.0
print(f"    audit §5.4 assumed : {audit_fetches:,} fetches, {audit_hours:.1f} h "
      f"(=> {audit_hours*3600/audit_fetches*1000:.0f} ms each)")
f_enfo = 87 * DATES
f_both = STREAMS_DATE * DATES
print(f"    measured per index : {IDX_S*1000:.0f} ms  (fetch + json parse, "
      f"8,500 rows)")
for label, n in (("enfo only", f_enfo), ("enfo + oper/fc", f_both)):
    h = hr(n * IDX_S)
    print(f"    {label:16s}   : {n:,} fetches -> {h:7.1f} h serial"
          f"   ({h/HOURS*100:5.1f}% of the build)"
          f"   {h*3600/READERS/60:6.1f} min at {READERS} readers")
print(f"    => §5.4's 4 h is low by ~{hr(f_enfo*IDX_S)/audit_hours:.1f}x on the "
      f"enfo-only count alone -- NOT the ~15x this script")
print(f"       previously reported, which used Herbie's DataFrame build "
      f"({HERBIE_DF*1000:.0f} ms) in place of")
print( "       index acquisition.  It remains SMALL against 111.7M message")
print( "       fetches (0.07% more requests), so §5.4's conclusion survives.")

print("\n  -- object construction: the part §5.4 never priced --------------------")
per_task = MSGS_DATE * CTOR_S
per_file = STREAMS_DATE * CTOR_S
print(f"    one Herbie object PER TASK   : {MSGS_DATE:,} x {CTOR_S*1000:.0f} ms"
      f" = {hr(per_task):6.1f} h/date -> {hr(per_task)*DATES/24:7.0f} days for "
      f"the corpus")
print(f"      (Herbie.__init__ probes SOURCES over the network; it is not a "
      f"value object)")
print(f"    one Herbie object PER FILE   : {STREAMS_DATE} x {CTOR_S*1000:.0f} ms"
      f" = {per_file:6.1f} s/date  -- fine, BUT:")
print(f"      a WARM HERBIE OBJECT is {IDX_MB:.2f} MB cloudpickled "
      f"({STREAMS_DATE*IDX_MB:,.0f} MB/date) -- but")
print(f"      nobody would ship that.  The right object is an offset table:")
print(f"      {OFF_MB:.2f} MB/file, {STREAMS_DATE*OFF_MB:,.0f} MB/date -- about the size of "
      f"stage (1)'s own output.")
print(f"      Costing the Herbie object here was the same error §5.4 made. "
      f"The")
print(f"      payload is NOT a blocker; object CONSTRUCTION is.")
print(f"    frisky                        : 0 objects/task, 46 B of task args.")

print("\n  -- per-message read --------------------------------------------------")
tot_a, tot_b = MSGS * MSG_A, MSGS * MSG_B
print(f"    path A (store)  {MSG_A*1000:5.0f} ms -> {hr(tot_a)/READERS:7.1f} h "
      f"at {READERS} readers   [measured build: {HOURS} h]")
print(f"    path B (Herbie) {MSG_B*1000:5.0f} ms -> {hr(tot_b)/READERS:7.1f} h "
      f"at {READERS} readers")
print(f"    ratio {MSG_B/MSG_A:.1f}x  -- but REALIZATION_COST §2 measures the "
      f"SOURCE as the")
print( "    bound, not the client, so the real gap at 48 readers is smaller")
print( "    than this ratio and cannot be settled without running both at scale.")

print("\n" + "=" * 78)
print("  verdict")
print("=" * 78)
print(f"""  §5.4's index arithmetic is off by ~{hr(f_enfo*IDX_S)/audit_hours:.0f}x but its conclusion holds:
  index fetching is NOT what makes stage (1)/(2) load-bearing.

  What §5.4 never priced, and what actually blocks the substitution:
    1. Herbie's unit is a FILE object costing 173 ms to construct (a network
       probe in __init__) plus 1.8-2.7 s to build its DataFrame.  The
       realization's unit is a MESSAGE.  Per-task construction is 6.1 h/date,
       or {per_task*DATES/3600/24:.0f} days for the corpus.  That is the blocker --
       not the payload, which an offset table reduces to ~{STREAMS_DATE*OFF_MB:.0f} MB/date.
    2. `download()` returns a Path.  There is no in-memory single-message
       read, so message-level work means bypassing Herbie's own fetch.
    3. The control member is not addressable as `number=0` in ANY era, and
       on 50r1 it is not in `enfo` at all -- see 03_member_axis.py.
  None of that is overhead.  It is the coordinate space the store provides.""")
