#!/usr/bin/env python3
"""C1 -- does a Herbie `enfo` request yield all 51 members, in every era?

Audit GIK_VS_HERBIE_AUDIT.md §5.4 claims stage (1)/(2) are replaceable by
Herbie at ~2% overhead.  That prices index FETCHES.  This checks whether the
substitute returns the same DATA.

Ground truth is read straight from the `.index` sidecars (no Herbie), so the
result stands even if Herbie's file discovery misbehaves.  Herbie is then run
over the same dates and the two are compared.

Expectation being tested (from REALIZED_EA_SWIO_STORE.md:147):
    49r1 enfo  -> number 0..50  = 51 members (control bundled)
    50r1 enfo  -> number 1..50  = 50 members (control moved to oper/fc)
"""
import json, os, sys, urllib.request, collections

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
FXX  = 24

# one date per era, all 00z.  Boundaries per CLAUDE.md, verified by listing.
DATES = [
    ("0p4-beta", "20230601", "0p4-beta"),
    ("49r1-9lv", "20240529", "0p25"),   # the legacy template reference date
    ("49r1-13lv", "20250515", "0p25"),
    ("50r1", "20260901", "0p25"),       # post 2026-05-12 dual-stream
]


def fetch(url):
    return urllib.request.urlopen(url, timeout=90).read().decode()


def index_url(date, res, product):
    suffix = "ef" if product in ("enfo", "waef") else "fc"
    stem = f"{date}000000-{FXX}h-{product}-{suffix}.index"
    if res == "0p4-beta":                       # pre 2024-02-28 06z layout
        return f"{BASE}/{date}/00z/{res}/{product}/{stem}"
    return f"{BASE}/{date}/00z/ifs/{res}/{product}/{stem}"


def members_in(date, res, product):
    """-> (sorted member set, n index lines) or (None, reason)."""
    try:
        txt = fetch(index_url(date, res, product))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    recs = [json.loads(l) for l in txt.strip().split("\n") if l]
    nums = set()
    for r in recs:
        n = r.get("number")
        nums.add(int(n) if n is not None else -1)   # -1 = field carries no number
    return sorted(nums), len(recs)


print("=" * 78)
print("C1  member axis: what an `enfo` request actually returns, per era")
print("=" * 78)

truth = {}
for era, date, res in DATES:
    nums, n = members_in(date, res, "enfo")
    if nums is None:
        print(f"\n{era:10s} {date}  enfo    UNAVAILABLE  {n}")
        continue
    numbered = [x for x in nums if x >= 0]
    truth[era] = set(numbered)
    print(f"\n{era:10s} {date}  enfo    {n:6d} index lines")
    print(f"{'':10s} {'':8s}          members = {len(numbered)}"
          f"  range {min(numbered) if numbered else '-'}..{max(numbered) if numbered else '-'}"
          f"   unnumbered fields: {'yes' if -1 in nums else 'no'}")

    # if the control is not in enfo, is it in oper/fc?
    if numbered and 0 not in numbered:
        onums, on = members_in(date, res, "oper")
        if onums is None:
            print(f"{'':10s} {'':8s}          control NOT in enfo; oper/fc: {on}")
        else:
            print(f"{'':10s} {'':8s}          control NOT in enfo -> oper/fc has "
                  f"{on} lines, number field {sorted(onums)}")
            print(f"{'':10s} {'':8s}          => 51st member requires a SECOND "
                  f"stream + a splice")

# ---------------------------------------------------------------- Herbie ----
print("\n" + "=" * 78)
print("C1b  the same request through Herbie itself")
print("=" * 78)
sys.path.insert(0, os.environ.get("HERBIE_SRC", "herbie/src"))
try:
    from herbie import Herbie
    from datetime import datetime
    for era, date, res in DATES:
        kw = dict(model="ifs", product="enfo", fxx=FXX)
        if res == "0p4-beta":
            kw["resolution"] = "0p4-beta"
        try:
            H = Herbie(datetime.strptime(date, "%Y%m%d"), **kw)
            df = H.index_as_dataframe
            nums = sorted({int(x) for x in df["number"].dropna().unique()})
            flag = ""
            if era in truth and set(nums) != truth[era]:
                flag = "  <-- DIFFERS FROM .index GROUND TRUTH"
            print(f"{era:10s} {date}  herbie enfo -> {len(nums):2d} members "
                  f"({min(nums)}..{max(nums)}){flag}")
        except Exception as exc:
            print(f"{era:10s} {date}  herbie enfo -> {type(exc).__name__}: {exc}")
except ImportError as exc:
    print(f"herbie not importable ({exc}); set HERBIE_SRC=<clone>/src")

# ------------------------------------------------- static template check ----
print("\n" + "=" * 78)
print("C1c  does Herbie's `ifs` template implement the 50r1 control split?")
print("=" * 78)
src = os.path.join(os.environ.get("HERBIE_SRC", "herbie/src"),
                   "herbie/models/ecmwf.py")
try:
    text = open(src).read()
    ifs_block = text.split("class ifs")[1].split("class aifs")[0]
    hits = {w: ifs_block.count(w) for w in ("get_control", "control", "'cf'", '"cf"')}
    print(f"  occurrences inside `class ifs`: {hits}")
    print(f"  verdict: {'IMPLEMENTED' if any(hits.values()) else 'NOT IMPLEMENTED'}"
          f" -- an enfo request post-2026-05-12 silently omits the control")
except Exception as exc:
    print(f"  could not read {src}: {exc}")
