# /// script
# requires-python = ">=3.11"
# dependencies = ["google-cloud-storage"]
# ///
"""Inventory the par bucket and emit one --dates-file per (era, run) group.

Option A of the multi-run design: twelve independent groups, `{era}/{run}z`, each
with its own preallocated time axis. This produces the twelve date lists that
`build_ecmwf_icechunk.py --create-schema` consumes, straight from what actually
exists in the bucket rather than from the published catalog.parquet -- which
lists 00z only (1,256 rows) and so cannot describe the other three runs.

Era assignment is per (date, run), not per date, because every era transition
happens at 06z. See ERA_BOUNDS in grids.py.

    uv run make_run_date_lists.py --out dates/
    uv run make_run_date_lists.py --out dates/ --report-only

Then, per group:

    build_ecmwf_icechunk.py --era 49r1 --run 06 --create-schema \
        --dates-file dates/dates-49r1-06z.txt --store <store>
    backfill_parallel.py --era 49r1 --run 06 --store <store> --par-source gcs

Reads the bucket with whatever GOOGLE_APPLICATION_CREDENTIALS points at; needs
only storage.objects.list. Verified working with coiled-data@sewaa-416306
(coiled-data-e4drr@e4drr-crafd is 403 on this bucket, for pars and store alike).
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grids import ERA_BOUNDS, LONG_RUNS, era_for  # noqa: E402

RUNS = ("00z", "06z", "12z", "18z")
N_MEMBERS = 51


def inventory(bucket: str, root: str, sa_key: str | None):
    """{(date, run): n_pars} for the whole tree. ~256k objects, ~25 s."""
    from google.cloud import storage
    client = (storage.Client.from_service_account_json(sa_key) if sa_key
              else storage.Client())
    cnt = collections.Counter()
    root = root.rstrip("/") + "/"
    depth = len(root.rstrip("/").split("/"))
    for b in client.list_blobs(bucket, prefix=root,
                               fields="items(name),nextPageToken"):
        p = b.name.split("/")
        # <root>/YYYY/MM/YYYYMMDD/<run>z/<date><run>z-<member>.parquet
        # -> five components below the root
        if len(p) == depth + 5 and p[-1].endswith(".parquet"):
            cnt[(p[-3], p[-2])] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default="gik-ecmwf-aws-tf")
    ap.add_argument("--root", default="v20260623_run_par_ecmwf")
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--out", default="dates")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    cnt = inventory(a.bucket, a.root, a.sa_key)
    dates = sorted({d for d, _ in cnt})
    print(f"gs://{a.bucket}/{a.root}")
    print(f"  {len(dates)} date dirs {dates[0]}..{dates[-1]}, "
          f"{len(cnt)} (date,run) pairs, {sum(cnt.values()):,} pars")

    short = [(d, r, n) for (d, r), n in sorted(cnt.items()) if n != N_MEMBERS]
    absent = [(d, r) for d in dates for r in RUNS if (d, r) not in cnt]
    print(f"  incomplete (date,run): {len(short)}"
          + ("" if not short else "  " + ", ".join(f"{d} {r} ({n})"
                                                   for d, r, n in short[:10])))
    print(f"  absent (date,run)    : {len(absent)}"
          + ("" if not absent else "  " + ", ".join(f"{d} {r}" for d, r in absent)))

    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for d, r in sorted(cnt):
        groups[(era_for(d, r), r)].append(d)

    print(f"\n  {'group':12s} {'dates':>6s} {'steps':>6s} {'first':>10s} "
          f"{'last':>10s}")
    for era in ERA_BOUNDS:
        for r in RUNS:
            dd = sorted(groups.get((era, r), []))
            if not dd:
                continue
            steps = 85 if int(r[:2]) in LONG_RUNS else 49
            print(f"  {era + '/' + r:12s} {len(dd):>6d} {steps:>6d} "
                  f"{dd[0]:>10s} {dd[-1]:>10s}")

    if a.report_only:
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for (era, r), dd in sorted(groups.items()):
        f = out / f"dates-{era}-{r}.txt"
        f.write_text("\n".join(sorted(dd)) + "\n")
    print(f"\nwrote {len(groups)} date lists to {out}/")


if __name__ == "__main__":
    main()
