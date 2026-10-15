#!/usr/bin/env python
"""Delete the realized stores that hold the wrong region.

Every `ea-swio` corpus was built by selecting `longitude=slice(15, 80)` from a
source store whose longitude coordinate is 180 deg out (see
../HANDOVER_LONGITUDE_FIX.md). They therefore contain the eastern Pacific, not
East Africa, and are of no use to CRMA. Relabelling cannot save them -- the
wrong subset was read, so the bytes themselves are the wrong place.

DRY RUN BY DEFAULT. Nothing is deleted without --yes.

    $P purge_wrong_stores.py --env .env                 # show what would go
    $P purge_wrong_stores.py --env .env --yes           # actually delete
    $P purge_wrong_stores.py --env .env --include-ea-cgan --yes

Guards, because 3.3 TB is not recoverable:

  * KEEP is checked key-by-key, not prefix-by-prefix. If any key queued for
    deletion falls under a KEEP prefix the run aborts before deleting anything.
  * Trailing slashes are mandatory on every prefix. Bare "ea-swio/v1-" would
    match four different stores.
  * A size/count tripwire aborts if the collected set is far from expectation,
    which catches a mistyped prefix that silently matches more than intended.
  * The two `ea-cgan` stores are NOT deleted by default. They carry the same
    defect, but CRMA's flood scenario may still reference them, so removing
    them is a separate decision (--include-ea-cgan).
"""
import argparse
import sys

# Wrong region, no value, safe to remove. Trailing slash is required.
PURGE = [
    "ea-swio/v1-49r1-mam2024/",
    "ea-swio/v1-49r1-mam2025/",
    "ea-swio/v1-49r1-mar-may2026/",
    "ea-swio/v1-50r1-mam2026-tail/",
]
# Same defect, but possibly still referenced downstream -- opt in explicitly.
EA_CGAN = [
    "ea-cgan/v3-june2026/",
    "ea-cgan/v4-mar2026-49r1/",
]
# Never touch, whatever else happens. Anything queued for deletion that falls
# under one of these aborts the entire run before a single object is removed.
KEEP = [
    # v2-49r1-mam2026 was here while it was deliberately retained. Released
    # 2026-08-19 once v3-49r1-mam2026 verified complete (170,820 chunks, none
    # missing) at the wider 40N-40S extent -- and because the bucket was
    # heading for 5.37 TB against a 5 TB hard limit with MAM 2024 still to go.
]

ENDPOINT = "https://object-store.os-api.cci1.ecmwf.int"
# Tripwire. Raised 2026-08-22 for the v3 purge: the four v3 stores are the
# widened 40N-40S build at 38 channels, so they are larger than the v1 set this
# was first sized for -- 722,370 objects / 4,898 GB against 660k / 3,364 GB.
# Raise this only against a reconciliation you have actually read. The point of
# the guard is that a mistyped prefix silently matches more than intended, and
# a limit bumped merely to make an abort go away is no limit at all.
MAX_OBJECTS = 900_000
MAX_GB = 5_000


def load_env(path):
    creds = {}
    with open(path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                creds[k.strip()] = v.strip()
    return creds["AK"], creds["SK"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env")
    p.add_argument("--bucket", default="must-icechunk")
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--include-ea-cgan", action="store_true",
                   help="also delete ea-cgan/v3-june2026 and v4-mar2026-49r1")
    p.add_argument("--prefix", action="append", default=None, metavar="PFX",
                   help="delete this prefix instead of the built-in list. "
                        "Repeatable. A trailing '/' is added if missing")
    p.add_argument("--yes", action="store_true",
                   help="actually delete. Without this it is a dry run")
    args = p.parse_args()

    import boto3

    if args.prefix:
        targets = [p if p.endswith("/") else p + "/" for p in args.prefix]
    else:
        targets = list(PURGE) + (list(EA_CGAN) if args.include_ea_cgan else [])
    bad = [t for t in targets if not t.endswith("/")]
    if bad:
        sys.exit(f"ABORT: prefixes must end in '/': {bad}")

    ak, sk = load_env(args.env)
    s3 = boto3.client("s3", endpoint_url=args.endpoint,
                      aws_access_key_id=ak, aws_secret_access_key=sk,
                      region_name="RegionOne")

    print(f"bucket {args.bucket}   mode "
          f"{'DELETE' if args.yes else 'DRY RUN (nothing will be removed)'}\n")
    keys, total = [], 0
    for pfx in targets:
        n, sz, tok = 0, 0, {}
        while True:
            r = s3.list_objects_v2(Bucket=args.bucket, Prefix=pfx, **tok)
            for o in r.get("Contents", []):
                keys.append(o["Key"]); n += 1; sz += o["Size"]
            if not r.get("IsTruncated"):
                break
            tok = {"ContinuationToken": r["NextContinuationToken"]}
        total += sz
        print(f"  {pfx:<34} {n:>9,} objects  {sz/1e9:>9.2f} GB")
    print(f"  {'TOTAL':<34} {len(keys):>9,} objects  {total/1e9:>9.2f} GB")

    # --- guards ------------------------------------------------------------
    if KEEP:
        clash = [k for k in keys if any(k.startswith(x) for x in KEEP)]
        if clash:
            sys.exit(f"\nABORT: {len(clash)} key(s) under a KEEP prefix, "
                     f"e.g. {clash[:2]}")
    if len(keys) > MAX_OBJECTS or total / 1e9 > MAX_GB:
        sys.exit(f"\nABORT: {len(keys):,} objects / {total/1e9:.0f} GB exceeds "
                 f"the tripwire ({MAX_OBJECTS:,} / {MAX_GB} GB). A prefix is "
                 f"probably wrong -- check before raising the limit.")
    if not keys:
        print("\nnothing to do")
        return 0
    print(f"\nguards passed: no KEEP collisions, within tripwire")

    if not args.yes:
        print("\nDRY RUN — pass --yes to delete. This cannot be undone: "
              "icechunk history does not survive object deletion.")
        return 0

    done = 0
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i:i + 1000]]
        r = s3.delete_objects(Bucket=args.bucket,
                              Delete={"Objects": batch, "Quiet": True})
        for e in r.get("Errors", [])[:3]:
            print(f"  ERROR {e}")
        done += len(batch)
        if done % 50_000 < 1000:
            print(f"  {done:,}/{len(keys):,} deleted", flush=True)
    print(f"\ndeleted {done:,} objects, {total/1e9:.2f} GB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
