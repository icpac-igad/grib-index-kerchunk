# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "gcsfs"]
# ///
"""Mirror a GCS-hosted Icechunk store to source.coop by plain object copy.

GCS counterpart of publish_to_source_coop.py (which mirrors a *local* store).
The GIK Icechunk stores live in GCS but their virtual chunks point at public
AWS S3 GRIB (noaa-gefs-pds / ecmwf-forecasts), so a byte-for-byte object copy
of the store prefix is a fully-functional published repo -- readers just need
anonymous GET/LIST on source.coop plus anonymous GET on the AWS GRIB bucket.
Nothing about the virtual references changes; only the store metadata objects
(repo pointer + snapshots + manifests + transactions + native chunks) move.

Icechunk never reads `overwritten/` on open (it is superseded `repo`-file
versions from ref rotation), so it is skipped by default.

Correctness: the top-level `repo` file is the entry point that names the
branch tips. It is uploaded LAST, after every snapshot/manifest it can
reference is already present, so the published store is never seen pointing
at a not-yet-copied object.

Resumable: skips destination objects already present with the same size, so
when the 1-hour source.coop STS token expires mid-run you just refresh .env
and re-run -- it picks up where it stopped.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/coiled-data.json   # GCS read
  source .env                                                    # source.coop STS
  uv run mirror_gcs_to_source_coop.py \
      --gcs-store gs://gik-gefs-aws-tf/icechunk/gefs-ens \
      --dest      forecasts/noaa_gefs_aws_s3_icechunk_vd         # bucket = e4drr-project
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import gcsfs
from botocore.config import Config

# Functional icechunk prefixes, in copy order. `repo` is handled separately
# (uploaded last). `overwritten` is intentionally excluded (see module docstring).
DATA_SUBDIRS = ["manifests", "snapshots", "transactions", "chunks"]


def list_source(fs, store, subdirs):
    """(relpath, size) for every file object under the store's data subdirs."""
    out = []
    for sub in subdirs:
        p = f"{store}/{sub}"
        if not fs.exists(p):
            continue
        for k, v in fs.find(p, detail=True).items():
            if v.get("type") == "file":
                out.append((k[len(store):].lstrip("/"), v.get("size", 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-store", required=True,
                    help="gs://bucket/prefix of the Icechunk store")
    ap.add_argument("--dest", required=True, help="key prefix under the S3 bucket")
    ap.add_argument("--bucket", default="e4drr-project")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--sa-key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    ap.add_argument("--include-overwritten", action="store_true",
                    help="also copy overwritten/ (superseded ref versions; not needed)")
    args = ap.parse_args()

    assert args.gcs_store.startswith("gs://"), "--gcs-store must be gs://..."
    store = args.gcs_store[5:].rstrip("/")
    dest = args.dest.rstrip("/")

    fs = gcsfs.GCSFileSystem(token=args.sa_key)
    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                      config=Config(s3={"addressing_style": "path"},
                                    max_pool_connections=args.threads + 4))

    # destination inventory (name -> size) for resume/skip
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=args.bucket, Prefix=dest + "/"):
        for o in page.get("Contents", []):
            have[o["Key"]] = o["Size"]
    print(f"destination already holds {len(have)} objects under {dest}/")

    subdirs = DATA_SUBDIRS + (["overwritten"] if args.include_overwritten else [])
    print("listing source objects ...", flush=True)
    t_list = time.time()
    src = list_source(fs, store, subdirs)
    total_bytes = sum(sz for _, sz in src)
    todo = [(rel, sz) for rel, sz in src
            if have.get(f"{dest}/{rel}") != sz]
    todo_bytes = sum(sz for _, sz in todo)
    print(f"source: {len(src)} objects ({total_bytes/1e9:.2f} GB), "
          f"listed in {time.time()-t_list:.0f}s")
    print(f"to upload: {len(todo)} objects ({todo_bytes/1e9:.2f} GB); "
          f"{len(src)-len(todo)} already in sync")

    def put(rel):
        data = fs.cat_file(f"{store}/{rel}")   # single GET; single PUT (no multipart)
        s3.put_object(Bucket=args.bucket, Key=f"{dest}/{rel}", Body=data)
        return len(data)

    t0 = time.time()
    done = done_bytes = 0
    errors = []
    expired = False
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(put, rel): rel for rel, _ in todo}
        for fut in as_completed(futs):
            try:
                done_bytes += fut.result()
                done += 1
            except Exception as e:
                errors.append((futs[fut], str(e)))
                if "ExpiredToken" in str(e) or "AccessDenied" in str(e):
                    expired = True
            if done and done % 2000 == 0:
                el = time.time() - t0
                rate = done / el
                eta = (len(todo) - done) / rate if rate else float("nan")
                print(f"  {done}/{len(todo)} objs ({done_bytes/1e9:.2f} GB) "
                      f"{rate:.0f} obj/s  ETA {eta/60:.1f} min", flush=True)
            if expired:
                break

    if expired:
        print(f"\nSTS token expired/denied after {done} uploads. "
              f"Refresh .env (source .env) and re-run -- it resumes from here.")
        raise SystemExit(2)

    # Upload the branch-pointer `repo` file LAST, now that all data it can
    # reference is present at the destination.
    if not errors:
        repo_key = f"{dest}/repo"
        if have.get(repo_key) != fs.info(f"{store}/repo").get("size"):
            s3.put_object(Bucket=args.bucket, Key=repo_key,
                          Body=fs.cat_file(f"{store}/repo"))
            print("uploaded repo pointer (last)")
        else:
            print("repo pointer already in sync")

    ok = len(todo) - len(errors)
    print(f"\nuploaded {ok}/{len(todo)} data objects in {time.time()-t0:.1f}s "
          f"-> s3://{args.bucket}/{dest}")
    if errors:
        print(f"{len(errors)} errors, first: {errors[0]}")
        raise SystemExit(1)
    print("MIRROR COMPLETE")


if __name__ == "__main__":
    main()
