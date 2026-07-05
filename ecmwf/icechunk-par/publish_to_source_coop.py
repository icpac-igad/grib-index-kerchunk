# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Publish a locally-built Icechunk store to source.coop by plain file sync.

Why this exists: Icechunk COMMITS cannot go through the data.source.coop proxy
(missing S3 ops) and the proxy creds are 1-hour STS tokens. So the build runs
locally and credential-free (see build_0p4_ensemble_icechunk.py), and only this
upload needs credentials. Icechunk's local layout is byte-identical to its
object-store layout, so a plain recursive PUT is a valid published repo; readers
need only anonymous GET/LIST.

Resumable: skips objects already present at the destination with the same size,
so if the STS token expires mid-run you just refresh .env and re-run -- it picks
up where it stopped. Run it repeatedly during a long build to keep source.coop
incrementally in sync (each new date is one sealed manifest shard, ~7 MB).

Usage:
  source .env            # fresh 1-hour STS creds from the source.coop dashboard
  uv run publish_to_source_coop.py \
      --local /tmp/ecmwf-0p4-real \
      --dest  forecasts/ecmwf_ifs_0p4_icechunk   # bucket = e4drr-project
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="local Icechunk store dir")
    ap.add_argument("--dest", required=True, help="key prefix under the bucket")
    ap.add_argument("--bucket", default="e4drr-project")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    src = Path(args.local)
    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                      config=Config(s3={"addressing_style": "path"},
                                    max_pool_connections=args.threads + 4))

    # existing objects (name -> size) for resume/skip
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=args.bucket, Prefix=args.dest.rstrip("/") + "/"):
        for o in page.get("Contents", []):
            have[o["Key"]] = o["Size"]

    files = [p for p in src.rglob("*") if p.is_file()]
    todo = [p for p in files
            if have.get(f"{args.dest}/{p.relative_to(src)}") != p.stat().st_size]
    skip = len(files) - len(todo)
    total = sum(p.stat().st_size for p in todo)
    print(f"{len(files)} objects local, {skip} already published, "
          f"{len(todo)} to upload ({total/1e6:.1f} MB)")
    if not todo:
        print("destination already in sync.")
        return

    t0 = time.time()
    errors = []

    def put(p):
        s3.upload_file(str(p), args.bucket, f"{args.dest}/{p.relative_to(src)}")

    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(put, p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:
                errors.append((str(futs[fut]), str(e)))
                if "ExpiredToken" in str(e):
                    print(f"\nSTS token expired after {i} uploads. "
                          f"Refresh .env and re-run -- it resumes from here.")
                    break
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} uploaded ({time.time()-t0:.0f}s)")

    ok = len(todo) - len(errors)
    print(f"uploaded {ok}/{len(todo)} objects in {time.time()-t0:.1f}s "
          f"-> {args.bucket}/{args.dest}")
    if errors and "ExpiredToken" not in errors[0][1]:
        print(f"{len(errors)} errors, first: {errors[0]}")


if __name__ == "__main__":
    main()
