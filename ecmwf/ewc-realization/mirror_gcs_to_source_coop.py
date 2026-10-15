# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "gcsfs", "icechunk>=2.1"]
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

`--live-only` copies just what the `main` tip references -- that snapshot, its
manifests, and the native chunks -- instead of every object under the prefix.
A store built with restarts and retries carries a lot of dead weight: v4 was
92 GB on GCS against 18.7 GB reachable from its tip, 6,728 snapshot objects
against 1,179 in `main`'s history. At ~20 MB/s through the proxy that is the
difference between fitting one STS hour and not. What it gives up is history:
the published repo reads, but not *as of* an earlier commit. Same end state as
`expire_snapshots` + `garbage_collect`, without mutating the source store.
Safe to run against a store being written concurrently -- the tip is pinned
once and the pinned set is internally consistent; a later re-run copies the
delta.

`--prune` then deletes destination objects the source does not have, which is
how the remains of a previous store under that prefix go away. Run it only
after the new `repo` is in place and the store has been read back.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/coiled-data.json   # GCS read
  source .env                                                    # source.coop STS
  uv run mirror_gcs_to_source_coop.py \
      --gcs-store gs://gik-gefs-aws-tf/icechunk/gefs-ens \
      --dest      forecasts/noaa_gefs_aws_s3_icechunk_vd         # bucket = e4drr-project

  # replace a published store with the current state of a much larger one
  uv run mirror_gcs_to_source_coop.py --live-only --threads 80 \
      --gcs-store gs://gik-ecmwf-aws-tf/icechunk/ecmwf-ens-v4 \
      --dest      forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd
  uv run mirror_gcs_to_source_coop.py --live-only --prune ...    # after it reads
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


def live_only(gcs_store, fs, sa_key, src):
    """Filter a full source listing down to what the `main` tip references.

    Returns (live_objects, repo_bytes_at_pin). The `repo` bytes are read at the
    moment the tip is pinned and uploaded verbatim at the end, NOT re-read from
    GCS. Against a store still being written that distinction is the whole
    correctness argument: re-reading `repo` at the end would publish a pointer
    to a snapshot committed after the pin, whose manifests were never copied.

    Chunks are kept wholesale: there are a few dozen (the coordinate arrays),
    and attributing them to the tip is not worth a manifest walk.
    """
    import icechunk

    bucket, _, prefix = gcs_store[5:].partition("/")
    repo = icechunk.Repository.open(
        icechunk.gcs_storage(bucket=bucket, prefix=prefix.rstrip("/"),
                             service_account_file=sa_key),
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {"s3://ecmwf-forecasts/": icechunk.s3_anonymous_credentials(),
             "s3://noaa-gefs-pds/": icechunk.s3_anonymous_credentials()}))
    store = gcs_store[5:].rstrip("/")

    tip = repo.lookup_branch("main")
    repo_bytes = fs.cat_file(f"{store}/repo")     # pinned with the tip, see docstring
    manifests = repo.list_manifest_files(tip)
    sizes = dict(src)
    keys = [f"snapshots/{tip}"] + [f"manifests/{m.id}" for m in manifests]
    keys += [k for k in sizes if k.startswith("chunks/")]

    # A conversion running against the same store commits every ~30 s, so the
    # tip can reference objects written after the listing was taken. Size those
    # few directly rather than re-walking the whole prefix.
    late = [k for k in keys if k not in sizes]
    for k in late:
        sizes[k] = fs.info(f"{store}/{k}")["size"]
    if late:
        print(f"  {len(late)} objects newer than the listing, sized directly")

    live = [(k, sizes[k]) for k in keys]
    print(f"live set: tip {tip}, {len(manifests)} manifests, {len(live)} objects, "
          f"{sum(s for _, s in live)/1e9:.2f} GB "
          f"(of {len(src)} / {sum(s for _, s in src)/1e9:.2f} GB in the store)")
    return live, repo_bytes


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
    ap.add_argument("--live-only", action="store_true",
                    help="copy only what the `main` tip references (drops history)")
    ap.add_argument("--prune", action="store_true",
                    help="delete destination objects absent from the source set")
    args = ap.parse_args()

    assert args.gcs_store.startswith("gs://"), "--gcs-store must be gs://..."
    store = args.gcs_store[5:].rstrip("/")
    dest = args.dest.rstrip("/")

    # No listings cache. Against a store that is still being written, `main`
    # moves on while this runs, and a cached directory listing answers
    # FileNotFoundError for objects the new tip references -- on the size probe
    # and again on the read -- instead of going to GCS for them.
    fs = gcsfs.GCSFileSystem(token=args.sa_key, use_listings_cache=False)
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
    repo_bytes = None
    if args.live_only:
        src, repo_bytes = live_only(args.gcs_store, fs, args.sa_key, src)
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
        # Under load the proxy returns a bare 520, which botocore's retry policy
        # does not recognise -- retry here or lose objects to a transient blip.
        for attempt in range(4):
            try:
                s3.put_object(Bucket=args.bucket, Key=f"{dest}/{rel}", Body=data)
                return len(data)
            except Exception as e:                          # noqa: BLE001
                if attempt == 3 or "ExpiredToken" in str(e) or "AccessDenied" in str(e):
                    raise
                time.sleep(0.5 * 2 ** attempt)

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
        body = repo_bytes if repo_bytes is not None else fs.cat_file(f"{store}/repo")
        if have.get(repo_key) != len(body):
            s3.put_object(Bucket=args.bucket, Key=repo_key, Body=body)
            print("uploaded repo pointer (last)")
        else:
            print("repo pointer already in sync")

    ok = len(todo) - len(errors)
    print(f"\nuploaded {ok}/{len(todo)} data objects in {time.time()-t0:.1f}s "
          f"-> s3://{args.bucket}/{dest}")
    if errors:
        print(f"{len(errors)} errors, first: {errors[0]}")
        raise SystemExit(1)

    if args.prune:
        keep = {f"{dest}/{rel}" for rel, _ in src} | {f"{dest}/repo"}
        stale = [k for k in have if k not in keep]
        print(f"prune: {len(stale)} destination objects not in the source set")
        # the source.coop proxy does not implement DeleteObjects -- one call each
        with ThreadPoolExecutor(args.threads) as ex:
            list(ex.map(lambda k: s3.delete_object(Bucket=args.bucket, Key=k), stale))
        print(f"pruned {len(stale)}")

    print("MIRROR COMPLETE")


if __name__ == "__main__":
    main()
