#!/usr/bin/env python
"""Mirror a REALIZED Icechunk store from EWC Ceph to source.coop.

Sibling of `mirror_gcs_to_source_coop.py`, which does the same job for the
GCS-hosted *virtual* stores. The difference that matters is what is being
moved. A virtual store is metadata: a few GB of manifests pointing at GRIB
bytes that stay in AWS. A realized store IS the data -- 215,280 native chunk
objects and ~1.4 TB for one 92-date season -- so throughput, not correctness
of the virtual references, is the whole problem.

Why this exists at all: `must-icechunk` on Ceph has a 5 TB hard limit and the
three MAM seasons come to ~4.4 TB on their own. source.coop cannot replace
Ceph as the build target -- its proxy will not take icechunk commits, and even
bulk DeleteObjects returns NoSuchBucket -- so the only way the ceiling stops
binding is the cycle

    build season -> verify -> mirror here -> verify -> delete from Ceph

which holds Ceph at roughly one season (~1.5 TB) instead of all of them.

MEASURED THROUGHPUT (2026-08-19, against a live source.coop STS token, with
the cluster concurrently running a realization so these are contended):

    48 threads, 1 process, gateway      55 MB/s     199 GB/h
    96 threads, 1 process, gateway      78 MB/s     282 GB/h   2 errors
    48 threads x 2 processes, gateway   78 MB/s     281 GB/h
    48 threads x 4 processes, gateway   90 MB/s     325 GB/h   1 error
    48 threads x 6 VMs                 166 MB/s     599 GB/h   0 errors

Two things follow, and the first is the reason this script forks at all.
**Threads plateau; processes keep scaling.** 96 threads in one process is
slower than 48 in two, and starts throwing errors -- the same per-process
connection cap SCALING.md documents for icechunk reads on the way in. So
`--procs` is the real lever and `--threads` should be left alone.

And a 60-minute STS token is worth ~325 GB from this host, ~600 GB spread
across the six VMs. That is not a cap on the total: the run skips objects
already present at the same size, so refreshing the token and re-running
resumes. ~4.4 TB is ~14 h of gateway time, or ~7.5 h if driven from the VMs.

Correctness, inherited from the GCS version and equally load-bearing here:
the top-level `repo` file names the branch tips, so it is uploaded LAST,
after every snapshot and manifest it can reference is already at the
destination. A half-finished mirror is then an incomplete store, never a
complete-looking store pointing at objects that are not there.

`overwritten/` is skipped: icechunk never reads it on open (superseded `repo`
versions from ref rotation).

Usage:
    source <this folder>/.env
    $P mirror_ceph_to_source_coop.py \
        --src-prefix ea-swio/v3-49r1-mam2026 \
        --dest       forecasts/ea-swio/v3-49r1-mam2026 \
        --procs 4 --env .env

    $P mirror_ceph_to_source_coop.py ... --dry-run     # plan only
    $P mirror_ceph_to_source_coop.py ... --prune       # after it reads back
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CEPH_ENDPOINT = "https://object-store.os-api.cci1.ecmwf.int"
DATA_SUBDIRS = ["manifests", "snapshots", "transactions", "chunks", "refs"]


def load_env(path):
    creds = {}
    with open(path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                creds[k.strip().replace("export ", "")] = v.strip().strip('"').strip("'")
    return creds


def ceph_client(ak, sk, pool=64):
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=CEPH_ENDPOINT, aws_access_key_id=ak,
                        aws_secret_access_key=sk, region_name="RegionOne",
                        config=Config(max_pool_connections=pool))


def sc_client(pool=64):
    """source.coop, from the AWS_* STS variables in the environment."""
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}, max_pool_connections=pool))


def list_prefix(client, bucket, prefix):
    """(key, size) for everything under prefix."""
    out = []
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out.append((o["Key"], o["Size"]))
    return out


def _worker(args):
    """One process: copy its slice of the work list, Ceph GET -> source.coop PUT."""
    shard, todo, cfg = args
    ceph = ceph_client(cfg["ak"], cfg["sk"], cfg["threads"] + 8)
    sc = sc_client(cfg["threads"] + 8)
    done = [0, 0, 0]          # objects, bytes, errors
    expired = [False]

    def one(item):
        rel, size = item
        if expired[0]:
            return
        try:
            body = ceph.get_object(Bucket=cfg["src_bucket"],
                                   Key=f"{cfg['src_prefix']}/{rel}")["Body"].read()
        except Exception:
            done[2] += 1
            return
        # Under load the proxy returns a bare 520 that botocore's retry policy
        # does not recognise. Retry here or lose objects to a transient blip.
        for attempt in range(4):
            try:
                sc.put_object(Bucket=cfg["bucket"], Key=f"{cfg['dest']}/{rel}",
                              Body=body)
                done[0] += 1
                done[1] += len(body)
                return
            except Exception as e:                                # noqa: BLE001
                msg = str(e)
                if "ExpiredToken" in msg or "AccessDenied" in msg:
                    expired[0] = True
                    done[2] += 1
                    return
                if attempt == 3:
                    done[2] += 1
                    return
                time.sleep(0.5 * 2 ** attempt)

    with ThreadPoolExecutor(cfg["threads"]) as ex:
        list(ex.map(one, todo))
    return shard, done[0], done[1], done[2], expired[0]


def _dask_shard(shard, cfg, sc_env):
    """Runs ON a worker VM. Shells out to the frisky venv, which has boto3;
    the pinned /opt/mamba/envs/dask does not and must not be made to.

    Credentials travel as arguments and then as subprocess env -- never
    written to disk on the VM.
    """
    import json, os, socket, subprocess, tempfile, textwrap
    fd, path = tempfile.mkstemp(suffix=".json", prefix="mirror_shard_")
    with os.fdopen(fd, "w") as f:
        json.dump(shard, f)
    code = textwrap.dedent("""
        import boto3, json, os, time, sys
        from concurrent.futures import ThreadPoolExecutor
        from botocore.config import Config
        cfg = json.loads(os.environ["MIRROR_CFG"])
        shard = json.load(open(os.environ["MIRROR_SHARD"]))
        ceph = boto3.client("s3", endpoint_url=cfg["ceph_endpoint"],
            aws_access_key_id=cfg["ak"], aws_secret_access_key=cfg["sk"],
            region_name="RegionOne",
            config=Config(max_pool_connections=cfg["threads"]+8))
        sc = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=os.environ.get("AWS_DEFAULT_REGION","us-east-1"),
            config=Config(s3={"addressing_style":"path"},
                          max_pool_connections=cfg["threads"]+8))
        st = {"n":0,"b":0,"e":0,"expired":False}
        def one(item):
            rel, size = item
            if st["expired"]: return
            try:
                body = ceph.get_object(Bucket=cfg["src_bucket"],
                          Key=cfg["src_prefix"]+"/"+rel)["Body"].read()
            except Exception:
                st["e"] += 1; return
            for a in range(4):
                try:
                    sc.put_object(Bucket=cfg["bucket"],
                                  Key=cfg["dest"]+"/"+rel, Body=body)
                    st["n"] += 1; st["b"] += len(body); return
                except Exception as ex:
                    m = str(ex)
                    if "ExpiredToken" in m or "AccessDenied" in m:
                        st["expired"] = True; st["e"] += 1; return
                    if a == 3: st["e"] += 1; return
                    time.sleep(0.5 * 2**a)
        t0 = time.time()
        with ThreadPoolExecutor(cfg["threads"]) as ex: list(ex.map(one, shard))
        st["secs"] = time.time()-t0; st["host"] = os.uname().nodename
        print(json.dumps(st))
    """)
    env = dict(os.environ)
    env["MIRROR_CFG"] = json.dumps(cfg)
    env["MIRROR_SHARD"] = path
    env.update(sc_env)
    # boto3 validates TLS against certifi's bundle, which does not carry the
    # ECMWF internal CA that fronts the Ceph endpoint. icechunk is Rust and uses
    # the system trust store, so the frisky build writes to Ceph happily while
    # boto3 on the same VM fails with SSLError -- and because the shard only
    # fetches from Ceph, the failure shows up as silence: processes alive, no
    # traffic, no objects, no error. Point boto3 at the system bundle, which
    # does carry the CA. Harmless where the path is absent.
    for _ca in ("/etc/ssl/certs/ca-certificates.crt",
                "/etc/pki/tls/certs/ca-bundle.crt"):
        if os.path.exists(_ca):
            env.setdefault("AWS_CA_BUNDLE", _ca)
            break
    # Scale the timeout to the shard, do not hardcode it. A flat 7200 s was
    # fine for the 302 GB tail (~42 min/shard) and killed 49r1-mam2026
    # mid-transfer: 1,162 GB / 6 = 194 GB per shard, ~2.7 h at the ~20 MB/s a
    # single VM sustains. Budget 4 MB/s -- five times pessimistic -- so a slow
    # VM or a source.coop wobble cannot trip it, with a 1 h floor.
    shard_bytes = sum(sz for _, sz in shard)
    budget = max(3600, int(shard_bytes / (4 * 1024 * 1024)) + 600)
    try:
        r = subprocess.run(["/tmp/frisky-ea/.venv/bin/python", "-c", code],
                           capture_output=True, text=True, env=env,
                           timeout=budget)
        out = r.stdout.strip().splitlines()
        return json.loads(out[-1]) if out else {"host": socket.gethostname(),
                                                "error": r.stderr[-300:]}
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-prefix", required=True,
                    help="store prefix under the Ceph bucket, e.g. ea-swio/v3-49r1-mam2026")
    ap.add_argument("--dest", required=True, help="key prefix under the source.coop bucket")
    ap.add_argument("--src-bucket", default="must-icechunk")
    ap.add_argument("--bucket", default="e4drr-project")
    ap.add_argument("--env", default=".env", help=".env holding the Ceph AK/SK")
    ap.add_argument("--procs", type=int, default=4,
                    help="THE lever. Threads plateau, processes scale -- see the "
                         "measurements in the module docstring")
    ap.add_argument("--threads", type=int, default=48, help="per process; 48 is the knee")
    ap.add_argument("--include-overwritten", action="store_true")
    ap.add_argument("--via-dask", metavar="ADDR", default=None,
                    help="dispatch across the worker VMs through the Dask "
                         "cluster (used purely as a remote-exec channel). "
                         "Measured 166 MB/s vs 90 from this host alone. "
                         "Requires the frisky venv on each VM to have boto3 "
                         "and the realization to be STOPPED -- they compete")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prune", action="store_true",
                    help="delete destination objects the source does not have. "
                         "Only after the new repo is in place and reads back")
    args = ap.parse_args()

    for v in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if v not in os.environ:
            sys.exit(f"{v} not set -- `source` the source.coop .env first")

    creds = load_env(args.env)
    src_prefix = args.src_prefix.rstrip("/")
    dest = args.dest.rstrip("/")
    ceph = ceph_client(creds["AK"], creds["SK"])
    sc = sc_client()

    print(f"source  ceph://{args.src_bucket}/{src_prefix}")
    print(f"dest    s3://{args.bucket}/{dest}")
    print(f"workers {args.procs} processes x {args.threads} threads\n")

    print("listing source ...", flush=True)
    t = time.time()
    src_all = list_prefix(ceph, args.src_bucket, src_prefix + "/")
    keep = DATA_SUBDIRS + (["overwritten"] if args.include_overwritten else [])
    src, repo_obj = [], None
    for key, size in src_all:
        rel = key[len(src_prefix) + 1:]
        if rel == "repo":
            repo_obj = size
        elif rel.split("/", 1)[0] in keep:
            src.append((rel, size))
    print(f"  {len(src)} objects, {sum(s for _, s in src)/1e9:.2f} GB "
          f"in {time.time()-t:.0f}s"
          + ("" if repo_obj is not None else "   WARNING: no `repo` object found"))

    have = {k: s for k, s in list_prefix(sc, args.bucket, dest + "/")}
    print(f"  destination already holds {len(have)} objects")
    todo = [(rel, sz) for rel, sz in src if have.get(f"{dest}/{rel}") != sz]
    tb = sum(sz for _, sz in todo)
    print(f"  to upload: {len(todo)} objects, {tb/1e9:.2f} GB "
          f"({len(src)-len(todo)} already in sync)")
    if tb:
        # Rate depends on the mode actually chosen. Printing the single-host
        # number for a --via-dask run said "56 min" for a 6-VM transfer.
        rate, how = ((118.0, "6 VMs, measured") if args.via_dask
                     else (90.0, f"{args.procs} procs, one host"))
        print(f"  at {rate:.0f} MB/s ({how}): {tb/1e6/rate/60:.1f} min "
              f"-> {tb/1e6/rate/3600:.2f} h")

    if args.dry_run:
        print("\nDRY RUN -- nothing uploaded")
        return 0
    if not todo:
        print("\nnothing to upload")
    else:
        cfg = dict(ak=creds["AK"], sk=creds["SK"], src_bucket=args.src_bucket,
                   src_prefix=src_prefix, bucket=args.bucket, dest=dest,
                   threads=args.threads, ceph_endpoint=CEPH_ENDPOINT)
        t0 = time.time()
        if args.via_dask:
            from distributed import Client
            sc_env = {k: os.environ[k] for k in
                      ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID",
                       "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
                       "AWS_SESSION_TOKEN") if k in os.environ}
            with Client(args.via_dask, timeout=60) as client:
                addrs = sorted(client.scheduler_info()["workers"])
                print(f"dispatching across {len(addrs)} VMs x {args.threads} threads")
                shards = [todo[i::len(addrs)] for i in range(len(addrs))]
                futs = [client.submit(_dask_shard, sh, cfg, sc_env,
                                      workers=[a], pure=False)
                        for a, sh in zip(addrs, shards)]
                out = client.gather(futs)
            ok = by = er = 0; exp = False
            for d in out:
                if "error" in d:
                    print(f"  {d.get('host','?')}: FAILED {d['error'][:150]}")
                    er += 1; continue
                print(f"  {d['host']:<18}{d['b']/1e6/max(d['secs'],1e-9):>8.1f} MB/s"
                      f"  {d['n']:>7} objs  {d['e']} errors")
                ok += d["n"]; by += d["b"]; er += d["e"]; exp |= d["expired"]
        else:
            shards = [todo[i::args.procs] for i in range(args.procs)]
            with mp.Pool(args.procs) as pool:
                res = pool.map(_worker, [(i, s, cfg) for i, s in enumerate(shards)])
            ok = sum(r[1] for r in res); by = sum(r[2] for r in res)
            er = sum(r[3] for r in res); exp = any(r[4] for r in res)
        el = time.time() - t0
        print(f"\nuploaded {ok}/{len(todo)} objects, {by/1e9:.2f} GB in {el:.0f}s "
              f"= {by/1e6/el:.1f} MB/s ({by/1e9/el*3600:.0f} GB/h), {er} errors")
        if exp:
            print("STS token expired. Refresh the .env, `source` it, and re-run "
                  "-- this resumes from where it stopped.")
            return 2
        if er:
            print("errors remain -- re-run to retry them before uploading `repo`")
            return 1

    # `repo` LAST: only now is everything it can point at present.
    if repo_obj is not None:
        body = ceph.get_object(Bucket=args.src_bucket,
                               Key=f"{src_prefix}/repo")["Body"].read()
        if have.get(f"{dest}/repo") != len(body):
            sc.put_object(Bucket=args.bucket, Key=f"{dest}/repo", Body=body)
            print("uploaded `repo` pointer (last)")
        else:
            print("`repo` pointer already in sync")

    if args.prune:
        want = {f"{dest}/{rel}" for rel, _ in src} | {f"{dest}/repo"}
        dead = [k for k in have if k not in want]
        print(f"\nprune: {len(dead)} destination objects absent from the source")
        # source.coop rejects bulk DeleteObjects (NoSuchBucket); single works.
        with ThreadPoolExecutor(16) as ex:
            n = sum(1 for _ in ex.map(
                lambda k: sc.delete_object(Bucket=args.bucket, Key=k), dead))
        print(f"  deleted {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
