"""Restore the EWC object-store credentials on the Dask workers.

Why this is needed: reading the published ECMWF store requires AWS_* to be
absent from the worker process (the EWC Ceph endpoint otherwise hijacks the
virtual-chunk fetch and it dies on a DNS lookup). Earlier versions of the
tooling here scrubbed AWS_* *permanently* -- via a sticky WorkerPlugin, and
via a task function that popped the variables and never put them back. Either
one leaves workers unable to reach must-icechunk, and it affects unrelated
jobs on the same cluster, not just ours.

Both root causes are fixed:
  * materialize_ea_icechunk_ewc.py  -- release_cluster() unregisters the plugin
  * realize_smoke_test.py           -- save/restore around the source open

This script repairs workers that were left stripped by the earlier versions.

    check    report which workers have credentials
    restore  push them back into the running processes, from .env (no restart,
             safe while a job is in flight)
    restart  bounce the workers so they inherit the pristine service
             environment -- the durable fix, but it KILLS RUNNING WORK

Usage
-----
    P=/opt/mamba/envs/dask/bin/python
    $P fix_worker_credentials.py check
    $P fix_worker_credentials.py restore
    $P fix_worker_credentials.py restart
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ENDPOINT = "https://object-store.os-api.cci1.ecmwf.int"
REGION = "RegionOne"


def load_env(path=".env"):
    env = {}
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"{path} not found -- needed for AK/SK")
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        env[k.strip().replace("export ", "")] = v.strip().strip('"').strip("'")
    ak = env.get("AK") or env.get("AWS_ACCESS_KEY_ID")
    sk = env.get("SK") or env.get("AWS_SECRET_ACCESS_KEY")
    if not ak or not sk:
        sys.exit(f"{path}: need AK and SK")
    return ak, sk


def report(client):
    """Which workers can see credentials, and can they actually authenticate?"""
    def probe():
        import os
        have = sorted(k for k in os.environ if k.startswith("AWS_"))
        ok = None
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            try:
                import s3fs
                fs = s3fs.S3FileSystem(
                    key=os.environ["AWS_ACCESS_KEY_ID"],
                    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
                    client_kwargs={
                        "endpoint_url": os.environ.get("AWS_ENDPOINT_URL"),
                        "region_name": os.environ.get("AWS_DEFAULT_REGION")})
                ok = "must-icechunk" in [b.strip("/")
                                         for b in fs.ls("", detail=False)]
            except Exception as e:                              # noqa: BLE001
                ok = f"FAIL {type(e).__name__}"
        return {"vars": have, "s3_ok": ok}

    res = client.run(probe)
    good = 0
    for addr in sorted(res):
        r = res[addr]
        has = bool(r["vars"])
        good += bool(r["s3_ok"] is True)
        print(f"  {addr.split('/')[-1]:22s} "
              f"{'AWS_* present' if has else 'AWS_* MISSING':16s} "
              f"{len(r['vars'])} vars  s3={r['s3_ok']}")
    print(f"\n  {good}/{len(res)} workers can reach must-icechunk")
    return good, len(res)


def cmd_check(args, client):
    print("credential status:")
    report(client)


def cmd_restore(args, client):
    ak, sk = load_env(args.env)

    def setter(ak=ak, sk=sk, ep=ENDPOINT, rg=REGION):
        import os
        os.environ["AWS_ACCESS_KEY_ID"] = ak
        os.environ["AWS_SECRET_ACCESS_KEY"] = sk
        os.environ["AWS_ENDPOINT_URL"] = ep
        os.environ["AWS_DEFAULT_REGION"] = rg
        return True

    print(f"before:")
    report(client)
    client.run(setter)
    print(f"\nafter pushing AWS_* from {args.env} into the worker processes:")
    good, total = report(client)
    if good < total:
        print("\n  NOTE: some workers still cannot authenticate. If a job that "
              "scrubs\n  AWS_* is in flight it will strip them again -- re-run "
              "once it finishes,\n  or use `restart` for the durable fix.")


def cmd_restart(args, client):
    n = len(client.scheduler_info()["workers"])
    print(f"restarting {n} workers -- THIS KILLS ANY RUNNING WORK")
    client.restart(timeout=300)
    client.wait_for_workers(n, timeout=300)
    print("\nafter restart (service environment restored):")
    report(client)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "restore", "restart"])
    ap.add_argument("--scheduler", default=os.environ.get(
        "DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"))
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    from dask.distributed import Client
    client = Client(args.scheduler, timeout=60)
    try:
        {"check": cmd_check, "restore": cmd_restore,
         "restart": cmd_restart}[args.command](args, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
