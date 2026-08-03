"""One place to answer "what state is the EWC Dask cluster in?".

Written because the same questions kept being answered with throwaway inline
snippets, inconsistently, and that led to misreadings -- a credential count
taken mid-task, and workers that had silently restarted being spotted only
because their port numbers changed.

What it reports, and why each matters here:

  memory        The important one. Reading a source array leaves several GB
                resident per worker and it is NOT released (see BLOCKERS.md
                section 5). `managed` is what Dask knows about; the gap between
                `managed` and `rss` is the part Dask cannot see and will not
                account for when scheduling -- which is exactly why it
                over-subscribes workers and the nanny kills them.

  headroom      rss against memory_limit. Dask spills at 70%, pauses at 80%,
                kills at 95%. A worker above ~70% with nothing running is
                holding leaked memory and should be restarted.

  identity      worker address + PID. If these change between runs the workers
                restarted -- which silently destroys in-flight futures and
                resets any cached state the tasks relied on.

  credentials   AWS_* present AND actually able to reach must-icechunk. Present
                is not the same as working, and earlier tooling here stripped
                them mid-run.

  tasks         what is actually executing, so "the cluster is idle" is a
                measurement rather than an assumption.

Usage
-----
    P=/opt/mamba/envs/dask/bin/python
    $P cluster_status.py                 # the standard check
    $P cluster_status.py --source        # also probe the AWS read path
    $P cluster_status.py --watch 30      # re-check every 30 s
"""
from __future__ import annotations

import argparse
import os
import time


def worker_probe():
    """Runs on each worker. Cheap, no network unless asked."""
    import os
    out = {"pid": os.getpid()}
    try:
        with open("/proc/self/statm") as f:
            out["rss_gb"] = int(f.read().split()[1]) * 4096 / 1e9
    except OSError:
        out["rss_gb"] = None
    out["aws_vars"] = sorted(k for k in os.environ if k.startswith("AWS_"))
    # NOTE: deliberately not reporting whether a task cached a source
    # handle. Task functions shipped by value get a synthetic module
    # namespace, so checking globals() here inspects the wrong dict and
    # always says "no". Better absent than misleading.
    return out


def s3_probe():
    """Can this worker actually reach must-icechunk? Present != working."""
    import os
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return "no AWS_ACCESS_KEY_ID"
    try:
        import s3fs
        fs = s3fs.S3FileSystem(
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            client_kwargs={"endpoint_url": os.environ.get("AWS_ENDPOINT_URL"),
                           "region_name": os.environ.get("AWS_DEFAULT_REGION")})
        return "OK" if "must-icechunk" in [b.strip("/")
                                           for b in fs.ls("", detail=False)] \
            else "no bucket"
    except Exception as e:                                      # noqa: BLE001
        return f"FAIL {type(e).__name__}"


def source_probe():
    """Is the AWS virtual-chunk path healthy right now? One small range GET.

    Deliberately opt-in (--source): we have been throttled by hammering this
    bucket, and a status command should not add to that by default.
    """
    import urllib.request, urllib.error, time
    url = ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
           "20260702/00z/ifs/0p25/enfo/20260702000000-0h-enfo-ef.grib2")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1999999"})
    try:
        t0 = time.time()
        n = len(urllib.request.urlopen(req, timeout=60).read())
        return f"OK {n/1e6/(time.time()-t0):.1f} MB/s"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read()[:200].decode("utf-8", "replace")
        except Exception:
            pass
        code = "SlowDown" if "SlowDown" in body else str(e.code)
        return f"HTTP {e.code} {code}"
    except Exception as e:                                      # noqa: BLE001
        return f"FAIL {type(e).__name__}"


def report(client, args):
    info = client.scheduler_info()["workers"]
    n = len(info)
    print(f"scheduler  {args.scheduler}")
    print(f"workers    {n}   threads {sum(w['nthreads'] for w in info.values())}"
          f"   total limit "
          f"{sum(w['memory_limit'] for w in info.values())/1e9:.0f} GB")

    try:
        probes = client.run(worker_probe)
    except Exception as e:                                      # noqa: BLE001
        print(f"  worker probe failed: {type(e).__name__} -- "
              f"cluster may be mid-restart")
        probes = {}
    creds = client.run(s3_probe) if args.creds else {}

    print(f"\n{'worker':16s} {'addr':22s} {'pid':>7s} {'rss':>8s} "
          f"{'managed':>8s} {'limit':>7s} {'use':>6s} {'exec':>5s} "
          f"{'AWS_*':>6s} {'s3':>10s}")
    print("-" * 110)
    tot_rss = tot_managed = 0.0
    hot = []
    for addr in sorted(info):
        w = info[addr]
        m = w["metrics"]
        rss = m.get("memory", 0) / 1e9
        managed = m.get("managed_bytes", 0) / 1e9
        lim = w["memory_limit"] / 1e9
        use = rss / lim if lim else 0
        tc = m.get("task_counts", {}) or {}
        ex = tc.get("executing", 0)
        p = probes.get(addr, {})
        tot_rss += rss
        tot_managed += managed
        if use > 0.70 and ex == 0:
            hot.append((w["id"], rss, use))
        print(f"{w['id']:16s} {addr.split('/')[-1]:22s} "
              f"{p.get('pid', '?'):>7} {rss:7.2f}G {managed:7.2f}G "
              f"{lim:6.1f}G {use*100:5.0f}% {ex:5d} "
              f"{len(p.get('aws_vars', [])):6d} "
              f"{str(creds.get(addr, '-')):>10s}")
    print("-" * 110)
    print(f"{'TOTAL':16s} {'':22s} {'':>7s} {tot_rss:7.2f}G {tot_managed:7.2f}G")

    # --- interpretation, not just numbers ------------------------------------
    print()
    unmanaged = tot_rss - tot_managed
    if unmanaged > 1.0:
        print(f"  ! {unmanaged:.1f} GB is UNMANAGED (rss minus what Dask "
              f"tracks).")
        print(f"    Dask does not count this when scheduling, so it will "
              f"over-subscribe")
        print(f"    these workers and the nanny will kill them. See "
              f"BLOCKERS.md section 5.")
    if hot:
        print(f"  ! {len(hot)} worker(s) above 70% of limit with NOTHING "
              f"running:")
        for wid, rss, use in hot:
            print(f"      {wid}  {rss:.2f} GB  ({use*100:.0f}%)")
        print(f"    That is leaked memory. Reclaim it with:")
        print(f"      python fix_worker_credentials.py restart")
    if args.creds:
        ok = sum(1 for v in creds.values() if v == "OK")
        print(f"  {'.' if ok == n else '!'} {ok}/{n} workers can reach "
              f"must-icechunk")
    if not hot and unmanaged <= 1.0:
        print("  . workers look clean")

    if args.source:
        print(f"\nAWS virtual-chunk path (one 2 MB range GET per worker):")
        for addr, v in sorted(client.run(source_probe).items()):
            print(f"  {addr.split('/')[-1]:22s} {v}")
        print("  (503 SlowDown = we are rate-limited; back off, do not retry "
              "harder)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheduler", default=os.environ.get(
        "DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"))
    ap.add_argument("--source", action="store_true",
                    help="also probe the AWS read path (opt-in: we have been "
                         "throttled for hammering it)")
    ap.add_argument("--no-creds", dest="creds", action="store_false",
                    help="skip the per-worker must-icechunk auth check")
    ap.add_argument("--watch", type=int, default=0,
                    help="re-check every N seconds")
    args = ap.parse_args()

    from dask.distributed import Client
    client = Client(args.scheduler, timeout=60)
    try:
        while True:
            print("=" * 110)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            report(client, args)
            if not args.watch:
                break
            time.sleep(args.watch)
    finally:
        client.close()


if __name__ == "__main__":
    main()
