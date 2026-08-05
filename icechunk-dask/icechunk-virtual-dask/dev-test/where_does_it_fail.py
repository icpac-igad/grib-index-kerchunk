"""Client vs worker: same read, repeated. Which one actually fails, and why?

The contradiction this resolves: quick-run.py reads t2m from AWS on the
JupyterHub VM in ~2 s with finite 1.000, while the same VM issuing a raw
urllib range GET got HTTP 503 SlowDown three times in a row, and the Dask
workers mostly fail. Those cannot all be "the egress is throttled".

Candidate explanations, and what would distinguish them:

  A. Egress address is throttled          -> client AND workers both fail at
                                             a similar rate
  B. Workers leave by a different address -> egress IPs differ; one side fails
  C. It is the raw-HTTP probe that is wrong, because the AWS SDK retries 503
     internally and plain urllib does not
                                          -> icechunk reads succeed where raw
                                             GETs fail, on the SAME host
  D. Something about the Dask worker path -> client succeeds, workers fail,
     (env, pickling, concurrency)            same egress address

So: report the egress address from client and every worker, then run the
identical single-message icechunk read N times on each, and compare success
rates. No writing, no concurrency, one message at a time.

Usage
-----
    P=/opt/mamba/envs/dask/bin/python
    $P where_does_it_fail.py                 # 5 reads each side
    $P where_does_it_fail.py --reps 10 --raw # also do raw urllib GETs
"""
from __future__ import annotations

import argparse
import os
import time

SRC_ENDPOINT = "https://data.source.coop"
SRC_BUCKET = "e4drr-project"
SRC_PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
SRC_CONTAINER = "s3://ecmwf-forecasts/"


def egress_ip():
    """How AWS sees us. If client and workers differ, that alone explains a
    lot. checkip is an AWS endpoint, so it is the same network path."""
    import urllib.request
    try:
        return urllib.request.urlopen("https://checkip.amazonaws.com",
                                      timeout=30).read().decode().strip()
    except Exception as e:                                      # noqa: BLE001
        return f"FAIL {type(e).__name__}"


def raw_get():
    """Plain HTTP range GET -- no SDK, and crucially NO RETRY."""
    import urllib.request, urllib.error, time
    url = ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
           "20260702/00z/ifs/0p25/enfo/20260702000000-0h-enfo-ef.grib2")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-999999"})
    try:
        t0 = time.time()
        n = len(urllib.request.urlopen(req, timeout=60).read())
        return {"ok": True, "s": round(time.time() - t0, 2),
                "mb_s": round(n / 1e6 / (time.time() - t0), 1)}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read()[:200].decode("utf-8", "replace")
        except Exception:
            pass
        return {"ok": False, "why": f"HTTP {e.code}"
                + (" SlowDown" if "SlowDown" in body else "")}
    except Exception as e:                                      # noqa: BLE001
        return {"ok": False, "why": type(e).__name__}


_SRC = {}


def icechunk_read(date="2026-07-02", var="t2m"):
    """Exactly what quick-run.py does: one global message, via icechunk, whose
    Rust object-store client DOES retry throttling internally."""
    import os
    saved = {k: os.environ[k] for k in list(os.environ) if k.startswith("AWS_")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        import numpy as np, xarray as xr, icechunk, time
        import gribberish.zarr  # noqa: F401
        if "ds" not in _SRC:
            storage = icechunk.s3_storage(
                bucket=SRC_BUCKET, prefix=SRC_PREFIX,
                endpoint_url=SRC_ENDPOINT, region="us-east-1",
                anonymous=True, from_env=False, force_path_style=True)
            auth = icechunk.containers_credentials(
                {SRC_CONTAINER: icechunk.s3_anonymous_credentials()})
            cfg = icechunk.RepositoryConfig.default()
            cfg.manifest = icechunk.ManifestConfig(
                preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                                       max_arrays_to_scan=0))
            repo = icechunk.Repository.open(
                storage, config=cfg, authorize_virtual_chunk_access=auth)
            _SRC["ds"] = xr.open_zarr(repo.readonly_session("main").store,
                                      group="50r1/00z", consolidated=False,
                                      zarr_format=3, decode_timedelta=True)
        ds = _SRC["ds"]
        t0 = time.time()
        v = ds[var].sel(time=np.datetime64(date), number=0,
                        step=np.timedelta64(0, "h")).compute()
        dt = time.time() - t0
        return {"ok": True, "s": round(dt, 2),
                "finite": round(float(np.isfinite(v.values).mean()), 3),
                "mean": round(float(np.nanmean(v.values)), 1)}
    except Exception as e:                                      # noqa: BLE001
        t = " ".join(str(e).split())
        why = ("SlowDown" if "SlowDown" in t
               else "DNS" if "dns error" in t
               else "timeout" if "timeout" in t.lower()
               else type(e).__name__)
        # Keep the raw text: an earlier version reported only the label and
        # that hid what was actually going wrong.
        # the ROOT cause is at the END of icechunk's error chain
        return {"ok": False, "why": why, "raw": t[-260:]}
    finally:
        os.environ.update(saved)


def summarise(label, results):
    ok = [r for r in results if r.get("ok")]
    fails = {}
    for r in results:
        if not r.get("ok"):
            fails[r.get("why", "?")] = fails.get(r.get("why", "?"), 0) + 1
    times = [r["s"] for r in ok if "s" in r]
    print(f"  {label:34s} {len(ok)}/{len(results)} ok"
          + (f"   median {sorted(times)[len(times)//2]:.2f}s" if times else "")
          + (f"   failures: {fails}" if fails else ""))
    for r in results[:2]:
        if not r.get("ok") and r.get("raw"):
            print(f"      e.g. {r['raw'][:150]}")
    return len(ok), len(results)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--raw", action="store_true",
                    help="also do raw urllib GETs (no SDK retry) for contrast")
    ap.add_argument("--scheduler", default=os.environ.get(
        "DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"))
    ap.add_argument("--gap", type=float, default=2.0,
                    help="seconds between reads, to stay polite")
    args = ap.parse_args()

    from dask.distributed import Client
    client = Client(args.scheduler, timeout=60)
    workers = sorted(client.scheduler_info()["workers"])

    print("=" * 78)
    print("EGRESS ADDRESS as AWS sees it")
    print("=" * 78)
    mine = egress_ip()
    print(f"  {'jupyter client (this VM)':34s} {mine}")
    theirs = client.run(egress_ip)
    for a in sorted(theirs):
        print(f"  {a.split('/')[-1]:34s} {theirs[a]}")
    distinct = {mine} | set(theirs.values())
    print(f"\n  {len(distinct)} distinct address(es): {distinct}")
    if len(distinct) == 1:
        print("  -> client and workers share ONE egress address, so any "
              "AWS-side rate\n     limit applies to both equally. A difference "
              "in behaviour is NOT egress.")
    else:
        print("  -> client and workers leave by DIFFERENT addresses. An "
              "AWS-side limit\n     could hit one and not the other.")

    print("\n" + "=" * 78)
    print(f"SAME ICECHUNK READ, {args.reps}x EACH, one message at a time")
    print("=" * 78)

    cli = []
    for i in range(args.reps):
        cli.append(icechunk_read())
        time.sleep(args.gap)
    c_ok, c_n = summarise("jupyter client (like quick-run)", cli)

    w = workers[0]
    wk = []
    for i in range(args.reps):
        wk.append(client.submit(icechunk_read, workers=[w],
                                allow_other_workers=False,
                                pure=False).result(timeout=600))
        time.sleep(args.gap)
    w_ok, w_n = summarise(f"one dask worker", wk)

    if args.raw:
        print()
        rc = []
        for i in range(args.reps):
            rc.append(raw_get()); time.sleep(args.gap)
        summarise("raw urllib GET, client (no retry)", rc)
        rw = [client.submit(raw_get, workers=[w], allow_other_workers=False,
                            pure=False).result(timeout=300)
              for _ in range(args.reps)]
        summarise("raw urllib GET, worker (no retry)", rw)

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    if c_ok and w_ok:
        print("  Both sides work. The failures are intermittent, so any single")
        print("  observation (success OR 503) proves nothing on its own.")
    elif c_ok and not w_ok:
        print("  Client works, workers do not, on the SAME read.")
        if len(distinct) == 1:
            print("  Same egress address -> this is NOT an AWS rate limit on "
                  "the address.\n  Look at the worker environment or the dask "
                  "execution path.")
        else:
            print("  Different egress addresses -> the workers' address is the "
                  "problem.")
    elif w_ok and not c_ok:
        print("  Workers work, the client does not -- the opposite of what we "
              "assumed.")
    else:
        print("  Neither side can read. That is consistent with an AWS-side "
              "limit on a\n  shared egress address, or an outage.")
    if args.raw:
        print("\n  If icechunk succeeds where raw urllib 503s on the SAME host,")
        print("  the raw probe is simply missing the SDK's internal retry --")
        print("  and our '503 on a single request' evidence was measuring that,")
        print("  not the true availability of the path.")
    client.close()


if __name__ == "__main__":
    main()
