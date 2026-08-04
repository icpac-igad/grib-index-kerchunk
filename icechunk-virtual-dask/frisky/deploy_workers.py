"""Stand up standalone Frisky workers on the six EWC VMs, via the Dask cluster.

Why standalone rather than `frisky.hijack(dask_client)`:

  * hijack runs a Frisky worker INSIDE each Dask worker process, so it inherits
    that process's environment -- including the Ceph `AWS_*` that makes the
    icechunk virtual-chunk client resolve a hostname that does not exist
    (README.md §2).  Clearing it would strip the Dask workers' credentials for
    every other job on the cluster, and README §2 rule 1 says putting it back
    mid-read does not work because the config is cached process-wide.
  * hijack also inherits `worker.state.nthreads` (hijack.py:605), which is 4.
    That is fine now -- see --nthreads below -- but it is not a choice.
  * hijack replaces the Dask dashboard at `/`.

A separate process per VM has its own environment, leaves the Dask workers
completely alone, and is torn down with `--stop`.

The Dask cluster is used only as a remote-exec channel: there is no SSH key on
the gateway, and `client.run()` reaches all six VMs.

On --nthreads 4: `NEXT_SESSION.md` §4.2 says 1, because Dask could not see this
workload's memory and over-subscribed until the nanny killed workers.  That was
a property of the EAGER task shape.  One task here holds one GRIB message
(~4 MB decoded) and returns ~0.1 MB, so 4 threads is ~20 MB of concurrent
working set against a 10 GB limit.  The reason for 1 does not apply, and 4
gives 24 concurrent readers across the six VMs.

Usage
-----
    P=.venv/bin/python
    $P deploy_workers.py --install     # build the venv on all six (~3 min)
    $P deploy_workers.py --start       # launch the workers
    $P deploy_workers.py --status
    $P deploy_workers.py --stop        # kill them; nothing else is touched
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REMOTE_DIR = "/tmp/frisky-ea"
REMOTE_PY = f"{REMOTE_DIR}/.venv/bin/python"
REMOTE_FRISKY = f"{REMOTE_DIR}/.venv/bin/frisky"
DASK_SCHEDULER = "tcp://127.0.0.1:8786"

# Quoted individually when written into the install script -- an unquoted
# `frisky>=0.3.0` is a shell redirect, not a version specifier.
PACKAGES = ["frisky>=0.3.0", "icechunk==2.1.1", "zarr==3.2.1",
            "xarray==2026.7.0", "gribberish==1.6.0", "numpy==2.5.1"]


# ── remote functions.  Each returns fast; nothing blocks the Dask worker's
#    event loop for more than a moment, so heartbeats keep flowing. ──────────

def _kick_off_install(dirname, base_python, packages):
    """Start the venv build detached and return immediately."""
    import os, shlex, subprocess, socket
    os.makedirs(dirname, exist_ok=True)
    specs = " ".join(shlex.quote(s) for s in packages)
    script = f"""#!/bin/bash
set -e
rm -f {dirname}/INSTALL_OK {dirname}/INSTALL_FAIL
{base_python} -m venv {dirname}/.venv
{dirname}/.venv/bin/pip install -q --upgrade pip
{dirname}/.venv/bin/pip install -q {specs}
touch {dirname}/INSTALL_OK
"""
    path = f"{dirname}/install.sh"
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)
    log = open(f"{dirname}/install.log", "wb")
    subprocess.Popen(["/bin/bash", "-c",
                      f"{path} || touch {dirname}/INSTALL_FAIL"],
                     stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    return {"host": socket.gethostname(), "started": True}


def _check_install(dirname):
    import os, socket
    ok = os.path.exists(f"{dirname}/INSTALL_OK")
    fail = os.path.exists(f"{dirname}/INSTALL_FAIL")
    tail = ""
    if fail:
        try:
            with open(f"{dirname}/install.log") as f:
                tail = f.read()[-400:]
        except Exception:
            pass
    return {"host": socket.gethostname(),
            "state": "ok" if ok else ("FAILED" if fail else "building"),
            "tail": tail}


def _push_module(dirname, name, source):
    """Frisky pickles task functions BY REFERENCE, so the module defining them
    must be importable on the worker.  Ship it rather than rely on a shared
    filesystem."""
    import os, socket
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, name), "w") as f:
        f.write(source)
    return {"host": socket.gethostname(), "wrote": name, "bytes": len(source)}


def _start_worker(dirname, frisky_bin, sched, nthreads, memory_limit):
    """Launch `frisky worker` detached, with AWS_* scrubbed from ITS env only.

    The Dask worker process keeps its own AWS_* untouched -- this is a child
    with a modified copy, so no other job on this VM is affected.
    """
    import os, subprocess, socket
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env["PYTHONPATH"] = dirname
    env["FRISKY_EA_HOME"] = dirname
    log = open(f"{dirname}/worker.log", "ab")
    p = subprocess.Popen(
        [frisky_bin, "worker", sched, "--nthreads", str(nthreads),
         "--memory-limit", memory_limit],
        env=env, cwd=dirname, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    with open(f"{dirname}/worker.pid", "w") as f:
        f.write(str(p.pid))
    return {"host": socket.gethostname(), "pid": p.pid}


def _start_scheduler(dirname, frisky_bin, bind, dashboard, tracing_capacity):
    """Run the Frisky scheduler on a worker VM, not the gateway.

    The gateway is the JupyterHub session and is capped at 8 GiB
    (BLOCKERS.md). A scheduler left up across several 82,710-task runs grew
    past that and was OOM-killed mid-run -- the client saw only
    `RuntimeError: Failed to send: channel closed`, and the store was left
    with nothing but its initial commit. A worker VM has 15 GB and no cgroup
    cap, and FRISKY_TRACING_CAPACITY bounds the span buffer that does most of
    the growing.
    """
    import os, subprocess, socket
    subprocess.run(["pkill", "-f", f"{dirname}/.venv/bin/frisky scheduler"],
                   capture_output=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env["FRISKY_TRACING_CAPACITY"] = str(tracing_capacity)
    log = open(f"{dirname}/scheduler.log", "ab")
    p = subprocess.Popen(
        [frisky_bin, "scheduler", "--address", bind,
         "--dashboard-address", dashboard],
        env=env, cwd=dirname, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    with open(f"{dirname}/scheduler.pid", "w") as f:
        f.write(str(p.pid))
    return {"host": socket.gethostname(), "pid": p.pid, "bind": bind}


def _stop_scheduler(dirname):
    import socket, subprocess
    r = subprocess.run(["pkill", "-f",
                        f"{dirname}/.venv/bin/frisky scheduler"],
                       capture_output=True)
    return {"host": socket.gethostname(), "pkill_rc": r.returncode}


def _kick_off_probe(dirname, python, messages, levels):
    """Run bandwidth_probe.py on this VM, detached.

    All six fire at once, so the summed result answers the only question that
    matters for sizing: does per-VM throughput ADD, or is there a shared EWC
    egress ceiling? Six VMs measured one at a time cannot tell the difference.
    """
    import os, subprocess, socket
    os.makedirs(dirname, exist_ok=True)
    lv = " ".join(str(x) for x in levels)
    cmd = (f"cd {dirname} && rm -f PROBE_OK bw.log && "
           f"{python} bandwidth_probe.py --messages {messages} "
           f"--levels {lv} > bw.log 2>&1; touch PROBE_OK")
    subprocess.Popen(["/bin/bash", "-c", cmd], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"host": socket.gethostname(), "started": True}


def _check_probe(dirname):
    import os, socket
    done = os.path.exists(f"{dirname}/PROBE_OK")
    body = ""
    try:
        with open(f"{dirname}/bw.log") as f:
            body = f.read()
    except Exception:
        pass
    return {"host": socket.gethostname(), "done": done, "log": body}


def _worker_status(dirname):
    import os, socket
    pid, alive, tail = None, False, ""
    try:
        with open(f"{dirname}/worker.pid") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        pass
    try:
        with open(f"{dirname}/worker.log") as f:
            tail = f.read()[-300:]
    except Exception:
        pass
    have_frisky = os.path.exists(f"{dirname}/.venv/bin/frisky")
    return {"host": socket.gethostname(), "pid": pid, "alive": alive,
            "venv": have_frisky, "log": tail.strip()[-200:]}


def _stop_worker(dirname):
    """Kill EVERY frisky worker on this VM, not just the pid we last recorded.

    worker.pid holds only the most recent launch, so a --start that did not
    stop first leaves orphans the pid file can no longer name.  That is not
    cosmetic: an orphan runs whatever version of the task module it was
    started with, and Frisky pickles task functions by reference -- so a stale
    worker fails every task calling a function added since.  Match on the
    command line instead.
    """
    import socket, subprocess
    r = subprocess.run(["pkill", "-f", f"{dirname}/.venv/bin/frisky worker"],
                       capture_output=True, text=True)
    left = subprocess.run(["pgrep", "-fc",
                           f"{dirname}/.venv/bin/frisky worker"],
                          capture_output=True, text=True).stdout.strip()
    return {"host": socket.gethostname(), "pkill_rc": r.returncode,
            "still_running": left or "0"}


# ── client side ────────────────────────────────────────────────────────────

def show(label, res):
    print(f"\n{label}")
    for addr, r in sorted(res.items(), key=lambda kv: str(kv[1].get("host"))):
        host = r.pop("host", addr)
        print(f"  {host:16s} {r}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--install", action="store_true")
    p.add_argument("--scheduler-on", metavar="IP",
                   help="worker IP to host the Frisky scheduler, e.g. "
                        "192.168.1.74.  Keeps it off the 8 GiB gateway")
    p.add_argument("--start", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--bandwidth", action="store_true",
                   help="run bandwidth_probe.py on ALL VMs simultaneously and "
                        "sum the peaks -- tests whether egress scales")
    p.add_argument("--probe-messages", type=int, default=200)
    p.add_argument("--probe-levels", type=int, nargs="+", default=[32])
    p.add_argument("--tracing-capacity", type=int, default=20000,
                   help="scheduler span buffer; the default 200k is what "
                        "grew unbounded across runs")
    p.add_argument("--scheduler", default="192.168.1.129:8796",
                   help="the FRISKY scheduler the workers should dial")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--memory-limit", default="10GB")
    p.add_argument("--dask", default=DASK_SCHEDULER)
    p.add_argument("--timeout", type=int, default=900,
                   help="seconds to wait for the venv build")
    args = p.parse_args()

    if not any([args.install, args.start, args.status, args.stop,
                args.scheduler_on, args.bandwidth]):
        p.error("pick one of --install / --scheduler-on / --start / "
                "--status / --stop / --bandwidth")

    from distributed import Client

    with Client(args.dask, timeout=30) as c:
        if args.install:
            base = "/opt/mamba/envs/dask/bin/python3.12"
            show("kicking off venv builds",
                 c.run(_kick_off_install, REMOTE_DIR, base, PACKAGES))
            t0 = time.time()
            while time.time() - t0 < args.timeout:
                time.sleep(20)
                res = c.run(_check_install, REMOTE_DIR)
                states = [r["state"] for r in res.values()]
                done = sum(s == "ok" for s in states)
                print(f"  {time.time() - t0:5.0f}s  ok={done}/{len(states)}  "
                      f"{sorted(set(states))}")
                if all(s == "ok" for s in states):
                    break
                if any(s == "FAILED" for s in states):
                    show("FAILED", res)
                    sys.exit(1)
            else:
                show("timed out", c.run(_check_install, REMOTE_DIR))
                sys.exit(1)
            print("\nvenv ready on all workers")

        if args.bandwidth:
            here = os.path.dirname(os.path.abspath(__file__))
            for mod in ("frisky_daily_dag.py", "bandwidth_probe.py"):
                with open(os.path.join(here, mod)) as f:
                    c.run(_push_module, REMOTE_DIR, mod, f.read())
            show("probing all six at once",
                 c.run(_kick_off_probe, REMOTE_DIR, REMOTE_PY,
                       args.probe_messages, args.probe_levels))
            t0 = time.time()
            while time.time() - t0 < 900:
                time.sleep(20)
                res = c.run(_check_probe, REMOTE_DIR)
                n = sum(r["done"] for r in res.values())
                print(f"  {time.time() - t0:5.0f}s  done={n}/{len(res)}")
                if n == len(res):
                    break
            total = 0.0
            for addr, r in sorted(res.items(), key=lambda kv: kv[1]["host"]):
                print(f"\n--- {r['host']}")
                for line in r["log"].splitlines():
                    if line.strip() and not line.startswith(("warming", "-")):
                        print("   ", line)
                for line in r["log"].splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].isdigit():
                        total = max(total, 0) + 0  # placeholder, summed below
            # Sum each VM's PEAK MB/s
            peaks = []
            for r in res.values():
                best = 0.0
                for line in r["log"].splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].isdigit():
                        try:
                            best = max(best, float(parts[3]))
                        except ValueError:
                            pass
                peaks.append(best)
            agg = sum(peaks)
            print(f"\n{'=' * 62}")
            print(f"per-VM peaks : {['%.1f' % x for x in sorted(peaks)]}")
            print(f"AGGREGATE    : {agg:.1f} MB/s  ({agg * 8 / 1000:.2f} Gbps)"
                  f"  across {len(peaks)} VMs")
            print(f"mean per VM  : {agg / max(len(peaks), 1):.1f} MB/s")
            if agg:
                print(f"\nVMs for 1 GB/s at this per-VM rate: "
                      f"{1000 / (agg / len(peaks)):.0f}")
            print("Linear scaling holds only if the mean per-VM rate here "
                  "matches\nthe 31 MB/s measured on ONE VM alone. If it is "
                  "lower, the EWC\negress is shared and no VM count reaches "
                  "1 GB/s.")
            return

        if args.scheduler_on:
            host = args.scheduler_on
            target = [a for a in c.scheduler_info()["workers"]
                      if f"//{host}:" in a]
            if not target:
                p.error(f"no Dask worker at {host}; known: "
                        f"{list(c.scheduler_info()['workers'])}")
            bind = f"{host}:8796"
            show(f"starting frisky scheduler on {host}",
                 c.run(_start_scheduler, REMOTE_DIR, REMOTE_FRISKY, bind,
                       f"{host}:8791", args.tracing_capacity,
                       workers=target[:1]))
            args.scheduler = bind
            print(f"\nscheduler  {bind}   dashboard http://{host}:8791")
            print(f"tracing capacity {args.tracing_capacity} "
                  f"(default 200k is what grew unbounded)")

        if args.install or args.start:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "frisky_daily_dag.py")) as f:
                src = f.read()
            show("shipping frisky_daily_dag.py",
                 c.run(_push_module, REMOTE_DIR, "frisky_daily_dag.py", src))

        if args.start:
            # Always clear first.  Starting on top of a running worker leaves
            # an orphan holding a stale copy of the task module.
            show("clearing any running workers",
                 c.run(_stop_worker, REMOTE_DIR))
            time.sleep(3)
            show("starting frisky workers",
                 c.run(_start_worker, REMOTE_DIR, REMOTE_FRISKY,
                       args.scheduler, args.nthreads, args.memory_limit))
            time.sleep(5)
            show("status", c.run(_worker_status, REMOTE_DIR))
            print(f"\nfrisky workers dialling {args.scheduler} "
                  f"at {args.nthreads} threads each "
                  f"({6 * args.nthreads} concurrent readers)")

        if args.status:
            show("status", c.run(_worker_status, REMOTE_DIR))

        if args.stop:
            show("stopping", c.run(_stop_worker, REMOTE_DIR))
            print("\nDask workers untouched -- their AWS_* was never modified")


if __name__ == "__main__":
    main()
