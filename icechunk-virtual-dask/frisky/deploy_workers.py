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
    import os, signal, socket
    killed = None
    try:
        with open(f"{dirname}/worker.pid") as f:
            pid = int(f.read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        killed = pid
    except Exception as e:
        killed = f"none ({type(e).__name__})"
    return {"host": socket.gethostname(), "killed": killed}


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
    p.add_argument("--start", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--scheduler", default="192.168.1.129:8796",
                   help="the FRISKY scheduler the workers should dial")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--memory-limit", default="10GB")
    p.add_argument("--dask", default=DASK_SCHEDULER)
    p.add_argument("--timeout", type=int, default=900,
                   help="seconds to wait for the venv build")
    args = p.parse_args()

    if not any([args.install, args.start, args.status, args.stop]):
        p.error("pick one of --install / --start / --status / --stop")

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

        if args.install or args.start:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "frisky_daily_dag.py")) as f:
                src = f.read()
            show("shipping frisky_daily_dag.py",
                 c.run(_push_module, REMOTE_DIR, "frisky_daily_dag.py", src))

        if args.start:
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
