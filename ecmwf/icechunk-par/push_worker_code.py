"""Ship the builder/driver modules to the frisky workers, safely.

frisky pickles module-level functions BY REFERENCE, so a worker must be able to
import `backfill_parallel` before it can unpickle write_date. The modules are
copied into one directory (grids.py lands NEXT TO the builder, which is what
_locate_grids searches first) and that directory is put on sys.path.

The locking is not incidental. This is fanned out as many tasks to reach all six
VMs, and each worker process runs 4 threads, so several tasks execute inside one
interpreter at once. An earlier version popped sys.modules and re-imported with
no lock; concurrent threads then tore each other's imports down mid-flight and
every worker ended up logging `KeyError: 'backfill_parallel'` from
importlib._load_unlocked. flock serialises across BOTH processes and threads
(separate fds contend on the same file), and modules are only evicted when the
bytes on disk actually changed.
"""
import frisky
from pathlib import Path

SRC = Path("/var/lib/private/nishadhka/grib-index-kerchunk/ecmwf")
FILES = {"grids.py": (SRC / "grids.py").read_bytes(),
         "build_ecmwf_icechunk.py": (SRC / "icechunk-par" / "build_ecmwf_icechunk.py").read_bytes(),
         "backfill_parallel.py": (SRC / "icechunk-par" / "backfill_parallel.py").read_bytes()}
DEST = "/tmp/frisky-ea/gik"


def place(files, dest):
    import os, sys, socket, fcntl, importlib
    os.makedirs(dest, exist_ok=True)
    with open("/tmp/.gik_code.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        changed = False
        for name, data in files.items():
            p = os.path.join(dest, name)
            old = open(p, "rb").read() if os.path.exists(p) else None
            if old != data:
                with open(p, "wb") as f:
                    f.write(data)
                changed = True
        if dest not in sys.path:
            sys.path.insert(0, dest)
        if changed:                      # only evict when the bytes moved
            for m in ("backfill_parallel", "build_ecmwf_icechunk", "grids"):
                sys.modules.pop(m, None)
            importlib.invalidate_caches()
        import backfill_parallel as BP
        return socket.gethostname(), BP.__file__, changed


if __name__ == "__main__":
    c = frisky.Client("192.168.1.74:8796")
    res = {}
    for f in [c.submit(place, FILES, DEST, key=f"code-{i}") for i in range(36)]:
        try:
            h, path, ch = f.result()
            res.setdefault(h, (path, ch))
        except Exception as e:
            print("  task failed:", str(e).splitlines()[0][:140])
    for h, (path, ch) in sorted(res.items()):
        print(f"  {h:20s} {'updated' if ch else 'unchanged'}  {path}")
    print(f"hosts: {len(res)}")
