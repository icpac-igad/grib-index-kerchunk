# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1",
#   "pandas", "pyarrow", "gribberish>=1.4",
#   "google-cloud-storage", "requests",
# ]
# ///
"""Build many dates into one Icechunk group concurrently, via fork/merge.

The serial driver (`backfill_all_eras.py`) took 20.64 h for 1,256 dates because
one date is one commit and commits serialize. This one fans dates out across
worker processes and commits a whole batch at once.

    coordinator  create schema (time axis preallocated, sorted)
                 |
                 +-- ensure every array this batch needs exists   <- ONE commit
                 |
                 +-- session.fork() -> worker: parse 51 pars,
                 |                             set_virtual_refs at its own ti
                 |   ... N dates concurrently, each returning a ForkSession ...
                 |
                 `-- session.merge(*forks); session.commit()       <- ONE commit

Why this is legal: a preallocated group gives every date a known, distinct time
index, so chunk indices `[ti, number, step, level, 0, 0]` never collide between
workers. That is the whole constraint icechunk puts on the caller. It is also why
`--create-schema` is mandatory here -- see the refusal in `check_group`.

Two things workers must never touch, because they are shared state that chunk
disjointness does not cover:

  * coordinate arrays  -- written once by --create-schema
  * array creation     -- done by the coordinator before each batch fans out

Hence the "ensure arrays" step: the coordinator reads only the `key` column of
ONE par per date (~0.1 s) to learn that date's variable set, and creates anything
missing. Schema drift across an era is real (49r1 gains and loses vars between
2024 and 2026), so this cannot be assumed constant.

Usage
-----
    # 1. freeze the axis (once per era/run)
    uv run build_ecmwf_icechunk.py --era 49r1 --run 00 --create-schema \
        --dates-file dates-49r1.txt --store gs://bucket/icechunk/ecmwf-ens

    # 2. fan out
    uv run backfill_parallel.py --era 49r1 --run 00 \
        --store gs://bucket/icechunk/ecmwf-ens \
        --par-source gcs --executor dask --scheduler 192.168.1.74:8796 --batch 20

    # local pars (testing, or a pre-staged corpus)
    uv run backfill_parallel.py --era 0p4 --par-source local \
        --pars-root ./pars --store ./store --workers 4 --batch 4

Par sources: `gcs` (authoritative), `local`, `hf`. **HuggingFace still holds the
defective pars for 20260319 / 20260322-27 / 20260329-31** -- rebuilding those
from `hf` reintroduces the collapsed-pl-level bug, which is why `gcs` is the
default. See published-ecmwf-store-defects.

Resume: dates whose refs are already in the manifest are skipped, so a killed run
restarts safely. Nothing is ever resized or reordered, so a partial store is a
valid store with holes.
"""
from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import sys
import time as _time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from pathlib import Path

# icechunk holds a tokio runtime; fork()ing a process that owns one deadlocks the
# child. Linux defaults to fork, so the pool MUST be spawn -- symptom otherwise is
# a hang with no output and no error, which is exactly what it did the first time.
MP = multiprocessing.get_context("spawn")

import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ecmwf_icechunk as B  # noqa: E402

HF = "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par-v2"


# ---------------------------------------------------------------------------
# par staging
# ---------------------------------------------------------------------------
def stage_pars(date: str, loc: dict, dest: str | None, limit: int | None = None):
    """Fetch a date's pars and return the local directory holding them.

    RUNS IN THE WORKER. The pars are the I/O-heavy input -- ~21 MB of 51 small,
    latency-bound objects -- and the measured cost of a date is ~45 s wall against
    ~5.5 s of CPU. Fetching them where the compute is means download concurrency
    scales with the worker count and no par ever transits the coordinator.

    `loc` is a plain dict (picklable across a process pool or a Dask cluster);
    `limit` fetches only the first N members, which the coordinator uses to grab
    a single par for its variable peek.
    """
    source = loc["source"]
    if source == "local":
        d = Path(loc["pars_root"]) / date
        n = len(list(d.glob("*.parquet")))
        if n != B.N_MEMBERS:
            raise RuntimeError(f"{d}: found {n} pars, expected {B.N_MEMBERS}")
        return d
    d = Path(dest) / date
    d.mkdir(parents=True, exist_ok=True)
    if limit is None and len(list(d.glob("*.parquet"))) == B.N_MEMBERS:
        return d

    if source == "gcs":
        from google.cloud import storage
        bucket_name, _, prefix = loc["gcs_path"][5:].partition("/")   # gs://b/p
        blobs = [b for b in storage.Client().list_blobs(
                     bucket_name, prefix=prefix.rstrip("/") + "/")
                 if b.name.endswith(".parquet")]
        if limit is None and len(blobs) != B.N_MEMBERS:
            raise RuntimeError(f"{loc['gcs_path']}: {len(blobs)} pars, "
                               f"expected {B.N_MEMBERS}")
        blobs = sorted(blobs, key=lambda b: b.name)[:limit]

        def get(b):
            b.download_to_filename(d / b.name.split("/")[-1])
    elif source == "hf":
        import requests
        tree = requests.get(
            f"https://huggingface.co/api/datasets/E4DRR/gik-ecmwf-par-v2/"
            f"tree/main/{loc['hf_path']}", timeout=60).json()
        blobs = sorted(x["path"].split("/")[-1] for x in tree
                       if x["path"].endswith(".parquet"))[:limit]

        def get(n):
            for attempt in range(3):
                try:
                    r = requests.get(f"{HF}/resolve/main/{loc['hf_path']}/{n}",
                                     timeout=120)
                    r.raise_for_status()
                    (d / n).write_bytes(r.content)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    _time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"unknown par source {source}")

    # latency-bound, so overlap them; this is inside one worker's own task
    with ThreadPoolExecutor(min(8, max(1, len(blobs)))) as ex:
        list(ex.map(get, blobs))
    return d


def peek_vars(pars_dir: Path) -> tuple[list[str], list[str]]:
    """(sfc_vars, pl_vars) from ONE par's key column -- cheap, no full parse.

    Reading a single member is enough: every member of a date carries the same
    variable set. This is what lets the coordinator create arrays before the
    workers (which must not create anything) start.
    """
    one = sorted(pars_dir.glob("*.parquet"))[0]
    keys = pd.read_parquet(one, columns=["key"])["key"]
    keys = keys[keys.str.startswith("step_")]
    parts = keys.str.split("/")
    df = pd.DataFrame({"var": parts.map(lambda p: p[1]),
                       "levtype": parts.map(lambda p: p[2])}).drop_duplicates()
    return B.split_levtypes(df)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def init_worker() -> None:
    """Pin each worker to one thread.

    `pd.read_parquet` uses pyarrow, which is multi-threaded by default, so a
    single worker already spreads across cores. Left alone, N workers then fight
    each other: measured on 4 dates, summed worker CPU rose from 20 s at 1 worker
    to 45 s at 4 while wall time only fell from 21 s to 14 s. One thread per
    process turns that contention into clean process-level scaling.
    """
    try:
        import pyarrow
        pyarrow.set_cpu_count(1)
        pyarrow.set_io_thread_count(1)
    except Exception:
        pass



def write_date(fork, grp: str, era_name: str, date: str, loc: dict, staging: str,
               ti: int, steps: list[int]):
    """Parse a date's pars and set its virtual refs into `fork`. Runs remote.

    Returns `(fork, ti, n_refs)` -- the fork must travel BACK, because the
    changeset lives in it and the coordinator merges it. Returning only counts
    would silently drop every ref this worker wrote.

    Creates nothing and resizes nothing: every array already exists at full time
    length, which is what makes this safe to run alongside other dates.
    """
    t0 = _time.time()
    era = B.ERAS[era_name]
    lev_idx = {float(v): i for i, v in enumerate(era["levels"])}
    step_idx = {h: i for i, h in enumerate(steps)}
    pars_dir = stage_pars(date, loc, staging)     # fetch where the compute is
    t_stage = _time.time() - t0
    t0 = _time.time()
    refs = B.load_date_refs(Path(pars_dir))
    t_parse = _time.time() - t0
    dsteps = sorted(refs.step_h.unique())
    if dsteps != list(steps):
        raise RuntimeError(f"step mismatch: date has {len(dsteps)}, "
                           f"group has {len(steps)}")
    bad = set(refs.loc[refs.levtype == "pl", "level"].dropna()) - set(lev_idx)
    if bad:
        raise RuntimeError(f"levels {bad} outside the {era_name} superset")
    sfc_vars, pl_vars = B.split_levtypes(refs)
    t1 = _time.time()
    n = 0
    for var, is_pl in [(v, False) for v in sfc_vars] + [(v, True) for v in pl_vars]:
        zname = B.channel_name(var, is_pl, pl_vars)
        specs = B.ref_specs(refs, var, is_pl, ti, step_idx, lev_idx)
        rejected = fork.store.set_virtual_refs(f"{grp}/{zname}", specs)
        if rejected:
            raise RuntimeError(f"{grp}/{zname}: rejected refs {rejected}")
        n += len(specs)
    if loc["source"] != "local" and not loc.get("keep_pars"):
        for f in Path(pars_dir).glob("*.parquet"):
            f.unlink()
    return fork, ti, n, t_stage, t_parse, _time.time() - t1


# ---------------------------------------------------------------------------
# coordinator
# ---------------------------------------------------------------------------
def make_executor(args):
    """A local process pool, or the Dask cluster the realize stage already uses.

    `write_date` is executor-agnostic -- a ForkSession pickles either way -- so the
    choice is purely about where the network is. Local is right when the pars are
    already on disk. Dask is right for a real backfill: the work is I/O-bound
    (~45 s wall per date against ~5.5 s CPU), so what scales is putting the fetch
    next to more bandwidth, ideally in the bucket's own region. Local process
    parallelism does not help with that -- measured on this 8-core host, 4
    concurrent pandas parsers only reached ~1.8x because they contend for memory
    bandwidth, not CPU.
    """
    if args.executor == "dask":
        from distributed import Client
        client = Client(args.scheduler)
        print(f"  dask cluster {args.scheduler}: "
              f"{len(client.scheduler_info()['workers'])} worker(s)")
        return client
    return ProcessPoolExecutor(args.workers, mp_context=MP,
                               initializer=init_worker)


def shutdown_executor(ex) -> None:
    (ex.close if hasattr(ex, "scheduler_info") else ex.shutdown)()


def check_group(session, grp: str, era_name: str):
    """The group must exist, be preallocated, and have the right axes."""
    try:
        g = zarr.open_group(store=session.store, path=grp, mode="r+",
                            zarr_format=3)
    except Exception:
        raise SystemExit(
            f"group {grp} does not exist. Create it first, which also freezes "
            f"the time axis:\n"
            f"    build_ecmwf_icechunk.py --era {era_name} --create-schema "
            f"--dates-file <dates> --store <store>")
    if B.group_mode(g) != B.PREALLOC:
        raise SystemExit(
            f"REFUSING: group {grp} is in '{B.group_mode(g)}' mode.\n"
            f"  Parallel writes need a preallocated time axis -- in append mode "
            f"every date derives its\n  index from the current length and "
            f"resizes ~40 shared arrays, so concurrent dates would\n  race on "
            f"both. Build it with --create-schema, or use the serial driver.")
    B.check_coords(g, era_name)          # the longitude guard
    return g


def ensure_arrays(repo, grp: str, era_name: str, vars_by_date: dict, steps,
                  n_time: int):
    """Create any array this batch needs that does not exist yet. ONE commit.

    Workers cannot do this: creating an array is a metadata write on the group,
    which no amount of chunk disjointness makes concurrent-safe.
    """
    era = B.ERAS[era_name]
    ny, nx, n_levels = era["ny"], era["nx"], len(era["levels"])
    wanted = {}
    for sfc, pl in vars_by_date.values():
        for var, is_pl in [(v, False) for v in sfc] + [(v, True) for v in pl]:
            wanted[B.channel_name(var, is_pl, pl)] = (var, is_pl)
    session = repo.writable_session("main")
    g = zarr.open_group(store=session.store, path=grp, mode="r+", zarr_format=3)
    made = []
    for zname, (var, is_pl) in sorted(wanted.items()):
        if zname in g:
            continue
        shape, chunks, dims = B.array_geometry(is_pl, n_time, len(steps),
                                               n_levels, ny, nx)
        zarr.create_array(session.store, name=f"{grp}/{zname}", shape=shape,
                          chunks=chunks, dtype="float32",
                          fill_value=float("nan"),
                          serializer=B.GribberishCodec(var=zname),
                          compressors=None, filters=None, dimension_names=dims,
                          attributes={"grib_shortName": var}, overwrite=False)
        made.append(zname)
    if made:
        session.commit(f"{grp}: create {len(made)} array(s) {', '.join(made)}")
        print(f"  created {len(made)} array(s): {', '.join(made)}")
    return made


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", required=True, choices=list(B.ERAS))
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--store", required=True)
    ap.add_argument("--sa-key", default=None)
    ap.add_argument("--par-source", default="gcs", choices=["gcs", "local", "hf"],
                    help="gcs is authoritative; hf still holds defective March "
                         "2026 pars")
    ap.add_argument("--pars-root", default=None,
                    help="directory of <date>/ subdirs (--par-source local)")
    ap.add_argument("--catalog", default=None,
                    help="catalog.parquet path; default fetches from HF")
    ap.add_argument("--staging", default="./par-staging",
                    help="scratch for downloaded pars; deleted per batch")
    ap.add_argument("--executor", default="local", choices=["local", "dask"],
                    help="local process pool, or the Dask cluster the realize "
                         "stage uses. Dask is what scales a real backfill: the "
                         "work is I/O-bound, so more workers = more bandwidth")
    ap.add_argument("--scheduler", default="192.168.1.74:8796",
                    help="dask scheduler address (--executor dask)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="process count (--executor local)")
    ap.add_argument("--batch", type=int, default=20,
                    help="dates per commit. Bigger = fewer snapshots, but a "
                         "failed merge loses the batch")
    ap.add_argument("--start", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-pars", action="store_true")
    args = ap.parse_args()
    t_start = _time.time()

    grp = f"{args.era}/{args.run}z"
    storage = B.resolve_storage(args.store, args.sa_key)
    repo = B.open_or_create_repo(storage)
    session = repo.writable_session("main")
    g = check_group(session, grp, args.era)
    tvals = g["time"][:]
    steps = [int(s) for s in g["step"][:]]
    n_time = int(tvals.size)
    ti_of = {int(t): i for i, t in enumerate(tvals)}
    print(f"{grp}: preallocated {n_time} dates, {len(steps)} steps, "
          f"{len(B.ERAS[args.era]['levels'])} levels")

    # which dates does the axis want, and which are already written?
    dates = []
    for tv, ti in sorted(ti_of.items(), key=lambda kv: kv[1]):
        d = (B.EPOCH + pd.Timedelta(hours=int(tv))).strftime("%Y%m%d")
        dates.append((d, ti))
    if args.start:
        dates = [(d, i) for d, i in dates if d >= args.start]
    if args.end:
        dates = [(d, i) for d, i in dates if d <= args.end]
    todo = [(d, i) for d, i in dates if not B.date_written(session, grp, g, i)]
    skipped = len(dates) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"  {len(todo)} to build, {skipped} already written"
          f"{f', limited to {args.limit}' if args.limit else ''}")
    if not todo:
        print("nothing to do")
        return

    cat = None
    if args.par_source != "local":
        if args.catalog:
            cat = pd.read_parquet(args.catalog)
        else:
            import requests
            r = requests.get(f"{HF}/resolve/main/catalog.parquet", timeout=90)
            r.raise_for_status()
            cat = pd.read_parquet(io.BytesIO(r.content))
        cat = cat[(cat.era == args.era) & (cat.run == f"{args.run}z")]
        cat = cat.set_index(cat.date.astype(str))

    def loc_for(d):
        base = {"source": args.par_source, "pars_root": args.pars_root,
                "keep_pars": args.keep_pars}
        if cat is not None:
            r = cat.loc[d]
            base.update(gcs_path=str(r.gcs_path), hf_path=str(r.hf_path))
        return base

    total_refs, done, failed = 0, 0, []
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"  {len(batches)} batch(es) of up to {args.batch}, "
          f"{args.workers} {args.executor} worker(s), pars from {args.par_source}\n")

    ex = make_executor(args)
    for bi, batch in enumerate(batches, 1):
        tb = _time.time()
        # --- coordinator: learn this batch's variable set --------------------
        # One par per date (1/51 of the data) is enough, and array creation is a
        # group-metadata write that workers must not do concurrently.
        vars_by_date, staged = {}, {}
        for d, ti in batch:
            try:
                one = stage_pars(d, loc_for(d), args.staging, limit=1)
                vars_by_date[d] = peek_vars(one)
                staged[(d, ti)] = one
            except Exception as e:
                failed.append((d, f"peek: {str(e).splitlines()[0][:80]}"))
        if not staged:
            print(f"batch {bi}/{len(batches)}: no date could be inspected")
            continue
        t_peek = _time.time() - tb
        ensure_arrays(repo, grp, args.era, vars_by_date, steps, n_time)

        # --- fan out: each worker fetches its own pars, parses, sets refs ----
        # fresh session: fork() refuses one with uncommitted changes, and
        # ensure_arrays just committed.
        session = repo.writable_session("main")
        t_fan = _time.time()
        forks, nrefs, wrote, wt = [], 0, [], []
        futs = {ex.submit(write_date, session.fork(), grp, args.era, d,
                          loc_for(d), args.staging, ti, steps): (d, ti)
                for (d, ti) in staged}
        for f in as_completed(futs):
            d, ti = futs[f]
            try:
                fork, _ti, n, tst, tp, ts = f.result()  # fork carries the changeset
                forks.append(fork); nrefs += n; wrote.append(d)
                wt.append((tst, tp, ts))
            except Exception as e:
                failed.append((d, f"write: {str(e).splitlines()[0][:80]}"))
        t_write = _time.time() - t_fan

        # --- one commit for the whole batch ----------------------------------
        if not forks:
            session.discard_changes()
            print(f"batch {bi}/{len(batches)}: every date failed, nothing committed")
            continue
        t_m = _time.time()
        session.merge(*forks)
        snap = session.commit(
            f"{grp}: {len(wrote)} dates {min(wrote)}..{max(wrote)}, "
            f"{nrefs} refs, 51 members")
        agg = [sum(x[i] for x in wt) for i in range(3)]
        print(f"batch {bi}/{len(batches)}: {len(wrote)} dates | peek {t_peek:.0f}s "
              f"| fanout {t_write:.0f}s (worker-seconds: fetch {agg[0]:.0f} "
              f"parse {agg[1]:.0f} setrefs {agg[2]:.0f}) "
              f"| merge+commit {_time.time()-t_m:.0f}s | {nrefs:,} refs "
              f"-> {str(snap)[:12]}")
        done += len(wrote); total_refs += nrefs

    shutdown_executor(ex)
    el = _time.time() - t_start
    print(f"\n{done} dates, {total_refs:,} refs in {el/3600:.2f} h "
          f"({el/max(1,done):.1f} s/date)")
    if failed:
        print(f"{len(failed)} failed:")
        for d, why in failed[:20]:
            print(f"  {d}: {why}")


if __name__ == "__main__":
    main()
