"""Write a realized Icechunk store to EWC Ceph, one channel per commit.

Client-side only.  Never imported by a worker.

Why the write happens here and not on the workers
-------------------------------------------------
`ICECHUNK_DASK_GUIDANCE.md` §1: a writable Session accumulates a change-set,
and several processes cannot each hold one and stay coherent.  The supported
shapes are (a) one process writes, or (b) fork -> write remotely -> merge ->
commit.  We use (a): the workers hand back numpy arrays, and this process holds
the only Session.  `to_icechunk` with numpy-backed data is a local write, which
is the case `realize_smoke_test.py` was already judged "within spec" for.

Memory: one channel at a time.  51 members x 53 steps x 262 x 261 float32 is
739 MB, so the client peak is ~1.5 GB against the 8 GiB cgroup — the reason
the caller gathers per channel rather than building the whole 26.6 GB Dataset.

Credentials: the source read runs anonymous with `AWS_*` stripped (README §2),
and this write passes the Ceph key explicitly rather than through `AWS_*`.  The
two never contend, which is why the process can do both.
"""
from __future__ import annotations

import os

DST_BUCKET = "must-icechunk"
DST_REGION = "RegionOne"
DST_ENDPOINT_DEFAULT = "https://object-store.os-api.cci1.ecmwf.int"


#: Which object store the realized output goes to.
#:   unset / "ceph"  -> EWC Ceph, must-icechunk            (the default; unchanged)
#:   "gcs"           -> Google Cloud Storage, CRMA_SINK_BUCKET
#: Set by the caller's environment so no existing invocation changes behaviour.
#: GCS removes the EWC dependency entirely: no AK/SK, no ECMWF endpoint, no
#: AWS_CA_BUNDLE trust-store problem (boto3 validates against certifi, which
#: does not carry the ECMWF CA -- the failure presents as a credentials error
#: when it is a trust problem).
SINK_BACKEND = os.environ.get("CRMA_SINK", "ceph").lower()
GCS_BUCKET = os.environ.get("CRMA_SINK_BUCKET", "")


def load_env(path):
    """AK/SK from the gitignored .env next to the other scripts.

    On the GCS backend there is nothing to load -- authentication is the VM's
    service account or GOOGLE_APPLICATION_CREDENTIALS -- so this returns
    (None, None) and the caller needs no change.
    """
    if SINK_BACKEND == "gcs":
        return None, None
    creds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    missing = [k for k in ("AK", "SK") if k not in creds]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")
    return creds["AK"], creds["SK"]


def open_sink(prefix, ak, sk, endpoint=None, create=True):
    """Open, or create, the realized repo at must-icechunk/<prefix>."""
    import icechunk as ic

    if SINK_BACKEND == "gcs":
        if not GCS_BUCKET:
            raise SystemExit("CRMA_SINK=gcs needs CRMA_SINK_BUCKET set")
        # from_env=True picks up the VM service account or
        # GOOGLE_APPLICATION_CREDENTIALS. Keep the bucket in the same region as
        # the workers: a same-region GCS write is free, cross-region is charged
        # per GB (devops/crma-gik-spot-vm/README.md).
        storage = ic.gcs_storage(bucket=GCS_BUCKET, prefix=prefix, from_env=True)
        try:
            return ic.Repository.open(storage), False
        except Exception:
            if not create:
                raise
            return ic.Repository.create(storage), True

    storage = ic.s3_storage(
        bucket=DST_BUCKET, prefix=prefix, region=DST_REGION,
        endpoint_url=endpoint or os.environ.get("AWS_ENDPOINT_URL",
                                                DST_ENDPOINT_DEFAULT),
        access_key_id=ak, secret_access_key=sk,
        force_path_style=True, from_env=False,
    )
    try:
        return ic.Repository.open(storage), False
    except Exception:
        if not create:
            raise
        return ic.Repository.create(storage), True


def channel_dataset(name, arr, coords, date):
    """One channel as a Dataset with a leading length-1 `time`.

    Dims (time, step, number, latitude, longitude).  `time` is length 1 and is
    the append dimension, so later dates extend the store rather than rewrite
    it.
    """
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {name: (("time", "step", "number", "latitude", "longitude"),
                arr[np.newaxis, ...])},
        coords={
            "time": np.array([np.datetime64(date, "ns")]),
            "step": coords["step"],
            "number": coords["number"],
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
        },
    )


def encoding_for(name, n_number, n_lat, n_lon):
    """One chunk per (date, step): 51 x 262 x 261 float32 = 13.9 MB.

    Keeps the store chunks a divisor of anything a later dask read would use,
    which is the constraint ICECHUNK_DASK_GUIDANCE.md §1 calls out as the
    caller's responsibility.
    """
    return {name: {"chunks": (1, 1, n_number, n_lat, n_lon)}}


def write_channel(repo, name, arr, coords, date, first_channel, first_date):
    """Commit one channel.  Returns the snapshot id.

    first_channel + first_date -> mode="w" to lay down the schema.
    later channels, same date  -> mode="a", adding a new variable.
    later dates                -> append along `time`.
    """
    from icechunk.xarray import to_icechunk

    ds = channel_dataset(name, arr, coords, date)
    session = repo.writable_session("main")

    if first_date:
        mode = "w" if first_channel else "a"
        to_icechunk(ds, session, mode=mode,
                    encoding=encoding_for(name, ds.sizes["number"],
                                          ds.sizes["latitude"],
                                          ds.sizes["longitude"]))
    else:
        to_icechunk(ds, session, append_dim="time")

    snap = session.commit(f"{name} {date}")
    return snap


def create_schema(repo, channel_names, coords, dates, dtype="float32"):
    """Lay down empty arrays for every channel, metadata only, and commit.

    Written from the coordinator before any fork, because a ForkSession can
    only fill regions of arrays that already exist.

    `to_zarr(compute=False)` writes the metadata and none of the data, so this
    costs nothing even though the arrays total 26.6 GB.  It goes through xarray
    rather than `zarr.create_array` deliberately: the `realized-test` store in
    this same bucket was written with raw zarr and was never readable by
    xarray, because its arrays carried no `dimension_names`. (Deleted
    2026-08-09; the lesson is why this goes through xarray.)

    Returns the snapshot id.
    """
    import dask.array as dsa
    import numpy as np
    import xarray as xr

    if isinstance(dates, str):
        dates = [dates]
    shape = (len(dates), len(coords["step"]), len(coords["number"]),
             len(coords["latitude"]), len(coords["longitude"]))
    chunks = (1, 1, shape[2], shape[3], shape[4])
    dims = ("time", "step", "number", "latitude", "longitude")

    ds = xr.Dataset(
        {name: (dims, dsa.zeros(shape, chunks=chunks, dtype=dtype))
         for name in channel_names},
        coords={
            "time": np.array([np.datetime64(d, "ns") for d in dates]),
            "step": coords["step"],
            "number": coords["number"],
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
        },
    )
    session = repo.writable_session("main")
    ds.to_zarr(session.store, compute=False, zarr_format=3,
               consolidated=False, mode="w")
    return session.commit(
        f"schema: {len(channel_names)} channels x {len(dates)} dates "
        f"({dates[0]} .. {dates[-1]})")


def merge_and_commit(repo, forks, message):
    """Collect the workers' changesets into one commit.

    `Session.fork()` refuses to fork a session that already has uncommitted
    changes, so the coordinator session used here must be freshly opened after
    the schema commit.
    """
    session = repo.writable_session("main")
    session.merge(*forks)
    return session.commit(message)


def existing_channels(repo):
    """Channels already committed, so a killed run can resume.

    Every channel is its own commit, so whatever landed before a crash is
    durable and must not be redone -- each one costs 2,703 GRIB messages and
    ~2 GB of AWS egress.
    """
    import xarray as xr

    try:
        ds = xr.open_zarr(repo.readonly_session("main").store,
                          consolidated=False, zarr_format=3)
    except Exception:
        return set()          # nothing but the initial commit
    return set(ds.data_vars)


def date_exists(repo, date):
    """Is this date already on the `time` axis?  Makes a rerun idempotent."""
    import numpy as np
    import xarray as xr

    try:
        ds = xr.open_zarr(repo.readonly_session("main").store,
                          consolidated=False, zarr_format=3)
    except Exception:
        return False
    if "time" not in ds.coords:
        return False
    return np.datetime64(date, "ns") in set(ds.time.values.tolist() and
                                            ds.time.values)
