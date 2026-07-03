# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas<2.2",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "cfgrib",
#     "eccodes",
#     "kerchunk==0.2.7",
#     "h5py",
#     "zarr>=2.18,<3",
#     "xarray==2024.10.0",
#     "numcodecs<0.13",
# ]
# ///
#!/usr/bin/env python3
"""
CFS template builder — orchestrator, modeled on
`ecmwf/lithops-cr-gik-ecmwf/build_ecmwf_50r1_template.py` and the per-era
Coiled preprocessing drivers `ecmwf/dev-test/ecmwf_{0p4,49r1,50r1}_coiled_preprocessing.py`.

WHAT THE ECMWF ROUTINE DOES (per era)
-------------------------------------
1. Coiled preprocessing: in parallel on a Coiled cluster, `scan_grib` EVERY
   forecast-hour GRIB of the era's stream(s) and dump per-hour per-message
   parquets to GCS `fmrc/scan_grib{date}/`.
2. Realigner (`ecmwf_par_to_ensemble_members.py`): `grib_tree` +
   `strip_datavar_chunks` over those dumps, split into per-(member,hour)
   deflated `.par` files -> the template tar
   `gik-fmrc/v2ecmwf_fmrc/{ens_control|ens_NN}/ecmwf-{REF}{run}-{m}-rt{HHH}.par`.

HOW CFS DIFFERS (why this is a single self-contained orchestrator)
------------------------------------------------------------------
CFS publishes ONE file per member per hour (member "01"), so there is no
51-members-in-one-file split — the realigner collapses to a single `grib_tree`.
And `lithops-cr-gik-cfs/run_lithops_cfs.py` reads only a SINGLE deflated
skeleton at runtime:
    gik-fmrc/v2cfs_fmrc/{init_date}_{run}/cfs-{init_date}{run}-member01-rt000.par
then overlays fresh `.idx` byte-ranges per step (`merge_with_template`, a pure
dict overlay). So CFS needs exactly ONE schema-complete skeleton, not a
per-(member,hour) corpus.

EMPIRICAL SCHEMA NOTE (verified 2026-07-02 against noaa-cfs-pds flxf)
--------------------------------------------------------------------
For CFS flux files the full variable/level/stepType schema is already complete
after f000 + f006:
    f000 -> 90 zarray-prefixes; f006 -> 92 (+2 avg-stepType fields that need a
    6h interval, e.g. PRATE/radiation); f240 and f1500 add NOTHING.
So a whole-corpus scan produces the IDENTICAL skeleton to the 2-file scan; the
`--full-corpus` mode below therefore exists as an AUDIT (prove no later hour
introduces new schema) and for exact parity with the ECMWF routine — it is not
required to obtain a correct skeleton. Default `--hours 0,6` is the cheap,
provably-complete build.

BACKENDS
--------
  --backend local  (default): scan the hours sequentially here (needs cfgrib;
                    the uv deps above provide it). ~2-4 s/file.
  --backend coiled: parallelize the per-hour scans on Coiled, mirroring the
                    ECMWF drivers' harness (anon-S3 forcing, gik-coiled-pinned
                    with gik-coiled-v6 fallback, n2-standard-2, us-east1,
                    workspace gcp-sewaa-nka). Workers RETURN their scan groups
                    to the client (no GCS write -> no coiled-data.json needed),
                    and the client assembles one grib_tree. NOTE: returning raw
                    groups is ~0.9 MB/file, so `--full-corpus` over Coiled ships
                    ~790 MB back — only use coiled for large `--hours` sets when
                    you actually need the audit.

Usage:
    # Build + package the template NOW from the proven-complete 2-file scan (free)
    uv run build_cfs_template.py --date 20260630 --run 00 --out ./cfs_template

    # Exact-routine parity / completeness audit: scan the whole corpus on Coiled
    uv run build_cfs_template.py --date 20260630 --run 00 --backend coiled \
        --full-corpus --audit --out ./cfs_template

    # Plan only, no scan / no Coiled / no cost
    uv run build_cfs_template.py --date 20260630 --run 00 --full-corpus --dry-run
"""
import argparse
import copy
import io
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")

S3_BUCKET = "noaa-cfs-pds"
CFS_MEMBER = "01"                      # 6hrly_grib_01 (matches run_lithops_cfs.py)
FORECAST_INTERVAL = 6                  # CFS flux files are 6-hourly
DEFAULT_MAX_FORECAST_HOURS = 5160      # ~215 days -> 861 files
SCHEMA_HOURS = [0, 6]                  # provably schema-complete (see header)


# --------------------------------------------------------------------------- #
# URL + plan
# --------------------------------------------------------------------------- #
def cfs_flux_url(date_str, run, forecast_hour, member=CFS_MEMBER):
    """S3 URL for one CFS flux GRIB (matches run_cfs_template_creation.py)."""
    init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
    forecast_dt = init_dt + timedelta(hours=forecast_hour)
    return (
        f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{member}/"
        f"flxf{forecast_dt:%Y%m%d%H}.{member}.{init_dt:%Y%m%d%H}.grb2"
    )


def resolve_hours(args):
    if args.full_corpus:
        return list(range(0, args.max_forecast_hours + 1, FORECAST_INTERVAL))
    if args.hours:
        return [int(h) for h in args.hours.split(",")]
    return list(SCHEMA_HOURS)


# --------------------------------------------------------------------------- #
# scan / assemble (client side, cfgrib-free except the local backend)
# --------------------------------------------------------------------------- #
def _force_anon_s3():
    import fsspec
    import s3fs
    fsspec.config.conf["s3"] = {"anon": True}
    s3fs.S3FileSystem.clear_instance_cache()


def scan_hour_local(hour, date, run):
    """scan_grib one flux file here; return (hour, groups)."""
    from kerchunk.grib2 import scan_grib
    _force_anon_s3()
    url = cfs_flux_url(date, run, hour)
    groups = scan_grib(url, storage_options={"anon": True})
    return hour, groups


def zarray_prefixes(groups):
    """Set of '<var>/<stepType>/<level>/<var>/' prefixes = the schema signature."""
    sig = set()
    for g in groups:
        for k in g.get("refs", {}):
            if k.endswith(".zarray"):
                sig.add(k[: -len(".zarray")])
    return sig


def deflate_tree(all_groups):
    """grib_tree(all_groups) -> deflated store (metadata only), matching
    run_cfs_template_creation.create_zarr_template exactly."""
    from kerchunk.grib2 import grib_tree
    try:
        tree = grib_tree(all_groups, remote_options={"anon": True})
    except TypeError:                                   # older kerchunk signature
        tree = grib_tree(all_groups)
    deflated = copy.deepcopy(tree)
    refs = deflated.get("refs", {})
    drop = [
        k for k in refs
        if not k.startswith(".") and "/" in k
        and not any(k.endswith(ext) for ext in (".zattrs", ".zarray", ".zgroup"))
    ]
    for k in drop:
        del refs[k]
    return refs, len(drop)


def store_to_keyvalue_df(refs):
    """Encode the deflated store as the ['key','value'] parquet the runtime reads
    (bytes; dict/list -> json, str -> utf-8)."""
    import pandas as pd
    rows = []
    for key, value in refs.items():
        if isinstance(value, (dict, list)):
            enc = json.dumps(value).encode("utf-8")
        elif isinstance(value, str):
            enc = value.encode("utf-8")
        else:
            enc = str(value).encode("utf-8")
        rows.append((key, enc))
    return pd.DataFrame(rows, columns=["key", "value"])


# --------------------------------------------------------------------------- #
# Coiled backend — mirrors ecmwf_*_coiled_preprocessing.py harness
# --------------------------------------------------------------------------- #
def scan_hours_coiled(hours, date, run, args):
    import coiled
    from distributed import Client

    @coiled.function(
        vm_type=args.vm_type,
        software=args.software,
        workspace=args.workspace,
        name="func-cfs-template",
        region=args.region,
        arm=False,
        idle_timeout="30 minutes",
    )
    def scan_one(hour):
        # gik-coiled image bakes stale AWS creds; force ANONYMOUS S3 (public
        # CFS bucket), exactly like the ECMWF drivers.
        import os
        import fsspec
        import s3fs
        fsspec.config.conf["s3"] = {"anon": True}
        s3fs.S3FileSystem.clear_instance_cache()
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/nonexistent"
        from datetime import datetime, timedelta
        idt = datetime.strptime(f"{date}{run}", "%Y%m%d%H")
        fdt = idt + timedelta(hours=hour)
        url = (f"s3://{S3_BUCKET}/cfs.{date}/{run}/6hrly_grib_{CFS_MEMBER}/"
               f"flxf{fdt:%Y%m%d%H}.{CFS_MEMBER}.{idt:%Y%m%d%H}.grb2")
        # kerchunk.grib2 on the image (0.2.6) or via uv (0.2.7) — both expose scan_grib
        from kerchunk.grib2 import scan_grib
        groups = scan_grib(url, storage_options={"anon": True})
        return hour, groups

    cluster = scan_one.cluster
    cluster.adapt(min=1, max=args.max_workers)
    _ = Client(cluster)                                 # bind client to the cluster
    return list(scan_one.map(hours))


# --------------------------------------------------------------------------- #
# packaging (tar layout consumed by run_lithops_cfs.py:230-232)
# --------------------------------------------------------------------------- #
def write_template(refs, out_root, ref_date, run):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    df = store_to_keyvalue_df(refs)

    # loose skeleton parquet (usable via run_lithops_cfs.py --local-template)
    skeleton = out_root / f"cfs-zarr-template-{ref_date}-{run}.parquet"
    df.to_parquet(skeleton)

    # tar.gz in the EXACT layout the runtime expects for one (ref_date, run)
    member_path = (f"gik-fmrc/v2cfs_fmrc/{ref_date}_{run}/"
                   f"cfs-{ref_date}{run}-member{CFS_MEMBER}-rt000.par")
    tar_path = out_root / "gik-fmrc-v2cfs_fmrc.tar.gz"
    buf = io.BytesIO()
    df.to_parquet(buf)
    data = buf.getvalue()
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo(name=member_path)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return skeleton, tar_path, member_path, len(df)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="scan date YYYYMMDD")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"])
    ap.add_argument("--ref-date", default=None,
                    help="reference date embedded in the tar path (default: --date)")
    ap.add_argument("--hours", default=None,
                    help="comma list of forecast hours to scan "
                         "(default: 0,6 = provably schema-complete)")
    ap.add_argument("--full-corpus", action="store_true",
                    help="scan every 6-hourly hour 0..max (audit / ECMWF parity)")
    ap.add_argument("--max-forecast-hours", type=int, default=DEFAULT_MAX_FORECAST_HOURS)
    ap.add_argument("--audit", action="store_true",
                    help="report any hour whose schema exceeds the 0,6 baseline")
    ap.add_argument("--backend", choices=["local", "coiled"], default="local")
    ap.add_argument("--out", default="./cfs_template", help="output dir")
    # Coiled config (mirrors ecmwf_*_coiled_preprocessing.py)
    ap.add_argument("--software", default="gik-coiled-pinned",
                    help="Coiled software env (fallback: --software gik-coiled-v6)")
    ap.add_argument("--workspace", default="gcp-sewaa-nka")
    ap.add_argument("--vm-type", default="n2-standard-2")
    ap.add_argument("--region", default="us-east1")
    ap.add_argument("--max-workers", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; no scan, no Coiled, no cost")
    args = ap.parse_args()

    ref_date = args.ref_date or args.date
    hours = resolve_hours(args)

    print(f"CFS template build: date={args.date} run={args.run}z ref={ref_date} "
          f"backend={args.backend} hours={len(hours)} "
          f"({'full corpus' if args.full_corpus else ','.join(map(str, hours))})")
    print(f"  sample: {cfs_flux_url(args.date, args.run, hours[0])}")
    print(f"  tar member -> gik-fmrc/v2cfs_fmrc/{ref_date}_{args.run}/"
          f"cfs-{ref_date}{args.run}-member{CFS_MEMBER}-rt000.par")
    if args.backend == "coiled":
        payload_mb = len(hours) * 0.92
        print(f"  coiled: software={args.software} workspace={args.workspace} "
              f"max_workers={args.max_workers}  (~{payload_mb:.0f} MB returned)")
    if args.dry_run:
        return 0

    # --- scan ---
    t0 = time.time()
    if args.backend == "coiled":
        results = scan_hours_coiled(hours, args.date, args.run, args)
    else:
        results = [scan_hour_local(h, args.date, args.run) for h in hours]
    results = [r for r in results if r and r[1]]
    print(f"scanned {len(results)}/{len(hours)} hours in {time.time()-t0:.1f}s")

    by_hour = {h: g for h, g in results}

    # --- audit (does any hour exceed the 0,6 baseline?) ---
    if args.audit or args.full_corpus:
        base = set()
        for h in SCHEMA_HOURS:
            if h in by_hour:
                base |= zarray_prefixes(by_hour[h])
        extra_hours = []
        for h in sorted(by_hour):
            new = zarray_prefixes(by_hour[h]) - base
            if new:
                extra_hours.append((h, sorted(new)))
        if extra_hours:
            print(f"AUDIT: {len(extra_hours)} hour(s) add schema beyond f000+f006:")
            for h, new in extra_hours:
                print(f"  f{h:04d}: +{new}")
        else:
            print(f"AUDIT: no hour beyond f000+f006 adds schema "
                  f"({len(base)} prefixes) — 2-file scan is complete.")

    # --- assemble one deflated skeleton over all scanned groups ---
    all_groups = []
    for h in sorted(by_hour):
        all_groups.extend(by_hour[h])
    refs, dropped = deflate_tree(all_groups)
    print(f"deflated store: {len(refs)} metadata keys (dropped {dropped} data refs)")

    skeleton, tar_path, member_path, nkeys = write_template(
        refs, args.out, ref_date, args.run)
    print(f"\nSkeleton parquet : {skeleton}  ({nkeys} keys)")
    print(f"Template tar.gz  : {tar_path}")
    print(f"  member         : {member_path}")
    print("\nNext:")
    print("  # replicate the skeleton across an init month + upload to HuggingFace")
    print(f"  uv run cfs_package_template.py --skeleton {skeleton} \\")
    print(f"      --ref-date {ref_date} --run {args.run} --replicate-month {ref_date[:6]} \\")
    print(f"      --out gik-fmrc-v2cfs_fmrc.tar.gz --upload "
          f"--repo E4DRR/grib-index-kerchunk-templates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
