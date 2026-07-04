#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "pyarrow",
#     "huggingface_hub",
# ]
# ///
"""
CFS template packaging — bundle the zarr skeleton into gik-fmrc-v2cfs_fmrc.tar.gz
and (optionally) upload to HuggingFace.

This is the step AFTER Stage-1 preprocessing that makes the CFS Cloud Run path
work. It mirrors the ECMWF runbook
(ecmwf/.../2026-06-03-three-era-scan-run-verification-and-perlevel-fix.md §6.3-6.4):
    skeleton parquet  ->  tar.gz in the layout the runtime expects  ->  HuggingFace

What run_lithops_cfs.py expects (build_deflated_store_from_template, line ~230):
    tar member path:
        gik-fmrc/v2cfs_fmrc/{init_date}_{run}/cfs-{init_date}{run}-member{MM}-rt000.par
    member content:
        a parquet with ['key','value'] columns = the deflated zarr skeleton
        (exactly what `run_cfs_template_creation.py --zarr-template` produces).

IMPORTANT — lookup is by PROCESSING date, not a fixed reference date.
`build_deflated_store_from_template(template_path, init_date, run)` is called with
the date being processed (run_lithops_cfs.py:790), and that date is embedded in
the tar member path. A pre-built tar.gz therefore needs ONE entry per
(init_date, run) it will be asked to serve. Because `merge_with_template` is a
pure dict overlay that keeps the skeleton metadata as-is and overlays only the
fresh date-specific byte-refs, the skeleton STRUCTURE (variable/dim/chunk shapes)
is date-independent and can be replicated across dates. Use --replicate-month /
--replicate-dates to write the same skeleton under every member of a CFS init
month (31 days x 4 runs = 124 members).

  CAVEAT: replication reuses the reference skeleton's coordinate metadata for all
  dates. Validate a streamed decode for a non-reference date before trusting it
  in production (no CFS GIK-vs-Herbie validation exists yet).

Build the skeleton first (one-time, ~24s, not this script's job):
    uv run run_cfs_template_creation.py --zarr-template \
        --init-date 20251101 --run 00 \
        --local-dir ./cfs_test/template --max-forecast-hours 48
    # -> cfs_test/template/cfs-zarr-template-20251101-00.parquet

Then package + upload:
    # 1. dry-run: show the tar layout, no file written, no upload
    uv run cfs_package_template.py \
        --skeleton cfs_test/template/cfs-zarr-template-20251101-00.parquet \
        --ref-date 20251101 --run 00 --dry-run

    # 2. write the tar.gz locally (reference member only)
    uv run cfs_package_template.py \
        --skeleton cfs_test/template/cfs-zarr-template-20251101-00.parquet \
        --ref-date 20251101 --run 00 --out gik-fmrc-v2cfs_fmrc.tar.gz

    # 3. replicate across a whole init month + upload to HuggingFace
    uv run cfs_package_template.py \
        --skeleton cfs_test/template/cfs-zarr-template-20251101-00.parquet \
        --ref-date 20251101 --run 00 \
        --replicate-month 202511 \
        --out gik-fmrc-v2cfs_fmrc.tar.gz \
        --upload --repo E4DRR/grib-index-kerchunk-templates

After upload, point run_lithops_cfs.py at it: its TEMPLATE_URL currently targets
`Nishadhka/gfs_s3_gik_refs/.../gik-fmrc-v2cfs_fmrc.tar.gz` (404). Set env
TEMPLATE_URL (or edit line ~107) to the repo you upload to here.
"""
import argparse
import io
import os
import sys
import tarfile
from calendar import monthrange

import pandas as pd

CFS_MEMBER = "01"  # 6hrly_grib_01 — matches run_lithops_cfs.py CFS_MEMBER
DEFAULT_OUT = "gik-fmrc-v2cfs_fmrc.tar.gz"
DEFAULT_REPO = "E4DRR/grib-index-kerchunk-templates"
RUNS = ["00", "06", "12", "18"]


def tar_member_path(date: str, run: str, member: str) -> str:
    """The exact arcname run_lithops_cfs.py looks up (line ~230-233)."""
    return (
        f"gik-fmrc/v2cfs_fmrc/{date}_{run}/"
        f"cfs-{date}{run}-member{member}-rt000.par"
    )


def expand_month(month: str) -> list:
    """All (date, run) pairs of an init month -> up to 31x4 = 124 members."""
    year, mon = int(month[:4]), int(month[4:6])
    _, last_day = monthrange(year, mon)
    return [(f"{year}{mon:02d}{day:02d}", run)
            for day in range(1, last_day + 1) for run in RUNS]


def parse_dates_arg(spec: str) -> list:
    """Parse --replicate-dates 'YYYYMMDD:HH,YYYYMMDD:HH' (HH optional -> all runs)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            d, r = tok.split(":", 1)
            out.append((d, r))
        else:
            out.extend((tok, r) for r in RUNS)
    return out


def load_skeleton_bytes(path: str) -> bytes:
    """Read + validate the skeleton parquet; return its raw bytes."""
    df = pd.read_parquet(path)
    if list(df.columns) != ["key", "value"] and not {"key", "value"} <= set(df.columns):
        sys.exit(f"skeleton must have ['key','value'] columns; got {list(df.columns)}")
    meta = sum(1 for k in df["key"] if str(k).endswith((".zarray", ".zattrs", ".zgroup")))
    print(f"skeleton: {path}  ({len(df)} keys, {meta} zarr-metadata keys)")
    with open(path, "rb") as fh:
        return fh.read()


def build_member_set(args) -> list:
    """Resolve the full (date, run) set to write into the tar."""
    members = [(args.ref_date, args.run)]
    if args.replicate_month:
        members += expand_month(args.replicate_month)
    if args.replicate_dates:
        members += parse_dates_arg(args.replicate_dates)
    # de-dup, stable order
    seen, uniq = set(), []
    for dr in members:
        if dr not in seen:
            seen.add(dr)
            uniq.append(dr)
    return uniq


def count_mapping_members(mappings_tar: str) -> int:
    with tarfile.open(mappings_tar, "r:gz") as mtar:
        return sum(1 for m in mtar.getmembers() if m.isfile())


def write_tar(out_path: str, skeleton: bytes, members: list, member_id: str,
              mappings_tar: str = None):
    """Write the skeleton bytes under one arcname per (date, run), and (optionally)
    fold in every mapping parquet from `mappings_tar` at its own arcname."""
    n_map = 0
    with tarfile.open(out_path, "w:gz") as tar:
        for date, run in members:
            arc = tar_member_path(date, run, member_id)
            info = tarfile.TarInfo(name=arc)
            info.size = len(skeleton)
            info.mtime = 0  # reproducible tar
            tar.addfile(info, io.BytesIO(skeleton))
        if mappings_tar:
            with tarfile.open(mappings_tar, "r:gz") as mtar:
                for m in mtar.getmembers():
                    if not m.isfile():
                        continue
                    data = mtar.extractfile(m).read()
                    info = tarfile.TarInfo(name=m.name)
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                    n_map += 1
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"wrote {out_path}  ({len(members)} skeleton member(s)"
          f"{f' + {n_map} mapping parquet(s)' if mappings_tar else ''}, {size_mb:.2f} MB)")


def upload_hf(out_path: str, repo: str):
    """Upload the tar.gz to a HuggingFace dataset repo."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        # try a repo-root / cwd .env (KEY=VALUE lines), like the ECMWF uploader
        for env_path in (".env", os.path.join("..", ".env")):
            if os.path.exists(env_path):
                for line in open(env_path):
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") and "=" in line:
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if token:
                break
    # token=None falls back to a cached `huggingface-cli login`
    from huggingface_hub import upload_file
    dest = os.path.basename(out_path)
    print(f"uploading {dest} -> hf://datasets/{repo}/{dest} ...")
    upload_file(
        path_or_fileobj=out_path,
        path_in_repo=dest,
        repo_id=repo,
        repo_type="dataset",
        token=token,
    )
    print(f"done: https://huggingface.co/datasets/{repo}/resolve/main/{dest}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skeleton", required=True,
                    help="zarr skeleton parquet from `--zarr-template`")
    ap.add_argument("--ref-date", required=True, help="reference init date YYYYMMDD")
    ap.add_argument("--run", default="00", choices=RUNS)
    ap.add_argument("--member", default=CFS_MEMBER, help="CFS member id (default 01)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output tar.gz (default {DEFAULT_OUT})")
    ap.add_argument("--replicate-month", metavar="YYYYMM",
                    help="also write the skeleton under every (date,run) of this init month")
    ap.add_argument("--replicate-dates", metavar="LIST",
                    help="also write under 'YYYYMMDD:HH,YYYYMMDD,...' (HH optional -> all runs)")
    ap.add_argument("--mappings-tar", metavar="TAR",
                    help="fold the mapping parquets from this tar.gz "
                         "(cfs_coiled_preprocessing.py output) into the same tar -> the "
                         "full GEFS-model template (deflated skeleton + idx->grib mappings)")
    ap.add_argument("--upload", action="store_true", help="upload the tar.gz to HuggingFace")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF dataset repo for --upload (default {DEFAULT_REPO})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; no tar written, no upload")
    args = ap.parse_args()

    if not os.path.exists(args.skeleton):
        sys.exit(f"skeleton not found: {args.skeleton}")
    if args.mappings_tar and not os.path.exists(args.mappings_tar):
        sys.exit(f"mappings tar not found: {args.mappings_tar}")

    members = build_member_set(args)
    print(f"plan: {len(members)} skeleton member(s), member id {args.member}")
    print(f"  e.g. {tar_member_path(*members[0], args.member)}")
    if len(members) > 1:
        print(f"       ... and {len(members) - 1} more "
              f"({members[1][0]}_{members[1][1]} .. {members[-1][0]}_{members[-1][1]})")
    if args.mappings_tar:
        print(f"  + mappings: {count_mapping_members(args.mappings_tar)} parquet(s) "
              f"from {args.mappings_tar}")
    print(f"  out: {args.out}" + ("  (+ upload to " + args.repo + ")" if args.upload else ""))

    if args.dry_run:
        return 0

    skeleton = load_skeleton_bytes(args.skeleton)
    write_tar(args.out, skeleton, members, args.member, args.mappings_tar)

    # quick self-check: re-open and confirm the reference member resolves
    with tarfile.open(args.out, "r:gz") as tar:
        probe = tar_member_path(args.ref_date, args.run, args.member)
        names = tar.getnames()
        assert probe in names, f"self-check failed: {probe} not in tar"
        df = pd.read_parquet(io.BytesIO(tar.extractfile(probe).read()))
        print(f"self-check OK: {probe} -> {len(df)} keys")

    if args.upload:
        upload_hf(args.out, args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
