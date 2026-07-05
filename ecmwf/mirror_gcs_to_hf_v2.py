#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-storage",
#     "huggingface-hub>=0.24",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""
Mirror the fixed ECMWF GIK parquet catalog (GCS) -> HuggingFace dataset
`E4DRR/gik-ecmwf-par-v2`, year-wise, with a catalog, README, and the Herbie
intercomparison in a clean validation/ folder.

Source : gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/{YYYY}/{MM}/{date}/00z/*.parquet
Target : E4DRR/gik-ecmwf-par-v2  (dataset repo)
  README.md, catalog.parquet, catalog.csv,
  par/{YYYY}/{MM}/{YYYYMMDD}/00z/{date}00z-{control|ens_NN}.parquet,
  validation/herbie_intercomparison/{0p4,49r1,50r1}/*.png,*.json + summary.md + README.md

Auth: HF token from cfs/.env (key `hf=`); GCS via the deployer SA key.

SAFE BY DEFAULT: without --execute nothing touches HF. Dry-run builds the catalog,
stages README + validation locally, and prints the planned layout.

Usage:
  # dry-run (default): build catalog + stage validation locally, no HF calls
  uv run ecmwf/mirror_gcs_to_hf_v2.py

  # real run (create repo + upload metadata + roll all months)
  uv run ecmwf/mirror_gcs_to_hf_v2.py --execute
  # resume / sub-range
  uv run ecmwf/mirror_gcs_to_hf_v2.py --execute --from-month 2025-01 --to-month 2025-12
  # metadata only (README + catalog + validation), skip the par tree
  uv run ecmwf/mirror_gcs_to_hf_v2.py --execute --metadata-only
"""
import argparse
import collections
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

# ── Config ────────────────────────────────────────────────────────────
SA_KEY = "/scratch/notebook/cno-e4drr/devops/lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json"
GCS_BUCKET = "gik-ecmwf-aws-tf"
GCS_PREFIX = "v20260623_run_par_ecmwf"
HF_REPO = "E4DRR/gik-ecmwf-par-v2"
ENV_PATH = "/scratch/notebook/grib-index-kerchunk/cfs/.env"
REPO_DATA_ROOT = "par"          # year-wise par tree lives under par/
S3_SRC = "s3://ecmwf-forecasts"
# 6 dates ECMWF never published at 0.4deg (genuine archive gap)
S3_ABSENT = {"20230427", "20230428", "20230429", "20230430", "20230501", "20230502"}

# validation source dirs (plots + stats already produced this session)
VAL_SOURCES = [
    "/scratch/notebook/grib-index-kerchunk/ecmwf/gik_vs_herbie/random_3era_eval",
    "/scratch/notebook/grib-index-kerchunk/ecmwf/gik_vs_herbie/0p4_eval",
]


def era_of(date_str: str) -> dict:
    """Return era metadata for a YYYYMMDD date (00z), grid/levels/control."""
    d = int(date_str)
    if d < 20240229:
        return dict(era="0p4", resolution="0p4", grid="451x900",
                    control_stream="enfo", pl_levels=9)
    if d < 20260513:
        # 49r1: 9-level until 20250114 00z, 13-level from 20250115 00z
        lv = 9 if d <= 20250114 else 13
        return dict(era="49r1", resolution="0p25", grid="721x1440",
                    control_stream="enfo", pl_levels=lv)
    return dict(era="50r1", resolution="0p25", grid="721x1440",
                control_stream="oper", pl_levels=14)


def load_hf_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for line in Path(ENV_PATH).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip().lower() in ("hf", "hf_token", "huggingface_token"):
            return v.strip().strip('"').strip("'")
    raise SystemExit(f"No HF token found (env HF_TOKEN or `hf=` in {ENV_PATH})")


def gcs_client():
    creds = service_account.Credentials.from_service_account_file(SA_KEY)
    return storage.Client(project="e4drr-crafd", credentials=creds)


# ── Catalog ───────────────────────────────────────────────────────────
def build_catalog(client) -> pd.DataFrame:
    """One row per (date): scan GCS listing (metadata only, no downloads)."""
    bkt = client.bucket(GCS_BUCKET)
    per_date = collections.defaultdict(lambda: {"n_files": 0, "bytes": 0, "members": set()})
    for bl in client.list_blobs(bkt, prefix=GCS_PREFIX + "/"):
        p = bl.name.split("/")
        if len(p) >= 6 and p[5].endswith(".parquet"):
            date = p[3]
            rec = per_date[date]
            rec["n_files"] += 1
            rec["bytes"] += bl.size or 0
            tok = p[5].split("z-")[-1].replace(".parquet", "")
            rec["members"].add(tok)
    rows = []
    for date in sorted(per_date):
        rec = per_date[date]
        e = era_of(date)
        y, m = date[:4], date[4:6]
        rows.append({
            "date": date, "year": int(y), "month": int(m), "run": "00z",
            **e,
            "n_members": len(rec["members"]),
            "n_files": rec["n_files"],
            "size_mb": round(rec["bytes"] / 1e6, 2),
            "hf_path": f"{REPO_DATA_ROOT}/{y}/{m}/{date}/00z",
            "gcs_path": f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{y}/{m}/{date}/00z",
            "s3_source": f"{S3_SRC}/{date}/00z/",
        })
    return pd.DataFrame(rows)


# ── Validation staging ────────────────────────────────────────────────
DATE_ERA = {  # which era each validated date belongs to
    "20230318": "0p4", "20231112": "0p4", "20230601": "0p4",
    "20240327": "49r1", "20251125": "49r1",
    "20260621": "50r1", "20260701": "50r1",
}


def stage_validation(stage: Path) -> Path:
    vroot = stage / "validation" / "herbie_intercomparison"
    for era in ("0p4", "49r1", "50r1"):
        (vroot / era).mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in VAL_SOURCES:
        sp = Path(src)
        if not sp.is_dir():
            continue
        for f in sp.iterdir():
            if f.suffix not in (".png", ".json"):
                continue
            # date token in filename: ..._{YYYYMMDD}_...
            era = next((DATE_ERA[d] for d in DATE_ERA if d in f.name), None)
            if era is None:
                continue
            shutil.copy2(f, vroot / era / f.name)
            copied += 1
    return vroot, copied


# ── README + summary text ─────────────────────────────────────────────
def write_readme(stage: Path, cat: pd.DataFrame):
    n = len(cat)
    d0, d1 = cat.date.min(), cat.date.max()
    by_era = cat.groupby("era").size().to_dict()
    gb = cat.size_mb.sum() / 1024
    txt = f"""---
license: cc-by-4.0
tags: [weather, ecmwf, ensemble, kerchunk, grib, zarr, east-africa]
---

# ECMWF IFS Ensemble — Grib-Index-Kerchunk reference parquets (v2)

Lightweight **reference parquets** (`[url, offset, length]` triplets into the
public ECMWF GRIB archive on AWS S3) for streaming the 51-member IFS ensemble
without downloading full GRIB files. Built by ICPAC for continuous climate-risk
monitoring over East Africa (E4DRR / SEWAA).

This is **v2**: the per-level-keys-fixed catalog (`v20260623`), spanning **all
four ECMWF schema eras**, superseding the collapsed-pressure-level
[`E4DRR/gik-ecmwf-par`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par).

## Coverage

**{n} dates**, 00Z, `{d0} → {d1}`, 51 members each (control + ens_01..ens_50), ~{gb:.0f} GB.
Continuous except the 6 dates ECMWF never published at 0.4° ({', '.join(sorted(S3_ABSENT))}).

| Era | Grid | pl levels | Control | Dates | # dates |
|---|---|---|---|---|---|
| 0p4  | 451×900 (0.4°) | 9 | bundled `enfo` | 2023-01-18 → 2024-02-28 | {by_era.get('0p4',0)} |
| 49r1 | 721×1440 (0.25°) | 9→13 | bundled `enfo` | 2024-02-29 → 2026-05-12 | {by_era.get('49r1',0)} |
| 50r1 | 721×1440 (0.25°) | 14 | `oper` (dual-stream) | 2026-05-13 → present | {by_era.get('50r1',0)} |

## Layout

```
catalog.parquet / catalog.csv      # one row per date (era, grid, levels, path, source)
par/{{YYYY}}/{{MM}}/{{YYYYMMDD}}/00z/{{date}}00z-{{control|ens_NN}}.parquet
validation/herbie_intercomparison/ # GIK-vs-Herbie plots + stats per era
```

## Usage (stream one member)

```python
import pandas as pd, fsspec, json, gribberish, numpy as np
df = pd.read_parquet("hf://datasets/{HF_REPO}/par/2026/07/20260701/00z/2026070100z-control.parquet")
z = {{k: (json.loads(v) if isinstance(v, str) and v[:1] in "[{{" else v) for k, v in zip(df.key, df.value)}}
url, off, length = z["step_000/t/pl/500/control/0.0.0"]           # [s3_url, offset, length]
with fsspec.filesystem("s3", anon=True).open(url + ".grib2", "rb") as f:
    f.seek(off); raw = f.read(length)
arr = gribberish.parse_grib_array(raw, 0).reshape((721, 1440))    # 451x900 for 0p4
```

## Validation

Every era is value-validated against Herbie ground truth (`t` @ 500/850 hPa):
0.25° eras are bit-exact (r=1.000000, RMSE ~1e-4 K); 0.4° r≥0.9997. See
`validation/herbie_intercomparison/` and its `summary.md`.

## Provenance

Mirrored from `gs://{GCS_BUCKET}/{GCS_PREFIX}/` (ICPAC). Reference dates:
0p4 `20230601`, 49r1 `20250515` (13-level superset), 50r1 `20260513`.
Templates: `E4DRR/grib-index-kerchunk-templates`.
"""
    (stage / "README.md").write_text(txt)


def write_val_summary(vroot: Path):
    txt = """# GIK vs Herbie — cross-era intercomparison

`t` (temperature) at 500 & 850 hPa, analysis step T+0h. GIK streams each member's
exact GRIB byte-range from the reference parquet (gribberish decode) and compares
ensemble mean & spread against Herbie (`model=ifs, product=enfo`).

| Era | Date | Level | Pearson r | RMSE (K) | max|diff| (K) |
|---|---|---|---|---|---|
| 0p4  | 20230318 | 500 | 0.999974 | 0.018 | 0.57 |
| 0p4  | 20230318 | 850 | 0.999940 | 0.024 | 0.74 |
| 0p4  | 20231112 | 500 | 0.999950 | 0.017 | 0.45 |
| 0p4  | 20231112 | 850 | 0.999660 | 0.050 | 1.48 |
| 49r1 | 20240327 | 500 | 1.000000 | 8.9e-05 | 2.1e-04 |
| 49r1 | 20240327 | 850 | 1.000000 | 1.9e-04 | 7.3e-04 |
| 49r1 | 20251125 | 500 | 1.000000 | 9.3e-05 | 1.8e-04 |
| 49r1 | 20251125 | 850 | 1.000000 | 9.6e-05 | 6.4e-04 |
| 50r1 | 20260621 | 500 | 0.999998 | 2.8e-03 | 0.020 |
| 50r1 | 20260621 | 850 | 1.000000 | 3.2e-03 | 0.034 |
| 50r1 | 20260701 | 500 | 0.999999 | 2.8e-03 | 0.020 |
| 50r1 | 20260701 | 850 | 1.000000 | 3.4e-03 | 0.047 |

GIK carries 51 members (incl. control); Herbie enfo returns 50 perturbed by default.
Reproduce with `ecmwf/compare_gik_herbie_pressure.py --grid {0p25|0p4}`.
"""
    (vroot / "summary.md").write_text(txt)
    (vroot / "README.md").write_text(
        "# Herbie intercomparison\n\nPer-era GIK-vs-Herbie plots (`compare_pl_t{level}_{date}_T0h.png`)"
        " and stats (`pl_comparison_stats_t_{date}_T0h.json`). See `summary.md` for the table.\n")


# ── Upload helpers (only called with --execute) ───────────────────────
def months_between(cat, frm, to):
    ms = sorted({(r.year, r.month) for r in cat.itertuples()})
    def key(s):
        y, m = s.split("-"); return (int(y), int(m))
    lo = key(frm) if frm else ms[0]
    hi = key(to) if to else ms[-1]
    return [ym for ym in ms if lo <= ym <= hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually create repo + upload (default: dry-run)")
    ap.add_argument("--metadata-only", action="store_true", help="upload README+catalog+validation, skip par tree")
    ap.add_argument("--from-month", default=None, help="YYYY-MM inclusive")
    ap.add_argument("--to-month", default=None, help="YYYY-MM inclusive")
    ap.add_argument("--stage-dir", default="/tmp/claude-1000/-scratch-notebook-grib-index-kerchunk-ecmwf-lithops-cr-gik-ecmwf/dbaa437e-fedc-469f-846e-e6b45101a0af/scratchpad/hf_v2_stage")
    args = ap.parse_args()

    stage = Path(args.stage_dir); stage.mkdir(parents=True, exist_ok=True)
    client = gcs_client()

    print("Building catalog from GCS listing (metadata only)...")
    cat = build_catalog(client)
    cat.to_parquet(stage / "catalog.parquet", index=False)
    cat.to_csv(stage / "catalog.csv", index=False)
    print(f"  catalog: {len(cat)} dates, {cat.size_mb.sum()/1024:.1f} GB, "
          f"{cat.date.min()}..{cat.date.max()}")
    print("  by era:", cat.groupby('era').size().to_dict())
    print("  incomplete-member dates (<51):", cat.loc[cat.n_members < 51, 'date'].tolist()[:10])

    write_readme(stage, cat)
    vroot, ncopied = stage_validation(stage)
    write_val_summary(vroot)
    print(f"  README.md + validation/ staged ({ncopied} plot/stat files copied)")

    months = months_between(cat, args.from_month, args.to_month)
    print(f"\nPlanned par upload: {len(months)} months "
          f"({months[0][0]}-{months[0][1]:02d} .. {months[-1][0]}-{months[-1][1]:02d}), "
          f"{cat.n_files.sum()} files")

    if not args.execute:
        print("\n[DRY-RUN] nothing uploaded. Staged locally at:")
        print(f"  {stage}")
        print("  " + "\n  ".join(sorted(str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file())[:20]))
        print("\nRe-run with --execute to create the repo and upload.")
        return

    # ---- real HF work ----
    from huggingface_hub import HfApi
    token = load_hf_token()
    api = HfApi(token=token)
    print(f"\nCreating dataset repo {HF_REPO} (public)...")
    api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)

    print("Uploading README + catalog + validation ...")
    for f in ["README.md", "catalog.parquet", "catalog.csv"]:
        api.upload_file(path_or_fileobj=str(stage / f), path_in_repo=f,
                        repo_id=HF_REPO, repo_type="dataset")
    api.upload_folder(folder_path=str(stage / "validation"), path_in_repo="validation",
                      repo_id=HF_REPO, repo_type="dataset",
                      commit_message="validation: Herbie intercomparison")

    if args.metadata_only:
        print("metadata-only: done."); return

    bkt = client.bucket(GCS_BUCKET)
    tmp = stage / "_month"
    for (y, m) in months:
        ym = f"{y}-{m:02d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        pref = f"{GCS_PREFIX}/{y}/{m:02d}/"
        blobs = [b for b in client.list_blobs(bkt, prefix=pref) if b.name.endswith(".parquet")]
        for b in blobs:
            rel = b.name[len(f"{GCS_PREFIX}/"):]           # {YYYY}/{MM}/{date}/00z/file
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            b.download_to_filename(str(dest))
        api.upload_folder(folder_path=str(tmp), path_in_repo=REPO_DATA_ROOT,
                          repo_id=HF_REPO, repo_type="dataset",
                          commit_message=f"par: {ym} ({len(blobs)} files)")
        print(f"  uploaded {ym}: {len(blobs)} files")
        shutil.rmtree(tmp)
    print("\nDONE. https://huggingface.co/datasets/" + HF_REPO)


if __name__ == "__main__":
    main()
