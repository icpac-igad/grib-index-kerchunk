#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "pyyaml",
#     "pandas",
# ]
# ///
"""
02_measure_parquet_sizes.py
===========================

Measure the **storage volume of GIK parquet references** for the same sample
dates probed by ``01_measure_grib_sizes.py``, then compute the primary
compression ratio (GRIB bytes ÷ parquet bytes).

Why HuggingFace and not GCS? The production GCS buckets
(``gs://gik-{gefs,ecmwf}-aws-tf``) are not publicly listable. The same
parquets are mirrored to the public HuggingFace datasets
``E4DRR/gik-gefs-par`` and ``E4DRR/gik-ecmwf-par``, which expose per-file
sizes via the dataset metadata API — no downloads required.

Also measures the **one-time template tar.gz** sizes
(``gik-fmrc-gefs-20241112.tar.gz``, ``gik-fmrc-v2ecmwf_fmrc.tar.gz``) on
``Nishadhka/gfs_s3_gik_refs`` — these are the only "fixed cost" GIK adds
to the publisher's storage footprint.

Output schema (``parquet_sizes_<product>.csv``)::

    product, date, run, member, hf_path, size_bytes, status

``status`` is ``present`` if the parquet exists on HF, ``missing`` otherwise
(common for dates outside the HF mirror window — the script still reports
which sample dates are missing so the cost extrapolation is honest).

Usage
-----
    uv run cost-eval/02_measure_parquet_sizes.py
    uv run cost-eval/02_measure_parquet_sizes.py --products gefs
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import HfApi


# ---------------------------------------------------------------------------
# Constants — HF mirror layout
# ---------------------------------------------------------------------------

HF_PARQUET_REPOS = {
    "gefs":  "E4DRR/gik-gefs-par",
    "ecmwf": "E4DRR/gik-ecmwf-par",
}
HF_TEMPLATE_REPO = "Nishadhka/gfs_s3_gik_refs"
HF_TEMPLATE_FILES = {
    "gefs":  "gik-fmrc-gefs-20241112.tar.gz",
    "ecmwf": "gik-fmrc-v2ecmwf_fmrc.tar.gz",
}

# Member naming differs between products on the HF mirror:
#   GEFS  -> gep01..gep30
#   ECMWF -> control, ens_01..ens_50
ECMWF_MEMBERS = ["control"] + [f"ens_{i:02d}" for i in range(1, 51)]


def hf_path(product: str, date: str, run: str, member: str) -> str:
    yyyy, mm = date[:4], date[4:6]
    return (
        f"run_par_{product}/{yyyy}/{mm}/{date}/{run}z/"
        f"{date}{run}z-{member}.parquet"
    )


def expected_paths(product: str, dates: list[str], runs: list[str],
                   gefs_members: list[str]) -> list[dict]:
    members = gefs_members if product == "gefs" else ECMWF_MEMBERS
    rows = []
    for run in runs:
        for date in dates:
            for member in members:
                rows.append({
                    "product": product,
                    "date": date,
                    "run": run,
                    "member": member,
                    "hf_path": hf_path(product, date, run, member),
                })
    return rows


# ---------------------------------------------------------------------------
# HF lookup
# ---------------------------------------------------------------------------

def load_hf_index(repo: str) -> dict[str, int]:
    """One API call → {filename: size_bytes} for every file in the repo."""
    api = HfApi()
    info = api.dataset_info(repo, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings}


def annotate_with_sizes(rows: list[dict], index: dict[str, int]) -> pd.DataFrame:
    out = []
    for r in rows:
        r = dict(r)
        size = index.get(r["hf_path"])
        r["size_bytes"] = size
        r["status"] = "present" if size is not None else "missing"
        out.append(r)
    df = pd.DataFrame(out)
    df = df.sort_values(["product", "date", "run", "member"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _load_grib_summary(results_dir: Path) -> dict | None:
    path = results_dir / "01_grib_sizes_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def summarise(df: pd.DataFrame, product: str,
              parquet_runs: list[str],
              grib_summary: dict | None, archive_cfg: dict,
              template_size: int | None) -> dict:
    present = df[df["status"] == "present"]
    missing = df[df["status"] == "missing"]
    dates_present = present["date"].nunique()

    summary: dict = {
        "parquet_runs_mirrored": parquet_runs,
        "n_parquets_expected":   int(len(df)),
        "n_parquets_present":    int(len(present)),
        "n_parquets_missing":    int(len(missing)),
        "n_dates_sampled":       int(df["date"].nunique()),
        "n_dates_with_data":     int(dates_present),
        "dates_with_data":       sorted(present["date"].unique().tolist()),
        "files_per_run":         (
            present.groupby("run").size().astype(int).to_dict()
        ),
        "bytes_per_run":         (
            present.groupby("run")["size_bytes"].sum().astype(int).to_dict()
        ),
        "total_bytes_sampled":   int(present["size_bytes"].sum()),
        "total_mib_sampled":     round(present["size_bytes"].sum() / 2**20, 3),
        "mean_bytes_per_parquet": (
            float(present["size_bytes"].mean()) if len(present) else 0.0
        ),
    }
    if dates_present:
        per_date = present.groupby("date")["size_bytes"].sum()
        bytes_per_date = float(per_date.mean())   # mean across ALL mirrored runs
        n_full = archive_cfg["n_dates_published"]
        summary["mean_bytes_per_date_mirrored_runs"] = bytes_per_date
        summary["extrapolated_full_archive_bytes"] = int(bytes_per_date * n_full)
        summary["extrapolated_full_archive_gib"]   = round(
            bytes_per_date * n_full / 2**30, 3
        )

    # Apples-to-apples compression ratio against step-01 primary.
    # Both bytes are summed across the same `parquet_runs` cycles so we don't
    # mix a 4-run GEFS GRIB total with a 1-run GEFS parquet total.
    if grib_summary and dates_present:
        primary = grib_summary.get("primary", {}).get(product)
        if primary and primary.get("bytes_per_run"):
            grib_by_run = primary["bytes_per_run"]
            # GRIB primary covers ALL runs; restrict to parquet_runs.
            grib_matched_runs = sum(
                grib_by_run.get(r, 0) for r in parquet_runs
            )
            # primary is summed over n_dates_sampled dates; pro-rate to
            # dates_present (typically the same number).
            n_primary_dates = primary.get("n_dates_sampled") or dates_present
            g_matched = grib_matched_runs * dates_present / n_primary_dates
            p_total   = int(present["size_bytes"].sum())
            summary["matched_grib_bytes"]    = int(g_matched)
            summary["matched_parquet_bytes"] = p_total
            summary["compression_ratio"]     = (
                round(g_matched / p_total, 1) if p_total else None
            )
            summary["compression_note"] = (
                f"GRIB bytes for runs {parquet_runs} (from step 01 primary, "
                "same dates) ÷ parquet bytes for the same runs."
            )

    summary["template_tarball"] = HF_TEMPLATE_FILES[product]
    summary["template_bytes"]   = template_size
    summary["template_mib"]     = (
        round(template_size / 2**20, 3) if template_size else None
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--products", nargs="+", default=["gefs", "ecmwf"],
                    choices=["gefs", "ecmwf"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(__file__).with_name(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    grib_summary = _load_grib_summary(results_dir)
    if grib_summary is None:
        print("[warn] results/01_grib_sizes_summary.json not found — "
              "compression ratios will be omitted. Run 01 first.")

    template_index = load_hf_index(HF_TEMPLATE_REPO)

    headline: dict[str, dict] = {}
    for product in args.products:
        product_cfg = cfg["products"][product]
        archive_cfg = cfg["archive"][product]
        dates = cfg["sample"]["primary"][product]["dates"]
        parquet_runs = product_cfg.get("parquet_runs", ["00"])
        members_cfg = (
            product_cfg["members"] if product == "gefs" else ECMWF_MEMBERS
        )
        rows = expected_paths(product, dates, parquet_runs, members_cfg)

        repo = HF_PARQUET_REPOS[product]
        print(f"[{product}] loading HF file index for {repo} …")
        index = load_hf_index(repo)
        print(f"[{product}] repo has {len(index)} files; "
              f"checking {len(rows)} expected paths "
              f"({len(dates)} dates × {len(parquet_runs)} runs × "
              f"{len(members_cfg)} members)")

        df = annotate_with_sizes(rows, index)
        csv_path = results_dir / f"parquet_sizes_{product}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{product}] wrote {csv_path}")

        s = summarise(df, product, parquet_runs, grib_summary, archive_cfg,
                      template_index.get(HF_TEMPLATE_FILES[product]))
        headline[product] = s
        print(f"[{product}] {json.dumps(s, indent=2)}")

    summary_path = results_dir / "02_parquet_sizes_summary.json"
    summary_path.write_text(json.dumps(headline, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
