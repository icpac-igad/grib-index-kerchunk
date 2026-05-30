#!/usr/bin/env python3
"""Gate A0 — confirm the per-level bug in the live production template.

Pulls `gik-fmrc-v2ecmwf_fmrc.tar.gz` from the HF dataset
`E4DRR/grib-index-kerchunk-templates` (the 49r1 template ref 20240529 baked
into the production Cloud Run image), extracts one `control` rt000.par,
and inspects its zarr-store keys for the per-level collapse symptom
reported by the cGAN port.

DISCOVERED STRUCTURE (verified empirically 2026-05-29):
The legacy rt000.par uses path shape

    {var}/{typeOfLevel}/{N.0}/.zarray
    {var}/{typeOfLevel}/{N.0}/.zattrs

with NO `stepType` segment and a single numeric "N.0" segment. For pl
variables (u, v, gh, t, q, w, r, d, vo) there are 51 .zarray entries per
variable. The .zattrs report `level: 0.0` and `ens_number: -1` for some
entries -- structurally broken vs the (member × level) shape the cGAN
inputs need: 51 members × 13 isobaric levels = 663 distinct slots.

This Gate A0 confirms BUG-PRESENT on the artifact downstream consumers
actually see; it is NOT a test of the fix. Gate B1 (run scan_grib on
one real GRIB through the fixed aggregator) is the corresponding
fix-validation gate, and it will emit the NEW path shape

    {var}/{stepType}/{typeOfLevel}/{level}/...

which is incompatible with the legacy shape by design -- the legacy
shape is what the cGAN reports call physically incoherent.

Run:
    cd ecmwf/dev-test
    uv run --quiet --with pandas --with pyarrow --with huggingface-hub \\
      gate_a0_legacy_template.py
"""
from __future__ import annotations
import io
import sys
import tarfile
from pathlib import Path

import pandas as pd

HF_REPO = "E4DRR/grib-index-kerchunk-templates"
TAR_NAME = "gik-fmrc-v2ecmwf_fmrc.tar.gz"
PL_VARS = ["u", "v", "gh", "t", "q", "w", "r", "d", "vo"]


def chk(label, ok, *details):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        for d in details:
            print(f"         {d}")
    return ok


def find_rt000(tar) -> str:
    for m in tar.getmembers():
        if m.name.endswith("rt000.par") and "ens_control" in m.name:
            return m.name
    for m in tar.getmembers():
        if m.name.endswith("rt000.par"):
            return m.name
    raise SystemExit("no rt000.par found in tar.gz")


def main() -> int:
    print(f"Gate A0 — pull {HF_REPO} / {TAR_NAME}\n")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("re-run with --with huggingface-hub")

    tar_path = Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                                    filename=TAR_NAME))
    print(f"  tar.gz size:        {tar_path.stat().st_size:,} bytes")

    with tarfile.open(tar_path, "r:gz") as tar:
        rt_name = find_rt000(tar)
        print(f"  picked rt000.par:   {rt_name}")
        rt_bytes = tar.extractfile(rt_name).read()
    print(f"  rt000.par size:     {len(rt_bytes):,} bytes\n")

    df = pd.read_parquet(io.BytesIO(rt_bytes))
    keys = df["key"].astype(str).tolist()
    print(f"Parsed rt000.par: {len(keys)} zarr keys "
          f"(consolidated zarr-store shape ~2774 expected; "
          f"investigation MD §7)")

    all_ok = True

    # (A) Count .zarray entries per pl variable.
    # Pre-fix (BUG): 51 entries per pl var (one per "N.0" segment;
    # the 13 isobaric levels collapsed somewhere into this 51-slot axis).
    # Post-fix (target): 663 = 51 members × 13 levels.
    print("\n(A) .zarray entry counts for pl variables (legacy template):")
    print(f"    {'var':<6} {'.zarray':>8}  "
          f"{'expected (pre-fix)':>20}  {'expected (post-fix)':>20}")
    for var in PL_VARS:
        cnt = sum(1 for k in keys
                  if k.startswith(f"{var}/isobaricInhPa/")
                  and k.endswith("/.zarray"))
        print(f"    {var:<6} {cnt:>8}  "
              f"{'51 (BUG: collapsed)':>20}  "
              f"{'663 (51 mem * 13 lvl)':>20}")
        # Bug present when count is the legacy 51 and NOT the 663 we'd want.
        all_ok &= chk(
            f"{var}: legacy collapsed count == 51 (BUG present)",
            cnt == 51,
            f"observed: {cnt}",
        )

    # (B) Per-pl-var, are the path segments distinct member-or-level indices
    # but NOT the expected isobaric set {50,100,150,...,1000}?
    seg_values = sorted({
        k.split("/")[2] for k in keys
        if k.startswith("u/isobaricInhPa/") and k.endswith("/.zarray")
    }, key=lambda s: float(s))
    print(f"\n(B) u/isobaricInhPa path segments ({len(seg_values)}):")
    print(f"    first 5: {seg_values[:5]}  last 5: {seg_values[-5:]}")
    # Bug symptom: segment values are 0.0..50.0 (sequential indices) NOT the
    # isobaric set.
    expected_isobaric = {"50", "100", "150", "200", "250", "300", "400",
                         "500", "600", "700", "850", "925", "1000"}
    seg_as_int = {s.split(".")[0] for s in seg_values}
    matches_isobaric = seg_as_int >= expected_isobaric
    all_ok &= chk(
        "BUG: path segments are sequential indices, NOT the 13 isobaric levels",
        not matches_isobaric,
        f"observed (int parts): {sorted(seg_as_int, key=int)}",
    )

    # (C) Total chunk-ref keys vs zarr metadata keys.
    chunk_refs = df["key"].apply(
        lambda k: not (k.endswith(".zarray") or k.endswith(".zattrs")
                       or k.endswith(".zgroup")
                       or k in ("zarr_consolidated_format", "metadata"))
    ).sum()
    print(f"\n(C) chunk-ref keys (non-metadata): {chunk_refs}")
    print(f"    -> the template is metadata-only "
          f"(\"deflated store\"); chunk refs are filled in at runtime "
          f"by Stage 2 against fresh .index files.")

    # (D) Show one pl .zattrs to expose the broken `level` field.
    sample_attrs_key = "u/isobaricInhPa/0.0/.zattrs"
    if sample_attrs_key in df["key"].values:
        row_val = df.loc[df["key"] == sample_attrs_key, "value"].iloc[0]
        snippet = row_val.decode() if isinstance(row_val, (bytes, bytearray)) else row_val
        print(f"\n(D) {sample_attrs_key} .zattrs preview:")
        print(f"    {snippet[:200]}")

    print()
    if all_ok:
        print("Gate A0 CONFIRMS the per-level bug in the live production "
              "template:")
        print("  - 51 .zarray entries per pl var (one per N.0 segment, "
              "0.0..50.0), NOT 663 (51 × 13).")
        print("  - Path segments are sequential indices, NOT the 13 "
              "isobaric levels {50,100,...,1000}.")
        print("  - .zattrs.level reports stale values (e.g. 0.0) "
              "inconsistent with isobaric metadata.")
        print("Rebuild against the fixed aggregator is required (per "
              "2026-05-29-49r1-perlevel-reprocess-plan.md §4-§5). The "
              "new template will use a DIFFERENT path shape "
              "(`{var}/{stepType}/{typeOfLevel}/{level}/...`) -- breakage "
              "of pl-key consumers is intentional, per the synthesis MD §4.")
        return 0
    print("Gate A0 — unexpected structure; investigate before assuming "
          "rebuild path is correct.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
