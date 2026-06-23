#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""
Upload the GIK (per-level-keys fix) vs Herbie PRESSURE-LEVEL validation
artifacts (PNGs + stats JSON + README) to HuggingFace.

Source (default): gik_vs_herbie/pl_eval/out/   (the 12 maps + stats JSON)
                  + gik_vs_herbie/pl_eval/README.md
Target:           E4DRR/gik-ecmwf-par (dataset), under
                  validation_gik_vs_herbie_pl_correction/

Mirrors gefs/dev-test/upload_validation_pngs_to_hf.py.

Auth: reads a write-scoped token from (in order) HF_TOKEN,
HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_HUB_TOKEN, or the lowercase `hf_token`
key in ecmwf/.env (loaded automatically), or a cached `huggingface-cli login`.

Usage:
    uv run upload_pl_validation_to_hf.py --dry-run    # list, no upload
    uv run upload_pl_validation_to_hf.py              # upload
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SCRIPT_DIR = Path(__file__).resolve().parent

HF_REPO = "E4DRR/gik-ecmwf-par"
HF_SUBFOLDER = "validation_gik_vs_herbie_pl_correction"
DEFAULT_SRC = SCRIPT_DIR / "gik_vs_herbie" / "pl_eval" / "out"
README_SRC = SCRIPT_DIR / "gik_vs_herbie" / "pl_eval" / "README.md"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, val)


def resolve_hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("hf_token")          # lowercase key used in ecmwf/.env
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload pressure-level validation to HF")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help=f"PNG/JSON dir (default: {DEFAULT_SRC})")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--subfolder", default=HF_SUBFOLDER)
    ap.add_argument("--commit-message",
                    default="Add GIK per-level-keys fix vs Herbie pressure-level validation (u/v/gh, 2026-03-01)")
    ap.add_argument("--no-readme", action="store_true", help="do not include the README.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--create-repo", action="store_true")
    args = ap.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")
    token = resolve_hf_token()

    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"ERROR: source folder not found: {src}", file=sys.stderr)
        return 2

    files = sorted(src.glob("*.png")) + sorted(src.glob("*.json"))
    if README_SRC.exists() and not args.no_readme:
        files.append(README_SRC)
    if not files:
        print(f"ERROR: no PNG/JSON files in {src}", file=sys.stderr)
        return 3

    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print("=" * 70)
    print("Upload pressure-level validation → HuggingFace")
    print("=" * 70)
    print(f"Source:    {src}")
    print(f"HF repo:   {args.repo}")
    print(f"Subfolder: {args.subfolder}")
    print(f"Files:     {len(files)} ({total_mb:.1f} MB)")
    print(f"Token:     {'yes' if token else 'NONE — relying on cached login'}")
    print("=" * 70)
    for f in files:
        print(f"  {f.name:<46} {f.stat().st_size/1024:>8.1f} KB → {args.repo}/{args.subfolder}/{f.name}")

    if args.dry_run:
        print("\n[DRY RUN] nothing uploaded")
        return 0

    api = HfApi(token=token)
    if args.create_repo:
        create_repo(args.repo, repo_type="dataset", exist_ok=True, token=token)

    # README sits at the same dir level via a separate add; upload_folder handles
    # the out/ files, then upload the README explicitly if outside src.
    try:
        api.upload_folder(
            folder_path=str(src),
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo=args.subfolder,
            allow_patterns=["*.png", "*.json"],
            commit_message=args.commit_message,
        )
        if README_SRC.exists() and not args.no_readme:
            api.upload_file(
                path_or_fileobj=str(README_SRC),
                path_in_repo=f"{args.subfolder}/README.md",
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=args.commit_message,
            )
    except Exception as e:
        print(f"\nUPLOAD FAILED: {e}", file=sys.stderr)
        print("  - token needs 'write' scope; repo is E4DRR/gik-ecmwf-par")
        return 5

    print(f"\nUploaded {len(files)} files ({total_mb:.1f} MB) → "
          f"https://huggingface.co/datasets/{args.repo}/tree/main/{args.subfolder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
