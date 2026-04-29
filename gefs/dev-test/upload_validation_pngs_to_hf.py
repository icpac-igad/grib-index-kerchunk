#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""
Upload GIK-vs-Herbie validation PNGs (and companion stats) to HuggingFace.

Source (default): gefs/gik_vs_herbie_first5/
Target:           E4DRR/gik-gefs-par (dataset repo), under
                  validation_gik_vs_herbie_50days/

Mirrors the pattern of ecmwf/upload_parquets_to_hf.py but uploads PNGs
(plus validation_stats.json, scatter_all_dates.png, validation_log.txt)
as a single folder.

Auth:
    HF_TOKEN or HUGGING_FACE_HUB_TOKEN env var (write-scoped), or the
    token saved by `huggingface-cli login`. If a .env file sits next to
    this script, variables there are loaded first.

Usage:
    uv run upload_validation_pngs_to_hf.py                        # upload everything
    uv run upload_validation_pngs_to_hf.py --dry-run               # list without uploading
    uv run upload_validation_pngs_to_hf.py --src some/other/dir
    uv run upload_validation_pngs_to_hf.py --subfolder validation_other
    uv run upload_validation_pngs_to_hf.py --include-logs          # also upload .txt logs
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Defaults ────────────────────────────────────────────────────────
HF_REPO = "E4DRR/gik-gefs-par"
HF_SUBFOLDER = "validation_gik_vs_herbie_50days"
DEFAULT_SRC = SCRIPT_DIR / "gik_vs_herbie_first5"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dep)."""
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
    )


def collect_files(src: Path, include_logs: bool) -> list[Path]:
    patterns = ["*.png", "*.json"]
    if include_logs:
        patterns.append("*.txt")
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(src.glob(pat)))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload GIK-vs-Herbie validation PNGs to HuggingFace")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC),
                        help=f"Source folder (default: {DEFAULT_SRC})")
    parser.add_argument("--repo", type=str, default=HF_REPO,
                        help=f"HF dataset repo (default: {HF_REPO})")
    parser.add_argument("--subfolder", type=str, default=HF_SUBFOLDER,
                        help=f"Subfolder in repo (default: {HF_SUBFOLDER})")
    parser.add_argument("--commit-message", type=str,
                        default="Upload GIK vs Herbie validation PNGs (50 cost-test dates)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    parser.add_argument("--include-logs", action="store_true",
                        help="Also include *.txt log files")
    parser.add_argument("--create-repo", action="store_true",
                        help="Create the dataset repo if it does not exist")
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")
    token = resolve_hf_token()

    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"ERROR: source folder not found: {src}", file=sys.stderr)
        return 2

    files = collect_files(src, include_logs=args.include_logs)
    if not files:
        print(f"ERROR: no PNG/JSON files in {src}", file=sys.stderr)
        return 3

    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    print("=" * 70)
    print("Upload GIK vs Herbie validation PNGs → HuggingFace")
    print("=" * 70)
    print(f"Source:     {src}")
    print(f"HF repo:    {args.repo}")
    print(f"Subfolder:  {args.subfolder}")
    print(f"Files:      {len(files)} ({total_mb:.1f} MB)")
    print(f"Token:      {'yes (env/login)' if token else 'NONE — will rely on cached login'}")
    print("=" * 70)

    for f in files:
        size_kb = f.stat().st_size / 1024
        rel = f.name
        hf_path = f"{args.subfolder}/{rel}"
        print(f"  {rel:<40}  {size_kb:>8.1f} KB  →  {args.repo}/{hf_path}")

    if args.dry_run:
        print("\n[DRY RUN] nothing uploaded")
        return 0

    api = HfApi(token=token)

    if args.create_repo:
        try:
            create_repo(args.repo, repo_type="dataset",
                        exist_ok=True, token=token)
            print(f"\n  repo ready: {args.repo}")
        except Exception as e:
            print(f"ERROR creating repo: {e}", file=sys.stderr)
            return 4

    try:
        api.upload_folder(
            folder_path=str(src),
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo=args.subfolder,
            allow_patterns=[f.name for f in files],
            commit_message=args.commit_message,
        )
    except Exception as e:
        print(f"\nUPLOAD FAILED: {e}", file=sys.stderr)
        print("\nCommon causes:")
        print("  - HF_TOKEN missing or read-only (need 'write' scope)")
        print("  - Dataset repo does not exist yet (add --create-repo)")
        print("  - Repo lives under a different namespace; check --repo")
        return 5

    print(f"\nUploaded {len(files)} files ({total_mb:.1f} MB) to "
          f"https://huggingface.co/datasets/{args.repo}/tree/main/{args.subfolder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
