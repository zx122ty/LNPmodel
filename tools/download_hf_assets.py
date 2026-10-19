#!/usr/bin/env python3
"""Restore the LipidFM data/model bundle from Hugging Face.

The expected Hugging Face repository mirrors this GitHub repository's directory
layout. For example, ``benchmark/input_data/dataset.xlsx`` in the Hub repository
is downloaded to ``<project>/benchmark/input_data/dataset.xlsx``.

Example:
    python tools/download_hf_assets.py --repo-id your-name/LipidFM-assets
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LipidFM datasets and model checkpoints from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository id, e.g. your-name/LipidFM-assets.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Hugging Face repository type (default: model).",
    )
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Destination directory (default: this project's root).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        metavar="GLOB",
        help="Only download matching repository paths. Default: download everything.",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token for a private repository. HF_TOKEN is also supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with "
            "`pip install huggingface-hub`."
        ) from exc

    local_dir = args.local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=local_dir,
        allow_patterns=args.include,
        token=args.token,
    )
    print(f"Assets downloaded to {local_dir}")
    print("See ASSETS.md for the expected layout and required/optional files.")


if __name__ == "__main__":
    main()
