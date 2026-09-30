#!/usr/bin/env python3
"""Download laptop models that cannot live in git (GitHub 100 MB file limit).

Qwen GGUF → server/models/
NLLB-200 1.3B → HuggingFace cache (used by local Telugu fallback)

Usage (from repo root or server/):
  python server/download_local_models.py
  python server/download_local_models.py --nllb
  python server/download_local_models.py --size 3B --nllb
  python server/download_local_models.py --7b
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))

from qwen_config import (  # noqa: E402
    get_kaggle_qwen_gguf_info,
    get_local_qwen_gguf_info,
)

NLLB_MODEL_NAME = "facebook/nllb-200-1.3B"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(total_size, block_num * block_size)
    pct = done * 100 // total_size
    mb = done // (1024 * 1024)
    total_mb = total_size // (1024 * 1024)
    print(f"\r  {pct:3d}% ({mb}/{total_mb} MB)", end="", flush=True)


def download_gguf(size_label: str, filename: str, url: str) -> Path:
    dest = SERVER_DIR / "models" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 50_000_000:
        print(f"Already present: {dest.name} ({dest.stat().st_size // (1024 * 1024)} MB)")
        return dest
    print(f"Downloading Qwen 2.5 {size_label} GGUF …")
    print(f"  {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\nDone: {dest} ({dest.stat().st_size // (1024 * 1024)} MB)")
    return dest


def download_nllb() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is missing. Install vendor deps first:\n"
            "  python -m pip install --target server/vendor -r server/requirements.txt"
        ) from exc
    print(f"Downloading {NLLB_MODEL_NAME} into the HuggingFace cache (~2.5 GB) …")
    path = snapshot_download(NLLB_MODEL_NAME)
    print(f"NLLB ready: {path}")


def main() -> None:
    vendor = SERVER_DIR / "vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))

    parser = argparse.ArgumentParser(description="Download Veda laptop / Kaggle GGUF models.")
    parser.add_argument(
        "--size",
        default=None,
        help="Local Qwen size: 0.5B, 1.5B, 3B (default: LOCAL_LLM_SIZE or 3B)",
    )
    parser.add_argument("--nllb", action="store_true", help="Also download NLLB-200 1.3B for Telugu fallback")
    parser.add_argument("--7b", dest="seven_b", action="store_true", help="Also download Kaggle 7B GGUF")
    args = parser.parse_args()

    if args.size:
        os.environ["LOCAL_LLM_SIZE"] = args.size.strip().upper()

    size, filename, url = get_local_qwen_gguf_info(args.size)
    download_gguf(size, filename, url)

    if args.seven_b:
        k_size, k_name, k_url = get_kaggle_qwen_gguf_info("7B")
        download_gguf(k_size, k_name, k_url)

    if args.nllb:
        download_nllb()

    print("Model download finished.")


if __name__ == "__main__":
    main()
