#!/usr/bin/env python3
"""Download Qwen 2.5 7B GGUF into server/models/ for Kaggle project-models."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from qwen_config import get_kaggle_qwen_gguf_info  # noqa: E402

def main() -> None:
    _size, filename, url = get_kaggle_qwen_gguf_info("7B")
    dest = SERVER_DIR / "models" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and dest.stat().st_size > 1_000_000_000:
        print(f"Already present: {dest} ({dest.stat().st_size // (1024**2)} MB)")
        return

    print(f"Downloading Qwen 2.5 7B (~3.8 GB, q3_k_m) …")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        mb = done // (1024 * 1024)
        total_mb = total_size // (1024 * 1024)
        print(f"\r  {pct:3d}% ({mb}/{total_mb} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\nDone: {dest}")


if __name__ == "__main__":
    main()
