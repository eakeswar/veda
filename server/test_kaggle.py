#!/usr/bin/env python3
"""
Step-by-step Kaggle test from your laptop.

Usage (from repo root):
  # 1) Default — everything local on laptop
  python server/test_kaggle.py

  # 2) After Kaggle notebook is running, add to server/.env:
  #    KAGGLE_ENABLED=true
  #    KAGGLE_API_BASE_URL=https://xxxx.trycloudflare.com
  #    KAGGLE_API_SECRET=<strong-random-secret>
  #    LLM_PROVIDER=kaggle
  #    TRANSLATE_PROVIDER=local
  #    UPSCALE_PROVIDER=local
  #  Then:
  python server/test_kaggle.py --compare

  # 3) Ping Kaggle GPU server only (no local server needed):
  python server/test_kaggle.py --kaggle-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=SERVER_DIR / ".env")
except ImportError:
    pass

SAMPLE_TEXT = (
    "Photosynthesis is the process plants use to convert sunlight into chemical energy. "
    "Chlorophyll in leaf cells absorbs light, driving reactions that produce glucose and "
    "release oxygen. This process supports nearly all life on Earth by forming the base "
    "of most food chains and replenishing atmospheric oxygen."
)

LOCAL_ANALYZE_URL = os.environ.get("VITE_ANALYZE_API_URL", "http://127.0.0.1:8765/analyze")
LOCAL_HEALTH_URL = LOCAL_ANALYZE_URL.replace("/analyze", "/health")


def _print_env():
    print("Current provider settings (from server/.env):")
    for key in (
        "KAGGLE_ENABLED",
        "KAGGLE_API_BASE_URL",
        "KAGGLE_API_SECRET",
        "LLM_PROVIDER",
        "TRANSLATE_PROVIDER",
        "UPSCALE_PROVIDER",
        "LOCAL_LLM_SIZE",
    ):
        val = os.environ.get(key, "")
        if key == "KAGGLE_API_SECRET" and val:
            val = val[:4] + "…"
        print(f"  {key}={val or '(not set)'}")
    print()


def test_local_health() -> bool:
    import requests

    print(f"Local server: GET {LOCAL_HEALTH_URL}")
    try:
        r = requests.get(LOCAL_HEALTH_URL, timeout=10)
        data = r.json()
        print(f"  ok={data.get('ok')} providers={data.get('providers')}")
        ks = data.get("kaggle") or {}
        print(f"  kaggle.enabled={ks.get('enabled')} configured={ks.get('configured')} reachable={ks.get('reachable')} gpus={ks.get('gpus')}")
        return r.ok
    except Exception as exc:
        print(f"  FAILED: {exc}")
        print("  Start the laptop server: python server/tts_server.py")
        return False


def timed_analyze_local() -> float | None:
    import requests

    print("\nLocal /analyze …")
    t0 = time.perf_counter()
    try:
        r = requests.post(
            LOCAL_ANALYZE_URL,
            json={"text": SAMPLE_TEXT, "is_digest": False, "language": "en-US"},
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        if not r.ok:
            print(f"  FAILED ({r.status_code}): {r.text[:300]}")
            return None
        data = r.json()
        print(f"  OK in {elapsed:.1f}s — title: {data.get('title', '?')!r}")
        return elapsed
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None


def timed_analyze_kaggle_direct() -> float | None:
    from kaggle_client import kaggle_analyze, kaggle_status, should_use_kaggle

    ks = kaggle_status()
    print("\nKaggle GPU direct /analyze …")
    print(f"  status={ks}")
    if not should_use_kaggle(os.environ.get("LLM_PROVIDER", "local")):
        print("  Skipped — set KAGGLE_ENABLED=true and LLM_PROVIDER=kaggle in server/.env")
        return None

    t0 = time.perf_counter()
    try:
        data = kaggle_analyze(SAMPLE_TEXT, is_digest=False)
        elapsed = time.perf_counter() - t0
        print(f"  OK in {elapsed:.1f}s — title: {data.get('title', '?')!r}")
        return elapsed
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Veda Kaggle GPU integration from laptop")
    parser.add_argument("--compare", action="store_true", help="Run local then Kaggle and show timings")
    parser.add_argument("--kaggle-only", action="store_true", help="Only test direct Kaggle GPU server")
    args = parser.parse_args()

    _print_env()

    if args.kaggle_only:
        timed_analyze_kaggle_direct()
        return

    if not test_local_health():
        sys.exit(1)

    if args.compare:
        local_s = timed_analyze_local()
        kaggle_s = timed_analyze_kaggle_direct()
        print("\n── Summary ──")
        if local_s is not None:
            print(f"  Local laptop:  {local_s:.1f}s")
        if kaggle_s is not None:
            print(f"  Kaggle GPU:    {kaggle_s:.1f}s")
        if local_s and kaggle_s:
            ratio = local_s / kaggle_s
            print(f"  Speedup:       {ratio:.1f}× faster on Kaggle" if ratio > 1 else f"  Local was {1/ratio:.1f}× faster")
        return

    print("\nTip: run with --compare after Kaggle notebook is up to benchmark local vs GPU.")
    print("Tip: run with --kaggle-only to test the tunnel without the local server.")


if __name__ == "__main__":
    main()
