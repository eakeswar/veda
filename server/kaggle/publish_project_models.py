#!/usr/bin/env python3
"""Build and publish the Veda Kaggle dataset via the official Kaggle API.

Usage (from repo root):
  python server/kaggle/publish_project_models.py

Requires in server/.env (either format):
  KAGGLE_API_TOKEN=KGAT_...                    # new token (kaggle.com/settings/api)
  KAGGLE_USERNAME=eakeswar                     # still needed for dataset id
  KAGGLE_DATASET_SLUG=project-models           # optional

  — or legacy —
  KAGGLE_USERNAME=...
  KAGGLE_KEY=...
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
SERVER_DIR = KAGGLE_DIR.parent
REPO_ROOT = SERVER_DIR.parent
BUNDLE = KAGGLE_DIR / "veda-models-bundle"
PREPARE_BAT = KAGGLE_DIR / "prepare_kaggle_dataset.bat"


def load_env() -> None:
    env_path = SERVER_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def ensure_kaggle_cli() -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing kaggle CLI (>=1.8 for KGAT tokens) …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.8"])


def run_prepare() -> None:
    print("Building bundle (prepare_kaggle_dataset.bat) …")
    subprocess.check_call(["cmd", "/c", str(PREPARE_BAT)], cwd=KAGGLE_DIR)


def write_metadata(username: str, dataset_slug: str) -> Path:
    metadata = {
        "title": dataset_slug,
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    path = BUNDLE / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path} -> {metadata['id']}")
    return path


def kaggle_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        cwd=KAGGLE_DIR,
        env=env,
        text=True,
        capture_output=True,
    )


def publish(username: str, dataset_slug: str) -> None:
    write_metadata(username, dataset_slug)
    msg = "Updated from Veda (API publish)"

    print(f"Publishing {username}/{dataset_slug} …")
    version = kaggle_cmd("datasets", "version", "-p", str(BUNDLE), "-m", msg)
    if version.returncode == 0:
        print(version.stdout.strip() or "New dataset version published.")
        return

    print("Version failed (first publish?). Trying create …")
    if version.stderr:
        print(version.stderr.strip())

    create = kaggle_cmd("datasets", "create", "-p", str(BUNDLE))
    if create.returncode != 0:
        print(create.stderr.strip() or create.stdout.strip())
        sys.exit(1)
    print(create.stdout.strip() or "Dataset created on Kaggle.")


def main() -> None:
    load_env()
    ensure_kaggle_cli()

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    api_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    api_key = os.environ.get("KAGGLE_KEY", "").strip()
    dataset_slug = os.environ.get("KAGGLE_DATASET_SLUG", "project-models").strip()

    if not api_token and not (username and api_key):
        print("Missing Kaggle API credentials.")
        print()
        print("Add to server/.env (new token):")
        print("  KAGGLE_USERNAME=your_kaggle_username")
        print("  KAGGLE_API_TOKEN=KGAT_...")
        print("  KAGGLE_DATASET_SLUG=project-models")
        print()
        print("Or legacy: KAGGLE_USERNAME + KAGGLE_KEY")
        print("Create token: https://www.kaggle.com/settings/api")
        sys.exit(1)

    if not username:
        print("KAGGLE_USERNAME is required (for dataset id username/project-models).")
        sys.exit(1)

    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if api_key:
        os.environ["KAGGLE_KEY"] = api_key

    run_prepare()
    publish(username, dataset_slug)
    print()
    print("Done. In your Kaggle notebook: Add Data ->", dataset_slug, "-> use latest version.")


if __name__ == "__main__":
    main()
