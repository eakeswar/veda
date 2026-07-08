"""Shared request limits and validation for Veda servers."""
from __future__ import annotations

import os

MAX_PDF_BYTES = int(os.environ.get("VEDA_MAX_PDF_BYTES", str(50 * 1024 * 1024)))
MAX_TTS_TEXT_CHARS = int(os.environ.get("VEDA_MAX_TTS_TEXT_CHARS", "10000"))
MAX_ANALYZE_TEXT_CHARS = int(os.environ.get("VEDA_MAX_ANALYZE_TEXT_CHARS", "50000"))
MAX_IMAGE_B64_CHARS = int(os.environ.get("VEDA_MAX_IMAGE_B64_CHARS", str(20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("VEDA_MAX_IMAGE_PIXELS", str(4096 * 4096)))

PDF_MAGIC = b"%PDF-"
WEAK_KAGGLE_SECRET = "veda-kaggle-dev"


def cors_origins() -> list[str]:
    raw = os.environ.get(
        "VEDA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def require_kaggle_secret(secret: str | None) -> str:
    value = (secret or "").strip()
    if not value:
        raise SystemExit(
            "KAGGLE_API_SECRET must be set to a strong random value before starting the GPU server."
        )
    if value == WEAK_KAGGLE_SECRET:
        raise SystemExit(
            "KAGGLE_API_SECRET cannot be the default 'veda-kaggle-dev'. "
            "Generate a new secret (e.g. openssl rand -hex 32)."
        )
    if len(value) < 16:
        raise SystemExit("KAGGLE_API_SECRET must be at least 16 characters.")
    return value


def validate_pdf_bytes(data: bytes) -> None:
    if len(data) > MAX_PDF_BYTES:
        max_mb = MAX_PDF_BYTES // (1024 * 1024)
        raise ValueError(f"PDF exceeds maximum size ({max_mb} MB)")
    if not data.startswith(PDF_MAGIC):
        raise ValueError("File is not a valid PDF (missing %PDF- header)")


def validate_text_length(text: str, max_chars: int, field: str = "text") -> None:
    if len(text) > max_chars:
        raise ValueError(f"{field} exceeds maximum length ({max_chars} characters)")


def validate_b64_payload(b64: str, max_chars: int = MAX_IMAGE_B64_CHARS) -> None:
    if len(b64) > max_chars:
        raise ValueError("Image payload exceeds maximum allowed size")


def redact_kaggle_status(status: dict) -> dict:
    """Return a public-safe view of Kaggle connectivity (no tunnel URL)."""
    return {
        "enabled": status.get("enabled", False),
        "configured": bool(status.get("base_url")),
        "reachable": status.get("reachable", False),
        "gpus": status.get("gpus"),
        "error": status.get("error"),
    }
