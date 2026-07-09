"""HTTP client for a Veda GPU server running on Kaggle (cloudflared tunnel)."""
from __future__ import annotations

import os

import requests

DEFAULT_TIMEOUT = int(os.environ.get("KAGGLE_API_TIMEOUT", "120"))


def kaggle_base_url() -> str | None:
    url = os.environ.get("KAGGLE_API_BASE_URL", "").strip().rstrip("/")
    return url or None


def kaggle_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    secret = os.environ.get("KAGGLE_API_SECRET", "").strip()
    if not secret:
        if kaggle_enabled():
            print("WARNING: KAGGLE_ENABLED=true but KAGGLE_API_SECRET is not set")
        return headers
    headers["Authorization"] = f"Bearer {secret}"
    return headers


def kaggle_enabled() -> bool:
    return os.environ.get("KAGGLE_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def kaggle_available() -> bool:
    return kaggle_enabled() and kaggle_base_url() is not None


def should_use_kaggle(provider: str) -> bool:
    """Return True when this provider name routes to the Kaggle GPU server."""
    mode = provider.strip().lower()
    if mode == "local":
        return False
    if mode == "kaggle":
        return kaggle_available()
    if mode == "auto":
        return kaggle_available()
    return False


def kaggle_status(timeout: int = 5) -> dict:
    base = kaggle_base_url()
    status = {
        "enabled": kaggle_enabled(),
        "base_url": base or None,
        "reachable": False,
        "gpus": None,
    }
    if not kaggle_available():
        return status
    try:
        resp = requests.get(f"{base}/health", headers=kaggle_headers(), timeout=timeout)
        if resp.ok:
            data = resp.json()
            status["reachable"] = data.get("ok") is True
            status["gpus"] = data.get("gpus")
    except Exception as exc:
        status["error"] = str(exc)
    return status


def kaggle_health(timeout: int = 10) -> bool:
    base = kaggle_base_url()
    if not base:
        return False
    try:
        resp = requests.get(f"{base}/health", headers=kaggle_headers(), timeout=timeout)
        return resp.ok and resp.json().get("ok") is True
    except Exception:
        return False


def kaggle_analyze(
    text: str,
    is_digest: bool = False,
    page_layout: str = "plain",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    resp = requests.post(
        f"{base}/analyze",
        headers=kaggle_headers(),
        json={"text": text, "is_digest": is_digest, "page_layout": page_layout},
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /analyze failed ({resp.status_code}): {detail}")
    return resp.json()


def kaggle_translate(texts: list[str], target_lang: str = "tel_Telu", timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    resp = requests.post(
        f"{base}/translate",
        headers=kaggle_headers(),
        json={"texts": texts, "target_lang": target_lang},
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /translate failed ({resp.status_code}): {detail}")
    data = resp.json()
    translated = data.get("translated")
    if not isinstance(translated, list):
        raise RuntimeError("Kaggle /translate returned invalid payload")
    return translated


def kaggle_upload_pdf(pdf_bytes: bytes, timeout: int = DEFAULT_TIMEOUT) -> dict:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    headers = kaggle_headers()
    headers["Content-Type"] = "application/octet-stream"
    resp = requests.post(f"{base}/upload_pdf", headers=headers, data=pdf_bytes, timeout=timeout)
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /upload_pdf failed ({resp.status_code}): {detail}")
    return resp.json()


def kaggle_page_layout(page: int, timeout: int = DEFAULT_TIMEOUT) -> dict:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    resp = requests.get(
        f"{base}/page_layout",
        headers=kaggle_headers(),
        params={"page": page},
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /page_layout failed ({resp.status_code}): {detail}")
    return resp.json()


def kaggle_generate_prompt(
    image_b64: str,
    page_text: str = "",
    timeout: int | None = None,
) -> dict:
    """Layer 1 trial — SmolVLM caption + Qwen SDXL prompt via Kaggle GPU."""
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    req_timeout = int(os.environ.get("KAGGLE_IMAGE_GEN_TIMEOUT", "180"))
    effective_timeout = timeout if timeout is not None else req_timeout

    resp = requests.post(
        f"{base}/generate_prompt",
        headers=kaggle_headers(),
        json={"image": image_b64, "page_text": page_text},
        timeout=effective_timeout,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /generate_prompt failed ({resp.status_code}): {detail}")
    data = resp.json()
    if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
        raise RuntimeError("Kaggle /generate_prompt returned invalid payload")
    return data


def resolve_provider(env_name: str, default: str = "local") -> str:
    return os.environ.get(env_name, default).strip().lower()
