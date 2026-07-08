"""Qwen GGUF model selection — laptop vs Kaggle GPU use different sizes."""
from __future__ import annotations

import os

# size key -> (filename, HuggingFace download URL)
QWEN_GGUF: dict[str, tuple[str, str]] = {
    "0.5B": (
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ),
    "1.5B": (
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    "3B": (
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
    "7B": (
        "qwen2.5-7b-instruct-q3_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf",
    ),
    "14B": (
        "qwen2.5-14b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf",
    ),
}

DEFAULT_LOCAL_LLM_SIZE = "3B"
DEFAULT_KAGGLE_LLM_SIZE = "7B"


def _normalize_size(size: str | None, env_name: str, default: str) -> str:
    raw = (size or os.environ.get(env_name, default)).strip().upper()
    if raw in QWEN_GGUF:
        return raw
    return default


def get_local_qwen_gguf_info(size: str | None = None) -> tuple[str, str, str]:
    """Laptop CPU fallback — default 3B."""
    key = _normalize_size(size, "LOCAL_LLM_SIZE", DEFAULT_LOCAL_LLM_SIZE)
    filename, url = QWEN_GGUF[key]
    return key, filename, url


def get_kaggle_qwen_gguf_info(size: str | None = None) -> tuple[str, str, str]:
    """Kaggle GPU — default 7B."""
    key = _normalize_size(size, "KAGGLE_LLM_SIZE", DEFAULT_KAGGLE_LLM_SIZE)
    filename, url = QWEN_GGUF[key]
    return key, filename, url


def get_qwen_gguf_info(size: str | None = None) -> tuple[str, str, str]:
    """Backward-compatible alias for laptop local model."""
    return get_local_qwen_gguf_info(size)
