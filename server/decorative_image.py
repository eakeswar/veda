"""Heuristic + people-aware routing: informational → Lanczos, subject photos → img2img."""
from __future__ import annotations

import os

OVERLAP_INFORMATIONAL = 8
OVERLAP_CHART = 5
OVERLAP_SCREENSHOT = 3
TINY_IMAGE_MAX_DIM = 400
IMAGE_GEN_MAX_DIM = int(os.environ.get("IMAGE_GEN_MAX_DIM", "1024"))


def is_tiny_image(width: int, height: int, max_dim: int = TINY_IMAGE_MAX_DIM) -> bool:
    """True when either dimension is below max_dim."""
    return width < max_dim or height < max_dim


def _image_dims(image_meta: dict) -> tuple[float, float, float, int, float]:
    overlap = int(image_meta.get("overlapWordCount") or 0)
    w = float(image_meta.get("w") or 0)
    h = float(image_meta.get("h") or 0)
    aspect = w / h if h > 0 else 1.0
    return w, h, aspect, overlap, 0.0  # area_ratio filled by caller


def is_informational_image(
    image_meta: dict,
    *,
    page_width: float,
    page_height: float,
) -> bool:
    """
    Charts, screenshots, text-heavy figures — always Lanczos, never img2img.
    Conservative: high overlap with PDF text → informational.
    """
    if not image_meta:
        return False

    w, h, aspect, overlap, _ = _image_dims(image_meta)
    if w < 1 or h < 1 or page_width < 1 or page_height < 1:
        return False

    area_ratio = (w * h) / (page_width * page_height)

    if overlap >= OVERLAP_INFORMATIONAL:
        return True

    if overlap >= OVERLAP_SCREENSHOT and 1.3 <= aspect <= 1.85 and w >= 400:
        return True

    if overlap >= OVERLAP_CHART and 0.85 <= aspect <= 1.15 and 0.10 <= area_ratio <= 0.40:
        return True

    return False


def is_photo_like_decorative(
    image_meta: dict,
    *,
    page_width: float,
    page_height: float,
    is_image_primary: bool = False,
) -> bool:
    """Inset photo / illustration with little overlapping text (animals, objects, scenes)."""
    if not image_meta:
        return False

    w, h, aspect, overlap, _ = _image_dims(image_meta)
    if w < 1 or h < 1 or page_width < 1 or page_height < 1:
        return False

    area_ratio = (w * h) / (page_width * page_height)

    if is_image_primary:
        return True

    if overlap < 3 and 0.05 <= area_ratio <= 0.45 and 0.4 <= aspect <= 2.5:
        return True

    return False


def is_decorative_image(
    image_meta: dict,
    *,
    page_width: float,
    page_height: float,
    page_word_count: int = 0,
    is_image_primary: bool = False,
) -> bool:
    """Backward-compatible: not informational and photo-like decorative."""
    _ = page_word_count
    if is_informational_image(image_meta, page_width=page_width, page_height=page_height):
        return False
    return is_photo_like_decorative(
        image_meta,
        page_width=page_width,
        page_height=page_height,
        is_image_primary=is_image_primary,
    )


def _within_img2img_size(width: int, height: int) -> bool:
    return max(width, height) <= IMAGE_GEN_MAX_DIM


def should_route_img2img(
    pil_img,
    image_meta: dict | None,
    *,
    width: int,
    height: int,
    page_width: float = 0,
    page_height: float = 0,
    is_image_primary: bool = False,
) -> tuple[bool, str]:
    """
    Phase 1 routing:
      • informational → Lanczos
      • people (YuNet/HOG) → img2img up to IMAGE_GEN_MAX_DIM
      • photo-like decorative + tiny → img2img
      • else → Lanczos
    Returns (use_img2img, reason_tag).
    """
    if not _within_img2img_size(width, height):
        return False, "too_large"

    meta = image_meta or {}
    if page_width and page_height and is_informational_image(
        meta, page_width=page_width, page_height=page_height
    ):
        return False, "informational"

    from upscale_service import detect_people_for_routing

    has_people, people_reason = detect_people_for_routing(pil_img)
    if has_people:
        return True, f"people_{people_reason}"

    if page_width and page_height and is_photo_like_decorative(
        meta,
        page_width=page_width,
        page_height=page_height,
        is_image_primary=is_image_primary,
    ) and is_tiny_image(width, height):
        return True, "photo_tiny"

    return False, "lanczos"
