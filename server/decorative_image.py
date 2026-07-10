"""Heuristic decorative vs informational image classification (no ML model)."""
from __future__ import annotations

OVERLAP_INFORMATIONAL = 8
OVERLAP_CHART = 5
OVERLAP_SCREENSHOT = 3


def is_decorative_image(
    image_meta: dict,
    *,
    page_width: float,
    page_height: float,
    page_word_count: int = 0,
    is_image_primary: bool = False,
) -> bool:
    """
    Return True when an image is decorative (cover, illustration, generic photo).
    Conservative default: False → Lanczos when uncertain.
    """
    if not image_meta:
        return False

    overlap = int(image_meta.get("overlapWordCount") or 0)
    w = float(image_meta.get("w") or 0)
    h = float(image_meta.get("h") or 0)
    if w < 1 or h < 1 or page_width < 1 or page_height < 1:
        return False

    aspect = w / h

    # Informational: many PDF text words overlap the image bbox
    if overlap >= OVERLAP_INFORMATIONAL:
        return False

    # Screenshot-like: wide aspect + moderate overlap
    if overlap >= OVERLAP_SCREENSHOT and 1.3 <= aspect <= 1.85 and w >= 400:
        return False

    # Chart/diagram-like: near-square + overlap in mid page area
    area_ratio = (w * h) / (page_width * page_height)
    if overlap >= OVERLAP_CHART and 0.85 <= aspect <= 1.15 and 0.10 <= area_ratio <= 0.40:
        return False

    # Decorative: image-primary pages (covers, photo spreads)
    if is_image_primary:
        return True

    # Decorative: inset photo/illustration with little overlapping text
    if overlap < 3 and 0.05 <= area_ratio <= 0.45 and 0.4 <= aspect <= 2.5:
        return True

    _ = page_word_count  # reserved for future heuristics
    return False
