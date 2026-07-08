"""Face- and person-aware image upscaling for Veda PDF slide images (Lanczos only).

Routing:
  • Faces or full bodies detected → Lanczos (×4 if tiny, ×2 otherwise)
  • No people + tiny            → Lanczos ×4 + Unsharp Mask
  • No people + normal size     → Lanczos ×2 + Unsharp Mask
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

TINY_SIZE_THRESHOLD = int(os.environ.get("TINY_SIZE_THRESHOLD", os.environ.get("ESRGAN_SIZE_THRESHOLD", "400")))
FACE_SCORE_THRESHOLD = float(os.environ.get("FACE_SCORE_THRESHOLD", "0.45"))

FACE_MODEL_NAMES = (
    "face_detection_yunet_2026may.onnx",
    "face_detection_yunet_2023mar.onnx",
)

_detector = None
_detector_unavailable = False
_hog = None
_hog_unavailable = False


def _resolve_face_model_path() -> Path | None:
    names = list(FACE_MODEL_NAMES)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for name in names:
            for path in kaggle_input.rglob(name):
                if path.is_file():
                    return path
    for name in names:
        for base in (BASE_DIR / "data", Path(__file__).resolve().parent / "data"):
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def _get_face_detector():
    global _detector, _detector_unavailable
    if _detector_unavailable:
        return None
    if _detector is not None:
        return _detector
    model_path = _resolve_face_model_path()
    if model_path is None:
        print("YuNet face model not found — treating tiny images as people-safe (Lanczos only).")
        _detector_unavailable = True
        return None
    try:
        import cv2

        _detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), FACE_SCORE_THRESHOLD, 0.3, 5000,
        )
    except Exception as exc:
        print(f"YuNet face detector load failed ({exc}) — using conservative Lanczos routing.")
        _detector_unavailable = True
    return _detector


def _get_hog_detector():
    """OpenCV HOG person detector — catches full bodies YuNet misses on stylized/AI art."""
    global _hog, _hog_unavailable
    if _hog_unavailable:
        return None
    if _hog is not None:
        return _hog
    try:
        import cv2

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _hog = hog
    except Exception as exc:
        print(f"HOG person detector load failed ({exc}) — face-only routing.")
        _hog_unavailable = True
    return _hog


def image_has_faces(pil_img, score_threshold: float | None = None) -> bool:
    """Return True when YuNet detects at least one face."""
    threshold = FACE_SCORE_THRESHOLD if score_threshold is None else score_threshold
    detector = _get_face_detector()
    if detector is None:
        return True

    import numpy as np

    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(rgb)
    if faces is None or len(faces) == 0:
        return False
    return bool((faces[:, 2] >= threshold).any())


def image_has_person_hog(pil_img) -> bool:
    """Return True when OpenCV HOG detects at least one person-shaped region."""
    hog = _get_hog_detector()
    if hog is None:
        return False

    import cv2
    import numpy as np

    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    min_dim = min(h, w)
    if min_dim < 128:
        scale = 128 / min_dim
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    rects, _weights = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    return len(rects) > 0


def detect_people(pil_img) -> tuple[bool, str | None]:
    """Return (has_people, reason). reason is 'face', 'person', or None."""
    if image_has_faces(pil_img):
        return True, "face"
    if image_has_person_hog(pil_img):
        return True, "person"
    return False, None


def lanczos_upscale(pil_img, scale: int = 2):
    """Lanczos resize + mild Unsharp Mask — faithful, no AI detail invention."""
    from PIL import Image, ImageFilter

    scale = max(1, int(scale))
    new_w = pil_img.width * scale
    new_h = pil_img.height * scale
    resample = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
    upscaled = pil_img.resize((new_w, new_h), resample)
    return upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=3))


def decide_upscale_method(width: int, height: int) -> int:
    """Return Lanczos scale factor (2 or 4)."""
    is_tiny = width < TINY_SIZE_THRESHOLD or height < TINY_SIZE_THRESHOLD
    return 4 if is_tiny else 2


def upscale_image(pil_img):
    """
    Upscale a PIL RGB image using face/person-aware Lanczos routing.

    Returns (output_pil, method_name).
    """
    w, h = pil_img.size
    has_people, people_reason = detect_people(pil_img)
    lanczos_scale = decide_upscale_method(w, h)

    if has_people:
        print(f"Upscale {w}×{h}: people detected ({people_reason}) → Lanczos ×{lanczos_scale}")
    else:
        print(f"Upscale {w}×{h}: no people → Lanczos ×{lanczos_scale}")

    out = lanczos_upscale(pil_img, lanczos_scale)
    if has_people:
        return out, f"lanczos_people_x{lanczos_scale}"
    return out, f"lanczos_x{lanczos_scale}"
