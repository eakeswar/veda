"""Face-aware image upscaling for Veda PDF slide images.

Routing:
  • Faces detected  → Lanczos (×4 if tiny, ×2 otherwise) — no AI hallucination on people
  • No faces + tiny → Real-ESRGAN ×4 when a model is available
  • No faces + normal size → Lanczos ×2 + Unsharp Mask
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

ESRGAN_SIZE_THRESHOLD = int(os.environ.get("ESRGAN_SIZE_THRESHOLD", "400"))
MAX_ESRGAN_INPUT_PX = 600
ESRGAN_TILE_SIZE = 256
ESRGAN_TILE_PAD = 10

FACE_MODEL_NAMES = (
    "face_detection_yunet_2026may.onnx",
    "face_detection_yunet_2023mar.onnx",
)

_detector = None
_detector_unavailable = False


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
        print("YuNet face model not found — treating tiny images as face-safe (Lanczos only).")
        _detector_unavailable = True
        return None
    try:
        import cv2

        _detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.55, 0.3, 5000)
    except Exception as exc:
        print(f"YuNet face detector load failed ({exc}) — using conservative Lanczos routing.")
        _detector_unavailable = True
    return _detector


def image_has_faces(pil_img, score_threshold: float = 0.55) -> bool:
    """Return True when YuNet detects at least one face."""
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
    return bool((faces[:, 2] >= score_threshold).any())


def lanczos_upscale(pil_img, scale: int = 2):
    """Lanczos resize + mild Unsharp Mask — faithful, no AI detail invention."""
    from PIL import Image, ImageFilter

    scale = max(1, int(scale))
    new_w = pil_img.width * scale
    new_h = pil_img.height * scale
    resample = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
    upscaled = pil_img.resize((new_w, new_h), resample)
    return upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=3))


def esrgan_upscale(img_np, model, device=None):
    """Run Real-ESRGAN 4× on a uint8 HWC RGB numpy array."""
    import numpy as np
    import torch

    h, w = img_np.shape[:2]
    img_t = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        img_t = img_t.to(device)
        model = model.to(device)

    tile = ESRGAN_TILE_SIZE
    pad = ESRGAN_TILE_PAD
    scale = 4
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, 3, out_h, out_w, dtype=torch.float32, device=img_t.device)

    tiles_x = max(1, (w + tile - 1) // tile)
    tiles_y = max(1, (h + tile - 1) // tile)

    with torch.no_grad():
        for iy in range(tiles_y):
            for ix in range(tiles_x):
                x0 = ix * tile
                y0 = iy * tile
                x1 = min(x0 + tile, w)
                y1 = min(y0 + tile, h)

                px0 = max(x0 - pad, 0)
                py0 = max(y0 - pad, 0)
                px1 = min(x1 + pad, w)
                py1 = min(y1 + pad, h)

                tile_in = img_t[:, :, py0:py1, px0:px1]
                tile_out = model(tile_in)

                ox0 = (x0 - px0) * scale
                oy0 = (y0 - py0) * scale
                ox1 = ox0 + (x1 - x0) * scale
                oy1 = oy0 + (y1 - y0) * scale

                output[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = tile_out[:, :, oy0:oy1, ox0:ox1]

    out_np = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (out_np * 255).astype(np.uint8)


def decide_upscale_method(width: int, height: int, has_faces: bool, esrgan_available: bool) -> tuple[str, int]:
    """Return (method, lanczos_scale). method is 'lanczos' or 'esrgan'."""
    is_tiny = width < ESRGAN_SIZE_THRESHOLD or height < ESRGAN_SIZE_THRESHOLD
    if has_faces:
        return "lanczos", (4 if is_tiny else 2)
    if is_tiny and esrgan_available:
        return "esrgan", 4
    if is_tiny:
        return "lanczos", 4
    return "lanczos", 2


def upscale_image(pil_img, *, esrgan_model=None, device=None):
    """
    Upscale a PIL RGB image using face-aware routing.

    Returns (output_pil, method_name).
    """
    import numpy as np
    from PIL import Image

    w, h = pil_img.size
    has_faces = image_has_faces(pil_img)
    esrgan_available = esrgan_model is not None
    method, lanczos_scale = decide_upscale_method(w, h, has_faces, esrgan_available)

    if method == "esrgan":
        work_img = pil_img
        if w > MAX_ESRGAN_INPUT_PX or h > MAX_ESRGAN_INPUT_PX:
            work_img = pil_img.copy()
            work_img.thumbnail((MAX_ESRGAN_INPUT_PX, MAX_ESRGAN_INPUT_PX), Image.LANCZOS)
        upscaled_np = esrgan_upscale(np.array(work_img), esrgan_model, device=device)
        return Image.fromarray(upscaled_np), "esrgan"

    out = lanczos_upscale(pil_img, lanczos_scale)
    if has_faces:
        return out, f"lanczos_faces_x{lanczos_scale}"
    return out, f"lanczos_x{lanczos_scale}"
