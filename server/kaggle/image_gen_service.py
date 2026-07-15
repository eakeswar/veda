"""Kaggle-only image generation: Layer 1 SmolVLM+Qwen prompt, Layer 2 SSD-1B img2img (SDXL-arch)."""
from __future__ import annotations

import base64
import os
import re
import threading
from io import BytesIO
from typing import Any

SMOLVLM_MODEL = os.environ.get("SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
SMOLVLM_DEVICE = os.environ.get("SMOLVLM_DEVICE", "cuda:0")
# SDXL-architecture img2img; SSD-1B is ~50% smaller/faster than base SDXL with near-identical quality.
SDXL_MODEL = os.environ.get("SDXL_MODEL", "segmind/SSD-1B")
SDXL_DEVICE = os.environ.get("SDXL_DEVICE", "cuda:1")
SDXL_STEPS = int(os.environ.get("SDXL_STEPS", "20"))
SDXL_GUIDANCE = float(os.environ.get("SDXL_GUIDANCE", "6.0"))
SDXL_STRENGTH = float(os.environ.get("SDXL_STRENGTH", "0.35"))
SDXL_MIN_SIDE = int(os.environ.get("SDXL_MIN_SIDE", "512"))
SDXL_MAX_SIDE = int(os.environ.get("SDXL_MAX_SIDE", "1024"))
SDXL_VRAM_1024_MB = int(os.environ.get("SDXL_VRAM_1024_MB", "12000"))
SDXL_VRAM_768_MB = int(os.environ.get("SDXL_VRAM_768_MB", "9000"))

CAPTION_MAX_TOKENS = int(os.environ.get("SMOLVLM_CAPTION_MAX_TOKENS", "128"))
PROMPT_MAX_TOKENS = int(os.environ.get("IMAGE_PROMPT_MAX_TOKENS", "220"))
PAGE_TEXT_MAX_CHARS = int(os.environ.get("IMAGE_GEN_PAGE_TEXT_MAX_CHARS", "1500"))

NEGATIVE_PROMPT = (
    "text, watermark, logo, words, letters, caption, signature, writing, label, "
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, different subject, "
    "changed composition, new scene"
)

CAPTION_USER_TEXT = (
    "Describe this educational magazine illustration in 2-4 sentences. "
    "Focus on the main subject, setting, colors, and mood. "
    "Do not invent readable text, watermarks, or logos."
)

PROMPT_SYSTEM = (
    "You write concise Stable Diffusion XL img2img prompts for enhancing educational "
    "magazine illustrations.\n"
    "Given an image caption and page context, output ONE paragraph prompt (40-80 words) "
    "that preserves the original subjects, layout, colors, and composition while "
    "improving clarity, sharpness, and detail.\n"
    "Use phrases like: same scene, same subjects, faithful enhancement, preserve layout.\n"
    "Do NOT invent new subjects, change the scene, or add readable text, watermarks, "
    "logos, captions, or words.\n"
    "Output ONLY the prompt text — no explanation, no JSON, no markdown."
)

_smolvlm_processor = None
_smolvlm_model = None
_smolvlm_lock = threading.Lock()

_sdxl_pipe = None
_sdxl_lock = threading.Lock()


def _img2img_method_tag() -> str:
    """Response method suffix derived from SDXL_MODEL env (ssd1b, vega, or sdxl)."""
    model = SDXL_MODEL.lower()
    if "ssd-1b" in model or "ssd_1b" in model:
        return "ssd1b_img2img"
    if "vega" in model:
        return "vega_img2img"
    return "sdxl_img2img"


def _img2img_download_hint() -> str:
    model = SDXL_MODEL.lower()
    if "ssd-1b" in model or "ssd_1b" in model:
        return "~3 GB"
    if "vega" in model:
        return "~2 GB"
    return "~6 GB"


def get_vram_stats() -> dict[str, Any]:
    """Per-GPU VRAM snapshot via torch.cuda.mem_get_info()."""
    try:
        import torch
    except ImportError:
        return {"available": False, "devices": []}

    if not torch.cuda.is_available():
        return {"available": False, "devices": []}

    devices = []
    for idx in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(idx)
        devices.append({
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "total_mb": round(total_b / (1024 * 1024)),
            "free_mb": round(free_b / (1024 * 1024)),
            "used_mb": round((total_b - free_b) / (1024 * 1024)),
        })
    return {"available": True, "devices": devices}


def log_vram_snapshot(label: str) -> None:
    stats = get_vram_stats()
    if not stats.get("available"):
        print(f"VRAM [{label}]: CUDA not available")
        return
    parts = [
        f"gpu{d['index']} free={d['free_mb']}/{d['total_mb']} MB"
        for d in stats.get("devices", [])
    ]
    print(f"VRAM [{label}]: " + ", ".join(parts))


def _hf_hub_kwargs() -> dict:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return {"token": token} if token else {}


def _load_smolvlm():
    global _smolvlm_processor, _smolvlm_model
    if _smolvlm_model is not None:
        return _smolvlm_processor, _smolvlm_model

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    hf_kw = _hf_hub_kwargs()
    print(f"Veda image-gen: loading SmolVLM ({SMOLVLM_MODEL}) on {SMOLVLM_DEVICE} …")
    log_vram_snapshot("before_smolvlm_load")

    print("Veda image-gen: downloading SmolVLM processor …")
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL, **hf_kw)
    dtype = torch.float16 if SMOLVLM_DEVICE.startswith("cuda") else torch.float32
    print("Veda image-gen: downloading SmolVLM weights …")
    model = AutoModelForVision2Seq.from_pretrained(
        SMOLVLM_MODEL,
        torch_dtype=dtype,
        **hf_kw,
    )
    print(f"Veda image-gen: moving SmolVLM to {SMOLVLM_DEVICE} …")
    model = model.to(SMOLVLM_DEVICE)
    model.eval()

    _smolvlm_processor = processor
    _smolvlm_model = model
    log_vram_snapshot("after_smolvlm_load")
    print("Veda image-gen: SmolVLM ready.")
    return processor, model


def get_smolvlm():
    if _smolvlm_model is not None:
        return _smolvlm_processor, _smolvlm_model
    with _smolvlm_lock:
        return _load_smolvlm()


def caption_image(pil_img) -> str:
    """Layer 1a — SmolVLM caption for a PIL RGB image."""
    import torch

    processor, model = get_smolvlm()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_USER_TEXT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[pil_img.convert("RGB")], return_tensors="pt")
    inputs = {k: v.to(SMOLVLM_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]
    with _smolvlm_lock:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=CAPTION_MAX_TOKENS,
                do_sample=False,
            )

    new_ids = generated_ids[0, input_len:]
    caption = processor.decode(new_ids, skip_special_tokens=True).strip()
    caption = re.sub(r"\s+", " ", caption)
    if not caption:
        raise RuntimeError("SmolVLM returned an empty caption")
    return caption


def build_prompt_refinement_messages(caption: str, page_text: str) -> list[dict]:
    """Layer 1b — chat messages for Qwen prompt refinement."""
    context = (page_text or "").strip()
    if len(context) > PAGE_TEXT_MAX_CHARS:
        context = context[:PAGE_TEXT_MAX_CHARS].rstrip() + "…"

    user_body = f"Image caption:\n{caption}\n\nPage context:\n{context or '(no page text)'}"
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user_body},
    ]


def refine_generation_prompt(caption: str, page_text: str, llm, llm_lock) -> str:
    """Layer 1b — use the shared Qwen 7B instance to refine an SDXL prompt."""
    messages = build_prompt_refinement_messages(caption, page_text)
    with llm_lock:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=PROMPT_MAX_TOKENS,
            temperature=0.4,
            repeat_penalty=1.05,
        )
    prompt = response["choices"][0]["message"]["content"].strip()
    prompt = re.sub(r"^[\"']|[\"']$", "", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    if not prompt:
        raise RuntimeError("Qwen returned an empty generation prompt")
    return prompt


def run_layer1(pil_img, page_text: str, llm, llm_lock) -> dict[str, str]:
    """
    Full Layer 1 pipeline: SmolVLM caption → Qwen SDXL prompt.
    SmolVLM and Qwen share cuda:0; calls are serialized via llm_lock for Qwen
    and _smolvlm_lock for VLM load/inference.
    """
    log_vram_snapshot("layer1_start")
    caption = caption_image(pil_img)
    print(f"Veda image-gen: caption ({len(caption)} chars): {caption[:120]}…")

    prompt = refine_generation_prompt(caption, page_text, llm, llm_lock)
    print(f"Veda image-gen: SDXL prompt ({len(prompt)} chars): {prompt[:120]}…")
    log_vram_snapshot("layer1_done")

    return {
        "caption": caption,
        "prompt": prompt,
        "method": "layer1_smolvlm_qwen",
    }


def _free_vram_mb(device: str) -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    idx = int(device.split(":")[1]) if ":" in device else 0
    free_b, _total_b = torch.cuda.mem_get_info(idx)
    return int(free_b / (1024 * 1024))


def _max_output_side(device: str = SDXL_DEVICE) -> int:
    """Pick SDXL max side from free VRAM on the generation GPU (NLLB shares cuda:1)."""
    free_mb = _free_vram_mb(device)
    if free_mb is None:
        return min(SDXL_MAX_SIDE, 768)
    if free_mb >= SDXL_VRAM_1024_MB:
        return SDXL_MAX_SIDE
    if free_mb >= SDXL_VRAM_768_MB:
        return 768
    return SDXL_MIN_SIDE


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def compute_output_size(src_w: int, src_h: int, max_side: int | None = None) -> tuple[int, int]:
    """Clamp output to 512–1024px, preserve aspect ratio, multiples of 8."""
    if src_w < 1 or src_h < 1:
        side = max_side or SDXL_MIN_SIDE
        return side, side

    cap = max_side or _max_output_side()
    cap = max(SDXL_MIN_SIDE, min(SDXL_MAX_SIDE, cap))
    scale = cap / max(src_w, src_h)
    out_w = _round_to_multiple(src_w * scale)
    out_h = _round_to_multiple(src_h * scale)

    longest = max(out_w, out_h)
    if longest < SDXL_MIN_SIDE:
        boost = SDXL_MIN_SIDE / longest
        out_w = _round_to_multiple(out_w * boost)
        out_h = _round_to_multiple(out_h * boost)

    longest = max(out_w, out_h)
    if longest > SDXL_MAX_SIDE:
        shrink = SDXL_MAX_SIDE / longest
        out_w = _round_to_multiple(out_w * shrink)
        out_h = _round_to_multiple(out_h * shrink)

    return out_w, out_h


def _load_sdxl_pipeline():
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    hf_kw = _hf_hub_kwargs()
    print(f"Veda image-gen: loading img2img ({SDXL_MODEL}) on {SDXL_DEVICE} …")
    log_vram_snapshot("before_sdxl_load")

    dtype = torch.float16 if SDXL_DEVICE.startswith("cuda") else torch.float32
    print(
        f"Veda image-gen: downloading img2img weights "
        f"(largest download in pre-warm, {_img2img_download_hint()}) …"
    )
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_MODEL,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
        **hf_kw,
    )
    print(f"Veda image-gen: moving img2img pipeline to {SDXL_DEVICE} …")
    pipe = pipe.to(SDXL_DEVICE)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)

    _sdxl_pipe = pipe
    log_vram_snapshot("after_sdxl_load")
    print(f"Veda image-gen: img2img ready ({SDXL_MODEL}).")
    return _sdxl_pipe


def get_sdxl_pipeline():
    if _sdxl_pipe is not None:
        return _sdxl_pipe
    with _sdxl_lock:
        if _sdxl_pipe is not None:
            return _sdxl_pipe
        return _load_sdxl_pipeline()


def _prepare_init_image(pil_img, out_w: int, out_h: int):
    """Resize source image to SDXL output dimensions for img2img."""
    if pil_img.width == out_w and pil_img.height == out_h:
        return pil_img.convert("RGB")
    return pil_img.convert("RGB").resize((out_w, out_h), resample=1)  # LANCZOS


def generate_image_sdxl(prompt: str, pil_img, src_w: int, src_h: int, strength: float | None = None):
    """Layer 2 — SDXL img2img enhancement from original pixels + refined prompt."""
    import torch

    out_w, out_h = compute_output_size(src_w, src_h)
    denoise = SDXL_STRENGTH if strength is None else strength
    denoise = max(0.05, min(0.95, denoise))
    init_image = _prepare_init_image(pil_img, out_w, out_h)
    print(
        f"Veda image-gen: img2img {out_w}×{out_h} px, "
        f"steps={SDXL_STEPS}, strength={denoise:.2f}"
    )

    pipe = get_sdxl_pipeline()
    generator = None
    if SDXL_DEVICE.startswith("cuda"):
        device_idx = int(SDXL_DEVICE.split(":")[1]) if ":" in SDXL_DEVICE else 0
        generator = torch.Generator(device=f"cuda:{device_idx}").manual_seed(42)

    with _sdxl_lock:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=init_image,
                negative_prompt=NEGATIVE_PROMPT,
                strength=denoise,
                num_inference_steps=SDXL_STEPS,
                guidance_scale=SDXL_GUIDANCE,
                generator=generator,
            )

    image = result.images[0].convert("RGB")
    log_vram_snapshot("layer2_done")
    return image, out_w, out_h


def pil_to_data_url(pil_img, fmt: str = "png") -> str:
    buf = BytesIO()
    if fmt == "jpeg":
        pil_img.save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"
    else:
        pil_img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def run_layer2(prompt: str, pil_img, src_w: int, src_h: int) -> dict[str, Any]:
    """Layer 2 only — SDXL img2img from original image + prompt."""
    log_vram_snapshot("layer2_start")
    try:
        image, out_w, out_h = generate_image_sdxl(prompt, pil_img, src_w, src_h)
    except Exception as exc:
        err_lower = str(exc).lower()
        if "out of memory" in err_lower or "cuda" in err_lower:
            print(f"Veda image-gen: SDXL OOM ({exc}) — retrying at {SDXL_MIN_SIDE}px …")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            small = pil_img.resize((SDXL_MIN_SIDE, SDXL_MIN_SIDE), resample=1)
            image, out_w, out_h = generate_image_sdxl(prompt, small, SDXL_MIN_SIDE, SDXL_MIN_SIDE)
        else:
            raise
    return {
        "image": pil_to_data_url(image),
        "width": out_w,
        "height": out_h,
        "method": _img2img_method_tag(),
    }


def run_full_pipeline(pil_img, page_text: str, llm, llm_lock) -> dict[str, Any]:
    """Layer 1 + Layer 2: caption → prompt → img2img PNG."""
    layer1 = run_layer1(pil_img, page_text, llm, llm_lock)
    layer2 = run_layer2(layer1["prompt"], pil_img, pil_img.width, pil_img.height)
    tag = _img2img_method_tag().replace("_img2img", "")
    return {
        "caption": layer1["caption"],
        "prompt": layer1["prompt"],
        "image": layer2["image"],
        "width": layer2["width"],
        "height": layer2["height"],
        "method": f"layer2_smolvlm_qwen_{tag}_img2img",
    }
