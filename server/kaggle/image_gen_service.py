"""Kaggle-only Layer 1 image prompt pipeline: SmolVLM caption → Qwen 7B SDXL prompt."""
from __future__ import annotations

import os
import re
import threading
from typing import Any

SMOLVLM_MODEL = os.environ.get("SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
SMOLVLM_DEVICE = os.environ.get("SMOLVLM_DEVICE", "cuda:0")
CAPTION_MAX_TOKENS = int(os.environ.get("SMOLVLM_CAPTION_MAX_TOKENS", "128"))
PROMPT_MAX_TOKENS = int(os.environ.get("IMAGE_PROMPT_MAX_TOKENS", "220"))
PAGE_TEXT_MAX_CHARS = int(os.environ.get("IMAGE_GEN_PAGE_TEXT_MAX_CHARS", "1500"))

CAPTION_USER_TEXT = (
    "Describe this educational magazine illustration in 2-4 sentences. "
    "Focus on the main subject, setting, colors, and mood. "
    "Do not invent readable text, watermarks, or logos."
)

PROMPT_SYSTEM = (
    "You write concise Stable Diffusion XL prompts for educational magazine illustrations.\n"
    "Given an image caption and page context, output ONE paragraph prompt (40-80 words) "
    "describing a high-quality illustration to generate.\n"
    "Include style cues: photorealistic or illustrated, lighting, composition, and subject.\n"
    "Do NOT include readable text, watermarks, logos, captions, or words in the scene.\n"
    "Output ONLY the prompt text — no explanation, no JSON, no markdown."
)

_smolvlm_processor = None
_smolvlm_model = None
_smolvlm_lock = threading.Lock()


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


def _load_smolvlm():
    global _smolvlm_processor, _smolvlm_model
    if _smolvlm_model is not None:
        return _smolvlm_processor, _smolvlm_model

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Veda image-gen: loading SmolVLM ({SMOLVLM_MODEL}) on {SMOLVLM_DEVICE} …")
    log_vram_snapshot("before_smolvlm_load")

    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL)
    dtype = torch.float16 if SMOLVLM_DEVICE.startswith("cuda") else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        SMOLVLM_MODEL,
        torch_dtype=dtype,
    ).to(SMOLVLM_DEVICE)
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
