#!/usr/bin/env python3
"""
Veda GPU inference server — run inside a Kaggle notebook (GPU T4 x2).

Setup (one-time):
  1. Create a Kaggle Dataset "veda-models" with:
       - qwen2.5-3b-instruct-q4_k_m.gguf  (or 1.5B / 0.5B)
       - analyze_prompts.py, page_layout_service.py, security.py
  2. New Kaggle notebook → Settings:
       - Accelerator: GPU T4 x2
       - Internet: ON
       - Add dataset veda-models
  3. Paste this file (or upload) and run. Copy the printed tunnel URL into server/.env:
       KAGGLE_API_BASE_URL=https://xxxx.trycloudflare.com
       KAGGLE_API_SECRET=your-secret
       LLM_PROVIDER=kaggle
       TRANSLATE_PROVIDER=kaggle

Stop the notebook session when done — GPU hours tick while the session is open.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from io import BytesIO
from pathlib import Path

# ── Quiet startup (Kaggle notebook cell output) ───────────────────────────────
def _configure_quiet_startup() -> None:
    os.environ.setdefault("PIP_PROGRESS_BAR", "off")
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    for name in ("transformers", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")


def _quiet() -> bool:
    return os.environ.get("VEDA_QUIET", "1").strip().lower() not in ("0", "false", "no")


def _log(msg: str) -> None:
    if _quiet():
        return
    print(msg)


_configure_quiet_startup()

# ── Kaggle paths ─────────────────────────────────────────────────────────────
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

# Kaggle ships torch; do not pip-install it (avoids huge re-downloads).
_PIP_PACKAGES: list[tuple[str, list[str]]] = [
    ("fastapi", ["fastapi"]),
    ("uvicorn", ["uvicorn"]),
    ("transformers", ["transformers"]),
    ("fitz", ["pymupdf"]),
    ("pdfplumber", ["pdfplumber"]),
]


def _quiet_pip(*pip_args: str) -> None:
    env = {**os.environ, "PIP_PROGRESS_BAR": "off", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *pip_args],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"pip install failed: {detail[:500]}")


def _install_llama_cpp() -> None:
    """Install llama-cpp-python with CUDA wheels for Kaggle GPU."""
    print("Veda: installing llama-cpp-python (CUDA) …")
    cuda_indexes = (
        "https://abetlen.github.io/llama-cpp-python/whl/cu124",
        "https://abetlen.github.io/llama-cpp-python/whl/cu122",
        "https://abetlen.github.io/llama-cpp-python/whl/cu121",
    )
    for index in cuda_indexes:
        try:
            _quiet_pip("llama-cpp-python", "--extra-index-url", index)
            print("Veda: llama-cpp-python ready.")
            return
        except RuntimeError:
            continue
    _quiet_pip("llama-cpp-python")
    print("Veda: llama-cpp-python ready (CPU wheel).")


def _fix_kaggle_pillow_stack() -> None:
    """Align Pillow 12.x + torchvision; keep transformers 4.x with huggingface_hub 0.x."""
    if not INPUT_ROOT.exists():
        return
    try:
        print("Veda: aligning Pillow / torchvision / transformers stack for Kaggle …")
        _quiet_pip(
            "--force-reinstall",
            "pillow>=12.2.0",
            "torchvision",
            "transformers>=4.46,<5",
            "huggingface_hub>=0.34,<1.0",
        )
    except Exception as exc:
        print(f"Veda: Pillow/torchvision fix skipped ({exc})")


def _ensure_packages() -> None:
    for import_name, pip_args in _PIP_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            _quiet_pip(*pip_args)
    try:
        __import__("llama_cpp")
    except ImportError:
        _install_llama_cpp()


_ensure_packages()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import uvicorn

# Import shared prompts (search full dataset tree, e.g. .../project-models/)
for candidate in INPUT_ROOT.rglob("analyze_prompts.py"):
    sys.path.insert(0, str(candidate.parent))
    break
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_prompts import build_analyze_messages, extract_analysis_json  # noqa: E402

for svc in INPUT_ROOT.rglob("page_layout_service.py"):
    sys.path.insert(0, str(svc.parent))
    break
else:
    sys.path.insert(0, str(WORKING))

from page_layout_service import PDFMetadataAnalyzer, extract_page_layout  # noqa: E402

for qc in INPUT_ROOT.rglob("qwen_config.py"):
    sys.path.insert(0, str(qc.parent))
    break
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_config import get_kaggle_qwen_gguf_info  # noqa: E402

for ig in INPUT_ROOT.rglob("image_gen_service.py"):
    sys.path.insert(0, str(ig.parent))
    break
else:
    sys.path.insert(0, str(WORKING))

from image_gen_service import get_vram_stats, log_vram_snapshot, run_layer1  # noqa: E402

for sec in INPUT_ROOT.rglob("security.py"):
    sys.path.insert(0, str(sec.parent))
    break
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from security import (  # noqa: E402
    MAX_IMAGE_B64_CHARS,
    MAX_IMAGE_PIXELS,
    MAX_PDF_BYTES,
    cors_origins,
    require_kaggle_secret,
    validate_b64_payload,
    validate_pdf_bytes,
)

API_SECRET = require_kaggle_secret(os.environ.get("KAGGLE_API_SECRET"))
PORT = int(os.environ.get("VEDA_GPU_PORT", "8766"))
NLLB_MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_DEVICE = os.environ.get("NLLB_DEVICE", "cuda:1")
TRANSLATE_BATCH_SIZE = int(os.environ.get("TRANSLATE_BATCH_SIZE", "4"))

app = FastAPI(title="Veda Kaggle GPU Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_bearer = HTTPBearer(auto_error=False)


def verify_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if creds is None or creds.credentials != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _find_file(name: str) -> Path | None:
    for path in INPUT_ROOT.rglob(name):
        if path.is_file():
            return path
    local = WORKING / "models" / name
    return local if local.exists() else None


# ── Qwen LLM (GPU 0) ─────────────────────────────────────────────────────────
_llm = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is not None:
            return _llm
        from llama_cpp import Llama

        _size, preferred_name, _url = get_kaggle_qwen_gguf_info()
        gguf = _find_file(preferred_name)
        if gguf is None:
            for pattern in ("qwen*.gguf", "*.gguf"):
                matches = sorted(INPUT_ROOT.rglob(pattern))
                if matches:
                    gguf = matches[0]
                    print(f"Veda: preferred {_size} ({preferred_name}) not in dataset; using {gguf.name}")
                    break
        if gguf is None:
            raise RuntimeError(
                f"No Qwen GGUF found in /kaggle/input — add {preferred_name} to project-models."
            )

        n_gpu = int(os.environ.get("LLM_GPU_LAYERS", "-1"))
        print(f"Veda: loading Qwen {_size} from {gguf.name} (n_gpu_layers={n_gpu}, gpu=0) …")
        _llm = Llama(
            model_path=str(gguf),
            n_ctx=4096,
            n_gpu_layers=n_gpu,
            main_gpu=0,
            verbose=False,
        )
        print("Veda: Qwen LLM ready on GPU 0.")
        return _llm


# ── NLLB translation (GPU 1 by default, LLM pinned to GPU 0) ─────────────────
_nllb_model = None
_nllb_tokenizer = None
_nllb_device_name = None
_nllb_lock = threading.Lock()


def _nllb_target_lang_id(tokenizer, target_lang: str) -> int:
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_map, dict) and target_lang in lang_map:
        return lang_map[target_lang]
    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if token_id != tokenizer.unk_token_id:
        return token_id
    raise ValueError(f"Unknown target language code: {target_lang}")


def get_nllb():
    global _nllb_model, _nllb_tokenizer, _nllb_device_name
    if _nllb_model is not None:
        return _nllb_model, _nllb_tokenizer
    with _nllb_lock:
        if _nllb_model is not None:
            return _nllb_model, _nllb_tokenizer
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Veda: loading NLLB ({NLLB_MODEL}) …")
        tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        device_name = NLLB_DEVICE
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        if device_name.startswith("cuda"):
            idx = int(device_name.split(":")[1]) if ":" in device_name else 0
            if torch.cuda.device_count() <= idx:
                device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

        load_kwargs = {"use_safetensors": True}

        def _load_model(dev: str):
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            m = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, **load_kwargs)
            return m.to(dev)

        model = None
        try:
            model = _load_model(device_name)
        except Exception as first_err:
            err_text = str(first_err)
            if "PIL" in err_text or "_Ink" in err_text:
                _fix_kaggle_pillow_stack()
                try:
                    model = _load_model(device_name)
                except Exception as retry_err:
                    first_err = retry_err
            if model is None:
                print(f"Veda: NLLB GPU load failed ({first_err}) — trying CPU.")
                device_name = "cpu"
                model = _load_model(device_name)

        print(f"Veda: NLLB ready on {device_name}.")

        model.eval()
        _nllb_model, _nllb_tokenizer, _nllb_device_name = model, tok, device_name
        return _nllb_model, _nllb_tokenizer


def run_nllb_translate(texts: list[str], target_lang: str = "tel_Telu") -> list[str]:
    """Translate English strings to target_lang using NLLB on GPU."""
    import torch

    if not texts:
        return []

    model, tokenizer = get_nllb()
    device = next(model.parameters()).device
    target_id = _nllb_target_lang_id(tokenizer, target_lang)
    clean_texts = [t if str(t).strip() else " " for t in texts]
    translated: list[str] = []

    for start in range(0, len(clean_texts), TRANSLATE_BATCH_SIZE):
        batch = clean_texts[start : start + TRANSLATE_BATCH_SIZE]
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_words = max(len(t.split()) for t in batch)
        max_new_tokens = min(max_words * 3 + 20, 256)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                num_beams=2,
                max_new_tokens=max_new_tokens,
                length_penalty=1.0,
            )
        translated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    _log(f"NLLB translated {len(texts)} strings → {target_lang} on {_nllb_device_name}")
    return translated


def _prewarm_nllb() -> None:
    try:
        get_nllb()
        print("Veda: NLLB pre-warm complete.")
    except Exception as exc:
        print(f"Veda: NLLB pre-warm failed (will retry on first /translate): {exc}")


# ── PDF layout extraction (mirrors laptop server) ────────────────────────────
_pdf_doc = None
_pdf_analyzer = None
_pdf_lock = threading.Lock()
PDF_PATH = WORKING / "active_doc.pdf"


class AnalyzeBody(BaseModel):
    text: str
    is_digest: bool = False
    page_layout: str = "plain"


class TranslateBody(BaseModel):
    texts: list[str]
    target_lang: str = "tel_Telu"


class GeneratePromptBody(BaseModel):
    image: str
    page_text: str = ""


@app.get("/health")
def health(_: None = Depends(verify_token)):
    import torch

    return {
        "ok": True,
        "service": "veda-kaggle-gpu",
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "nllb_loaded": _nllb_model is not None,
        "llm_loaded": _llm is not None,
        "image_gen_layer": 1,
        "vram": get_vram_stats(),
    }


@app.post("/analyze")
def analyze(body: AnalyzeBody, _: None = Depends(verify_token)):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    llm = get_llm()
    messages = build_analyze_messages(
        text, is_digest=body.is_digest, page_layout=body.page_layout
    )
    with _llm_lock:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
            repeat_penalty=1.1,
        )
    output = response["choices"][0]["message"]["content"].strip()
    try:
        return extract_analysis_json(output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"JSON parse failed: {exc}") from exc


@app.post("/translate")
def translate(body: TranslateBody, _: None = Depends(verify_token)):
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts is required")

    try:
        return {"translated": run_nllb_translate(body.texts, target_lang=body.target_lang)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Kaggle /translate failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate_prompt")
def generate_prompt(body: GeneratePromptBody, _: None = Depends(verify_token)):
    """
    Layer 1 trial — SmolVLM caption + Qwen 7B SDXL prompt refinement.
    Does not generate an image yet (Layer 2 SDXL comes later).
    """
    from PIL import Image

    b64 = body.image.strip()
    if not b64:
        raise HTTPException(status_code=400, detail="image is required")

    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        validate_b64_payload(b64, max_chars=MAX_IMAGE_B64_CHARS)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        pil_img = Image.open(BytesIO(__import__("base64").b64decode(b64))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    page_text = (body.page_text or "").strip()
    print(
        f"Kaggle /generate_prompt: input {pil_img.width}×{pil_img.height} px, "
        f"page_text={len(page_text)} chars"
    )

    try:
        llm = get_llm()
        result = run_layer1(pil_img, page_text, llm, _llm_lock)
        return result
    except Exception as exc:
        err_text = str(exc)
        if "PIL" in err_text or "_Ink" in err_text:
            print(f"Kaggle /generate_prompt: Pillow stack error ({exc}) — retrying after fix …")
            _fix_kaggle_pillow_stack()
            try:
                llm = get_llm()
                result = run_layer1(pil_img, page_text, llm, _llm_lock)
                return result
            except Exception as retry_err:
                exc = retry_err
        print(f"Kaggle /generate_prompt failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload_pdf")
async def upload_pdf(request: Request, _: None = Depends(verify_token)):
    global _pdf_doc, _pdf_analyzer
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_PDF_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds maximum size ({MAX_PDF_BYTES // (1024 * 1024)} MB)",
                )
        except ValueError:
            pass

    pdf_bytes = await request.body()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")
    try:
        validate_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _pdf_lock:
        if _pdf_doc is not None:
            _pdf_doc.close()
        PDF_PATH.write_bytes(pdf_bytes)
        import fitz
        _pdf_doc = fitz.open(str(PDF_PATH))
        _pdf_analyzer = PDFMetadataAnalyzer(_pdf_doc)
    print(f"Kaggle loaded PDF ({len(pdf_bytes)} bytes, {len(_pdf_doc)} pages)")
    return {"ok": True, "num_pages": len(_pdf_doc)}


@app.get("/page_layout")
def page_layout(page: int, _: None = Depends(verify_token)):
    global _pdf_doc, _pdf_analyzer
    with _pdf_lock:
        if _pdf_doc is None:
            if not PDF_PATH.exists():
                raise HTTPException(status_code=400, detail="No PDF uploaded to Kaggle server")
            import fitz
            _pdf_doc = fitz.open(str(PDF_PATH))
            _pdf_analyzer = PDFMetadataAnalyzer(_pdf_doc)
        if page < 1 or page > len(_pdf_doc):
            raise HTTPException(status_code=400, detail=f"Invalid page number {page}")
        try:
            result = extract_page_layout(_pdf_doc, _pdf_analyzer, page)
            print(f"Kaggle layout: {len(result.get('lines', []))} lines for page {page}")
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def _install_cloudflared() -> Path:
    dest = Path("/usr/local/bin/cloudflared")
    if dest.exists():
        return dest
    import urllib.request

    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    if not _quiet():
        print("Veda: downloading cloudflared …")
    urllib.request.urlretrieve(url, dest)
    dest.chmod(0o755)
    return dest


def start_tunnel() -> subprocess.Popen:
    cloudflared = _install_cloudflared()
    proc = subprocess.Popen(
        [str(cloudflared), "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def main():
    print("Veda GPU server starting (quiet mode). Set VEDA_QUIET=0 for verbose logs.")
    print(f"  port={PORT}  gpus=2 expected  nllb={NLLB_MODEL}")
    if INPUT_ROOT.exists():
        _fix_kaggle_pillow_stack()
    log_vram_snapshot("startup")

    threading.Thread(target=_prewarm_nllb, daemon=True).start()

    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning"),
        daemon=True,
    )
    server.start()
    time.sleep(2)

    tunnel = start_tunnel()
    public_url = None
    deadline = time.time() + 60
    while time.time() < deadline and public_url is None:
        line = tunnel.stdout.readline() if tunnel.stdout else ""
        if not line:
            time.sleep(0.2)
            continue
        match = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if match:
            public_url = match.group(1)

    if public_url:
        print("\n" + "=" * 60)
        print("Add to server/.env on your Veda machine:")
        print(f"  KAGGLE_API_BASE_URL={public_url}")
        print("  KAGGLE_API_SECRET=<same secret you set in this notebook>")
        print("  KAGGLE_ENABLED=true")
        print("  LLM_PROVIDER=kaggle")
        print("  TRANSLATE_PROVIDER=kaggle")
        print("  LAYOUT_PROVIDER=kaggle")
        print("=" * 60)
        print("Ready. Keep this notebook running while using Veda.")
    else:
        print("Veda: tunnel URL not detected — set VEDA_QUIET=0 and check cloudflared output.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        tunnel.terminate()


if __name__ == "__main__":
    main()
