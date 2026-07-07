#!/usr/bin/env python3
"""
Veda GPU inference server — run inside a Kaggle notebook (GPU T4 x2).

Setup (one-time):
  1. Create a Kaggle Dataset "veda-models" with:
       - qwen2.5-3b-instruct-q4_k_m.gguf  (or 1.5B / 0.5B)
       - RealESRGAN_x4plus.pth            (optional, for /upscale_image)
       - rrdb_net.py                      (copy from server/rrdb_net.py)
       - analyze_prompts.py               (copy from server/analyze_prompts.py)
  2. New Kaggle notebook → Settings:
       - Accelerator: GPU T4 x2
       - Internet: ON
       - Add dataset veda-models
  3. Paste this file (or upload) and run. Copy the printed tunnel URL into server/.env:
       KAGGLE_API_BASE_URL=https://xxxx.trycloudflare.com
       KAGGLE_API_SECRET=your-secret
       LLM_PROVIDER=kaggle
       UPSCALE_PROVIDER=kaggle

Stop the notebook session when done — GPU hours tick while the session is open.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

# ── Kaggle paths ─────────────────────────────────────────────────────────────
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

# import_name -> pip install args (llama_cpp needs package name llama-cpp-python + CUDA wheel)
_PIP_PACKAGES: list[tuple[str, list[str]]] = [
    ("fastapi", ["fastapi"]),
    ("uvicorn", ["uvicorn"]),
    ("transformers", ["transformers"]),
    ("torch", ["torch"]),
    ("fitz", ["pymupdf"]),
    ("pdfplumber", ["pdfplumber"]),
    ("cv2", ["opencv-python-headless"]),
]


def _install_llama_cpp() -> None:
    """Install llama-cpp-python with CUDA wheels for Kaggle GPU."""
    cuda_indexes = (
        "https://abetlen.github.io/llama-cpp-python/whl/cu124",
        "https://abetlen.github.io/llama-cpp-python/whl/cu122",
        "https://abetlen.github.io/llama-cpp-python/whl/cu121",
    )
    for index in cuda_indexes:
        print(f"Installing llama-cpp-python (CUDA wheel from {index}) …")
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "llama-cpp-python",
                "--extra-index-url", index,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        print(result.stderr.strip() or result.stdout.strip())
    print("CUDA wheels failed — installing CPU-only llama-cpp-python …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python"])


def _fix_kaggle_pillow_stack() -> None:
    """Align Pillow + torchvision on Kaggle (avoids _Ink and _imaging version skew).

    Do NOT downgrade Pillow — pdfplumber 0.11+ requires Pillow>=12.2.
    The original _Ink error is from an outdated torchvision against Pillow 12.
    """
    if not INPUT_ROOT.exists():
        return
    try:
        print("Fixing Pillow/torchvision stack for NLLB (force-reinstall Pillow 12 + upgrade torchvision) …")
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "--force-reinstall", "pillow>=12.2.0",
            ],
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "torchvision"],
        )
    except Exception as exc:
        print(f"Pillow/torchvision fix skipped ({exc})")


def _ensure_packages() -> None:
    _fix_kaggle_pillow_stack()
    for import_name, pip_args in _PIP_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pip_args])
    try:
        __import__("llama_cpp")
    except ImportError:
        _install_llama_cpp()


def _setup_upscale_service_path() -> None:
    """Load upscale_service from working (manual upload) or copy from dataset."""
    import shutil

    dest = WORKING / "upscale_service.py"
    if dest.is_file():
        if str(WORKING) not in sys.path:
            sys.path.insert(0, str(WORKING))
        print(f"upscale_service: using {dest}")
        return
    for svc in INPUT_ROOT.rglob("upscale_service.py"):
        shutil.copy2(svc, dest)
        if str(WORKING) not in sys.path:
            sys.path.insert(0, str(WORKING))
        print(f"upscale_service: copied from {svc} → working")
        return
    if str(WORKING) not in sys.path:
        sys.path.insert(0, str(WORKING))


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

_setup_upscale_service_path()

from upscale_service import upscale_image as run_face_aware_upscale  # noqa: E402

API_SECRET = os.environ.get("KAGGLE_API_SECRET", "veda-kaggle-dev")
PORT = int(os.environ.get("VEDA_GPU_PORT", "8766"))
NLLB_MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_DEVICE = os.environ.get("NLLB_DEVICE", "cuda:1")
TRANSLATE_BATCH_SIZE = int(os.environ.get("TRANSLATE_BATCH_SIZE", "4"))

app = FastAPI(title="Veda Kaggle GPU Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_bearer = HTTPBearer(auto_error=False)


def verify_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if not API_SECRET:
        return
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

        gguf = _find_file("qwen2.5-3b-instruct-q4_k_m.gguf")
        if gguf is None:
            for pattern in ("qwen*.gguf", "*.gguf"):
                matches = list(INPUT_ROOT.rglob(pattern))
                if matches:
                    gguf = matches[0]
                    break
        if gguf is None:
            raise RuntimeError("No Qwen GGUF found in /kaggle/input — add to your dataset.")

        n_gpu = int(os.environ.get("LLM_GPU_LAYERS", "-1"))
        print(f"Loading LLM from {gguf} (n_gpu_layers={n_gpu}, main_gpu=0) …")
        _llm = Llama(
            model_path=str(gguf),
            n_ctx=4096,
            n_gpu_layers=n_gpu,
            main_gpu=0,
            verbose=False,
        )
        print("LLM ready on GPU.")
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

        print(f"Loading NLLB ({NLLB_MODEL}) …")
        tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        device_name = NLLB_DEVICE
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        if device_name.startswith("cuda"):
            idx = int(device_name.split(":")[1]) if ":" in device_name else 0
            if torch.cuda.device_count() <= idx:
                device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            if device_name.startswith("cuda"):
                torch.cuda.empty_cache()
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
            model = model.to(device_name)
            print(f"NLLB ready on {device_name}.")
        except Exception as gpu_err:
            print(f"NLLB load on {device_name} failed ({gpu_err}) — using CPU.")
            device_name = "cpu"
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device_name)
            print("NLLB ready on CPU.")

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

    print(f"NLLB translated {len(texts)} strings → {target_lang} on {_nllb_device_name}")
    return translated


def _prewarm_nllb() -> None:
    try:
        run_nllb_translate(["Hello world"], "tel_Telu")
        print("NLLB pre-warm complete.")
    except Exception as exc:
        print(f"NLLB pre-warm failed (will retry on first /translate): {exc}")


# ── Real-ESRGAN (GPU 1) ──────────────────────────────────────────────────────
_esrgan = None
_esrgan_lock = threading.Lock()


def get_esrgan():
    global _esrgan
    if _esrgan is not None:
        return _esrgan
    with _esrgan_lock:
        if _esrgan is not None:
            return _esrgan
        import torch

        weights = _find_file("RealESRGAN_x4plus.pth")
        if weights is None:
            raise RuntimeError("RealESRGAN_x4plus.pth not in dataset")

        rrdb_dir = weights.parent
        if (rrdb_dir / "rrdb_net.py").exists():
            sys.path.insert(0, str(rrdb_dir))
        else:
            for p in INPUT_ROOT.rglob("rrdb_net.py"):
                sys.path.insert(0, str(p.parent))
                break

        from rrdb_net import RRDBNet

        device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state = torch.load(str(weights), map_location="cpu")
        w = state.get("params_ema", state.get("params", state))
        model.load_state_dict(w, strict=True)
        model.eval().to(device)
        _esrgan = (model, device)
        print(f"Real-ESRGAN ready on {device}.")
        return _esrgan


# ── PDF layout extraction (mirrors laptop server) ────────────────────────────
_pdf_doc = None
_pdf_analyzer = None
_pdf_lock = threading.Lock()
PDF_PATH = WORKING / "active_doc.pdf"


class AnalyzeBody(BaseModel):
    text: str
    is_digest: bool = False


class TranslateBody(BaseModel):
    texts: list[str]
    target_lang: str = "tel_Telu"


@app.get("/health")
def health():
    import torch

    return {
        "ok": True,
        "service": "veda-kaggle-gpu",
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "nllb_loaded": _nllb_model is not None,
        "nllb_device": _nllb_device_name,
    }


@app.post("/analyze")
def analyze(body: AnalyzeBody, _: None = Depends(verify_token)):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    llm = get_llm()
    messages = build_analyze_messages(text, is_digest=body.is_digest)
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


@app.post("/upload_pdf")
async def upload_pdf(request: Request, _: None = Depends(verify_token)):
    global _pdf_doc, _pdf_analyzer
    pdf_bytes = await request.body()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")
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


@app.post("/upscale_image")
async def upscale_image(request: Request, _: None = Depends(verify_token)):
    import base64

    from PIL import Image

    payload = await request.json()
    b64 = payload.get("image", "")
    fmt = payload.get("format", "png").lower()
    if not b64:
        raise HTTPException(status_code=400, detail="Missing image")

    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    w, h = img.size

    esrgan_model = None
    device = None
    try:
        esrgan_model, device = get_esrgan()
    except RuntimeError:
        pass

    out, method = run_face_aware_upscale(img, esrgan_model=esrgan_model, device=device)
    print(f"Kaggle /upscale_image: {w}×{h} px → {method}")

    buf = BytesIO()
    if fmt == "jpeg":
        out.save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"
    else:
        out.save(buf, format="PNG")
        mime = "image/png"
    out_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": f"data:{mime};base64,{out_b64}", "method": method}


def _install_cloudflared() -> Path:
    dest = Path("/usr/local/bin/cloudflared")
    if dest.exists():
        return dest
    import urllib.request

    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    print("Downloading cloudflared …")
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
    print(f"API secret (set KAGGLE_API_SECRET on Veda server): {API_SECRET}")
    print(f"Starting FastAPI on port {PORT} …")

    threading.Thread(target=_prewarm_nllb, daemon=True).start()

    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info"),
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
        print(line.rstrip())
        match = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if match:
            public_url = match.group(1)

    if public_url:
        print("\n" + "=" * 60)
        print("Add to server/.env on your Veda machine:")
        print(f"  KAGGLE_API_BASE_URL={public_url}")
        print(f"  KAGGLE_API_SECRET={API_SECRET}")
        print("  LLM_PROVIDER=kaggle")
        print("  TRANSLATE_PROVIDER=kaggle")
        print("  UPSCALE_PROVIDER=kaggle")
        print("=" * 60)
        print("Keep this notebook running while using Veda. Stop session when done.")
    else:
        print("Tunnel URL not detected — check cloudflared output above.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        tunnel.terminate()


if __name__ == "__main__":
    main()
