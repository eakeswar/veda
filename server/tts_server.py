from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import threading

# Windows console defaults to cp1252 which can't encode most Unicode characters.
# Reconfigure stdout/stderr to UTF-8 so LLM output with arrows, dashes, etc. never
# crashes a print() call and kills the request handler.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import fitz
import wave
from io import BytesIO
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load dotenv from BASE_DIR
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

import requests

from analyze_prompts import build_analyze_messages, extract_analysis_json
from decorative_image import should_route_img2img
from kaggle_client import (
    kaggle_analyze,
    kaggle_generate_image,
    kaggle_page_layout,
    kaggle_status,
    kaggle_translate,
    kaggle_upload_pdf,
    resolve_provider,
    should_use_kaggle,
)
from page_layout_service import PDFMetadataAnalyzer, extract_page_layout, merge_background_images
from qwen_config import get_local_qwen_gguf_info
from security import (
    MAX_ANALYZE_TEXT_CHARS,
    MAX_IMAGE_PIXELS,
    MAX_PDF_BYTES,
    MAX_TTS_TEXT_CHARS,
    cors_origins,
    redact_kaggle_status,
    validate_b64_payload,
    validate_pdf_bytes,
    validate_text_length,
)
from upscale_service import upscale_image as run_face_aware_upscale

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor"

if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

try:
    # pyrefly: ignore [missing-import]
    import edge_tts
except ModuleNotFoundError as exc:
    raise SystemExit(
        "edge-tts is not installed. Install it with:\n"
        "python -m pip install --target server/vendor -r server/requirements.txt"
    ) from exc

try:
    import pdfplumber as _pdfplumber  # layout-aware text extraction
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    print("pdfplumber not available — text extraction will use PyMuPDF only")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ModuleNotFoundError:
    # Fallback to standard library if pip installation hasn't fully registered in-process yet
    print("FastAPI is not installed in the target vendor path. Installing requirements...")
    raise

app = FastAPI(title="TTS & Slide Analysis Server")

VEDA_API_KEY = os.environ.get("VEDA_API_KEY", "").strip()

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def veda_api_key_middleware(request: Request, call_next):
    """Optional API key auth — active only when VEDA_API_KEY is set in .env."""
    if not VEDA_API_KEY or request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    header_key = request.headers.get("X-Veda-Api-Key", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else header_key.strip()
    if token != VEDA_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


VOICE_MAP = {
    "en-US": "en-US-AriaNeural",
    "te-IN": "te-IN-ShrutiNeural",
}

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")

SARVAM_CREDITS_MESSAGE = (
    "Your Sarvam API credits are exhausted. "
    "Add credits at dashboard.sarvam.ai to restore TTS, STT, translation, and read-along."
)


class SarvamCreditsExhaustedError(Exception):
    """Raised when Sarvam API credits or quota are exhausted."""


def _sarvam_error_text(body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error")
        if isinstance(err, dict):
            return f"{err.get('code', '')} {err.get('message', '')}".strip()
        if isinstance(err, str):
            return err
        return str(data.get("message", body))
    except Exception:
        return body


def is_sarvam_credits_exhausted(status_code: int, body: str) -> bool:
    if status_code == 402:
        return True
    text = _sarvam_error_text(body).lower()
    keywords = (
        "credit", "credits", "quota", "balance", "exhausted",
        "insufficient", "payment required", "billing",
        "limit exceeded", "no more credit", "out of credit",
        "usage limit", "subscription",
    )
    if status_code in (403, 429) and any(keyword in text for keyword in keywords):
        return True
    return False


def check_sarvam_response(response: requests.Response) -> None:
    if is_sarvam_credits_exhausted(response.status_code, response.text):
        raise SarvamCreditsExhaustedError(SARVAM_CREDITS_MESSAGE)
    response.raise_for_status()


def sarvam_credits_http_exception() -> HTTPException:
    return HTTPException(
        status_code=402,
        detail={"code": "sarvam_credits_exhausted", "message": SARVAM_CREDITS_MESSAGE},
    )

def get_sarvam_speaker(voice: str, lang: str) -> str:
    voice_lower = voice.lower() if voice else ""
    lang_lower = lang.lower() if lang else ""
    if "shruti" in voice_lower or lang_lower == "te-in":
        return "kavitha"
    if "aria" in voice_lower or lang_lower in ["en-us", "en-in"]:
        return "priya"
    return "shubh"

def parse_rate_to_pace(rate_str: str) -> float:
    if not rate_str:
        return 1.0
    match = re.search(r'([+-]?\d+)\s*%', rate_str)
    if match:
        try:
            pct = float(match.group(1))
            pace = 1.0 + (pct / 100.0)
            return max(0.5, min(2.0, pace))
        except ValueError:
            pass
    return 1.0

class SarvamMemoryCache:
    _lock = threading.Lock()
    last_text = None
    last_language = None
    last_voice = None
    last_rate = None
    last_audio_bytes = None
    last_word_boundaries = None

    @classmethod
    def get(cls, text: str, language: str, voice: str, rate: str):
        with cls._lock:
            if (cls.last_text == text and 
                cls.last_language == language and 
                cls.last_voice == voice and 
                cls.last_rate == rate):
                return cls.last_audio_bytes, cls.last_word_boundaries
        return None, None

    @classmethod
    def set(cls, text: str, language: str, voice: str, rate: str, audio_bytes: bytes, word_boundaries: list):
        with cls._lock:
            cls.last_text = text
            cls.last_language = language
            cls.last_voice = voice
            cls.last_rate = rate
            cls.last_audio_bytes = audio_bytes
            cls.last_word_boundaries = word_boundaries

def call_sarvam_tts(text: str, target_lang: str, speaker: str, pace: float) -> bytes:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY environment variable is not set")
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "target_language_code": target_lang,
        "speaker": speaker,
        "model": "bulbul:v3",
        "properties": {
            "output_audio_codec": "wav",
            "speech_sample_rate": 24000,
            "pace": pace
        }
    }
    print(f"Calling Sarvam TTS API for text of len {len(text)} ({target_lang}, speaker: {speaker}, pace: {pace})...")
    try:
        response = requests.post("https://api.sarvam.ai/text-to-speech", json=payload, headers=headers, timeout=15)
        check_sarvam_response(response)
    except requests.exceptions.HTTPError as err:
        if err.response is not None and is_sarvam_credits_exhausted(err.response.status_code, err.response.text):
            raise SarvamCreditsExhaustedError(SARVAM_CREDITS_MESSAGE) from err
        print(f"Sarvam HTTP Error: {err.response.text if err.response is not None else err}")
        raise
    data = response.json()
    
    if "audios" not in data or not data["audios"]:
        raise ValueError("No audios returned from Sarvam TTS")
    
    audio_base64 = data["audios"][0]
    return base64.b64decode(audio_base64)

def call_sarvam_stt(audio_bytes: bytes, lang_code: str) -> list[dict]:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY environment variable is not set")
        
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }
    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav")
    }
    data = {
        "model": "saaras:v3",
        "language_code": lang_code,
        "with_timestamps": "true"
    }
    print(f"Calling Sarvam STT API for audio alignment ({lang_code})...")
    response = requests.post("https://api.sarvam.ai/speech-to-text", files=files, data=data, headers=headers, timeout=20)
    try:
        check_sarvam_response(response)
    except requests.exceptions.HTTPError as err:
        if err.response is not None and is_sarvam_credits_exhausted(err.response.status_code, err.response.text):
            raise SarvamCreditsExhaustedError(SARVAM_CREDITS_MESSAGE) from err
        raise
    stt_data = response.json()
    
    word_boundaries = []
    if "timestamps" in stt_data:
        ts = stt_data["timestamps"]
        if isinstance(ts, dict) and "words" in ts:
            words_list = ts["words"]
            if isinstance(words_list, list) and len(words_list) > 0:
                if isinstance(words_list[0], dict):
                    for item in words_list:
                        word_boundaries.append({
                            "text": item.get("word", ""),
                            "start": float(item.get("start_time_seconds", 0.0)),
                            "end": float(item.get("end_time_seconds", 0.0))
                        })
                elif isinstance(words_list[0], str):
                    starts = ts.get("start_time_seconds", [])
                    ends = ts.get("end_time_seconds", [])
                    for i in range(min(len(words_list), len(starts), len(ends))):
                        word_boundaries.append({
                            "text": words_list[i],
                            "start": float(starts[i]),
                            "end": float(ends[i])
                        })
    
    print(f"Sarvam STT alignment complete. Extracted {len(word_boundaries)} words.")
    return word_boundaries

def get_wav_duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception as e:
        print(f"Failed to read WAV duration: {e}")
    return 0.0

def detect_speech_boundaries(wav_bytes: bytes, threshold: int = 300) -> tuple[float, float]:
    try:
        import struct
        with wave.open(BytesIO(wav_bytes), "rb") as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            
            if sampwidth != 2 or framerate <= 0 or n_frames <= 0:
                return 0.20, 0.15
                
            raw_data = wav.readframes(n_frames)
            num_samples = len(raw_data) // 2
            samples = struct.unpack(f"{num_samples}h", raw_data)
            
            window_size = int(framerate * 0.01) * n_channels
            if window_size <= 0:
                window_size = 1
                
            first_speech_idx = None
            last_speech_idx = None
            
            for i in range(0, num_samples, window_size):
                chunk = samples[i:i+window_size]
                if any(abs(s) > threshold for s in chunk):
                    first_speech_idx = i
                    break
                    
            for i in range(num_samples - window_size, -1, -window_size):
                chunk = samples[i:i+window_size]
                if any(abs(s) > threshold for s in chunk):
                    last_speech_idx = i + window_size
                    break
                    
            if first_speech_idx is None:
                first_speech_idx = 0
            if last_speech_idx is None:
                last_speech_idx = num_samples
                
            lead_in = (first_speech_idx / n_channels) / framerate
            lead_out = ((num_samples - last_speech_idx) / n_channels) / framerate
            
            return round(lead_in, 3), round(lead_out, 3)
    except Exception as e:
        print(f"Error detecting speech boundaries: {e}")
        return 0.20, 0.15

def adjust_boundaries_to_silence(boundaries: list[dict], wav_bytes: bytes) -> list[dict]:
    if not boundaries:
        return boundaries
    try:
        lead_in, lead_out = detect_speech_boundaries(wav_bytes)
        duration = get_wav_duration(wav_bytes)
        if duration <= 0.0:
            return boundaries
            
        actual_start = lead_in
        actual_end = duration - lead_out
        
        if actual_end <= actual_start:
            return boundaries
            
        orig_start = boundaries[0]["start"]
        orig_end = boundaries[-1]["end"]
        orig_range = orig_end - orig_start
        
        if orig_range <= 0.0:
            usable = actual_end - actual_start
            for i, b in enumerate(boundaries):
                fraction = i / len(boundaries)
                next_fraction = (i + 1) / len(boundaries)
                b["start"] = round(actual_start + usable * fraction, 3)
                b["end"] = round(actual_start + usable * next_fraction, 3)
            return boundaries
            
        actual_range = actual_end - actual_start
        scale = actual_range / orig_range
        
        for b in boundaries:
            b["start"] = round(actual_start + (b["start"] - orig_start) * scale, 3)
            b["end"] = round(actual_start + (b["end"] - orig_start) * scale, 3)
            
        return boundaries
    except Exception as e:
        print(f"Error adjusting boundaries: {e}")
        return boundaries

def generate_fallback_boundaries(text: str, duration: float, lead_in: float = None, lead_out: float = None) -> list[dict]:
    words = text.split()
    if not words:
        return []
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []
    
    if lead_in is None:
        lead_in = min(0.20, duration * 0.05)
    if lead_out is None:
        lead_out = min(0.15, duration * 0.04)
        
    usable_duration = duration - (lead_in + lead_out)
    if usable_duration <= 0.1:
        usable_duration = duration
        lead_in = 0.0
        
    boundaries = []
    current_time = lead_in
    for w in words:
        char_fraction = len(w) / total_chars
        w_duration = usable_duration * char_fraction
        boundaries.append({
            "text": w,
            "start": round(current_time, 3),
            "end": round(current_time + w_duration, 3)
        })
        current_time += w_duration
    return boundaries

async def generate_sarvam_audio_and_boundaries(text: str, language: str, voice: str, rate: str) -> tuple[bytes, list[dict]]:
    cached_audio, cached_boundaries = SarvamMemoryCache.get(text, language, voice, rate)
    if cached_audio is not None:
        print("Using cached Sarvam audio and boundaries.")
        return cached_audio, cached_boundaries

    lang_code = "te-IN" if language == "te-IN" else "en-IN"
    speaker = get_sarvam_speaker(voice, language)
    pace = parse_rate_to_pace(rate)
    
    audio_bytes = await asyncio.to_thread(call_sarvam_tts, text, lang_code, speaker, pace)
    
    lead_in, lead_out = detect_speech_boundaries(audio_bytes)
    duration = get_wav_duration(audio_bytes)
    if duration <= 0.0:
        words_count = len(text.split())
        duration = max(1.0, words_count / 2.5)
        
    try:
        word_boundaries = await asyncio.to_thread(call_sarvam_stt, audio_bytes, lang_code)
        
        # Check if STT returned grouped segments rather than individual words
        input_words = text.split()
        if word_boundaries and len(word_boundaries) < len(input_words) * 0.7:
            print(f"Sarvam STT returned chunked segments ({len(word_boundaries)} segments for {len(input_words)} words). Re-aligning using fallback heuristic...")
            word_boundaries = generate_fallback_boundaries(text, duration, lead_in, lead_out)
        else:
            word_boundaries = adjust_boundaries_to_silence(word_boundaries, audio_bytes)
    except SarvamCreditsExhaustedError:
        raise
    except Exception as stt_err:
        print(f"Sarvam STT failed: {stt_err}. Using fallback heuristic alignment...")
        word_boundaries = generate_fallback_boundaries(text, duration, lead_in, lead_out)
        
    SarvamMemoryCache.set(text, language, voice, rate, audio_bytes, word_boundaries)
    return audio_bytes, word_boundaries

active_doc = None
active_analyzer = None
active_doc_lock = threading.Lock()

MIN_MODEL_BYTES = 50 * 1024 * 1024  # reject truncated/corrupt GGUF files

def resolve_model_path(model_paths: list[Path]) -> Path | None:
    for path in model_paths:
        if path.exists() and path.stat().st_size >= MIN_MODEL_BYTES:
            return path
        if path.exists():
            print(f"Skipping corrupt/incomplete model at {path} ({path.stat().st_size} bytes)")
    return None

# Shared LLM state
def get_selected_model_info() -> tuple[str, str, list[Path]]:
    _size, filename, url = get_local_qwen_gguf_info()
    model_paths = [BASE_DIR / "models" / filename]
    return filename, url, model_paths

class LLMState:
    _llm = None
    _model_downloading = False
    _lock = threading.Lock()

    @classmethod
    def get_llm(cls):
        if cls._llm is not None:
            return cls._llm

        filename, url, model_paths = get_selected_model_info()

        found_path = resolve_model_path(model_paths)

        if not found_path:
            if cls._model_downloading:
                raise Exception("Local LLM model is still downloading in the background. Please wait a moment...")
            else:
                start_model_download()
                raise Exception(f"Local LLM model ({filename}) is not present. Download started in background. Please try again in a moment...")

        from llama_cpp import Llama
        n_threads = min(8, os.cpu_count() or 4)
        print(f"Loading Qwen LLM model from {found_path} with {n_threads} threads...")
        cls._llm = Llama(model_path=str(found_path), n_ctx=4096, n_threads=n_threads, verbose=False)
        print("Model loaded successfully.")
        return cls._llm

INDICTRANS_LANG_MAP = {
    "te-IN": "tel_Telu",
    "hi-IN": "hin_Deva",
    "ta-IN": "tam_Taml",
    "kn-IN": "kan_Knda",
    "ml-IN": "mal_Mlym",
    "mr-IN": "mar_Deva",
}

# NLLB-200 (facebook/nllb-200-1.3B) — works on CPU, no C extensions required.
# Replaces IndicTrans2/IndicTransToolkit which requires MSVC build tools on Windows.
NLLB_MODEL_NAME = "facebook/nllb-200-1.3B"

class IndicTransState:
    _model = None
    _tokenizer = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        if cls._model is not None:
            return cls._model, cls._tokenizer
        with cls._lock:
            if cls._model is not None:
                return cls._model, cls._tokenizer
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    f"transformers not installed: {e}. "
                    "Run: pip install --target server/vendor transformers sentencepiece"
                )
            print(f"Loading NLLB-200 translation model ({NLLB_MODEL_NAME}) — first load downloads ~2.5GB and may take a few minutes...")
            cls._tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
            cls._model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
            print("NLLB-200 translation model loaded successfully.")
            return cls._model, cls._tokenizer

def translate_strings_indictrans2(texts: list[str], target_lang: str = "tel_Telu") -> list[str]:
    """Translate a list of English strings to target_lang using NLLB-200 locally."""
    import torch
    import time
    model, tokenizer = IndicTransState.get_model()
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    if target_lang_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown target language code for NLLB-200: {target_lang}")
    # Dynamic max_length: 3× the longest input's word count + 20 buffer, capped at 200.
    # Narration is now capped at 70 words → output ~230 tokens → 200 is sufficient.
    input_max_words = max(len(t.split()) for t in texts)
    dynamic_max_length = min(input_max_words * 3 + 20, 200)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            num_beams=2,
            max_new_tokens=dynamic_max_length,
            length_penalty=1.0,
            early_stopping=True,
        )
    elapsed = time.perf_counter() - t0
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"NLLB-200 translated {len(texts)} strings to {target_lang} in {elapsed:.1f}s (max_len={dynamic_max_length})")
    return result

def analyze_text_semantic_local(text: str, is_digest: bool = False, language: str = "en-US", page_layout: str = "plain") -> dict:
    llm = LLMState.get_llm()
    if llm is None:
        raise Exception("LLM model not available yet.")

    messages = build_analyze_messages(text, is_digest=is_digest, page_layout=page_layout)

    with LLMState._lock:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
            repeat_penalty=1.1,
        )
        output_text = response["choices"][0]["message"]["content"].strip()

    try:
        return extract_analysis_json(output_text)
    except Exception as e:
        print("--- LLM OUTPUT PARSE FAILURE ---")
        print(output_text)
        print(f"Error: {e}")
        print("--------------------------------")
        raise Exception(f"Failed to parse JSON response from local LLM: {e}") from e


def run_semantic_analysis(
    text: str,
    is_digest: bool = False,
    language: str = "en-US",
    page_layout: str = "plain",
) -> dict:
    provider = resolve_provider("LLM_PROVIDER", "auto")

    if should_use_kaggle(provider):
        try:
            print(f"Calling Kaggle GPU /analyze (is_digest={is_digest}) …")
            return kaggle_analyze(text, is_digest=is_digest, page_layout=page_layout)
        except Exception as kaggle_err:
            if provider == "kaggle":
                raise
            print(f"Kaggle analyze failed ({kaggle_err}); falling back to local LLM …")

    return analyze_text_semantic_local(
        text, is_digest=is_digest, language=language, page_layout=page_layout
    )


def translate_strings_for_deck(texts: list[str], target_lang: str) -> list[str]:
    provider = resolve_provider("TRANSLATE_PROVIDER", "auto")

    if should_use_kaggle(provider):
        try:
            print(f"Calling Kaggle GPU /translate ({len(texts)} strings) …")
            return kaggle_translate(texts, target_lang=target_lang)
        except Exception as kaggle_err:
            print(f"Kaggle translate failed ({kaggle_err}); falling back to local NLLB …")

    return translate_strings_indictrans2(texts, target_lang)

# Request/Response Pydantic schemas
class TTSRequest(BaseModel):
    text: str
    language: str = "en-US"
    voice: str = ""
    rate: str = "+0%"

class AnalyzeRequest(BaseModel):
    text: str
    is_digest: bool = False
    language: str = "en-US"
    page_layout: str = "plain"

class TranslateDeckRequest(BaseModel):
    deck: dict

class TranscribeRequest(BaseModel):
    audio: str
    text: str
    filename: str = "audio.wav"

@app.get("/health")
def health_check():
    ks = kaggle_status()
    return {
        "ok": True,
        "service": "veda-tts",
        "auth_required": bool(VEDA_API_KEY),
        "providers": {
            "llm": resolve_provider("LLM_PROVIDER", "local"),
            "translate": resolve_provider("TRANSLATE_PROVIDER", "local"),
            "layout": resolve_provider("LAYOUT_PROVIDER", "local"),
            "upscale": resolve_provider("UPSCALE_PROVIDER", "local"),
            "image_gen": resolve_provider("IMAGE_GEN_PROVIDER", "auto"),
        },
        "kaggle": redact_kaggle_status(ks),
    }

async def generate_audio_and_boundaries(text: str, voice: str, rate: str) -> tuple[bytes, list[dict]]:
    stream = edge_tts.Communicate(text=text, voice=voice, rate=rate, boundary="WordBoundary")
    audio_buffer = BytesIO()
    word_boundaries = []

    async for chunk in stream.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            word_boundaries.append({
                "text": chunk["text"],
                "start": chunk["offset"] / 10000000.0,
                "end": (chunk["offset"] + chunk["duration"]) / 10000000.0
            })

    return audio_buffer.getvalue(), word_boundaries

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    text = req.text.strip()
    language = req.language.strip() or "en-US"
    voice = req.voice.strip() or VOICE_MAP.get(language, VOICE_MAP["en-US"])
    rate = req.rate.strip() or "+0%"

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        validate_text_length(text, MAX_TTS_TEXT_CHARS, "TTS text")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        sarvam_credits_exhausted = False
        if SARVAM_API_KEY:
            print(f"Generating Sarvam TTS for {len(text)} chars (voice: {voice}, rate: {rate})...")
            try:
                audio_bytes, word_boundaries = await generate_sarvam_audio_and_boundaries(text, language, voice, rate)
            except SarvamCreditsExhaustedError as sarvam_exc:
                print(f"Sarvam credits exhausted: {sarvam_exc}. Falling back to edge-tts...")
                sarvam_credits_exhausted = True
                audio_bytes, word_boundaries = await generate_audio_and_boundaries(text, voice, rate)
            except Exception as sarvam_exc:
                print(f"Sarvam TTS generation failed: {sarvam_exc}. Falling back to edge-tts...")
                audio_bytes, word_boundaries = await generate_audio_and_boundaries(text, voice, rate)
        else:
            print(f"Generating edge-tts for {len(text)} chars (voice: {voice}, rate: {rate})...")
            audio_bytes, word_boundaries = await generate_audio_and_boundaries(text, voice, rate)
        
        print("TTS generation successful")
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "audio": audio_base64,
            "word_boundaries": word_boundaries,
            "sarvam_credits_exhausted": sarvam_credits_exhausted,
        }
    except Exception as exc:
        print(f"TTS generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/tts_stream")
async def text_to_speech_stream(text: str, language: str = "en-US", voice: str = "", rate: str = "+0%"):
    text = text.strip()
    language = language.strip() or "en-US"
    voice = voice.strip() or VOICE_MAP.get(language, VOICE_MAP["en-US"])
    rate = rate.strip() or "+0%"

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        validate_text_length(text, MAX_TTS_TEXT_CHARS, "TTS text")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def audio_generator():
        try:
            if SARVAM_API_KEY:
                print(f"Streaming Sarvam TTS for {len(text)} chars (voice: {voice}, rate: {rate})...")
                try:
                    audio_bytes, _ = await generate_sarvam_audio_and_boundaries(text, language, voice, rate)
                    yield audio_bytes
                    return
                except SarvamCreditsExhaustedError as sarvam_exc:
                    print(f"Sarvam credits exhausted: {sarvam_exc}. Falling back to edge-tts...")
                except Exception as sarvam_exc:
                    print(f"Sarvam streaming failed: {sarvam_exc}. Falling back to edge-tts...")
            
            print(f"Streaming edge-tts for {len(text)} chars (voice: {voice}, rate: {rate})...")
            stream = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            async for chunk in stream.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            print("Streaming TTS generation completed")
        except Exception as exc:
            print(f"Streaming TTS generator failed: {exc}")

    media_type = "audio/wav" if SARVAM_API_KEY else "audio/mpeg"
    return StreamingResponse(audio_generator(), media_type=media_type)

@app.post("/tts_boundaries")
async def text_to_speech_boundaries(req: TTSRequest):
    text = req.text.strip()
    language = req.language.strip() or "en-US"
    voice = req.voice.strip() or VOICE_MAP.get(language, VOICE_MAP["en-US"])
    rate = req.rate.strip() or "+0%"

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        validate_text_length(text, MAX_TTS_TEXT_CHARS, "TTS text")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        word_boundaries = []
        use_fallback = False
        sarvam_credits_exhausted = False
        
        if SARVAM_API_KEY:
            print(f"Generating Sarvam boundaries for {len(text)} chars (voice: {voice}, rate: {rate})...")
            try:
                _, word_boundaries = await generate_sarvam_audio_and_boundaries(text, language, voice, rate)
            except SarvamCreditsExhaustedError as sarvam_exc:
                print(f"Sarvam credits exhausted: {sarvam_exc}. Falling back to edge-tts...")
                sarvam_credits_exhausted = True
                use_fallback = True
            except Exception as sarvam_exc:
                print(f"Sarvam boundaries failed: {sarvam_exc}. Falling back to edge-tts...")
                use_fallback = True
        else:
            use_fallback = True
            
        if use_fallback:
            print(f"Generating edge-tts boundaries for {len(text)} chars (voice: {voice}, rate: {rate})...")
            stream = edge_tts.Communicate(text=text, voice=voice, rate=rate, boundary="WordBoundary")
            async for chunk in stream.stream():
                if chunk["type"] == "WordBoundary":
                    word_boundaries.append({
                        "text": chunk["text"],
                        "start": chunk["offset"] / 10000000.0,
                        "end": (chunk["offset"] + chunk["duration"]) / 10000000.0
                    })
        print(f"TTS boundary generation successful: {len(word_boundaries)} words")
        return {
            "word_boundaries": word_boundaries,
            "sarvam_credits_exhausted": sarvam_credits_exhausted,
        }
    except Exception as exc:
        print(f"TTS boundary generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

def extract_strings(obj, paths, current_path=[], skip_keys: set | None = None):
    skip_keys = skip_keys or set()
    if isinstance(obj, str):
        paths.append((current_path, obj))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            extract_strings(item, paths, current_path + [idx], skip_keys)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in skip_keys:
                continue
            if key in ["title", "subtitle", "summary", "narration", "body"]:
                extract_strings(value, paths, current_path + [key], skip_keys)
            elif key in ["highlights", "supportingPoints", "topics"]:
                extract_strings(value, paths, current_path + [key], skip_keys)

def set_by_path(obj, path, value):
    current = obj
    for step in path[:-1]:
        current = current[step]
    current[path[-1]] = value

def translate_strings_sarvam(texts: list[str], api_key: str) -> list[str]:
    delimiter = " |#| "
    combined = delimiter.join(texts)
    
    url = "https://api.sarvam.ai/translate"
    payload = {
        "input": combined,
        "source_language_code": "en-IN",
        "target_language_code": "te-IN",
        "model": "sarvam-translate:v1"
    }
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        check_sarvam_response(response)
        translated_combined = response.json()["translated_text"]
        translated_texts = re.split(r'\s*\|#\|\s*', translated_combined)
        if len(translated_texts) == len(texts):
            return [t.strip() for t in translated_texts]
        else:
            print(f"Translation split mismatch: got {len(translated_texts)}, expected {len(texts)}. Translating individually...")
    except SarvamCreditsExhaustedError:
        raise
    except Exception as e:
        print(f"Batch translation failed: {e}. Translating individually...")
        
    translated_texts = []
    for t in texts:
        try:
            payload["input"] = t
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            check_sarvam_response(response)
            translated_texts.append(response.json()["translated_text"].strip())
        except SarvamCreditsExhaustedError:
            raise
        except Exception as err:
            print(f"Failed to translate '{t}': {err}")
            translated_texts.append(t)
    return translated_texts

def translate_deck_fields(deck: dict) -> dict:
    translated_deck = json.loads(json.dumps(deck))
    # Snapshot English titles before in-place translation
    for topic in translated_deck.get("topics", []):
        topic["title_en"] = topic.get("title", "")
    string_paths: list[tuple[list, str]] = []
    extract_strings(translated_deck, string_paths)
    if not string_paths:
        return translated_deck

    texts_to_translate = [text for _, text in string_paths]
    try:
        translated_texts = translate_strings_for_deck(texts_to_translate, "tel_Telu")
        print("translate_deck: translation succeeded.")
    except Exception as it_err:
        print(f"translate_deck: translation failed ({it_err}), falling back to Sarvam...")
        if not SARVAM_API_KEY:
            raise ValueError("Translation unavailable and SARVAM_API_KEY is not set") from it_err
        translated_texts = translate_strings_sarvam(texts_to_translate, SARVAM_API_KEY)
    for (path, _), trans_val in zip(string_paths, translated_texts):
        set_by_path(translated_deck, path, trans_val)

    if translated_deck.get("narration"):
        translated_deck["narration_te"] = translated_deck["narration"]
    for topic in translated_deck.get("topics", []):
        title = topic.get("title", "")
        body = topic.get("summary", topic.get("body", ""))
        combined = f"{title}. {body}".strip()
        if combined:
            topic["narration_te"] = combined

    translated_deck["isTelugu"] = True
    return translated_deck

class TranslateRequest(BaseModel):
    texts: list[str]
    source_lang: str = "eng_Latn"
    target_lang: str = "tel_Telu"

@app.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    """Translate strings via Kaggle GPU when enabled, else local NLLB with Sarvam fallback."""
    if not req.texts:
        return {"translations": []}
    try:
        result = await asyncio.to_thread(translate_strings_for_deck, req.texts, req.target_lang)
        engine = "kaggle" if should_use_kaggle(resolve_provider("TRANSLATE_PROVIDER", "local")) else "nllb"
        return {"translations": result, "engine": engine}
    except Exception as e:
        print(f"Translation failed: {e}. Falling back to Sarvam...")
        if not SARVAM_API_KEY:
            raise HTTPException(status_code=500, detail=f"Translation failed and SARVAM_API_KEY not set: {e}")
        try:
            result = await asyncio.to_thread(translate_strings_sarvam, req.texts, SARVAM_API_KEY)
            return {"translations": result, "engine": "sarvam_fallback"}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))

@app.post("/translate_deck")
def translate_deck(req: TranslateDeckRequest):
    try:
        print("Translating deck fields into Telugu...")
        return translate_deck_fields(req.deck)
    except SarvamCreditsExhaustedError:
        raise sarvam_credits_http_exception()
    except Exception as exc:
        print(f"Deck translation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

class TranslateFieldsRequest(BaseModel):
    deck: dict
    fields: list[str] = ["highlights", "supportingPoints"]
    language: str = "te-IN"

@app.post("/translate_fields")
async def translate_fields_endpoint(req: TranslateFieldsRequest):
    """Translate secondary deck fields (highlights, supportingPoints) lazily after the slide renders.
    Only the requested fields are extracted and translated; the rest of the deck is returned as-is."""
    tgt_lang_code = INDICTRANS_LANG_MAP.get(req.language, "tel_Telu")
    import copy
    deck_copy = copy.deepcopy(req.deck)

    # Extract only the requested fields
    only_keys = set(req.fields)
    string_paths: list[tuple[list, str]] = []
    for key in req.fields:
        if key in deck_copy:
            extract_strings({key: deck_copy[key]}, string_paths, skip_keys=set())

    if not string_paths:
        return deck_copy

    texts = [t for _, t in string_paths]
    print(f"/translate_fields: translating {len(texts)} strings ({req.fields}) to {req.language}...")
    try:
        translated = await asyncio.to_thread(translate_strings_for_deck, texts, tgt_lang_code)
        for (path, _), trans_val in zip(string_paths, translated):
            # path[0] is the field key — route back into deck_copy
            set_by_path(deck_copy, path, trans_val)
        print(f"/translate_fields: done")
    except Exception as e:
        print(f"/translate_fields: translation failed ({e}), returning untranslated fields")

    return deck_copy

@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        validate_text_length(text, MAX_ANALYZE_TEXT_CHARS, "Analysis text")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        mode = "digest" if req.is_digest else "single-topic"
        llm_provider = resolve_provider("LLM_PROVIDER", "local")
        backend = "kaggle" if should_use_kaggle(llm_provider) else "local"
        print(f"Starting LLM analysis ({mode}, backend={backend}, language: {req.language}) for {len(text)} chars...")

        # We always generate the semantic structure in English first for high accuracy
        analysis = run_semantic_analysis(
            text,
            is_digest=req.is_digest,
            language="en-US",
            page_layout=req.page_layout,
        )
        sarvam_credits_exhausted = False
        
        # If Telugu is selected, translate via IndicTrans2 (local) with Sarvam as fallback
        if req.language == "te-IN":
            tgt_lang_code = INDICTRANS_LANG_MAP.get(req.language, "tel_Telu")
            print(f"Translating slide content into Telugu (IndicTrans2 → Sarvam fallback)...")
            try:
                # Snapshot English titles before in-place translation so the frontend
                # can still match images by English title (title_en) even after Telugu
                # titles overwrite the title field.
                for topic in analysis.get("topics", []):
                    topic["title_en"] = topic.get("title", "")

                # Translate only priority fields immediately so the slide renders fast.
                # highlights and supportingPoints are translated lazily via /translate_fields.
                SECONDARY_FIELDS = {"highlights", "supportingPoints"}
                string_paths: list[tuple[list, str]] = []
                extract_strings(analysis, string_paths, skip_keys=SECONDARY_FIELDS)
                if string_paths:
                    texts_to_translate = [t for _, t in string_paths]
                    try:
                        translated = translate_strings_for_deck(texts_to_translate, tgt_lang_code)
                        print("Translation succeeded.")
                    except Exception as it_err:
                        print(f"IndicTrans2 failed ({it_err}), falling back to Sarvam...")
                        if not SARVAM_API_KEY:
                            raise Exception("IndicTrans2 unavailable and SARVAM_API_KEY not set.")
                        translated = translate_strings_sarvam(texts_to_translate, SARVAM_API_KEY)
                    for (path, _), trans_val in zip(string_paths, translated):
                        set_by_path(analysis, path, trans_val)

                # narration_te for karaoke/TTS (fields are already Telugu after in-place translation)
                if analysis.get("narration"):
                    analysis["narration_te"] = analysis["narration"]
                for topic in analysis.get("topics", []):
                    title = topic.get("title", "")
                    body = topic.get("summary", topic.get("body", ""))
                    combined = f"{title}. {body}".strip()
                    if combined:
                        topic["narration_te"] = combined

                print("Translation to Telugu complete.")
            except SarvamCreditsExhaustedError as trans_err:
                sarvam_credits_exhausted = True
                print(f"Translation to Telugu failed — Sarvam credits exhausted: {trans_err}")
            except Exception as trans_err:
                print(f"Translation to Telugu failed: {trans_err}.")
        
        if sarvam_credits_exhausted:
            analysis["sarvam_credits_exhausted"] = True

        print("LLM analysis successful")
        return analysis
    except Exception as exc:
        print(f"LLM analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/upload_pdf")
async def upload_pdf(request: Request):
    global active_doc, active_analyzer
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
    
    with active_doc_lock:
        try:
            if active_doc is not None:
                active_doc.close()
                active_doc = None
                active_analyzer = None
            
            pdf_path = BASE_DIR / "active_doc.pdf"
            pdf_path.write_bytes(pdf_bytes)
            
            active_doc = fitz.open(str(pdf_path))
            active_analyzer = PDFMetadataAnalyzer(active_doc)
            print(f"Loaded active PDF document: {pdf_path} ({len(pdf_bytes)} bytes, {len(active_doc)} pages)")
            if should_use_kaggle(resolve_provider("LAYOUT_PROVIDER", "local")):
                try:
                    kaggle_upload_pdf(pdf_bytes)
                    print("Mirrored PDF to Kaggle GPU server for layout extraction.")
                except Exception as kaggle_err:
                    layout_mode = resolve_provider("LAYOUT_PROVIDER", "local")
                    if layout_mode == "kaggle":
                        raise HTTPException(status_code=502, detail=f"Kaggle PDF upload failed: {kaggle_err}") from kaggle_err
                    print(f"Kaggle PDF mirror failed ({kaggle_err}); using local layout extraction.")
            return {"ok": True, "num_pages": len(active_doc)}
        except Exception as e:
            print(f"Failed to load PDF document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load PDF: {e}")

@app.get("/page_layout")
def get_page_layout(page: int):
    global active_doc, active_analyzer
    layout_provider = resolve_provider("LAYOUT_PROVIDER", "local")

    if should_use_kaggle(layout_provider):
        try:
            print(f"GET /page_layout?page={page} via Kaggle …")
            result = kaggle_page_layout(page)
            try:
                if active_doc is None:
                    pdf_path = BASE_DIR / "active_doc.pdf"
                    if pdf_path.exists():
                        with active_doc_lock:
                            if active_doc is None:
                                active_doc = fitz.open(str(pdf_path))
                                active_analyzer = PDFMetadataAnalyzer(active_doc)
                if active_doc is not None:
                    if active_analyzer is None:
                        active_analyzer = PDFMetadataAnalyzer(active_doc)
                    with active_doc_lock:
                        local = extract_page_layout(
                            active_doc,
                            active_analyzer,
                            page,
                            pdfplumber_available=_PDFPLUMBER_AVAILABLE,
                        )
                    result["images"] = merge_background_images(
                        result.get("images") or [],
                        local.get("images") or [],
                    )
            except Exception as enrich_err:
                print(f"Background merge skipped for page {page} ({enrich_err})")
            return result
        except Exception as kaggle_err:
            if layout_provider == "kaggle":
                raise HTTPException(status_code=502, detail=str(kaggle_err)) from kaggle_err
            print(f"Kaggle layout failed ({kaggle_err}); using local extraction …")

    if active_doc is None:
        pdf_path = BASE_DIR / "active_doc.pdf"
        if pdf_path.exists():
            try:
                active_doc = fitz.open(str(pdf_path))
                active_analyzer = PDFMetadataAnalyzer(active_doc)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to open saved PDF: {e}")
        else:
            raise HTTPException(status_code=400, detail="No active PDF document uploaded")

    if active_analyzer is None and active_doc is not None:
        active_analyzer = PDFMetadataAnalyzer(active_doc)

    if page < 1 or page > len(active_doc):
        raise HTTPException(status_code=400, detail=f"Invalid page number {page}")

    with active_doc_lock:
        try:
            result = extract_page_layout(
                active_doc, active_analyzer, page, pdfplumber_available=_PDFPLUMBER_AVAILABLE
            )
            print(f"Local layout extracted {len(result.get('lines', []))} lines for page {page}")
            return result
        except Exception as e:
            print(f"Failed to extract page layout for page {page}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return {
            "text": "",
            "word_boundaries": [],
            "error": "OpenAI API Key is not set in the server environment. Please configure it in your environment variables or server/.env."
        }

    audio_data = req.audio.strip()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Audio base64 data is required")

    try:
        audio_bytes = base64.b64decode(audio_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 audio: {exc}")

    import tempfile
    ext = os.path.splitext(req.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        from openai import OpenAI
        print(f"Calling OpenAI Whisper API for file {tmp_path} ({len(audio_bytes)} bytes)...")
        client = OpenAI(api_key=openai_key)

        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )

        word_boundaries = []
        words_list = getattr(transcript, "words", []) or []
        for w in words_list:
            if isinstance(w, dict):
                word_text = w.get("word")
                word_start = w.get("start")
                word_end = w.get("end")
            else:
                word_text = getattr(w, "word", None)
                word_start = getattr(w, "start", None)
                word_end = getattr(w, "end", None)

            if word_text is not None and word_start is not None and word_end is not None:
                word_boundaries.append({
                    "text": word_text,
                    "start": float(word_start),
                    "end": float(word_end)
                })

        print(f"Whisper transcription complete: {len(word_boundaries)} words extracted.")
        return {
            "text": getattr(transcript, "text", ""),
            "word_boundaries": word_boundaries
        }
    except Exception as exc:
        print(f"Whisper transcription failed: {exc}")
        return {
            "text": "",
            "word_boundaries": [],
            "error": f"Whisper API call failed: {exc}"
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

@app.post("/upscale_image")
async def upscale_image(request: Request):
    """
    Upscale or regenerate a base64-encoded image.

    Decorative tiny images (either dim < 400px) route to Kaggle SDXL img2img when
    IMAGE_GEN_PROVIDER allows and Kaggle is reachable; larger decorative and all
    informational images use Lanczos.

    Request body (JSON):
        {
          "image": "<base64 or data-URL>",
          "format": "png" | "jpeg",
          "page_text": "...",
          "page_width": 612,
          "page_height": 792,
          "page_word_count": 120,
          "is_image_primary": false,
          "image_meta": { "x", "y", "w", "h", "overlapWordCount", "url" }
        }

    Response (JSON):
        { "image": "<data-URL>", "method": "sdxl" | "lanczos_x2" | ... }
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Pillow not installed: {exc}") from exc

    body = await request.json()
    image_field = body.get("image", "")
    fmt = body.get("format", "png").lower()
    page_text = (body.get("page_text") or "").strip()
    page_width = body.get("page_width")
    page_height = body.get("page_height")
    page_word_count = int(body.get("page_word_count") or 0)
    is_image_primary = bool(body.get("is_image_primary"))
    image_meta = body.get("image_meta") or {}

    if not image_field:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body.")

    b64 = image_field
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        validate_b64_payload(b64)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    img_bytes = base64.b64decode(b64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    w, h = pil_img.width, pil_img.height
    use_img2img = False
    route_reason = "lanczos"
    try:
        use_img2img, route_reason = should_route_img2img(
            pil_img,
            image_meta,
            width=w,
            height=h,
            page_width=float(page_width or 0),
            page_height=float(page_height or 0),
            is_image_primary=is_image_primary,
        )
    except Exception as meta_err:
        print(f"/upscale_image: routing check failed ({meta_err}) — Lanczos")

    image_gen_provider = resolve_provider("IMAGE_GEN_PROVIDER", "auto")
    use_kaggle_gen = use_img2img and should_use_kaggle(image_gen_provider)
    if use_img2img and not use_kaggle_gen:
        print(f"/upscale_image: {route_reason} → img2img skipped (Kaggle off/unreachable)")
    elif use_img2img:
        print(f"/upscale_image: {route_reason} → Kaggle img2img ({w}×{h} px)")
    elif route_reason == "informational":
        print(f"/upscale_image: informational ({w}×{h} px) → Lanczos")
    elif route_reason == "background":
        print(f"/upscale_image: background ({w}×{h} px) → Lanczos")
    elif route_reason == "too_large":
        print(f"/upscale_image: subject too large ({w}×{h} px) → Lanczos")

    if use_kaggle_gen:
        loop = asyncio.get_event_loop()
        try:
            gen_result = await loop.run_in_executor(
                None,
                lambda: kaggle_generate_image(image_field, page_text=page_text),
            )
            out_url = gen_result.get("image", "")
            if out_url:
                method = gen_result.get("method", "ssd1b_img2img")
                print(f"/upscale_image: img2img output [{method}]")
                return {
                    "image": out_url,
                    "method": method,
                    "prompt": gen_result.get("prompt"),
                    "caption": gen_result.get("caption"),
                }
        except Exception as gen_err:
            print(f"/upscale_image: Kaggle generate failed ({gen_err}) — Lanczos fallback")

    print(f"/upscale_image: input {w}×{h} px (Lanczos), fmt={fmt}, route={route_reason}")

    loop = asyncio.get_event_loop()
    out_pil, method = await loop.run_in_executor(None, lambda: run_face_aware_upscale(pil_img))

    buf = BytesIO()
    if fmt == "jpeg":
        out_pil.save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"
    else:
        out_pil.save(buf, format="PNG")
        mime = "image/png"
    out_b64 = base64.b64encode(buf.getvalue()).decode()
    out_w, out_h = out_pil.width, out_pil.height
    print(f"/upscale_image: output {out_w}×{out_h} px [{method}], {len(out_b64)//1024} KB b64")

    return {"image": f"data:{mime};base64,{out_b64}", "method": method}


def start_model_download() -> None:
    filename, url, model_paths = get_selected_model_info()

    if resolve_model_path(model_paths):
        print("Valid local LLM model already present.")
        return

    target_dir = BASE_DIR / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    def download():
        import urllib.request
        try:
            LLMState._model_downloading = True
            print(f"Downloading local LLM model from {url} to {target_path}...")
            urllib.request.urlretrieve(url, str(target_path))
            print("Download completed successfully.")
        except Exception as e:
            print(f"Download failed: {e}")
        finally:
            LLMState._model_downloading = False

    t = threading.Thread(target=download)
    t.daemon = True
    t.start()

def _prewarm_nllb() -> None:
    """Load the NLLB translation model into RAM at startup so the first /analyze
    call in Telugu mode does not pay the model-load penalty (~120s)."""
    try:
        IndicTransState.get_model()
        print("NLLB-200 pre-warm complete.")
    except Exception as e:
        print(f"NLLB-200 pre-warm failed (will load on first use): {e}")

if __name__ == "__main__":
    import uvicorn
    # Pre-warm NLLB in background so it's ready before the first Telugu request
    t = threading.Thread(target=_prewarm_nllb, daemon=True)
    t.start()
    start_model_download()
    print("Starting FastAPI Uvicorn server on port 8765...")
    uvicorn.run("tts_server:app", host="127.0.0.1", port=8765, reload=False)
