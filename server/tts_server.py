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
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import requests

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
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ModuleNotFoundError:
    # Fallback to standard library if pip installation hasn't fully registered in-process yet
    print("FastAPI is not installed in the target vendor path. Installing requirements...")
    raise

app = FastAPI(title="TTS & Slide Analysis Server")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def reconstruct_line_text(spans) -> str:
    line_text = ""
    sorted_spans = sorted(spans, key=lambda s: s.get("origin", (0, 0))[0])
    for s in sorted_spans:
        text = s.get("text", "")
        if not text:
            continue
        if not line_text:
            line_text = text
        else:
            needs_space = not line_text.endswith('-') and not text.startswith(',') and not text.startswith('.')
            line_text += " " + text if needs_space else text
    return line_text.strip()

class PDFMetadataAnalyzer:
    def __init__(self, doc):
        self.running_headers_footers = set()
        self._analyze(doc)
        
    def _analyze(self, doc):
        from collections import Counter
        top_lines = []
        bottom_lines = []
        
        for page in doc:
            height = page.rect.height
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") == 0:
                    for l in b.get("lines", []):
                        spans = l.get("spans", [])
                        if not spans:
                            continue
                        line_text = reconstruct_line_text(spans)
                        if not line_text:
                            continue
                        y = spans[0]["origin"][1]
                        if y < height * 0.10:
                            top_lines.append(line_text)
                        elif y > height * 0.90:
                            bottom_lines.append(line_text)
                            
        top_counts = Counter(top_lines)
        bottom_counts = Counter(bottom_lines)
        
        total_pages = len(doc)
        min_count = max(2, min(3, total_pages))
        
        for text, count in top_counts.items():
            if count >= min_count:
                self.running_headers_footers.add(text)
                
        for text, count in bottom_counts.items():
            if count >= min_count:
                self.running_headers_footers.add(text)

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
    size = os.environ.get("LOCAL_LLM_SIZE", "0.5B").strip().upper()
    if size == "1.5B":
        filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    elif size == "3B":
        filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
        url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    else:  # Default to 0.5B
        filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        
    model_paths = [
        BASE_DIR / "models" / filename,
    ]
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
            max_length=dynamic_max_length,
            length_penalty=1.0,
            early_stopping=True,
        )
    elapsed = time.perf_counter() - t0
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"NLLB-200 translated {len(texts)} strings to {target_lang} in {elapsed:.1f}s (max_len={dynamic_max_length})")
    return result

ESRGAN_MODEL_PATH = BASE_DIR / "models" / "RealESRGAN_x4plus.pth"
ESRGAN_TILE_SIZE = 256   # Process this many pixels at a time; keeps RAM ~1GB on CPU
ESRGAN_TILE_PAD  = 10    # Overlap between tiles to avoid edge seams

GFPGAN_MODEL_PATH = BASE_DIR / "models" / "GFPGANv1.4.pth"

class ESRGANState:
    """Lazy-load singleton for the Real-ESRGAN 4× upscaler."""
    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        if cls._model is not None:
            return cls._model
        with cls._lock:
            if cls._model is not None:
                return cls._model
            import torch
            sys.path.insert(0, str(BASE_DIR))
            from rrdb_net import RRDBNet
            print("Loading Real-ESRGAN model (RealESRGAN_x4plus.pth) …")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            state = torch.load(str(ESRGAN_MODEL_PATH), map_location="cpu")
            weights = state.get("params_ema", state.get("params", state))
            model.load_state_dict(weights, strict=True)
            model.eval()
            print("Real-ESRGAN model loaded.")
            cls._model = model
            return cls._model


class GFPGANState:
    """Lazy-load singleton for the GFPGAN face restoration enhancer.
    Only initialised when the model file is present — silently skipped otherwise."""
    _enhancer = None
    _unavailable = False
    _lock = threading.Lock()

    @classmethod
    def get_enhancer(cls):
        if cls._unavailable:
            return None
        if cls._enhancer is not None:
            return cls._enhancer
        with cls._lock:
            if cls._enhancer is not None or cls._unavailable:
                return cls._enhancer
            if not GFPGAN_MODEL_PATH.exists():
                print("GFPGANv1.4.pth not found — face enhancement disabled.")
                cls._unavailable = True
                return None
            try:
                from gfpgan import GFPGANer
                print("Loading GFPGAN face restoration model…")
                cls._enhancer = GFPGANer(
                    model_path=str(GFPGAN_MODEL_PATH),
                    upscale=1,          # we apply it after ESRGAN so don't scale again
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,  # background already handled by ESRGAN
                )
                print("GFPGAN model loaded.")
            except Exception as e:
                print(f"GFPGAN failed to load ({e}) — face enhancement disabled.")
                cls._unavailable = True
            return cls._enhancer


def _gfpgan_enhance_faces(pil_img) -> "PIL.Image.Image":
    """Run GFPGAN face restoration on a PIL image. Returns enhanced PIL image,
    or the original image unchanged if no faces are detected or GFPGAN is unavailable."""
    import numpy as np
    from PIL import Image
    enhancer = GFPGANState.get_enhancer()
    if enhancer is None:
        return pil_img
    try:
        img_bgr = np.array(pil_img)[:, :, ::-1]  # RGB → BGR for OpenCV/GFPGAN
        _, _, restored_img = enhancer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        if restored_img is None:
            return pil_img
        return Image.fromarray(restored_img[:, :, ::-1])  # BGR → RGB
    except Exception as e:
        print(f"GFPGAN face enhancement failed ({e}) — using ESRGAN output.")
        return pil_img


def _lanczos_upscale(pil_img, target_scale: int = 2):
    """
    Upscale a PIL image using Lanczos resampling + Unsharp Mask sharpening.

    This is the primary upscaler for PDF-extracted images: it produces clean,
    artefact-free results because PDF images are already at decent quality and
    only need crisp interpolation, not AI-hallucinated detail.

    Steps:
      1. Lanczos ×target_scale for clean high-quality interpolation
      2. Unsharp Mask to restore the slight softness Lanczos introduces
    """
    from PIL import Image, ImageFilter

    new_w = pil_img.width * target_scale
    new_h = pil_img.height * target_scale
    upscaled = pil_img.resize((new_w, new_h), Image.LANCZOS)
    # Mild unsharp mask: radius=1.5, percent=60, threshold=3
    sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=3))
    return sharpened


def _esrgan_upscale(img_np):
    """
    Run Real-ESRGAN 4× upscaling on a uint8 HWC RGB numpy array.
    Reserved for genuinely tiny / highly degraded images.
    Returns a uint8 HWC RGB numpy array.
    """
    import numpy as np
    import torch

    model = ESRGANState.get_model()

    h, w = img_np.shape[:2]
    img_t = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    tile = ESRGAN_TILE_SIZE
    pad  = ESRGAN_TILE_PAD
    scale = 4

    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, 3, out_h, out_w, dtype=torch.float32)

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

    out_np = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return (out_np * 255).astype(np.uint8)


# Images with either dimension below this threshold go through Real-ESRGAN (AI
# detail reconstruction). Larger images use Lanczos for speed. Raised to 400 so
# typical PDF portrait photos (200–400px) get ESRGAN rather than blurry Lanczos.
ESRGAN_SIZE_THRESHOLD = 400


def _extract_json(text: str) -> dict:
    text = text.strip()
    
    # Clean markdown
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Extract outer braces
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        raise ValueError("No JSON object found in LLM response.")
    json_str = text[first_brace:last_brace + 1]
    
    # Try direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    # Attempt simple auto-fixes
    print("Direct JSON parse failed. Attempting auto-fixes...")
    
    # Fix Qwen's trailing bracket bug: "narration": "..." \n ] \n }
    fixed_str = re.sub(r'"\s*\]\s*\}', '"\n}', json_str)
    fixed_str = re.sub(r'"\s*,\s*\]\s*\}', '"\n}', fixed_str)
    
    # Close open quotes and braces if cut off
    open_braces = fixed_str.count('{')
    close_braces = fixed_str.count('}')
    if open_braces > close_braces:
        if fixed_str.count('"') % 2 != 0:
            fixed_str += '"'
        fixed_str += '}' * (open_braces - close_braces)
        
    try:
        return json.loads(fixed_str)
    except Exception as e:
        print(f"Auto-fix failed: {e}")
        # Last resort fallback: extract key-values via regex
        fallback_dict = {}
        for key in ["title", "subtitle", "summary", "narration"]:
            match = re.search(r'"' + key + r'"\s*:\s*"([^"]*)"', json_str)
            if match:
                fallback_dict[key] = match.group(1)
        
        for key in ["highlights", "supportingPoints"]:
            array_match = re.search(r'"' + key + r'"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if array_match:
                items = re.findall(r'"([^"]*)"', array_match.group(1))
                fallback_dict[key] = items
                
        if "title" in fallback_dict:
            print("Extracted fields via regex fallback.")
            if "highlights" not in fallback_dict:
                fallback_dict["highlights"] = []
            if "supportingPoints" not in fallback_dict:
                fallback_dict["supportingPoints"] = []
            return fallback_dict
            
        raise ValueError(f"Failed to parse or repair JSON response: {e}")

def analyze_text_semantic(text: str, is_digest: bool = False, language: str = "en-US") -> dict:
    llm = LLMState.get_llm()
    if llm is None:
        raise Exception("LLM model not available yet.")

    # Shared preamble — applied to both prompt variants
    NOISE_EXCLUSION = (
        "Never include page numbers, running headers, footers, publication dates, "
        "captions, image labels, or any repeating metadata in any field.\n"
    )
    AUDIENCE = (
        "You are narrating an educational presentation for curious students aged 14–18. "
        "Use clear, engaging language — avoid jargon unless you briefly explain it.\n"
    )

    if is_digest:
        system_instructions = (
            f"{AUDIENCE}"
            "The user will give you the full text of a magazine or newspaper page that contains "
            "MULTIPLE separate articles or news items.\n"
            "Identify every distinct article on the page and output a JSON object with exactly these fields:\n"
            "- \"title\": The section heading of the whole page (e.g. 'Science Updates') — under 8 words.\n"
            "- \"subtitle\": A short description like '5 stories this issue' — under 12 words.\n"
            "- \"summary\": One sentence summarising the page as a whole (20–30 words).\n"
            "- \"topics\": A JSON array where each element has:\n"
            "    - \"title\": The article headline — under 10 words.\n"
            "    - \"summary\": One vivid sentence (20–35 words) summarising that article.\n"
            "- \"narration\": A flowing narration (60–80 words) that briefly introduces each article "
            "one by one in order, with natural transitions between them.\n"
            f"{NOISE_EXCLUSION}"
            "Output ONLY the raw JSON object — no markdown fences, no explanation.\n\n"
            "Example output format:\n"
            "{\"title\": \"Science Updates\", \"subtitle\": \"4 stories this issue\", "
            "\"summary\": \"This page covers breakthroughs in space, medicine, climate, and AI.\", "
            "\"topics\": [{\"title\": \"Mars Mission Milestone\", \"summary\": \"NASA's Perseverance rover collected its 20th rock sample, a major step toward returning Martian material to Earth.\"}], "
            "\"narration\": \"Welcome to Science Updates. This week we have four fascinating stories ...\"}"
        )
    else:
        system_instructions = (
            f"{AUDIENCE}"
            "Convert the user's text into a slide JSON object with exactly these fields:\n"
            "- \"title\": The main topic — concise, under 8 words.\n"
            "- \"subtitle\": A compelling subheading — under 12 words.\n"
            "- \"summary\": Two clear sentences (30–50 words total) explaining the core concept.\n"
            "- \"highlights\": An array of exactly 3 bullet points — key takeaways, 12–25 words each, "
            "starting with a strong action verb or number.\n"
            "- \"supportingPoints\": An array of exactly 4 interesting supporting facts or details "
            "(15–30 words each) that deepen understanding.\n"
            "- \"narration\": An engaging narration script (50–70 words) for audio playback — "
            "written in present tense, conversational but informative, as if speaking directly to students.\n"
            f"{NOISE_EXCLUSION}"
            "Output ONLY the raw JSON object — no markdown fences, no explanation.\n\n"
            "Example output format:\n"
            "{\"title\": \"How Black Holes Form\", \"subtitle\": \"Stars that collapse under their own gravity\", "
            "\"summary\": \"Black holes are regions in space where gravity is so strong that nothing, not even light, can escape. They form when massive stars exhaust their fuel and collapse.\", "
            "\"highlights\": [\"Stars 20× the Sun's mass can become black holes after a supernova explosion.\", "
            "\"The event horizon is the point of no return around a black hole.\", "
            "\"Supermassive black holes lurk at the centres of most large galaxies.\"], "
            "\"supportingPoints\": [\"The nearest known black hole, Gaia BH1, is 1,560 light-years from Earth.\", "
            "\"Time dilation near a black hole means clocks tick slower the closer you get.\", "
            "\"Stephen Hawking proposed that black holes slowly lose energy through Hawking radiation.\", "
            "\"The first image of a black hole was captured by the Event Horizon Telescope in 2019.\"], "
            "\"narration\": \"Black holes are one of the universe's most extreme objects. When a star much larger than our Sun runs out of fuel, it can no longer hold itself up against gravity. It collapses inward, triggering a massive explosion called a supernova, and leaving behind a region so dense that nothing — not even light — can escape. We call this boundary the event horizon. Scientists believe supermassive black holes, millions of times heavier than the Sun, sit at the heart of nearly every large galaxy, including our own Milky Way.\"}"
        )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": text}
    ]

    with LLMState._lock:
        max_tokens = 1500
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            repeat_penalty=1.1
        )
        output_text = response["choices"][0]["message"]["content"].strip()
    
    try:
        return _extract_json(output_text)
    except Exception as e:
        print("--- LLM OUTPUT PARSE FAILURE ---")
        print(output_text)
        print(f"Error: {e}")
        print("--------------------------------")
        raise Exception(f"Failed to parse JSON response from local LLM: {e}")

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

class TranslateDeckRequest(BaseModel):
    deck: dict

class TranscribeRequest(BaseModel):
    audio: str
    text: str
    filename: str = "audio.wav"

@app.get("/health")
def health_check():
    return {"ok": True, "service": "edge-tts"}

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
        translated_texts = translate_strings_indictrans2(texts_to_translate, "tel_Telu")
        print("translate_deck: IndicTrans2 succeeded.")
    except Exception as it_err:
        print(f"translate_deck: IndicTrans2 failed ({it_err}), falling back to Sarvam...")
        if not SARVAM_API_KEY:
            raise ValueError("IndicTrans2 unavailable and SARVAM_API_KEY is not set")
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
    """Translate a list of strings using IndicTrans2 (local model), falling back to Sarvam."""
    if not req.texts:
        return {"translations": []}
    try:
        result = await asyncio.to_thread(
            translate_strings_indictrans2,
            req.texts,
            req.target_lang
        )
        return {"translations": result, "engine": "indictrans2"}
    except Exception as e:
        print(f"IndicTrans2 translation failed: {e}. Falling back to Sarvam...")
        if not SARVAM_API_KEY:
            raise HTTPException(status_code=500, detail=f"IndicTrans2 failed and SARVAM_API_KEY not set: {e}")
        try:
            result = await asyncio.to_thread(translate_strings_sarvam, req.texts, SARVAM_API_KEY)
            return {"translations": result, "engine": "sarvam_fallback"}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))

@app.post("/translate_deck")
def translate_deck(req: TranslateDeckRequest):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=400, detail="SARVAM_API_KEY is not set")
    try:
        print("Translating deck fields into Telugu via Sarvam Translation API...")
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
        translated = await asyncio.to_thread(translate_strings_indictrans2, texts, tgt_lang_code)
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
        mode = "digest" if req.is_digest else "single-topic"
        print(f"Starting local LLM analysis ({mode}, language: {req.language}) for {len(text)} chars...")
        
        # We always generate the semantic structure in English first for high accuracy
        analysis = analyze_text_semantic(text, is_digest=req.is_digest, language="en-US")
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
                        translated = translate_strings_indictrans2(texts_to_translate, tgt_lang_code)
                        print("IndicTrans2 translation succeeded.")
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

        print("Local LLM analysis successful")
        return analysis
    except Exception as exc:
        print(f"Local LLM analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/upload_pdf")
async def upload_pdf(request: Request):
    global active_doc, active_analyzer
    pdf_bytes = await request.body()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")
    
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
            return {"ok": True, "num_pages": len(active_doc)}
        except Exception as e:
            print(f"Failed to load PDF document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load PDF: {e}")

def _extract_lines_pymupdf(page_obj, width, height, active_analyzer):
    """
    PyMuPDF-based text line extraction — used as fallback when pdfplumber is
    unavailable or fails.  Applies the same header/footer/margin filters as the
    pdfplumber path.
    """
    d = page_obj.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue
            line_text = reconstruct_line_text(spans)
            if not line_text:
                continue
            if active_analyzer and line_text in active_analyzer.running_headers_footers:
                continue
            origin_y = spans[0]["origin"][1]
            is_margin = (origin_y < height * 0.10) or (origin_y > height * 0.90)
            if is_margin:
                if re.match(r'^\d+$', line_text) or re.match(r'^[ivxIVX]+$', line_text):
                    continue
                if re.match(r'^(page|slide|p\.)\s*\d+$', line_text, re.IGNORECASE):
                    continue
            max_font_size = 0.0
            min_x = float("inf")
            baseline_y = 0.0
            for s in spans:
                max_font_size = max(max_font_size, s["size"])
                min_x = min(min_x, s["origin"][0])
                baseline_y = max(baseline_y, height - s["origin"][1])
            lines.append({
                "text": line_text,
                "x": min_x,
                "y": baseline_y,
                "fontSize": max_font_size,
            })
    return lines


def _group_words_into_lines(words):
    """Group pdfplumber word dicts into lines by proximity of their `top` coordinate."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur = [], [sorted_w[0]]
    cur_top = sorted_w[0]["top"]
    for w in sorted_w[1:]:
        if abs(w["top"] - cur_top) <= 4:
            cur.append(w)
        else:
            lines.append(sorted(cur, key=lambda w: w["x0"]))
            cur, cur_top = [w], w["top"]
    lines.append(sorted(cur, key=lambda w: w["x0"]))
    return lines


def _extract_lines_pdfplumber(pdf_path, page_num, img_rects_topdown, width, height, active_analyzer):
    """
    Layout-aware text extraction combining pdfplumber (text) and pre-extracted image
    rectangles (to filter text that bleeds over image zones).

    img_rects_topdown: list of (x0, y0, x1, y1) in top-down screen coords.

    Returns list of {"text", "x", "y" (bottom-up), "fontSize"} to match the
    existing PyMuPDF output format expected by the frontend.

    Column-detection approach:
    - Sample horizontal coverage across the middle 40 % of the page.
    - A gap ≥ 3 % of page width in that region signals a column gutter.
    - Full-width spans (crossing the gutter) are treated as headers/titles and
      placed before/after column content depending on their vertical position.
    """
    with _pdfplumber.open(pdf_path) as pdf:
        pl_page = pdf.pages[page_num - 1]

        # Words with per-word font size (falls back gracefully if attr missing)
        try:
            words = pl_page.extract_words(
                extra_attrs=["size"],
                keep_blank_chars=False,
                x_tolerance=3,
                y_tolerance=3,
            )
        except Exception:
            words = pl_page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
            for w in words:
                w.setdefault("size", 12.0)

        if not words:
            return []

        # Filter words that substantially overlap any image zone (> 30 % of word width)
        def _overlaps_image(w):
            wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]
            for (ix0, iy0, ix1, iy1) in img_rects_topdown:
                ox = min(wx1, ix1) - max(wx0, ix0)
                oy = min(wy1, iy1) - max(wy0, iy0)
                if ox > 0 and oy > 0:
                    word_w = wx1 - wx0 or 1
                    if ox / word_w > 0.3:
                        return True
            return False

        filtered = [w for w in words if not _overlaps_image(w)]
        if not filtered:
            return []

        # ── Column detection ─────────────────────────────────────────────────
        mid_start = int(width * 0.30)
        mid_end = int(width * 0.70)
        x_spans = [(w["x0"], w["x1"]) for w in filtered]

        # Count how many words cover each x position (sampled every 2 pts)
        coverage = {x: sum(1 for (x0, x1) in x_spans if x0 <= x <= x1)
                    for x in range(mid_start, mid_end, 2)}

        # Find the widest contiguous zero-coverage gap in the middle band
        gutter_x = None
        max_gap = 0
        gap_start = None
        for x in range(mid_start, mid_end, 2):
            if coverage.get(x, 0) == 0:
                if gap_start is None:
                    gap_start = x
            else:
                if gap_start is not None:
                    gap = x - gap_start
                    if gap > max_gap:
                        max_gap, gutter_x = gap, (gap_start + x) // 2
                    gap_start = None
        if gap_start is not None:
            gap = mid_end - gap_start
            if gap > max_gap:
                max_gap, gutter_x = gap, (gap_start + mid_end) // 2

        is_two_col = max_gap >= width * 0.03 and gutter_x is not None

        # ── Group into lines then classify ───────────────────────────────────
        all_line_groups = _group_words_into_lines(filtered)

        if is_two_col:
            # A line "spans the gutter" if it has words on both sides
            def _spans(line):
                return any(w["x0"] < gutter_x for w in line) and any(w["x1"] > gutter_x for w in line)

            full_lines   = [l for l in all_line_groups if _spans(l)]
            left_lines   = [l for l in all_line_groups if not _spans(l) and all(w["x1"] <= gutter_x for w in l)]
            right_lines  = [l for l in all_line_groups if not _spans(l) and all(w["x0"] >= gutter_x for w in l)]

            # Column content starts where the first left or right column word appears
            col_words = [w for l in left_lines + right_lines for w in l]
            col_start = min((w["top"] for w in col_words), default=0)
            col_end   = max((w["top"] for w in col_words), default=height)

            pre_col  = sorted([l for l in full_lines if l[0]["top"] < col_start],  key=lambda l: l[0]["top"])
            post_col = sorted([l for l in full_lines if l[0]["top"] > col_end],   key=lambda l: l[0]["top"])
            mid_full = sorted([l for l in full_lines if col_start <= l[0]["top"] <= col_end], key=lambda l: l[0]["top"])

            # Primary column = the one with more words (main article body).
            # Secondary column = the one with fewer words (sidebar / captions).
            # If counts are within 1.8× of each other, use left-first (standard order).
            left_wc  = sum(len(l) for l in left_lines)
            right_wc = sum(len(l) for l in right_lines)
            if right_wc > left_wc * 1.8:
                primary_col   = sorted(right_lines, key=lambda l: l[0]["top"])
                secondary_col = sorted(left_lines,  key=lambda l: l[0]["top"])
            else:
                primary_col   = sorted(left_lines,  key=lambda l: l[0]["top"])
                secondary_col = sorted(right_lines, key=lambda l: l[0]["top"])

            ordered = (
                pre_col
                + primary_col
                + mid_full
                + secondary_col
                + post_col
            )
        else:
            ordered = all_line_groups

        # ── Build output, applying header/footer filters ─────────────────────
        result = []
        for line_words in ordered:
            text = " ".join(w["text"] for w in line_words).strip()
            if not text:
                continue

            if active_analyzer and text in active_analyzer.running_headers_footers:
                continue

            top_y = line_words[0]["top"]
            is_margin = top_y < height * 0.10 or top_y > height * 0.90
            if is_margin:
                if re.match(r'^\d+$', text) or re.match(r'^[ivxIVX]+$', text):
                    continue
                if re.match(r'^(page|slide|p\.)\s*\d+$', text, re.IGNORECASE):
                    continue

            sizes = [w.get("size") or 12.0 for w in line_words]
            font_size = max(sizes)
            min_x = min(w["x0"] for w in line_words)
            # Convert top-down bottom coord → bottom-up y (matches PyMuPDF output)
            y_bottom_up = height - line_words[0]["bottom"]

            result.append({
                "text": text,
                "x": min_x,
                "y": y_bottom_up,
                "fontSize": font_size,
            })

        return result


@app.get("/page_layout")
def get_page_layout(page: int):
    global active_doc, active_analyzer
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
            page_obj = active_doc[page - 1]
            rect = page_obj.rect
            width = rect.width
            height = rect.height
            
            lines = []
            images = []

            # ── Image extraction via get_images() ────────────────────────────────
            # get_text("dict") type-1 blocks miss most PDF images because the
            # majority of PDFs embed images as XObjects (placed with Do operator)
            # rather than inline. get_images(full=True) finds all of them.
            seen_xrefs = set()
            for img_info in page_obj.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    rects = page_obj.get_image_rects(xref)
                except Exception:
                    rects = []

                for rect in rects:
                    img_w = rect.width
                    img_h = rect.height
                    # Skip invisible or near-zero rendered size
                    if img_w < 5 or img_h < 5:
                        continue
                    # Skip tiny decorative elements (icons, bullets, dividers)
                    if img_w < 50 or img_h < 50:
                        continue
                    # Skip full-page-width backgrounds / decorative banners
                    if img_w > width * 0.90:
                        continue

                    try:
                        img_dict = active_doc.extract_image(xref)
                    except Exception:
                        continue

                    image_bytes = img_dict.get("image")
                    if not image_bytes:
                        continue
                    ext = img_dict.get("ext", "png")
                    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    url = f"data:image/{ext};base64,{img_base64}"
                    images.append({
                        "url": url,
                        "x": rect.x0,
                        "y": height - rect.y1,
                        "w": img_w,
                        "h": img_h,
                        "cx": (rect.x0 + rect.x1) / 2,
                        "cy": height - (rect.y0 + rect.y1) / 2
                    })
                    break  # one rect per xref is enough for position info

            # ── Text extraction ───────────────────────────────────────────────
            # pdfplumber gives column-aware, image-zone-filtered text.
            # Fall back to PyMuPDF's get_text("dict") if pdfplumber is unavailable
            # or raises an unexpected error.
            if _PDFPLUMBER_AVAILABLE:
                try:
                    # Convert image positions to top-down coords for overlap filtering
                    img_rects_td = [
                        (img["x"], height - img["y"] - img["h"],
                         img["x"] + img["w"], height - img["y"])
                        for img in images
                    ]
                    lines = _extract_lines_pdfplumber(
                        str(active_doc.name), page, img_rects_td,
                        width, height, active_analyzer
                    )
                    print(f"pdfplumber extracted {len(lines)} lines for page {page}")
                except Exception as pl_err:
                    print(f"pdfplumber failed for page {page} ({pl_err}), falling back to PyMuPDF")
                    lines = _extract_lines_pymupdf(page_obj, width, height, active_analyzer)
            else:
                lines = _extract_lines_pymupdf(page_obj, width, height, active_analyzer)

            return {
                "width": width,
                "height": height,
                "lines": lines,
                "images": images
            }
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
    Upscale a base64-encoded image and return a sharper version.

    Routing logic (based on input dimensions):
      • width < ESRGAN_SIZE_THRESHOLD or height < ESRGAN_SIZE_THRESHOLD
          → Real-ESRGAN ×4 (adds AI-reconstructed detail for tiny/degraded images)
      • otherwise
          → Lanczos ×2 + Unsharp Mask (clean, artefact-free for normal PDF images)

    Real-ESRGAN is intentionally avoided for larger images because it hallucinates
    detail that doesn't exist, causing the "morphed wax figure" distortion on people,
    faces, and complex photo content.

    Request body (JSON):
        { "image": "<base64 PNG or JPEG>", "format": "png" | "jpeg" }

    Response (JSON):
        { "image": "<base64 upscaled image>", "method": "lanczos" | "esrgan" }
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Pillow not installed: {exc}") from exc

    body = await request.json()
    b64 = body.get("image", "")
    fmt = body.get("format", "png").lower()
    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body.")

    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    w, h = pil_img.width, pil_img.height
    is_tiny = w < ESRGAN_SIZE_THRESHOLD or h < ESRGAN_SIZE_THRESHOLD
    use_esrgan = is_tiny and ESRGAN_MODEL_PATH.exists()

    print(f"/upscale_image: input {w}×{h} px, method={'esrgan' if use_esrgan else 'lanczos'}, fmt={fmt}")

    loop = asyncio.get_event_loop()

    if use_esrgan:
        import numpy as np
        # Cap input to keep ESRGAN fast on CPU. 400px → 1600px output is the expected
        # portrait case; 600px cap handles larger inputs while staying under ~2s per tile.
        MAX_INPUT_PX = 600
        if w > MAX_INPUT_PX or h > MAX_INPUT_PX:
            pil_img.thumbnail((MAX_INPUT_PX, MAX_INPUT_PX), Image.LANCZOS)
        img_np = np.array(pil_img)
        upscaled_np = await loop.run_in_executor(None, _esrgan_upscale, img_np)
        out_pil = Image.fromarray(upscaled_np)
        method = "esrgan"
        # Apply GFPGAN face restoration on top of ESRGAN if the model is available
        if GFPGANState.get_enhancer() is not None:
            out_pil = await loop.run_in_executor(None, _gfpgan_enhance_faces, out_pil)
            method = "esrgan+gfpgan"
    else:
        # Lanczos is fast enough to run inline; wrap in executor anyway for consistency
        out_pil = await loop.run_in_executor(None, _lanczos_upscale, pil_img, 2)
        method = "lanczos"

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
