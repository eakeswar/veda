# Veda — AI-Powered PDF to Interactive Presentation

Veda turns an uploaded PDF into an interactive narrated presentation with read-along (karaoke-style) word highlighting. English and Telugu are supported.

**Public repo:** [https://github.com/eakeswar/veda](https://github.com/eakeswar/veda)

The laptop always runs the FastAPI orchestrator (`server/tts_server.py`, port **8765**) and the React UI (Vite, port **5173**). Heavy work can optionally go to a **Kaggle GPU** notebook (Qwen 7B, NLLB, SSD-1B img2img). If Kaggle is off or the tunnel is down, everything falls back to **local CPU**.

Architecture, routing tables, known issues, and dated incident notes live in **`.cursorrules`** (keep that file; it is the detailed engineering log). Internship write-up: `docs/Internship_Report_Veda.md` (also `.docx` / `.pdf`).

---

## Before you delete a local copy

GitHub does **not** contain secrets or downloaded models. Save these **outside** the folder (password manager / notes) before deleting the laptop directory:

| Keep a copy of | Why it is not in git |
|----------------|----------------------|
| `SARVAM_API_KEY` | TTS / STT / last-resort translate |
| `OPENAI_API_KEY` (if you used `/transcribe`) | Optional Whisper |
| `KAGGLE_API_SECRET` | Laptop ↔ Kaggle notebook password |
| `KAGGLE_API_BASE_URL` | Changes every notebook session |
| `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` | Publishing `project-models` |
| HuggingFace `HF_TOKEN` (Kaggle secret) | Faster model downloads on Kaggle |

After a fresh clone you recreate `server/.env` from `server/.env.example` and paste those keys. GGUF models download again on first `/analyze` (or via `server/kaggle/download_qwen_7b.py` for the 7B bundle).

**Do not delete the folder until** you have confirmed the GitHub repo has your latest commits: open [github.com/eakeswar/veda](https://github.com/eakeswar/veda) and check `README.md`, `.cursorrules`, `docs/`, and `src/`.

---

## What is in git vs what you reinstall

**In the repo (clone this):** source (`src/`, `server/*.py`, Kaggle scripts, notebook), `package.json` / lockfile, `server/requirements.txt`, `.env.example` files, YuNet ONNX (`server/data/face_detection_yunet_2023mar.onnx`), docs, `.cursorrules`.

**Not in git (GitHub rejects files over 100 MB):** Qwen GGUF (~0.5–4 GB) and NLLB (~2.5 GB) cannot be pushed. YuNet face weights **are** in git (`server/data/*.onnx`).

| Path | How to restore |
|------|----------------|
| `node_modules/` | `npm install` |
| `server/vendor/` | `pip install --target server/vendor ...` |
| `server/models/*.gguf` | `python server/download_local_models.py` (or `setup.ps1`) |
| HuggingFace NLLB cache | `python server/download_local_models.py --nllb` |
| `server/.env` | Copy from `server/.env.example` |
| Root `.env` | Optional; copy from `.env.example` |
| `server/active_doc.pdf` | Created when you upload a PDF |

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|--------|
| **Git** | recent | Clone this repo |
| **Python** | **3.11** recommended | Backend |
| **Node.js** | **20+** | Frontend (Vite 8) |
| **RAM** | 16 GB+ if local 3B + NLLB | 32–64 GB is more comfortable |
| **GPU** | not required | Laptop path is CPU; GPU is Kaggle T4 × 2 |

---

## Setup on a new machine (Windows PowerShell)

### 1. Clone

```powershell
git clone https://github.com/eakeswar/veda.git
cd veda
```

### 2. One command (packages + Qwen GGUF + NLLB)

This installs npm deps, `server/vendor`, CPU PyTorch, then downloads the local Qwen GGUF (default **3B**, from `LOCAL_LLM_SIZE`) and **NLLB-200 1.3B**. Several GB; needs disk space and a stable network.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

To match a small laptop that only used **0.5B**, set that before download:

```powershell
$env:LOCAL_LLM_SIZE="0.5B"
python server\download_local_models.py --nllb
```

Optional: `copy .env.example .env` (only if you need to change `VITE_API_BASE_URL`).

### 3. Backend Python packages (if you skip setup.ps1)

Always install into **`server/vendor`** (this project does not use a global site-packages install for app deps):

```powershell
cd server
python -m pip install --target vendor -r requirements.txt
python -m pip install --target vendor torch --index-url https://download.pytorch.org/whl/cpu
cd ..
```

If pdfplumber / pdfminer fails at runtime, install once **system-wide**:

```powershell
python -m pip install cryptography cffi
```

### 4. Environment file (required)

```powershell
copy server\.env.example server\.env
```

Edit `server/.env`:

- Set **`SARVAM_API_KEY`**. If it is missing or invalid, TTS falls back to **edge-tts** (no karaoke-quality Sarvam STT).
- Leave **`KAGGLE_ENABLED`** unset/false until a notebook tunnel is running.
- Keep **`LLM_PROVIDER=local`**, **`TRANSLATE_PROVIDER=local`**, **`LAYOUT_PROVIDER=local`** for laptop-only.

Generate a Kaggle tunnel secret when you need GPU (never use `veda-kaggle-dev`):

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Put the same value in `server/.env` as `KAGGLE_API_SECRET` and in the Kaggle notebook cell.

### 5. Run both servers

Terminal A — backend (port **8765**):

```powershell
python server/tts_server.py
```

Terminal B — frontend (port **5173**):

```powershell
npm run dev
```

Open **http://localhost:5173**. Check **http://127.0.0.1:8765/health**.

If you already ran `setup.ps1`, Qwen and NLLB are on disk. Otherwise the first **Analyze** still downloads the local GGUF, and the first Telugu translate downloads NLLB.

---

## Setup (macOS / Linux)

Same steps; use `cp` instead of `copy`, and `python3` if needed:

```bash
git clone https://github.com/eakeswar/veda.git
cd veda
npm install
python3 -m pip install --target server/vendor -r server/requirements.txt
python3 -m pip install --target server/vendor torch --index-url https://download.pytorch.org/whl/cpu
cp server/.env.example server/.env
# edit server/.env
python3 server/tts_server.py
# other terminal:
npm run dev
```

---

## How the app runs (product flow)

1. Upload a PDF in the left panel → `POST /upload_pdf` (mirrored to Kaggle if layout provider is Kaggle).
2. `GET /page_layout?page=N` extracts lines and images (pdfplumber + PyMuPDF).
3. Frontend post-processes magazine layout (`src/utils/pdfUtils.js`).
4. `POST /analyze` → Qwen produces slide JSON (7B on Kaggle or 3B local).
5. Telugu: `POST /translate_deck` / field endpoints (NLLB, then Sarvam).
6. Play → `POST /tts` (Sarvam WAV + STT word times, silence-adjusted; edge-tts fallback).
7. Images → `POST /upscale_image` (Lanczos on laptop, or Kaggle SSD-1B img2img for some photos/people).
8. **Open HTML** exports the current page to a new tab (`src/utils/exportPageHtml.js`).

---

## API endpoints (laptop)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, providers, Kaggle reachability (tunnel URL redacted) |
| POST | `/upload_pdf` | Save `server/active_doc.pdf` |
| GET | `/page_layout?page=N` | `{ width, height, lines[], images[] }` |
| POST | `/analyze` | Slide JSON |
| POST | `/translate`, `/translate_deck`, `/translate_fields` | Translation |
| POST | `/tts` | Audio + `word_boundaries` |
| GET | `/tts_stream` | Streaming audio |
| POST | `/tts_boundaries` | Timestamps only |
| POST | `/transcribe` | Optional OpenAI Whisper |
| POST | `/upscale_image` | Lanczos or Kaggle img2img |

---

## Environment variables

### `server/.env`

See **`server/.env.example`** for the full commented list. Summary:

| Variable | Purpose |
|----------|---------|
| `SARVAM_API_KEY` | TTS, STT, translation last resort |
| `OPENAI_API_KEY` | Optional `/transcribe` |
| `VEDA_API_KEY` | Optional auth on the laptop API |
| `VEDA_CORS_ORIGINS` | Defaults to local Vite origins |
| `LOCAL_LLM_SIZE` | Laptop GGUF: `0.5B` / `1.5B` / `3B` (default 3B) |
| `KAGGLE_LLM_SIZE` | Notebook GGUF (default **7B**) |
| `LLM_PROVIDER` / `TRANSLATE_PROVIDER` / `LAYOUT_PROVIDER` | `local` \| `kaggle` \| `auto` |
| `UPSCALE_PROVIDER` | Lanczos always runs on the laptop |
| `IMAGE_GEN_PROVIDER` | `auto` \| `kaggle` \| `local` (local = no img2img) |
| `KAGGLE_ENABLED` | `true` only while the notebook + tunnel are up |
| `KAGGLE_API_BASE_URL` | `https://….trycloudflare.com` |
| `KAGGLE_API_SECRET` | Shared secret (≥16 chars, not the old default) |
| `KAGGLE_USERNAME` / `KAGGLE_API_TOKEN` | Official Kaggle API to publish datasets |

`tts_server.py` loads dotenv with **`override=True`**: save `.env` then restart the backend so a new tunnel URL is picked up.

### Root `.env` (frontend, optional)

| Variable | Purpose |
|----------|---------|
| `VITE_API_BASE_URL` | Backend base (default `http://127.0.0.1:8765`) |
| `VITE_VEDA_API_KEY` | Bearer token if backend auth is on |

---

## Kaggle GPU (optional)

Two **different** credentials:

1. **`KAGGLE_API_SECRET`** — password between laptop and notebook (you invent it).
2. **`KAGGLE_USERNAME` + `KAGGLE_API_TOKEN`** — [kaggle.com/settings](https://www.kaggle.com/settings) API token, only for **publishing** the `project-models` dataset.

### Publish / attach models

1. Download 7B GGUF if needed: `python server/kaggle/download_qwen_7b.py`
2. `server\kaggle\prepare_kaggle_dataset.bat`
3. `server\kaggle\publish_kaggle_dataset.bat`
4. In Kaggle: notebook, GPU **T4 x2**, Internet **ON**, attach dataset **`project-models`** (latest version). Add secret **`HF_TOKEN`**.
5. Run `server/kaggle/veda_kaggle.ipynb`:
   - Cell 1 copies scripts from the dataset into `/kaggle/working`
   - Cell 2 sets `KAGGLE_API_SECRET`, `HF_HOME=/kaggle/working/hf_cache`, starts the GPU server
6. Wait for **`all model pre-warm complete`** (first run ~7–15 min). Do not cancel.
7. Copy the Cloudflare URL into laptop `server/.env` as `KAGGLE_API_BASE_URL`, set `KAGGLE_ENABLED=true` and providers to `kaggle` or `auto`, **save the file**, restart `tts_server.py`.

When GPU hours are gone or you get Cloudflare **530**: set `KAGGLE_ENABLED=false` and use laptop CPU.

Health / trials:

```powershell
python server/test_kaggle.py
python server/test_kaggle.py --compare
python server/test_kaggle.py --prompt-trial
python server/test_kaggle.py --generate-trial
python server/test_kaggle.py --translate-trial
```

---

## Project layout

```
veda/
├── .cursorrules                 # Full architecture + incidents
├── README.md                    # This setup guide
├── docs/                        # Internship report
├── package.json                 # React / Vite / Tailwind
├── src/
│   ├── config/api.js
│   ├── App.jsx
│   ├── components/              # LeftPanel, RightPanel, AIAvatar
│   ├── context/PDFContext.jsx
│   └── utils/                   # pdfUtils, speechUtils, exportPageHtml
└── server/
    ├── tts_server.py            # Orchestrator :8765
    ├── requirements.txt
    ├── .env.example
    ├── data/                    # YuNet ONNX (in git)
    ├── kaggle/
    │   ├── veda_gpu_server.py
    │   ├── veda_kaggle.ipynb
    │   ├── image_gen_service.py
    │   ├── prepare_kaggle_dataset.bat
    │   └── publish_kaggle_dataset.bat
    ├── models/                  # gitignored GGUF
    └── vendor/                  # gitignored pip target
```

---

## Security

- Default bind is localhost. Set `VEDA_API_KEY` before exposing the API.
- Never commit `server/.env`.
- CORS is restricted to local Vite origins unless you set `VEDA_CORS_ORIGINS`.
- PDFs are validated (`%PDF-` header, size cap).

---

## Typical problems

| Symptom | What to do |
|---------|------------|
| Frontend loads, no analysis | Is `:8765` running? CORS / `VITE_API_BASE_URL`? |
| Telugu still English | NLLB/vendor install; do not rely on a placeholder Sarvam key |
| TTS works but no highlighting | Sarvam STT needs a valid key; otherwise timestamps are heuristic |
| Kaggle 524 / 503 on images | Wait for pre-warm; `image_gen_prewarm: true` on Kaggle `/health` |
| Kaggle 530 | Notebook or tunnel died; new URL + restart laptop backend |
| Dark mode looks inverted | Theme must be on `<html data-theme>` (already in `PDFContext.jsx`) |

---

## License / internship

Built as an internship project at **ADS Softek**. Code in this repository is the project source; models and API keys remain subject to their own terms (HuggingFace, Sarvam, OpenAI, Kaggle).
