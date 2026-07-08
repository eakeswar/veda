# Veda — AI-Powered PDF to Interactive Presentation

Veda converts uploaded PDFs into interactive narrated presentations with karaoke-style word highlighting. The backend extracts page layout, a Qwen LLM generates slide JSON, and Sarvam AI handles TTS + STT for word-level timestamps. Optional Kaggle GPU offloading accelerates analyze, translate, and upscale.

---

## Quick Start

### Backend (port 8765)

```bash
pip install --target server/vendor -r server/requirements.txt
python server/tts_server.py
```

Copy `server/.env.example` → `server/.env` and add your `SARVAM_API_KEY`.

### Frontend (port 5173)

```bash
npm install
npm run dev
```

Optional: copy `.env.example` → `.env` for `VITE_*` overrides.

---

## Architecture

```
veda/
├── server/
│   ├── tts_server.py          # Laptop orchestrator (FastAPI, :8765)
│   ├── kaggle_client.py       # HTTP client for Kaggle GPU tunnel
│   ├── security.py            # Shared limits, validation, CORS helpers
│   ├── analyze_prompts.py     # LLM prompts + JSON extract
│   ├── page_layout_service.py # PDF layout extraction
│   ├── upscale_service.py     # Face-aware upscaling (YuNet + ESRGAN)
│   └── kaggle/
│       ├── veda_gpu_server.py           # Run in Kaggle notebook (GPU T4 x2)
│       ├── prepare_kaggle_dataset.bat     # Build veda-models bundle locally
│       ├── publish_kaggle_dataset.bat     # Push bundle via Kaggle API
│       └── veda-models-bundle/          # Generated — not edited by hand
├── src/
│   ├── config/api.js          # Central backend URL config
│   ├── components/            # React UI (LeftPanel, RightPanel, AIAvatar)
│   └── utils/                 # pdfUtils.js, speechUtils.js
└── .cursorrules               # Detailed architecture reference
```

**Dual-mode:** Heavy ML routes to Kaggle when `KAGGLE_ENABLED=true`; everything falls back to local CPU automatically when the tunnel is down.

---

## Environment Variables

### `server/.env` (backend secrets)

| Variable | Purpose |
|----------|---------|
| `SARVAM_API_KEY` | TTS, STT, translation fallback |
| `OPENAI_API_KEY` | Optional `/transcribe` (Whisper) |
| `KAGGLE_ENABLED` | `true` while Kaggle notebook session is running |
| `KAGGLE_API_BASE_URL` | Cloudflare tunnel URL from notebook |
| `KAGGLE_API_SECRET` | Strong random secret (shared with notebook) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Official Kaggle API — publish datasets from laptop |
| `VEDA_API_KEY` | Optional auth for laptop server (non-localhost deploy) |
| `LLM_PROVIDER` etc. | `local` \| `kaggle` \| `auto` |

See `server/.env.example` for the full list.

### Root `.env` (frontend, optional)

| Variable | Purpose |
|----------|---------|
| `VITE_API_BASE_URL` | Backend base URL (default `http://127.0.0.1:8765`) |
| `VITE_VEDA_API_KEY` | Sent as Bearer token if backend auth is enabled |

---

## Kaggle GPU Workflow

### Two different “Kaggle keys”

1. **`KAGGLE_API_SECRET`** — Your own shared password between the laptop server and the GPU server running in a notebook (via Cloudflare tunnel). Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. Set the same value in `server/.env` and in the notebook environment before starting `veda_gpu_server.py`.

2. **`KAGGLE_USERNAME` + `KAGGLE_KEY`** — Official Kaggle API credentials from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token. Used to **publish dataset updates** from your laptop via CLI — not for runtime inference.

### Runtime (GPU inference)

1. Build bundle: `server\kaggle\prepare_kaggle_dataset.bat`
2. Publish dataset (optional, when code/models change): `server\kaggle\publish_kaggle_dataset.bat`
3. Open Kaggle notebook → attach `veda-models` dataset → GPU T4 x2, Internet ON
4. Set `KAGGLE_API_SECRET` in notebook, run `veda_gpu_server.py`
5. Copy tunnel URL → `server/.env` → restart `tts_server.py`

### Publish dataset from laptop

```bash
pip install kaggle
# Copy server/kaggle/dataset-metadata.json.example → dataset-metadata.json
# Set id to YOUR_USERNAME/veda-models
server\kaggle\publish_kaggle_dataset.bat
```

First time: `kaggle datasets create -p veda-models-bundle`

---

## Security Notes

- Laptop server binds to `127.0.0.1` by default — safe for local dev
- Set `VEDA_API_KEY` before exposing the server beyond localhost
- Never use the old default `veda-kaggle-dev` as `KAGGLE_API_SECRET`
- CORS restricted to Vite dev origins by default (`VEDA_CORS_ORIGINS`)
- PDF uploads validated (`%PDF-` header, 50 MB cap)

---

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + provider modes |
| POST | `/upload_pdf` | Save active PDF |
| GET | `/page_layout?page=N` | Text + image layout |
| POST | `/analyze` | LLM slide JSON |
| POST | `/translate*` | Telugu translation |
| POST | `/tts` | Sarvam TTS + word boundaries |
| POST | `/upscale_image` | Face-aware upscaling |

---

## Testing Kaggle

```bash
python server/test_kaggle.py
python server/test_kaggle.py --compare
python server/test_kaggle.py --kaggle-only
```
