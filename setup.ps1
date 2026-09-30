# One-shot setup after cloning https://github.com/eakeswar/veda
# Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "==> Frontend: npm install"
npm install

Write-Host "==> Backend packages into server\vendor"
python -m pip install --target server\vendor -r server\requirements.txt
python -m pip install --target server\vendor torch --index-url https://download.pytorch.org/whl/cpu

if (-not (Test-Path "server\.env")) {
    Copy-Item "server\.env.example" "server\.env"
    Write-Host "Created server\.env from example — add SARVAM_API_KEY before running TTS."
}

Write-Host "==> Qwen GGUF (LOCAL_LLM_SIZE, default 3B) + NLLB-200"
$env:PYTHONPATH = "server\vendor"
python server\download_local_models.py --nllb

Write-Host ""
Write-Host "Setup complete."
Write-Host "  1. Edit server\.env and paste your API keys"
Write-Host "  2. python server\tts_server.py"
Write-Host "  3. npm run dev"
