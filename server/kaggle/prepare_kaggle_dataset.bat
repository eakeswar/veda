@echo off
REM Copy model files into a folder you can upload as a Kaggle Dataset named "veda-models"
set DEST=%~dp0veda-models-bundle
mkdir "%DEST%" 2>nul

copy /Y "%~dp0..\analyze_prompts.py" "%DEST%\"
copy /Y "%~dp0..\page_layout_service.py" "%DEST%\"
copy /Y "%~dp0..\rrdb_net.py" "%DEST%\"
if exist "%~dp0..\models\RealESRGAN_x4plus.pth" (
  copy /Y "%~dp0..\models\RealESRGAN_x4plus.pth" "%DEST%\"
) else (
  echo WARNING: RealESRGAN_x4plus.pth not found - upscale on Kaggle will use Lanczos only
)
if exist "%~dp0..\models\qwen2.5-3b-instruct-q4_k_m.gguf" (
  copy /Y "%~dp0..\models\qwen2.5-3b-instruct-q4_k_m.gguf" "%DEST%\"
) else (
  echo WARNING: Qwen GGUF not found in server\models\ - download first or add manually
)

echo.
echo Bundle ready: %DEST%
echo Upload this folder as a Kaggle Dataset named "veda-models"
echo Then attach it to your notebook and run veda_gpu_server.py
