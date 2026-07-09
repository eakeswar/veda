@echo off
REM Build veda-models-bundle for Kaggle dataset project-models (7B only)
setlocal
set DEST=%~dp0veda-models-bundle
set MODELS=%~dp0..\models

REM Clean stale files from previous bundles (old 3B, ESRGAN, etc.)
if exist "%DEST%" rmdir /S /Q "%DEST%"
mkdir "%DEST%" 2>nul

copy /Y "%~dp0..\security.py" "%DEST%\"
copy /Y "%~dp0..\qwen_config.py" "%DEST%\"
copy /Y "%~dp0..\analyze_prompts.py" "%DEST%\"
copy /Y "%~dp0..\page_layout_service.py" "%DEST%\"
copy /Y "%~dp0veda_gpu_server.py" "%DEST%\"
copy /Y "%~dp0image_gen_service.py" "%DEST%\"

if exist "%MODELS%\qwen2.5-7b-instruct-q3_k_m.gguf" (
  copy /Y "%MODELS%\qwen2.5-7b-instruct-q3_k_m.gguf" "%DEST%\"
  echo Bundled Qwen 2.5 7B ^(q3_k_m^) for Kaggle GPU
) else (
  echo ERROR: qwen2.5-7b-instruct-q3_k_m.gguf not found in server\models\
  echo.
  echo Download it from repo root:
  echo   python server/kaggle/download_qwen_7b.py
  echo.
  echo Laptop keeps 3B via LOCAL_LLM_SIZE=3B — only 7B goes to Kaggle.
  exit /b 1
)

echo.
echo Bundle ready: %DEST%
echo Publish: server\kaggle\publish_kaggle_dataset.bat
