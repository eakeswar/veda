@echo off
REM Publish the veda-models bundle to Kaggle using the official Kaggle API.
REM
REM Prerequisites:
REM   1. pip install kaggle
REM   2. KAGGLE_USERNAME + KAGGLE_KEY in server/.env (or ~/.kaggle/kaggle.json)
REM   3. Create dataset-metadata.json from dataset-metadata.json.example
REM      (set YOUR_KAGGLE_USERNAME/veda-models as the id)
REM
REM First-time create:
REM   kaggle datasets create -p veda-models-bundle
REM Subsequent updates:
REM   kaggle datasets version -p veda-models-bundle -m "Updated from Veda"

setlocal
cd /d "%~dp0"

call prepare_kaggle_dataset.bat
if errorlevel 1 exit /b 1

if not exist "dataset-metadata.json" (
  echo.
  echo ERROR: server\kaggle\dataset-metadata.json not found.
  echo Copy dataset-metadata.json.example and set your Kaggle username in the id field.
  exit /b 1
)

REM Load KAGGLE_USERNAME and KAGGLE_KEY from server/.env if present
if exist "..\..\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("..\..\.env") do (
    if "%%A"=="KAGGLE_USERNAME" set KAGGLE_USERNAME=%%B
    if "%%A"=="KAGGLE_KEY" set KAGGLE_KEY=%%B
  )
)
if exist "..\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("..\.env") do (
    if "%%A"=="KAGGLE_USERNAME" set KAGGLE_USERNAME=%%B
    if "%%A"=="KAGGLE_KEY" set KAGGLE_KEY=%%B
  )
)

if not defined KAGGLE_USERNAME (
  echo WARNING: KAGGLE_USERNAME not set — ensure ~/.kaggle/kaggle.json exists
)
if not defined KAGGLE_KEY (
  echo WARNING: KAGGLE_KEY not set — ensure ~/.kaggle/kaggle.json exists
)

echo.
echo Publishing veda-models-bundle to Kaggle...
kaggle datasets version -p veda-models-bundle -m "Updated from Veda %date% %time%"
if errorlevel 1 (
  echo.
  echo If this is the first publish, run instead:
  echo   kaggle datasets create -p veda-models-bundle
  exit /b 1
)

echo Done. Attach the updated dataset to your Kaggle notebook.
