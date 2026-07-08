@echo off
REM Publish project-models to Kaggle using API token from server/.env
python "%~dp0publish_project_models.py"
exit /b %ERRORLEVEL%
