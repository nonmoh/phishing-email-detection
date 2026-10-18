@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv does not exist yet. Run INSTALL_EVERYTHING_AND_OPEN_NOTEBOOK.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" src\download_all_project_data.py --force
if %errorlevel% neq 0 (
    pause
    exit /b %errorlevel%
)
".venv\Scripts\python.exe" src\prepare_web_sources.py
".venv\Scripts\python.exe" src\build_master_dataset.py

echo.
echo All datasets were downloaded and the master dataset was rebuilt.
pause
