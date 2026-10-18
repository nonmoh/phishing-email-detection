@echo off
setlocal
cd /d "%~dp0"
title AI Phishing Project - Full Setup

echo ================================================================
echo AI PHISHING PROJECT - ONE CLICK FULL SETUP
echo ================================================================
echo This will:
echo   1. Create/reuse .venv
echo   2. Install ALL Python libraries
echo   3. Download ALL public project datasets from the Internet
echo   4. Rebuild the cleaned master dataset
echo   5. Download the base DistilBERT model
echo   6. Register the Jupyter kernel
echo   7. Open START_HERE_LARGE_10WEEKS.ipynb in JupyterLab
echo.
echo Internet is required. Large downloads may take time.
echo ================================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 bootstrap_everything.py --preload-distilbert --open-notebook
) else (
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python was not found.
        echo Install Python 3.11 or newer from python.org, then run this file again.
        pause
        exit /b 1
    )
    python bootstrap_everything.py --preload-distilbert --open-notebook
)

if %errorlevel% neq 0 (
    echo.
    echo Setup stopped because of an error. Read the message above, then rerun this file.
    pause
    exit /b %errorlevel%
)

exit /b 0
