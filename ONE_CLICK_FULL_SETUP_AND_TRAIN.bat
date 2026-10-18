@echo off
setlocal
cd /d "%~dp0"
title AI Phishing Project - Setup, Data, Train, Notebook

echo ================================================================
echo AI PHISHING PROJECT - FULL SETUP + TRAINING
echo ================================================================
echo This performs the complete Internet setup and then trains/evaluates
echo Dummy, Logistic Regression, Linear SVM, cross-generator robustness,
echo error analysis, and figures before opening JupyterLab.
echo.
echo NOTE: DistilBERT BASE weights are downloaded, but DistilBERT phishing
echo fine-tuning remains optional inside the notebook because it can take
echo a long time on CPU.
echo ================================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 bootstrap_everything.py --preload-distilbert --train --open-notebook
) else (
    python bootstrap_everything.py --preload-distilbert --train --open-notebook
)

if %errorlevel% neq 0 (
    echo.
    echo Setup/training stopped because of an error. Read the message above.
    pause
    exit /b %errorlevel%
)

exit /b 0
