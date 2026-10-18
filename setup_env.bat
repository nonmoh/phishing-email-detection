@echo off
cd /d %~dp0
py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Environment ready. Open this folder in Visual Studio Code and select .venv as the Python interpreter.
pause
