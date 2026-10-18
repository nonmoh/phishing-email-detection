@echo off
cd /d %~dp0
call .venv\Scripts\activate
python src\build_master_dataset.py
pause
