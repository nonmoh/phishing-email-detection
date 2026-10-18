@echo off
cd /d %~dp0
call .venv\Scripts\activate
python src\train_baselines.py
python src\evaluate_cross_generator.py
python src\export_visual_ready.py
pause
