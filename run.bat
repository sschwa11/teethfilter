@echo off
cd /d "%~dp0"
if not exist .venv (
  py -3.12 -m venv .venv || py -3 -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
)
.venv\Scripts\python -m pip install -q -r requirements.txt
.venv\Scripts\python teeth_filter.py --width 1280 --height 720 %*
pause
