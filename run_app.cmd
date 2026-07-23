@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
set PYTHONPATH=
.
.venv\Scripts\python.exe -m streamlit run app.py
