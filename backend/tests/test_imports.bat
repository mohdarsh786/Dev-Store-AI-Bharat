@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python ingestion\test_import.py
pause
