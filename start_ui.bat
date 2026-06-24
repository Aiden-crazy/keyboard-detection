@echo off
cd /d "%~dp0"

REM Check for Python (try python, then python3, then py)
set PYTHON=
python --version >nul 2>nul && set PYTHON=python
if not defined PYTHON python3 --version >nul 2>nul && set PYTHON=python3
if not defined PYTHON py --version >nul 2>nul && set PYTHON=py
if not defined PYTHON (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo Installing dependencies...
%PYTHON% -m pip install -q -r code/requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Default pip failed, trying Tsinghua mirror...
    %PYTHON% -m pip install -q -r code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo Checking EasyOCR models...
%PYTHON% code/download_ocr_models.py
if %errorlevel% neq 0 (
    echo [WARN] OCR models not fully downloaded. Auto-annotation may not work.
    echo        You can re-run: %PYTHON% code/download_ocr_models.py
)

echo.
echo Starting UI - open http://127.0.0.1:7860 in your browser
%PYTHON% ui/app.py
pause
