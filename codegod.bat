@echo off
REM Code-God Launcher Script for Windows

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is required but not installed.
    echo Please install Python 3.8 or higher from https://www.python.org/
    exit /b 1
)

REM Check if venv exists, create if not
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/update dependencies
if not exist "venv\.deps_installed" (
    echo Installing dependencies...
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
    type nul > venv\.deps_installed
) else (
    REM Check if requirements.txt is newer
    for %%i in (requirements.txt) do set req_time=%%~ti
    for %%i in (venv\.deps_installed) do set deps_time=%%~ti
    if "!req_time!" gtr "!deps_time!" (
        echo Updating dependencies...
        python -m pip install --quiet -r requirements.txt
        type nul > venv\.deps_installed
    )
)

REM Run Code-God
python codegod.py %*
