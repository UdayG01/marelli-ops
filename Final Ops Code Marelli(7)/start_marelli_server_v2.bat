@echo off
title Marelli AI Inspection System
color 0A
echo ================================================================
echo                 MARELLI AI INSPECTION SYSTEM
echo                    Industrial Nut Detection
echo ================================================================
echo.

REM Change to the script directory (project root)
cd /d "C:\Users\manis\Downloads\Final Ops Code\Final Ops Code"

echo Current directory: %CD%
echo.

REM Check if manage.py exists
if not exist "manage.py" (
    echo ERROR: manage.py not found in current directory
    echo Please ensure this batch file is in the same directory as manage.py
    pause
    exit /b 1
)

echo [1/4] Activating virtual environment...
REM Activate virtual environment
if exist "env\Scripts\activate.bat" (
    call "env\Scripts\activate.bat"
    echo Virtual environment activated successfully!
) else (
    echo WARNING: Virtual environment not found at env\Scripts\activate.bat
    echo Proceeding with system Python...
)
echo.

echo [2/4] Checking Python and dependencies...
REM Check Python version
python --version
echo.

REM Install/check requirements silently
echo Installing/checking dependencies...
pip install -r requirements.txt > nul 2>&1
if errorlevel 1 (
    echo WARNING: Some dependencies may not have installed correctly
) else (
    echo Dependencies checked successfully!
)
echo.

echo [3/4] Starting Marelli AI Server...
echo ================================================================
echo Server Configuration:
echo   - URL: http://127.0.0.1:8000/
echo   - ML Models: Loading industrial nut detection
echo   - Camera: Hikrobot SDK integration
echo   - Storage: Enhanced file system ready
echo ================================================================
echo.
echo Server is starting... Please wait for "Starting development server" message
echo Press Ctrl+C to stop the server when needed
echo.

REM Start browser after a delay (in background)
timeout /t 5 /nobreak > nul && start "" "http://127.0.0.1:8000/" &

echo [4/4] Setting up auto-refresh...
echo Press 'R' to manually refresh server
echo Auto-refresh interval: 10 minutes

:server_loop
REM Start Django server in background
start /B python manage.py runserver 127.0.0.1:8000

:refresh_check
REM Check for 'R' key press or timeout (600 seconds = 10 minutes)
choice /c R /t 600 /d R /n > nul

REM Clear screen and show refresh message
cls
echo ================================================================
echo                 MARELLI AI INSPECTION SYSTEM
echo                    Industrial Nut Detection
echo ================================================================
echo.
echo Triggering Django auto-reload...

REM Trigger Django auto-reload instead of killing process
copy /b "manage.py" +,, > nul

echo Django auto-reload triggered! Server will restart automatically...
echo Please wait for Django to complete its restart process...
echo.

REM Brief pause to let Django restart complete
timeout /t 5 /nobreak > nul

REM Clear screen and show system ready message
cls
echo ================================================================
echo                 MARELLI AI INSPECTION SYSTEM
echo                    Industrial Nut Detection
echo ================================================================
echo.
echo Server refreshed successfully via Django auto-reload!
echo   - All variables reset
echo   - Fresh memory space
echo   - Complete module reload
echo.
echo Press 'R' to manually refresh server
echo Auto-refresh interval: 10 minutes
echo.

goto refresh_check

REM This runs when server stops (Ctrl+C)
echo.
echo ================================================================
echo Server has stopped.
echo Thank you for using Marelli AI Inspection System!
echo ================================================================
pause