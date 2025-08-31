@echo off
echo ========================================
echo Ultra Signals Trading Helper Supervisor
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

echo Starting supervisor with automatic restart capability...
echo.
echo Configuration:
echo - Max restarts: 10
echo - Base backoff: 60 seconds
echo - Max backoff: 1 hour
echo - Output capture: Last 100 lines
echo.
echo Press Ctrl+C to stop the supervisor
echo.

REM Run the supervisor
python supervisor.py --max-restarts 10 --backoff-base 60 --backoff-max 3600

echo.
echo Supervisor has stopped.
pause
