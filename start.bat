@echo off
REM start.bat — Launch API + Frontend (Windows)

set ROOT=%~dp0
set UVICORN=%ROOT%.venv\Scripts\uvicorn.exe

if not exist "%UVICORN%" (
    echo ERROR: virtualenv not found.
    echo        Run "poetry install" first.
    pause
    exit /b 1
)

echo Starting API on http://localhost:8000 ...
start "HK Travel Guide - API" cmd /k "cd /d "%ROOT%backend" && "%UVICORN%" api:app --reload --port 8000"

echo Starting Frontend on http://localhost:5173 ...
start "HK Travel Guide - Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

echo.
echo Both services launched in separate windows.
