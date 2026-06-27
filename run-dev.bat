@echo off
REM Caption Forge - development launcher. Runs the FastAPI backend (uvicorn,
REM auto-reload) and the Vite dev server (HMR, proxies /api and /ws to the
REM backend) side by side. Open http://127.0.0.1:5173 for the live front-end.

setlocal
title Caption Forge (dev)

if not exist ".\venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

call ".\venv\Scripts\activate.bat"

echo Starting FastAPI backend (uvicorn --reload) on port 7776...
start "Caption Forge API" cmd /k python -m uvicorn server.main:app --host 127.0.0.1 --port 7776 --reload

echo Starting Vite dev server on port 5173...
pushd web
call npm run dev
popd

endlocal
