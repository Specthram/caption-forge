@echo off
REM Caption Forge - launcher (FastAPI + React). Run install.bat first if the
REM venv or the web build is missing.

setlocal
title Caption Forge

if not exist ".\venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please run install.bat first to set up Caption Forge.
    echo.
    pause
    exit /b 1
)

call ".\venv\Scripts\activate.bat"

:launch
REM Build the front on every start/restart. FastAPI serves web\dist; the
REM restart button loops back here, so a code change is live after restart.
echo Building React front-end...
pushd web
call npm run build
popd
if not exist ".\web\dist\index.html" (
    echo FAILED to build the React front-end. Run install.bat again.
    pause
    exit /b 1
)

echo Launching Caption Forge on http://127.0.0.1:7776 ...
python -m uvicorn server.main:app --host 127.0.0.1 --port 7776

REM Exit code 3 = the in-app "Restart server" button (src.constants.
REM RESTART_EXIT_CODE): relaunch. Any other code (crash, normal close)
REM falls through instead of looping.
if %errorlevel% equ 3 (
    echo.
    echo Restarting Caption Forge...
    goto :launch
)

echo.
echo Application closed. Press any key to exit.
pause
endlocal
