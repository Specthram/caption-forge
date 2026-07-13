@echo off
REM Caption Forge - interactive installer (Windows).
REM Creates the virtual environment and installs everything needed. Use
REM run.bat afterwards to launch the app.

setlocal EnableExtensions EnableDelayedExpansion
title Caption Forge - Installer

echo ==========================================================
echo   Caption Forge - Installation
echo ==========================================================
echo.
echo This will create a local Python environment in .\venv and
echo install Caption Forge and its dependencies.
echo.
echo Requirements: Python 3.12 and an NVIDIA GPU with up-to-date
echo drivers (CUDA 12.x).
echo.

REM --- Python 3.12 check (required by the CUDA llama-cpp wheel) ---
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.12 was not found via the 'py' launcher.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo then run this installer again.
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('py -3.12 --version') do echo Found %%v
echo.

REM --- CUDA version choice ---
echo Which CUDA build should be installed?
echo   [1] CUDA 12.8  (recommended - RTX 20/30/40/50 series)
echo   [2] CUDA 12.6  (RTX 20/30/40 series)
echo   [3] CUDA 12.4  (older drivers)
echo.
echo If unsure, choose 1. RTX 50-series (5080, 5090...) REQUIRES 12.8.
set "CUDA=128"
set /p "SEL=Enter choice [1-3] (default 1): "
if "!SEL!"=="2" set "CUDA=126"
if "!SEL!"=="3" set "CUDA=124"
echo Selected: CUDA 12.!CUDA:~-1!  (cu!CUDA!)
echo.

REM --- GGUF support choice ---
echo GGUF support lets you caption with .gguf vision models (Qwen3-VL,
echo Gemma...). It installs llama-cpp-python (JamePeng CUDA build, ~330 MB).
echo Skip it if you only use .safetensors models.
set "GGUF=Y"
set /p "GG=Install GGUF support? [Y/n] (default Y): "
if /i "!GG!"=="n" set "GGUF=N"
set "LLVER="
if /i "!GGUF!"=="Y" (
    echo.
    echo Leave blank for the latest available build, or type a specific
    echo version such as 0.3.40.
    set /p "LLVER=llama-cpp version (blank = latest): "
)
echo.

REM --- Virtual environment ---
if not exist ".\venv\Scripts\activate.bat" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv venv
    if !errorlevel! neq 0 (
        echo FAILED to create the virtual environment.
        pause
        exit /b 1
    )
)
call ".\venv\Scripts\activate.bat"

REM --- torch first, from the CUDA index ---
REM Must precede requirements.txt: transformers/accelerate/timm depend on
REM torch, so installing them first would pull the CPU build from PyPI and the
REM CUDA install would then be skipped as "already satisfied".
echo.
echo Installing torch / torchvision (cu!CUDA!)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu!CUDA!
if !errorlevel! neq 0 (
    echo FAILED to install torch.
    pause
    exit /b 1
)

REM --- Core dependencies ---
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo FAILED to install requirements.
    pause
    exit /b 1
)

REM --- GGUF support (optional) ---
if /i not "!GGUF!"=="Y" goto done

echo.
echo Resolving llama-cpp-python wheel for cu!CUDA!...
set "WHL_URL="
set "WHL_SHA="
set "WHL_NAME="
for /f "usebackq tokens=1,2,3 delims= " %%a in (`py -3.12 tools\resolve_llama_wheel.py cu!CUDA! "!LLVER!"`) do (
    set "WHL_URL=%%a"
    set "WHL_SHA=%%b"
    set "WHL_NAME=%%c"
)
if not defined WHL_URL (
    echo Could not find a matching llama-cpp wheel for cu!CUDA! !LLVER!.
    echo Check your CUDA choice / version, or see the JamePeng releases:
    echo   https://github.com/JamePeng/llama-cpp-python/releases
    pause
    exit /b 1
)

if not exist "!WHL_NAME!" (
    echo Downloading !WHL_NAME! ...
    curl -L -o "!WHL_NAME!" "!WHL_URL!"
    if !errorlevel! neq 0 (
        echo FAILED to download the llama-cpp wheel.
        pause
        exit /b 1
    )
)

if not "!WHL_SHA!"=="-" (
    echo Verifying checksum...
    certutil -hashfile "!WHL_NAME!" SHA256 | findstr /i "!WHL_SHA!" >nul
    if !errorlevel! neq 0 (
        echo ERROR: checksum mismatch for !WHL_NAME! - deleting it.
        echo Re-run this installer to download it again.
        del "!WHL_NAME!"
        pause
        exit /b 1
    )
)

echo Installing !WHL_NAME! ...
pip install "!WHL_NAME!"
if !errorlevel! neq 0 (
    echo FAILED to install the llama-cpp wheel.
    pause
    exit /b 1
)

REM Remove the downloaded wheel now that it is installed (~330 MB freed).
del "!WHL_NAME!"

:done

REM --- React front-end (web/) ---
REM The new UI is a Vite + React app served by FastAPI. It needs Node.js to
REM install its packages and build the production bundle once.
echo.
echo ==========================================================
echo   React front-end setup (web\)
echo ==========================================================
where node >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Node.js was not found on PATH.
    echo The Python backend is installed, but the React UI needs a build.
    echo Install Node.js 20+ from https://nodejs.org/ then run:
    echo     cd web ^&^& npm ci ^&^& npm run build
    goto finish
)
for /f "delims=" %%v in ('node --version') do echo Found Node %%v
pushd web
echo Installing npm dependencies...
call npm ci
if errorlevel 1 (
    echo npm ci failed - falling back to npm install...
    call npm install
    if errorlevel 1 (
        echo FAILED to install the React dependencies.
        popd
        pause
        exit /b 1
    )
)
echo Building the production bundle...
call npm run build
if errorlevel 1 (
    echo FAILED to build the React front-end.
    popd
    pause
    exit /b 1
)
popd

:finish
echo.
echo ==========================================================
echo   Installation complete. Launch the app with run.bat
echo ==========================================================
echo.
pause
endlocal
