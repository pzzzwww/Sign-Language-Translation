@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title SL-Translator
cd /d "%~dp0"

set PYTHON=python
set PORT=8000

echo.
echo ================================================
echo   SL-Translator
echo ================================================
echo.


if not exist ".deps_installed" (
    echo [*] Installing deps...
    "%PYTHON%" -m pip install -r requirements.txt -q
    echo. > ".deps_installed"
)

for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    set IP=%%a
    set IP=!IP:~1!
    if not "!IP!"=="127.0.0.1" goto :show
)
:show
if "%IP%"=="" set IP=127.0.0.1

echo.
echo ================================================
echo   Local:  https://localhost:%PORT%
echo   LAN:    https://%IP%:%PORT%
echo ================================================
echo.
echo   First time: click Advanced -^> Proceed
echo   LAN users: run firewall.bat as Admin first
echo ================================================
echo.

start "" https://localhost:%PORT%
"%PYTHON%" -m src.backend.main
pause
