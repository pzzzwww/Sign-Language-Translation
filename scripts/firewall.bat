@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
:: Run as Administrator to open firewall for LAN access

netsh advfirewall firewall add rule name="SL-Translator" dir=in action=allow protocol=TCP localport=8000 profile=any >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Port 8000 opened
) else (
    netsh advfirewall firewall show rule name="SL-Translator" >nul 2>&1
    if %errorlevel% equ 0 (echo [OK] Rule exists) else (echo [FAIL] Run as Admin)
)

for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    set IP=%%a
    set IP=!IP:~1!
    if not "!IP!"=="127.0.0.1" goto :done
)
:done

echo.
echo ================================================
echo   LAN URL: https://!IP!:8000
echo ================================================
echo.
echo   First time: Advanced -^> Proceed
pause
