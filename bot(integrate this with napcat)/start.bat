@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ========================================
echo   QQ Style Bot
echo ========================================
echo.
echo [*] Starting NoneBot2 on port 7001...
echo.
set "PORT_IN_USE="
for /f "tokens=5" %%p in ('netstat -ano -p tcp ^| findstr /R /C:":7001 .*LISTENING"') do (
    set "PORT_IN_USE=1"
    echo [!] Port 7001 is already in use by PID %%p
)
if defined PORT_IN_USE (
    echo [!] Close the old bot window or stop those PIDs, then start again.
    echo.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" bot.py
pause
