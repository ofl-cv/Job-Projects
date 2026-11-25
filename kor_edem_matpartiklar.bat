@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist edem_env\Scripts\python.exe (
    echo Kor "installera_edem_verktyg.bat" forst ^(en gang^).
    pause
    exit /b 1
)
edem_env\Scripts\python edem_food_clumps.py %*
echo.
pause
