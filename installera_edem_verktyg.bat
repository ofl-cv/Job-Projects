@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo ==============================================
echo  EDEM-verktyg: engångsinstallation
echo ==============================================
echo.

rem ---- hitta Python -------------------------------------------------------
set "PYCMD="
py -3 --version >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD python --version >nul 2>nul && set "PYCMD=python"
if not defined PYCMD (
    echo Python hittades inte pa datorn.
    echo Installera fran https://www.python.org/downloads/
    echo och bocka i "Add Python to PATH" i installationen. Kor sedan om denna fil.
    pause
    exit /b 1
)

rem ---- skapa miljon och installera paket ----------------------------------
echo Skapar Python-miljon "edem_env" ^(gors bara en gang^)...
%PYCMD% -m venv edem_env || goto :fail
echo Installerar numpy och h5py, kan ta en minut...
edem_env\Scripts\python -m pip install --quiet --upgrade pip
edem_env\Scripts\python -m pip install --quiet "numpy<2" h5py || goto :fail

rem ---- leta efter edempy-hjulet fran EDEM-installationen ------------------
set "WHEEL="
for /f "delims=" %%W in ('dir /b /s "%~dp0edempy*.whl" 2^>nul') do set "WHEEL=%%W"
if not defined WHEEL for /f "delims=" %%W in ('dir /b /s "C:\Program Files\Altair\edempy*.whl" 2^>nul') do set "WHEEL=%%W"
if defined WHEEL (
    echo Installerar EDEMpy: !WHEEL!
    edem_env\Scripts\python -m pip install --quiet "!WHEEL!" || goto :fail
) else (
    echo.
    echo OBS: hittade inget edempy-hjul nu, men verktyget letar ocksa sjalvt
    echo efter EDEM nar det kors. Om det klagar da: kopiera edempy-*.whl
    echo fran EDEM-installationens EDEMpy-mapp hit och kor om denna fil.
)

echo.
echo Klart! Dubbelklicka pa "kor_edem_replicator.bat" for att kora verktyget.
pause
exit /b 0

:fail
echo.
echo Nagot gick fel, se felmeddelandet ovanfor.
pause
exit /b 1
