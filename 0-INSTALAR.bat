@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalar PosturIA

echo ===============================================
echo   INSTALACION - solo hace falta una vez
echo ===============================================
echo.

if exist ".venv\Scripts\python.exe" goto instalar
if exist "venv\Scripts\python.exe" goto instalar

rem Las dependencias solo existen para Python 3.10, 3.11 y 3.12.
set "PY="
for %%V in (3.11 3.12 3.10) do (
    if not defined PY (
        py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo No encontre Python 3.10, 3.11 ni 3.12 en este equipo.
    echo Instala Python 3.11 desde https://www.python.org/downloads/ y vuelve a abrir este archivo.
    pause
    exit /b 1
)

echo Creando el entorno con: %PY%
%PY% -m venv .venv
if errorlevel 1 (
    echo No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

:instalar
set "VENV=.venv\Scripts\python.exe"
if not exist "%VENV%" set "VENV=venv\Scripts\python.exe"
echo Instalando dependencias ^(puede tardar varios minutos^)...
"%VENV%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo La instalacion fallo. Revisa el mensaje de arriba.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   LISTO. Ahora abre 1-GRABAR-DATOS.bat
echo ===============================================
pause
