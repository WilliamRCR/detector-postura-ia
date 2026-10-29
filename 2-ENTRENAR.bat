@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No encuentro el entorno virtual ^(.venv o venv^).
    echo Crealo con:  python -m venv .venv
    echo e instala:   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
title Paso 2 - Entrenar el modelo

echo ===============================================
echo   PASO 2 de 3 - ENTRENAR EL MODELO
echo ===============================================
echo.
"%PY%" train.py
echo.
echo ===============================================
echo   Si arriba dice "Modelo guardado", abre 3-ABRIR-APP.bat
echo ===============================================
pause
