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
title Paso 1 - Grabar tus posturas

echo ===============================================
echo   PASO 1 de 3 - GRABAR TUS POSTURAS
echo ===============================================
echo.
echo Son 4 grabaciones de 20 segundos cada una.
echo Ponte DE FRENTE a la camara, con los dos hombros visibles.
echo.
pause

echo.
echo --- 1 de 4: POSTURA CORRECTA, SENTADO ---
echo Sientate bien: espalda recta, hombros parejos, cabeza atras.
pause
"%PY%" collect_data.py 1 20 --contexto sentado

echo.
echo --- 2 de 4: POSTURA CORRECTA, DE PIE ---
echo Ponte de pie, derecho.
pause
"%PY%" collect_data.py 1 20 --contexto de_pie

echo.
echo --- 3 de 4: POSTURA INCORRECTA, SENTADO ---
echo Encorvate: hombros caidos y cabeza adelantada.
pause
"%PY%" collect_data.py 2 20 --contexto sentado

echo.
echo --- 4 de 4: POSTURA INCORRECTA, DE PIE ---
echo De pie pero mal: encorvado o ladeado.
pause
"%PY%" collect_data.py 2 20 --contexto de_pie

echo.
echo ===============================================
echo   LISTO. Ahora abre 2-ENTRENAR.bat
echo ===============================================
pause
