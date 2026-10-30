@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No encuentro el entorno virtual ^(.venv o venv^).
    echo Abre primero 0-INSTALAR.bat ^(crea el entorno e instala todo^).
    pause
    exit /b 1
)
title PosturIA
"%PY%" app_gui.py
if errorlevel 1 pause
