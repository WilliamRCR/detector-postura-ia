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
title Detector de Postura IA
"%PY%" app_gui.py
if errorlevel 1 pause
