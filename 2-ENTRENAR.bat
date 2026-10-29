@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Paso 2 - Entrenar el modelo

echo ===============================================
echo   PASO 2 de 3 - ENTRENAR EL MODELO
echo ===============================================
echo.
".venv\Scripts\python.exe" train.py
echo.
echo ===============================================
echo   Si arriba dice "Modelo guardado", abre 3-ABRIR-APP.bat
echo ===============================================
pause
