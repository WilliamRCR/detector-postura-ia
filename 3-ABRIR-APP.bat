@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Detector de Postura IA
".venv\Scripts\python.exe" app_gui.py
if errorlevel 1 pause
