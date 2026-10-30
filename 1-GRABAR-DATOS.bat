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
title Paso 1 - Grabar tus posturas

echo ===============================================
echo   PASO 1 de 3 - GRABAR TUS POSTURAS
echo ===============================================
echo.
echo Son 8 grabaciones de 20 segundos, en 4 situaciones:
echo   sentado de frente, sentado de perfil, de pie de frente y de pie de perfil
echo   ^(en cada una, una postura correcta y una incorrecta^).
echo.
echo DE FRENTE: mirando a la camara, con la cabeza y los dos hombros visibles.
echo DE PERFIL: completamente de lado a la camara ^(no a 45 grados^), con la
echo            nariz y las orejas visibles. Da igual hacia que lado mires.
echo.
echo Mientras mas variedad ^(distancia, luz, ropa, otro dia^), mejor aprende.
echo Conviene hacer DOS pasadas ^(mejor en dias distintos^): las grabaciones se suman
echo y asi cada situacion queda con mas de una sesion.
echo.
pause

echo.
echo --- 1 de 8: SENTADO DE FRENTE - POSTURA CORRECTA ---
echo Sientate bien de frente: espalda recta, hombros parejos, cabeza atras.
pause
"%PY%" collect_data.py 1 20 --vista frente --contexto sentado

echo.
echo --- 2 de 8: SENTADO DE FRENTE - POSTURA INCORRECTA ---
echo Encorvate de frente: hombros caidos y cabeza adelantada.
pause
"%PY%" collect_data.py 2 20 --vista frente --contexto sentado

echo.
echo --- 3 de 8: DE PIE DE FRENTE - POSTURA CORRECTA ---
echo Ponte de pie, derecho, de frente a la camara.
pause
"%PY%" collect_data.py 1 20 --vista frente --contexto de_pie

echo.
echo --- 4 de 8: DE PIE DE FRENTE - POSTURA INCORRECTA ---
echo De pie pero mal: encorvado o ladeado.
pause
"%PY%" collect_data.py 2 20 --vista frente --contexto de_pie

echo.
echo --- 5 de 8: SENTADO DE PERFIL - POSTURA CORRECTA ---
echo Sientate de lado a la camara, espalda recta y la oreja sobre el hombro.
pause
"%PY%" collect_data.py 1 20 --vista perfil --contexto sentado

echo.
echo --- 6 de 8: SENTADO DE PERFIL - POSTURA INCORRECTA ---
echo Sentado de lado: espalda encorvada y cabeza adelantada.
pause
"%PY%" collect_data.py 2 20 --vista perfil --contexto sentado

echo.
echo --- 7 de 8: DE PIE DE PERFIL - POSTURA CORRECTA ---
echo De pie y de lado a la camara, derecho, oreja sobre el hombro.
pause
"%PY%" collect_data.py 1 20 --vista perfil --contexto de_pie

echo.
echo --- 8 de 8: DE PIE DE PERFIL - POSTURA INCORRECTA ---
echo De pie y de lado: encorvado, con la cabeza adelantada.
pause
"%PY%" collect_data.py 2 20 --vista perfil --contexto de_pie

echo.
echo ===============================================
echo   LISTO. Ahora abre 2-ENTRENAR.bat
echo   ^(te dira si falta alguna situacion o si conviene otra pasada^)
echo ===============================================
pause
