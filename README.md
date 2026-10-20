# Detector de Postura Corporal

Sistema de detección de postura corporal en tiempo real usando visión por computadora (MediaPipe) y una red neuronal (scikit-learn). Te evalúa unos segundos al iniciar, te dice por voz y en pantalla qué mejorar, y luego te corrige de forma continua.

## Características

- **Cámara activa desde el inicio**: en `app_gui.py` y `app_web.py` la cámara se enciende al abrir la app y te saluda por voz antes de empezar.
- **Evaluación inicial de 30s**: al presionar INICIAR mantienes tu postura habitual; al terminar ves un checklist (✓/✗ de hombros, cabeza y cuello, con el % de cada uno) y escuchas un veredicto corto.
- **Coaching continuo**: después de la evaluación te avisa (por voz y texto) cuando tu postura es incorrecta y te felicita una vez cuando la corriges.
- **Voz local**: usa `pyttsx3` (sin internet). Se puede silenciar con el botón "Voz".
- **Predicción suavizada**: voto mayoritario de los últimos frames para evitar parpadeo entre CORRECTA/INCORRECTA.
- **Feedback independiente de la distancia**: hombros, cabeza y cuello se miden con razones normalizadas por tu tamaño corporal (ancho de hombros, largo de torso), no con coordenadas crudas.
- **Detección sentado / de pie**: a partir del ángulo cadera-rodilla-tobillo. Si las piernas no se ven en el encuadre lo indica ("Piernas no visibles") en vez de adivinar.
- **Estadísticas de sesión**: % de tiempo en buena postura durante toda la sesión.
- **Dataset por código**: `collect_data.py` graba desde la webcam y aumenta los datos automáticamente (espejo + ruido).

---

## Requisitos Previos

### 1. Python 3.10 o superior
- **Windows**: Descarga de [python.org](https://www.python.org/downloads/) o desde la Microsoft Store
- **Mac**: `brew install python3` o descargarlo de python.org
- **Linux**: `sudo apt install python3 python3-tk` (Ubuntu/Debian) o `sudo yum install python3 python3-tkinter` (Fedora)

**IMPORTANTE**: Durante la instalación de Windows, marca la opción **"Add Python to PATH"**

### 2. Cámara web
- Necesitas una cámara integrada o externa conectada

### 3. Altavoces o audífonos (opcional)
- Para el feedback por voz. Sin audio el detector funciona igual.

---

## Instalación

### Paso 1: Descargar el proyecto

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd detector-postura-ia
```

O si ya tienes el proyecto descargado, simplemente abre la carpeta.

### Paso 2: Crear un entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

> En Windows también puedes saltarte la activación y llamar directamente al Python del entorno: `.\venv\Scripts\python.exe app_gui.py`. Esto evita problemas con la política de ejecución de PowerShell.

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas probadas en conjunto (opencv-python, mediapipe 0.10.13, numpy, scikit-learn, joblib, pyttsx3, flask, pillow).

---

## Uso del Proyecto

El flujo completo es: **recolectar datos → entrenar → usar el detector**.

### 1. Recolectar datos (por código, desde la webcam)

```bash
python collect_data.py <etiqueta> [segundos] --contexto <sentado|de_pie|mixto>
```

- `etiqueta`: `1` = postura correcta, `2` = postura incorrecta
- `segundos`: duración de la grabación (por defecto 20)
- `--contexto`: solo para llevar registro de qué combinaciones ya cubriste

Tienes 3 segundos para prepararte y luego se graba automáticamente (no hay que ir foto por foto). Presiona `q` para cancelar.

**Repite para cada combinación** para que el modelo generalice bien:

```bash
python collect_data.py 1 20 --contexto sentado
python collect_data.py 1 20 --contexto de_pie
python collect_data.py 2 20 --contexto sentado
python collect_data.py 2 20 --contexto de_pie
```

Para la postura incorrecta, mantén errores típicos: hombros caídos o desnivelados, cabeza adelantada, espalda encorvada, etc. Conviene variar el tipo de error entre corridas.

Detalles:
- Cada muestra real se **aumenta x3** (original + espejo con intercambio de lado izquierdo/derecho + variante con ruido leve).
- Cada corrida **se agrega** a `dataset_posturas.npz` (no lo sobreescribe).
- Cada corrida queda registrada en `dataset_sessions.json` (fecha, etiqueta, contexto, muestras) para que verifiques qué combinaciones te faltan.
- Alternativa: si ya tienes fotos, colócalas en `dataset/correcta` y `dataset/incorrecta` y ejecuta `python collect_from_images.py` (esto **sobreescribe** `dataset_posturas.npz` con lo que haya en esas carpetas).

### 2. Entrenar el modelo

```bash
python train.py
```
- Lee `dataset_posturas.npz`, entrena una red neuronal (MLP) y genera `modelo_postura.pkl` y `scaler.pkl`
- Vuelve a ejecutarlo cada vez que agregues datos nuevos

### 3. Usar el detector

Hay cuatro formas de ejecutarlo. Las dos primeras son las recomendadas.

#### Interfaz de escritorio (Tkinter)
```bash
python app_gui.py
```

#### Interfaz web (Flask)
```bash
python app_web.py
```
Luego abre `http://localhost:5000` en el navegador. El servidor enciende la cámara apenas arranca y la voz suena en el equipo donde corre el servidor.

Flujo en ambas interfaces:
1. **Vista previa**: la cámara está activa y el programa te saluda por voz. Ubícate frente a la cámara (sentado o de pie).
2. **INICIAR**: comienza la evaluación de 30 segundos. Mantén tu postura habitual; durante este tiempo no te corrige en vivo, para no sesgar el resultado. El estado muestra "EVALUANDO... Ns".
3. **Resultado**: durante ~10 segundos ves un checklist como este (lo que hay que mejorar aparece primero) y escuchas un veredicto corto por voz:
   ```
   60% correcta en la evaluación
   ✗ Hombros nivelados (58%)
   ✗ Cabeza no adelantada (40%)
   ✓ Buena altura de cabeza/cuello (85%)
   ```
4. **Coaching continuo**: el estado pasa a CORRECTA/INCORRECTA en tiempo real, con el consejo específico (por ejemplo "Endereza los hombros") y la voz te avisa. Debajo ves el contexto detectado (Sentado / De pie) y la estadística de sesión.
5. **DETENER**: vuelve a la vista previa (la cámara sigue encendida). Para apagarla, cierra la ventana (`app_gui.py`) o detén el servidor con `Ctrl+C` (`app_web.py`). Puedes volver a presionar INICIAR para una nueva evaluación.

El botón **Voz** (checkbox en `app_gui.py`, botón "VOZ: ON/OFF" en `app_web.py`) silencia o reactiva el audio sin cerrar la app.

#### Versiones de consola (ventana de OpenCV)
```bash
python predict.py
python main.py
```
- Muestran la cámara con el estado, el consejo, el contexto (sentado/de pie) y la estadística de sesión; presiona `q` para salir.
- Tienen voz y predicción suavizada, pero **no** incluyen la evaluación inicial de 30s ni el saludo: arrancan directo en coaching continuo.
- `main.py` es casi idéntico a `predict.py` (distinto layout de pantalla); ambos requieren el modelo entrenado.

### Cómo funciona el feedback

Hay dos señales independientes:
- **Estado CORRECTA/INCORRECTA**: lo decide la red neuronal entrenada con tu dataset (99 valores = 33 landmarks de MediaPipe × x, y, z).
- **Consejo específico y checklist**: lo calculan reglas geométricas en `posture_metrics.py` sobre tres aspectos: hombros nivelados, cabeza no adelantada y altura de cabeza/cuello. Los umbrales son proporciones del tamaño de tu cuerpo, y se pueden ajustar en `PostureThresholds`.

Parámetros que puedes cambiar:
- `assessment.py`: `duration=30.0` (segundos de evaluación) y `good_threshold=70.0` (% mínimo para considerarlo buena postura)
- `voice_feedback.py`: `cooldown=6.0` (segundos antes de repetir el mismo aviso) y `rate=175` (velocidad de la voz)
- `smoothing.py`: tamaño de la ventana de suavizado (en cada script se crea con `window=7` para el estado y `window=5` para el consejo)

---

## Solución de Problemas Comunes

### ❌ "pip no se reconoce como comando interno o externo"
**Solución**: 
- Windows: Busca "Python" en el menú inicio → "Modify Python" → repair installation y marca "Add to PATH"
- O usa: `py -m pip install -r requirements.txt`

### ❌ El `venv` no funciona ("No Python at ...")
**Causa**: el entorno virtual apunta a una instalación de Python que ya no existe (por ejemplo, desinstalaste esa versión).

**Solución**: bórralo y vuélvelo a crear:
```bash
rmdir /s /q venv
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### ❌ "ModuleNotFoundError: No module named 'cv2'" (o mediapipe, flask, PIL, pyttsx3...)
**Solución**: instala las dependencias en el entorno que estás usando:
```bash
pip install -r requirements.txt
```

### ❌ "Modelo no encontrado. Ejecuta train.py primero."
**Causa**: aún no existen `modelo_postura.pkl` / `scaler.pkl` (están en `.gitignore`, así que no vienen al clonar el repo).

**Solución**: recolecta datos (paso 1) y ejecuta `python train.py`. Con el modelo ausente la interfaz web solo muestra la vista previa y deshabilita INICIAR.

### ❌ Error de cámara no encontrada
**Solución**:
- Verifica que otra aplicación no esté usando la cámara
- En `app_gui.py`, `app_web.py`, `predict.py`, `main.py` y `collect_data.py`, busca la línea `cv2.VideoCapture(0)` y cambia el `0` a `1`, `2`, etc. según tu cámara
- En `app_gui.py` aparece "Error: no se pudo abrir la cámara" y INICIAR queda deshabilitado; en `app_web.py` verás el mensaje en lugar del video

### ❌ No se escucha la voz / "Voz no disponible"
**Solución**:
- Revisa que el volumen y la salida de audio del equipo estén bien
- Si `pyttsx3` no puede iniciar el motor de voz (en Linux requiere `espeak`: `sudo apt install espeak`), el detector sigue funcionando sin audio y la interfaz lo indica
- En `app_web.py` la voz suena en el equipo donde corre el servidor, no en el navegador

### ❌ "ImportError: DLL load failed" (Windows)
**Solución**:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```
(Nota: `opencv-python-headless` no incluye ventanas, así que `predict.py`, `main.py` y `collect_data.py` no podrán mostrar video.)

### ❌ Error con numpy, scikit-learn o incompatibilidad de versiones
**Síntoma**:  
Errores como "_ARRAY_API not found", "numpy.core._multiarray_umath failed to import", o `InconsistentVersionWarning` al cargar `modelo_postura.pkl`/`scaler.pkl`

**Causa**:  
`requirements.txt` ya fija las versiones exactas probadas en conjunto con `mediapipe==0.10.13` (numpy 2.x funciona bien con esta versión de mediapipe). Si el error persiste es porque tienes paquetes de una instalación previa con otras versiones.

**Solución**: reinstala limpio desde `requirements.txt`:
```bash
pip install -r requirements.txt --force-reinstall
```
Si el modelo (`modelo_postura.pkl`) fue entrenado con otra versión de scikit-learn, el warning no rompe nada pero conviene reentrenarlo: `python train.py`.

### ❌ El entrenamiento falla o la precisión es mala
**Solución**: probablemente hay pocas muestras o están desbalanceadas. Revisa `dataset_sessions.json` y completa las combinaciones que falten:
```bash
python collect_data.py 1 30 --contexto sentado    # más muestras correctas
python collect_data.py 2 30 --contexto sentado    # más muestras incorrectas
python train.py
```

### ❌ El veredicto o el contexto no parecen correctos
- **"Piernas no visibles"**: es normal si la cámara solo capta torso y cabeza; la detección sentado/de pie necesita ver cadera, rodilla y tobillo.
- **"Ubícate de frente a la cámara"**: aparece cuando el ancho de hombros se ve demasiado pequeño (por ejemplo, estás de perfil o muy lejos).
- Si el checklist marca ✗ en un aspecto que consideras correcto, ajusta los umbrales en `PostureThresholds` (`posture_metrics.py`).

### ❌ La cámara se ve muy lenta o Lag
**Solución**: Cierra otras aplicaciones que usen la cámara o reduce la resolución en el código cambiando:
```python
cap.set(3, 640)  # ancho
cap.set(4, 480)  # alto
```

---

## Estructura de Archivos

```
detector-postura-ia/
├── app_gui.py              # Interfaz de escritorio (Tkinter)
├── app_web.py              # Interfaz web (Flask)
├── templates/index.html    # Frontend de la interfaz web
├── predict.py              # Detector en ventana de OpenCV (sin evaluación inicial)
├── main.py                 # Igual que predict.py, con otro layout de pantalla
├── collect_data.py         # Recolectar datos desde la webcam (automático, con aumento de datos)
├── collect_from_images.py  # Alternativa: generar el dataset a partir de fotos ya tomadas
├── train.py                # Entrenar el modelo
├── posture_metrics.py      # Métricas normalizadas, aspectos, sentado/de pie y estadísticas de sesión
├── assessment.py           # Evaluación inicial de 30s (postura habitual)
├── smoothing.py            # Suavizado temporal de predicciones (voto mayoritario)
├── voice_feedback.py       # Feedback por voz (TTS local, con anti-spam)
├── requirements.txt        # Dependencias con versiones fijas
├── dataset/                # (opcional) fotos para collect_from_images.py
├── dataset_posturas.npz    # Dataset acumulado (se crea/actualiza al ejecutar)
├── dataset_sessions.json   # Registro de qué combinaciones ya recolectaste
├── modelo_postura.pkl      # Modelo entrenado (se crea con train.py)
└── scaler.pkl              # Escalador (se crea con train.py)
```

---

## Notas

- Para mejores resultados, usa buena iluminación
- Mantén una distancia adecuada de la cámara (1-2 metros) y asegúrate de que se vean tus hombros
- El modelo funciona mejor cuando se entrena con datos de la misma cámara y el mismo encuadre en el que lo vas a usar
- Todo se procesa en tu equipo: no se envían imágenes ni audio a ningún servicio externo, y el video no se guarda (solo se guardan landmarks numéricos al recolectar datos)
- Si tus amigos tienen problemas, sugiere que actualicen sus drivers de cámara
