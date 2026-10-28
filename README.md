# Detector de Postura Corporal

Sistema de detección de postura corporal en tiempo real usando visión por computadora (MediaPipe) y una red neuronal (scikit-learn). Te evalúa unos segundos al iniciar, calibra los umbrales a tu cuerpo, te dice por voz y en pantalla qué mejorar, y luego te corrige de forma continua.

## Características

- **Cámara activa desde el inicio**: en `app_gui.py` y `app_web.py` la cámara se enciende al abrir la app y te saluda por voz antes de empezar.
- **Evaluación inicial de 30s**: al presionar INICIAR mantienes tu postura habitual; al terminar ves un checklist (✓/✗ de hombros, cabeza y cuello, con el % de cada uno) y escuchas un veredicto corto.
- **Calibración personal**: esos mismos 30 segundos sirven para ajustar los umbrales a tu cuerpo y tu cámara, en vez de usar los mismos números para todo el mundo.
- **Avisos por tiempo, no por frames**: la voz solo te corrige cuando la mala postura se sostiene varios segundos seguidos, así no salta por estirarte o girarte a hablar.
- **Recordatorio de pausas**: cada 30-45 minutos te pide levantarte y estirarte, aunque tu postura sea buena.
- **Historial persistente**: cada sesión se guarda en SQLite (fecha, minutos, % correcta, problema más común) y se ve como gráfica de la última semana.
- **Modo privacidad**: oculta el video y deja solo el esqueleto sobre fondo negro.
- **Notificaciones y bandeja**: avisos nativos de Windows y, al cerrar la ventana de escritorio, la app sigue trabajando desde la bandeja del sistema.
- **Configuración guardada**: umbrales, voz, cámara y tiempos se editan desde la propia interfaz y quedan en `config.json`.
- **Características normalizadas**: el modelo recibe los puntos centrados en el medio de los hombros y divididos entre el ancho de hombros, más ángulos de cuello y torso. Moverte o acercarte a la cámara ya no cambia lo que ve el modelo.
- **Evaluación honesta del modelo**: `train.py` divide entrenamiento/prueba por sesión o foto original (no por muestra) y reporta matriz de confusión, no solo accuracy.
- **Voz local**: usa `pyttsx3` (sin internet). Se puede silenciar con el botón "Voz".
- **Detección sentado / de pie**: a partir del ángulo cadera-rodilla-tobillo. Si las piernas no se ven en el encuadre lo indica ("Piernas no visibles") en vez de adivinar.

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
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

> En Windows también puedes saltarte la activación y llamar directamente al Python del entorno: `.\.venv\Scripts\python.exe app_gui.py`. Esto evita problemas con la política de ejecución de PowerShell.

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas probadas en conjunto. `plyer` (notificaciones) y `pystray` (bandeja) son opcionales: si faltan, la app funciona igual y solo pierde esas dos funciones.

---

## Uso del Proyecto

El flujo completo es: **recolectar datos → entrenar → usar el detector**.

### 1. Recolectar datos (desde la webcam, de frente)

```bash
python collect_data.py <etiqueta> [segundos] --contexto <sentado|de_pie|mixto>
```

- `etiqueta`: `1` = postura correcta, `2` = postura incorrecta
- `segundos`: duración de la grabación (por defecto 20)
- `--contexto`: solo para llevar registro de qué combinaciones ya cubriste
- `--camara N`: usar otra cámara (por defecto, la de `config.json`)

Tienes 3 segundos para prepararte y luego se graba automáticamente. Presiona `q` para cancelar.

**Repite para cada combinación** para que el modelo generalice bien:

```bash
python collect_data.py 1 20 --contexto sentado
python collect_data.py 1 20 --contexto de_pie
python collect_data.py 2 20 --contexto sentado
python collect_data.py 2 20 --contexto de_pie
```

Para la postura incorrecta, mantén errores típicos: hombros caídos o desnivelados, cabeza adelantada, espalda encorvada, etc. Conviene variar el tipo de error entre corridas.

Detalles importantes:
- **Graba siempre de frente**, con los dos hombros visibles. Las muestras de perfil se descartan automáticamente y el contador en pantalla te avisa cuántas se están perdiendo.
- Se guardan las **muestras crudas, sin espejo ni ruido**. El aumento lo hace `train.py` *después* de dividir entrenamiento/prueba, para que la precisión reportada sea real.
- Cada corrida queda como una **sesión** identificada, y `train.py` usa esa sesión para dividir sin mezclar frames casi idénticos entre entrenamiento y prueba. Por eso conviene hacer **varias corridas cortas** en vez de una sola larga: con menos de 4 sesiones el reparto vuelve a ser por muestra y la precisión se infla.
- Cada corrida **se agrega** a `dataset_posturas.npz` (no lo sobreescribe) y se registra en `dataset_sessions.json`.

### 2. Entrenar el modelo

```bash
python train.py
```

Opciones: `--dataset otro.npz`, `--test-size 0.25`, `--sin-aumento`.

Genera `modelo_postura.pkl`, `scaler.pkl` y `modelo_meta.json` (que guarda con qué formato de características se entrenó). Vuelve a ejecutarlo cada vez que agregues datos nuevos.

Además del accuracy, imprime **matriz de confusión** y **precisión/recall por clase**. Con clases desbalanceadas el accuracy solo puede verse muy bien sin que el modelo sirva: mira la matriz.

Si el accuracy de entrenamiento supera mucho al de prueba, te avisa de que el modelo se está memorizando y hay que grabar más variedad.

### 3. Usar el detector

#### Interfaz de escritorio (Tkinter)
```bash
python app_gui.py
```
Tiene tres pestañas: **Detector**, **Historial** y **Ajustes**.

#### Interfaz web (Flask)
```bash
python app_web.py
```
Luego abre `http://localhost:5000`. Mismas tres pestañas. El servidor enciende la cámara apenas arranca y la voz suena en el equipo donde corre el servidor, no en el navegador.

#### Versión de consola (ventana de OpenCV)
```bash
python main.py
```
Arranca directo en coaching continuo, sin evaluación inicial. `q` sale, `p` alterna el modo privacidad.

> `predict.py` se eliminó: era un duplicado casi idéntico de `main.py`.

### Flujo en las interfaces completas

1. **Vista previa**: la cámara está activa y el programa te saluda por voz. Ubícate frente a la cámara (sentado o de pie).
2. **INICIAR**: comienza la evaluación de 30 segundos. Mantén tu postura habitual; durante este tiempo no te corrige en vivo, para no sesgar el resultado.
3. **Resultado**: durante ~10 segundos ves un checklist como este (lo que hay que mejorar aparece primero) y escuchas un veredicto corto por voz:
   ```
   60% correcta en la evaluación
   ✗ Hombros nivelados (58%)
   ✗ Cabeza no adelantada (40%)
   ✓ Buena altura de cabeza/cuello (85%)
   Umbrales ajustados a tu cuerpo y tu cámara (74 muestras).
   ```
4. **Coaching continuo**: el estado pasa a CORRECTA/INCORRECTA en tiempo real. Cuando la postura es incorrecta ves el contador de segundos; la voz recién avisa al cumplirse el tiempo configurado.
5. **DETENER**: guarda la sesión en el historial y vuelve a la vista previa (la cámara sigue encendida).

---

## Calibración personal

Los umbrales por defecto son iguales para todos, pero cada cuerpo y cada encuadre miden distinto. Al terminar la evaluación de 30s, la app usa tus propias mediciones como referencia y ajusta `PostureThresholds`.

Dos salvaguardas, para que calibrar no sea una forma de aprobar la mala postura:

- **No calibra si tu postura de referencia fue mala** (menos de 60% correcta). Primero mejora la postura y vuelve a evaluarte; mientras tanto se usan los umbrales generales.
- **Los umbrales calibrados no se alejan sin límite** de los generales (entre 0.7x y 1.6x según el aspecto).

El resultado queda en `config.json` con `"calibrated": true`. Si editas un umbral a mano desde Ajustes, ese flag vuelve a `false`.

---

## Configuración (`config.json`)

Se crea solo la primera vez que guardas desde la pestaña **Ajustes**. Si falta, se usan los valores por defecto; si tiene valores fuera de rango, se recortan en vez de romper la app.

| Clave | Por defecto | Qué hace |
|---|---|---|
| `camera_index` | `0` | Qué cámara usar (requiere reiniciar la app) |
| `flip_camera` | `true` | Espejar la imagen |
| `voice_enabled` | `true` | Voz activada |
| `voice_rate` | `175` | Velocidad de la voz (80-400) |
| `voice_cooldown` | `6.0` | Segundos antes de repetir el mismo aviso |
| `shoulder_tilt_max` | `0.20` | Inclinación de hombros máxima (proporción del ancho de hombros) |
| `forward_head_max` | `0.55` | Cabeza adelantada máxima (proporción del ancho de hombros) |
| `neck_compression_min` | `0.23` | Altura de cabeza mínima (proporción del largo del torso) |
| `calibrated` | `false` | Si los umbrales vienen de la calibración automática |
| `alert_after_seconds` | `6.0` | Segundos de mala postura sostenida antes de avisar |
| `alert_repeat_seconds` | `25.0` | Cada cuánto repetir el aviso si no te corriges |
| `break_minutes` | `35` | Cada cuánto pedir una pausa |
| `break_reminders` | `true` | Recordatorios de pausa |
| `desktop_notifications` | `true` | Notificaciones nativas del sistema |
| `assessment_seconds` | `30.0` | Duración de la evaluación inicial (mínimo 5) |
| `privacy_mode` | `false` | Ocultar el video y dejar solo el esqueleto |
| `save_history` | `true` | Guardar las sesiones en SQLite |

---

## Historial

Cada sesión con datos se guarda en `historial_postura.db` (SQLite, en la carpeta del proyecto) al presionar DETENER o al cerrar la app. Las sesiones de menos de 30 segundos se descartan por ruido.

La pestaña **Historial** muestra una barra por día de los últimos 7 días con el % de tiempo en buena postura, los totales de la semana y el problema más frecuente. El porcentaje diario se pondera por minutos: una sesión de 2 minutos no pesa igual que una de una hora.

---

## Sobre las fotos de `dataset/`

`dataset/correcta` y `dataset/incorrecta` traen fotos de stock y de Roboflow que **no sirven para entrenar este detector**, por dos razones medidas sobre las propias fotos:

- **Son casi todas de perfil.** De 428 fotos en las que MediaPipe detecta una persona, solo 3 están tomadas de frente. La webcam y `posture_metrics.py` ven al usuario de frente: el modelo aprendería una vista y se usaría en otra.
- **Hay muchas menos fotos distintas de las que parece.** Las 160 "correctas" de Roboflow son 57 originales con recortes repetidos, y las 276 "incorrectas" son 92. Al dividir al azar, copias de la misma foto caían en entrenamiento y prueba a la vez.

Por eso el camino recomendado es `collect_data.py`. Si de todas formas quieres inspeccionar las fotos:

```bash
python collect_from_images.py                  # solo frontales, agrupando por foto original
python collect_from_images.py --incluir-perfil # todo (no recomendado)
```

Escribe en `dataset_imagenes.npz`, **aparte** del dataset de la webcam, para no pisarlo.

`dataset/descartadas/` guarda las fotos que se sacaron del entrenamiento: tres archivos AVIF que OpenCV no puede leer (dos de ellos venían con extensión `.jpg`, así que se saltaban en silencio) y una foto con dos personas, donde MediaPipe elige una cualquiera.

---

## Cómo funciona el feedback

Hay dos señales independientes:

- **Estado CORRECTA/INCORRECTA**: lo decide la red neuronal. Ya no recibe las 99 coordenadas crudas, sino 107 características: los 33 puntos centrados en el medio de los hombros y divididos entre el ancho de hombros, más las razones de postura y los ángulos de cuello y torso.
- **Consejo específico y checklist**: lo calculan reglas geométricas en `posture_metrics.py` sobre tres aspectos (hombros nivelados, cabeza no adelantada, altura de cabeza/cuello), con los umbrales de `config.json`.

Cuando no se puede medir con confianza (estás de perfil, o muy lejos), el estado no adivina: muestra "Ponte de frente a la cámara".

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

105 tests que corren sin cámara ni modelo entrenado: usan poses falsas de MediaPipe (`tests/fake_pose.py`) para ejercitar el código real.

Cubren la normalización (que acercarte a la cámara no cambie lo que ve el modelo), la detección de perfil, el espejo, los avisos por tiempo sostenido, los recordatorios de pausa, la calibración con sus salvaguardas, el historial SQLite, la configuración con sus límites y todas las rutas de `app_web`.

---

## Solución de Problemas Comunes

### ❌ "pip no se reconoce como comando interno o externo"
**Solución**:
- Windows: Busca "Python" en el menú inicio → "Modify Python" → repair installation y marca "Add to PATH"
- O usa: `py -m pip install -r requirements.txt`

### ❌ El `venv` no funciona ("No Python at ...")
**Causa**: el entorno virtual apunta a una instalación de Python que ya no existe.

**Solución**: bórralo y vuélvelo a crear:
```bash
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### ❌ "ModuleNotFoundError: No module named 'cv2'" (o mediapipe, flask, PIL, pyttsx3...)
**Solución**: instala las dependencias en el entorno que estás usando:
```bash
pip install -r requirements.txt
```

### ❌ "Modelo no encontrado. Ejecuta train.py primero."
**Causa**: aún no existen `modelo_postura.pkl` / `scaler.pkl` (están en `.gitignore`, así que no vienen al clonar el repo).

**Solución**: recolecta datos (paso 1) y ejecuta `python train.py`.

### ❌ "El modelo se entrenó con un formato de características anterior"
**Causa**: tienes un `modelo_postura.pkl` entrenado con las 99 coordenadas crudas de la versión vieja.

**Solución**: `python train.py`. La app detecta el desajuste y lo dice en vez de dar predicciones sin sentido.

### ❌ Error de cámara no encontrada
**Solución**:
- Verifica que otra aplicación no esté usando la cámara
- Cambia `camera_index` en la pestaña **Ajustes** (o en `config.json`) a `1`, `2`, etc. y reinicia la app
- En `app_gui.py` aparece "Error: no se pudo abrir la cámara" y INICIAR queda deshabilitado; en `app_web.py` verás el mensaje en lugar del video

### ❌ No se escucha la voz / "Voz no disponible"
**Solución**:
- Revisa que el volumen y la salida de audio del equipo estén bien
- Si `pyttsx3` no puede iniciar el motor de voz (en Linux requiere `espeak`: `sudo apt install espeak`), el detector sigue funcionando sin audio y la interfaz lo indica
- En `app_web.py` la voz suena en el equipo donde corre el servidor, no en el navegador

### ❌ No aparecen notificaciones ni el icono de la bandeja
**Causa**: faltan `plyer` o `pystray`, que son opcionales.

**Solución**: `pip install plyer pystray`. Sin ellos todo lo demás funciona igual; con `pystray` ausente, cerrar la ventana cierra la app en vez de minimizarla.

### ❌ La voz avisa demasiado seguido (o demasiado tarde)
**Solución**: ajusta `alert_after_seconds` y `alert_repeat_seconds` en la pestaña **Ajustes**.

### ❌ "ImportError: DLL load failed" (Windows)
**Solución**:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```
(Nota: `opencv-python-headless` no incluye ventanas, así que `main.py` y `collect_data.py` no podrán mostrar video.)

### ❌ Error con numpy, scikit-learn o incompatibilidad de versiones
**Solución**: reinstala limpio desde `requirements.txt`:
```bash
pip install -r requirements.txt --force-reinstall
```
Si el modelo fue entrenado con otra versión de scikit-learn, conviene reentrenarlo: `python train.py`.

### ❌ El entrenamiento avisa "solo hay N grupos distintos"
**Causa**: grabaste todo en una o dos corridas, así que no hay suficientes sesiones para dividir por grupo.

**Solución**: graba varias corridas cortas en vez de una larga:
```bash
python collect_data.py 1 20 --contexto sentado
python collect_data.py 1 20 --contexto de_pie
python collect_data.py 2 20 --contexto sentado
python collect_data.py 2 20 --contexto de_pie
python train.py
```

### ❌ "No se guardó ninguna muestra frontal"
**Causa**: estabas de perfil o muy de lado durante toda la grabación.

**Solución**: ponte de frente a la cámara, con los dos hombros visibles, y vuelve a grabar.

### ❌ El veredicto o el contexto no parecen correctos
- **"Piernas no visibles"**: es normal si la cámara solo capta torso y cabeza; la detección sentado/de pie necesita ver cadera, rodilla y tobillo.
- **"Ponte de frente a la cámara"**: el ancho de hombros se ve demasiado pequeño (estás de perfil o muy lejos).
- Si el checklist marca ✗ en un aspecto que consideras correcto, haz la evaluación de 30s con buena postura para calibrar, o ajusta los umbrales en **Ajustes**.

### ❌ La cámara se ve muy lenta o lag
**Solución**: cierra otras aplicaciones que usen la cámara, o activa el modo privacidad (dibuja menos).

---

## Estructura de Archivos

```
detector-postura-ia/
├── core.py                 # Keypoints, características normalizadas, carga del modelo y predicción
├── config.py               # Configuración en config.json, con límites y valores por defecto
├── history.py              # Historial de sesiones en SQLite
├── posture_metrics.py      # Umbrales, calibración personal, sentado/de pie y estadísticas de sesión
├── assessment.py           # Evaluación inicial de 30s (postura habitual + calibración)
├── smoothing.py            # Suavizado por frames y avisos por tiempo sostenido
├── voice_feedback.py       # Voz, notificaciones de escritorio y recordatorios de pausa
├── app_gui.py              # Interfaz de escritorio (Tkinter) con bandeja del sistema
├── app_web.py              # Interfaz web (Flask)
├── templates/index.html    # Frontend de la interfaz web
├── main.py                 # Detector en ventana de OpenCV (sin evaluación inicial)
├── collect_data.py         # Recolectar datos desde la webcam (solo frontales, por sesión)
├── collect_from_images.py  # Inspeccionar las fotos de dataset/ (no recomendado para entrenar)
├── train.py                # Entrenar con división por grupo y matriz de confusión
├── tests/                  # Tests que corren sin cámara ni modelo
├── requirements.txt        # Dependencias con versiones fijas
├── dataset/                # Fotos de stock (ver la sección sobre dataset/)
│   └── descartadas/        # Fotos sacadas del entrenamiento, con el motivo en el nombre
├── config.json             # Configuración (se crea al guardar desde Ajustes)
├── dataset_posturas.npz    # Dataset acumulado (se crea/actualiza al ejecutar)
├── dataset_sessions.json   # Registro de qué combinaciones ya recolectaste
├── historial_postura.db    # Historial de sesiones (se crea al usar el detector)
├── modelo_postura.pkl      # Modelo entrenado (se crea con train.py)
├── scaler.pkl              # Escalador (se crea con train.py)
└── modelo_meta.json        # Con qué formato y con cuántos datos se entrenó
```

---

## Notas

- Para mejores resultados, usa buena iluminación
- Mantén una distancia adecuada de la cámara (1-2 metros) y asegúrate de que se vean tus hombros
- **Entrena con la misma cámara y el mismo encuadre en el que lo vas a usar**, y siempre de frente
- Todo se procesa en tu equipo: no se envían imágenes ni audio a ningún servicio externo, y el video no se guarda (solo landmarks numéricos al recolectar datos y estadísticas en el historial)
- Si tus amigos tienen problemas, sugiere que actualicen sus drivers de cámara
