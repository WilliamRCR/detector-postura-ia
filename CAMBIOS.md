# Cambios de esta rama

Implementa las 14 ideas del cuadro del PDF *"Detector de Postura IA — Ideas de mejora"* (Oct 2026), que está incluido en el repo.

**Resumen:** 4 arreglos del modelo (prioridad alta), 7 funciones nuevas, 3 de limpieza. Más 105 tests que no existían, un bug de layout que apareció al probar, y una corrección a lo que decía el PDF.

---

## Cómo revisar esto rápido

Si tienes poco tiempo, mira estos cuatro archivos en este orden:

1. **`core.py`** — el módulo nuevo del que cuelga todo lo demás
2. **`train.py`** — aquí está el cambio que hace honesta la precisión
3. **`posture_metrics.py`** — la calibración personal y sus salvaguardas
4. **`smoothing.py`** — el aviso por tiempo sostenido

Para correr los tests no hace falta cámara ni modelo entrenado:

```bash
python -m unittest discover -s tests -t .
```

Deben pasar los 105.

---

## Los 4 arreglos del modelo (prioridad alta)

### 1. `core.py`: un solo lugar para los datos del modelo

`get_keypoints` estaba duplicado en 5 archivos (`main.py`, `predict.py`, `collect_data.py`, `collect_from_images.py`, `app_gui.py`, `app_web.py`), así que cualquier cambio en el formato obligaba a editarlos todos y a acordarse de todos.

Ahora `core.py` concentra keypoints, características, carga del modelo y predicción (`PosturePredictor`). El PDF recomendaba hacer esto **antes** que las características normalizadas, y así se hizo.

> Detalle de diseño: `core.PosturePredictor.predict` importa `posture_metrics` dentro de la función. Es a propósito: `posture_metrics` importa `core`, y el import diferido rompe el ciclo. Está comentado en el código.

### 2. Características normalizadas (`core.features_from_keypoints`)

**El problema:** el modelo recibía las 99 coordenadas crudas de MediaPipe. Si el usuario se movía en el cuadro o se acercaba a la cámara, cambiaban los 99 números aunque la postura fuera idéntica.

**La solución:** los 33 puntos se centran en el medio de los hombros y se dividen entre el ancho de hombros. Se agregan 8 valores más: las 3 razones de postura, la proporción torso/hombros y 4 ángulos (cuello, torso, inclinación de cabeza y de hombros). **107 características** en total.

Hay dos tests que demuestran que el arreglo funciona, en `tests/test_core.py`:

- `test_acercarse_a_la_camara_no_cambia_las_caracteristicas` — la misma postura al doble de distancia da características idénticas (±1e-5), mientras las coordenadas crudas cambian más de 0.05
- `test_moverse_en_el_cuadro_no_cambia_las_caracteristicas`

**Compatibilidad:** un `modelo_postura.pkl` viejo (99 features) ya no sirve. En vez de dar predicciones sin sentido, `core.load_model()` lanza `ModelOutdated` con el mensaje *"Vuelve a ejecutar train.py"*. Se guarda `modelo_meta.json` con la versión del formato.

### 3. División por grupo en `train.py`

**Dos cosas inflaban la precisión:**

- Las 160 fotos "correctas" de Roboflow son **57 originales** con recortes repetidos (las 276 "incorrectas" son 92). Al dividir al azar, copias de la misma foto caían en entrenamiento y prueba a la vez.
- El espejo y el ruido se agregaban **antes** de dividir, así que la prueba contenía el espejo de muestras de entrenamiento.

**Ahora:** `GroupShuffleSplit` por sesión de grabación o foto original, y el aumento se aplica **solo al lado de entrenamiento, después de dividir**. Además se reporta matriz de confusión y precision/recall por clase, no solo accuracy.

`collect_data.py` guarda a qué sesión pertenece cada muestra (array `groups` en el `.npz`). Los datasets viejos sin `groups` siguen cargando, pero `train.py` avisa de que la división vuelve a ser por muestra.

> Ojo al revisar: al dividir por grupo ya no se puede estratificar por clase, así que el reparto correcta/incorrecta en la prueba puede quedar desparejo. Es el precio correcto a pagar; está contemplado y avisado por consola.

### 4. Solo datos frontales

**Medido sobre las propias fotos del repo:** de 428 imágenes donde MediaPipe detecta a alguien, **solo 3 están tomadas de frente**. El PDF decía "casi todas de perfil"; es el 99.3%.

El criterio es `ancho_hombros / largo_torso` (`core.frontality`). De perfil los hombros se juntan y la razón colapsa. La distribución del dataset tiene mediana ~0.25 y percentil 99 ~0.50; una pose frontal real ronda 0.85. El umbral quedó en **0.55**, entre medio de ambas.

- `collect_data.py` descarta las muestras de perfil mientras grabas y te muestra el contador
- `collect_from_images.py` filtra por defecto, agrupa por foto original y escribe en `dataset_imagenes.npz` **aparte**, para no pisar el dataset bueno

---

## Las 7 funciones nuevas

| Función | Dónde | Nota |
|---|---|---|
| Calibración personal | `posture_metrics.PostureCalibrator`, `assessment.py` | Ver salvaguardas abajo |
| Alerta por tiempo | `smoothing.SustainedStateTimer` | Ver abajo |
| Historial persistente | `history.py` (SQLite) + gráfica semanal en ambas interfaces | |
| Recordatorio de pausas | `voice_feedback.BreakReminder` | Cada 35 min por defecto |
| Notificaciones + bandeja | `voice_feedback.DesktopNotifier`, `app_gui._setup_tray` | `plyer` y `pystray`, **opcionales** |
| Modo privacidad | `app_gui.py`, `app_web.py`, `main.py` | Esqueleto sobre fondo negro |
| Configuración guardada | `config.py` → `config.json` | Editable desde la interfaz |

### Calibración: dos salvaguardas que conviene revisar

Calibrar sobre la postura del usuario tiene un riesgo obvio: si se evalúa encorvado, los umbrales aprenderían a aprobar la mala postura para siempre. Por eso:

- **No calibra si la evaluación salió por debajo del 60% correcta.** Devuelve los umbrales generales y un mensaje que lo explica.
- **Los umbrales calibrados no se alejan de los generales más de 0.7x–1.6x** (`CALIBRATION_BOUNDS`).

Esto significa que la función es menos útil justo para quien peor se sienta. Es deliberado, pero es una decisión de producto discutible: si prefieres otro balance, se cambia en `CALIBRATION_MIN_PERCENT` y `CALIBRATION_BOUNDS`.

Tests en `tests/test_calibration.py`.

### Alerta por tiempo

El suavizado votaba sobre 7 frames, menos de medio segundo, así que la voz saltaba por cualquier movimiento corto. `SustainedStateTimer` exige **mala postura sostenida** (6 s por defecto) antes de avisar, repite cada 25 s si no se corrige, y felicita una vez al corregirse.

Tiene `grace_seconds`: un frame bueno aislado (la IA dudando un instante) no reinicia el contador. El voto por frames se mantiene para el cartel en pantalla, que sí debe ser reactivo.

---

## Limpieza

- **Eliminado `predict.py`.** Era un duplicado casi idéntico de `main.py` (solo cambiaba el layout). Se quedó `main.py`.
- **Dataset.** Se creó `dataset/descartadas/` con 4 archivos y el motivo en el nombre:
  - 3 archivos **AVIF**, que OpenCV no puede leer. El PDF mencionaba uno; había **tres**, y dos venían con extensión `.jpg`, así que se saltaban en silencio sin que nadie lo notara.
  - 1 foto con **dos personas**, donde MediaPipe elige una cualquiera.
- **Renombrada** la imagen cuyo nombre tenía tilde (`Siéntate` → `sientate`).

---

## ⚠️ Una cosa del PDF que resultó estar mal

El cuadro dice que en `incorrecta/` están mal etiquetadas `Postura-correcta.jpg` y `Siéntate-bien…buena-postura.jpg`, y pide moverlas.

**Abrí las dos fotos: ambas muestran mala postura** (una persona encorvada sobre el laptop). Los nombres vienen del título del artículo de donde se descargaron, no de lo que se ve en la imagen. **La etiqueta `incorrecta` es correcta**, así que no las moví: solo se renombró la del acento.

Vale la pena que lo confirmes tú también antes de aprobar, están en `dataset/incorrecta/`.

---

## Bug encontrado al probar (ya corregido)

La app se abría con los botones INICIAR/DETENER **fuera de la pantalla** en monitores de 768 px. Tres causas, todas introducidas al agregar las pestañas:

1. `video_label` se creaba con `width=640, height=480` **sin imagen**, y un `tk.Label` sin imagen mide eso en **caracteres y líneas**, no en píxeles: pedía 480 líneas de texto y la ventana calculaba 7448 px de alto.
2. El `Notebook` adopta el alto de su pestaña más alta, y Ajustes (778 px) estiraba toda la ventana aunque estuvieras en Detector. Ahora Ajustes tiene scroll y pide 265 px.
3. La ventana era de tamaño fijo, así que lo que no entraba era inalcanzable.

Ahora el video se ajusta **midiendo la interfaz ya construida** en vez de estimarla (el alto del encabezado y los botones cambia con la fuente y el tema de Windows). Verificado en 768, 800, 900, 1080 y 1440 px.

---

## Otro bug, este del historial

`history.py` usaba `with sqlite3.connect(...)`, que **confirma la transacción pero no cierra la conexión**. En Windows eso dejaba el archivo bloqueado y acumulaba conexiones sesión tras sesión. Lo detectaron los tests al no poder borrar el `.db` temporal. Corregido con un `contextmanager` que sí cierra.

---

## Tests

**105 tests, ninguno necesita cámara ni modelo entrenado.** Usan poses falsas de MediaPipe (`tests/fake_pose.py`) para ejercitar el código real.

| Archivo | Cubre |
|---|---|
| `test_core.py` | Normalización, detección de perfil, espejo, ruido, métricas |
| `test_alerts.py` | Aviso por tiempo sostenido, pausas, coach de voz |
| `test_calibration.py` | Calibración y sus salvaguardas, evaluación de 30s |
| `test_history_config.py` | SQLite, estadísticas de sesión, config con sus límites |
| `test_app_web.py` | Las 8 rutas de Flask, con el test client (sin abrir la cámara) |

Además se verificó a mano, con cámara simulada (sin encender la webcam):

- Pipeline completo: generar datos → `train.py` → cargar modelo → predecir
- `app_gui.py` recorriendo las 4 fases, ajustes, historial y cierre
- `main.py` con sus teclas
- Los guardas de modelo viejo (`ModelOutdated`) y modelo ausente (`ModelNotFound`)

**Lo que NO se pudo probar automáticamente** y conviene que mires tú con hardware real:

- La webcam de verdad (todo se probó con frames simulados desde fotos)
- Que la voz suene (la lógica está testeada, el audio no)
- El icono de la bandeja del sistema y las notificaciones de Windows

---

## Dependencias nuevas

```
plyer==2.1.0     # notificaciones de escritorio
pystray==0.19.5  # icono en la bandeja
```

**Ambas son opcionales.** Si faltan, la app arranca igual y solo pierde esas dos funciones (sin `pystray`, cerrar la ventana cierra la app en vez de minimizarla). El import está protegido en ambos casos.

---

## Archivos nuevos que genera la app

Están todos en `.gitignore`:

```
config.json              # configuración del usuario
historial_postura.db     # historial de sesiones
modelo_meta.json         # formato y métricas del modelo entrenado
dataset_imagenes.npz     # salida de collect_from_images.py
```

---

## Para probarlo tú

Después de clonar la rama:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

Y para el flujo con cámara, los tres `.bat` en orden: `1-GRABAR-DATOS`, `2-ENTRENAR`, `3-ABRIR-APP`. El `README.md` tiene el detalle de qué deberías ver en cada paso.
