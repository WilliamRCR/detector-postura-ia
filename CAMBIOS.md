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

Deben pasar los 159.

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

## Revisión posterior: errores corregidos

- **La voz solo hablaba la bienvenida.** `pyttsx3` en Windows solo suena la primera vez si se reutiliza el mismo motor: el segundo `runAndWait()` vuelve al instante sin hablar (medido: 3,4 s la primera frase, 0,3 s las siguientes). Ahora `VoiceFeedback` crea un motor nuevo por frase dentro de su hilo (con COM inicializado). Además: un error suelto ya no mata el hilo para siempre, la voz se puede activar aunque la app haya arrancado con ella apagada, un aviso descartado por cola llena ya no gasta el cooldown, y `stop()` ya no se cuelga con la cola llena.
- **El aviso podía decir "Buena postura".** Si la red decía "incorrecta" pero las reglas por umbral decían "Buena postura", esa era la frase que se hablaba como corrección. Ahora se usa la última corrección real o un aviso genérico.
- **La sesión se guardaba dos veces** (DETENER y luego cerrar la app). `SessionStats.reset()` tras guardar.
- **Los `.bat` buscaban solo `.venv`** y aquí el entorno se llama `venv`. Ahora prueban ambos y, si no hay ninguno, explican qué hacer.
- **`train.py`**: con las 4 sesiones de `1-GRABAR-DATOS.bat`, la prueba era una sola sesión (una sola clase). Ahora cada clase aporta al menos un grupo completo a la prueba.
- **Ajustes**: la rueda del ratón dejaba de funcionar al pasar sobre los controles; y cambiar la cámara avisa de que se aplica al reiniciar.
- Textos con tilde que OpenCV dibujaba como `?` (ventana de `main.py` y de grabación).

## Revisión del entrenamiento: las cuatro situaciones

El detector tiene que funcionar **sentado de frente, sentado de perfil, de pie de frente y de pie de perfil**. Antes todo asumía vista frontal (idea 4 del PDF, "solo datos frontales"), así que el perfil no se podía ni grabar ni medir.

Lo medido con las 23 grabaciones de webcam y las 428 fotos antiguas:

- **El modelo entrenado con las fotos antiguas daba "correcta" al 100% de las grabaciones de webcam**, incluidas las 17 de mala postura: aprendía la vista de perfil de otras cámaras y se usaba de frente.
- **El criterio de "de frente" descartaba la mitad de la webcam**: usaba hombros/torso >= 0.55 y su mediana en la webcam era 0.56, porque con una cámara de portátil las caderas no se ven y MediaPipe las inventa.
- **Con los umbrales por defecto, la regla de cabeza adelantada marcaba mal todas las posturas buenas de la webcam** (miden 1.03-1.34 contra un umbral de 0.55) y la calibración en vivo no podía corregirlo (tope 0.88).
- Las 3 medidas de perfil que se probaron sobre las fotos separan bien correctas de incorrectas (AUC 0.90-0.93).

### Qué cambió

| Dónde | Cambio |
|---|---|
| `core.py` | `view_of(kp)` -> `frente` / `perfil` / `intermedia` (3/4 o lejos: no se mide). De frente usa nariz centrada entre las orejas, hombros a similar profundidad y ancho >= 0.08 (sin caderas). De perfil usa nariz fuera del eje de las orejas y hombros angostos frente al cuello. Mediciones sobre datos reales: webcam de frente 23/23, fotos antiguas 346 perfil / 81 intermedias / 1 frente |
| `core.py` | **Escala por vista**: ancho de hombros de frente, tamaño de la cabeza (nariz-oreja) de perfil. De perfil la muestra se espeja para mirar siempre al mismo lado: da igual hacia dónde mires. **111 características** (`FEATURE_VERSION = 3`): hay que volver a entrenar |
| `posture_metrics.py` | Reglas de perfil: oreja por delante del hombro, altura de cabeza y inclinación del tronco ("Endereza la espalda"). Calibración en vivo y umbrales personales **por vista**; la vista que no usaste queda intacta |
| `config.py`, apps | 3 umbrales de perfil (`side_*`) editables en Ajustes (escritorio y web). La app muestra "De perfil · Sentado" y pide "Ubícate de frente o de perfil" solo a 45° o muy lejos |
| `collect_data.py` | `--vista frente|perfil`: solo guarda lo que está en la vista pedida. Guarda vista y contexto de cada muestra (los datasets viejos siguen cargando; el contexto se lee del nombre de la sesión) |
| `train.py` | Ya no filtra por vista. Imprime la **cobertura de las 4 situaciones** y avisa de las que faltan o tienen pocas muestras/una sola sesión; precisión **por vista**; umbrales personales por vista (solo con tus grabaciones, no las fotos). `--solo-vista`, `--solo-grabaciones`. El aumento no espeja el perfil y el ruido es proporcional a la escala de cada vista |
| `1-GRABAR-DATOS.bat` | 8 grabaciones (4 situaciones x correcta/incorrecta) |
| `collect_from_images.py` | Conserva frente y perfil; solo descarta las intermedias |

### Lo que conviene saber

- **El modelo guardado ya no sirve** (107 características vs 111): la app lo dice y hay que correr `2-ENTRENAR.bat`. Con los datos actuales entrena, pero las cuatro situaciones salen como "sin muestras" hasta que grabes.
- **Las fotos de perfil sirven como complemento**, no como dataset principal: vienen de otras cámaras y no sabemos si la persona está sentada o de pie.
- **De perfil, la inclinación del tronco usa la cadera**, que con una cámara de portátil puede quedar fuera de cuadro. Si el consejo de espalda falla, ajusta su umbral en Ajustes.
- Los umbrales de la vista de perfil se validaron solo con las fotos antiguas (reglas: marcan mal el 95% de las incorrectas y el 17% de las correctas). Falta confirmarlos con grabaciones propias de perfil.
- **Lo que falta es de datos, no de código:** grabar las 8 combinaciones, mejor varias veces y en días distintos. El tipo de modelo no es la palanca: red actual, red pequeña, Random Forest y solo 8 características quedan entre 88 y 90% sobre las fotos antiguas.

## Tests

**159 tests (105 + 54 de las revisiones), ninguno necesita cámara ni modelo entrenado.** Usan poses falsas de MediaPipe (`tests/fake_pose.py`) para ejercitar el código real.

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
