# Documento técnico de PosturIA

Cómo funciona el sistema por dentro. Está pensado para quien vaya a modificar el código o explicar el proyecto con detalle.

## 1. Vista general

```text
Cámara → MediaPipe Pose → 33 puntos del cuerpo
                              ├─ Juegos: reglas geométricas → conteo, consejos, voz
                              └─ App de postura: características → red neuronal → CORRECTA/INCORRECTA
```

Todo corre en la compu, sin internet. El video se analiza en memoria y no se guarda.

| Tecnología | Para qué se usa |
|---|---|
| Python 3.11 | Lenguaje. Las dependencias solo funcionan de 3.10 a 3.12 |
| MediaPipe Pose | Red neuronal ya entrenada que detecta 33 puntos |
| OpenCV | Leer la cámara |
| scikit-learn | Red neuronal propia de la app de postura |
| pygame | Pantalla, teclado y audio de los juegos |
| Tkinter y Flask | Interfaces de la app de postura |
| SQLite | Ranking e historial |
| edge-tts | Generar las voces una sola vez, con internet |

## 2. Archivos

### Juegos

| Archivo | Responsabilidad |
|---|---|
| `reto.py` | Pantallas, flujo de la partida, teclado e hilo de cámara |
| `reto_logica.py` | Reglas sin cámara ni pantalla, por eso se pueden probar |
| `reto_audio.py` | Voz, efectos y música en canales separados |
| `reto_frases.py` | Entrenadores, frases y consejos |
| `tools/generar_audio.py` | Genera `audio/`. Solo se usa para cambiar frases |
| `audio/` | 579 frases y 9 sonidos |

### App de postura

| Archivo | Responsabilidad |
|---|---|
| `core.py` | Puntos, características, vistas, modelo y búsqueda de cámara |
| `posture_metrics.py` | Umbrales, calibración y estadísticas |
| `assessment.py` | Evaluación inicial de 30 segundos |
| `smoothing.py` | Suavizado y avisos por tiempo sostenido |
| `voice_feedback.py` | Voz, notificaciones y pausas |
| `train.py` | Entrenamiento y evaluación del modelo |
| `collect_data.py` | Grabar datos con la webcam |
| `app_gui.py`, `app_web.py`, `main.py` | Interfaces |
| `config.py`, `history.py` | Configuración e historial |

## 3. Flujo de los juegos

`reto.py` es una máquina de estados:

```text
menu → apodo → marca → cuenta → juego → resultado → consejos → menu
                 ↑                                      │
                 └────────── segundo turno ─────────────┤
                                                        └→ duelo → menu
```

La cámara corre en su propio hilo (`Camara`), y la detección toma unos 28 ms por cuadro en la compu del proyecto. Así la pantalla se mantiene a 30 cuadros por segundo. Si el hilo falla, muestra el error en pantalla en vez de quedarse esperando.

### Búsqueda de la cámara

Las cámaras virtuales, como las de OBS o Animaze, abren sin problema, pero entregan una imagen fija. `core.buscar_camara` abre cada índice y mide la diferencia entre cuadros: un sensor real siempre tiene ruido, aunque nada se mueva. Usa la primera con imagen viva y la guarda en `config.json`.

## 4. Sentadillas

`ContadorSentadillas` mide cuánto baja la cadera respecto a la persona de pie:

```text
profundidad = (cadera_actual - cadera_de_pie) / largo_del_muslo
```

Una profundidad de 1 equivale al muslo horizontal. Se eligió así por tres razones:

- De frente, el ángulo de la rodilla casi no cambia en la imagen.
- Con poco espacio, los tobillos pueden quedar fuera de cuadro.
- Agacharse como una reverencia baja los hombros, pero no la cadera, y por eso no cuenta.

Es una máquina de dos estados con histéresis:

| Paso | Condición |
|---|---|
| Empieza a contar | Baja de `nivel.bajar` |
| Repetición terminada | Sube de 0.20 |
| Es perfecta | Llegó a `nivel.perfecta` sin inclinarse |
| Fallida | Pasó de 0.25 pero no llegó a contar |

| Nivel | `bajar` | `perfecta` |
|---|---|---|
| Fácil | 0.40 | 0.55 |
| Normal | 0.50 | 0.70 |
| Difícil | 0.60 | 0.85 |

### Revisión de técnica

Se revisa en la parte baja de cada repetición, según lo que cada vista permite ver:

| Vista | Qué mide | Límite |
|---|---|---|
| Perfil | Inclinación del tronco | Más de 55° |
| Perfil | Ángulo oreja-hombro-cadera | Menos de 135° |
| Frente | Torso acortado en la imagen | Menos de la mitad |
| Frente | Rodillas respecto a tobillos | Menos de 0.7 |
| Frente | Inclinación de hombros | Más de 0.18 |
| Ambas | Duración de la repetición | Menos de 0.9 s |

**Limitación importante:** MediaPipe no ve la columna. El ángulo oreja-hombro-cadera es una aproximación, no una medida real de la curvatura de la espalda.

## 5. Estatua perfecta

### Comparar poses

Cada pose define hacia dónde apunta cada segmento, en grados: 0 derecha, 90 abajo, 180 izquierda y 270 arriba. Se comparan direcciones, no posiciones, y por eso sirve para cualquier estatura.

- Los lados se toman según la pantalla, no según el cuerpo, porque el jugador se ve como en un espejo.
- Las coordenadas de MediaPipe no son cuadradas, así que se corrigen con `ASPECTO = 4/3`.
- Lo que apunta hacia abajo tiene 1.7 veces más margen, porque un brazo pegado al cuerpo se detecta mal. Si no se ve, cuenta como bien.

### Fases de cada pose

| Fase | Qué pasa | Si falla |
|---|---|---|
| `preparar` | Muestra la pose. No se evalúa | Nada |
| `imitar` | Debe coincidir durante 0.3 s | Pierde una vida al acabarse el tiempo |
| `aguantar` | Debe sostenerla sin moverse | Pierde una vida si se mueve |

### Medir el movimiento

Al empezar `aguantar` se guarda la posición de los puntos vigilados. Cada cuadro se suaviza la posición y se mide la distancia a esa referencia, dividida entre el ancho de hombros.

Se usa la **segunda distancia más alta**: un movimiento real siempre mueve más de un punto, y así un solo punto mal detectado no quita una vida.

| Nivel | Puntos | Tolerancia | Movimiento | Preparar | Imitar | Aguantar |
|---|---|---|---|---|---|---|
| Fácil | 9 | 35° | 0.20 | 4 s | 5 s | 3 s |
| Normal | 13 | 28° | 0.14 | 3 s | 4 s | 4 s |
| Difícil | 19 | 22° | 0.10 | 2.5 s | 3.5 s | 5 s |

## 6. Caza Estrellas

`JuegoEstrellas` crea objetivos alrededor de la persona, a una distancia proporcional a su cuerpo, para que siempre queden a su alcance. Un toque se detecta cuando la muñeca o el índice entra en el radio del objetivo.

| Nivel | Duración | Radio | Bombas |
|---|---|---|---|
| Fácil | 2.6 s | 0.085 | 12% |
| Normal | 2.0 s | 0.07 | 20% |
| Difícil | 1.5 s | 0.055 | 30% |

## 7. Consejos finales

Cada juego acumula sus errores en un `Counter`. `elegir_consejos` toma los más repetidos, siempre que hayan pasado en al menos el 25% de los intentos, y muestra un máximo de 3. Así, un error aislado no genera un consejo.

## 8. Aprendizaje entre partidas

`Aprendizaje` guarda estadísticas en `reto_aprendizaje.json` y ajusta los umbrales dentro de límites fijos.

| Qué aprende | Cómo lo usa | Límite |
|---|---|---|
| Qué poses se logran | Da más o menos tolerancia | De -4° a +8° |
| Qué poses tienen pocos datos | Las elige más seguido | Mínimo 5 intentos |
| Profundidad real de las sentadillas | Ajusta qué cuenta como perfecta | Hasta -0.10 |
| Temblor real de la detección | Ajusta el movimiento tolerado | De 0.85 a 1.3 veces |

**Esto no es reentrenar una red neuronal.** Es adaptación estadística: simple, explicable y sin riesgo de desconfigurar el juego, porque cada valor tiene límites. Se puede borrar con `--borrar-aprendizaje`.

## 9. Audio

Cada frase tiene varias versiones. `Audio._frase` elige una al azar, sin repetir la anterior. La voz tiene un canal reservado para no cortar los efectos, y si ya está hablando, solo queda en espera la última frase.

Las voces se generan con `edge-tts`, se les recorta el silencio y se guardan como OGG. La abuelita lleva además un efecto de temblor (vibrato y trémolo).

Para cambiar una frase, edita `reto_frases.py` y corre `tools/generar_audio.py` con `edge-tts`, `soundfile`, `scipy` y `numpy` instalados.

## 10. App de postura

1. `collect_data.py` guarda los 33 puntos, nunca el video.
2. `core.features_from_keypoints` convierte cada muestra en 111 características: centra los puntos y los divide entre una escala, para que no importe la distancia a la cámara.
3. `train.py` divide por sesión, entrena para medir y luego vuelve a entrenar con todas las sesiones.
4. La red neuronal decide CORRECTA o INCORRECTA, y reglas geométricas explican qué corregir.

## 11. Datos que se guardan

| Archivo | Contenido | ¿Va al repo? |
|---|---|---|
| `reto_ranking.db` | Apodo, juego y puntaje | No |
| `reto_aprendizaje.json` | Estadísticas sin datos personales | No |
| `config.json` | Ajustes y cámara elegida | No |
| `historial_postura.db` | Sesiones de la app | No |
| `dataset_posturas.npz` | Puntos de las grabaciones | No |

**Nunca se guarda** video ni fotos.

## 12. Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Son 302 pruebas y no necesitan cámara. Usan poses simuladas (`tests/fake_pose.py`) y cubren reglas, audio, ranking, aprendizaje y partidas completas.

**Lo que las pruebas no cubren:** el comportamiento con personas reales. Los umbrales de los juegos deben validarse jugando.
