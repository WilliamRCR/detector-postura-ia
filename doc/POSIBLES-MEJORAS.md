# Posibles mejoras de PosturIA

**Documento vivo.** Se actualiza cada vez que el programa cambia: lo que se implementa pasa a "Ya implementado", y lo que aparezca al probar se agrega aquí.

**Última actualización:** 30 de octubre de 2026, con los tres juegos, las 37 poses, los consejos, el aprendizaje y el nombre PosturIA.

Las ideas iniciales para la feria están en [IDEAS-FERIA.md](IDEAS-FERIA.md).

---

## 1. Antes de la feria

Es lo que más importa hoy. No agrega funciones: evita fallos frente al público.

| Prioridad | Tarea | Por qué |
|---|---|---|
| Crítica | Probar con 3 o más personas distintas | Los umbrales salen de movimientos simulados |
| Crítica | Probar con TV y bocina conectadas | Nunca se han usado juntas |
| Crítica | Elegir la dificultad para el stand | Depende de la luz y la distancia reales |
| Alta | Probar sentadillas de perfil | La señal de espalda encorvada no se ha validado |
| Alta | Revisar las poses de brazos doblados | Son las que más pueden fallar |
| Alta | Ajustar el tamaño de las estrellas | Depende de la distancia a la cámara |
| Media | Preparar tarjetas con nombre y número de stand | Ayudan a que los recuerden al votar |
| Media | Grabar un video de respaldo | Por si algo falla |

---

## 2. Mejoras a lo que ya existe

### Juegos

| Mejora | Beneficio | Dificultad |
|---|---|---|
| Elegir el tiempo de cada reto | Partidas más cortas cuando hay fila | Baja |
| Mostrar cómo vas contra el récord | Más emoción durante la partida | Baja |
| Pantalla de espera animada | Atrae gente cuando nadie juega | Baja |
| Retrato de esqueleto para llevarse | Un recuerdo sin mostrar la cara | Media |
| Ver dos cámaras en la estatua | Evaluar de frente y de lado | Alta |

### Técnica

| Mejora | Beneficio | Dificultad |
|---|---|---|
| Usar `pose_world_landmarks` | Medir ángulos en 3D, menos dependientes de la cámara | Media |
| Validar los umbrales con un fisioterapeuta | Consejos con más respaldo | Media |
| Detectar si la persona está de lado al inicio | Elegir mejor qué revisar | Baja |
| Filtrar a otras personas en el fondo | Menos confusiones en el stand | Media |

### Aprendizaje

Hoy el programa ajusta márgenes con estadísticas, dentro de límites.

| Mejora | Beneficio | Dificultad |
|---|---|---|
| Mostrar qué aprendió | Hace visible el aprendizaje en el stand | Baja |
| Guardar repeticiones etiquetadas | Base para un modelo real | Media |
| Entrenar un clasificador de técnica | Aprender de verdad qué es una buena sentadilla | Alta |
| Calibración rápida por persona | Se adapta a cada cuerpo en 5 segundos | Media |

**Para que un modelo aprenda de verdad,** hacen falta ejemplos etiquetados por alguien que sepa de técnica. Las partidas por sí solas no dicen qué repetición estuvo bien.

### Audio

| Mejora | Beneficio | Dificultad |
|---|---|---|
| Más entrenadores | Más variedad y más risas | Baja |
| Decir el nombre del jugador | Más personal, pero exige generar voz en vivo | Alta |
| Comandos de voz | Jugar sin tocar el teclado, aunque el ruido del stand lo complica | Media |

### App de postura

| Mejora | Beneficio | Dificultad |
|---|---|---|
| Entrenar con varias personas | Funciona mejor con desconocidos | Media |
| Unirla con los juegos en un solo programa | Una sola experiencia | Alta |
| Convertir los juegos a la versión web | Jugar desde otros dispositivos | Alta |

---

## 3. Limitaciones conocidas

| Limitación | Efecto |
|---|---|
| MediaPipe detecta a una sola persona | Alguien en el fondo puede confundirlo |
| La cámara no ve la columna | La espalda encorvada se estima |
| El video es de 640×480 | Las manos y los dedos se detectan con menos precisión |
| La cámara falla a 1280×720 con esta versión de OpenCV | No se puede subir la resolución |
| De frente no se ve la profundidad real | Por eso la técnica de espalda se revisa mejor de perfil |
| Las voces no dicen el nombre del jugador | Están generadas de antemano |
| El aprendizaje no reentrena la red | Solo ajusta márgenes |

---

## 4. Ya implementado

| Mejora | Dónde |
|---|---|
| Tres juegos: sentadillas, estatua y Caza Estrellas | `reto.py` |
| 37 poses por dificultad | `reto_logica.py` |
| Esqueleto completo de 33 puntos | `reto.py` |
| Cuenta regresiva antes de cada pose | `reto_logica.py` |
| Consejos finales por voz y en pantalla | `reto_frases.py` |
| 579 frases sin repetición seguida | `reto_frases.py`, `audio/` |
| Aprendizaje con límites entre partidas | `reto_logica.py` |
| Ranking del día | `reto_logica.py` |
| Por turnos | `reto.py` |
| Búsqueda automática de la cámara real | `core.py` |
| Instalador | `0-INSTALAR.bat` |

---

## 5. Cómo mantener este documento

Cada vez que se modifique el programa:

1. Mueve lo terminado a "Ya implementado".
2. Agrega los problemas o ideas que aparezcan al probar.
3. Revisa si alguna limitación ya se resolvió.
4. Cambia la fecha de la última actualización.
