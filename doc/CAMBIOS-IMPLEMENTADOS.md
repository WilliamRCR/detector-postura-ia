# Cambios implementados en PosturIA

Todo lo que se implementó, cambió y arregló en esta etapa de preparación para la feria. El detalle de revisiones anteriores está en [CAMBIOS.md](../CAMBIOS.md).

**Estado actual:** pasan 302 pruebas automáticas. El juego corre a 30 cuadros por segundo en la compu del proyecto.

**Todavía no se ha validado con personas reales:** los umbrales de los juegos salen de movimientos simulados.

---

## 1. Lo nuevo

### PosturIA: tres juegos para el stand

Es un programa aparte (`reto.py`). No usa el modelo entrenado, así que funciona con cualquier persona y no cambia la app de postura.

| Función | Qué hace |
|---|---|
| **Sentadillas** | 30 segundos, con porcentaje en vivo y repeticiones buenas, a medias y malas |
| **Estatua perfecta** | 37 poses para imitar, con cuenta regresiva, figura guía sobre el cuerpo y 3 vidas |
| **Caza Estrellas** | Tocar estrellas con las manos y esquivar bombas |
| **Por turnos** | Dos personas compiten una tras otra |
| **Tres entrenadores** | Sargento, Abuelita y Maestro Zen, con voz de IA |
| **Consejos finales** | Al terminar, dice qué mejorar y cómo, en pantalla y por voz |
| **Aprendizaje** | Ajusta los márgenes según lo que pasa en cada partida |
| **Ranking del día** | Por juego, con contador de partidas |
| **Dificultades** | Fácil, Normal y Difícil, cambian las reglas de cada juego |
| **Pantalla completa** | Pensada para una TV |

### Audio

- **579 frases** generadas con voces de IA, guardadas en `audio/`. El juego funciona sin internet.
- La mayoría de las frases tiene **5 versiones** y nunca repite la misma dos veces seguidas.
- **Sonidos y música** generados por código, sin derechos de autor.

### Técnica y consejos

- **De perfil:** detecta el tronco muy inclinado y una señal de espalda encorvada.
- **De frente:** detecta rodillas que se juntan y hombros torcidos.
- **Ritmo:** avisa si las repeticiones van demasiado rápido.
- **Estatua:** registra si fallas por tiempo, por moverte, o por brazos o piernas.
- **Caza Estrellas:** detecta si usas una sola mano, si reaccionas lento o si tocas muchas bombas.

### Instalación y documentos

- `0-INSTALAR.bat` busca una versión compatible de Python e instala todo.
- `4-JUGAR-RETO.bat` abre los juegos.
- Documentos nuevos en `doc/`: manual, este registro, documento técnico y posibles mejoras.

---

## 2. Lo que se cambió

| Parte | Antes | Ahora |
|---|---|---|
| **Estatua** | Quedarse quieto, vigilando 7 puntos | Imitar poses, vigilando 9, 13 o 19 puntos según la dificultad |
| **Poses** | 14 | 37 |
| **Esqueleto** | 13 puntos del cuerpo | Los 33 puntos, con cara, manos y pies |
| **Inicio de cada pose** | Se evaluaba de inmediato | Cuenta regresiva sin evaluar |
| **Brazos pegados al cuerpo** | Se detectaban como doblados | Más margen, y si no se ven cuentan como bien |
| **Resultado de sentadillas** | Número de perfectas | Porcentaje grande y conteo de buenas, a medias y malas |
| **Duelo** | Pantalla dividida (descartado) | Por turnos: menos riesgo y necesita menos espacio |
| **Voces** | Una frase por situación | Cinco versiones por situación |
| **Nombre** | Detector de Postura IA | PosturIA en los juegos y en la app |

---

## 3. Errores corregidos

### En la app de postura

| Error | Qué pasaba | Solución |
|---|---|---|
| Evaluación injusta | Lo que no se podía medir contaba como mala postura | Ya no cuenta a favor ni en contra |
| Calibración sin límite | Cada evaluación relajaba más los umbrales, hasta aprobar cualquier postura | Los topes se miden desde una referencia fija |
| Modelo incompleto | Se guardaba sin las sesiones de prueba | Se vuelve a entrenar con todas |
| Aviso de cobertura | Una sesión por clase parecía suficiente | Se cuentan las sesiones por clase |
| Cabeza adelantada | Se medía contra un solo hombro | Se mide contra el centro de los hombros |
| Avisos falsos | El tiempo sin medir contaba como mala postura | Tras 15 segundos sin medir se cierra el episodio |
| Parpadeo | El estado saltaba entre CORRECTA y "Ubícate" | Se conserva el último estado unos fotogramas |
| Ajustes en la web | Cada guardado apagaba la calibración y avisaba de la cámara | Solo cuentan los cambios reales |
| Notificaciones | No se podían activar después de arrancar | Se pueden activar en cualquier momento |
| Configuración | Un valor infinito hacía fallar la carga | Se usa el valor por defecto |
| Restaurar ajustes | Borraba los umbrales entrenados | Los conserva |
| Instalación | La guía creaba un entorno con una versión incompatible de Python | `0-INSTALAR.bat` elige una compatible |

### En los juegos

| Error | Qué pasaba | Solución |
|---|---|---|
| Cámara equivocada | Abría una cámara virtual y solo se veía un mensaje | Busca sola la cámara real, y `C` permite cambiarla |
| Conteo lento | Las voces traían silencio y "Siete" duraba casi 2 segundos | Se recortaron los silencios |
| Audio largo | Generar la música cerraba el programa sin error | Se escribe por bloques |
| Avisos encimados | El aviso grande tapaba "Imita la pose" | Ahora tiene su propio fondo |
| Colores distintos | El esqueleto se ponía verde antes que la barra | Usan el mismo criterio |
| Cámara a 1280×720 | El programa fallaba al pedir esa resolución | Se usa 640×480 |

---

## 4. Lo que falta validar

1. **Sentadillas con personas reales**, de frente y de perfil.
2. **Señal de espalda encorvada:** es una aproximación, porque la cámara no ve la columna.
3. **Poses:** comprobar que se reconozcan bien, sobre todo las de brazos doblados y piernas.
4. **Caza Estrellas:** ajustar el tamaño de las estrellas a la distancia real.
5. **TV y bocina Bluetooth** conectadas juntas.
6. **App de postura:** falta grabar los datos y entrenar el modelo.
