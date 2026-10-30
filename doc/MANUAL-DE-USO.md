# Manual de uso de PosturIA

**PosturIA** es un entrenador de postura y movimiento con inteligencia artificial. Mira tu cuerpo con una cámara común, te corrige por voz y te dice qué mejorar. Funciona sin internet y no guarda ninguna imagen.

Tiene dos programas:

- **Los juegos** (`4-JUGAR-RETO.bat`): para el stand. No hay que entrenar nada.
- **La app de postura** (`3-ABRIR-APP.bat`): vigila tu postura mientras trabajas o estudias. Necesita un modelo entrenado.

---

## 1. Instalación

Solo la primera vez, en una compu nueva:

1. Instala Python 3.11 desde [python.org](https://www.python.org/downloads/).
2. Haz doble clic en `0-INSTALAR.bat` y espera a que termine.

En la compu del proyecto ya está instalado.

---

## 2. Preparar el stand

### Lo que necesitas

- La compu, con su cargador y una regleta.
- La TV y un cable HDMI, de unos 3 metros si es posible.
- La bocina.

### Cómo colocar todo

```text
        Pared o fondo liso
    (la persona se para aquí)
              ↓
    [Compu con cámara, a 1 m de alto]    [TV en ángulo]
  ------------------ pasillo ------------------
```

- **Fondo:** que el jugador le dé la espalda a una pared. Si pasa gente detrás, la cámara puede confundirse.
- **Luz:** que le dé de frente. Evita tener una ventana detrás de la persona.
- **Marca en el piso:** pon cinta donde deben pararse.

### Conectar la TV y el sonido

1. Conecta el HDMI y presiona `Windows + P`.
2. Elige **Duplicar**.
3. Conecta la bocina.
4. En el ícono de sonido de Windows, elige la bocina como salida. Al conectar el HDMI, Windows puede mandar el sonido a la TV.

### Antes de abrir el stand

Abre PowerShell en la carpeta del proyecto y borra las partidas de prueba:

```powershell
.\.venv\Scripts\python.exe reto.py --borrar-ranking
```

Lo aprendido de las partidas no se borra con ese comando, y conviene conservarlo.

---

## 3. Abrir los juegos

Doble clic en `4-JUGAR-RETO.bat`.

Verás "Buscando tu cámara..." unos 3 segundos. El juego elige solo la cámara real y salta las virtuales.

### Teclas

| Tecla | Qué hace |
|---|---|
| `1` | Sentadillas |
| `2` | Estatua perfecta |
| `3` | Caza Estrellas |
| `S` | Entrenador Sargento |
| `A` | Entrenadora Abuelita |
| `Z` | Maestro Zen |
| `T` | Individual o por turnos |
| `D` | Dificultad: Normal, Difícil o Fácil |
| `P` | Privacidad: oculta el video y deja solo el esqueleto |
| `C` | Pasar a la siguiente cámara |
| `ENTER` | Jugar, confirmar o saltar la espera |
| `ESC` | Cancelar la partida. En el menú, dos veces para salir |
| `F11` | Pantalla completa o ventana |

### Cómo es una partida

1. Elige el juego y el entrenador.
2. Presiona `ENTER` y escribe el apodo del jugador.
3. El jugador se para en la marca, de frente. Si algo falla, la pantalla le dice qué hacer.
4. Cuenta regresiva y a jugar.
5. Aparece el resultado y luego los consejos de qué mejorar.

**Por turnos:** dos personas juegan una tras otra. La segunda ve la marca que debe superar, y al final el entrenador anuncia al ganador.

---

## 4. Los juegos

### Sentadillas

Haz todas las que puedas en 30 segundos.

- La barra de la izquierda sube mientras bajas. El porcentaje indica qué tan cerca vas de una sentadilla perfecta.
- **Buena:** llegaste a la profundidad y mantuviste la técnica.
- **A medias:** cuenta, pero no bajaste lo suficiente o te inclinaste mucho.
- **Mala:** intentaste, pero no llegaste a contar.

Al final ves el porcentaje de bien hechas y cuántas fueron de cada tipo. El ranking se ordena por sentadillas buenas.

**De lado o de frente:** de perfil se revisa mejor la espalda. De frente se revisan las rodillas y los hombros.

### Estatua perfecta

Imita poses y aguántalas sin moverte, hasta 10 poses y con 3 vidas.

1. **Prepárate:** aparece la pose y una cuenta regresiva. Aquí no se evalúa nada.
2. **Imita:** la figura sobre tu cuerpo marca en magenta lo que falta y en verde lo que ya está bien.
3. **Aguanta:** sostén la pose sin moverte hasta que se acabe la barra.

Pierdes una vida si no llegas a tiempo, si te mueves o si rompes la pose. Cada pose completada da menos tiempo para imitar y pide sostener un poco más. Gana quien complete más poses, y la precisión desempata.

### Caza Estrellas

Toca todas las estrellas que puedas con las manos en 30 segundos.

| Objetivo | Puntos |
|---|---|
| Estrella blanca | +1 |
| Estrella dorada | +3 |
| Bomba roja | -2 |

Cada 5 aciertos seguidos haces una racha. Es el juego más fácil de entender: sirve para cualquier persona, incluso con ropa formal o poco espacio.

### Dificultad

| | Fácil | Normal | Difícil |
|---|---|---|---|
| **Sentadillas:** profundidad perfecta | Menor | Media | Mayor |
| **Estatua:** puntos vigilados | 9 | 13 | 19 |
| **Estatua:** poses | Brazos rectos | También doblados | Doblados y piernas |
| **Estrellas:** duración | 2.6 s | 2 s | 1.5 s |
| **Estrellas:** bombas | Pocas | Algunas | Muchas |

Si el juego cuenta de más o de menos, cambia la dificultad con `D`.

---

## 5. La app de postura

Primero hay que entrenarla, una sola vez:

```powershell
.\1-GRABAR-DATOS.bat
.\1-GRABAR-DATOS.bat
.\2-ENTRENAR.bat
.\3-ABRIR-APP.bat
```

1. **Grabar:** son 8 grabaciones de 20 segundos, sentado y de pie, de frente y de perfil, con postura correcta e incorrecta. Hazlas dos veces.
2. **Entrenar:** debe decir "Modelo guardado".
3. **Usar:** presiona INICIAR, mantén tu postura 30 segundos y luego te corrige en vivo.

---

## 6. Si algo falla

| Problema | Qué hacer |
|---|---|
| No apareces en la pantalla | Presiona `C` hasta ver tu cámara |
| "No veo tus rodillas" | Da un paso atrás. Si no hay espacio, espera 4 segundos y el juego sigue |
| "No veo tu cabeza" | Da un paso atrás o inclina la pantalla de la compu |
| No suena nada | Revisa la salida de sonido de Windows y que la bocina siga conectada |
| Cuenta sentadillas de más | Sube la dificultad con `D` |
| No cuenta sentadillas | Baja la dificultad con `D` y revisa que se vean la cadera y las rodillas |
| La estatua quita vidas sin moverte | Usa Fácil y mejora la luz |
| Detecta a otra persona | Pon al jugador frente a un fondo sin gente |
| El ranking tiene datos raros | Sal del juego y bórralo con el comando del punto 2 |
| El juego se cerró | Vuelve a abrir `4-JUGAR-RETO.bat`. El ranking queda guardado |

---

## 7. Preguntas frecuentes

**¿Dónde está la IA?** Una red neuronal detecta 33 puntos del cuerpo en tiempo real. Los juegos analizan esos puntos para contar, corregir y aconsejar.

**¿Guarda mi cara o mi video?** No. El video se analiza en la compu y no se guarda. El ranking solo tiene apodo y puntaje.

**¿Aprende de los juegos?** Sí, de forma limitada: guarda qué poses cuestan más y cómo mide esa cámara, y ajusta los márgenes para las siguientes partidas. No reentrena una red neuronal.

**¿Reemplaza a un fisioterapeuta?** No. Ayuda a crear buenos hábitos, pero no es un diagnóstico.
