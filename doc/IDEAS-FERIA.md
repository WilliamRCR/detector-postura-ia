# Ideas para ganar la feria de proyectos

> Este documento recoge las ideas iniciales. Varias ya se implementaron: el estado actual está en [POSIBLES-MEJORAS.md](POSIBLES-MEJORAS.md).

Mejoras visuales y auditivas para el PosturIA. Todas se pueden hacer con lo que el proyecto ya usa (MediaPipe, OpenCV, Flask y Tkinter), sin internet y sin sensores extra.

**Cómo es la feria:** no hay exposición ni jurado. El proyecto queda en un stand, ustedes lo muestran a quien pase y **el público lo califica al final**. Este documento está pensado para eso.

## La recomendación en 30 segundos

Con voto del público no gana el proyecto más complejo, sino **el que más gente probó, disfrutó y recuerda a la hora de votar**. Hoy el proyecto hace algo útil, pero se ve tranquilo: una persona quieta y un letrero que dice CORRECTA. Desde el pasillo eso no llama a nadie.

**El cambio de fondo es pasar de "una app que te vigila" a "un juego al que todos quieren entrar".** Cada visita debe cumplir cuatro cosas:

- **Se entiende en 5 segundos,** sin explicación.
- **Dura menos de 2 minutos,** para que pase mucha gente.
- **Divierte también a quien mira,** porque mira mientras espera su turno.
- **Deja un recuerdo,** porque el voto llega al final y compiten muchos proyectos.

Si solo pueden hacer seis cosas, que sean estas, en este orden:

| # | Mejora | Qué aporta |
|---|---|---|
| 1 | **Modo Ejercicio** (sentadillas y saltos) | Es el motor del juego: la IA cuenta y corrige en vivo |
| 2 | **Reto de 30 segundos con ranking** | Hace que la gente compita, se quede a mirar y traiga amigos |
| 3 | **Duelo 1 contra 1** | Los grupos de amigos no pueden resistirse, y se oye desde lejos |
| 4 | **Espejo inteligente con puntaje** | Atrae desde el pasillo: esqueleto de colores, ángulos y un número grande |
| 5 | **Audio con personalidad** | Hace reír, y la risa atrae gente |
| 6 | **Recuerdo para llevarse** | Es lo que les recuerda a ustedes en el momento de votar |

Lo demás del documento es opcional. **No intenten hacer todo:** cinco cosas pulidas ganan a quince a medias.

---

## Antes de programar: cuatro cosas que deciden la feria

### 1. Averigüen las reglas de la votación
Es lo primero, porque cambia lo que conviene hacer:

- ¿Cómo se vota, cuándo, y quién puede votar?
- ¿Cuántos votos tiene cada persona? ¿Hay que probar el proyecto para calificarlo?
- ¿Se califica con nota o se elige un favorito?
- ¿Se permite pedir el voto o entregar recuerdos?
- ¿Cómo identifica el público a cada proyecto: nombre, número de stand o QR?
- ¿Qué tamaño tiene el stand, y hay enchufes, mesa, pantalla y luz?

### 2. El punto débil: el modelo está entrenado con ustedes
El detector aprende de quien graba los datos y de esa cámara. **En la feria, quien se pare enfrente será un desconocido**, con otra estatura, otra ropa y otra distancia. Si falla frente a la fila, se corre la voz y no se recupera.

Qué hacer:

- **Basen el juego en reglas geométricas.** El Modo Ejercicio mide ángulos, así que funciona con cualquier cuerpo sin entrenar. La red neuronal queda como complemento.
- **Graben a 10 o 15 personas distintas** (altas, bajas, con ropa holgada), con la misma cámara y una ubicación parecida a la del stand.
- **Quiten la evaluación de 30 segundos en el modo feria.** Es demasiado tiempo quieto: cámbienla por una medición de 5 a 10 segundos.

### 3. La cámara del portátil no alcanza
Una webcam de portátil solo ve de la cintura para arriba. Para ejercicios hace falta ver el cuerpo completo, a unos 2 o 3 metros. Necesitan una **webcam USB en trípode**, o un teléfono usado como cámara. Confirmen que el stand tiene ese espacio: si no lo tiene, el reto debe ser de la cintura para arriba (ver el Reto de brazos).

### 4. Todos deben poder jugar
Votan todas las personas, no solo las que quieren hacer ejercicio. Habrá gente con ropa formal, tacones, poca movilidad o simplemente vergüenza. **Si solo ofrecen sentadillas, pierden esos votos.** Por eso conviene tener un reto para cada caso (ver la idea 2).

---

## Las ideas estrella

### 1. Modo Ejercicio

**Qué se ve:** la persona elige un ejercicio y hace repeticiones. En pantalla, un contador grande, el esqueleto de color y una barra que marca qué tan abajo llegó. Al terminar, un resumen: "12 repeticiones, 10 bien hechas, 83%".

**Qué se oye:** la voz cuenta ("uno, dos, tres") y corrige en el momento: "baja un poco más", "mantén la espalda recta".

**Cómo funciona:** cada ejercicio se reduce a medir un ángulo. Una repetición es pasar de un ángulo a otro y volver. El proyecto ya calcula ángulos entre tres puntos (`_joint_angle` en `posture_metrics.py`), así que la base existe.

Los ángulos son puntos de partida para ajustar probando:

| Ejercicio | Cámara | Qué se mide | Cuenta una repetición | Qué puede corregir | Fiabilidad esperada |
|---|---|---|---|---|---|
| **Sentadilla** | Lado o frente | Cadera-rodilla-tobillo | De unos 170° a menos de 100° y volver | No baja lo suficiente, tronco muy inclinado | Alta |
| **Saltos de tijera** | Frente | Muñecas sobre los hombros | Brazos arriba y abajo | Brazos a medias | Alta |
| **Curl de bíceps** | Frente o lado | Hombro-codo-muñeca | De unos 160° a menos de 50° | Mueve el codo, no sube completo | Alta |
| **Elevación lateral** | Frente | Altura de la muñeca | La muñeca llega al nivel del hombro | Un brazo más bajo que el otro | Alta |
| **Zancada** | Lado | Ángulo de la rodilla | Baja hasta cerca de 90° | Rodilla muy adelantada | Media |
| **Plancha** | Lado, cámara baja | Hombro-cadera-tobillo | Cronómetro mientras la línea siga recta | Cadera caída o levantada | Media |
| **Flexión** | Lado, cámara baja | Hombro-codo-muñeca | De unos 160° a menos de 90° | Cadera caída, no baja completo | Media |

**Advertencia sobre las flexiones:** ustedes las mencionaron, pero son las más difíciles. La cámara tiene que estar casi al nivel del piso, y MediaPipe rinde peor con personas acostadas. Además, en una feria casi nadie se tira al suelo, y menos con ropa formal. **Empiecen por sentadillas y saltos**, y dejen flexiones y plancha como extra para una demostración de ustedes.

**Dificultad:** media. Una sentadilla funcionando se logra en 1 o 2 días; el modo completo con 3 ejercicios y su interfaz, en 1 o 2 semanas.

### 2. Reto de la feria y duelo 1 contra 1

**Reto con ranking.** Una pantalla grande muestra el top 10 del día: "Sentadillas perfectas en 30 segundos". Cada visitante escribe un apodo, compite y aparece en la tabla. Es lo que hace que la gente se quede mirando y que llame a sus amigos. Podrían dar un premio pequeño al primer lugar del día.

**Por qué es la idea clave con voto del público:** convierte a los visitantes en parte del espectáculo. Quien mira quiere ganarle al que va primero, y quien entra en la tabla vuelve con sus amigos a ver si sigue ahí.

**Retos para todo tipo de persona.** Ofrezcan tres y dejen elegir:

| Reto | Qué hace la persona | Para quién |
|---|---|---|
| **Sentadillas** | Las más posibles y bien hechas en 30 segundos | Quien quiere competir de verdad |
| **Brazos** | Saltos de tijera sin saltar o elevaciones laterales | Ropa formal, poco espacio o menos movilidad. Solo necesita ver de la cintura para arriba |
| **Estatua perfecta** | Aguantar la mejor postura el mayor tiempo posible, de pie o sentado | Cualquiera, incluso quien prefiere no moverse |

**Duelo 1 contra 1.** Dos personas juegan a la vez, con la pantalla dividida y marcadores separados. La forma sencilla es cortar cada imagen en dos mitades y analizar cada una con su propio detector, para no depender de un modelo multipersona. Se necesita una cámara con buen ángulo.

**Dificultad:** el ranking es baja (es una tabla más en el SQLite que ya existe). El duelo es media.

**Privacidad:** guarden solo apodo y puntaje, nunca imágenes. Tener un cartel que lo diga da confianza para participar.

### 3. Espejo inteligente

La idea es que el público **vea** lo que la IA está pensando.

- **Articulaciones por color:** verde si está bien, amarillo si va mal, rojo si está mal. El cuello se pone rojo cuando la cabeza se adelanta.
- **Ángulos sobre el cuerpo:** un arco dibujado en la rodilla o el cuello con el número en vivo ("87°"). Esto se ve muy técnico y muy impresionante.
- **Línea guía:** una línea vertical ideal de oreja a cadera y otra con la postura actual; la distancia entre ambas muestra el error.
- **Silueta fantasma:** en la evaluación, una figura transparente muestra la postura correcta y la persona se acomoda hasta encajar.
- **Modo esqueleto de neón:** el modo privacidad actual, pero con fondo oscuro y líneas brillantes. Sirve para el stand y para explicar la privacidad.

**Dificultad:** baja a media. Es dibujar con OpenCV sobre lo que ya se calcula. **Es la mejora con mejor relación entre esfuerzo e impacto.**

### 4. Puntaje de postura de 0 a 100

Cambien CORRECTA/INCORRECTA por un número grande y un medidor tipo velocímetro. La gente entiende "72/100" al instante y quiere subirlo.

- **Radar del cuerpo:** una gráfica con cuello, hombros, espalda y cabeza.
- **Antes y después:** dos esqueletos lado a lado, el del inicio y el corregido ("subiste de 58 a 86"). Es la imagen más potente del stand.
- **Nivel en palabras:** "Postura de oficina", "Postura de atleta", "Postura de bailarín". Es un detalle simple que da risa y se comparte.

**Dificultad:** baja.

### 5. Recuerdo para llevarse

El voto llega al final, después de ver muchos proyectos. **Hay que lograr que se acuerden de ustedes.** Antes de irse, cada visitante recibe algo con su resultado y, sobre todo, con el nombre del proyecto y su número de stand.

- **Tarjeta o sticker:** lo más simple y lo que mejor funciona. Una tarjeta pequeña con espacio para escribir: "Hice 18 sentadillas perfectas. Stand 12". No necesita programación.
- **Retrato de esqueleto:** la app genera una imagen de neón de su pose final, con su puntaje. Se comparte bien y no muestra la cara, así que no hay problema de privacidad.
- **En el teléfono:** un código QR abre su resultado. Funciona sin internet si el teléfono y el portátil están en la misma red, porque Flask ya sirve páginas. Hay que permitir conexiones de la red y ajustar el firewall de Windows. **En una feria esto puede dar problemas:** tengan la tarjeta como opción principal.
- **Impreso en el momento:** una impresora térmica pequeña es muy llamativa, si consiguen una.

**Dificultad:** la tarjeta es muy baja. El retrato o el QR son baja a media; hay librerías para generarlos (`qrcode`, `fpdf2`).

### 6. Audio con personalidad

El sonido es lo que menos cuidan los proyectos de feria, y por eso destaca.

- **Elige a tu entrenador:** tres personalidades con frases propias:
  - **Sargento:** "¡Esa espalda, soldado!"
  - **Abuela:** "Mijo, siéntese bien."
  - **Zen:** "Respira y alinea tus hombros."
  Esta sola idea hace reír, y la gente la cuenta.
- **Graben las voces ustedes:** suena mucho mejor que el motor actual y evita depender de las voces instaladas en Windows. También pueden probar Piper, una voz neuronal que funciona sin internet.
- **Sensor de estacionamiento:** un pitido que se acelera mientras la postura empeora y se apaga al corregirla. Se entiende sin palabras.
- **Sonidos de videojuego:** un "ding" por repetición bien hecha, aplausos al batir el récord y música con ritmo durante el reto.
- **Metrónomo de ejercicio:** marca el ritmo de bajar y subir.

**Dificultad:** baja. **En el stand habrá ruido:** lleven una bocina, y que todo lo importante también se vea en pantalla.

---

## Ideas de apoyo

Elijan solo si ya tienen lo anterior funcionando.

### Modo Salud

- **Prueba de la silla de 30 segundos:** cuenta cuántas veces una persona se sienta y se levanta en 30 segundos. Es una prueba real de fuerza de piernas en adultos mayores (forma parte del programa STEADI de los CDC). Da un ángulo de **impacto social**, y el programa ya distingue sentado de de pie.
- **Prueba de equilibrio:** cronometra cuánto tiempo aguanta alguien en un pie.
- **Evaluación ergonómica:** un puntaje de riesgo inspirado en RULA o REBA, métodos reales de ergonomía laboral. Es interesante, pero el público no lo aprecia tanto como un juego. Hay que decir "inspirado en" y aclarar que **no es un diagnóstico médico**.
- **Revisión de la vista:** avisa si estás muy cerca de la pantalla, usando el tamaño de la cara, y recuerda la regla 20-20-20 para descansar los ojos.

### Modo Oficina y Estudio

- **Pausa activa guiada:** cuando toca la pausa, un entrenador en pantalla guía estiramientos de cuello y hombros, y la cámara **verifica que los hiciste**. El recordatorio ya existe; esto lo vuelve interactivo.
- **Borde de pantalla:** una franja fina alrededor del monitor cambia de color según tu postura, sin interrumpir el trabajo.
- **Modo aula:** para que un profesor vea un semáforo durante una clase. Sirve para el discurso de impacto, aunque sea solo un prototipo.

### Mascota y juego

- **Mascota virtual:** una planta o personaje que crece y se ve feliz con buena postura y se marchita con mala. Es simple de hacer con unas pocas imágenes, y gusta a todas las edades.
- **Logros y rachas:** "5 días seguidos", "100 sentadillas", "1 hora sin encorvarte". Ya existe el historial para calcularlos.
- **Minijuego controlado con el cuerpo:** por ejemplo, esquivar obstáculos moviéndose a los lados o saltar con una sentadilla. Es muy llamativo, aunque se aleja del tema de la postura.

### Accesibilidad

- **Comandos de voz:** decir "iniciar" o "detener" sin tocar nada, con Vosk, que funciona sin internet. Luce muy bien en una demo si el stand no es muy ruidoso.
- **Modo sin audio y modo sin pantalla:** avisos solo visuales para personas sordas, y solo sonoros para personas ciegas. Pocos proyectos lo consideran, y permite que más personas participen.
- **Modo silla de ruedas:** evalúa solo el tronco, el cuello y los hombros.
- **Texto grande y alto contraste,** para adultos mayores.

### Presentación de la app

- **Pantalla de espera:** cuando no hay nadie, la pantalla muestra una animación, el ranking y el mensaje "Párate aquí y mide tu postura". El stand trabaja solo.
- **Interfaz a pantalla completa:** botones grandes, pensada para verse a 3 metros.
- **Marca propia:** nombre, logo y colores. "PosturIA" es descriptivo, pero no se recuerda.
- **Cuenta regresiva y celebración:** "3, 2, 1, ¡ya!" y confeti al terminar.

---

## Comparación

El impacto mide cuánto ayuda a ganar votos del público. Las estimaciones son aproximadas: días de trabajo de una persona que ya conoce Python.

| Idea | Impacto en votos | Dificultad | Días |
|---|---|---|---|
| Sentadillas y saltos | Muy alto | Media | 3 a 5 |
| Reto con ranking | Muy alto | Baja | 1 a 2 |
| Tarjeta o sticker con el stand | Muy alto | Muy baja | Menos de 1 |
| Pantalla de espera y pantalla completa | Muy alto | Baja | 1 a 2 |
| Espejo inteligente | Muy alto | Baja a media | 2 a 4 |
| Duelo 1 contra 1 | Muy alto | Media | 3 a 4 |
| Retos de brazos y estatua | Alto | Baja | 1 a 2 |
| Entrenadores con voces propias | Alto | Baja | 1 a 2 |
| Sensor de estacionamiento y sonidos | Alto | Baja | 1 |
| Puntaje 0 a 100 y antes/después | Alto | Baja | 1 a 3 |
| Retrato de esqueleto | Medio a alto | Baja a media | 1 a 2 |
| Mascota virtual | Medio | Baja | 1 a 2 |
| Prueba de la silla | Medio | Baja | 1 |
| Comandos de voz | Bajo a medio, por el ruido | Media | 2 |
| Pausa activa guiada | Bajo | Media | 2 a 3 |
| Plancha y flexiones | Bajo | Media a alta | 3 a 5 |
| Puntaje ergonómico tipo RULA | Bajo | Media a alta | 3 a 5 |

---

## Plan sugerido

No conozco la fecha de la feria, así que el plan va por etapas. **Si queda poco tiempo, corten desde abajo.**

**Etapa 1: que no falle.** Averiguar las reglas de votación y el espacio del stand. Conseguir webcam y trípode, y probar con desconocidos.

**Etapa 2: el juego.** Sentadillas y saltos, reto de 30 segundos y ranking. Con esto ya tienen algo que mostrar.

**Etapa 3: lo que atrae desde el pasillo.** Espejo inteligente, puntaje grande, pantalla completa y pantalla de espera.

**Etapa 4: la personalidad.** Voces, sonidos, marca, tarjetas y retos para todo tipo de persona.

**Etapa 5: el duelo.** Solo si lo anterior funciona bien.

**Etapa 6: ensayo.** Congelen el código al menos una semana antes. Hagan un ensayo general con compañeros que nunca lo hayan visto y midan cuánto dura cada visita.

---

## Cuando alguien llega al stand

Todo debe tomar **90 segundos o menos**. Ensáyenlo hasta que salga solo:

1. **Gancho (10 s):** la pantalla muestra el récord. "¿Le ganas al primer lugar?" o "Hagamos un duelo".
2. **Ubicación (10 s):** se para en la marca del piso y escribe su apodo, o se lo escribe un miembro del equipo.
3. **Medición (10 s):** ve su esqueleto y su puntaje de postura.
4. **Reto (30 s):** cuenta regresiva, contador grande, voz y sonidos.
5. **Resultado (10 s):** puntaje, aplausos y su lugar en el ranking. "¡Quedaste tercero!"
6. **Recuerdo (10 s):** recibe su tarjeta con el nombre y el número del stand. Si las reglas lo permiten: "Si te gustó, acuérdate de nosotros al votar".

Si alguien quiere saber más, ahí sí expliquen. Quien solo quiere jugar no debe escuchar una charla.

### Lo que preguntará la gente

Respuestas de una frase:

- **¿Dónde está la IA?** "Detecta 33 puntos de tu cuerpo en tiempo real, mide tus ángulos y una red neuronal clasifica tu postura."
- **¿Graba mi cara?** "No. Todo se procesa en esta computadora y el video no se guarda."
- **¿Para qué sirve en la vida real?** "Para corregir tu postura mientras estudias o trabajas, y para hacer ejercicio bien hecho en casa."
- **¿Sirve en cualquier computadora?** "Solo necesita una cámara y funciona sin internet."
- **¿Reemplaza a un fisioterapeuta?** "No. Ayuda a crear buenos hábitos, pero no es un diagnóstico."

---

## El stand

El stand es parte del proyecto.

- **Pantalla grande o TV hacia el pasillo,** con el esqueleto y el ranking a la vista. Es su mejor anuncio.
- **Un letrero enorme con una sola frase,** como "¿Cuántas sentadillas perfectas puedes hacer?". Se debe leer al pasar.
- **Marca en el piso** con cinta: "Párate aquí". Soluciona el problema de la distancia a la cámara.
- **Fondo liso y buena luz.** Lleven una lámpara. Eviten tener una ventana detrás de la persona.
- **Un lugar claro para esperar,** desde donde se vea la pantalla. Una pequeña fila atrae más gente.
- **Una silla** para el reto de estatua.
- **Un cartel de privacidad:** "No se guarda tu imagen".
- **Roles por turnos:** uno llama a la gente en el pasillo, otro guía al jugador y otro maneja el equipo. Roten para no cansarse: serán muchas horas.
- **Energía:** alargador, cargadores y una regleta. Dejen el portátil siempre conectado.
- **Sonido a un volumen justo:** que se oiga en el stand sin molestar a los vecinos, o podrían pedirles que lo bajen.

### Plan B

Lleven esto preparado:

- **Un video de la demo funcionando,** por si falla todo.
- **Un segundo portátil** con el proyecto instalado y probado.
- **Copia del proyecto en una memoria USB,** con el `.venv` listo o los instaladores.
- **Una bocina con cable** y adaptadores.
- **Todo funciona sin internet:** pruébenlo con el Wi-Fi apagado.

---

## Números para mostrar

Con voto del público no hacen falta estudios, pero unos pocos números reales ayudan:

- **Contador en vivo:** "Hoy han jugado 143 personas". Si otros ya participaron, más gente se anima.
- **Récord del día,** grande y visible.
- **Precisión real:** si alguien pregunta, digan el número de `train.py`. No lo inflen.

**No inventen ni copien cifras de salud sin fuente.** Si usan un dato en el póster, que sea de la OMS o de un artículo científico, con la referencia.

---

## Advertencias

- **Nada de esto garantiza el primer lugar.** Depende de quién vote, de las reglas y de los otros proyectos. Lo que sí mejora mucho sus posibilidades es que mucha gente lo pruebe, se divierta y los recuerde.
- **Lo más peligroso es agregar cosas a última hora.** Una función que falla frente a la fila resta más de lo que suma.
- **Prueben pronto con hardware real.** Las estimaciones de fiabilidad de este documento son expectativas, no mediciones: hay que confirmarlas con su cámara y su espacio.
