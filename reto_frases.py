"""Los tres entrenadores de PosturIA, todo lo que dicen y los consejos finales.

Cada frase tiene varias versiones y el juego elige una sin repetir la anterior.
Las frases ya estan generadas en audio/: el juego no necesita internet. Para
cambiar una, edita este archivo y corre tools/generar_audio.py (eso si la usa).
"""
MAX_NUMERO = 40  # hasta aqui se cuentan las repeticiones en voz alta

NUMEROS = (
    "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
    "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho",
    "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
    "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve", "treinta",
    "treinta y uno", "treinta y dos", "treinta y tres", "treinta y cuatro", "treinta y cinco",
    "treinta y seis", "treinta y siete", "treinta y ocho", "treinta y nueve", "cuarenta",
)

# Lo que se ve en pantalla al final: {clave: (que mejorar, como hacerlo)}.
CONSEJOS = {
    "profundidad": ("Baja más en cada sentadilla", "Lleva la cadera hacia atrás, como si te sentaras en una silla, hasta que los muslos queden casi horizontales."),
    "espalda": ("Inclinas mucho el tronco", "Saca el pecho, mira al frente y mantén el peso en los talones para que el torso quede más erguido."),
    "encorvada": ("Se te encorva la espalda", "Junta suavemente los omóplatos y aprieta el abdomen. Mantén la cabeza alineada con la espalda."),
    "rodillas": ("Tus rodillas se juntan", "Empuja las rodillas hacia afuera, en la misma dirección que las puntas de los pies."),
    "hombros": ("Bajas con los hombros torcidos", "Reparte el peso por igual en las dos piernas y mantén los hombros nivelados."),
    "ritmo": ("Vas demasiado rápido", "Baja en dos segundos y sube en uno. Una repetición lenta y controlada vale más que tres rápidas."),
    "pocas": ("Hiciste pocas repeticiones", "Empieza con un ritmo constante y no te detengas: mejor seguido que a tirones."),
    "tiempo": ("Tardas en llegar a la pose", "Mira la figura durante la preparación y empieza a moverte antes de que termine la cuenta."),
    "quieto": ("Te cuesta quedarte quieto", "Respira lento, aprieta el abdomen y fija la mirada en un punto de la pared."),
    "brazos": ("Ajusta mejor los brazos", "Compara tus brazos con la figura verde: lleva primero el codo a su sitio y después la mano."),
    "piernas": ("Ajusta mejor las piernas", "Abre la pierna hacia el lado, no hacia el frente, y apóyate bien en el pie contrario."),
    "un_lado": ("Usas más una mano", "Trabaja con las dos manos: la que menos usas también necesita coordinación."),
    "reaccion": ("Puedes reaccionar más rápido", "Mantén las manos arriba y a la altura del pecho, listas para salir hacia cualquier lado."),
    "bombas": ("Tocaste muchas bombas", "Mira antes de mover la mano: un segundo de atención vale más que tres puntos perdidos."),
}

ENTRENADORES = {
    "sargento": {
        "nombre": "Sargento",
        "voz": {"voice": "es-GT-AndresNeural", "rate": "+12%", "pitch": "-8Hz"},
        "formato_numero": "¡{}!",
        "frases": {
            "hola": ("¡Atención! ¡Aquí manda el sargento!", "¡Firmes! ¡Se presenta su sargento!", "¡Soldado, a sus órdenes no, a las mías!", "¡Se acabó el descanso, recluta!", "¡Hoy vengo a sacarle el sudor!"),
            "marca": ("¡Párese en la marca, soldado!", "¡A la marca, de frente a la cámara!", "¡Recluta, a su posición!", "¡Ubíquese en la marca, ahora mismo!", "¡Formación! ¡Párese donde le digo!"),
            "listo": ("¡Prepárese, soldado!", "¡Atento a mi señal!", "¡Concéntrese, recluta!", "¡Respire y prepárese!", "¡Esto va en serio!"),
            "tres": ("¡Tres!",),
            "dos": ("¡Dos!",),
            "uno": ("¡Uno!",),
            "ya": ("¡Ahora!", "¡Adelante, soldado!", "¡Muévase!", "¡A trabajar!", "¡Comience ya!"),
            "baja": ("¡Baje más, soldado!", "¡Eso no cuenta, más abajo!", "¡Más profundo, recluta!", "¡No me engañe, baje!", "¡A media sentadilla no se gana la guerra!"),
            "pecho": ("¡Pecho arriba!", "¡Espalda recta, soldado!", "¡Enderécese!", "¡Levante ese pecho!", "¡No se me encorve!"),
            "rodillas": ("¡Rodillas afuera!", "¡Abra esas rodillas!", "¡Las rodillas no se juntan, soldado!", "¡Empuje las rodillas hacia afuera!", "¡Cuide esas rodillas!"),
            "rapido": ("¡Más despacio, con control!", "¡Esto no es una carrera!", "¡Controle el movimiento!", "¡Calma, soldado, técnica primero!", "¡Lento y bien hecho!"),
            "bien": ("¡Eso es, soldado!", "¡Así me gusta!", "¡Siga así!", "¡Excelente disciplina!", "¡Ahora sí, recluta!"),
            "animo": ("¡Diez segundos! ¡No se me rinda!", "¡Últimos diez segundos!", "¡Deje todo, soldado!", "¡Ya casi! ¡Más fuerte!", "¡El final se gana aquí!"),
            "alto": ("¡Alto!", "¡Descanse!", "¡Se acabó!", "¡Pare ahí, soldado!", "¡Fin del entrenamiento!"),
            "imita": ("¡Imite la pose, soldado!", "¡Copie esa posición!", "¡A la pose, ya!", "¡Ejecute la figura!", "¡Mire y obedezca!"),
            "aguanta": ("¡Aguante! ¡Ni un músculo!", "¡Firme ahí!", "¡No se mueva, recluta!", "¡Como una roca!", "¡Quieto hasta que yo diga!"),
            "pose_ok": ("¡Correcto! ¡Siguiente!", "¡Aprobado!", "¡Bien ejecutado!", "¡Esa pasa!", "¡Siguiente orden!"),
            "vida": ("¡Se movió! ¡Quieto, soldado!", "¡Falló, recluta!", "¡Eso no fue lo que ordené!", "¡Concéntrese!", "¡Una vida menos!"),
            "caida": ("¡Inaceptable! ¡Fuera de formación!", "¡Misión fallida, soldado!", "¡Se acabó para usted!", "¡A descansar, recluta!", "¡Le falta entrenamiento!"),
            "res_bien": ("¡Muy bien! ¡Usted es material de élite!", "¡Buen trabajo, soldado!", "¡Me sorprendió, recluta!", "¡Merece un ascenso!", "¡Así se hace en este batallón!"),
            "res_mal": ("¡Mediocre! ¡Vuelva cuando sea un verdadero soldado!", "¡Eso no alcanza, recluta!", "¡Le falta disciplina!", "¡A entrenar más, soldado!", "¡Mañana quiero verlo mejor!"),
            "record": ("¡Nuevo récord! ¡Le concedo una medalla!", "¡Récord del batallón!", "¡Soldado del día!", "¡Histórico, recluta!", "¡Nadie lo había logrado!"),
            "turno2": ("¡Siguiente soldado, a la marca!", "¡Otro recluta, adelante!", "¡Le toca, soldado!", "¡Veamos si usted puede superarlo!", "¡Al frente el siguiente!"),
            "gana1": ("¡Victoria para el primer soldado!", "¡Gana el primero!", "¡El primer recluta se lleva la medalla!", "¡Primer soldado, misión cumplida!", "¡El primero manda hoy!"),
            "gana2": ("¡Victoria para el segundo soldado!", "¡Gana el segundo!", "¡El segundo recluta se lleva la medalla!", "¡Segundo soldado, misión cumplida!", "¡El segundo manda hoy!"),
            "empate": ("¡Empate! ¡Los dos son dignos del batallón!", "¡Igualados, soldados!", "¡Nadie cede terreno!", "¡Empate técnico!", "¡Los dos sirven a la patria!"),
            "estrellas": ("¡Atrape las estrellas, soldado!", "¡Blancos a la vista!", "¡Toque las estrellas, evite las bombas!", "¡Reflejos, recluta!", "¡Manos arriba y a cazar!"),
            "bomba": ("¡Era una bomba, soldado!", "¡Campo minado!", "¡Mire lo que toca!", "¡Explosión! ¡Atención!", "¡Esa no, recluta!"),
            "racha": ("¡Racha imparable!", "¡Excelente puntería!", "¡Siga disparando!", "¡Tirador de élite!", "¡No falle ninguna!"),
            "dorada": ("¡Estrella de oro!", "¡Condecoración!", "¡Blanco dorado!", "¡Tres puntos, soldado!", "¡Eso brilla!"),
            "tip_profundidad": ("¡Soldado, baje la cadera como si se sentara!",),
            "tip_espalda": ("¡Pecho al frente y peso en los talones!",),
            "tip_encorvada": ("¡Enderece esa espalda, soldado, junte los hombros atrás!",),
            "tip_rodillas": ("¡Empuje las rodillas hacia afuera!",),
            "tip_hombros": ("¡Nivele esos hombros, recluta!",),
            "tip_ritmo": ("¡Menos prisa y más control!",),
            "tip_pocas": ("¡Quiero más repeticiones, soldado!",),
            "tip_tiempo": ("¡Reaccione antes, recluta!",),
            "tip_quieto": ("¡Apriete el abdomen y quédese firme!",),
            "tip_brazos": ("¡Revise esos brazos, soldado!",),
            "tip_piernas": ("¡Abra la pierna hacia el lado, recluta!",),
            "tip_un_lado": ("¡Use las dos manos, soldado!",),
            "tip_reaccion": ("¡Manos arriba y más reflejos!",),
            "tip_bombas": ("¡Mire antes de tocar, recluta!",),
            "tip_ok": ("¡Técnica impecable, soldado! ¡Nada que corregir!",),
        },
    },
    "abuela": {
        "nombre": "Abuelita",
        "voz": {"voice": "es-MX-DaliaNeural", "rate": "-18%", "pitch": "-2Hz"},
        "formato_numero": "{}.",
        "efecto": "temblor",
        "frases": {
            "hola": ("Ay, mijito, aquí está tu abuelita.", "Venga, mi amor, salude a su abuela.", "Qué grande está, mijito.", "Hola, mi cielo. ¿Ya comió?", "Aquí estoy, con mis huesitos."),
            "marca": ("Venga, mijito, párese ahí en la marca.", "Póngase ahí, mi amor, de frente.", "Más para acá, mijito, en la marca.", "Ándele, a su lugar.", "Ahí, mi cielo, donde lo vea."),
            "listo": ("Prepárese, mi amor, con cuidadito.", "Despacito, mijito.", "Respire, mi cielo.", "A ver, con calma.", "Ya mero empezamos, corazón."),
            "tres": ("Tres.",),
            "dos": ("Dos.",),
            "uno": ("Uno.",),
            "ya": ("¡Ándele!", "Ahora sí, mijito.", "Vamos, mi amor.", "Empiece, corazón.", "A darle, mi cielo."),
            "baja": ("Baje otro poquito, mi amor.", "Un poquito más, mijito.", "Más abajo, corazón.", "Así no, mi cielo, más abajo.", "Ándele, otro poquito."),
            "pecho": ("Esa espalda, mijito, derechito.", "Enderécese, mi amor.", "No se me jorobe, corazón.", "Derechito, como le enseñé.", "Pecho arriba, mi cielo."),
            "rodillas": ("Cuide esas rodillas, mijito.", "Rodillitas para afuera, mi amor.", "No junte las rodillas, corazón.", "Ábralas tantito, mi cielo.", "Ay, sus rodillitas."),
            "rapido": ("Despacito, mijito, que se lastima.", "No corra, mi amor.", "Con calma, corazón.", "Más lento, mi cielo.", "Ay, qué prisa, mijito."),
            "bien": ("Eso, así me gusta.", "Muy bien, mi amor.", "Qué bonito, mijito.", "Así, corazón.", "Mi nieto favorito."),
            "animo": ("Ya mero, mijito, diez segunditos.", "Tantito más, mi amor.", "Ya casi, corazón.", "No se canse, mi cielo.", "Un esfuercito más."),
            "alto": ("Ya, ya, descanse.", "Suficiente, mijito.", "Ya estuvo, mi amor.", "Siéntese tantito, corazón.", "Ya, mi cielo, tómese un agüita."),
            "imita": ("A ver, mijito, haga como yo.", "Póngase así, mi amor.", "Copie la figurita, corazón.", "Mire bien, mi cielo.", "Ahora esta, mijito."),
            "aguanta": ("Quietito, mi amor, aguante.", "No se mueva, mijito.", "Tranquilo, corazón.", "Quietecito como en misa.", "Así, sin moverse, mi cielo."),
            "pose_ok": ("Eso, mijito, otra.", "Muy bonita, mi amor.", "Qué lindo, corazón.", "Sabía que podía, mi cielo.", "Así mero, mijito."),
            "vida": ("Ay, se movió, mijito.", "Casi, mi amor.", "No pasa nada, corazón.", "Otra vez, mi cielo.", "Ay, Dios mío, mijito."),
            "caida": ("Ay, mi amor, ya se cansó.", "Ya estuvo, mijito, a descansar.", "No se preocupe, corazón.", "Mañana le sale, mi cielo.", "Venga, le doy un abrazo."),
            "res_bien": ("¡Qué orgullo, mijito!", "Le voy a hacer su comidita.", "Así se hace, mi amor.", "Ya se lo voy a contar a sus tías.", "Mi corazón está contento, mijito."),
            "res_mal": ("Ay, mijito, le falta comer.", "Vuelva otro día, mi amor.", "No se aflija, corazón.", "Le voy a preparar un caldito.", "Con cariño todo se logra, mi cielo."),
            "record": ("¡Ay, Dios mío, un récord!", "¡Ese es mi nieto!", "¡Mi amor, nadie lo había hecho!", "¡Voy a llamar a toda la familia!", "¡Récord, corazón, qué emoción!"),
            "turno2": ("A ver, el otro mijito, venga para acá.", "Le toca a usted, mi amor.", "Pásele, corazón.", "Ahora usted, mi cielo.", "Venga, el siguiente nietecito."),
            "gana1": ("Ganó el primer mijito. Qué lindo.", "El primero, mi amor.", "Ganó el primero, corazón.", "Felicidades al primero, mi cielo.", "El primer nietecito ganó."),
            "gana2": ("Ganó el segundo mijito. Qué lindo.", "El segundo, mi amor.", "Ganó el segundo, corazón.", "Felicidades al segundo, mi cielo.", "El segundo nietecito ganó."),
            "empate": ("Empate. Los dos son mis favoritos.", "A los dos los quiero igual.", "Empataron, mis amores.", "Les toca galletita a los dos.", "Qué bonito, los dos iguales."),
            "estrellas": ("Atrape las estrellitas, mijito.", "Cuidado con las bombitas, mi amor.", "A ver esas manitas, corazón.", "Toque las estrellitas, mi cielo.", "Ándele, que se van."),
            "bomba": ("Ay, una bomba, mijito.", "Cuidado, mi amor.", "Esa no, corazón.", "Ay, qué susto, mi cielo.", "Se me para el corazón."),
            "racha": ("Qué manitas tan rápidas.", "Así, mi amor, sin parar.", "Mire nada más, corazón.", "Ni yo en mis tiempos, mijito.", "Qué bárbaro, mi cielo."),
            "dorada": ("Una estrellita de oro.", "Ay, qué brillante, mijito.", "Esa vale tres, mi amor.", "Qué bonita, corazón.", "Se la merece, mi cielo."),
            "tip_profundidad": ("Mijito, baje como si se sentara en su silla.",),
            "tip_espalda": ("Saque el pecho, mi amor, y mire al frente.",),
            "tip_encorvada": ("No se encorve, mijito, que le va a doler la espalda.",),
            "tip_rodillas": ("Rodillitas hacia afuera, mi cielo.",),
            "tip_hombros": ("Esos hombros parejitos, corazón.",),
            "tip_ritmo": ("Más despacito, mijito, sin prisa.",),
            "tip_pocas": ("La próxima, unas cuantas más, mi amor.",),
            "tip_tiempo": ("Mire la figurita con tiempo, corazón.",),
            "tip_quieto": ("Respire despacito y quédese quieto, mi cielo.",),
            "tip_brazos": ("Fíjese bien en sus bracitos, mijito.",),
            "tip_piernas": ("Abra la piernita hacia el lado, mi amor.",),
            "tip_un_lado": ("Use las dos manitas, corazón.",),
            "tip_reaccion": ("Tenga las manitas listas, mi cielo.",),
            "tip_bombas": ("Fíjese antes de tocar, mijito.",),
            "tip_ok": ("Todo perfecto, mijito. Qué orgullo.",),
        },
    },
    "zen": {
        "nombre": "Maestro Zen",
        "voz": {"voice": "es-MX-JorgeNeural", "rate": "-25%", "pitch": "-6Hz"},
        "formato_numero": "{}.",
        "frases": {
            "hola": ("Bienvenido. Respira. Estás aquí.", "El camino empieza con un paso.", "Saludo a tu cuerpo.", "Cierra los ojos. Ábrelos.", "Todo es ahora."),
            "marca": ("Camina hasta la marca.", "Encuentra tu lugar.", "Sitúate frente a la cámara.", "Tu sitio te espera.", "Avanza con calma hasta la marca."),
            "listo": ("Respira profundo.", "Calma la mente.", "Siente tus pies.", "Inhala.", "Prepara tu espíritu."),
            "tres": ("Tres.",),
            "dos": ("Dos.",),
            "uno": ("Uno.",),
            "ya": ("Comienza.", "Fluye.", "Ahora.", "Empieza tu camino.", "Muévete con calma."),
            "baja": ("Desciende un poco más.", "La profundidad enseña.", "Baja como la lluvia.", "Un poco más abajo.", "Desciende sin miedo."),
            "pecho": ("Abre el pecho.", "Tu espalda es un bambú.", "Crece hacia el cielo.", "Alinea tu columna.", "Eleva el corazón."),
            "rodillas": ("Abre las rodillas.", "Tus rodillas siguen a tus pies.", "Dale espacio a tus piernas.", "Rodillas hacia afuera.", "Alinea tus piernas."),
            "rapido": ("Más lento.", "El río no tiene prisa.", "La calma es fuerza.", "Baja el ritmo.", "Respira en cada movimiento."),
            "bien": ("Muy bien.", "Fluyes.", "Así es.", "Armonía.", "Tu cuerpo escucha."),
            "animo": ("Diez segundos. Sigue fluyendo.", "El final también es camino.", "Ya casi.", "Mantén la calma.", "Un último esfuerzo sereno."),
            "alto": ("Descansa.", "Detente.", "Exhala.", "Ha terminado.", "Agradece a tu cuerpo."),
            "imita": ("Observa la postura. Imítala.", "Sé esa forma.", "Adopta la figura.", "Tu cuerpo la conoce.", "Encuentra esta postura."),
            "aguanta": ("Permanece. Sé la montaña.", "No te muevas.", "Quietud.", "Sé como la piedra.", "Solo respira."),
            "pose_ok": ("Bien. Fluye a la siguiente.", "Equilibrio.", "Así es.", "Hermosa quietud.", "Continúa."),
            "vida": ("La montaña se movió.", "Vuelve a tu centro.", "Respira. Inténtalo de nuevo.", "Todo error enseña.", "Suelta y regresa."),
            "caida": ("Todo termina. Respira.", "La caída también es maestra.", "El camino continúa mañana.", "Acepta y descansa.", "Volverás más sabio."),
            "res_bien": ("Has encontrado el equilibrio.", "Tu cuerpo y tu mente son uno.", "Gran armonía.", "Estás en el camino correcto.", "Has fluido muy bien."),
            "res_mal": ("Cada caída es una enseñanza.", "El bambú también se dobla.", "La paciencia te guiará.", "Vuelve con calma.", "Todo maestro fue aprendiz."),
            "record": ("Has alcanzado la iluminación. Es un nuevo récord.", "Un nuevo récord nace.", "Has superado a todos. Y a ti mismo.", "El universo sonríe.", "La perfección te visita."),
            "turno2": ("Que se acerque el siguiente.", "Otro viajero llega.", "Es tu momento.", "El camino te llama.", "Avanza, siguiente alumno."),
            "gana1": ("Hoy el camino favorece al primero.", "El primero encuentra la victoria.", "Gana el primero.", "El primero alcanzó la cima.", "La victoria descansa en el primero."),
            "gana2": ("Hoy el camino favorece al segundo.", "El segundo encuentra la victoria.", "Gana el segundo.", "El segundo alcanzó la cima.", "La victoria descansa en el segundo."),
            "empate": ("Empate. Ambos son uno.", "Dos caminos, una cima.", "No hay vencedor ni vencido.", "Equilibrio perfecto.", "Sois iguales."),
            "estrellas": ("Toca las estrellas con calma.", "Las estrellas buscan tus manos.", "Atrapa la luz.", "Evita la oscuridad.", "Tus manos son el cielo."),
            "bomba": ("La oscuridad te tocó.", "Atención.", "Suelta esa sombra.", "Observa antes de actuar.", "Eso no era luz."),
            "racha": ("Tu mano fluye.", "Armonía perfecta.", "Estás en el presente.", "Sigues el ritmo del universo.", "Nada te detiene."),
            "dorada": ("Una luz dorada.", "El oro de la calma.", "Tres veces bendecido.", "Gran fortuna.", "El sol en tu mano."),
            "tip_profundidad": ("Desciende más, como si te sentaras.",),
            "tip_espalda": ("Abre el pecho y siente tus talones.",),
            "tip_encorvada": ("Tu espalda es un bambú. Mantenla recta.",),
            "tip_rodillas": ("Tus rodillas siguen la dirección de tus pies.",),
            "tip_hombros": ("Equilibra tus hombros.",),
            "tip_ritmo": ("Más lento. El control es fuerza.",),
            "tip_pocas": ("La constancia es el camino.",),
            "tip_tiempo": ("Observa primero. Muévete después.",),
            "tip_quieto": ("Respira lento y encuentra la quietud.",),
            "tip_brazos": ("Observa tus brazos con atención.",),
            "tip_piernas": ("Abre la pierna hacia el lado.",),
            "tip_un_lado": ("Busca el equilibrio entre tus dos manos.",),
            "tip_reaccion": ("Mantén las manos preparadas.",),
            "tip_bombas": ("Observa antes de actuar.",),
            "tip_ok": ("Tu técnica es armonía. Nada que corregir.",),
        },
    },
}


def clave_numero(n):
    return f"n{n}"


def variantes(entrenador, clave):
    """Nombres de archivo de las versiones de una frase."""
    frases = ENTRENADORES[entrenador]["frases"].get(clave)
    if frases is None:
        return [clave]
    return [f"{clave}_{i}" for i in range(len(frases))]


def textos(entrenador):
    """Todo lo que dice un entrenador: {archivo: texto}, numeros incluidos."""
    datos = ENTRENADORES[entrenador]
    todas = {}
    for clave, frases in datos["frases"].items():
        for archivo, texto in zip(variantes(entrenador, clave), frases):
            todas[archivo] = texto
    for i, numero in enumerate(NUMEROS[:MAX_NUMERO], start=1):
        todas[clave_numero(i)] = datos["formato_numero"].format(numero.capitalize())
    return todas
