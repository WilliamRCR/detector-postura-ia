"""PosturIA: entrenador de postura y movimiento con inteligencia artificial.

Tres juegos para el stand: sentadillas, estatua perfecta y Caza Estrellas, con
entrenadores por voz, ranking, consejos al terminar y ajustes que mejoran con cada
partida.

Es un programa aparte de app_gui.py: no necesita el modelo entrenado y no toca
nada de la app de postura. Se abre a pantalla completa, pensado para una TV.

    python reto.py                        # pantalla completa
    python reto.py --ventana              # en una ventana
    python reto.py --camara 1             # usar esa camara, sin buscar la real
    python reto.py --borrar-ranking       # deja el ranking en cero
    python reto.py --borrar-aprendizaje   # olvida lo aprendido de las partidas

Se maneja con el teclado de la compu; las teclas estan en pantalla.
"""
import argparse
import math
import random
import threading
import time

import cv2
import pygame

import core
import reto_frases
from config import AppConfig
from reto_audio import Audio
from reto_logica import (
    ANKLE_L,
    ANKLE_R,
    KNEE_L,
    KNEE_R,
    NIVELES,
    RETO_ESTATUA,
    RETO_ESTRELLAS,
    RETO_SENTADILLAS,
    SEGUNDOS_ESTRELLAS,
    SEGUNDOS_SENTADILLAS,
    WRIST_L,
    WRIST_R,
    Aprendizaje,
    ContadorSentadillas,
    JuegoEstrellas,
    JuezEstatua,
    Ranking,
    angulo,
    elegir_consejos,
    medir_cuerpo,
)

# Largo de cada segmento de la figura guia, en anchos de hombros.
LARGO_SEGMENTO = {"bi": 0.9, "ai": 0.8, "bd": 0.9, "ad": 0.8, "mi": 1.3, "md": 1.3}

TITULO = "PosturIA"
ANCHO, ALTO = 1280, 720
VIDEO_W = 960
FPS = 30
MAX_APODO = 12

SEGUNDOS_MARCA = 1.2           # quieto en la marca antes de empezar
ESPERA_RODILLAS = 4.0          # si no se ven, se juega igual tras este tiempo
SEGUNDOS_RESULTADO = 6.0
SEGUNDOS_CONSEJOS = 10.0
SEGUNDOS_DUELO = 7.0
# Desde aqui el entrenador felicita.
PUNTAJE_BUENO = {RETO_SENTADILLAS: 8, RETO_ESTATUA: 5, RETO_ESTRELLAS: 15}
ESPERA_CORRECCION = 3.0        # entre una correccion de voz y la siguiente
MIN_INTENTOS_SENTADILLAS = 5   # menos que esto en 30 segundos merece un consejo

FONDO = (13, 15, 33)
PANEL = (20, 24, 52)
BLANCO = (245, 247, 255)
GRIS = (150, 156, 190)
VERDE = (46, 230, 130)
AMARILLO = (255, 205, 60)
ROJO = (255, 75, 90)
CIAN = (60, 220, 255)
MAGENTA = (240, 80, 200)

# El esqueleto completo de MediaPipe: los 33 puntos unidos.
CONEXIONES_CUERPO = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
)
CONEXIONES_DETALLE = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),     # cara
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),  # manos
    (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32),                  # pies
)
PUNTOS_CUERPO = {i for par in CONEXIONES_CUERPO for i in par}

NOMBRE_RETO = {RETO_SENTADILLAS: "Sentadillas", RETO_ESTATUA: "Estatua perfecta", RETO_ESTRELLAS: "Caza Estrellas"}
TECLAS_ENTRENADOR = {pygame.K_s: "sargento", pygame.K_a: "abuela", pygame.K_z: "zen"}
NIVELES_ORDEN = tuple(NIVELES)


class Camara(threading.Thread):
    """Lee la camara y detecta el cuerpo en su propio hilo, para que la pantalla
    no se trabe. Con buscar=True arranca con la primera camara real: la numero 0
    puede ser una virtual (OBS, Animaze) que solo muestra un mensaje."""

    def __init__(self, indice, espejo, buscar=True, al_elegir=None):
        super().__init__(daemon=True)
        self.indice = indice
        self.espejo = espejo
        self.buscar = buscar
        self.al_elegir = al_elegir
        self.error = ""
        self.aviso = "Buscando tu cámara..."
        self._lock = threading.Lock()
        self._datos = (None, None)
        self._parar = threading.Event()
        self._cambiar = threading.Event()

    def run(self):
        try:
            if self.buscar:
                self._elegida(core.buscar_camara(self.indice))
            self.aviso = "Encendiendo la cámara..."
            pose = core.new_pose()
            while not self._parar.is_set():
                self._transmitir(pose)
                if self._cambiar.is_set():
                    self._cambiar.clear()
                    with self._lock:
                        self._datos = (None, None)
                    self._elegida((self.indice + 1) % core.MAX_CAMARAS)
        except Exception as exc:  # sin esto el hilo muere callado y la pantalla se queda esperando
            self.error = f"Error con la cámara: {exc}"

    def _elegida(self, indice):
        self.indice = indice
        self.error = ""
        if self.al_elegir:
            self.al_elegir(indice)

    def _transmitir(self, pose):
        cap = core.abrir_camara(self.indice)
        try:
            if not cap.isOpened():
                self.error = f"No se pudo abrir la cámara {self.indice}. Presiona C para probar otra"
                self._cambiar.wait(timeout=0.5)
                return
            while not self._parar.is_set() and not self._cambiar.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "Se perdió la conexión con la cámara. Presiona C para probar otra"
                    self._cambiar.wait(timeout=0.5)
                    return
                self.error = ""
                if self.espejo:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = pose.process(rgb)
                puntos = resultado.pose_landmarks.landmark if resultado.pose_landmarks else None
                with self._lock:
                    self._datos = (rgb, puntos)
        finally:
            cap.release()

    def leer(self):
        with self._lock:
            return self._datos

    def siguiente(self):
        """Pasa a la siguiente camara (tecla C)."""
        self._cambiar.set()

    def detener(self):
        self._parar.set()
        if self.is_alive():
            self.join(timeout=2.0)


class Confeti:
    COLORES = (VERDE, AMARILLO, ROJO, CIAN, MAGENTA, BLANCO)

    def __init__(self):
        self.piezas = []

    def lanzar(self, n=220):
        self.piezas = [
            [random.uniform(0, ANCHO), random.uniform(-ALTO, 0), random.uniform(-40, 40),
             random.uniform(160, 420), random.choice(self.COLORES), random.randint(6, 13)]
            for _ in range(n)
        ]

    def dibujar(self, pantalla, dt):
        for p in self.piezas:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            pygame.draw.rect(pantalla, p[4], (p[0], p[1], p[5], p[5] * 0.6))
        self.piezas = [p for p in self.piezas if p[1] < ALTO + 20]


class Juego:
    def __init__(self, pantalla, camara, audio, ranking, aprendizaje=None):
        self.aprendizaje = aprendizaje
        self.pantalla = pantalla
        self.camara = camara
        self.audio = audio
        self.ranking = ranking
        self.confeti = Confeti()
        self._fuentes = {}

        self.estado = "menu"
        self.t_estado = 0.0
        self.salir = False
        self.reto = RETO_SENTADILLAS
        self.entrenador = "sargento"
        self.por_turnos = False
        self.nivel = "Normal"
        self.privacidad = False
        self.audio.elegir_entrenador(self.entrenador)

        self.jugadores = []
        self.turno = 0
        self.apodo = ""
        self.marca_desde = None
        self.cuenta_paso = 0
        self.contador = None
        self.juez = None
        self.estrellas = None
        self.aviso_10 = False
        self.ultima_correccion = -10.0
        self.ultimo_puntos = None
        self.ultimo_tick = 0
        self.flash_hasta = 0.0
        self.flash_texto = ""
        self.flash_color = VERDE
        self.resultado = {}
        self.esc_hasta = 0.0
        self._t_anterior = None

    # ---------------------------------------------------------------- flujo
    @property
    def jugador(self):
        return self.jugadores[self.turno]

    @property
    def nombre_entrenador(self):
        return reto_frases.ENTRENADORES[self.entrenador]["nombre"]

    @property
    def rival(self):
        return self.jugadores[0] if self.por_turnos and self.turno == 1 else None

    def ir(self, estado, ahora):
        self.estado = estado
        self.t_estado = ahora
        if estado == "marca":
            self.marca_desde = None
            self.audio.decir("turno2" if self.turno else "marca", ahora_mismo=True)
        elif estado == "cuenta":
            self.cuenta_paso = 0
            self.audio.decir("listo", ahora_mismo=True)
        elif estado == "consejos":
            consejos = self.resultado["consejos"]
            self.audio.decir(f"tip_{consejos[0]}" if consejos else "tip_ok", ahora_mismo=True)

    def nueva_partida(self, ahora):
        self.jugadores = []
        self.turno = 0
        self.apodo = ""
        self.ir("apodo", ahora)

    def cancelar(self, ahora):
        self.audio.parar_musica()
        self.audio.silenciar_voz()
        self.ir("menu", ahora)

    def empezar_juego(self, puntos, ahora):
        cuerpo = medir_cuerpo(puntos)
        if cuerpo is None:
            self.ir("marca", ahora)
            return
        nivel = NIVELES[self.nivel]
        if self.aprendizaje:
            nivel = self.aprendizaje.nivel_ajustado(nivel)
        self.aviso_10 = False
        self.audio.efecto("go")
        if self.reto == RETO_SENTADILLAS:
            self.contador = ContadorSentadillas(nivel)
            self.contador.calibrar(cuerpo)
            self.audio.decir("ya", ahora_mismo=True)
            self.audio.musica("musica_reto")
        elif self.reto == RETO_ESTATUA:
            self.juez = JuezEstatua(nivel, piernas=cuerpo.rodillas_visibles, aprendizaje=self.aprendizaje)
            self.juez.iniciar(ahora)
            self.audio.decir("imita", ahora_mismo=True)
            self.audio.musica("musica_estatua")
        else:
            self.estrellas = JuegoEstrellas(nivel)
            self.audio.decir("estrellas", ahora_mismo=True)
            self.audio.musica("musica_reto")
        self.ir("juego", ahora)

    def consejos_de_la_partida(self):
        if self.reto == RETO_SENTADILLAS:
            c = self.contador
            errores = c.errores.copy()
            if c.intentos < MIN_INTENTOS_SENTADILLAS:
                errores["pocas"] = max(c.intentos, 1)
            return elegir_consejos(errores, c.intentos)
        if self.reto == RETO_ESTATUA:
            j = self.juez
            return elegir_consejos(j.errores, j.poses + sum(j.errores.values()))
        e = self.estrellas
        return elegir_consejos(e.errores, e.aciertos + e.perdidas + e.bombas)

    def aprender_de_la_partida(self):
        if not self.aprendizaje:
            return
        if self.reto == RETO_SENTADILLAS:
            for pico in self.contador.picos:
                self.aprendizaje.registrar_profundidad(pico)
        self.aprendizaje.terminar_partida()

    def terminar_juego(self, puntaje, extra, ahora):
        anterior = self.ranking.record(self.reto)
        self.jugador["puntaje"] = puntaje
        self.jugador["extra"] = extra
        posicion = self.ranking.guardar(self.reto, self.jugador["apodo"], puntaje, extra)
        record = puntaje > 0 and (anterior is None or puntaje > anterior)
        buena = puntaje >= PUNTAJE_BUENO[self.reto]
        self.resultado = {
            "puntaje": puntaje, "extra": extra, "posicion": posicion, "record": record,
            "consejos": self.consejos_de_la_partida(),
        }
        self.aprender_de_la_partida()

        self.audio.parar_musica()
        if record:
            self.audio.efecto("fanfarria")
            self.audio.efecto("aplausos")
            self.audio.decir("record", ahora_mismo=True)
            self.confeti.lanzar()
        elif buena:
            self.audio.efecto("aplausos")
            self.audio.decir("res_bien", ahora_mismo=True)
        else:
            self.audio.decir("res_mal", ahora_mismo=True)
        self.ir("resultado", ahora)

    def despues_de_los_consejos(self, ahora):
        if self.por_turnos and self.turno == 0:
            self.turno = 1
            self.ir("marca", ahora)
        elif self.por_turnos:
            self.mostrar_ganador(ahora)
        else:
            self.ir("menu", ahora)

    def mostrar_ganador(self, ahora):
        a, b = self.jugadores
        pa, pb = (a["puntaje"], a["extra"]), (b["puntaje"], b["extra"])
        self.resultado["ganador"] = None if pa == pb else (0 if pa > pb else 1)
        clave = {None: "empate", 0: "gana1", 1: "gana2"}[self.resultado["ganador"]]
        self.audio.efecto("fanfarria")
        self.audio.efecto("aplausos")
        self.audio.decir(clave, ahora_mismo=True)
        self.confeti.lanzar()
        self.ir("duelo", ahora)

    # -------------------------------------------------------------- teclado
    def manejar(self, evento, ahora):
        if evento.type == pygame.QUIT:
            self.salir = True
        elif evento.type == pygame.TEXTINPUT and self.estado == "apodo":
            limpio = "".join(c for c in evento.text if c.isprintable())
            self.apodo = (self.apodo + limpio)[:MAX_APODO]
        elif evento.type == pygame.KEYDOWN:
            self._tecla(evento.key, ahora)

    def _tecla(self, tecla, ahora):
        if tecla == pygame.K_F11:
            pygame.display.toggle_fullscreen()
        elif tecla == pygame.K_ESCAPE:
            if self.estado != "menu":
                self.cancelar(ahora)
            elif ahora < self.esc_hasta:
                self.salir = True
            else:
                self.esc_hasta = ahora + 2.0
        elif self.estado == "menu":
            self._tecla_menu(tecla, ahora)
        elif self.estado == "apodo":
            self._tecla_apodo(tecla, ahora)
        elif self.estado == "marca" and tecla == pygame.K_RETURN:
            self.ir("cuenta", ahora)
        elif tecla == pygame.K_RETURN and self.estado == "resultado":
            self.ir("consejos", ahora)
        elif tecla == pygame.K_RETURN and self.estado == "consejos":
            self.despues_de_los_consejos(ahora)
        elif tecla == pygame.K_RETURN and self.estado == "duelo":
            self.ir("menu", ahora)

    def _tecla_menu(self, tecla, ahora):
        if tecla == pygame.K_1:
            self.reto = RETO_SENTADILLAS
        elif tecla == pygame.K_2:
            self.reto = RETO_ESTATUA
        elif tecla == pygame.K_3:
            self.reto = RETO_ESTRELLAS
        elif tecla in TECLAS_ENTRENADOR:
            self.entrenador = TECLAS_ENTRENADOR[tecla]
            self.audio.elegir_entrenador(self.entrenador)
            self.audio.decir("hola", ahora_mismo=True)
        elif tecla == pygame.K_t:
            self.por_turnos = not self.por_turnos
        elif tecla == pygame.K_d:
            i = NIVELES_ORDEN.index(self.nivel)
            self.nivel = NIVELES_ORDEN[(i + 1) % len(NIVELES_ORDEN)]
        elif tecla == pygame.K_p:
            self.privacidad = not self.privacidad
        elif tecla == pygame.K_c:
            self.camara.siguiente()
        elif tecla == pygame.K_RETURN:
            self.nueva_partida(ahora)

    def _tecla_apodo(self, tecla, ahora):
        if tecla == pygame.K_BACKSPACE:
            self.apodo = self.apodo[:-1]
        elif tecla == pygame.K_RETURN:
            numero = len(self.jugadores) + 1
            self.jugadores.append({"apodo": self.apodo.strip() or f"Jugador {numero}", "puntaje": 0, "extra": 0})
            self.apodo = ""
            if not self.por_turnos or len(self.jugadores) == 2:
                self.ir("marca", ahora)

    # --------------------------------------------------------------- logica
    def actualizar(self, ahora):
        self.audio.actualizar()
        _, puntos = self.camara.leer()
        self.ultimo_puntos = puntos
        if self.estado == "marca":
            self._actualizar_marca(puntos, ahora)
        elif self.estado == "cuenta":
            self._actualizar_cuenta(puntos, ahora)
        elif self.estado == "juego":
            if self.reto == RETO_SENTADILLAS:
                self._actualizar_sentadillas(puntos, ahora)
            elif self.reto == RETO_ESTATUA:
                self._actualizar_estatua(puntos, ahora)
            else:
                self._actualizar_estrellas(puntos, ahora)
        elif self.estado == "resultado" and ahora - self.t_estado >= SEGUNDOS_RESULTADO:
            self.ir("consejos", ahora)
        elif self.estado == "consejos" and ahora - self.t_estado >= SEGUNDOS_CONSEJOS:
            self.despues_de_los_consejos(ahora)
        elif self.estado == "duelo" and ahora - self.t_estado >= SEGUNDOS_DUELO:
            self.ir("menu", ahora)

    def _problema_en_marca(self, puntos, ahora):
        """Que le falta a la persona para empezar, o "" si esta lista."""
        cuerpo = medir_cuerpo(puntos)
        if cuerpo is None:
            return "No te veo: párate en la marca, de frente"
        if not cuerpo.cabeza_visible:
            return "No veo tu cabeza: aléjate un paso"
        faltan_rodillas = self.reto == RETO_SENTADILLAS and not cuerpo.rodillas_visibles
        if faltan_rodillas and ahora - self.t_estado < ESPERA_RODILLAS:
            return "No veo tus rodillas: aléjate un paso"
        return ""

    def _actualizar_marca(self, puntos, ahora):
        if self._problema_en_marca(puntos, ahora):
            self.marca_desde = None
            return
        if self.marca_desde is None:
            self.marca_desde = ahora
        if ahora - self.marca_desde >= SEGUNDOS_MARCA:
            self.ir("cuenta", ahora)

    def _actualizar_cuenta(self, puntos, ahora):
        pasos = (("tres", 1.5), ("dos", 2.5), ("uno", 3.5))
        t = ahora - self.t_estado
        if self.cuenta_paso < len(pasos) and t >= pasos[self.cuenta_paso][1]:
            self.audio.efecto("tick")
            self.audio.decir(pasos[self.cuenta_paso][0], ahora_mismo=True)
            self.cuenta_paso += 1
        if t >= 4.5:
            self.empezar_juego(puntos, ahora)

    def _corregir(self, clave, ahora):
        if ahora - self.ultima_correccion >= ESPERA_CORRECCION:
            self.ultima_correccion = ahora
            self.audio.decir(clave)

    def _flash(self, texto, color, ahora):
        self.flash_texto, self.flash_color, self.flash_hasta = texto, color, ahora + 0.6

    def _actualizar_sentadillas(self, puntos, ahora):
        c = self.contador
        restante = SEGUNDOS_SENTADILLAS - (ahora - self.t_estado)
        for evento in c.actualizar(medir_cuerpo(puntos), ahora):
            if evento in ("perfecta", "media"):
                perfecta = evento == "perfecta"
                self.audio.efecto("ding" if perfecta else "media")
                self.audio.decir(reto_frases.clave_numero(min(c.total, reto_frases.MAX_NUMERO)))
                nota = f"{c.calidades[-1]:.0f}%"
                self._flash(f"¡PERFECTA! {nota}" if perfecta else f"A MEDIAS {nota}", VERDE if perfecta else AMARILLO, ahora)
                if c.total % 10 == 0:
                    self.audio.decir("bien")
            elif evento == "baja":
                self.audio.efecto("fallo")
                self._flash("¡BAJA MÁS!", ROJO, ahora)
                self._corregir("baja", ahora)
            elif evento in ("pecho", "rodillas", "rapido"):
                self._corregir(evento, ahora)

        if restante <= 10.0 and not self.aviso_10:
            self.aviso_10 = True
            self.audio.decir("animo", ahora_mismo=True)
        if restante <= 0:
            self.audio.decir("alto", ahora_mismo=True)
            self.terminar_juego(c.perfectas, c.total, ahora)

    def _actualizar_estatua(self, puntos, ahora):
        j = self.juez
        if j.fase == "preparar":
            segundo = math.ceil(j.restante(ahora))
            if segundo != self.ultimo_tick:
                self.ultimo_tick = segundo
                self.audio.efecto("tick")
        for evento in j.actualizar(puntos, ahora):
            if evento == "imita":
                self.audio.efecto("go")
            elif evento == "aguanta":
                self.audio.efecto("tick")
                self.audio.decir("aguanta", ahora_mismo=True)
            elif evento == "pose":
                self.audio.efecto("ding")
                self.audio.decir("pose_ok", ahora_mismo=True)
                self._flash("¡POSE LOGRADA!", VERDE, ahora)
            elif evento == "gana":
                self.audio.efecto("ding")
            else:  # "vida" o "caida"
                self.audio.efecto("fallo")
                self.audio.decir(evento, ahora_mismo=True)
                self._flash("¡FALLASTE!", ROJO, ahora)
        if j.terminado:
            self.terminar_juego(j.poses, round(j.precision, 1), ahora)

    def _actualizar_estrellas(self, puntos, ahora):
        e = self.estrellas
        restante = SEGUNDOS_ESTRELLAS - (ahora - self.t_estado)
        for evento in e.actualizar(puntos, ahora):
            if evento == "estrella":
                self.audio.efecto("ding")
                self._flash("+1", VERDE, ahora)
            elif evento == "dorada":
                self.audio.efecto("go")
                self.audio.decir("dorada")
                self._flash("+3", AMARILLO, ahora)
            elif evento == "bomba":
                self.audio.efecto("fallo")
                self.audio.decir("bomba", ahora_mismo=True)
                self._flash("¡BOMBA! -2", ROJO, ahora)
            elif evento == "racha":
                self.audio.decir("racha")
        if restante <= 10.0 and not self.aviso_10:
            self.aviso_10 = True
            self.audio.decir("animo", ahora_mismo=True)
        if restante <= 0:
            self.audio.decir("alto", ahora_mismo=True)
            self.terminar_juego(e.puntos, round(e.precision, 1), ahora)

    # --------------------------------------------------------------- dibujo
    def fuente(self, tam, negrita=True):
        clave = (tam, negrita)
        if clave not in self._fuentes:
            self._fuentes[clave] = pygame.font.SysFont("segoeui", tam, bold=negrita)
        return self._fuentes[clave]

    def texto(self, s, tam, color=BLANCO, pos=(0, 0), ancla="topleft", negrita=True):
        sup = self.fuente(tam, negrita).render(str(s), True, color)
        rect = sup.get_rect(**{ancla: pos})
        self.pantalla.blit(sup, rect)
        return rect

    def caja(self, rect, color, alfa=190, radio=14):
        sup = pygame.Surface(rect[2:], pygame.SRCALPHA)
        pygame.draw.rect(sup, (*color, alfa), sup.get_rect(), border_radius=radio)
        self.pantalla.blit(sup, rect[:2])

    def _a_pantalla(self, lm):
        return int(lm.x * VIDEO_W), int(lm.y * ALTO)

    def dibujar_video(self):
        rgb, puntos = self.camara.leer()
        self.pantalla.fill(FONDO)
        if rgb is None:
            mensaje = self.camara.error or self.camara.aviso
            self.texto(mensaje, 30, ROJO if self.camara.error else GRIS, (VIDEO_W // 2, 300), "center")
            return
        if not self.privacidad:
            alto, ancho = rgb.shape[:2]
            imagen = pygame.image.frombuffer(rgb.tobytes(), (ancho, alto), "RGB")
            imagen = pygame.transform.smoothscale(imagen, (VIDEO_W, ALTO))
            imagen.fill((135, 135, 150), special_flags=pygame.BLEND_RGB_MULT)
            self.pantalla.blit(imagen, (0, 0))
        if puntos:
            self.dibujar_esqueleto(puntos)

    def color_esqueleto(self):
        if self.estado == "juego" and self.reto == RETO_ESTATUA and self.juez:
            if self.juez.fase != "aguantar":
                return CIAN
            exceso = self.juez.movimiento / self.juez.nivel.quieto
            return VERDE if exceso < 0.6 else (AMARILLO if exceso < 1.0 else ROJO)
        if self.estado == "juego" and self.contador:
            c = self.contador
            return VERDE if c.profundidad >= c.nivel.perfecta else (AMARILLO if c.profundidad >= c.nivel.bajar else CIAN)
        return CIAN

    def dibujar_esqueleto(self, puntos):
        color = self.color_esqueleto()
        tenue = tuple(c // 3 for c in color)
        for a, b in CONEXIONES_DETALLE:
            if puntos[a].visibility >= 0.4 and puntos[b].visibility >= 0.4:
                pygame.draw.line(self.pantalla, color, self._a_pantalla(puntos[a]), self._a_pantalla(puntos[b]), 2)
        for a, b in CONEXIONES_CUERPO:
            if puntos[a].visibility < 0.4 or puntos[b].visibility < 0.4:
                continue
            pa, pb = self._a_pantalla(puntos[a]), self._a_pantalla(puntos[b])
            pygame.draw.line(self.pantalla, tenue, pa, pb, 14)
            pygame.draw.line(self.pantalla, color, pa, pb, 5)
        for i in range(core.NUM_LANDMARKS):
            if puntos[i].visibility >= 0.4:
                radio = 6 if i in PUNTOS_CUERPO else 3
                pygame.draw.circle(self.pantalla, BLANCO, self._a_pantalla(puntos[i]), radio)

        if self.estado != "juego":
            return
        if self.reto == RETO_ESTATUA and self.juez:
            for i in self.juez.nivel.puntos:
                if puntos[i].visibility >= 0.4:
                    pygame.draw.circle(self.pantalla, AMARILLO, self._a_pantalla(puntos[i]), 10, 2)
            self.dibujar_figura_guia(puntos)
        elif self.reto == RETO_ESTRELLAS:
            for i in (WRIST_L, WRIST_R):
                if puntos[i].visibility >= 0.4:
                    pygame.draw.circle(self.pantalla, CIAN, self._a_pantalla(puntos[i]), 22, 4)
        elif self.reto == RETO_SENTADILLAS:
            for cadera, rodilla, tobillo in ((core.HIP_L, KNEE_L, ANKLE_L), (core.HIP_R, KNEE_R, ANKLE_R)):
                if min(puntos[cadera].visibility, puntos[rodilla].visibility, puntos[tobillo].visibility) < 0.5:
                    continue
                grados = angulo(puntos[cadera], puntos[rodilla], puntos[tobillo])
                if grados is not None:
                    x, y = self._a_pantalla(puntos[rodilla])
                    self.texto(f"{grados:.0f}°", 26, BLANCO, (x + 14, y - 14))

    def dibujar_figura_guia(self, puntos):
        """La pose objetivo sobre el cuerpo de la persona: lo que falta en magenta y
        lo que ya esta bien en verde, para que se vea que corregir."""
        j = self.juez
        izquierda, derecha = sorted((core.SHOULDER_L, core.SHOULDER_R), key=lambda i: puntos[i].x)
        cadera_i, cadera_d = sorted((core.HIP_L, core.HIP_R), key=lambda i: puntos[i].x)
        hi, hd = self._a_pantalla(puntos[izquierda]), self._a_pantalla(puntos[derecha])
        ancho = max(40.0, math.hypot(hd[0] - hi[0], hd[1] - hi[1]))
        inicios = {
            "bi": hi, "bd": hd,
            "mi": self._a_pantalla(puntos[cadera_i]), "md": self._a_pantalla(puntos[cadera_d]),
        }
        for primero, segundo in (("bi", "ai"), ("bd", "ad"), ("mi", None), ("md", None)):
            if primero not in j.pose.segmentos:
                continue
            punta = inicios[primero]
            for segmento in filter(None, (primero, segundo)):
                grados = math.radians(j.pose.segmentos[segmento])
                largo = ancho * LARGO_SEGMENTO[segmento]
                fin = (punta[0] + largo * math.cos(grados), punta[1] + largo * math.sin(grados))
                color = VERDE if j.segmentos_bien.get(segmento) else MAGENTA
                pygame.draw.line(self.pantalla, color, punta, fin, 9)
                pygame.draw.circle(self.pantalla, color, fin, 8)
                punta = fin

    def pictograma(self, cx, cy, pose, color=BLANCO):
        """Figura pequena de la pose, para mostrarla en el panel."""
        escala = 46
        hombros = cy - 26
        cadera = cy + 44
        pygame.draw.circle(self.pantalla, color, (cx, hombros - 30), 17, 4)
        pygame.draw.line(self.pantalla, color, (cx, hombros - 12), (cx, cadera), 6)
        pygame.draw.line(self.pantalla, color, (cx - 26, hombros), (cx + 26, hombros), 6)
        for inicio, primero, segundo in (((cx - 26, hombros), "bi", "ai"), ((cx + 26, hombros), "bd", "ad")):
            punta = inicio
            for segmento in (primero, segundo):
                grados = math.radians(pose.segmentos[segmento])
                fin = (punta[0] + escala * 0.8 * math.cos(grados), punta[1] + escala * 0.8 * math.sin(grados))
                pygame.draw.line(self.pantalla, color, punta, fin, 6)
                punta = fin
        for segmento, x, normal in (("mi", cx - 10, 100), ("md", cx + 10, 80)):
            grados = math.radians(pose.segmentos.get(segmento, normal))
            fin = (x + escala * 1.5 * math.cos(grados), cadera + escala * 1.5 * math.sin(grados))
            pygame.draw.line(self.pantalla, color, (x, cadera), fin, 6)

    def dibujar_panel(self):
        pygame.draw.rect(self.pantalla, PANEL, (VIDEO_W, 0, ANCHO - VIDEO_W, ALTO))

    def dibujar_top(self, y):
        self.texto(f"TOP DE HOY", 30, AMARILLO, (VIDEO_W + 28, y))
        self.texto(NOMBRE_RETO[self.reto], 22, GRIS, (VIDEO_W + 28, y + 42), negrita=False)
        filas = self.ranking.top(self.reto, n=5)
        if not filas:
            self.texto("¡Sé el primero!", 26, BLANCO, (VIDEO_W + 28, y + 90))
        for i, fila in enumerate(filas):
            yy = y + 84 + i * 44
            self.texto(f"{i + 1}.", 26, AMARILLO, (VIDEO_W + 28, yy))
            self.texto(fila["apodo"], 26, BLANCO, (VIDEO_W + 66, yy), negrita=False)
            self.texto(self.puntaje_corto(fila["puntaje"]), 26, VERDE, (ANCHO - 24, yy), "topright")

    def puntaje_corto(self, puntaje):
        if self.reto == RETO_ESTATUA:
            return "1 pose" if puntaje == 1 else f"{puntaje:.0f} poses"
        if self.reto == RETO_ESTRELLAS:
            return f"{puntaje:.0f} pts"
        return f"{puntaje:.0f}"

    def dibujar_menu(self, ahora):
        self.dibujar_video()
        self.dibujar_panel()
        self.caja((0, 0, VIDEO_W, 112), FONDO, 215, 0)
        self.texto(TITULO, 58, BLANCO, (VIDEO_W // 2, 40), "center")
        self.texto("Tu entrenador de postura con inteligencia artificial", 24, CIAN, (VIDEO_W // 2, 88), "center", negrita=False)

        self.dibujar_top(36)
        self.texto(f"Hoy jugaron: {self.ranking.jugadas()}", 24, CIAN, (VIDEO_W + 28, 342))
        if self.aprendizaje:
            self.texto(f"La IA aprendió de {self.aprendizaje.partidas} partidas", 21, GRIS, (VIDEO_W + 28, 660), negrita=False)
        record = self.ranking.record(self.reto)
        if record is not None:
            self.texto("¿Le ganas al récord?", 26, BLANCO, (VIDEO_W + 28, 392))
            self.texto(self.puntaje_corto(record), 76, VERDE, (VIDEO_W + 28, 424))

        self.caja((24, 448, VIDEO_W - 48, 248), FONDO, 225)
        x, y = 48, 460
        self.fila_opciones(x, y, [
            (str(i + 1), NOMBRE_RETO[r], self.reto == r)
            for i, r in enumerate((RETO_SENTADILLAS, RETO_ESTATUA, RETO_ESTRELLAS))
        ])
        self.fila_opciones(x, y + 56, [
            (t, reto_frases.ENTRENADORES[n]["nombre"], self.entrenador == n)
            for t, n in (("S", "sargento"), ("A", "abuela"), ("Z", "zen"))
        ])
        self.fila_opciones(x, y + 112, [
            ("T", "Por turnos" if self.por_turnos else "Individual", self.por_turnos),
            ("D", self.nivel, False),
            ("P", "Privacidad", self.privacidad),
            ("C", f"Cámara {self.camara.indice}", False),
        ])
        latido = 200 + int(55 * math.sin(ahora * 4))
        self.texto("ENTER para jugar", 40, (latido, 255, latido), (VIDEO_W // 2, 650), "center")
        if ahora < self.esc_hasta:
            self.texto("ESC otra vez para salir", 24, ROJO, (VIDEO_W // 2, 420), "center")

    def fila_opciones(self, x, y, opciones):
        for tecla, nombre, activa in opciones:
            ancho = self.fuente(26).size(nombre)[0] + 72
            self.caja((x, y, ancho, 46), VERDE if activa else PANEL, 235, 10)
            color = FONDO if activa else BLANCO
            self.texto(tecla, 26, color, (x + 14, y + 23), "midleft")
            self.texto(nombre, 26, color, (x + 54, y + 23), "midleft")
            x += ancho + 14

    def dibujar_apodo(self):
        self.dibujar_video()
        self.caja((190, 210, 900, 300), FONDO, 240)
        numero = len(self.jugadores) + 1
        titulo = f"Jugador {numero}: escribe tu apodo" if self.por_turnos else "Escribe tu apodo"
        self.texto(titulo, 42, BLANCO, (640, 268), "center")
        self.texto((self.apodo or " ") + "_", 66, VERDE, (640, 360), "center")
        self.texto("ENTER para seguir  ·  ESC para volver", 26, GRIS, (640, 454), "center", negrita=False)

    def dibujar_marca(self, ahora):
        self.dibujar_video()
        self.dibujar_panel()
        self.dibujar_top(36)
        self.caja((70, 28, VIDEO_W - 140, 96), FONDO, 225)
        self.texto(f"{self.jugador['apodo']}: párate en la marca", 40, BLANCO, (VIDEO_W // 2, 76), "center")
        problema = self._problema_en_marca(self.ultimo_puntos, ahora)
        self.caja((70, 610, VIDEO_W - 140, 84), FONDO, 225)
        self.texto(problema or "¡Perfecto! No te muevas", 34, ROJO if problema else VERDE, (VIDEO_W // 2, 652), "center")
        if self.marca_desde is not None:
            avance = min(1.0, (ahora - self.marca_desde) / SEGUNDOS_MARCA)
            pygame.draw.rect(self.pantalla, VERDE, (70, 594, (VIDEO_W - 140) * avance, 8), border_radius=4)

    def dibujar_cuenta(self, ahora):
        self.dibujar_video()
        self.dibujar_panel()
        self.dibujar_top(36)
        t = ahora - self.t_estado
        numero = "3" if t < 2.5 else ("2" if t < 3.5 else "1")
        if t < 1.5:
            self.texto("¡Prepárate!", 96, AMARILLO, (VIDEO_W // 2, ALTO // 2), "center")
        else:
            escala = 1 + 0.5 * (1 - ((t - 1.5) % 1.0))
            self.texto(numero, int(190 * escala), BLANCO, (VIDEO_W // 2, ALTO // 2), "center")

    def dibujar_barra_profundidad(self):
        c = self.contador
        x, y, ancho, alto = 22, 110, 34, 500
        self.caja((x - 8, y - 8, ancho + 16, alto + 16), FONDO, 200, 10)
        fraccion = min(1.0, max(0.0, c.profundidad))
        lleno = alto * fraccion
        color = VERDE if fraccion >= c.nivel.perfecta else (AMARILLO if fraccion >= c.nivel.bajar else CIAN)
        pygame.draw.rect(self.pantalla, color, (x, y + alto - lleno, ancho, lleno), border_radius=6)
        linea = y + alto * (1 - c.nivel.perfecta)
        pygame.draw.line(self.pantalla, BLANCO, (x - 8, linea), (x + ancho + 8, linea), 3)
        porcentaje = c.porcentaje_actual
        color_texto = VERDE if porcentaje >= 100 else (AMARILLO if fraccion >= c.nivel.bajar else BLANCO)
        self.caja((12, 26, 150, 66), FONDO, 215, 12)
        self.texto(f"{porcentaje:.0f}%", 50, color_texto, (87, 59), "center")

    def corazon(self, cx, cy, r, color):
        pygame.draw.circle(self.pantalla, color, (cx - r // 2, cy), r // 2 + 2)
        pygame.draw.circle(self.pantalla, color, (cx + r // 2, cy), r // 2 + 2)
        pygame.draw.polygon(self.pantalla, color, ((cx - r - 1, cy + 3), (cx + r + 1, cy + 3), (cx, cy + r + 8)))

    def estrella(self, cx, cy, radio, color):
        puntas = []
        for i in range(10):
            r = radio if i % 2 == 0 else radio * 0.45
            a = math.radians(-90 + i * 36)
            puntas.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        pygame.draw.polygon(self.pantalla, color, puntas)

    def dibujar_objetivos(self, ahora):
        for o in self.estrellas.objetivos:
            cx, cy = o.x * VIDEO_W, o.y * ALTO
            radio = o.radio * ALTO * (1.0 + 0.08 * math.sin(ahora * 9))
            queda = max(0.0, (o.muere - ahora) / max(o.muere - o.nace, 1e-6))
            if o.tipo == "bomba":
                pygame.draw.circle(self.pantalla, (30, 10, 16), (cx, cy), radio)
                pygame.draw.circle(self.pantalla, ROJO, (cx, cy), radio, 5)
                self.texto("X", int(radio * 1.3), ROJO, (cx, cy), "center")
            else:
                color = AMARILLO if o.tipo == "dorada" else BLANCO
                self.estrella(cx, cy, radio * 1.25, color)
                if o.tipo == "dorada":
                    pygame.draw.circle(self.pantalla, AMARILLO, (cx, cy), radio * 1.5, 3)
            pygame.draw.arc(self.pantalla, CIAN, (cx - radio * 1.6, cy - radio * 1.6, radio * 3.2, radio * 3.2),
                            math.pi / 2, math.pi / 2 + 2 * math.pi * queda, 4)

    def dibujar_hud_estrellas(self, x, ahora):
        e = self.estrellas
        restante = max(0.0, SEGUNDOS_ESTRELLAS - (ahora - self.t_estado))
        self.dibujar_objetivos(ahora)
        self.texto(f"{math.ceil(restante)}", 74, ROJO if restante <= 10 else AMARILLO, (x, 104), "center")
        self.texto("segundos", 22, GRIS, (x, 150), "center", negrita=False)
        self.texto(e.puntos, 170, VERDE, (x, 290), "center")
        self.texto("puntos", 34, BLANCO, (x, 400), "center")
        self.texto(f"Racha: {e.racha}", 30, CIAN, (x, 454), "center")
        self.texto(f"Precisión: {e.precision:.0f}%", 24, GRIS, (x, 500), "center", negrita=False)

    def dibujar_juego(self, ahora):
        self.dibujar_video()
        self.dibujar_panel()
        x = VIDEO_W + (ANCHO - VIDEO_W) // 2
        self.texto(self.jugador["apodo"], 32, CIAN, (x, 34), "center")
        if self.reto == RETO_ESTRELLAS:
            self.dibujar_hud_estrellas(x, ahora)
        elif self.reto == RETO_SENTADILLAS:
            c = self.contador
            restante = max(0.0, SEGUNDOS_SENTADILLAS - (ahora - self.t_estado))
            self.dibujar_barra_profundidad()
            self.texto(f"{math.ceil(restante)}", 74, ROJO if restante <= 10 else AMARILLO, (x, 104), "center")
            self.texto("segundos", 22, GRIS, (x, 150), "center", negrita=False)
            self.texto(c.perfectas, 190, VERDE, (x, 300), "center")
            self.texto("perfectas", 34, BLANCO, (x, 420), "center")
            self.texto(f"Total: {c.total}", 28, GRIS, (x, 470), "center", negrita=False)
        else:
            self.dibujar_hud_estatua(x, ahora)
        rival = self.rival
        if rival:
            self.caja((VIDEO_W + 24, 560, ANCHO - VIDEO_W - 48, 120), FONDO, 255)
            self.texto(f"Rival: {rival['apodo']}", 26, GRIS, (x, 592), "center", negrita=False)
            self.texto(self.puntaje_corto(rival["puntaje"]), 54, MAGENTA, (x, 642), "center")
        if ahora < self.flash_hasta:
            self.caja((110, 8, VIDEO_W - 220, 92), FONDO, 240)
            self.texto(self.flash_texto, 62, self.flash_color, (VIDEO_W // 2, 54), "center")

    def dibujar_hud_estatua(self, x, ahora):
        j = self.juez
        for i in range(JuezEstatua.VIDAS):
            self.corazon(x - 70 + i * 70, 76, 17, ROJO if i < j.vidas else (60, 62, 90))
        self.texto(f"Poses: {j.poses}", 30, VERDE, (x, 124), "center")
        self.pictograma(x, 244, j.pose)
        self.texto(j.pose.nombre, 30, AMARILLO, (x, 394), "center")
        self.texto(f"{j.coincidencia:.0f}%", 54, VERDE if j.coincidencia >= 80 else BLANCO, (x, 446), "center")

        barra = (VIDEO_W + 40, 490, ANCHO - VIDEO_W - 80, 20)
        pygame.draw.rect(self.pantalla, (50, 54, 90), barra, border_radius=8)
        color = {"preparar": CIAN, "imitar": AMARILLO, "aguantar": VERDE}[j.fase]
        pygame.draw.rect(self.pantalla, color, (barra[0], barra[1], barra[2] * j.avance(ahora), barra[3]), border_radius=8)
        self.texto(f"Vigilando {j.puntos_vigilados} puntos", 20, GRIS, (x, 530), "center", negrita=False)

        mensaje, color = {
            "preparar": (f"PREPÁRATE: {math.ceil(j.restante(ahora))}", CIAN),
            "imitar": ("¡IMITA LA POSE!", MAGENTA),
            "aguantar": ("¡AGUANTA, NO TE MUEVAS!", VERDE),
        }[j.fase]
        self.caja((130, 12, VIDEO_W - 260, 84), FONDO, 220)
        self.texto(mensaje, 44, color, (VIDEO_W // 2, 54), "center")
        if j.fase == "aguantar":
            fraccion = min(1.0, j.movimiento / j.nivel.quieto)
            barra = (130, 668, VIDEO_W - 260, 26)
            self.caja((barra[0] - 8, barra[1] - 34, barra[2] + 16, 68), FONDO, 210)
            self.texto("movimiento", 20, GRIS, (VIDEO_W // 2, barra[1] - 18), "center", negrita=False)
            pygame.draw.rect(self.pantalla, (50, 54, 90), barra, border_radius=8)
            color = VERDE if fraccion < 0.6 else (AMARILLO if fraccion < 1 else ROJO)
            pygame.draw.rect(self.pantalla, color, (barra[0], barra[1], barra[2] * fraccion, barra[3]), border_radius=8)

    def dibujar_resumen_sentadillas(self, cx):
        """El porcentaje de bien hechas y, abajo, cuantas fueron de cada tipo."""
        c = self.contador
        self.texto(f"{c.porcentaje_bien:.0f}%", 112, VERDE, (cx, 272), "center")
        self.texto("de sentadillas bien hechas", 30, BLANCO, (cx, 348), "center")

        barra = (190, 378, VIDEO_W - 380, 24)
        pygame.draw.rect(self.pantalla, (50, 54, 90), barra, border_radius=8)
        x = barra[0]
        partes = ((c.perfectas, VERDE), (c.medias, AMARILLO), (c.fallidas, ROJO))
        for cantidad, color in partes:
            if c.intentos and cantidad:
                ancho = barra[2] * cantidad / c.intentos
                pygame.draw.rect(self.pantalla, color, (x, barra[1], ancho, barra[3]))
                x += ancho

        columnas = ((cx - 200, c.perfectas, "buenas", VERDE), (cx, c.medias, "a medias", AMARILLO), (cx + 200, c.fallidas, "malas", ROJO))
        for columna, cantidad, nombre, color in columnas:
            self.texto(cantidad, 56, color, (columna, 444), "center")
            self.texto(nombre, 26, BLANCO, (columna, 490), "center", negrita=False)

    def dibujar_resultado(self, ahora):
        self.dibujar_video()
        self.dibujar_panel()
        self.dibujar_top(36)
        r = self.resultado
        self.caja((110, 130, VIDEO_W - 220, 440), FONDO, 240)
        cx = VIDEO_W // 2
        self.texto("¡NUEVO RÉCORD!" if r["record"] else self.jugador["apodo"], 54, AMARILLO if r["record"] else BLANCO, (cx, 192), "center")
        if self.reto == RETO_SENTADILLAS:
            self.dibujar_resumen_sentadillas(cx)
        else:
            self.texto(self.puntaje_corto(r["puntaje"]), 118, VERDE, (cx, 302), "center")
            self.texto(f"Precisión: {r['extra']:.0f}%", 38, BLANCO, (cx, 420), "center")
        self.texto(f"Lugar {r['posicion']} de hoy", 34, CIAN, (cx, 538), "center")

    def lineas(self, texto, tam, ancho):
        """Parte un texto para que quepa en `ancho` pixeles."""
        fuente = self.fuente(tam, False)
        lineas, actual = [], ""
        for palabra in texto.split():
            prueba = f"{actual} {palabra}".strip()
            if fuente.size(prueba)[0] <= ancho:
                actual = prueba
            else:
                lineas.append(actual)
                actual = palabra
        return lineas + [actual]

    def dibujar_consejos(self):
        self.pantalla.fill(FONDO)
        consejos = self.resultado["consejos"]
        nombre = self.jugador["apodo"]
        self.texto(f"{nombre}: qué puedes mejorar" if consejos else f"¡Excelente técnica, {nombre}!", 52, AMARILLO, (640, 62), "center")
        self.texto(f"Tu entrenador: {self.nombre_entrenador}", 26, GRIS, (640, 112), "center", negrita=False)
        if not consejos:
            self.texto("No encontré nada importante que corregir. ¡Sigue así!", 38, VERDE, (640, 340), "center")
        y = 160
        for i, clave in enumerate(consejos, start=1):
            titulo, consejo = reto_frases.CONSEJOS[clave]
            self.caja((80, y, 1120, 152), PANEL, 255)
            self.texto(i, 68, CIAN, (136, y + 76), "center")
            self.texto(titulo, 36, VERDE, (196, y + 16))
            for n, linea in enumerate(self.lineas(consejo, 27, 960)):
                self.texto(linea, 27, BLANCO, (196, y + 66 + n * 36), negrita=False)
            y += 168
        self.texto("ENTER para seguir", 26, GRIS, (640, 692), "center", negrita=False)

    def dibujar_duelo(self):
        self.pantalla.fill(FONDO)
        ganador = self.resultado.get("ganador")
        titulo = "¡EMPATE!" if ganador is None else f"¡GANA {self.jugadores[ganador]['apodo'].upper()}!"
        self.texto(titulo, 82, AMARILLO, (640, 130), "center")
        for i, j in enumerate(self.jugadores):
            x = 340 + i * 600
            color = VERDE if ganador == i else (BLANCO if ganador is None else GRIS)
            self.texto(j["apodo"], 46, color, (x, 300), "center")
            self.texto(self.puntaje_corto(j["puntaje"]), 130, color, (x, 430), "center")
        self.texto("VS", 56, MAGENTA, (640, 410), "center")
        self.texto(f"Entrenador: {self.nombre_entrenador}", 30, GRIS, (640, 620), "center", negrita=False)

    def dibujar(self, ahora):
        dt = 0.0 if self._t_anterior is None else min(0.1, ahora - self._t_anterior)
        self._t_anterior = ahora
        {
            "menu": lambda: self.dibujar_menu(ahora),
            "apodo": self.dibujar_apodo,
            "marca": lambda: self.dibujar_marca(ahora),
            "cuenta": lambda: self.dibujar_cuenta(ahora),
            "juego": lambda: self.dibujar_juego(ahora),
            "resultado": lambda: self.dibujar_resultado(ahora),
            "consejos": self.dibujar_consejos,
            "duelo": self.dibujar_duelo,
        }[self.estado]()
        self.confeti.dibujar(self.pantalla, dt)


def main():
    parser = argparse.ArgumentParser(description="Reto de la feria.")
    parser.add_argument("--ventana", action="store_true", help="abrir en ventana, no a pantalla completa")
    parser.add_argument("--camara", type=int, default=None, help="usar esta camara, sin buscar la real")
    parser.add_argument("--borrar-ranking", action="store_true", help="dejar el ranking en cero y salir")
    parser.add_argument("--borrar-aprendizaje", action="store_true", help="olvidar lo aprendido de las partidas y salir")
    args = parser.parse_args()

    ranking = Ranking()
    if args.borrar_ranking:
        ranking.borrar()
        print("Ranking en cero.")
    aprendizaje = Aprendizaje()
    if args.borrar_aprendizaje:
        aprendizaje.borrar()
        print("Aprendizaje borrado.")
    if args.borrar_ranking or args.borrar_aprendizaje:
        return

    cfg = AppConfig.load()
    pygame.init()
    audio = Audio()
    banderas = pygame.SCALED if args.ventana else pygame.SCALED | pygame.FULLSCREEN
    pantalla = pygame.display.set_mode((ANCHO, ALTO), banderas)
    pygame.display.set_caption(TITULO)
    # Se guarda para que app_gui y collect_data tambien usen la camara real.
    camara = Camara(
        cfg.camera_index if args.camara is None else args.camara,
        cfg.flip_camera,
        buscar=args.camara is None,
        al_elegir=lambda indice: cfg.update({"camera_index": indice}),
    )
    camara.start()
    juego = Juego(pantalla, camara, audio, ranking, aprendizaje)

    reloj = pygame.time.Clock()
    while not juego.salir:
        ahora = time.monotonic()
        for evento in pygame.event.get():
            juego.manejar(evento, ahora)
        juego.actualizar(ahora)
        juego.dibujar(ahora)
        pygame.display.flip()
        reloj.tick(FPS)

    camara.detener()
    pygame.quit()


if __name__ == "__main__":
    main()
