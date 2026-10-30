"""Reglas de PosturIA: sentadillas, estatua, Caza Estrellas, consejos finales,
aprendizaje entre partidas y ranking. Aqui no hay camara, pantalla ni audio, para
poder probarlo sin nada de eso.

El juego no usa la red neuronal: mide posiciones del cuerpo, asi que funciona con
cualquier persona sin grabar datos antes.

Las sentadillas se miden por cuanto baja la cadera respecto a la persona de pie.
De frente el angulo de la rodilla casi no cambia, y con poco espacio los tobillos
pueden quedar fuera de cuadro; la cadera y las rodillas siempre se ven.
"""
import json
import math
import random
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import core

ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16
INDEX_L, INDEX_R = 19, 20
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28
FOOT_L, FOOT_R = 31, 32

RETO_SENTADILLAS = "sentadillas"
RETO_ESTATUA = "estatua"
RETO_ESTRELLAS = "estrellas"
RETOS = (RETO_SENTADILLAS, RETO_ESTATUA, RETO_ESTRELLAS)

SEGUNDOS_SENTADILLAS = 30.0
SEGUNDOS_ESTRELLAS = 30.0
ASPECTO = 640 / 480  # las coordenadas de MediaPipe no son cuadradas
MIN_VISIBILIDAD = 0.5


# Puntos del cuerpo que vigila la estatua. MediaPipe entrega 33: cada nivel usa
# mas, y moverse un poco ya se nota en alguno.
_PUNTOS_FACIL = (
    core.NOSE, core.SHOULDER_L, core.SHOULDER_R, core.HIP_L, core.HIP_R,
    ELBOW_L, ELBOW_R, WRIST_L, WRIST_R,
)
_PUNTOS_NORMAL = _PUNTOS_FACIL + (core.EAR_L, core.EAR_R, KNEE_L, KNEE_R)
_PUNTOS_DIFICIL = _PUNTOS_NORMAL + (INDEX_L, INDEX_R, ANKLE_L, ANKLE_R, FOOT_L, FOOT_R)


@dataclass(frozen=True)
class Nivel:
    bajar: float        # sentadilla: profundidad para contarla (1 = muslo horizontal)
    perfecta: float     # sentadilla: profundidad para que sea perfecta
    quieto: float       # estatua: movimiento tolerado, en anchos de hombros
    tolerancia: float   # estatua: grados de diferencia con la pose
    preparar: float     # estatua: segundos para ver la pose, sin evaluar
    limite: float       # estatua: segundos para llegar a la pose
    aguantar: float     # estatua: segundos que hay que sostener cada pose
    sostenido: float    # estatua: un movimiento de menos de esto se perdona
    puntos: tuple       # estatua: puntos del cuerpo vigilados
    grupos: tuple       # estatua: que poses salen en este nivel
    vida_estrella: float   # estrellas: segundos que dura cada una
    radio_estrella: float  # estrellas: tamano, en fracciones del alto de la imagen
    bombas: float          # estrellas: probabilidad de que salga una bomba


# Se elige en el menu: permite ajustar el juego en el stand sin tocar el codigo.
NIVELES = {
    "Fácil": Nivel(0.40, 0.55, 0.20, 35.0, 4.0, 5.0, 3.0, 0.40, _PUNTOS_FACIL, ("simple",), 2.6, 0.085, 0.12),
    "Normal": Nivel(0.50, 0.70, 0.14, 28.0, 3.0, 4.0, 4.0, 0.30, _PUNTOS_NORMAL, ("simple", "brazos"), 2.0, 0.07, 0.2),
    "Difícil": Nivel(0.60, 0.85, 0.10, 22.0, 2.5, 3.5, 5.0, 0.20, _PUNTOS_DIFICIL, ("brazos", "piernas"), 1.5, 0.055, 0.3),
}


def _visible(lm):
    return getattr(lm, "visibility", 1.0) >= MIN_VISIBILIDAD


def _medio(a, b):
    return (a + b) / 2.0


def angulo(a, b, c):
    """Angulo en grados en el punto b, o None si no se puede calcular."""
    return core._angle_between((a.x - b.x, a.y - b.y), (c.x - b.x, c.y - b.y))


@dataclass
class Cuerpo:
    """Lo que el juego necesita saber de la persona en un momento dado."""

    hombros_y: float
    cadera_y: float
    rodilla_y: float | None
    ancho_hombros: float
    inclinacion_hombros: float
    cabeza_visible: bool
    tronco_grados: float = 0.0           # inclinacion del tronco respecto a la vertical
    cuello_grados: float | None = None   # oreja-hombro-cadera: 180 es recto
    apertura_rodillas: float | None = None  # ancho entre rodillas / ancho entre tobillos

    @property
    def de_perfil(self):
        return self.ancho_hombros < self.torso * 0.5

    @property
    def torso(self):
        return max(self.cadera_y - self.hombros_y, 1e-6)

    @property
    def rodillas_visibles(self):
        return self.rodilla_y is not None


def medir_cuerpo(landmarks):
    """Devuelve un Cuerpo, o None si no se ven los hombros y la cadera."""
    if not landmarks:
        return None
    hl, hr = landmarks[core.SHOULDER_L], landmarks[core.SHOULDER_R]
    cl, cr = landmarks[core.HIP_L], landmarks[core.HIP_R]
    if not all(_visible(p) for p in (hl, hr, cl, cr)):
        return None

    rl, rr = landmarks[KNEE_L], landmarks[KNEE_R]
    dentro = _visible(rl) and _visible(rr) and max(rl.y, rr.y) < 1.0
    ancho = math.hypot(hl.x - hr.x, hl.y - hr.y)
    nariz = landmarks[core.NOSE]

    hombros_x = _medio(hl.x, hr.x) * ASPECTO
    cadera_x = _medio(cl.x, cr.x) * ASPECTO
    dy = _medio(cl.y, cr.y) - _medio(hl.y, hr.y)
    tronco = math.degrees(math.atan2(abs(hombros_x - cadera_x), max(abs(dy), 1e-6)))

    # La columna no se ve: la cadena oreja-hombro-cadera es la mejor senal que
    # hay de una espalda encorvada, y solo sirve de perfil.
    cuello = None
    oi, od = landmarks[core.EAR_L], landmarks[core.EAR_R]
    orejas = [o for o in (oi, od) if _visible(o)]
    if orejas:
        oreja_x = sum(o.x for o in orejas) / len(orejas) * ASPECTO
        oreja_y = sum(o.y for o in orejas) / len(orejas)
        h_y, c_y = _medio(hl.y, hr.y), _medio(cl.y, cr.y)
        cuello = core._angle_between((oreja_x - hombros_x, oreja_y - h_y), (cadera_x - hombros_x, c_y - h_y))

    apertura = None
    ti, td = landmarks[ANKLE_L], landmarks[ANKLE_R]
    if dentro and _visible(ti) and _visible(td):
        entre_tobillos = abs(ti.x - td.x)
        if entre_tobillos > 0.02:
            apertura = abs(rl.x - rr.x) / entre_tobillos

    return Cuerpo(
        hombros_y=_medio(hl.y, hr.y),
        cadera_y=_medio(cl.y, cr.y),
        rodilla_y=_medio(rl.y, rr.y) if dentro else None,
        ancho_hombros=ancho,
        inclinacion_hombros=abs(hl.y - hr.y) / max(ancho, 1e-6),
        cabeza_visible=_visible(nariz) and nariz.y > 0.0,
        tronco_grados=tronco,
        cuello_grados=cuello,
        apertura_rodillas=apertura,
    )


PENALIZA_INCLINACION = 25.0  # puntos que resta terminar muy inclinado


def _falla_una_vez(fallas, clave, evento):
    """Anota una falla de la repeticion; el evento sale solo la primera vez."""
    if clave in fallas:
        return []
    fallas.add(clave)
    return [evento]


class ContadorSentadillas:
    """Cuenta repeticiones: bajar la cadera hasta `nivel.bajar` y volver a subir.

    Eventos que devuelve `actualizar`:
    - "perfecta": repeticion completa y profunda
    - "media":    cuenta, pero no llego a lo mas profundo o se inclino mucho
    - "baja":     subio sin llegar a contar
    - "pecho":    va muy inclinado hacia adelante o encorvado
    - "rodillas": las rodillas se juntan al bajar
    - "rapido":   hizo la repeticion demasiado rapido

    `errores` acumula que fallo en cada repeticion para el resumen final.
    """

    SUBIDA = 0.20           # profundidad a la que ya se considera de pie
    INTENTO = 0.25          # desde aqui es un intento de sentadilla
    MIN_SEGUNDOS_REP = 0.5  # evita contar dos veces por temblor de la deteccion
    SUAVIZADO = 0.6
    TORSO_MINIMO = 0.5      # de frente, inclinarse acorta el torso en la imagen
    TRONCO_MAX = 55.0       # de perfil, un poco de inclinacion es normal al bajar
    CUELLO_MIN = 135.0      # de perfil, por debajo de esto la espalda va encorvada
    RODILLAS_MIN = 0.7      # rodillas mas juntas que esto respecto a los tobillos
    HOMBROS_MAX = 0.18
    SEGUNDOS_RAPIDA = 0.9

    def __init__(self, nivel):
        self.nivel = nivel
        self.total = 0
        self.perfectas = 0
        self.fallidas = 0
        self.calidades = []
        self.errores = Counter()
        self.picos = []
        self.profundidad = 0.0
        self._cadera0 = None
        self._muslo = 1.0
        self._torso0 = 1.0
        self._abajo = False
        self._pico = 0.0
        self._inclinado = False
        self._fallas_rep = set()
        self._inicio_rep = 0.0
        self._ultima_rep = -10.0

    @property
    def calibrado(self):
        return self._cadera0 is not None

    @property
    def porcentaje_actual(self):
        """Que tan cerca va de una sentadilla perfecta, de 0 a 100."""
        return self._porcentaje(self.profundidad)

    def _porcentaje(self, profundidad):
        return max(0.0, min(100.0, profundidad / self.nivel.perfecta * 100.0))

    @property
    def medias(self):
        return self.total - self.perfectas

    @property
    def intentos(self):
        return self.total + self.fallidas

    @property
    def porcentaje_bien(self):
        """De todos los intentos, cuantos fueron perfectos."""
        return self.perfectas / self.intentos * 100.0 if self.intentos else 0.0

    @property
    def calidad_media(self):
        return sum(self.calidades) / len(self.calidades) if self.calidades else 0.0

    def calibrar(self, cuerpo):
        """Toma como referencia a la persona de pie."""
        self._cadera0 = cuerpo.cadera_y
        self._torso0 = cuerpo.torso
        muslo = (cuerpo.rodilla_y - cuerpo.cadera_y) if cuerpo.rodillas_visibles else 0.0
        # Sin rodillas a la vista se estima: el muslo mide cerca de 0.85 torsos.
        self._muslo = muslo if muslo > 0.3 * cuerpo.torso else cuerpo.torso * 0.85
        self.profundidad = 0.0

    def _revisar_tecnica(self, cuerpo):
        """Que esta fallando en la parte baja de la sentadilla. Mide lo que cada
        vista permite: de perfil el tronco y la espalda, de frente rodillas y hombros."""
        eventos = []
        if cuerpo.de_perfil:
            if cuerpo.tronco_grados > self.TRONCO_MAX:
                self._inclinado = True
                eventos += _falla_una_vez(self._fallas_rep, "espalda", "pecho")
            if cuerpo.cuello_grados is not None and cuerpo.cuello_grados < self.CUELLO_MIN:
                self._inclinado = True
                eventos += _falla_una_vez(self._fallas_rep, "encorvada", "pecho")
            return eventos

        if cuerpo.torso / self._torso0 < self.TORSO_MINIMO:
            self._inclinado = True
            eventos += _falla_una_vez(self._fallas_rep, "espalda", "pecho")
        if cuerpo.apertura_rodillas is not None and cuerpo.apertura_rodillas < self.RODILLAS_MIN:
            eventos += _falla_una_vez(self._fallas_rep, "rodillas", "rodillas")
        if cuerpo.inclinacion_hombros > self.HOMBROS_MAX:
            self._fallas_rep.add("hombros")
        return eventos

    def actualizar(self, cuerpo, ahora):
        if cuerpo is None or not self.calibrado:
            return []

        cruda = (cuerpo.cadera_y - self._cadera0) / self._muslo
        self.profundidad += self.SUAVIZADO * (cruda - self.profundidad)
        p = self.profundidad
        eventos = []

        if not self._abajo:
            self._pico = max(self._pico, p)
            if p >= self.nivel.bajar:
                self._abajo = True
                self._inclinado = False
                self._fallas_rep = set()
                self._inicio_rep = ahora
            elif p <= self.SUBIDA:
                if self._pico >= self.INTENTO:
                    self.fallidas += 1
                    self.errores["profundidad"] += 1
                    self.calidades.append(self._porcentaje(self._pico))
                    eventos.append("baja")
                self._pico = 0.0
                # De pie: sigue los pequenos cambios de distancia a la camara.
                if p < 0.12:
                    self._cadera0 += 0.02 * (cuerpo.cadera_y - self._cadera0)
            return eventos

        self._pico = max(self._pico, p)
        eventos += self._revisar_tecnica(cuerpo)

        if p <= self.SUBIDA:
            self._abajo = False
            if ahora - self._ultima_rep >= self.MIN_SEGUNDOS_REP:
                self._ultima_rep = ahora
                self.total += 1
                self.picos.append(self._pico)
                calidad = self._porcentaje(self._pico) - (PENALIZA_INCLINACION if self._inclinado else 0.0)
                self.calidades.append(max(0.0, calidad))
                if self._pico >= self.nivel.perfecta and not self._inclinado:
                    self.perfectas += 1
                    eventos.append("perfecta")
                else:
                    eventos.append("media")
                if self._pico < self.nivel.perfecta:
                    self.errores["profundidad"] += 1
                self.errores.update(self._fallas_rep)
                if ahora - self._inicio_rep < self.SEGUNDOS_RAPIDA:
                    self.errores["ritmo"] += 1
                    eventos.append("rapido")
            self._pico = 0.0
        return eventos


# ------------------------------------------------------------------- poses
# Cada pose dice hacia donde apunta cada parte del cuerpo, en grados sobre la
# pantalla: 0 derecha, 90 abajo, 180 izquierda, 270 arriba. Comparar direcciones,
# y no posiciones, hace que sirva para cualquier estatura.
RANGO_ABAJO = 45.0     # grados alrededor de "hacia abajo" que tienen mas margen
MARGEN_ABAJO = 1.7
NOTA_NO_VISIBLE = 0.8


@dataclass(frozen=True)
class Pose:
    nombre: str
    grupo: str
    segmentos: dict  # bi/ai brazo y antebrazo izquierdos, bd/ad derechos, mi/md muslos


POSES = (
    # Brazos rectos: las puede hacer cualquiera en el primer intento.
    Pose("Cruz", "simple", {"bi": 180, "ai": 180, "bd": 0, "ad": 0}),
    Pose("Brazos arriba", "simple", {"bi": 240, "ai": 240, "bd": 300, "ad": 300}),
    Pose("Manos al cielo", "simple", {"bi": 270, "ai": 270, "bd": 270, "ad": 270}),
    Pose("Pingüino", "simple", {"bi": 130, "ai": 130, "bd": 50, "ad": 50}),
    Pose("Mano arriba", "simple", {"bi": 90, "ai": 90, "bd": 270, "ad": 270}),
    Pose("La otra mano", "simple", {"bi": 270, "ai": 270, "bd": 90, "ad": 90}),
    Pose("Las tres en punto", "simple", {"bi": 270, "ai": 270, "bd": 0, "ad": 0}),
    Pose("Las nueve en punto", "simple", {"bi": 180, "ai": 180, "bd": 270, "ad": 270}),
    Pose("Mano al lado", "simple", {"bi": 180, "ai": 180, "bd": 90, "ad": 90}),
    Pose("Al otro lado", "simple", {"bi": 90, "ai": 90, "bd": 0, "ad": 0}),
    Pose("Semáforo", "simple", {"bi": 225, "ai": 225, "bd": 45, "ad": 45}),
    Pose("Semáforo inverso", "simple", {"bi": 135, "ai": 135, "bd": 315, "ad": 315}),
    # Brazos doblados.
    Pose("Fortachón", "brazos", {"bi": 180, "ai": 270, "bd": 0, "ad": 270}),
    Pose("Manos a la cintura", "brazos", {"bi": 135, "ai": 45, "bd": 45, "ad": 135}),
    Pose("Una a la cintura", "brazos", {"bi": 135, "ai": 45, "bd": 270, "ad": 270}),
    Pose("La otra a la cintura", "brazos", {"bi": 270, "ai": 270, "bd": 45, "ad": 135}),
    Pose("Egipcio", "brazos", {"bi": 180, "ai": 270, "bd": 0, "ad": 90}),
    Pose("Egipcio al revés", "brazos", {"bi": 180, "ai": 90, "bd": 0, "ad": 270}),
    Pose("Rayo", "brazos", {"bi": 180, "ai": 270, "bd": 45, "ad": 90}),
    Pose("Manos a la nuca", "brazos", {"bi": 190, "ai": 320, "bd": 350, "ad": 220}),
    Pose("Aro", "brazos", {"bi": 225, "ai": 315, "bd": 315, "ad": 225}),
    Pose("Tetera", "brazos", {"bi": 135, "ai": 45, "bd": 0, "ad": 300}),
    Pose("Ola", "brazos", {"bi": 135, "ai": 225, "bd": 315, "ad": 45}),
    Pose("Saludo militar", "brazos", {"bi": 90, "ai": 90, "bd": 20, "ad": 230}),
    Pose("Al teléfono", "brazos", {"bi": 90, "ai": 90, "bd": 60, "ad": 290}),
    Pose("Hola", "brazos", {"bi": 90, "ai": 90, "bd": 0, "ad": 270}),
    Pose("Adiós", "brazos", {"bi": 180, "ai": 270, "bd": 90, "ad": 90}),
    # Con una pierna: solo salen si la camara ve las rodillas.
    Pose("Estrella", "piernas", {"bi": 215, "ai": 215, "bd": 325, "ad": 325, "mi": 110, "md": 70}),
    Pose("Árbol", "piernas", {"bi": 240, "ai": 240, "bd": 300, "ad": 300, "mi": 90, "md": 35}),
    Pose("El otro árbol", "piernas", {"bi": 240, "ai": 240, "bd": 300, "ad": 300, "mi": 145, "md": 90}),
    Pose("Avión", "piernas", {"bi": 180, "ai": 180, "bd": 0, "ad": 0, "mi": 90, "md": 45}),
    Pose("Avión al revés", "piernas", {"bi": 180, "ai": 180, "bd": 0, "ad": 0, "mi": 135, "md": 90}),
    Pose("Vaquero", "piernas", {"bi": 135, "ai": 45, "bd": 45, "ad": 135, "mi": 110, "md": 70}),
    Pose("Karateka", "piernas", {"bi": 135, "ai": 45, "bd": 0, "ad": 0, "mi": 90, "md": 50}),
    Pose("Bailarín", "piernas", {"bi": 225, "ai": 315, "bd": 315, "ad": 225, "mi": 90, "md": 50}),
    Pose("Superhéroe", "piernas", {"bi": 135, "ai": 45, "bd": 270, "ad": 270, "mi": 110, "md": 70}),
    Pose("Gol", "piernas", {"bi": 240, "ai": 240, "bd": 300, "ad": 300, "mi": 110, "md": 70}),
)


def diferencia_angulos(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _direccion(a, b):
    if not (_visible(a) and _visible(b)):
        return None
    dx, dy = (b.x - a.x) * ASPECTO, b.y - a.y
    if math.hypot(dx, dy) < 1e-4:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _lados(landmarks, izquierdo, derecho):
    """(el de la izquierda de la pantalla, el de la derecha). MediaPipe nombra los
    lados del cuerpo; a quien juega le importa de que lado lo ve en la pantalla."""
    if landmarks[izquierdo].x <= landmarks[derecho].x:
        return izquierdo, derecho
    return derecho, izquierdo


def medir_segmentos(landmarks):
    """Hacia donde apunta cada parte del cuerpo: {segmento: grados o None}."""
    hi, hd = _lados(landmarks, core.SHOULDER_L, core.SHOULDER_R)
    ci, cd = _lados(landmarks, core.HIP_L, core.HIP_R)
    # Cada hombro sigue con su propio codo y su muneca.
    brazos = {core.SHOULDER_L: (ELBOW_L, WRIST_L), core.SHOULDER_R: (ELBOW_R, WRIST_R)}
    pierna = {core.HIP_L: KNEE_L, core.HIP_R: KNEE_R}
    (ei, mi), (ed, md) = brazos[hi], brazos[hd]
    return {
        "bi": _direccion(landmarks[hi], landmarks[ei]),
        "ai": _direccion(landmarks[ei], landmarks[mi]),
        "bd": _direccion(landmarks[hd], landmarks[ed]),
        "ad": _direccion(landmarks[ed], landmarks[md]),
        "mi": _direccion(landmarks[ci], landmarks[pierna[ci]]),
        "md": _direccion(landmarks[cd], landmarks[pierna[cd]]),
    }


def _apunta_abajo(grados):
    return diferencia_angulos(grados, 90.0) <= RANGO_ABAJO


def evaluar_pose(landmarks, pose, tolerancia):
    """Devuelve (cumple, porcentaje, {segmento: bien}).

    Un brazo pegado al cuerpo se mide mal: la muneca queda junto a la cadera, se
    tapa o sale torcida, y se veia como un brazo doblado. Lo que debe ir hacia
    abajo tiene mas margen, y si no se ve se da por bien.
    """
    medidos = medir_segmentos(landmarks)
    bien = {}
    notas = []
    for segmento, objetivo in pose.segmentos.items():
        grados = medidos[segmento]
        abajo = _apunta_abajo(objetivo)
        if grados is None:
            bien[segmento] = abajo
            notas.append(NOTA_NO_VISIBLE if abajo else 0.0)
            continue
        margen = tolerancia * (MARGEN_ABAJO if abajo else 1.0)
        error = diferencia_angulos(grados, objetivo)
        bien[segmento] = error <= margen
        notas.append(max(0.0, 1.0 - error / (margen * 3.0)))
    return all(bien.values()), sum(notas) / len(notas) * 100.0, bien


class JuezEstatua:
    """Una pose tras otra: imitarla a tiempo y sostenerla sin moverse.

    Cada pose empieza con una preparacion en la que se muestra y no se evalua: sin
    eso, la persona apenas se estaba acomodando y ya le contaba como error.

    Eventos: "imita" (termino la preparacion), "aguanta" (ya la imito), "pose" (la
    completo), "vida" (fallo y pierde una vida), "caida" (se acabaron las vidas) y
    "gana" (las completo todas).
    """

    VIDAS = 3
    MAX_POSES = 10
    ENTRADA = 0.3         # segundos en la pose para aceptarla: evita pasar de largo
    PERDIDO = 1.0         # segundos sin verlo antes de contar como movimiento
    MARGEN_POSE = 1.35    # al sostener se tolera un poco mas que al entrar
    SUAVIZADO = 0.4
    SUBE_AGUANTE = 0.2    # cada pose completada exige mas...
    BAJA_LIMITE = 0.2     # ...y da menos tiempo
    LIMITE_MINIMO = 2.5

    def __init__(self, nivel, piernas=True, rng=None, aprendizaje=None):
        self.nivel = nivel
        self.rng = rng or random.Random()
        self.aprendizaje = aprendizaje
        self.errores = Counter()
        self.tolerancia = nivel.tolerancia
        self.opciones = [
            p for p in POSES if p.grupo in nivel.grupos and (piernas or p.grupo != "piernas")
        ]
        self.vidas = self.VIDAS
        self.poses = 0
        self.terminado = False
        self.pose = None
        self.fase = "preparar"
        self.coincidencia = 0.0
        self.segmentos_bien = {}
        self.movimiento = 0.0
        self.precisiones = []
        self._inicio_fase = 0.0
        self._fin_fase = 0.0
        self._cumple_desde = None
        self._referencia = {}
        self._actual = {}
        self._fuera_desde = None
        self._visto = 0.0

    @property
    def precision(self):
        """Que tan bien imito las poses que completo, de 0 a 100."""
        return sum(self.precisiones) / len(self.precisiones) if self.precisiones else 0.0

    @property
    def puntos_vigilados(self):
        return len(self.nivel.puntos)

    def avance(self, ahora):
        """De 1 a 0: cuanto queda de la fase actual."""
        total = max(self._fin_fase - self._inicio_fase, 1e-6)
        return max(0.0, min(1.0, (self._fin_fase - ahora) / total))

    def restante(self, ahora):
        return max(0.0, self._fin_fase - ahora)

    def iniciar(self, ahora):
        self._nueva_pose(ahora)

    def _nueva_pose(self, ahora):
        candidatas = [p for p in self.opciones if p is not self.pose]
        if self.aprendizaje:
            pesos = [self.aprendizaje.peso(p.nombre) for p in candidatas]
            self.pose = self.rng.choices(candidatas, weights=pesos)[0]
            self.tolerancia = self.nivel.tolerancia + self.aprendizaje.ajuste_tolerancia(self.pose.nombre)
        else:
            self.pose = self.rng.choice(candidatas)
        self.fase = "preparar"
        self.coincidencia = 0.0
        self.segmentos_bien = {}
        self.movimiento = 0.0
        self._cumple_desde = None
        self._fuera_desde = None
        self._visto = ahora
        self._inicio_fase = ahora
        self._fin_fase = ahora + self.nivel.preparar

    def _preparar(self, landmarks, ahora):
        if landmarks:
            _, self.coincidencia, self.segmentos_bien = evaluar_pose(landmarks, self.pose, self.tolerancia)
        if ahora < self._fin_fase:
            return []
        self.fase = "imitar"
        self._inicio_fase = ahora
        limite = self.nivel.limite - self.BAJA_LIMITE * self.poses
        self._fin_fase = ahora + max(self.LIMITE_MINIMO, limite)
        return ["imita"]

    def _fijar_referencia(self, landmarks):
        self._referencia = {
            i: (landmarks[i].x, landmarks[i].y) for i in self.nivel.puntos if _visible(landmarks[i])
        }
        self._actual = dict(self._referencia)

    def _fallo(self, ahora, causa):
        self.errores[causa] += 1
        for segmento, bien in self.segmentos_bien.items():
            if not bien:
                self.errores["piernas" if segmento in ("mi", "md") else "brazos"] += 1
        if self.aprendizaje:
            self.aprendizaje.registrar_pose(self.pose.nombre, False)
        self.vidas -= 1
        if self.vidas <= 0:
            self.terminado = True
            return ["caida"]
        self._nueva_pose(ahora)
        return ["vida"]

    def _medir_movimiento(self, landmarks, cuerpo):
        escala = max(cuerpo.ancho_hombros, cuerpo.torso * 0.5)
        distancias = []
        for i, (rx, ry) in self._referencia.items():
            if not _visible(landmarks[i]):
                continue
            ax, ay = self._actual[i]
            ax += self.SUAVIZADO * (landmarks[i].x - ax)
            ay += self.SUAVIZADO * (landmarks[i].y - ay)
            self._actual[i] = (ax, ay)
            distancias.append(math.hypot(ax - rx, ay - ry) / escala)
        # El segundo mas alto: atrapa cualquier movimiento real, que siempre
        # mueve mas de un punto, y perdona un punto con mala deteccion.
        distancias.sort(reverse=True)
        return distancias[1] if len(distancias) > 1 else 0.0

    def actualizar(self, landmarks, ahora):
        if self.terminado or self.pose is None:
            return []
        if self.fase == "preparar":
            return self._preparar(landmarks, ahora)
        cuerpo = medir_cuerpo(landmarks)
        if cuerpo is None:
            if self.fase == "aguantar" and ahora - self._visto >= self.PERDIDO:
                return self._fallo(ahora, "quieto")
            return self._fallo(ahora, "tiempo") if self.fase == "imitar" and ahora >= self._fin_fase else []
        self._visto = ahora
        if self.fase == "imitar":
            return self._imitar(landmarks, ahora)
        return self._aguantar(landmarks, cuerpo, ahora)

    def _imitar(self, landmarks, ahora):
        cumple, self.coincidencia, self.segmentos_bien = evaluar_pose(landmarks, self.pose, self.tolerancia)
        if cumple:
            if self._cumple_desde is None:
                self._cumple_desde = ahora
            if ahora - self._cumple_desde >= self.ENTRADA:
                self.fase = "aguantar"
                self._inicio_fase = ahora
                self._fin_fase = ahora + self.nivel.aguantar + self.SUBE_AGUANTE * self.poses
                self._fijar_referencia(landmarks)
                return ["aguanta"]
            return []
        self._cumple_desde = None
        return self._fallo(ahora, "tiempo") if ahora >= self._fin_fase else []

    def _aguantar(self, landmarks, cuerpo, ahora):
        tolerancia = self.tolerancia * self.MARGEN_POSE
        cumple, self.coincidencia, self.segmentos_bien = evaluar_pose(landmarks, self.pose, tolerancia)
        self.movimiento = self._medir_movimiento(landmarks, cuerpo)
        if cumple and self.movimiento <= self.nivel.quieto:
            self._fuera_desde = None
        else:
            if self._fuera_desde is None:
                self._fuera_desde = ahora
            if ahora - self._fuera_desde >= self.nivel.sostenido:
                return self._fallo(ahora, "quieto" if cumple else "brazos")

        if ahora >= self._fin_fase:
            self.poses += 1
            self.precisiones.append(self.coincidencia)
            if self.aprendizaje:
                self.aprendizaje.registrar_pose(self.pose.nombre, True)
                self.aprendizaje.registrar_movimiento(self.movimiento)
            if self.poses >= self.MAX_POSES:
                self.terminado = True
                return ["gana"]
            self._nueva_pose(ahora)
            return ["pose"]
        return []


# --------------------------------------------------------------- Caza Estrellas
@dataclass
class Objetivo:
    x: float
    y: float
    tipo: str      # "estrella", "dorada" o "bomba"
    nace: float
    muere: float
    radio: float


class JuegoEstrellas:
    """Tocar estrellas con las manos y esquivar bombas.

    Las estrellas salen alrededor de la persona, a distancia de brazo, asi que
    funciona con cualquier estatura y sin que se vean las piernas.

    Eventos: "estrella", "dorada", "bomba" y "perdida". "racha" sale cada 5 seguidas.
    """

    SIMULTANEOS = 2
    PROB_DORADA = 0.08
    PUNTOS = {"estrella": 1, "dorada": 3, "bomba": -2}
    ESPERA = 0.35          # entre una estrella y la siguiente
    RACHA = 5
    LENTO = 1.4            # segundos promedio para tocar una estrella

    def __init__(self, nivel, rng=None):
        self.nivel = nivel
        self.rng = rng or random.Random()
        self.puntos = 0
        self.racha = 0
        self.mejor_racha = 0
        self.objetivos = []
        self.aciertos = 0
        self.perdidas = 0
        self.bombas = 0
        self.reacciones = []
        self.por_mano = {"izquierda": 0, "derecha": 0}
        self._proxima = 0.0

    @property
    def precision(self):
        intentos = self.aciertos + self.perdidas + self.bombas
        return self.aciertos / intentos * 100.0 if intentos else 0.0

    def _crear(self, cuerpo, landmarks, ahora):
        centro = _medio(landmarks[core.SHOULDER_L].x, landmarks[core.SHOULDER_R].x)
        alcance = max(cuerpo.ancho_hombros, cuerpo.torso * 0.6)
        lado = self.rng.choice((-1, 1))
        x = centro + lado * self.rng.uniform(0.7, 1.9) * alcance / ASPECTO
        y = cuerpo.hombros_y + self.rng.uniform(-0.8, 0.9) * cuerpo.torso
        r = self.rng.random()
        tipo = "bomba" if r < self.nivel.bombas else ("dorada" if r < self.nivel.bombas + self.PROB_DORADA else "estrella")
        return Objetivo(
            x=max(0.08, min(0.92, x)), y=max(0.1, min(0.9, y)), tipo=tipo, nace=ahora,
            muere=ahora + self.nivel.vida_estrella, radio=self.nivel.radio_estrella,
        )

    def _mano_que_toca(self, objetivo, landmarks):
        for indices, mano in (((WRIST_L, INDEX_L), "izquierda"), ((WRIST_R, INDEX_R), "derecha")):
            for i in indices:
                lm = landmarks[i]
                if _visible(lm) and math.hypot((lm.x - objetivo.x) * ASPECTO, lm.y - objetivo.y) <= objetivo.radio:
                    return mano
        return None

    def _tocar(self, objetivo, mano, ahora):
        self.puntos = max(0, self.puntos + self.PUNTOS[objetivo.tipo])
        if objetivo.tipo == "bomba":
            self.bombas += 1
            self.racha = 0
            return [objetivo.tipo]
        self.aciertos += 1
        self.racha += 1
        self.mejor_racha = max(self.mejor_racha, self.racha)
        self.por_mano[mano] += 1
        self.reacciones.append(ahora - objetivo.nace)
        return [objetivo.tipo, "racha"] if self.racha % self.RACHA == 0 else [objetivo.tipo]

    def actualizar(self, landmarks, ahora):
        eventos = []
        cuerpo = medir_cuerpo(landmarks)
        for objetivo in list(self.objetivos):
            mano = self._mano_que_toca(objetivo, landmarks) if landmarks else None
            if mano:
                self.objetivos.remove(objetivo)
                eventos += self._tocar(objetivo, mano, ahora)
            elif ahora >= objetivo.muere:
                self.objetivos.remove(objetivo)
                if objetivo.tipo != "bomba":
                    self.perdidas += 1
                    self.racha = 0
                    eventos.append("perdida")
        if cuerpo and len(self.objetivos) < self.SIMULTANEOS and ahora >= self._proxima:
            self.objetivos.append(self._crear(cuerpo, landmarks, ahora))
            self._proxima = ahora + self.ESPERA
        return eventos

    @property
    def errores(self):
        errores = Counter()
        intentos = max(1, self.aciertos + self.perdidas + self.bombas)
        menos, mas = sorted(self.por_mano.values())
        if self.aciertos >= 6 and menos < mas * 0.35:
            errores["un_lado"] = intentos
        if self.reacciones and sum(self.reacciones) / len(self.reacciones) > self.LENTO:
            errores["reaccion"] = intentos
        errores["bombas"] = self.bombas
        return errores


# -------------------------------------------------------------- consejos finales
MIN_PROPORCION_CONSEJO = 0.25
MAX_CONSEJOS = 3


def elegir_consejos(errores, intentos):
    """Los problemas que mas se repitieron, de mayor a menor. Uno que paso una
    sola vez entre muchas repeticiones no merece un consejo."""
    base = max(intentos, 1)
    frecuentes = [(n, clave) for clave, n in errores.items() if n and n / base >= MIN_PROPORCION_CONSEJO]
    frecuentes.sort(key=lambda par: -par[0])
    return [clave for _, clave in frecuentes[:MAX_CONSEJOS]]


# ------------------------------------------------------------------ aprendizaje
APRENDIZAJE_FILE = "reto_aprendizaje.json"


class Aprendizaje:
    """Lo que el juego aprende de las partidas, de la gente y de esa camara.

    No reentrena una red neuronal: guarda estadisticas y ajusta los umbrales dentro
    de limites fijos, para que nada pueda desconfigurar el juego:

    - una pose que casi nadie logra da mas margen; una que todos logran, menos;
    - las poses con pocos datos salen mas, para poder aprender de ellas;
    - la profundidad de una sentadilla perfecta se adapta a lo que mide esa camara;
    - el movimiento tolerado se adapta al temblor real de la deteccion.
    """

    MIN_INTENTOS = 5
    MIN_MUESTRAS = 20
    MAX_MUESTRAS = 200
    AJUSTE_MIN, AJUSTE_MAX = -4.0, 8.0     # grados de tolerancia
    TASA_OBJETIVO = 0.6
    MAX_CAMBIO_PERFECTA = 0.10
    QUIETO_MIN, QUIETO_MAX = 0.85, 1.3     # veces el valor del nivel

    def __init__(self, path=APRENDIZAJE_FILE):
        self.path = Path(path)
        self.datos = {"partidas": 0, "poses": {}, "profundidad": [], "movimiento": []}
        try:
            guardado = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(guardado, dict):
                self.datos.update({k: guardado[k] for k in self.datos if k in guardado})
        except (OSError, ValueError):
            pass  # sin archivo, o danado: se empieza de cero

    @property
    def partidas(self):
        return self.datos["partidas"]

    def _pose(self, nombre):
        return self.datos["poses"].setdefault(nombre, {"intentos": 0, "logros": 0})

    def registrar_pose(self, nombre, logro):
        pose = self._pose(nombre)
        pose["intentos"] += 1
        pose["logros"] += 1 if logro else 0

    def ajuste_tolerancia(self, nombre):
        pose = self.datos["poses"].get(nombre)
        if not pose or pose["intentos"] < self.MIN_INTENTOS:
            return 0.0
        tasa = pose["logros"] / pose["intentos"]
        return max(self.AJUSTE_MIN, min(self.AJUSTE_MAX, (self.TASA_OBJETIVO - tasa) * 20.0))

    def peso(self, nombre):
        pose = self.datos["poses"].get(nombre)
        if not pose or pose["intentos"] < self.MIN_INTENTOS:
            return 3.0
        return 0.5 + pose["logros"] / pose["intentos"]

    def _anotar(self, lista, valor):
        muestras = self.datos[lista]
        muestras.append(round(float(valor), 4))
        del muestras[: -self.MAX_MUESTRAS]

    def registrar_profundidad(self, pico):
        self._anotar("profundidad", pico)

    def registrar_movimiento(self, movimiento):
        self._anotar("movimiento", movimiento)

    def nivel_ajustado(self, nivel):
        """El nivel con los umbrales que esta camara ha demostrado necesitar."""
        perfecta = nivel.perfecta
        muestras = sorted(self.datos["profundidad"])
        if len(muestras) >= self.MIN_MUESTRAS:
            mediana = muestras[len(muestras) // 2]
            perfecta = max(nivel.perfecta - self.MAX_CAMBIO_PERFECTA, min(nivel.perfecta, mediana))
            perfecta = max(perfecta, nivel.bajar + 0.05)

        quieto = nivel.quieto
        muestras = sorted(self.datos["movimiento"])
        if len(muestras) >= self.MIN_MUESTRAS:
            p90 = muestras[int(len(muestras) * 0.9)]
            quieto = max(nivel.quieto * self.QUIETO_MIN, min(nivel.quieto * self.QUIETO_MAX, p90 * 1.3))
        return replace(nivel, perfecta=perfecta, quieto=quieto)

    def terminar_partida(self):
        self.datos["partidas"] += 1
        self.guardar()

    def guardar(self):
        try:
            self.path.write_text(json.dumps(self.datos, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            print(f"No pude guardar lo aprendido ({exc}).")

    def borrar(self):
        self.datos = {"partidas": 0, "poses": {}, "profundidad": [], "movimiento": []}
        self.path.unlink(missing_ok=True)


RANKING_FILE = "reto_ranking.db"

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS puntajes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha    TEXT NOT NULL,
    momento  TEXT NOT NULL,
    reto     TEXT NOT NULL,
    apodo    TEXT NOT NULL,
    puntaje  REAL NOT NULL,
    extra    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_puntajes ON puntajes (fecha, reto, puntaje);
"""


class Ranking:
    """Top del dia en SQLite. Solo guarda apodo y puntaje: nunca imagenes."""

    def __init__(self, path=RANKING_FILE):
        self.path = str(path)
        with self._conectar() as conn:
            conn.executescript(_ESQUEMA)

    @contextmanager
    def _conectar(self):
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def guardar(self, reto, apodo, puntaje, extra=0.0, dia=None):
        """Guarda el puntaje y devuelve su posicion en el ranking del dia."""
        dia = dia or date.today()
        with self._conectar() as conn:
            conn.execute(
                "INSERT INTO puntajes (fecha, momento, reto, apodo, puntaje, extra) VALUES (?, ?, ?, ?, ?, ?)",
                (dia.isoformat(), datetime.now().isoformat(timespec="seconds"), reto, apodo, float(puntaje), float(extra)),
            )
            fila = conn.execute(
                "SELECT COUNT(*) AS n FROM puntajes WHERE fecha = ? AND reto = ?"
                " AND (puntaje > ? OR (puntaje = ? AND extra > ?))",
                (dia.isoformat(), reto, float(puntaje), float(puntaje), float(extra)),
            ).fetchone()
        return fila["n"] + 1

    def top(self, reto, n=5, dia=None):
        dia = dia or date.today()
        with self._conectar() as conn:
            filas = conn.execute(
                "SELECT apodo, puntaje, extra FROM puntajes WHERE fecha = ? AND reto = ?"
                " ORDER BY puntaje DESC, extra DESC, id ASC LIMIT ?",
                (dia.isoformat(), reto, n),
            ).fetchall()
        return [dict(f) for f in filas]

    def record(self, reto, dia=None):
        """El mejor puntaje del dia, o None si aun no hay."""
        mejores = self.top(reto, n=1, dia=dia)
        return mejores[0]["puntaje"] if mejores else None

    def jugadas(self, dia=None):
        dia = dia or date.today()
        with self._conectar() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM puntajes WHERE fecha = ?", (dia.isoformat(),)).fetchone()["n"]

    def borrar(self):
        with self._conectar() as conn:
            conn.execute("DELETE FROM puntajes")
