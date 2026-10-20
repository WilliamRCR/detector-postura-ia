"""Metricas de postura normalizadas por el tamano del cuerpo (ancho de hombros /
largo de torso) en vez de coordenadas crudas, para que los umbrales no dependan
de que tan cerca esta la persona de la camara ni de su tipo de cuerpo.

Tambien incluye una deteccion best-effort de si la persona esta sentada o de
pie, y un contador simple de estadisticas de sesion.
"""
import math
import time
from dataclasses import dataclass


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _visible(lm, threshold=0.5):
    return getattr(lm, "visibility", 1.0) >= threshold


def _joint_angle(a, b, c):
    """Angulo en grados formado en el punto b por los segmentos b-a y b-c."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


@dataclass
class PostureThresholds:
    """Umbrales como proporcion del tamano del cuerpo, no en coordenadas crudas.
    Los valores por defecto se calibraron para aproximar el comportamiento de
    los umbrales fijos originales (0.05 / 0.15 / 0.08) asumiendo un encuadre
    tipico (ancho de hombros ~25% del frame), pero ahora escalan correctamente
    sin importar que tan cerca este la persona de la camara."""

    shoulder_tilt_max: float = 0.20     # hombros: inclinacion / ancho de hombros
    forward_head_max: float = 0.55      # cabeza adelantada / ancho de hombros
    neck_compression_min: float = 0.23  # oreja-hombro / largo de torso


DEFAULT_THRESHOLDS = PostureThresholds()


def compute_metrics(landmarks):
    """Devuelve las razones normalizadas, o None si no se puede medir con
    confianza (p. ej. la persona esta de lado y el ancho de hombros colapsa)."""
    nose = landmarks[0]
    ls, rs = landmarks[11], landmarks[12]
    le, re = landmarks[7], landmarks[8]
    lh, rh = landmarks[23], landmarks[24]

    shoulder_width = _dist(ls, rs)
    if shoulder_width < 0.03:
        return None

    mid_shoulder_y = (ls.y + rs.y) / 2
    mid_hip_y = (lh.y + rh.y) / 2
    torso_length = abs(mid_hip_y - mid_shoulder_y)
    if torso_length < 0.03:
        torso_length = shoulder_width * 1.4  # respaldo si no se ven bien las caderas

    return {
        "shoulder_tilt": abs(ls.y - rs.y) / shoulder_width,
        "forward_head": (ls.z - nose.z) / shoulder_width,
        "neck_gap": abs((le.y + re.y) / 2 - mid_shoulder_y) / torso_length,
    }


def evaluate_posture(landmarks, thresholds=None):
    """Devuelve un mensaje de feedback especifico, usando razones normalizadas
    por el tamano del cuerpo en vez de coordenadas crudas."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metrics = compute_metrics(landmarks)
    if metrics is None:
        return "Ubícate de frente a la cámara"

    if metrics["shoulder_tilt"] > thresholds.shoulder_tilt_max:
        return "Endereza los hombros"
    if metrics["forward_head"] > thresholds.forward_head_max:
        return "Retrasa un poco la cabeza"
    if metrics["neck_gap"] < thresholds.neck_compression_min:
        return "Levanta un poco la cabeza"
    return "Buena postura"


ASPECT_LABELS = {
    "hombros": "Hombros nivelados",
    "cabeza_adelantada": "Cabeza no adelantada",
    "altura_cabeza": "Buena altura de cabeza/cuello",
}


def evaluate_aspects(landmarks, thresholds=None):
    """Evalua las 3 dimensiones de postura de forma independiente (a diferencia
    de evaluate_posture, que corta en la primera que falla), para poder armar
    un listado de aspectos positivos/negativos en vez de un unico mensaje.
    Devuelve {clave: True/False} o None si no se puede medir con confianza."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metrics = compute_metrics(landmarks)
    if metrics is None:
        return None
    return {
        "hombros": metrics["shoulder_tilt"] <= thresholds.shoulder_tilt_max,
        "cabeza_adelantada": metrics["forward_head"] <= thresholds.forward_head_max,
        "altura_cabeza": metrics["neck_gap"] >= thresholds.neck_compression_min,
    }


def summarize_aspects(aspect_percents, good_threshold=70.0):
    """A partir de {clave: % de tiempo que estuvo bien ese aspecto} arma una
    lista de lineas tipo checklist, con los aspectos a mejorar primero."""
    items = []
    for key, label in ASPECT_LABELS.items():
        if key not in aspect_percents:
            continue
        pct = aspect_percents[key]
        ok = pct >= good_threshold
        mark = "✓" if ok else "✗"
        items.append((ok, f"{mark} {label} ({pct:.0f}%)"))
    items.sort(key=lambda item: item[0])  # False (a mejorar) primero
    return [text for _, text in items]


def detect_context(landmarks):
    """Intenta inferir si la persona esta sentada o de pie a partir del angulo
    cadera-rodilla-tobillo. Si las piernas no son visibles en el encuadre (muy
    comun cuando la camara solo capta torso/cabeza), devuelve 'no_determinado'
    en vez de adivinar."""
    sides = [(23, 25, 27), (24, 26, 28)]  # (cadera, rodilla, tobillo) izq/der
    angles = []
    for hip_i, knee_i, ankle_i in sides:
        hip, knee, ankle = landmarks[hip_i], landmarks[knee_i], landmarks[ankle_i]
        if _visible(hip) and _visible(knee) and _visible(ankle):
            angle = _joint_angle(hip, knee, ankle)
            if angle is not None:
                angles.append(angle)

    if not angles:
        return "no_determinado"
    return "de_pie" if (sum(angles) / len(angles)) > 160 else "sentado"


CONTEXT_LABELS = {
    "sentado": "Sentado",
    "de_pie": "De pie",
    "no_determinado": "Piernas no visibles",
}


class SessionStats:
    """Contador simple de % de tiempo en buena postura durante toda la sesion
    de uso (no solo la evaluacion inicial de 30s)."""

    def __init__(self):
        self._start_time = None
        self._correct = 0
        self._incorrect = 0

    def start(self):
        self._start_time = time.time()
        self._correct = 0
        self._incorrect = 0

    def record(self, is_correct):
        if is_correct is None:
            return
        if is_correct:
            self._correct += 1
        else:
            self._incorrect += 1

    @property
    def percent_correct(self):
        total = self._correct + self._incorrect
        return (self._correct / total * 100) if total else 0.0

    @property
    def elapsed_minutes(self):
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) / 60.0

    def summary_text(self):
        total = self._correct + self._incorrect
        if total == 0:
            return ""
        minutos = self.elapsed_minutes
        tiempo = f"{minutos:.0f} min" if minutos >= 1 else "<1 min"
        return f"Sesión: {self.percent_correct:.0f}% correcta ({tiempo})"
