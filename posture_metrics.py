"""Metricas de postura normalizadas por el tamano del cuerpo (ancho de hombros /
largo de torso) en vez de coordenadas crudas, para que los umbrales no dependan
de que tan cerca esta la persona de la camara ni de su tipo de cuerpo.

Tambien incluye calibracion personal de umbrales, una deteccion best-effort de
si la persona esta sentada o de pie, y un contador de estadisticas de sesion.

La geometria vive en core.py: aqui solo quedan los umbrales y la lectura.
"""
import math
import time
from dataclasses import dataclass, replace

import core


def _visible(lm, threshold=0.5):
    return getattr(lm, "visibility", 1.0) >= threshold


def _joint_angle(a, b, c):
    """Angulo en grados formado en el punto b por los segmentos b-a y b-c."""
    return core._angle_between((a.x - b.x, a.y - b.y), (c.x - b.x, c.y - b.y))


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
    return core.metrics_from_keypoints(core.keypoints_from_landmarks(landmarks))


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


# ------------------------------------------------------- calibracion personal
# Cuanto se puede alejar un umbral calibrado del valor por defecto. Sin estos
# topes, una persona que se evalua encorvada terminaria con umbrales que
# aprueban su mala postura para siempre.
CALIBRATION_BOUNDS = {
    "shoulder_tilt_max": (0.7, 1.6),
    "forward_head_max": (0.7, 1.6),
    "neck_compression_min": (0.6, 1.3),
}

# Minimos para que valga la pena calibrar: pocas muestras o una postura de
# referencia mala no sirven como referencia.
CALIBRATION_MIN_SAMPLES = 20
CALIBRATION_MIN_PERCENT = 60.0


def _percentile(values, pct):
    """Percentil sin numpy (interpolacion lineal), para poder testear este
    modulo sin dependencias pesadas."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


class PostureCalibrator:
    """Acumula las metricas de la evaluacion inicial para derivar umbrales
    propios de esa persona y esa camara.

    Los umbrales por defecto son iguales para todos, pero cada cuerpo y cada
    encuadre son distintos: lo que para una persona es 'hombros nivelados'
    puede medir distinto en otra.
    """

    def __init__(self):
        self._samples = {"shoulder_tilt": [], "forward_head": [], "neck_gap": []}

    def record(self, metrics):
        if not metrics:
            return
        for key, bucket in self._samples.items():
            value = metrics.get(key)
            if value is not None and math.isfinite(value):
                bucket.append(float(value))

    @property
    def samples(self):
        return len(self._samples["shoulder_tilt"])

    def suggest(self, base=None, percent_correct=100.0):
        """Devuelve (umbrales, motivo). Si no hay datos suficientes o la postura
        de referencia era mala, devuelve los umbrales base sin tocar."""
        base = base or DEFAULT_THRESHOLDS

        if self.samples < CALIBRATION_MIN_SAMPLES:
            return base, f"Pocas muestras ({self.samples}) para calibrar; dejo los umbrales generales."
        if percent_correct < CALIBRATION_MIN_PERCENT:
            return base, (
                "No calibro con una postura de referencia mala: primero mejora la "
                "postura y vuelve a evaluarte."
            )

        # Para hombros y cabeza adelantada (menos es mejor) tomamos el percentil
        # 75 de la persona con un margen; para la altura de cabeza (mas es
        # mejor) el percentil 25 con margen hacia abajo.
        tilt = _percentile(self._samples["shoulder_tilt"], 75)
        head = _percentile(self._samples["forward_head"], 75)
        gap = _percentile(self._samples["neck_gap"], 25)

        def clamp(value, default, key):
            low, high = CALIBRATION_BOUNDS[key]
            return max(default * low, min(default * high, value))

        suggested = replace(
            base,
            shoulder_tilt_max=clamp(tilt * 1.15, base.shoulder_tilt_max, "shoulder_tilt_max"),
            forward_head_max=clamp(head * 1.15, base.forward_head_max, "forward_head_max"),
            neck_compression_min=clamp(gap * 0.90, base.neck_compression_min, "neck_compression_min"),
        )
        return suggested, f"Umbrales ajustados a tu cuerpo y tu cámara ({self.samples} muestras)."


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
    de uso (no solo la evaluacion inicial de 30s). Tambien lleva cual fue el
    problema mas frecuente, para guardarlo en el historial."""

    def __init__(self):
        self._start_time = None
        self._correct = 0
        self._incorrect = 0
        self._issues = {}

    def start(self):
        self._start_time = time.time()
        self._correct = 0
        self._incorrect = 0
        self._issues = {}

    def reset(self):
        """Deja la sesion vacia: ya guardada en el historial, no debe volver
        a guardarse (DETENER y luego cerrar la app duplicaba la fila)."""
        self.start()
        self._start_time = None

    def record(self, is_correct, feedback_text=None):
        if is_correct is None:
            return
        if is_correct:
            self._correct += 1
        else:
            self._incorrect += 1
            if feedback_text and feedback_text != "Buena postura":
                self._issues[feedback_text] = self._issues.get(feedback_text, 0) + 1

    @property
    def samples(self):
        return self._correct + self._incorrect

    @property
    def percent_correct(self):
        total = self.samples
        return (self._correct / total * 100) if total else 0.0

    @property
    def elapsed_minutes(self):
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) / 60.0

    @property
    def started_at(self):
        return self._start_time

    @property
    def common_issue(self):
        """El aviso que mas se repitio, o None si no hubo problemas."""
        if not self._issues:
            return None
        return max(self._issues.items(), key=lambda item: item[1])[0]

    def summary_text(self):
        total = self.samples
        if total == 0:
            return ""
        minutos = self.elapsed_minutes
        tiempo = f"{minutos:.0f} min" if minutos >= 1 else "<1 min"
        return f"Sesión: {self.percent_correct:.0f}% correcta ({tiempo})"
