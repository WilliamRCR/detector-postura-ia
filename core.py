"""Nucleo compartido: keypoints, caracteristicas normalizadas, carga del modelo
y prediccion.

Antes `get_keypoints` estaba copiado en 5 archivos y cada cambio en los datos
del modelo obligaba a editarlos todos. Ahora todo eso vive aqui y el resto de
los modulos solo importa de este archivo.
"""
import json
import math
from pathlib import Path

import numpy as np

MODEL_FILE = "modelo_postura.pkl"
SCALER_FILE = "scaler.pkl"
META_FILE = "modelo_meta.json"
DATASET_FILE = "dataset_posturas.npz"

NUM_LANDMARKS = 33
RAW_SIZE = NUM_LANDMARKS * 3  # 99 valores crudos (x, y, z por punto)

# Version del formato de caracteristicas. Si cambia, un modelo viejo ya no
# sirve y hay que volver a entrenar: por eso se guarda junto al .pkl.
FEATURE_VERSION = 2

# Indices de MediaPipe Pose que usamos por nombre.
NOSE = 0
EAR_L, EAR_R = 7, 8
SHOULDER_L, SHOULDER_R = 11, 12
HIP_L, HIP_R = 23, 24

# Ancho de hombros minimo (en coordenadas normalizadas de MediaPipe) para que
# valga la pena normalizar: por debajo de esto la persona esta de perfil o la
# deteccion es basura.
MIN_SHOULDER_WIDTH = 0.03

# Las piernas fuera del encuadre hacen que MediaPipe extrapole puntos lejanos.
# Tras normalizar, recortamos a este rango para que esos puntos inventados no
# dominen las caracteristicas.
COORD_CLIP = 6.0

EXTRA_FEATURE_NAMES = (
    "shoulder_tilt",
    "forward_head",
    "neck_gap",
    "torso_ratio",
    "neck_angle",
    "torso_angle",
    "head_tilt",
    "shoulder_line_angle",
)

NUM_FEATURES = RAW_SIZE + len(EXTRA_FEATURE_NAMES)


class ModelError(Exception):
    """Problema al cargar el modelo entrenado."""


class ModelNotFound(ModelError):
    pass


class ModelOutdated(ModelError):
    pass


# ---------------------------------------------------------------- keypoints
def keypoints_from_landmarks(landmarks):
    """Convierte los landmarks de MediaPipe en el vector crudo de 99 valores.

    Este es el formato que se guarda en el dataset: crudo, sin normalizar, para
    poder recalcular las caracteristicas sin volver a grabar datos.
    """
    kp = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]
    return np.array(kp, dtype=np.float32)


# Alias historico: varios scripts lo llamaban asi.
get_keypoints = keypoints_from_landmarks


def _as_points(kp):
    """(99,) -> (33, 3). Acepta lista, tupla o array."""
    arr = np.asarray(kp, dtype=np.float64).reshape(-1)
    if arr.size != RAW_SIZE:
        raise ValueError(f"Se esperaban {RAW_SIZE} valores crudos, llegaron {arr.size}")
    return arr.reshape(NUM_LANDMARKS, 3)


# ---------------------------------------------------------------- geometria
def _angle_between(v1, v2):
    """Angulo en grados entre dos vectores 2D, o None si alguno es degenerado."""
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-9 or mag2 < 1e-9:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _vector_angle_to_horizontal(v):
    """Inclinacion de un vector respecto a la horizontal, en grados (0 a 90)."""
    if abs(v[0]) < 1e-9 and abs(v[1]) < 1e-9:
        return 0.0
    return abs(math.degrees(math.atan2(v[1], v[0])))


def body_frame(kp):
    """Mide el cuerpo: devuelve (centro_hombros, ancho_hombros, largo_torso) o
    None si no se puede medir con confianza (persona de perfil, por ejemplo)."""
    pts = _as_points(kp)
    ls, rs = pts[SHOULDER_L], pts[SHOULDER_R]
    lh, rh = pts[HIP_L], pts[HIP_R]

    shoulder_width = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    if shoulder_width < MIN_SHOULDER_WIDTH:
        return None

    mid_shoulder = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0
    torso_length = abs(mid_hip[1] - mid_shoulder[1])
    if torso_length < 0.03:
        torso_length = shoulder_width * 1.4  # respaldo si no se ven las caderas
    return mid_shoulder, shoulder_width, torso_length


def frontality(kp):
    """Ancho de hombros / largo de torso. De frente ronda 0.7-1.2; de perfil los
    hombros se juntan y baja mucho. Devuelve None si no se puede medir."""
    frame = body_frame(kp)
    if frame is None:
        return None
    _, shoulder_width, torso_length = frame
    return shoulder_width / torso_length


FRONTAL_MIN_RATIO = 0.55


def is_frontal(kp, min_ratio=FRONTAL_MIN_RATIO):
    """True si la muestra parece tomada de frente (no de perfil)."""
    ratio = frontality(kp)
    return ratio is not None and ratio >= min_ratio


def metrics_from_keypoints(kp):
    """Razones de postura normalizadas por el tamano del cuerpo, o None si no
    se puede medir. Unica implementacion: posture_metrics la reutiliza."""
    frame = body_frame(kp)
    if frame is None:
        return None
    pts = _as_points(kp)
    mid_shoulder, shoulder_width, torso_length = frame

    nose = pts[NOSE]
    ls, rs = pts[SHOULDER_L], pts[SHOULDER_R]
    le, re = pts[EAR_L], pts[EAR_R]
    mid_ear = (le + re) / 2.0

    return {
        "shoulder_tilt": abs(ls[1] - rs[1]) / shoulder_width,
        "forward_head": (ls[2] - nose[2]) / shoulder_width,
        "neck_gap": abs(mid_ear[1] - mid_shoulder[1]) / torso_length,
    }


# --------------------------------------------------------- caracteristicas
def features_from_keypoints(kp):
    """Vector de caracteristicas listo para el modelo, o None si la muestra no
    es medible.

    Antes el modelo recibia las 99 coordenadas crudas: si te movias o te
    acercabas a la camara cambiaban todos los numeros aunque la postura fuera
    la misma. Aqui los puntos se centran en el medio de los hombros, se dividen
    entre el ancho de hombros (asi la escala deja de importar) y se agregan
    angulos de cuello y torso.
    """
    frame = body_frame(kp)
    if frame is None:
        return None

    pts = _as_points(kp)
    mid_shoulder, shoulder_width, torso_length = frame

    normalized = (pts - mid_shoulder) / shoulder_width
    normalized = np.clip(normalized, -COORD_CLIP, COORD_CLIP)

    ls, rs = pts[SHOULDER_L], pts[SHOULDER_R]
    lh, rh = pts[HIP_L], pts[HIP_R]
    le, re = pts[EAR_L], pts[EAR_R]
    mid_hip = (lh + rh) / 2.0
    mid_ear = (le + re) / 2.0

    metrics = metrics_from_keypoints(kp)

    # Angulo del cuello: oreja - hombro - cadera. 180 = cabeza alineada con el
    # torso; menos = cabeza caida hacia adelante o hacia un lado.
    neck_angle = _angle_between(mid_ear - mid_shoulder, mid_hip - mid_shoulder)
    if neck_angle is None:
        neck_angle = 180.0

    # Inclinacion del torso respecto a la vertical (0 = erguido).
    torso_vec = mid_hip - mid_shoulder
    torso_angle = 90.0 - _vector_angle_to_horizontal(torso_vec)

    head_tilt = _vector_angle_to_horizontal(re - le)
    shoulder_line_angle = _vector_angle_to_horizontal(rs - ls)

    extras = np.array(
        [
            metrics["shoulder_tilt"],
            metrics["forward_head"],
            metrics["neck_gap"],
            torso_length / shoulder_width,
            neck_angle / 180.0,
            torso_angle / 90.0,
            head_tilt / 90.0,
            shoulder_line_angle / 90.0,
        ],
        dtype=np.float64,
    )

    return np.concatenate([normalized.reshape(-1), extras]).astype(np.float32)


def features_from_landmarks(landmarks):
    """Atajo: landmarks de MediaPipe -> caracteristicas (o None)."""
    return features_from_keypoints(keypoints_from_landmarks(landmarks))


def features_matrix(raw_samples):
    """Convierte muchas muestras crudas a caracteristicas.

    Devuelve (matriz, indices_validos): las muestras que no se pueden medir
    (perfil, deteccion mala) se descartan, y los indices permiten filtrar las
    etiquetas y los grupos en paralelo.
    """
    rows = []
    keep = []
    for i, kp in enumerate(raw_samples):
        feats = features_from_keypoints(kp)
        if feats is None:
            continue
        rows.append(feats)
        keep.append(i)
    if not rows:
        return np.empty((0, NUM_FEATURES), dtype=np.float32), []
    return np.vstack(rows), keep


# --------------------------------------------------------- espejo / ruido
# Pares izquierda/derecha de MediaPipe Pose, para espejar una muestra
# intercambiando el lado correcto.
MIRROR_PAIRS = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)


def mirror_keypoints(kp):
    """Espeja una muestra cruda (x -> 1-x e intercambio de lados)."""
    pts = _as_points(kp).copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    for left, right in MIRROR_PAIRS:
        pts[[left, right]] = pts[[right, left]]
    return pts.reshape(-1).astype(np.float32)


def jitter_keypoints(kp, std=0.01, rng=None):
    """Agrega ruido leve a una muestra cruda."""
    arr = np.asarray(kp, dtype=np.float32).reshape(-1)
    rng = rng or np.random.default_rng()
    return (arr + rng.normal(0, std, size=arr.shape)).astype(np.float32)


# ------------------------------------------------------------------ modelo
def save_model(model, scaler, meta=None):
    """Guarda modelo, scaler y metadatos (version de caracteristicas)."""
    import joblib

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    info = {"feature_version": FEATURE_VERSION, "n_features": NUM_FEATURES}
    if meta:
        info.update(meta)
    Path(META_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model():
    """Carga (model, scaler, meta).

    Lanza ModelNotFound si falta entrenar y ModelOutdated si el modelo guardado
    quedo con el formato viejo de 99 coordenadas crudas.
    """
    import joblib

    if not Path(MODEL_FILE).exists() or not Path(SCALER_FILE).exists():
        raise ModelNotFound("Modelo no encontrado. Ejecuta train.py primero.")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    meta = {}
    if Path(META_FILE).exists():
        try:
            meta = json.loads(Path(META_FILE).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            meta = {}

    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and expected != NUM_FEATURES:
        raise ModelOutdated(
            f"El modelo guardado espera {expected} caracteristicas y ahora se "
            f"generan {NUM_FEATURES}. Vuelve a ejecutar train.py."
        )
    if meta.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        raise ModelOutdated(
            "El modelo se entreno con un formato de caracteristicas anterior. "
            "Vuelve a ejecutar train.py."
        )
    return model, scaler, meta


class PosturePredictor:
    """Modelo + scaler + suavizado, en un solo objeto que usan todas las apps.

    `predict` devuelve (is_correct, mensaje). is_correct es None cuando la
    muestra no es medible (persona de perfil o fuera de cuadro).
    """

    def __init__(self, model, scaler, thresholds=None, status_window=7, feedback_window=5):
        from smoothing import MajorityVoteSmoother  # noqa: PLC0415  (modulo local liviano)

        self.model = model
        self.scaler = scaler
        self.thresholds = thresholds
        self._status = MajorityVoteSmoother(window=status_window)
        self._feedback = MajorityVoteSmoother(window=feedback_window)

    @classmethod
    def load(cls, thresholds=None, **kwargs):
        model, scaler, _meta = load_model()
        return cls(model, scaler, thresholds=thresholds, **kwargs)

    def reset(self):
        self._status.reset()
        self._feedback.reset()

    def predict_from_keypoints(self, kp):
        feats = features_from_keypoints(kp)
        if feats is None:
            return None
        return int(self.model.predict(self.scaler.transform([feats]))[0])

    def predict(self, landmarks):
        # Import local: posture_metrics importa core, asi que al revés solo se
        # puede hacer en tiempo de ejecucion.
        from posture_metrics import evaluate_posture  # noqa: PLC0415

        kp = keypoints_from_landmarks(landmarks)
        raw = self.predict_from_keypoints(kp)
        if raw is None:
            self.reset()
            return None, "Ubícate de frente a la cámara"
        is_correct = self._status.update(raw) == 1
        feedback = self._feedback.update(evaluate_posture(landmarks, self.thresholds))
        return is_correct, feedback


def new_pose(static=False):
    """Crea un mp.solutions.pose.Pose con los ajustes que usan todas las apps."""
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=static,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
