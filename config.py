"""Configuracion en config.json: umbrales, voz y numero de camara dejan de
estar fijos en el codigo y se pueden editar desde la interfaz.

El archivo se crea solo la primera vez que se guarda. Si falta, o si tiene
claves raras, se usan los valores por defecto sin romper la app.
"""
import json
import math
from pathlib import Path

CONFIG_FILE = "config.json"
ANCHOR_KEY = "threshold_anchor"

DEFAULTS = {
    # Camara
    "camera_index": 0,
    "flip_camera": True,
    # Voz
    "voice_enabled": True,
    "voice_rate": 175,
    "voice_cooldown": 6.0,
    # Umbrales de postura (proporciones del tamano del cuerpo)
    "shoulder_tilt_max": 0.20,
    "forward_head_max": 0.55,
    "neck_compression_min": 0.23,
    # Umbrales de perfil (de lado la oreja, el hombro y la cadera se miden distinto)
    "side_head_forward_max": 0.65,   # oreja por delante del hombro / altura del cuello
    "side_neck_height_min": 1.70,    # altura de la cabeza sobre el hombro, en tamanos de cabeza
    "side_trunk_lean_max": 12.0,     # grados que el hombro queda por delante de la cadera
    "calibrated": False,
    # Umbrales que dejo train.py con tus grabaciones. La calibracion de 30s solo
    # se puede mover alrededor de estos valores; sin ellos, de los generales.
    ANCHOR_KEY: None,
    # Avisos
    "alert_after_seconds": 6.0,
    "alert_repeat_seconds": 25.0,
    "break_minutes": 35,
    "break_reminders": True,
    "desktop_notifications": True,
    # Evaluacion inicial
    "assessment_seconds": 30.0,
    # Vista
    "privacy_mode": False,
    # Historial
    "save_history": True,
}

# Rangos permitidos: evita que un config.json editado a mano deje la app
# inservible (camara -3, umbral 0, avisos cada 0 segundos...).
LIMITS = {
    "camera_index": (0, 10),
    "voice_rate": (80, 400),
    "voice_cooldown": (1.0, 60.0),
    "shoulder_tilt_max": (0.02, 1.0),
    "forward_head_max": (0.05, 3.0),
    "neck_compression_min": (0.02, 1.0),
    "side_head_forward_max": (0.05, 3.0),
    "side_neck_height_min": (0.3, 5.0),
    "side_trunk_lean_max": (2.0, 60.0),
    "alert_after_seconds": (1.0, 120.0),
    "alert_repeat_seconds": (5.0, 600.0),
    "break_minutes": (5, 240),
    "assessment_seconds": (5.0, 300.0),
}

THRESHOLD_KEYS = (
    "shoulder_tilt_max",
    "forward_head_max",
    "neck_compression_min",
    "side_head_forward_max",
    "side_neck_height_min",
    "side_trunk_lean_max",
)


def _coerce_anchor(value):
    """Solo acepta un dict con umbrales conocidos, cada uno dentro de sus limites."""
    if not isinstance(value, dict):
        return None
    anchor = {key: _coerce(key, value[key]) for key in THRESHOLD_KEYS if key in value}
    return anchor or None


def _coerce(key, value):
    """Convierte el valor al tipo del default y lo deja dentro de los limites."""
    if key == ANCHOR_KEY:
        return _coerce_anchor(value)

    default = DEFAULTS[key]
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "si", "sí", "on", "yes")
            value = bool(value)
        elif isinstance(default, int):
            value = int(round(float(value)))
        elif isinstance(default, float):
            value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if isinstance(value, float) and not math.isfinite(value):
        return default

    low, high = LIMITS.get(key, (None, None))
    if low is not None:
        value = max(low, min(high, value))
    return value


class AppConfig:
    """Acceso por atributo a la configuracion, con guardado en disco."""

    def __init__(self, values=None, path=CONFIG_FILE):
        self._path = Path(path)
        self._values = dict(DEFAULTS)
        if values:
            self.update(values, save=False)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path=CONFIG_FILE):
        cfg = cls(path=path)
        file_path = Path(path)
        if not file_path.exists():
            return cfg
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"config.json ilegible ({exc}), uso los valores por defecto.")
            return cfg
        if isinstance(raw, dict):
            cfg.update(raw, save=False)
        return cfg

    def save(self):
        try:
            self._path.write_text(
                json.dumps(self._values, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return True
        except OSError as exc:
            print(f"No pude guardar config.json ({exc}).")
            return False

    # ------------------------------------------------------------------
    def update(self, values, save=True):
        """Aplica solo las claves conocidas; ignora el resto."""
        changed = False
        for key, value in (values or {}).items():
            if key not in DEFAULTS:
                continue
            # Un campo vacio llega como None: no cambia nada (el umbral guardado
            # no debe volver al valor por defecto por dejar una casilla en blanco).
            if value is None and key != ANCHOR_KEY:
                continue
            new_value = _coerce(key, value)
            if self._values.get(key) != new_value:
                self._values[key] = new_value
                changed = True
        if save and changed:
            self.save()
        return changed

    def as_dict(self):
        return dict(self._values)

    def would_change(self, key, value):
        """True si guardar `value` cambiaria de verdad lo que hay en `key`."""
        if key not in DEFAULTS or value is None:
            return False
        return _coerce(key, value) != self._values[key]

    def thresholds_changed(self, values):
        return any(self.would_change(key, values.get(key)) for key in THRESHOLD_KEYS)

    def reset(self, save=True):
        """Vuelve a los valores por defecto. Los umbrales de train.py se conservan:
        con los generales, en una webcam de portatil, la regla de cabeza adelantada
        marcaba mal todas las posturas buenas."""
        anchor = self._values[ANCHOR_KEY]
        self._values = dict(DEFAULTS)
        if anchor:
            self._values.update(anchor)
            self._values[ANCHOR_KEY] = anchor
            self._values["calibrated"] = True
        if save:
            self.save()

    # ------------------------------------------------------------------
    def thresholds(self):
        """Devuelve los umbrales como PostureThresholds."""
        from posture_metrics import PostureThresholds  # noqa: PLC0415  (evita ciclo)

        return PostureThresholds(**{key: self._values[key] for key in THRESHOLD_KEYS})

    def anchor_thresholds(self):
        """Los umbrales de referencia de la calibracion: los de train.py o, si
        todavia no hay, los generales."""
        from posture_metrics import PostureThresholds  # noqa: PLC0415  (evita ciclo)

        anchor = self._values[ANCHOR_KEY] or {}
        return PostureThresholds(**{key: anchor.get(key, DEFAULTS[key]) for key in THRESHOLD_KEYS})

    def set_thresholds(self, thresholds, calibrated=True, save=True, anchor=False):
        """Guarda los umbrales. Con anchor=True quedan ademas como referencia de
        la calibracion en vivo (lo hace train.py, que parte de datos etiquetados)."""
        values = {key: getattr(thresholds, key) for key in THRESHOLD_KEYS}
        if anchor:
            values[ANCHOR_KEY] = dict(values)
        self.update({**values, "calibrated": calibrated}, save=save)

    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # Solo se llama si el atributo no existe normalmente.
        try:
            return self.__dict__["_values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("save",):
            super().__setattr__(name, value)
        elif name in DEFAULTS:
            self._values[name] = _coerce(name, value)
        else:
            super().__setattr__(name, value)

    def __getitem__(self, key):
        return self._values[key]

    def __repr__(self):
        return f"AppConfig({self._values!r})"
