"""Suavizado temporal de predicciones para evitar parpadeo frame a frame."""
import time
from collections import deque, Counter


class MajorityVoteSmoother:
    """Devuelve el valor mas frecuente de las ultimas `window` lecturas.

    Sirve para que el cartel en pantalla no parpadee, pero es muy corto para
    decidir un aviso: 7 frames son menos de medio segundo. Para los avisos se
    usa SustainedStateTimer.
    """

    def __init__(self, window=7):
        self._window = deque(maxlen=window)

    def update(self, value):
        self._window.append(value)
        return Counter(self._window).most_common(1)[0][0]

    def reset(self):
        self._window.clear()


class SustainedStateTimer:
    """Mide cuanto tiempo seguido lleva la postura en mal estado y avisa solo
    cuando pasa de `min_seconds`.

    El suavizado por frames votaba sobre menos de medio segundo, asi que la voz
    saltaba por cualquier movimiento corto (estirarse, girarse a hablar). Aqui
    el aviso se dispara recien cuando la mala postura se mantiene de verdad, y
    se repite cada `repeat_seconds` si no se corrige.
    """

    def __init__(self, min_seconds=6.0, repeat_seconds=25.0, grace_seconds=1.0, unknown_reset_seconds=15.0):
        self.min_seconds = min_seconds
        self.repeat_seconds = repeat_seconds
        # Un frame bueno aislado (la IA duda un instante) no deberia reiniciar
        # el contador: se tolera `grace_seconds` de postura buena.
        self.grace_seconds = grace_seconds
        # Girarse un momento no reinicia el contador, pero pasado este tiempo
        # sin poder medir el episodio se cierra: lo que paso mientras tanto no
        # se sabe y no puede contar como mala postura sostenida.
        self.unknown_reset_seconds = unknown_reset_seconds
        self._bad_since = None
        self._good_since = None
        self._last_alert_at = None
        self._last_reading_at = None
        self._was_bad = False

    # ------------------------------------------------------------------
    @property
    def bad_seconds(self):
        """Segundos seguidos en mala postura (0 si ahora esta bien)."""
        if self._bad_since is None:
            return 0.0
        return max(0.0, time.time() - self._bad_since)

    @property
    def alerted(self):
        return self._last_alert_at is not None

    @property
    def was_bad(self):
        """True si hubo un aviso pendiente de 'ya te corregiste'."""
        return self._was_bad

    # ------------------------------------------------------------------
    def update(self, is_correct, now=None):
        """Registra el estado actual y devuelve el evento que corresponde:

        - "alert"   -> toca avisar (mala postura sostenida)
        - "praise"  -> se corrigio despues de un aviso
        - None      -> nada que decir
        """
        now = time.time() if now is None else now

        last = self._last_reading_at
        if last is not None and (now - last) > self.unknown_reset_seconds:
            self.reset()

        if is_correct is None:
            return None
        self._last_reading_at = now

        if not is_correct:
            self._good_since = None
            if self._bad_since is None:
                self._bad_since = now
            elapsed = now - self._bad_since
            if elapsed < self.min_seconds:
                return None
            if self._last_alert_at is None or (now - self._last_alert_at) >= self.repeat_seconds:
                self._last_alert_at = now
                self._was_bad = True
                return "alert"
            return None

        # Postura correcta: se tolera un ratito antes de dar por cerrado el
        # episodio malo, para no reiniciar el contador por un frame dudoso.
        if self._bad_since is not None:
            if self._good_since is None:
                self._good_since = now
            if (now - self._good_since) < self.grace_seconds:
                return None
            self._bad_since = None
            self._good_since = None

        self._last_alert_at = None
        if self._was_bad:
            self._was_bad = False
            return "praise"
        return None

    def reset(self):
        self._bad_since = None
        self._good_since = None
        self._last_alert_at = None
        self._last_reading_at = None
        self._was_bad = False
