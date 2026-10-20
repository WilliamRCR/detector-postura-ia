"""Suavizado temporal de predicciones para evitar parpadeo frame a frame."""
from collections import deque, Counter


class MajorityVoteSmoother:
    """Devuelve el valor mas frecuente de las ultimas `window` lecturas."""

    def __init__(self, window=7):
        self._window = deque(maxlen=window)

    def update(self, value):
        self._window.append(value)
        return Counter(self._window).most_common(1)[0][0]

    def reset(self):
        self._window.clear()
