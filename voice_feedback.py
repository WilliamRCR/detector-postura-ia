"""Feedback por voz (TTS local, sin conexion a internet) con anti-spam."""
import queue
import threading
import time

import pyttsx3


class VoiceFeedback:
    """Habla mensajes en un hilo aparte, sin repetir el mismo aviso seguido."""

    def __init__(self, cooldown=6.0, rate=175, enabled=True, max_pending=3):
        self.enabled = enabled
        self._cooldown = cooldown
        self._last_message = None
        self._last_spoken_at = 0.0
        self._queue = queue.Queue(maxsize=max_pending)
        self._stop_event = threading.Event()
        self._engine = None

        if self.enabled:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
            except Exception as exc:
                print(f"Voz no disponible ({exc}), continuo sin audio.")
                self.enabled = False

        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def say(self, message):
        if not self.enabled or not message:
            return

        now = time.time()
        is_repeat = message == self._last_message
        if is_repeat and (now - self._last_spoken_at) < self._cooldown:
            return

        self._last_message = message
        self._last_spoken_at = now

        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # Ya hay varios avisos distintos esperando turno; se descarta
            # este para no acumular atraso, pero no se pisan los pendientes.
            pass

    def say_now(self, message):
        """Anuncio prioritario (p. ej. instrucciones o veredicto final):
        se salta el cooldown/duplicados y limpia lo pendiente para sonar ya."""
        if not self.enabled or not message:
            return
        self._last_message = message
        self._last_spoken_at = time.time()
        with self._queue.mutex:
            self._queue.queue.clear()
        self._queue.put(message)

    def set_enabled(self, enabled):
        self.enabled = enabled and self._engine is not None
        if not self.enabled:
            with self._queue.mutex:
                self._queue.queue.clear()

    def _run(self):
        while not self._stop_event.is_set():
            message = self._queue.get()
            if message is None:
                break
            self._engine.say(message)
            self._engine.runAndWait()

    def stop(self):
        if self._engine is None:
            return
        self._stop_event.set()
        self._queue.put(None)


class PostureVoiceCoach:
    """Decide que decir a partir del estado de postura ya suavizado.

    Mientras la postura es incorrecta, repite el consejo (respetando el
    cooldown de VoiceFeedback). Cuando se corrige, felicita una sola vez
    y queda en silencio hasta que vuelva a detectarse una mala postura.
    """

    def __init__(self, voice_feedback, praise="Buena postura, sigue asi"):
        self._voice = voice_feedback
        self._praise = praise
        self._was_bad = False

    def update(self, is_correct, feedback_text):
        if is_correct is None:
            return
        if not is_correct:
            self._voice.say(feedback_text)
            self._was_bad = True
        elif self._was_bad:
            self._voice.say(self._praise)
            self._was_bad = False

    def reset(self):
        self._was_bad = False
