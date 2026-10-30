"""Feedback por voz (TTS local, sin conexion a internet) con anti-spam,
recordatorios de pausa y notificaciones nativas de Windows."""
import queue
import threading
import time

from smoothing import SustainedStateTimer


class VoiceFeedback:
    """Habla mensajes en un hilo aparte, sin repetir el mismo aviso seguido.

    En Windows, pyttsx3 solo habla la PRIMERA vez si se reutiliza el mismo
    motor: los siguientes runAndWait() vuelven al instante sin sonar (la app
    saludaba al abrir y quedaba muda). Por eso cada frase se dice con un motor
    nuevo, creado dentro del hilo de voz (SAPI5 exige que el motor viva en el
    hilo que lo usa).
    """

    # Errores seguidos antes de rendirse; uno suelto no debe apagar la voz.
    MAX_FAILURES = 5

    def __init__(self, cooldown=6.0, rate=175, enabled=True, max_pending=3, speaker=None):
        self.enabled = enabled
        self._cooldown = cooldown
        self._rate = int(rate)
        self._last_message = None
        self._last_spoken_at = 0.0
        self._queue = queue.Queue(maxsize=max_pending)
        self._stop_event = threading.Event()
        self._failures = 0
        self._speaker = speaker or self._speak_pyttsx3
        self._thread = None

        # Se comprueba siempre, aunque arranque en silencio: asi se puede
        # activar despues desde la interfaz.
        self._available = speaker is not None or self._probe_engine()
        if not self._available:
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def available(self):
        """True si hay motor TTS, aunque la voz este apagada ahora mismo."""
        return self._available

    # ------------------------------------------------------------------
    @staticmethod
    def _com_init():
        """SAPI5 es COM: cada hilo que lo use debe inicializarlo."""
        try:
            import pythoncom  # noqa: PLC0415  (solo existe en Windows con pywin32)

            pythoncom.CoInitialize()
            return pythoncom.CoUninitialize
        except Exception:
            return None

    def _probe_engine(self):
        """Prueba que el motor arranca, en un hilo propio por lo de COM."""
        ok = []

        def check():
            release = self._com_init()
            try:
                import pyttsx3  # noqa: PLC0415  (opcional: la app funciona sin voz)

                engine = pyttsx3.init()
                engine.stop()
                ok.append(True)
            except Exception as exc:
                print(f"Voz no disponible ({exc}), continuo sin audio.")
            finally:
                if release:
                    release()

        probe = threading.Thread(target=check, daemon=True)
        probe.start()
        probe.join(timeout=15.0)
        return bool(ok)

    def _speak_pyttsx3(self, message):
        import pyttsx3  # noqa: PLC0415

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self._rate)
            engine.say(message)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _clear_pending(self):
        with self._queue.mutex:
            self._queue.queue.clear()

    # ------------------------------------------------------------------
    def say(self, message):
        if not self.enabled or not message:
            return

        now = time.time()
        is_repeat = message == self._last_message
        if is_repeat and (now - self._last_spoken_at) < self._cooldown:
            return

        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # Ya hay varios avisos distintos esperando turno; se descarta
            # este para no acumular atraso, pero no se pisan los pendientes.
            # No se anota como dicho: asi podra reintentarse enseguida.
            return

        self._last_message = message
        self._last_spoken_at = now

    def say_now(self, message):
        """Anuncio prioritario (p. ej. instrucciones o veredicto final):
        se salta el cooldown/duplicados y limpia lo pendiente para sonar ya."""
        if not self.enabled or not message:
            return
        self._last_message = message
        self._last_spoken_at = time.time()
        self._clear_pending()
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            pass  # otro hilo se adelanto y la cola ya tiene mensajes recientes

    def set_enabled(self, enabled):
        self.enabled = bool(enabled) and self._available
        if self.enabled:
            self._failures = 0
        else:
            self._clear_pending()

    def set_rate(self, rate):
        try:
            self._rate = int(rate)
        except (TypeError, ValueError):
            pass

    def _run(self):
        release = self._com_init()
        try:
            while not self._stop_event.is_set():
                message = self._queue.get()
                if message is None:
                    break
                if not self.enabled:
                    continue  # se silencio mientras esperaba en la cola
                try:
                    self._speaker(message)
                    self._failures = 0
                except Exception as exc:
                    # Un fallo suelto no mata el hilo; solo si el motor falla
                    # varias veces seguidas se apaga la voz.
                    self._failures += 1
                    print(f"Error al hablar ({exc}).")
                    if self._failures >= self.MAX_FAILURES:
                        print("La voz falla seguido, la apago.")
                        self.enabled = False
                        self._clear_pending()
        finally:
            if release:
                release()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._clear_pending()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


class DesktopNotifier:
    """Notificaciones nativas de Windows via plyer, para que los avisos lleguen
    aunque la ventana este minimizada.

    plyer es opcional: si no esta instalado, la app funciona igual y solo se
    pierden las notificaciones de escritorio.
    """

    APP_NAME = "Detector de Postura IA"

    def __init__(self, enabled=True, cooldown=30.0):
        self.enabled = enabled
        self._cooldown = cooldown
        self._last_at = 0.0
        self._notify = None

        if not enabled:
            return
        try:
            from plyer import notification  # noqa: PLC0415  (dependencia opcional)

            self._notify = notification.notify
        except Exception as exc:
            print(f"Notificaciones de escritorio no disponibles ({exc}).")
            self.enabled = False

    @property
    def available(self):
        return self._notify is not None

    def set_enabled(self, enabled):
        self.enabled = bool(enabled) and self._notify is not None

    def notify(self, title, message, force=False):
        if not self.enabled or self._notify is None:
            return False
        now = time.time()
        if not force and (now - self._last_at) < self._cooldown:
            return False
        self._last_at = now
        try:
            self._notify(
                title=title,
                message=message,
                app_name=self.APP_NAME,
                timeout=8,
            )
            return True
        except Exception as exc:
            print(f"No pude mostrar la notificacion ({exc}).")
            self.enabled = False
            return False


class BreakReminder:
    """Cada 30-45 minutos pide levantarse y estirarse.

    Pasar horas sentado cansa aunque la postura sea buena, asi que este aviso
    no depende de como estes sentado: solo del tiempo.
    """

    MESSAGES = (
        "Llevas un buen rato sentado. Levántate y estírate un minuto.",
        "Hora de una pausa: ponte de pie, estira la espalda y mueve el cuello.",
        "Pausa activa: camina un poco y descansa la vista de la pantalla.",
    )

    def __init__(self, interval_minutes=35, enabled=True):
        self.interval_minutes = interval_minutes
        self.enabled = enabled
        self._last_at = None
        self._count = 0

    def start(self, now=None):
        self._last_at = time.time() if now is None else now
        self._count = 0

    @property
    def minutes_until_next(self):
        if self._last_at is None:
            return float(self.interval_minutes)
        elapsed = (time.time() - self._last_at) / 60.0
        return max(0.0, self.interval_minutes - elapsed)

    def due(self, now=None):
        """True (una sola vez por intervalo) cuando toca la pausa."""
        if not self.enabled:
            return False
        now = time.time() if now is None else now
        if self._last_at is None:
            self._last_at = now
            return False
        if (now - self._last_at) < self.interval_minutes * 60.0:
            return False
        self._last_at = now
        self._count += 1
        return True

    def message(self):
        """Rota el texto para que no suene igual todas las veces."""
        return self.MESSAGES[(self._count - 1) % len(self.MESSAGES)]

    def reset(self):
        self._last_at = None
        self._count = 0


class PostureVoiceCoach:
    """Decide que decir a partir del estado de postura ya suavizado.

    Antes avisaba en cuanto el voto de 7 frames daba 'incorrecta', o sea por
    movimientos de menos de medio segundo. Ahora el aviso espera a que la mala
    postura se sostenga varios segundos seguidos (SustainedStateTimer), y la
    felicitacion suena una sola vez al corregirse.
    """

    # Textos que NO sirven como correccion. El modelo y las reglas por umbral
    # pueden discrepar: la red dice "incorrecta" mientras las reglas dicen
    # "Buena postura", y la voz terminaba diciendo "buena postura" como aviso.
    NOT_A_CORRECTION = ("Buena postura", "Ubícate de frente a la cámara", "Ubícate de frente o de perfil a la cámara")
    GENERIC_ALERT = "Corrige tu postura. Endereza la espalda y relaja los hombros."

    def __init__(
        self,
        voice_feedback,
        praise="Buena postura, sigue asi",
        min_bad_seconds=6.0,
        repeat_seconds=25.0,
        notifier=None,
    ):
        self._voice = voice_feedback
        self._praise = praise
        self._notifier = notifier
        self._timer = SustainedStateTimer(
            min_seconds=min_bad_seconds, repeat_seconds=repeat_seconds
        )
        self._last_feedback = ""

    @property
    def bad_seconds(self):
        return self._timer.bad_seconds

    def configure(self, min_bad_seconds=None, repeat_seconds=None):
        if min_bad_seconds is not None:
            self._timer.min_seconds = min_bad_seconds
        if repeat_seconds is not None:
            self._timer.repeat_seconds = repeat_seconds

    def update(self, is_correct, feedback_text, now=None):
        """Devuelve el evento emitido ("alert", "praise" o None)."""
        if feedback_text and feedback_text not in self.NOT_A_CORRECTION:
            self._last_feedback = feedback_text

        event = self._timer.update(is_correct, now=now)
        if event == "alert":
            usable = feedback_text if feedback_text not in self.NOT_A_CORRECTION else ""
            message = usable or self._last_feedback or self.GENERIC_ALERT
            self._voice.say(message)
            if self._notifier is not None:
                self._notifier.notify("Corrige tu postura", message)
        elif event == "praise":
            self._voice.say(self._praise)
        return event

    def reset(self):
        self._timer.reset()
        self._last_feedback = ""
