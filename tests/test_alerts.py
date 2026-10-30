"""Tests del aviso por tiempo sostenido, los recordatorios de pausa y el coach.

El punto de estos tests: antes la voz avisaba con 7 frames de mala postura
(menos de medio segundo). Ahora tiene que callarse ante movimientos cortos y
hablar solo cuando la mala postura dura de verdad.
"""
import unittest

from smoothing import MajorityVoteSmoother, SustainedStateTimer
from voice_feedback import BreakReminder, DesktopNotifier, PostureVoiceCoach


class VoiceSpy:
    """Doble de VoiceFeedback que solo anota lo que se le pide decir."""

    def __init__(self):
        self.said = []
        self.enabled = True

    def say(self, message):
        self.said.append(message)

    def say_now(self, message):
        self.said.append(message)


class NotifierSpy:
    def __init__(self):
        self.sent = []
        self.enabled = True

    def notify(self, title, message, force=False):
        self.sent.append((title, message))
        return True


class TestSustainedStateTimer(unittest.TestCase):
    def test_no_avisa_por_un_movimiento_corto(self):
        timer = SustainedStateTimer(min_seconds=6.0)
        # 3 segundos de mala postura: todavia no.
        for t in range(0, 4):
            self.assertIsNone(timer.update(False, now=float(t)))

    def test_avisa_al_pasar_el_umbral(self):
        timer = SustainedStateTimer(min_seconds=6.0)
        eventos = [timer.update(False, now=float(t)) for t in range(0, 8)]
        self.assertNotIn("alert", eventos[:6])
        self.assertIn("alert", eventos)

    def test_repite_el_aviso_pasado_el_intervalo(self):
        timer = SustainedStateTimer(min_seconds=5.0, repeat_seconds=20.0)
        self.assertIsNone(timer.update(False, now=0.0))
        self.assertEqual(timer.update(False, now=5.0), "alert")
        # Antes de los 20 s no repite.
        self.assertIsNone(timer.update(False, now=15.0))
        self.assertEqual(timer.update(False, now=25.0), "alert")

    def test_felicita_una_sola_vez_al_corregirse(self):
        timer = SustainedStateTimer(min_seconds=5.0, grace_seconds=1.0)
        timer.update(False, now=0.0)
        self.assertEqual(timer.update(False, now=6.0), "alert")
        # Se corrige: hay que esperar la gracia y luego felicita una vez.
        self.assertIsNone(timer.update(True, now=6.5))
        self.assertEqual(timer.update(True, now=8.0), "praise")
        self.assertIsNone(timer.update(True, now=9.0))

    def test_un_frame_bueno_aislado_no_reinicia_el_contador(self):
        timer = SustainedStateTimer(min_seconds=6.0, grace_seconds=1.0)
        timer.update(False, now=0.0)
        timer.update(False, now=3.0)
        timer.update(True, now=3.5)   # duda momentanea de la IA
        timer.update(False, now=4.0)
        # El contador sigue corriendo desde 0, asi que a los 6 s avisa.
        self.assertEqual(timer.update(False, now=6.5), "alert")

    def test_bad_seconds_reporta_el_tiempo_acumulado(self):
        timer = SustainedStateTimer(min_seconds=6.0)
        self.assertEqual(timer.bad_seconds, 0.0)
        timer.update(False)
        self.assertGreaterEqual(timer.bad_seconds, 0.0)

    def test_none_no_altera_el_estado(self):
        timer = SustainedStateTimer(min_seconds=5.0)
        timer.update(False, now=0.0)
        self.assertIsNone(timer.update(None, now=3.0))
        self.assertEqual(timer.update(False, now=5.0), "alert")

    def test_mucho_tiempo_sin_medir_cierra_el_episodio(self):
        """1 frame malo + 5 minutos a 45 grados + otro frame malo no son 5
        minutos de mala postura."""
        timer = SustainedStateTimer(min_seconds=6.0)
        timer.update(False, now=0.0)
        timer.update(None, now=300.0)
        self.assertIsNone(timer.update(False, now=300.1))
        self.assertEqual(timer.update(False, now=306.5), "alert")

    def test_tampoco_felicita_tras_mucho_tiempo_sin_medir(self):
        timer = SustainedStateTimer(min_seconds=5.0)
        timer.update(False, now=0.0)
        self.assertEqual(timer.update(False, now=6.0), "alert")
        self.assertIsNone(timer.update(True, now=120.0))
        self.assertFalse(timer.was_bad)

    def test_reset_limpia_todo(self):
        timer = SustainedStateTimer(min_seconds=5.0)
        timer.update(False, now=0.0)
        timer.update(False, now=6.0)
        timer.reset()
        self.assertFalse(timer.was_bad)
        self.assertIsNone(timer.update(False, now=7.0))


class TestPostureVoiceCoach(unittest.TestCase):
    def test_callado_ante_mala_postura_breve(self):
        voice = VoiceSpy()
        coach = PostureVoiceCoach(voice, min_bad_seconds=6.0)
        for t in range(0, 4):
            coach.update(False, "Endereza los hombros", now=float(t))
        self.assertEqual(voice.said, [])

    def test_avisa_tras_mala_postura_sostenida(self):
        voice = VoiceSpy()
        coach = PostureVoiceCoach(voice, min_bad_seconds=6.0)
        for t in range(0, 8):
            coach.update(False, "Endereza los hombros", now=float(t))
        self.assertIn("Endereza los hombros", voice.said)

    def test_notifica_en_el_escritorio_junto_con_la_voz(self):
        voice, notifier = VoiceSpy(), NotifierSpy()
        coach = PostureVoiceCoach(voice, min_bad_seconds=5.0, notifier=notifier)
        coach.update(False, "Levanta un poco la cabeza", now=0.0)
        coach.update(False, "Levanta un poco la cabeza", now=6.0)
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(notifier.sent[0][1], "Levanta un poco la cabeza")

    def test_felicita_al_corregirse(self):
        voice = VoiceSpy()
        coach = PostureVoiceCoach(voice, min_bad_seconds=5.0, praise="Buena postura")
        coach.update(False, "Endereza los hombros", now=0.0)
        coach.update(False, "Endereza los hombros", now=6.0)
        coach.update(True, "Buena postura", now=7.0)
        coach.update(True, "Buena postura", now=9.0)
        self.assertEqual(voice.said[-1], "Buena postura")

    def test_configure_cambia_los_tiempos_en_caliente(self):
        voice = VoiceSpy()
        coach = PostureVoiceCoach(voice, min_bad_seconds=30.0)
        coach.configure(min_bad_seconds=2.0)
        coach.update(False, "Endereza los hombros", now=0.0)
        coach.update(False, "Endereza los hombros", now=3.0)
        self.assertEqual(len(voice.said), 1)


class TestDesktopNotifier(unittest.TestCase):
    def _notificador(self, enabled=True):
        notificador = DesktopNotifier(enabled=enabled, cooldown=0.0)
        notificador._notify = lambda **_k: None
        return notificador

    def test_se_puede_activar_despues_de_arrancar_apagado(self):
        notificador = self._notificador(enabled=False)
        self.assertTrue(notificador.available)
        notificador.set_enabled(True)
        self.assertTrue(notificador.enabled)
        self.assertTrue(notificador.notify("t", "m"))

    def test_un_error_suelto_no_las_apaga(self):
        notificador = self._notificador()

        def falla(**_k):
            raise OSError("sin bandeja")

        notificador._notify = falla
        for _ in range(DesktopNotifier.MAX_FAILURES - 1):
            self.assertFalse(notificador.notify("t", "m", force=True))
            self.assertTrue(notificador.enabled)
        self.assertFalse(notificador.notify("t", "m", force=True))
        self.assertFalse(notificador.enabled)


class TestBreakReminder(unittest.TestCase):
    def test_no_avisa_antes_del_intervalo(self):
        breaks = BreakReminder(interval_minutes=30)
        breaks.start(now=0.0)
        self.assertFalse(breaks.due(now=10 * 60))
        self.assertFalse(breaks.due(now=29 * 60))

    def test_avisa_al_cumplirse_el_intervalo(self):
        breaks = BreakReminder(interval_minutes=30)
        breaks.start(now=0.0)
        self.assertTrue(breaks.due(now=30 * 60))

    def test_no_repite_el_mismo_aviso(self):
        breaks = BreakReminder(interval_minutes=30)
        breaks.start(now=0.0)
        self.assertTrue(breaks.due(now=30 * 60))
        self.assertFalse(breaks.due(now=31 * 60))
        self.assertTrue(breaks.due(now=60 * 60))

    def test_los_mensajes_rotan(self):
        breaks = BreakReminder(interval_minutes=30)
        breaks.start(now=0.0)
        mensajes = []
        for i in range(1, 4):
            breaks.due(now=i * 30 * 60)
            mensajes.append(breaks.message())
        self.assertEqual(len(set(mensajes)), 3)

    def test_desactivado_nunca_avisa(self):
        breaks = BreakReminder(interval_minutes=30, enabled=False)
        breaks.start(now=0.0)
        self.assertFalse(breaks.due(now=90 * 60))


class TestMajorityVoteSmoother(unittest.TestCase):
    def test_devuelve_el_valor_mas_frecuente(self):
        smoother = MajorityVoteSmoother(window=5)
        for value in (1, 1, 0, 1, 1):
            resultado = smoother.update(value)
        self.assertEqual(resultado, 1)

    def test_reset_vacia_la_ventana(self):
        smoother = MajorityVoteSmoother(window=3)
        smoother.update(1)
        smoother.reset()
        self.assertEqual(smoother.update(0), 0)


if __name__ == "__main__":
    unittest.main()
