"""Tests de los arreglos de la revision: la voz hablaba solo la primera vez,
el aviso podia decir "Buena postura", la sesion se guardaba dos veces y la
prueba de train.py podia quedar con una sola clase.

La voz se prueba con un "altavoz" falso (parametro `speaker`): el hilo, la cola
y el anti-spam son los reales, solo se reemplaza el motor TTS.
"""
import threading
import time
import unittest

import numpy as np

import train
from posture_metrics import SessionStats
from tests.test_alerts import VoiceSpy
from voice_feedback import PostureVoiceCoach, VoiceFeedback


def esperar(condicion, timeout=3.0):
    limite = time.time() + timeout
    while time.time() < limite:
        if condicion():
            return True
        time.sleep(0.01)
    return False


class Altavoz:
    """Anota lo dicho; opcionalmente falla o se queda esperando."""

    def __init__(self, falla_en=(), bloquear_primero=False):
        self.dicho = []
        self.falla_en = set(falla_en)
        self.liberar = threading.Event()
        if not bloquear_primero:
            self.liberar.set()
        self._bloquear_primero = bloquear_primero

    def __call__(self, mensaje):
        if self._bloquear_primero and not self.dicho:
            self.dicho.append(mensaje)
            self.liberar.wait(timeout=5.0)
            return
        if mensaje in self.falla_en:
            raise RuntimeError("motor caido")
        self.dicho.append(mensaje)


class TestVoiceFeedback(unittest.TestCase):
    def test_habla_todos_los_mensajes_no_solo_el_primero(self):
        altavoz = Altavoz()
        voz = VoiceFeedback(speaker=altavoz, cooldown=0.0)
        voz.say_now("bienvenida")
        voz.say("aviso uno")
        voz.say("aviso dos")
        self.assertTrue(esperar(lambda: len(altavoz.dicho) == 3), altavoz.dicho)
        self.assertEqual(altavoz.dicho, ["bienvenida", "aviso uno", "aviso dos"])
        voz.stop()

    def test_un_error_suelto_no_mata_la_voz(self):
        altavoz = Altavoz(falla_en={"malo"})
        voz = VoiceFeedback(speaker=altavoz, cooldown=0.0)
        voz.say("malo")
        voz.say("bueno")
        self.assertTrue(esperar(lambda: "bueno" in altavoz.dicho))
        self.assertTrue(voz.enabled)
        voz.stop()

    def test_se_puede_activar_despues_de_arrancar_apagada(self):
        altavoz = Altavoz()
        voz = VoiceFeedback(speaker=altavoz, enabled=False)
        self.assertTrue(voz.available)
        voz.say("no debe sonar")
        time.sleep(0.1)
        self.assertEqual(altavoz.dicho, [])
        voz.set_enabled(True)
        voz.say_now("ahora si")
        self.assertTrue(esperar(lambda: altavoz.dicho == ["ahora si"]))
        voz.stop()

    def test_un_aviso_descartado_por_cola_llena_no_gasta_el_cooldown(self):
        altavoz = Altavoz(bloquear_primero=True)
        voz = VoiceFeedback(speaker=altavoz, cooldown=60.0, max_pending=1)
        voz.say("A")
        self.assertTrue(esperar(lambda: altavoz.dicho == ["A"]))  # A en curso
        voz.say("B")   # queda en cola
        voz.say("C")   # cola llena: se descarta
        altavoz.liberar.set()
        self.assertTrue(esperar(lambda: altavoz.dicho == ["A", "B"]))
        # C nunca se dijo, asi que repetirlo no es spam y debe sonar.
        voz.say("C")
        self.assertTrue(esperar(lambda: altavoz.dicho[-1] == "C"), altavoz.dicho)
        voz.stop()

    def test_no_repite_el_mismo_aviso_dentro_del_cooldown(self):
        altavoz = Altavoz()
        voz = VoiceFeedback(speaker=altavoz, cooldown=60.0)
        voz.say("igual")
        voz.say("igual")
        self.assertTrue(esperar(lambda: altavoz.dicho == ["igual"]))
        time.sleep(0.1)
        self.assertEqual(altavoz.dicho, ["igual"])
        voz.stop()

    def test_stop_no_se_cuelga_con_la_cola_llena(self):
        altavoz = Altavoz(bloquear_primero=True)
        voz = VoiceFeedback(speaker=altavoz, max_pending=1, cooldown=0.0)
        voz.say("A")
        self.assertTrue(esperar(lambda: altavoz.dicho == ["A"]))
        voz.say("B")
        hilo = threading.Thread(target=voz.stop)
        hilo.start()
        hilo.join(timeout=2.0)
        self.assertFalse(hilo.is_alive(), "stop() quedo bloqueado")
        altavoz.liberar.set()


class TestCoachNoDiceBuenaPostura(unittest.TestCase):
    def _alerta(self, feedback, antes=()):
        voz = VoiceSpy()
        coach = PostureVoiceCoach(voz, min_bad_seconds=5.0)
        for texto in antes:
            coach.update(False, texto, now=0.0)
        coach.update(False, feedback, now=0.0)
        self.assertEqual(coach.update(False, feedback, now=6.0), "alert")
        return voz.said[-1]

    def test_si_las_reglas_dicen_buena_postura_se_avisa_algo_accionable(self):
        mensaje = self._alerta("Buena postura")
        self.assertNotEqual(mensaje, "Buena postura")
        self.assertEqual(mensaje, PostureVoiceCoach.GENERIC_ALERT)

    def test_usa_la_ultima_correccion_real_si_ahora_no_hay_ninguna(self):
        mensaje = self._alerta("Buena postura", antes=["Endereza los hombros"])
        self.assertEqual(mensaje, "Endereza los hombros")

    def test_una_correccion_real_se_dice_tal_cual(self):
        self.assertEqual(self._alerta("Retrasa un poco la cabeza"), "Retrasa un poco la cabeza")


class TestSessionStatsReset(unittest.TestCase):
    def test_reset_deja_la_sesion_vacia(self):
        sesion = SessionStats()
        sesion.start()
        sesion.record(True)
        sesion.record(False, "Endereza los hombros")
        self.assertEqual(sesion.samples, 2)
        sesion.reset()
        self.assertEqual(sesion.samples, 0)
        self.assertEqual(sesion.elapsed_minutes, 0.0)
        self.assertEqual(sesion.summary_text(), "")


class TestSplitByGroup(unittest.TestCase):
    def test_con_cuatro_sesiones_la_prueba_tiene_las_dos_clases(self):
        # Es el flujo de 1-GRABAR-DATOS.bat: 2 sesiones correctas + 2 incorrectas.
        grupos = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 50 + ["d"] * 50)
        y = np.array([1] * 100 + [0] * 100)
        X = np.zeros((200, 99))
        train_idx, test_idx = train.split_by_group(X, y, grupos)
        self.assertEqual(set(y[test_idx]), {0, 1})
        self.assertEqual(set(y[train_idx]), {0, 1})
        self.assertFalse(set(grupos[train_idx]) & set(grupos[test_idx]))


if __name__ == "__main__":
    unittest.main()
