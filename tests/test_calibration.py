"""Tests de la calibracion personal y de la evaluacion inicial."""
import unittest

from dataclasses import replace

from assessment import PostureAssessment, verdict_lines
from posture_metrics import (
    CALIBRATION_MIN_SAMPLES,
    DEFAULT_THRESHOLDS,
    PostureCalibrator,
    PostureThresholds,
    evaluate_aspects,
    evaluate_posture,
    _percentile,
)
from tests import fake_pose


def metrics_series(tilt, head, gap, n=40):
    return [{"shoulder_tilt": tilt, "forward_head": head, "neck_gap": gap} for _ in range(n)]


class TestPercentil(unittest.TestCase):
    def test_mediana(self):
        self.assertAlmostEqual(_percentile([1, 2, 3, 4, 5], 50), 3.0)

    def test_percentil_interpolado(self):
        self.assertAlmostEqual(_percentile([0, 10], 75), 7.5)

    def test_lista_vacia(self):
        self.assertIsNone(_percentile([], 50))

    def test_un_solo_valor(self):
        self.assertEqual(_percentile([4.2], 90), 4.2)


class TestPostureCalibrator(unittest.TestCase):
    def test_sin_muestras_no_calibra(self):
        cal = PostureCalibrator()
        thresholds, nota = cal.suggest()
        self.assertEqual(thresholds, DEFAULT_THRESHOLDS)
        self.assertIn("Pocas muestras", nota)

    def test_pocas_muestras_no_calibran(self):
        cal = PostureCalibrator()
        for m in metrics_series(0.05, 0.10, 0.40, n=CALIBRATION_MIN_SAMPLES - 1):
            cal.record(m)
        thresholds, _ = cal.suggest()
        self.assertEqual(thresholds, DEFAULT_THRESHOLDS)

    def test_no_calibra_sobre_postura_mala(self):
        """Calibrar con una postura de referencia mala dejaria umbrales que la
        aprueban para siempre."""
        cal = PostureCalibrator()
        for m in metrics_series(0.40, 0.90, 0.05):
            cal.record(m)
        thresholds, nota = cal.suggest(percent_correct=20.0)
        self.assertEqual(thresholds, DEFAULT_THRESHOLDS)
        self.assertIn("postura de referencia mala", nota)

    def test_persona_muy_derecha_obtiene_umbrales_mas_exigentes(self):
        cal = PostureCalibrator()
        for m in metrics_series(0.02, 0.05, 0.60):
            cal.record(m)
        thresholds, nota = cal.suggest(percent_correct=95.0)
        self.assertLess(thresholds.shoulder_tilt_max, DEFAULT_THRESHOLDS.shoulder_tilt_max)
        self.assertLess(thresholds.forward_head_max, DEFAULT_THRESHOLDS.forward_head_max)
        self.assertIn("tu cámara", nota)

    def test_los_umbrales_no_se_alejan_sin_limite(self):
        """Aunque la persona mida valores absurdos, los umbrales se quedan
        cerca de los generales."""
        cal = PostureCalibrator()
        for m in metrics_series(0.0001, 0.0001, 50.0):
            cal.record(m)
        thresholds, _ = cal.suggest(percent_correct=100.0)
        self.assertGreaterEqual(
            thresholds.shoulder_tilt_max, DEFAULT_THRESHOLDS.shoulder_tilt_max * 0.7 - 1e-9
        )
        self.assertLessEqual(
            thresholds.neck_compression_min, DEFAULT_THRESHOLDS.neck_compression_min * 1.3 + 1e-9
        )

    def test_evaluarse_varias_veces_no_acumula_el_tope(self):
        """Cada evaluacion partia del umbral anterior: 0.55 -> 0.88 -> 1.41 ->
        2.25 -> 3.60, hasta aprobar cualquier postura."""
        tope = DEFAULT_THRESHOLDS.forward_head_max * 1.6
        actuales = DEFAULT_THRESHOLDS
        for _ in range(6):
            cal = PostureCalibrator()
            for m in metrics_series(5.0, 5.0, 0.30):
                cal.record(m)
            actuales, _ = cal.suggest(base=actuales, percent_correct=95.0)
            self.assertLessEqual(actuales.forward_head_max, tope + 1e-9)

    def test_los_topes_se_miden_desde_los_umbrales_de_entrenamiento(self):
        """Una webcam de portatil mide cabeza ~1.2: esos umbrales de train.py no
        pueden volver a 0.88 por hacer la evaluacion."""
        ancla = replace(DEFAULT_THRESHOLDS, forward_head_max=1.5)
        cal = PostureCalibrator()
        for m in metrics_series(0.02, 1.2, 0.30):
            cal.record(m)
        sugeridos, _ = cal.suggest(base=ancla, percent_correct=95.0, anchor=ancla)
        self.assertAlmostEqual(sugeridos.forward_head_max, 1.2 * 1.15, places=6)

    def test_ignora_valores_no_finitos(self):
        cal = PostureCalibrator()
        cal.record({"shoulder_tilt": float("nan"), "forward_head": 0.1, "neck_gap": 0.4})
        self.assertEqual(cal.samples, 0)

    def test_record_acepta_none(self):
        cal = PostureCalibrator()
        cal.record(None)
        self.assertEqual(cal.samples, 0)


class TestPostureAssessment(unittest.TestCase):
    def test_sin_muestras_devuelve_veredicto_de_error(self):
        ev = PostureAssessment(duration=0.01)
        ev.start()
        result = ev.finish()
        self.assertEqual(result["samples"], 0)
        self.assertFalse(result["good"])
        self.assertIn("No pude verte", result["verdict"])

    def test_lo_que_no_se_pudo_medir_no_es_mala_postura(self):
        """A 45 grados is_correct es None: antes contaba como incorrecta."""
        ev = PostureAssessment(duration=60)
        ev.start()
        for _ in range(9):
            ev.record(True, "Buena postura")
        ev.record(False, "Endereza los hombros")
        for _ in range(90):
            ev.record(None, "Ubícate de frente o de perfil a la cámara")
        result = ev.finish()
        self.assertEqual(result["samples"], 10)
        self.assertAlmostEqual(result["percent_correct"], 90.0)
        self.assertAlmostEqual(result["measured_percent"], 10.0)
        self.assertEqual(result["dominant_issue"], "Endereza los hombros")
        self.assertTrue(any("Solo pude medirte" in linea for linea in verdict_lines(result)))

    def test_sin_nada_medible_pide_reubicarse_y_no_da_porcentaje(self):
        ev = PostureAssessment(duration=60)
        ev.start()
        for _ in range(50):
            ev.record(None, "Ubícate de frente o de perfil a la cámara")
        result = ev.finish()
        self.assertEqual(result["samples"], 0)
        self.assertEqual(verdict_lines(result), [result["verdict"]])
        self.assertIn("de frente o de perfil", result["verdict"])

    def test_el_resultado_normal_trae_porcentaje_aspectos_y_nota(self):
        ev = PostureAssessment(duration=60)
        ev.start()
        ev.record(True, "", aspects={"hombros": True})
        lineas = verdict_lines(ev.finish())
        self.assertEqual(lineas[0], "100% correcta en la evaluación")
        self.assertIn("Hombros nivelados", lineas[1])
        self.assertIn("Pocas muestras", lineas[-1])

    def test_finish_respeta_la_referencia_de_entrenamiento(self):
        ancla = replace(DEFAULT_THRESHOLDS, forward_head_max=1.5)
        ev = PostureAssessment(duration=60)
        ev.start()
        for m in metrics_series(0.02, 1.2, 0.30):
            ev.record(True, "Buena postura", metrics=m)
        result = ev.finish(base_thresholds=ancla, anchor_thresholds=ancla)
        self.assertAlmostEqual(result["thresholds"].forward_head_max, 1.2 * 1.15, places=6)

    def test_cuenta_correctas_e_incorrectas(self):
        ev = PostureAssessment(duration=60)
        ev.start()
        for _ in range(8):
            ev.record(True, "Buena postura")
        for _ in range(2):
            ev.record(False, "Endereza los hombros")
        result = ev.finish()
        self.assertEqual(result["samples"], 10)
        self.assertAlmostEqual(result["percent_correct"], 80.0)
        self.assertTrue(result["good"])
        self.assertEqual(result["dominant_issue"], "Endereza los hombros")

    def test_devuelve_umbrales_calibrados(self):
        ev = PostureAssessment(duration=60)
        ev.start()
        for m in metrics_series(0.02, 0.05, 0.60):
            ev.record(True, "Buena postura", context="sentado", metrics=m)
        result = ev.finish()
        self.assertIsNotNone(result["thresholds"])
        self.assertIsInstance(result["thresholds"], PostureThresholds)

    def test_no_devuelve_umbrales_si_no_cambian(self):
        ev = PostureAssessment(duration=60, calibrate=False)
        ev.start()
        for _ in range(30):
            ev.record(True, "Buena postura")
        result = ev.finish()
        self.assertIsNone(result["thresholds"])

    def test_record_se_ignora_despues_de_finish(self):
        ev = PostureAssessment(duration=60)
        ev.start()
        ev.record(True, "Buena postura")
        ev.finish()
        ev.record(False, "Endereza los hombros")
        self.assertFalse(ev.active)

    def test_aspectos_se_promedian(self):
        ev = PostureAssessment(duration=60)
        ev.start()
        for _ in range(3):
            ev.record(True, "", aspects={"hombros": True, "altura_cabeza": False})
        ev.record(True, "", aspects={"hombros": False, "altura_cabeza": False})
        result = ev.finish()
        self.assertAlmostEqual(result["aspect_percents"]["hombros"], 75.0)
        self.assertAlmostEqual(result["aspect_percents"]["altura_cabeza"], 0.0)


class TestEvaluatePosture(unittest.TestCase):
    def test_buena_postura(self):
        self.assertEqual(evaluate_posture(fake_pose.good_posture()), "Buena postura")

    def test_perfil_pide_ponerse_de_frente(self):
        self.assertIn("frente", evaluate_posture(fake_pose.make_profile_landmarks()))

    def test_hombros_ladeados(self):
        pose = fake_pose.make_landmarks(shoulder_tilt=0.15, ear_gap=0.12)
        self.assertEqual(evaluate_posture(pose), "Endereza los hombros")

    def test_cuello_hundido(self):
        pose = fake_pose.make_landmarks(ear_gap=0.01)
        self.assertEqual(evaluate_posture(pose), "Levanta un poco la cabeza")

    def test_umbrales_personalizados_cambian_el_veredicto(self):
        pose = fake_pose.make_landmarks(shoulder_tilt=0.05, ear_gap=0.12)
        estricto = PostureThresholds(shoulder_tilt_max=0.01)
        self.assertEqual(evaluate_posture(pose, estricto), "Endereza los hombros")
        self.assertEqual(evaluate_posture(pose), "Buena postura")

    def test_aspectos_se_evaluan_por_separado(self):
        pose = fake_pose.make_landmarks(shoulder_tilt=0.15, ear_gap=0.12)
        aspects = evaluate_aspects(pose)
        self.assertFalse(aspects["hombros"])
        self.assertTrue(aspects["altura_cabeza"])

    def test_aspectos_none_en_perfil(self):
        self.assertIsNone(evaluate_aspects(fake_pose.make_profile_landmarks()))


if __name__ == "__main__":
    unittest.main()
