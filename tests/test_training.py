"""Tests de los cambios al entrenamiento: criterio de frontalidad sin caderas,
filtro de solo-frontales en train.py y umbrales personales.

Contexto medido con una webcam de portatil: el criterio viejo (hombros/torso
>= 0.55) descartaba la mitad de las muestras reales, el modelo entrenado con
fotos de perfil daba 'correcta' al 100% de las grabaciones y, con los umbrales
por defecto, la regla de cabeza adelantada marcaba mal TODAS las posturas buenas.
"""
import unittest

import numpy as np

import core
from posture_metrics import PERSONAL_MIN_SAMPLES, PERSONAL_MIN_TILT, personal_thresholds
from tests import fake_pose


def kp_de(landmarks):
    return core.keypoints_from_landmarks(landmarks)


class TestFrontalSinCaderas(unittest.TestCase):
    def test_webcam_de_portatil_con_torso_largo_sigue_siendo_frontal(self):
        # Caderas lejisimos (cuerpo cortado a la cintura): hombros/torso < 0.55,
        # que con el criterio viejo se descartaba como "de perfil".
        landmarks = fake_pose.make_landmarks(shoulder_width=0.32, hip_y=1.2)
        self.assertLess(core.frontality(kp_de(landmarks)), 0.55)
        self.assertTrue(core.is_frontal(kp_de(landmarks)))

    def test_nariz_desplazada_es_vista_de_tres_cuartos(self):
        landmarks = fake_pose.good_posture()
        landmarks[core.NOSE].x += 0.20  # la nariz se va hacia un lado de las orejas
        self.assertFalse(core.is_frontal(kp_de(landmarks)))

    def test_hombros_a_distinta_profundidad_no_es_frontal(self):
        landmarks = fake_pose.good_posture()
        landmarks[core.SHOULDER_L].z = -0.60  # un hombro mucho mas cerca de la camara (2x el ancho)
        self.assertFalse(core.is_frontal(kp_de(landmarks)))

    def test_persona_muy_lejos_no_es_frontal(self):
        landmarks = fake_pose.make_landmarks(shoulder_width=0.05)
        self.assertFalse(core.is_frontal(kp_de(landmarks)))

    def test_perfil_no_se_puede_medir_o_no_es_frontal(self):
        self.assertFalse(core.is_frontal(kp_de(fake_pose.make_profile_landmarks())))


class TestUmbralesPersonales(unittest.TestCase):
    def _metricas(self, n, forward_head=0.32 * 3.5):
        # Postura correcta medida en una camara tipo portatil: cabeza ~1.1.
        rng = np.random.default_rng(3)
        lote = []
        for _ in range(n):
            lm = fake_pose.make_landmarks(head_forward=forward_head * rng.uniform(0.9, 1.15) * 0.32 / 1.12)
            lote.append(core.metrics_from_keypoints(kp_de(lm)))
        return lote

    def test_pocas_muestras_no_cambian_nada(self):
        umbrales, nota = personal_thresholds(self._metricas(PERSONAL_MIN_SAMPLES - 1))
        self.assertIsNone(umbrales)
        self.assertIn("como estaban", nota)

    def test_se_adaptan_a_una_camara_que_mide_alto(self):
        metricas = self._metricas(40)
        medida_max = max(m["forward_head"] for m in metricas)
        self.assertGreater(medida_max, 0.55)  # con el umbral por defecto fallarian todas
        umbrales, _ = personal_thresholds(metricas)
        self.assertIsNotNone(umbrales)
        # Ya no se marca mal casi ninguna de sus posturas correctas...
        marcadas = sum(m["forward_head"] > umbrales.forward_head_max for m in metricas)
        self.assertLessEqual(marcadas / len(metricas), 0.10)
        # ...y esto no es posible con la calibracion en vivo (tope 0.7x-1.6x = 0.88).
        self.assertGreater(umbrales.forward_head_max, 0.88)

    def test_el_umbral_de_hombros_no_baja_del_minimo(self):
        umbrales, _ = personal_thresholds(self._metricas(40))
        self.assertGreaterEqual(umbrales.shoulder_tilt_max, PERSONAL_MIN_TILT)

    def test_ignora_muestras_no_medibles(self):
        lote = self._metricas(30) + [None, None, None]
        umbrales, _ = personal_thresholds(lote)
        self.assertIsNotNone(umbrales)


if __name__ == "__main__":
    unittest.main()
