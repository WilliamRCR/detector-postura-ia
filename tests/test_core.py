"""Tests de core.py: keypoints, normalizacion y deteccion de perfil."""
import unittest

import numpy as np

import core
from tests import fake_pose


class TestKeypoints(unittest.TestCase):
    def test_keypoints_tienen_99_valores(self):
        kp = core.keypoints_from_landmarks(fake_pose.good_posture())
        self.assertEqual(kp.shape, (core.RAW_SIZE,))

    def test_alias_historico_sigue_existiendo(self):
        self.assertIs(core.get_keypoints, core.keypoints_from_landmarks)

    def test_orden_x_y_z(self):
        landmarks = fake_pose.good_posture()
        kp = core.keypoints_from_landmarks(landmarks)
        self.assertAlmostEqual(kp[0], landmarks[0].x, places=5)
        self.assertAlmostEqual(kp[1], landmarks[0].y, places=5)
        self.assertAlmostEqual(kp[2], landmarks[0].z, places=5)


class TestNormalizacion(unittest.TestCase):
    """Lo que motivo el cambio: la misma postura a distinta distancia o en
    distinto lugar del cuadro tiene que dar casi las mismas caracteristicas."""

    def test_acercarse_a_la_camara_no_cambia_las_caracteristicas(self):
        # La misma postura, el doble de lejos (la mitad de tamano en el cuadro).
        crudo_cerca = core.keypoints_from_landmarks(
            fake_pose.make_landmarks(shoulder_tilt=0.04, head_forward=0.10, scale=1.0)
        )
        crudo_lejos = core.keypoints_from_landmarks(
            fake_pose.make_landmarks(shoulder_tilt=0.04, head_forward=0.10, scale=0.5)
        )

        # Las coordenadas crudas si cambian mucho: por eso el modelo viejo se
        # confundia cuando el usuario se acercaba o se alejaba.
        self.assertGreater(np.abs(crudo_cerca - crudo_lejos).max(), 0.05)

        # Las razones de postura, en cambio, quedan practicamente iguales.
        for key in ("shoulder_tilt", "forward_head", "neck_gap"):
            a = core.metrics_from_keypoints(crudo_cerca)[key]
            b = core.metrics_from_keypoints(crudo_lejos)[key]
            self.assertAlmostEqual(a, b, delta=0.02, msg=f"{key} cambio demasiado")

        # Y las caracteristicas que recibe el modelo tambien.
        cerca = core.features_from_keypoints(crudo_cerca)
        lejos = core.features_from_keypoints(crudo_lejos)
        np.testing.assert_allclose(cerca, lejos, atol=1e-5)

    def test_moverse_en_el_cuadro_no_cambia_las_caracteristicas(self):
        centro = core.features_from_landmarks(fake_pose.make_landmarks(center_x=0.5))
        izquierda = core.features_from_landmarks(fake_pose.make_landmarks(center_x=0.2))
        np.testing.assert_allclose(centro, izquierda, atol=1e-5)

    def test_cantidad_de_caracteristicas(self):
        feats = core.features_from_landmarks(fake_pose.good_posture())
        self.assertEqual(len(feats), core.NUM_FEATURES)
        self.assertGreater(core.NUM_FEATURES, core.RAW_SIZE)

    def test_el_centro_de_hombros_queda_en_el_origen(self):
        feats = core.features_from_landmarks(fake_pose.make_landmarks(shoulder_tilt=0.0))
        puntos = feats[: core.RAW_SIZE].reshape(core.NUM_LANDMARKS, 3)
        medio = (puntos[core.SHOULDER_L] + puntos[core.SHOULDER_R]) / 2
        np.testing.assert_allclose(medio[:2], [0.0, 0.0], atol=1e-5)

    def test_sin_valores_infinitos_ni_nan(self):
        for pose in (fake_pose.good_posture(), fake_pose.bad_posture()):
            feats = core.features_from_landmarks(pose)
            self.assertTrue(np.all(np.isfinite(feats)))


class TestPerfil(unittest.TestCase):
    def test_perfil_no_es_frontal(self):
        kp = core.keypoints_from_landmarks(fake_pose.make_profile_landmarks())
        self.assertFalse(core.is_frontal(kp))

    def test_frontal_es_frontal(self):
        kp = core.keypoints_from_landmarks(fake_pose.good_posture())
        self.assertTrue(core.is_frontal(kp))

    def test_perfil_no_produce_caracteristicas(self):
        self.assertIsNone(core.features_from_landmarks(fake_pose.make_profile_landmarks()))

    def test_hombros_estrechos_tambien_se_descartan(self):
        # Hombros visibles pero muy juntos respecto al torso: vista de 3/4.
        kp = core.keypoints_from_landmarks(
            fake_pose.make_landmarks(shoulder_width=0.10, hip_y=0.95)
        )
        self.assertFalse(core.is_frontal(kp))


class TestEspejoYRuido(unittest.TestCase):
    def test_espejo_dos_veces_vuelve_al_original(self):
        kp = core.keypoints_from_landmarks(fake_pose.bad_posture())
        np.testing.assert_allclose(core.mirror_keypoints(core.mirror_keypoints(kp)), kp, atol=1e-5)

    def test_espejo_intercambia_los_hombros(self):
        kp = core.keypoints_from_landmarks(fake_pose.make_landmarks(shoulder_tilt=0.10))
        espejo = core.mirror_keypoints(kp).reshape(core.NUM_LANDMARKS, 3)
        original = kp.reshape(core.NUM_LANDMARKS, 3)
        # La altura del hombro izquierdo pasa a ser la del derecho.
        self.assertAlmostEqual(
            espejo[core.SHOULDER_L][1], original[core.SHOULDER_R][1], places=5
        )

    def test_el_ruido_es_leve(self):
        kp = core.keypoints_from_landmarks(fake_pose.good_posture())
        rng = np.random.default_rng(0)
        ruidoso = core.jitter_keypoints(kp, std=0.01, rng=rng)
        self.assertLess(np.abs(ruidoso - kp).max(), 0.08)
        self.assertGreater(np.abs(ruidoso - kp).max(), 0.0)

    def test_features_matrix_descarta_las_no_medibles(self):
        buenas = core.keypoints_from_landmarks(fake_pose.good_posture())
        perfil = core.keypoints_from_landmarks(fake_pose.make_profile_landmarks())
        matriz, keep = core.features_matrix([buenas, perfil, buenas])
        self.assertEqual(matriz.shape, (2, core.NUM_FEATURES))
        self.assertEqual(keep, [0, 2])


class TestMetricas(unittest.TestCase):
    def test_hombros_ladeados_suben_el_tilt(self):
        nivelados = core.metrics_from_keypoints(
            core.keypoints_from_landmarks(fake_pose.make_landmarks(shoulder_tilt=0.0))
        )
        ladeados = core.metrics_from_keypoints(
            core.keypoints_from_landmarks(fake_pose.make_landmarks(shoulder_tilt=0.12))
        )
        self.assertGreater(ladeados["shoulder_tilt"], nivelados["shoulder_tilt"])

    def test_cabeza_adelantada_sube_forward_head(self):
        atras = core.metrics_from_keypoints(
            core.keypoints_from_landmarks(fake_pose.make_landmarks(head_forward=0.0))
        )
        adelante = core.metrics_from_keypoints(
            core.keypoints_from_landmarks(fake_pose.make_landmarks(head_forward=0.25))
        )
        self.assertGreater(adelante["forward_head"], atras["forward_head"])

    def test_cuello_hundido_baja_neck_gap(self):
        estirado = core.metrics_from_keypoints(
            core.keypoints_from_landmarks(fake_pose.make_landmarks(ear_gap=0.15))
        )
        hundido = core.metrics_from_keypoints(
            core.keypoints_from_landmarks(fake_pose.make_landmarks(ear_gap=0.02))
        )
        self.assertLess(hundido["neck_gap"], estirado["neck_gap"])


class TestValidaciones(unittest.TestCase):
    def test_tamano_incorrecto_lanza_error(self):
        with self.assertRaises(ValueError):
            core.features_from_keypoints(np.zeros(50))


if __name__ == "__main__":
    unittest.main()
