"""Tests de las dos vistas (frente y perfil) y del entrenamiento con las cuatro
situaciones: sentado de frente, sentado de perfil, de pie de frente y de pie de
perfil.

Contexto medido: de perfil los hombros se superponen, asi que el ancho de
hombros ya no sirve de escala (se usa el tamano de la cabeza) y las reglas
miden otras cosas (oreja por delante del hombro, altura de cabeza, inclinacion
del tronco).
"""
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

import collect_data
import core
import train
from config import DEFAULTS, THRESHOLD_KEYS, AppConfig
from posture_metrics import (
    PERSONAL_MIN_SAMPLES,
    PostureCalibrator,
    PostureThresholds,
    evaluate_aspects,
    evaluate_posture,
    personal_thresholds,
    scene_label,
)
from tests import fake_pose


def kp_de(landmarks):
    return core.keypoints_from_landmarks(landmarks)


class TestDeteccionDeVista(unittest.TestCase):
    def test_de_frente(self):
        self.assertEqual(core.view_of(kp_de(fake_pose.good_posture())), core.VIEW_FRONT)

    def test_de_perfil_mirando_a_cualquier_lado(self):
        for facing in (1, -1):
            kp = kp_de(fake_pose.make_side_landmarks(facing=facing))
            self.assertEqual(core.view_of(kp), core.VIEW_SIDE, facing)

    def test_tres_cuartos_es_intermedia(self):
        landmarks = fake_pose.good_posture()
        landmarks[core.NOSE].x += 0.12  # nariz fuera del eje, pero hombros todavia anchos
        self.assertEqual(core.view_of(kp_de(landmarks)), core.VIEW_MIXED)

    def test_persona_muy_lejos_de_frente_es_intermedia(self):
        self.assertEqual(core.view_of(kp_de(fake_pose.make_landmarks(shoulder_width=0.05))), core.VIEW_MIXED)

    def test_intermedia_no_produce_medidas_ni_caracteristicas(self):
        kp = kp_de(fake_pose.make_landmarks(shoulder_width=0.05))
        self.assertIsNone(core.metrics_from_keypoints(kp))
        self.assertIsNone(core.features_from_keypoints(kp))


class TestCaracteristicasDePerfil(unittest.TestCase):
    def test_mismas_dimensiones_en_las_dos_vistas(self):
        for pose in (fake_pose.good_posture(), fake_pose.good_side_posture(), fake_pose.bad_side_posture()):
            feats = core.features_from_landmarks(pose)
            self.assertEqual(len(feats), core.NUM_FEATURES)
            self.assertTrue(np.all(np.isfinite(feats)))

    def test_indicador_de_vista(self):
        self.assertEqual(core.features_from_landmarks(fake_pose.good_posture())[-1], 0.0)
        self.assertEqual(core.features_from_landmarks(fake_pose.good_side_posture())[-1], 1.0)

    def test_da_igual_hacia_que_lado_mire(self):
        a = core.features_from_landmarks(fake_pose.make_side_landmarks(facing=1))
        b = core.features_from_landmarks(fake_pose.make_side_landmarks(facing=-1))
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_acercarse_a_la_camara_no_cambia_las_caracteristicas(self):
        lejos = core.features_from_landmarks(fake_pose.make_side_landmarks(scale=1.0))
        cerca = core.features_from_landmarks(fake_pose.make_side_landmarks(scale=1.6))
        np.testing.assert_allclose(lejos, cerca, atol=1e-4)

    def test_la_postura_mala_se_distingue_de_la_buena(self):
        buena = core.features_from_landmarks(fake_pose.good_side_posture())
        mala = core.features_from_landmarks(fake_pose.bad_side_posture())
        self.assertGreater(np.abs(buena - mala).max(), 0.5)

    def test_escala_de_perfil_es_la_cabeza_no_los_hombros(self):
        kp = kp_de(fake_pose.good_side_posture())
        frente = kp_de(fake_pose.good_posture())
        self.assertLess(core.body_scale(kp), 0.1)       # cabeza
        self.assertGreater(core.body_scale(frente), 0.2)  # ancho de hombros


class TestReglasDePerfil(unittest.TestCase):
    def test_postura_buena_de_perfil(self):
        self.assertEqual(evaluate_posture(fake_pose.good_side_posture()), "Buena postura")
        self.assertTrue(all(evaluate_aspects(fake_pose.good_side_posture()).values()))

    def test_postura_mala_de_perfil_dice_que_corregir(self):
        self.assertEqual(evaluate_posture(fake_pose.bad_side_posture()), "Endereza la espalda")
        aspectos = evaluate_aspects(fake_pose.bad_side_posture())
        self.assertEqual(set(aspectos), {"cabeza_adelantada", "altura_cabeza", "espalda"})
        self.assertFalse(any(aspectos.values()))

    def test_cabeza_adelantada_sola(self):
        pose = fake_pose.make_side_landmarks(ear_forward=0.10, ear_gap=0.12, trunk_lean_deg=3.0)
        self.assertEqual(evaluate_posture(pose), "Retrasa un poco la cabeza")

    def test_de_perfil_no_hay_aspecto_de_hombros(self):
        self.assertNotIn("hombros", evaluate_aspects(fake_pose.good_side_posture()))

    def test_de_frente_sigue_igual(self):
        self.assertEqual(evaluate_posture(fake_pose.good_posture()), "Buena postura")
        self.assertIn("hombros", evaluate_aspects(fake_pose.good_posture()))

    def test_fuera_de_vista_pide_reubicarse(self):
        pose = fake_pose.make_landmarks(shoulder_width=0.05)
        self.assertEqual(evaluate_posture(pose), core.UNMEASURABLE_MESSAGE)

    def test_etiqueta_de_escena(self):
        self.assertTrue(scene_label(fake_pose.good_side_posture()).startswith("De perfil"))
        self.assertTrue(scene_label(fake_pose.good_posture()).startswith("De frente"))


class TestCalibracionPorVista(unittest.TestCase):
    def _metricas_de_perfil(self, n):
        rng = np.random.default_rng(5)
        return [
            core.metrics_from_keypoints(kp_de(fake_pose.make_side_landmarks(
                ear_forward=rng.uniform(0.0, 0.03), ear_gap=rng.uniform(0.11, 0.13),
                trunk_lean_deg=rng.uniform(0, 6))))
            for _ in range(n)
        ]

    def test_calibra_la_vista_en_que_estuvo_y_deja_la_otra_intacta(self):
        calibrador = PostureCalibrator()
        for m in self._metricas_de_perfil(40):
            calibrador.record(m)
        self.assertEqual(calibrador.view, core.VIEW_SIDE)
        self.assertEqual(calibrador.samples, 40)
        base = PostureThresholds()
        sugeridos, _ = calibrador.suggest(base=base, percent_correct=90.0)
        self.assertEqual(sugeridos.shoulder_tilt_max, base.shoulder_tilt_max)  # frente intacto
        self.assertEqual(sugeridos.forward_head_max, base.forward_head_max)
        self.assertNotEqual(sugeridos.side_trunk_lean_max, base.side_trunk_lean_max)

    def test_umbrales_personales_de_perfil(self):
        metricas = self._metricas_de_perfil(40)
        sugeridos, _ = personal_thresholds(metricas, view=core.VIEW_SIDE)
        self.assertIsNotNone(sugeridos)
        marcadas = sum(m["trunk_lean"] > sugeridos.side_trunk_lean_max for m in metricas)
        self.assertEqual(marcadas, 0)

    def test_pocas_muestras_de_una_vista_no_cambian_nada(self):
        metricas = self._metricas_de_perfil(PERSONAL_MIN_SAMPLES - 1)
        sugeridos, nota = personal_thresholds(metricas, view=core.VIEW_SIDE)
        self.assertIsNone(sugeridos)
        self.assertIn("perfil", nota)

    def test_no_mezcla_vistas(self):
        # 40 muestras de perfil no cuentan para los umbrales de frente
        sugeridos, _ = personal_thresholds(self._metricas_de_perfil(40), view=core.VIEW_FRONT)
        self.assertIsNone(sugeridos)


class TestConfigDePerfil(unittest.TestCase):
    def test_los_umbrales_de_perfil_son_configurables(self):
        for clave in ("side_head_forward_max", "side_neck_height_min", "side_trunk_lean_max"):
            self.assertIn(clave, DEFAULTS)
            self.assertIn(clave, THRESHOLD_KEYS)

    def test_set_thresholds_guarda_los_seis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig(path=os.path.join(tmp, "config.json"))
            cfg.set_thresholds(PostureThresholds(side_trunk_lean_max=9.0, side_head_forward_max=0.5))
            recargado = AppConfig.load(os.path.join(tmp, "config.json"))
            self.assertEqual(recargado.thresholds().side_trunk_lean_max, 9.0)
            self.assertEqual(recargado.thresholds().side_head_forward_max, 0.5)

    def test_los_limites_se_respetan(self):
        cfg = AppConfig(path=os.path.join(tempfile.gettempdir(), "no_se_guarda.json"))
        cfg.update({"side_trunk_lean_max": 9999}, save=False)
        self.assertLessEqual(cfg.side_trunk_lean_max, 60.0)


class TestContextoYDataset(unittest.TestCase):
    def test_contexto_del_nombre_de_sesion(self):
        self.assertEqual(core.context_from_group("20261028_212539_correcta_sentado"), "sentado")
        self.assertEqual(core.context_from_group("20261029_101010_incorrecta_perfil_de_pie"), "de_pie")
        self.assertEqual(core.context_from_group("20261028_212157_incorrecta_mixto"), "mixto")
        self.assertEqual(core.context_from_group("legado_12"), core.UNKNOWN_CONTEXT)

    def test_guardar_y_cargar_conserva_vista_y_contexto(self):
        previo = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                kp = kp_de(fake_pose.good_side_posture())
                collect_data.save_dataset([kp], [1], ["s1_correcta_perfil_sentado"], ["perfil"], ["sentado"])
                X, y, groups, views, contexts = collect_data.load_existing()
                self.assertEqual((len(X), views, contexts), (1, ["perfil"], ["sentado"]))
                _, _, _, _, _, ctx = train.load_dataset()
                self.assertEqual(list(ctx), ["sentado"])
            finally:
                os.chdir(previo)

    def test_dataset_viejo_sin_vistas_las_deduce(self):
        previo = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                kps = np.vstack([kp_de(fake_pose.good_posture()), kp_de(fake_pose.good_side_posture())])
                np.savez(core.DATASET_FILE, X=kps, y=np.array([1, 1]), groups=np.array(["a_correcta_sentado", "b"]))
                _, _, _, _, views, contexts = train.load_dataset()
                self.assertEqual(list(views), ["frente", "perfil"])
                self.assertEqual(list(contexts), ["sentado", core.UNKNOWN_CONTEXT])
            finally:
                os.chdir(previo)


class TestEntrenamientoCuatroSituaciones(unittest.TestCase):
    def _cobertura(self, y, views, contexts):
        groups = np.array([f"g{i}" for i in range(len(y))])
        with redirect_stdout(StringIO()) as salida:
            faltan = train.print_coverage(np.array(y), np.array(views), np.array(contexts), groups)
        return faltan, salida.getvalue()

    def test_avisa_de_las_situaciones_que_faltan(self):
        # Solo hay sentado de frente (correcta e incorrecta): faltan las otras tres.
        n = train.MIN_PER_CELL
        y = [1] * n + [0] * n
        faltan, _ = self._cobertura(y, ["frente"] * 2 * n, ["sentado"] * 2 * n)
        texto = " ".join(faltan)
        self.assertIn("de_pie de frente", texto)
        self.assertIn("sentado de perfil", texto)
        self.assertIn("de_pie de perfil", texto)
        self.assertNotIn("sentado de frente", texto.replace("pocas muestras o una sola sesion", ""))

    def test_pocas_sesiones_tambien_se_avisa(self):
        n = train.MIN_PER_CELL
        y = [1] * n + [0] * n
        groups = np.array(["una"] * 2 * n)  # una sola sesion
        with redirect_stdout(StringIO()):
            faltan = train.print_coverage(np.array(y), np.array(["frente"] * 2 * n),
                                          np.array(["sentado"] * 2 * n), groups)
        self.assertTrue(any("sentado de frente" in f and "una sola sesion" in f for f in faltan))

    def test_una_sesion_por_clase_tambien_se_avisa(self):
        """Una pasada de 1-GRABAR-DATOS.bat: 2 sesiones por situacion, pero solo
        una de cada clase, y antes eso contaba como suficiente."""
        n = train.MIN_PER_CELL
        y = np.array([1] * n + [0] * n)
        groups = np.array(["correcta"] * n + ["incorrecta"] * n)
        with redirect_stdout(StringIO()):
            faltan = train.print_coverage(y, np.array(["frente"] * 2 * n), np.array(["sentado"] * 2 * n), groups)
        self.assertTrue(any("sentado de frente" in f and "por clase" in f for f in faltan))

    def test_las_cuatro_situaciones_completas_no_avisan(self):
        n = train.MIN_PER_CELL
        y, views, contexts, groups = [], [], [], []
        for view in core.VIEWS:
            for context in train.CONTEXTS:
                for label in (1, 0):
                    for sesion in range(2):
                        y += [label] * n
                        views += [view] * n
                        contexts += [context] * n
                        groups += [f"{view}{context}{label}{sesion}"] * n
        with redirect_stdout(StringIO()):
            faltan = train.print_coverage(np.array(y), np.array(views), np.array(contexts), np.array(groups))
        self.assertEqual(faltan, [])

    def test_el_aumento_no_espeja_las_muestras_de_perfil(self):
        frente = kp_de(fake_pose.good_posture())
        perfil = kp_de(fake_pose.good_side_posture())
        rng = np.random.default_rng(0)
        _, y_f, _ = train.augment(np.array([frente]), np.array([1]), np.array(["a"]), rng)
        _, y_p, _ = train.augment(np.array([perfil]), np.array([1]), np.array(["b"]), rng)
        self.assertEqual(len(y_f), 3)  # original + espejo + ruido
        self.assertEqual(len(y_p), 2)  # original + ruido (el espejo seria identico)

    def test_el_ruido_es_proporcional_al_tamano(self):
        frente = kp_de(fake_pose.good_posture())
        perfil = kp_de(fake_pose.good_side_posture())
        rng = np.random.default_rng(1)
        ruido_f = np.std([core.jitter_relative(frente, rng=rng) - frente for _ in range(200)])
        ruido_p = np.std([core.jitter_relative(perfil, rng=rng) - perfil for _ in range(200)])
        self.assertLess(ruido_p, ruido_f / 2)  # la cabeza mide mucho menos que los hombros

    def test_entrena_con_las_dos_vistas_y_las_distingue(self):
        # Mini dataset sintetico con las 4 situaciones; el modelo debe separar buena de mala en ambas vistas.
        rng = np.random.default_rng(7)
        X, y = [], []
        for _ in range(40):
            X.append(kp_de(fake_pose.make_landmarks(head_forward=0.0 + rng.uniform(0, 0.02), ear_gap=rng.uniform(0.11, 0.13),
                                                    shoulder_tilt=rng.uniform(0, 0.01), center_x=rng.uniform(0.45, 0.55)))); y.append(1)
            X.append(kp_de(fake_pose.make_landmarks(head_forward=0.25, ear_gap=0.04, shoulder_tilt=0.10,
                                                    center_x=rng.uniform(0.45, 0.55)))); y.append(0)
            X.append(kp_de(fake_pose.make_side_landmarks(ear_forward=rng.uniform(0, 0.03), ear_gap=rng.uniform(0.11, 0.13),
                                                         trunk_lean_deg=rng.uniform(0, 6), facing=rng.choice([-1, 1])))); y.append(1)
            X.append(kp_de(fake_pose.make_side_landmarks(ear_forward=rng.uniform(0.09, 0.12), ear_gap=rng.uniform(0.06, 0.08),
                                                         trunk_lean_deg=rng.uniform(18, 25), facing=rng.choice([-1, 1])))); y.append(0)
        F, keep = core.features_matrix(np.array(X))
        self.assertEqual(len(keep), len(X))
        y = np.array(y)[keep]
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(F)
        modelo = MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=0).fit(sc.transform(F), y)
        pred_f = modelo.predict(sc.transform(core.features_matrix(np.array([kp_de(fake_pose.good_posture()), kp_de(fake_pose.bad_posture())]))[0]))
        pred_p = modelo.predict(sc.transform(core.features_matrix(np.array([kp_de(fake_pose.good_side_posture()), kp_de(fake_pose.bad_side_posture())]))[0]))
        self.assertEqual(list(pred_f), [1, 0])
        self.assertEqual(list(pred_p), [1, 0])


if __name__ == "__main__":
    unittest.main()
