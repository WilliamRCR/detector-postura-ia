"""Tests del reto de la feria: contador, estatua, ranking, audio completo y el
juego de principio a fin, sin camara, ventana ni bocina."""
import math
import os
import pathlib
import random
import tempfile
import unittest
from datetime import date
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import core
import reto
import reto_frases
from reto_audio import CARPETA, Audio
from reto_logica import (
    ASPECTO,
    NIVELES,
    POSES,
    RETO_ESTATUA,
    RETO_ESTRELLAS,
    RETO_SENTADILLAS,
    SEGUNDOS_ESTRELLAS,
    SEGUNDOS_SENTADILLAS,
    Aprendizaje,
    ContadorSentadillas,
    JuegoEstrellas,
    JuezEstatua,
    Objetivo,
    Ranking,
    diferencia_angulos,
    elegir_consejos,
    evaluar_pose,
    medir_cuerpo,
    medir_segmentos,
)
from tests.fake_pose import FakeLandmark, make_landmarks, make_side_landmarks

NIVEL = NIVELES["Normal"]


def persona(baja=0.0, sin_rodillas=False, x=0.5, hombros_extra=0.0):
    """Persona de frente que baja la cadera una fraccion `baja` del muslo."""
    muslo, torso = 0.22, 0.28
    cadera = 0.45 + baja * muslo
    hombros = cadera - torso + hombros_extra
    puntos = make_landmarks(shoulder_y=hombros, hip_y=cadera, center_x=x, ear_gap=0.10, legs_visible=True)
    for i in (25, 26):
        puntos[i] = FakeLandmark(puntos[i].x, 0.67, 0.0, 0.1 if sin_rodillas else 1.0)
    for i in (15, 16):
        puntos[i] = FakeLandmark(x + (0.12 if i == 15 else -0.12), hombros + 0.3, 0.0, 1.0)
    return puntos


def en_pose(pose, error=0.0, x=0.5):
    """Persona haciendo `pose`, con todos sus segmentos desviados `error` grados."""
    p = persona(x=x)
    izq, der = (11, 13, 15), (12, 14, 16)
    if p[11].x > p[12].x:
        izq, der = der, izq

    def extender(a, indice, grados, largo):
        r = math.radians(grados + error)
        p[indice] = FakeLandmark(a.x + largo * math.cos(r) / ASPECTO, a.y + largo * math.sin(r), 0.0, 1.0)
        return p[indice]

    s = pose.segmentos
    for (hombro, codo, muneca), (brazo, ante) in ((izq, ("bi", "ai")), (der, ("bd", "ad"))):
        extender(extender(p[hombro], codo, s[brazo], 0.12), muneca, s[ante], 0.11)
    cadera_i, cadera_d = (23, 24) if p[23].x <= p[24].x else (24, 23)
    for cadera, rodilla, muslo in ((cadera_i, 25 if cadera_i == 23 else 26, "mi"), (cadera_d, 26 if cadera_d == 24 else 25, "md")):
        if muslo in s:
            extender(p[cadera], rodilla, s[muslo], 0.2)
    return p


def repeticion(contador, cuerpo_de, picos, t, paso=0.05):
    """Baja hasta `pico` y vuelve a subir. Devuelve (eventos, t final)."""
    eventos = []
    for pico in picos:
        bajada = list(np.linspace(0, pico, 12)) + list(np.linspace(pico, 0, 12)) + [0.0] * 3
        for p in bajada:
            eventos += contador.actualizar(cuerpo_de(p), t)
            t += paso
    return eventos, t


class TestSentadillas(unittest.TestCase):
    def setUp(self):
        self.c = ContadorSentadillas(NIVEL)
        self.c.calibrar(medir_cuerpo(persona()))
        self.de = lambda p: medir_cuerpo(persona(p))

    def test_cuenta_repeticiones_perfectas(self):
        eventos, _ = repeticion(self.c, self.de, [1.0] * 5, 0.0)
        self.assertEqual((self.c.total, self.c.perfectas), (5, 5))
        self.assertEqual(eventos.count("perfecta"), 5)

    def test_una_a_medias_cuenta_pero_no_es_perfecta(self):
        eventos, _ = repeticion(self.c, self.de, [0.6], 0.0)
        self.assertEqual((self.c.total, self.c.perfectas), (1, 0))
        self.assertIn("media", eventos)

    def test_si_no_baja_lo_suficiente_no_cuenta_y_lo_corrige(self):
        eventos, _ = repeticion(self.c, self.de, [0.35], 0.0)
        self.assertEqual(self.c.total, 0)
        self.assertIn("baja", eventos)

    def test_quedarse_de_pie_no_cuenta_nada(self):
        rng = np.random.default_rng(0)
        eventos = []
        for i in range(600):
            eventos += self.c.actualizar(self.de(float(rng.normal(0, 0.03))), i * 0.033)
        self.assertEqual((self.c.total, eventos), (0, []))

    def test_inclinarse_sin_doblar_las_piernas_no_cuenta(self):
        """Agacharse como reverencia baja los hombros, no la cadera."""
        for i in range(200):
            extra = 0.14 * abs(np.sin(i / 15))
            self.c.actualizar(medir_cuerpo(persona(0.0, hombros_extra=extra)), i * 0.033)
        self.assertEqual(self.c.total, 0)

    def test_funciona_sin_ver_las_rodillas(self):
        """Stand con poco espacio: la cadera se ve, las rodillas no."""
        c = ContadorSentadillas(NIVEL)
        cuerpo = medir_cuerpo(persona(sin_rodillas=True))
        self.assertFalse(cuerpo.rodillas_visibles)
        c.calibrar(cuerpo)
        repeticion(c, lambda p: medir_cuerpo(persona(p, sin_rodillas=True)), [1.0] * 3, 0.0)
        self.assertEqual(c.total, 3)

    def test_moverse_de_lado_no_cuenta(self):
        for i in range(200):
            self.c.actualizar(medir_cuerpo(persona(0.0, x=0.5 + 0.1 * np.sin(i / 10))), i * 0.033)
        self.assertEqual(self.c.total, 0)

    def test_perder_a_la_persona_no_rompe_ni_cuenta(self):
        self.assertEqual(self.c.actualizar(None, 1.0), [])
        self.assertEqual(self.c.total, 0)

    def test_un_temblor_no_cuenta_doble(self):
        c = ContadorSentadillas(NIVEL)
        c.calibrar(medir_cuerpo(persona()))
        repeticion(c, self.de, [1.0, 1.0], 0.0, paso=0.005)
        self.assertEqual(c.total, 1)

    def test_el_porcentaje_sube_con_la_profundidad(self):
        self.assertEqual(self.c.porcentaje_actual, 0.0)
        valores = []
        for p in (0.2, 0.4, 0.6, 0.9):
            for _ in range(10):
                self.c.actualizar(self.de(p), 0.0)
            valores.append(self.c.porcentaje_actual)
        self.assertEqual(valores, sorted(valores))
        self.assertEqual(valores[-1], 100.0)

    def test_el_resumen_dice_cuanto_salio_bien_y_mal(self):
        repeticion(self.c, self.de, [1.0, 1.0, 1.0, 0.6, 0.35], 0.0)
        self.assertEqual((self.c.perfectas, self.c.medias, self.c.fallidas), (3, 1, 1))
        self.assertEqual(self.c.intentos, 5)
        self.assertAlmostEqual(self.c.porcentaje_bien, 60.0)
        self.assertGreater(self.c.calidad_media, 50.0)

    def test_sin_intentos_el_resumen_no_divide_entre_cero(self):
        self.assertEqual(self.c.porcentaje_bien, 0.0)
        self.assertEqual(self.c.calidad_media, 0.0)

    def test_el_nivel_dificil_exige_mas(self):
        facil = ContadorSentadillas(NIVELES["Fácil"])
        dificil = ContadorSentadillas(NIVELES["Difícil"])
        for c in (facil, dificil):
            c.calibrar(medir_cuerpo(persona()))
            repeticion(c, self.de, [0.7], 0.0)
        self.assertEqual(facil.perfectas, 1)
        self.assertEqual(dificil.perfectas, 0)


class TestTecnicaSentadillas(unittest.TestCase):
    def perfil(self, baja=0.0, lean=3.0, ear_forward=0.01, ear_gap=0.12):
        p = make_side_landmarks(trunk_lean_deg=lean, ear_forward=ear_forward, ear_gap=ear_gap)
        for lm in (p[23], p[24]):
            lm.y += baja * 0.2
        for i in (25, 26):
            p[i] = FakeLandmark(p[23].x, 0.97, 0.0, 1.0)
        return p

    def contador_de_perfil(self):
        c = ContadorSentadillas(NIVEL)
        c.calibrar(medir_cuerpo(self.perfil()))
        return c

    def test_de_perfil_detecta_el_tronco_muy_inclinado(self):
        c = self.contador_de_perfil()
        eventos, _ = repeticion(c, lambda p: medir_cuerpo(self.perfil(p, lean=70 * p)), [1.0], 0.0)
        self.assertIn("pecho", eventos)
        self.assertEqual(c.errores["espalda"], 1)
        self.assertEqual(c.perfectas, 0)

    def test_de_perfil_detecta_la_espalda_encorvada(self):
        c = self.contador_de_perfil()
        eventos, _ = repeticion(c, lambda p: medir_cuerpo(self.perfil(p, lean=20 * p, ear_forward=0.14 * p, ear_gap=0.12 - 0.06 * p)), [1.0], 0.0)
        self.assertIn("pecho", eventos)
        self.assertEqual(c.errores["encorvada"], 1)

    def test_de_perfil_una_buena_sentadilla_no_marca_nada(self):
        c = self.contador_de_perfil()
        repeticion(c, lambda p: medir_cuerpo(self.perfil(p, lean=25 * p)), [1.0], 0.0)
        self.assertEqual((c.perfectas, c.errores["espalda"], c.errores["encorvada"]), (1, 0, 0))

    def test_de_frente_detecta_las_rodillas_juntas(self):
        def pose(p):
            lm = persona(p)
            lm[27], lm[28] = FakeLandmark(0.40, 0.9, 0, 1.0), FakeLandmark(0.60, 0.9, 0, 1.0)
            cierre = 0.09 * p
            lm[25], lm[26] = FakeLandmark(0.40 + cierre, 0.67, 0, 1.0), FakeLandmark(0.60 - cierre, 0.67, 0, 1.0)
            return medir_cuerpo(lm)

        c = ContadorSentadillas(NIVEL)
        c.calibrar(pose(0.0))
        eventos, _ = repeticion(c, pose, [1.0], 0.0)
        self.assertIn("rodillas", eventos)
        self.assertEqual(c.errores["rodillas"], 1)

    def test_avisa_si_va_demasiado_rapido(self):
        c = ContadorSentadillas(NIVEL)
        c.calibrar(medir_cuerpo(persona()))
        eventos, _ = repeticion(c, lambda p: medir_cuerpo(persona(p)), [1.0], 0.0, paso=0.02)
        self.assertIn("rapido", eventos)
        self.assertEqual(c.errores["ritmo"], 1)


class TestConsejos(unittest.TestCase):
    def test_ordena_por_lo_que_mas_se_repitio(self):
        self.assertEqual(elegir_consejos({"a": 2, "b": 8, "c": 5}, 10)[:2], ["b", "c"])

    def test_algo_que_paso_una_vez_no_merece_consejo(self):
        self.assertEqual(elegir_consejos({"a": 1}, 20), [])

    def test_maximo_tres(self):
        self.assertEqual(len(elegir_consejos({"a": 5, "b": 5, "c": 5, "d": 5}, 10)), 3)

    def test_sin_errores_no_hay_consejos(self):
        self.assertEqual(elegir_consejos({}, 0), [])

    def test_cada_consejo_tiene_texto_y_voz(self):
        for clave in reto_frases.CONSEJOS:
            for datos in reto_frases.ENTRENADORES.values():
                self.assertIn(f"tip_{clave}", datos["frases"])
        for datos in reto_frases.ENTRENADORES.values():
            self.assertIn("tip_ok", datos["frases"])


class TestFrases(unittest.TestCase):
    def test_las_frases_comunes_tienen_varias_versiones(self):
        for datos in reto_frases.ENTRENADORES.values():
            for clave, frases in datos["frases"].items():
                if clave not in ("tres", "dos", "uno") and not clave.startswith("tip_"):
                    self.assertGreaterEqual(len(frases), 5, clave)
                    self.assertEqual(len(set(frases)), len(frases), clave)

    def test_no_repite_la_misma_frase_seguida(self):
        with tempfile.TemporaryDirectory():
            audio = Audio()
            audio.elegir_entrenador("zen")
            anterior = None
            for _ in range(50):
                audio._frase("bien")
                actual = audio._ultima["bien"]
                self.assertNotEqual(actual, anterior)
                anterior = actual
            pygame.mixer.quit()


class TestPoses(unittest.TestCase):
    def test_hay_al_menos_treinta_poses_distintas(self):
        self.assertGreaterEqual(len(POSES), 30)
        self.assertEqual(len({p.nombre for p in POSES}), len(POSES))

    def test_cada_pose_coincide_consigo_misma(self):
        for pose in POSES:
            cumple, porcentaje, bien = evaluar_pose(en_pose(pose), pose, 10.0)
            self.assertTrue(cumple, pose.nombre)
            self.assertGreater(porcentaje, 95.0)
            self.assertTrue(all(bien.values()))

    def test_las_poses_se_distinguen_entre_si(self):
        for pose in POSES:
            for otra in POSES:
                if pose is not otra and set(otra.segmentos) == set(pose.segmentos):
                    self.assertFalse(evaluar_pose(en_pose(pose), otra, 20.0)[0], (pose.nombre, otra.nombre))

    def test_sirve_para_cualquier_lugar_del_cuadro(self):
        pose = POSES[0]
        self.assertTrue(evaluar_pose(en_pose(pose, x=0.3), pose, 10.0)[0])

    def test_una_parte_mal_se_marca_sola(self):
        pose = POSES[0]
        p = en_pose(pose)
        p[15].y -= 0.15
        cumple, _, bien = evaluar_pose(p, pose, 20.0)
        self.assertFalse(cumple)
        self.assertEqual(sum(1 for v in bien.values() if not v), 1)

    def pose_con_brazo_abajo(self):
        return next(p for p in POSES if p.nombre == "Mano arriba")

    def lado_abajo(self, p):
        """Indices (codo, muneca) del brazo que cuelga en la pose."""
        return (13, 15) if p[11].x <= p[12].x else (14, 16)

    def test_un_brazo_pegado_al_cuerpo_y_algo_doblado_cuenta(self):
        """Con el brazo colgando, la muneca sale torcida hacia la cadera."""
        pose = self.pose_con_brazo_abajo()
        p = en_pose(pose)
        codo, muneca = self.lado_abajo(p)
        p[muneca].x = p[codo].x + 0.07
        self.assertTrue(evaluar_pose(p, pose, NIVEL.tolerancia)[0])

    def test_un_brazo_pegado_al_cuerpo_tapado_cuenta(self):
        pose = self.pose_con_brazo_abajo()
        p = en_pose(pose)
        _, muneca = self.lado_abajo(p)
        p[muneca].visibility = 0.1
        self.assertTrue(evaluar_pose(p, pose, NIVEL.tolerancia)[0])

    def test_el_brazo_que_va_arriba_sigue_siendo_exigente(self):
        pose = self.pose_con_brazo_abajo()
        p = en_pose(pose)
        codo, muneca = (14, 16) if self.lado_abajo(p) == (13, 15) else (13, 15)
        p[muneca].x = p[codo].x + 0.07
        self.assertFalse(evaluar_pose(p, pose, NIVEL.tolerancia)[0])

    def test_el_brazo_abajo_tampoco_vale_en_cualquier_lado(self):
        pose = self.pose_con_brazo_abajo()
        p = en_pose(pose)
        codo, muneca = self.lado_abajo(p)
        p[muneca].x, p[muneca].y = p[codo].x - 0.1, p[codo].y
        self.assertFalse(evaluar_pose(p, pose, NIVEL.tolerancia)[0])

    def test_diferencia_de_angulos_da_la_vuelta(self):
        self.assertEqual(diferencia_angulos(350, 10), 20)

    def test_sin_codos_a_la_vista_no_cumple(self):
        pose = POSES[0]
        p = en_pose(pose)
        p[13].visibility = 0.1
        self.assertFalse(evaluar_pose(p, pose, 30.0)[0])
        self.assertIsNone(medir_segmentos(p)["bi"] if p[11].x <= p[12].x else medir_segmentos(p)["bd"])


class TestDificultad(unittest.TestCase):
    def test_cada_nivel_vigila_mas_puntos(self):
        cuantos = [len(NIVELES[n].puntos) for n in ("Fácil", "Normal", "Difícil")]
        self.assertEqual(cuantos, [9, 13, 19])

    def test_cada_nivel_exige_mas(self):
        f, n, d = (NIVELES[k] for k in ("Fácil", "Normal", "Difícil"))
        self.assertGreater(f.tolerancia, n.tolerancia)
        self.assertGreater(n.tolerancia, d.tolerancia)
        self.assertGreater(f.quieto, d.quieto)
        self.assertLess(f.aguantar, d.aguantar)
        self.assertGreater(f.limite, d.limite)
        self.assertGreater(f.preparar, d.preparar)

    def test_los_puntos_existen_en_mediapipe(self):
        for nivel in NIVELES.values():
            self.assertTrue(all(0 <= i < core.NUM_LANDMARKS for i in nivel.puntos))
            self.assertEqual(len(set(nivel.puntos)), len(nivel.puntos))

    def test_cada_nivel_tiene_poses(self):
        for nivel in NIVELES.values():
            self.assertGreaterEqual(len(JuezEstatua(nivel, piernas=False).opciones), 3)

    def test_solo_el_nivel_dificil_pide_piernas(self):
        self.assertTrue(any(p.grupo == "piernas" for p in JuezEstatua(NIVELES["Difícil"]).opciones))
        self.assertFalse(any(p.grupo == "piernas" for p in JuezEstatua(NIVELES["Normal"]).opciones))

    def test_sin_rodillas_a_la_vista_no_pide_piernas(self):
        opciones = JuezEstatua(NIVELES["Difícil"], piernas=False).opciones
        self.assertFalse(any(p.grupo == "piernas" for p in opciones))


class TestEstatua(unittest.TestCase):
    def setUp(self):
        self.j = JuezEstatua(NIVEL, rng=random.Random(4))
        self.t = 0.0
        self.j.iniciar(self.t)

    def correr(self, poses, paso=0.05):
        eventos = []
        for p in poses:
            self.t += paso
            eventos += self.j.actualizar(p, self.t)
        return eventos

    def segundos(self, s):
        return int(s / 0.05) + 1

    def preparar(self):
        return self.correr([persona()] * self.segundos(self.j.restante(self.t)))

    def llegar(self):
        self.preparar()
        return self.correr([en_pose(self.j.pose)] * 10)

    def test_empieza_mostrando_la_pose_sin_evaluar(self):
        self.assertEqual(self.j.fase, "preparar")
        self.assertIsNotNone(self.j.pose)

    def test_en_la_preparacion_nada_cuenta_como_error(self):
        eventos = self.correr([persona(x=0.3)] * 20 + [None] * 20)
        self.assertEqual(eventos, [])
        self.assertEqual(self.j.vidas, JuezEstatua.VIDAS)

    def test_ya_estar_en_pose_no_adelanta_la_preparacion(self):
        self.correr([en_pose(self.j.pose)] * 10)
        self.assertEqual(self.j.fase, "preparar")

    def test_la_preparacion_muestra_que_tan_cerca_va(self):
        self.correr([en_pose(self.j.pose)] * 3)
        self.assertGreater(self.j.coincidencia, 90.0)

    def test_al_terminar_la_preparacion_empieza_el_tiempo(self):
        self.assertEqual(self.preparar(), ["imita"])
        self.assertEqual(self.j.fase, "imitar")

    def test_cada_pose_nueva_tiene_su_preparacion(self):
        self.llegar()
        self.correr([en_pose(self.j.pose)] * self.segundos(NIVEL.aguantar))
        self.assertEqual(self.j.fase, "preparar")

    def test_tras_perder_una_vida_tambien_se_prepara(self):
        self.preparar()
        self.correr([persona()] * self.segundos(NIVEL.limite))
        self.assertEqual(self.j.fase, "preparar")

    def test_imitar_la_pose_empieza_a_contar(self):
        self.assertIn("aguanta", self.llegar())
        self.assertEqual(self.j.fase, "aguantar")

    def test_una_pose_mal_hecha_no_cuenta(self):
        self.preparar()
        self.assertEqual(self.correr([persona()] * 20), [])
        self.assertEqual(self.j.fase, "imitar")

    def test_no_llegar_a_tiempo_quita_una_vida(self):
        self.preparar()
        eventos = self.correr([persona()] * self.segundos(NIVEL.limite))
        self.assertEqual(eventos, ["vida"])
        self.assertEqual(self.j.vidas, JuezEstatua.VIDAS - 1)

    def test_aguantar_quieto_completa_la_pose(self):
        self.llegar()
        pose = self.j.pose
        eventos = self.correr([en_pose(pose)] * self.segundos(NIVEL.aguantar))
        self.assertEqual(eventos, ["pose"])
        self.assertEqual(self.j.poses, 1)
        self.assertIsNot(self.j.pose, pose)
        self.assertGreater(self.j.precision, 90.0)

    def test_quieto_con_ruido_normal_no_pierde_vidas(self):
        rng = np.random.default_rng(3)
        self.llegar()
        pose = self.j.pose
        poses = []
        for _ in range(self.segundos(NIVEL.aguantar)):
            p = en_pose(pose)
            for lm in p:
                lm.x += float(rng.normal(0, 0.003))
                lm.y += float(rng.normal(0, 0.003))
            poses.append(p)
        self.assertEqual(self.correr(poses), ["pose"])
        self.assertEqual(self.j.vidas, JuezEstatua.VIDAS)

    def test_moverse_quita_una_vida(self):
        self.llegar()
        pose = self.j.pose
        self.assertEqual(self.correr([en_pose(pose, x=0.56)] * 12), ["vida"])

    def test_un_instante_de_mala_deteccion_no_quita_vida(self):
        self.llegar()
        pose = self.j.pose
        eventos = self.correr([en_pose(pose, x=0.6)] * 2 + [en_pose(pose)] * 10)
        self.assertEqual(eventos, [])

    def test_romper_la_pose_quita_una_vida(self):
        self.llegar()
        self.assertEqual(self.correr([persona()] * 12), ["vida"])

    def test_en_dificil_se_nota_un_movimiento_que_en_facil_no(self):
        def prueba(nombre):
            j = JuezEstatua(NIVELES[nombre], piernas=False, rng=random.Random(1))
            j.iniciar(0.0)
            pose = j.pose
            t = 0.0
            while j.fase != "aguantar":
                t += 0.05
                j.actualizar(en_pose(pose), t)
            for _ in range(10):
                t += 0.05
                j.actualizar(en_pose(pose, x=0.535), t)
            return j.vidas

        self.assertEqual(prueba("Fácil"), JuezEstatua.VIDAS)
        self.assertEqual(prueba("Difícil"), JuezEstatua.VIDAS - 1)

    def test_perder_todas_las_vidas_termina(self):
        eventos = []
        for _ in range(JuezEstatua.VIDAS):
            self.preparar()
            eventos += self.correr([persona()] * self.segundos(NIVEL.limite))
        self.assertTrue(self.j.terminado)
        self.assertEqual(eventos[-1], "caida")

    def test_salirse_de_cuadro_mientras_aguanta_cuenta(self):
        self.llegar()
        self.assertEqual(self.correr([None] * 30), ["vida"])

    def test_cada_pose_es_mas_dura(self):
        self.llegar()
        primera = self.j.restante(self.t)
        self.correr([en_pose(self.j.pose)] * self.segundos(NIVEL.aguantar))
        self.llegar()
        self.assertGreater(self.j.restante(self.t), primera)

    def test_completar_todas_las_poses_gana(self):
        eventos = []
        while not self.j.terminado:
            eventos += self.llegar()
            eventos += self.correr([en_pose(self.j.pose)] * self.segundos(self.j.restante(self.t)))
        self.assertEqual(self.j.poses, JuezEstatua.MAX_POSES)
        self.assertEqual(eventos[-1], "gana")
        self.assertEqual(self.j.vidas, JuezEstatua.VIDAS)


class TestEstrellas(unittest.TestCase):
    def setUp(self):
        self.e = JuegoEstrellas(NIVEL, rng=random.Random(2))

    def poner(self, tipo, x=0.3, y=0.3):
        o = Objetivo(x, y, tipo, nace=0.0, muere=2.0, radio=0.07)
        self.e.objetivos.append(o)
        return o

    def mano_en(self, x, y, indice=15):
        p = persona()
        p[indice] = FakeLandmark(x, y, 0.0, 1.0)
        return p

    def test_tocar_una_estrella_da_un_punto(self):
        self.poner("estrella")
        self.assertIn("estrella", self.e.actualizar(self.mano_en(0.3, 0.3), 0.5))
        self.assertEqual((self.e.puntos, self.e.aciertos), (1, 1))
        self.assertEqual(self.e.reacciones, [0.5])

    def test_la_dorada_vale_tres(self):
        self.poner("dorada")
        self.e.actualizar(self.mano_en(0.3, 0.3), 0.5)
        self.assertEqual(self.e.puntos, 3)

    def test_la_bomba_resta_y_corta_la_racha(self):
        self.e.puntos, self.e.racha = 5, 3
        self.poner("bomba")
        self.assertIn("bomba", self.e.actualizar(self.mano_en(0.3, 0.3), 0.5))
        self.assertEqual((self.e.puntos, self.e.racha), (3, 0))

    def test_el_puntaje_no_baja_de_cero(self):
        self.poner("bomba")
        self.e.actualizar(self.mano_en(0.3, 0.3), 0.5)
        self.assertEqual(self.e.puntos, 0)

    def test_dejarla_pasar_corta_la_racha(self):
        self.e.racha = 4
        self.poner("estrella")
        self.assertIn("perdida", self.e.actualizar(persona(), 2.5))
        self.assertEqual((self.e.racha, self.e.perdidas), (0, 1))

    def test_una_bomba_que_se_va_no_cuenta_como_perdida(self):
        self.poner("bomba")
        self.assertEqual(self.e.actualizar(persona(), 2.5), [])

    def test_cada_cinco_seguidas_hay_racha(self):
        eventos = []
        for i in range(5):
            self.poner("estrella")
            eventos += self.e.actualizar(self.mano_en(0.3, 0.3), 0.5 + i * 0.1)
        self.assertEqual(eventos.count("racha"), 1)

    def test_salen_estrellas_dentro_de_la_imagen(self):
        for i in range(400):
            self.e.actualizar(persona(), i * 0.3)
            for o in self.e.objetivos:
                self.assertTrue(0.05 <= o.x <= 0.95 and 0.05 <= o.y <= 0.95)
            self.e.objetivos.clear()

    def test_sin_persona_no_salen_estrellas(self):
        self.assertEqual(self.e.actualizar(None, 1.0), [])
        self.assertEqual(self.e.objetivos, [])

    def test_avisa_si_usa_una_sola_mano(self):
        self.e.por_mano = {"izquierda": 9, "derecha": 1}
        self.e.aciertos = 10
        self.assertGreater(self.e.errores["un_lado"], 0)

    def test_avisa_si_reacciona_lento(self):
        self.e.reacciones = [1.9, 2.0]
        self.e.aciertos = 2
        self.assertGreater(self.e.errores["reaccion"], 0)

    def test_precision_sin_intentos(self):
        self.assertEqual(self.e.precision, 0.0)


class TestAprendizaje(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.ruta = pathlib.Path(self._dir.name) / "a.json"
        self.a = Aprendizaje(self.ruta)

    def tearDown(self):
        self._dir.cleanup()

    def intentos(self, nombre, logros, fallos):
        for _ in range(logros):
            self.a.registrar_pose(nombre, True)
        for _ in range(fallos):
            self.a.registrar_pose(nombre, False)

    def test_sin_datos_no_cambia_nada(self):
        self.assertEqual(self.a.ajuste_tolerancia("Cruz"), 0.0)
        self.assertEqual(self.a.nivel_ajustado(NIVEL), NIVEL)

    def test_una_pose_dificil_da_mas_margen(self):
        self.intentos("Cruz", 1, 9)
        self.assertGreater(self.a.ajuste_tolerancia("Cruz"), 0)

    def test_una_pose_facil_da_menos_margen(self):
        self.intentos("Cruz", 10, 0)
        self.assertLess(self.a.ajuste_tolerancia("Cruz"), 0)

    def test_el_ajuste_tiene_limites(self):
        self.intentos("Cruz", 0, 100)
        self.intentos("Aro", 100, 0)
        self.assertLessEqual(self.a.ajuste_tolerancia("Cruz"), Aprendizaje.AJUSTE_MAX)
        self.assertGreaterEqual(self.a.ajuste_tolerancia("Aro"), Aprendizaje.AJUSTE_MIN)

    def test_las_poses_sin_datos_salen_mas(self):
        self.intentos("Cruz", 5, 5)
        self.assertGreater(self.a.peso("Aro"), self.a.peso("Cruz"))

    def test_se_adapta_a_una_camara_que_mide_menos(self):
        for _ in range(30):
            self.a.registrar_profundidad(0.62)
        self.assertLess(self.a.nivel_ajustado(NIVEL).perfecta, NIVEL.perfecta)
        self.assertGreaterEqual(self.a.nivel_ajustado(NIVEL).perfecta, NIVEL.perfecta - Aprendizaje.MAX_CAMBIO_PERFECTA)

    def test_nunca_exige_menos_que_contar_la_repeticion(self):
        for _ in range(30):
            self.a.registrar_profundidad(0.1)
        self.assertGreater(self.a.nivel_ajustado(NIVEL).perfecta, NIVEL.bajar)

    def test_el_movimiento_tolerado_tiene_limites(self):
        for _ in range(30):
            self.a.registrar_movimiento(5.0)
        self.assertLessEqual(self.a.nivel_ajustado(NIVEL).quieto, NIVEL.quieto * Aprendizaje.QUIETO_MAX)

    def test_se_guarda_y_se_recupera(self):
        self.intentos("Cruz", 3, 2)
        self.a.terminar_partida()
        otra = Aprendizaje(self.ruta)
        self.assertEqual(otra.partidas, 1)
        self.assertEqual(otra.datos["poses"]["Cruz"]["intentos"], 5)

    def test_un_archivo_danado_no_rompe(self):
        self.ruta.write_text("{ esto no es json", encoding="utf-8")
        self.assertEqual(Aprendizaje(self.ruta).partidas, 0)

    def test_borrar_olvida_todo(self):
        self.a.terminar_partida()
        self.a.borrar()
        self.assertEqual(self.a.partidas, 0)
        self.assertFalse(self.ruta.exists())

    def test_el_juez_aprende_de_cada_pose(self):
        juez = JuezEstatua(NIVEL, rng=random.Random(4), aprendizaje=self.a)
        juez.iniciar(0.0)
        juez.actualizar(persona(), NIVEL.preparar + 0.1)
        juez.actualizar(persona(), NIVEL.preparar + NIVEL.limite + 1)
        self.assertEqual(sum(p["intentos"] for p in self.a.datos["poses"].values()), 1)


class TestRanking(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.r = Ranking(pathlib.Path(self._dir.name) / "r.db")

    def tearDown(self):
        self._dir.cleanup()

    def test_ordena_y_da_la_posicion(self):
        self.assertEqual(self.r.guardar(RETO_SENTADILLAS, "Ana", 10), 1)
        self.assertEqual(self.r.guardar(RETO_SENTADILLAS, "Luis", 15), 1)
        self.assertEqual(self.r.guardar(RETO_SENTADILLAS, "Eva", 12), 2)
        self.assertEqual([f["apodo"] for f in self.r.top(RETO_SENTADILLAS)], ["Luis", "Eva", "Ana"])
        self.assertEqual(self.r.record(RETO_SENTADILLAS), 15)

    def test_el_empate_lo_decide_el_total(self):
        self.r.guardar(RETO_SENTADILLAS, "Ana", 10, extra=11)
        self.assertEqual(self.r.guardar(RETO_SENTADILLAS, "Luis", 10, extra=14), 1)

    def test_cada_reto_tiene_su_tabla(self):
        self.r.guardar(RETO_SENTADILLAS, "Ana", 10)
        self.assertEqual(self.r.top(RETO_ESTATUA), [])
        self.assertIsNone(self.r.record(RETO_ESTATUA))

    def test_solo_cuenta_el_dia(self):
        self.r.guardar(RETO_SENTADILLAS, "Ayer", 99, dia=date(2026, 10, 29))
        self.assertEqual(self.r.top(RETO_SENTADILLAS, dia=date(2026, 10, 30)), [])
        self.assertEqual(self.r.jugadas(dia=date(2026, 10, 29)), 1)

    def test_borrar(self):
        self.r.guardar(RETO_ESTATUA, "Ana", 10)
        self.r.borrar()
        self.assertEqual(self.r.jugadas(), 0)


class TestAudioCompleto(unittest.TestCase):
    def test_hay_un_archivo_por_cada_frase(self):
        for nombre in reto_frases.ENTRENADORES:
            for clave in reto_frases.textos(nombre):
                self.assertTrue((CARPETA / nombre / f"{clave}.ogg").is_file(), (nombre, clave))

    def test_hay_todos_los_sonidos(self):
        for nombre in ("ding", "media", "fallo", "tick", "go", "fanfarria", "aplausos", "musica_reto", "musica_estatua"):
            self.assertTrue((CARPETA / "sonidos" / f"{nombre}.ogg").is_file(), nombre)

    def test_los_tres_entrenadores_tienen_las_mismas_frases(self):
        claves = [set(reto_frases.textos(n)) for n in reto_frases.ENTRENADORES]
        self.assertEqual(claves[0], claves[1])
        self.assertEqual(claves[1], claves[2])

    def test_la_voz_cuenta_rapido(self):
        """Un numero largo atrasaria el conteo de las sentadillas."""
        pygame.mixer.init(44100)
        self.addCleanup(pygame.mixer.quit)
        for nombre in reto_frases.ENTRENADORES:
            for n in range(1, 21):
                ruta = CARPETA / nombre / f"{reto_frases.clave_numero(n)}.ogg"
                self.assertLess(pygame.mixer.Sound(str(ruta)).get_length(), 1.6, (nombre, n))

    def test_sin_archivos_no_rompe(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Audio(tmp)
            audio.elegir_entrenador("zen")
            audio.decir("hola")
            audio.efecto("ding")
            audio.musica("musica_reto")
            pygame.mixer.quit()


class AudioEspia:
    def __init__(self):
        self.dicho = []
        self.efectos = []
        self.musicas = []

    def elegir_entrenador(self, nombre):
        pass

    def decir(self, clave, ahora_mismo=False):
        self.dicho.append(clave)

    def actualizar(self):
        pass

    def efecto(self, nombre):
        self.efectos.append(nombre)

    def musica(self, nombre):
        self.musicas.append(nombre)

    def parar_musica(self):
        pass

    def silenciar_voz(self):
        pass


class CamaraFalsa:
    error = ""
    aviso = "Buscando tu cámara..."
    indice = 1

    def __init__(self):
        self.puntos = persona()
        self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cambios = 0

    def leer(self):
        return self.rgb, self.puntos

    def siguiente(self):
        self.cambios += 1


class CapturaFalsa:
    """Camara de OpenCV: fija como una virtual, o con ruido como una real."""

    def __init__(self, real, abre=True):
        self.real = real
        self.abre = abre
        self.rng = np.random.default_rng(0)

    def isOpened(self):
        return self.abre

    def read(self):
        cuadro = np.full((48, 64, 3), 90, dtype=np.uint8)
        if self.real:
            cuadro = cuadro + self.rng.integers(0, 5, cuadro.shape, dtype=np.uint8)
        return True, cuadro

    def release(self):
        pass


class TestBuscarCamara(unittest.TestCase):
    def buscar(self, camaras, preferida=0):
        abrir = lambda i: camaras.get(i, CapturaFalsa(False, abre=False))
        with mock.patch.object(core, "abrir_camara", abrir):
            return core.buscar_camara(preferida)

    def test_salta_las_camaras_virtuales(self):
        """Asi esta el equipo del proyecto: 0 y 2 virtuales y 1 la webcam."""
        camaras = {0: CapturaFalsa(False), 1: CapturaFalsa(True), 2: CapturaFalsa(False)}
        self.assertEqual(self.buscar(camaras), 1)

    def test_respeta_la_preferida_si_es_real(self):
        self.assertEqual(self.buscar({0: CapturaFalsa(True), 3: CapturaFalsa(True)}, preferida=3), 3)

    def test_si_no_hay_ninguna_real_deja_la_preferida(self):
        self.assertEqual(self.buscar({0: CapturaFalsa(False)}, preferida=0), 0)


class TestJuegoCompleto(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self._dir = tempfile.TemporaryDirectory()
        self.ranking = Ranking(pathlib.Path(self._dir.name) / "r.db")
        self.audio = AudioEspia()
        self.cam = CamaraFalsa()
        self.pantalla = pygame.display.set_mode((reto.ANCHO, reto.ALTO))
        self.j = reto.Juego(self.pantalla, self.cam, self.audio, self.ranking)
        self.t = 100.0

    def tearDown(self):
        pygame.quit()
        self._dir.cleanup()

    def tecla(self, tecla):
        self.j.manejar(pygame.event.Event(pygame.KEYDOWN, key=tecla), self.t)

    def escribir(self, texto):
        self.j.manejar(pygame.event.Event(pygame.TEXTINPUT, text=texto), self.t)

    def correr(self, segundos, paso=1 / 30):
        fin = self.t + segundos
        while self.t < fin:
            self.t += paso
            self.j.actualizar(self.t)
            self.j.dibujar(self.t)

    def sentadillas(self, cuantas, pico=1.0):
        for _ in range(cuantas):
            for p in list(np.linspace(0, pico, 8)) + list(np.linspace(pico, 0, 8)) + [0.0] * 4:
                self.cam.puntos = persona(float(p))
                self.correr(1 / 30)

    def llegar_al_juego(self):
        self.correr(reto.SEGUNDOS_MARCA + 0.2)
        self.assertEqual(self.j.estado, "cuenta")
        self.correr(4.7)
        self.assertEqual(self.j.estado, "juego")

    def test_partida_individual_de_sentadillas(self):
        self.tecla(pygame.K_RETURN)
        self.assertEqual(self.j.estado, "apodo")
        self.escribir("José")
        self.tecla(pygame.K_RETURN)
        self.assertEqual(self.j.estado, "marca")
        self.llegar_al_juego()
        self.assertIn("musica_reto", self.audio.musicas)

        self.sentadillas(6)
        self.assertEqual(self.j.contador.perfectas, 6)
        self.assertIn("n6", self.audio.dicho)
        self.assertEqual(self.audio.efectos.count("ding"), 6)

        self.correr(SEGUNDOS_SENTADILLAS)
        self.assertEqual(self.j.estado, "resultado")
        self.assertTrue(self.j.resultado["record"])
        self.assertIn("record", self.audio.dicho)
        self.assertIn("animo", self.audio.dicho)
        self.assertEqual(self.ranking.top(RETO_SENTADILLAS)[0]["apodo"], "José")

        self.correr(reto.SEGUNDOS_RESULTADO + 0.2)
        self.assertEqual(self.j.estado, "consejos")
        self.correr(reto.SEGUNDOS_CONSEJOS + 0.2)
        self.assertEqual(self.j.estado, "menu")

    def test_partida_de_caza_estrellas(self):
        self.tecla(pygame.K_3)
        self.tecla(pygame.K_RETURN)
        self.tecla(pygame.K_RETURN)
        self.llegar_al_juego()
        self.assertIn("estrellas", self.audio.dicho)
        for _ in range(10):
            self.correr(0.4)
            objetivos = [o for o in self.j.estrellas.objetivos if o.tipo != "bomba"]
            if objetivos:
                self.cam.puntos = persona()
                self.cam.puntos[15] = FakeLandmark(objetivos[0].x, objetivos[0].y, 0.0, 1.0)
        self.cam.puntos = persona()
        self.assertGreater(self.j.estrellas.puntos, 0)
        self.correr(SEGUNDOS_ESTRELLAS)
        self.assertEqual(self.j.estado, "resultado")
        self.assertGreater(self.ranking.top(RETO_ESTRELLAS)[0]["puntaje"], 0)

    def test_al_final_da_consejos_y_los_dice_en_voz(self):
        self.tecla(pygame.K_RETURN)
        self.tecla(pygame.K_RETURN)
        self.llegar_al_juego()
        self.sentadillas(4, pico=0.35)
        self.correr(SEGUNDOS_SENTADILLAS)
        self.assertIn("profundidad", self.j.resultado["consejos"])
        self.tecla(pygame.K_RETURN)
        self.assertEqual(self.j.estado, "consejos")
        self.assertTrue(any(c.startswith("tip_") for c in self.audio.dicho))
        self.correr(0.2)
        self.tecla(pygame.K_RETURN)
        self.assertEqual(self.j.estado, "menu")

    def test_una_partida_guarda_lo_aprendido(self):
        ruta = pathlib.Path(self._dir.name) / "a.json"
        self.j.aprendizaje = Aprendizaje(ruta)
        self.tecla(pygame.K_RETURN)
        self.tecla(pygame.K_RETURN)
        self.llegar_al_juego()
        self.sentadillas(3)
        self.correr(SEGUNDOS_SENTADILLAS)
        self.assertEqual(Aprendizaje(ruta).partidas, 1)
        self.assertEqual(len(Aprendizaje(ruta).datos["profundidad"]), 3)

    def hacer_poses(self, cuantas):
        j = self.j.juez
        objetivo = j.poses + cuantas
        while j.poses < objetivo:
            self.cam.puntos = en_pose(j.pose)
            self.correr(0.2)

    def fallar_hasta_el_final(self):
        self.cam.puntos = persona()
        while self.j.estado == "juego":
            self.correr(0.5)

    def test_partida_por_turnos_de_estatua_anuncia_al_ganador(self):
        self.tecla(pygame.K_2)
        self.tecla(pygame.K_t)
        self.tecla(pygame.K_a)
        self.tecla(pygame.K_RETURN)
        self.escribir("Ana")
        self.tecla(pygame.K_RETURN)
        self.assertEqual(self.j.estado, "apodo")
        self.escribir("Luis")
        self.tecla(pygame.K_RETURN)

        self.llegar_al_juego()
        self.hacer_poses(1)
        self.fallar_hasta_el_final()
        self.assertEqual(self.j.estado, "resultado")
        self.assertEqual(self.j.jugador["puntaje"], 1)
        self.assertIn("aguanta", self.audio.dicho)
        self.assertIn("pose_ok", self.audio.dicho)
        self.correr(reto.SEGUNDOS_RESULTADO + reto.SEGUNDOS_CONSEJOS + 0.4)
        self.assertEqual((self.j.estado, self.j.turno), ("marca", 1))
        self.assertIn("turno2", self.audio.dicho)

        self.llegar_al_juego()
        self.assertIsNotNone(self.j.rival)
        self.hacer_poses(2)
        self.fallar_hasta_el_final()
        self.cam.puntos = persona()
        self.correr(reto.SEGUNDOS_RESULTADO + reto.SEGUNDOS_CONSEJOS + 0.4)
        self.assertEqual(self.j.estado, "duelo")
        self.assertEqual(self.j.resultado["ganador"], 1)
        self.assertIn("gana2", self.audio.dicho)
        self.correr(reto.SEGUNDOS_DUELO + 0.2)
        self.assertEqual(self.j.estado, "menu")

    def test_no_empieza_si_no_hay_nadie_en_la_marca(self):
        self.cam.puntos = None
        self.tecla(pygame.K_RETURN)
        self.tecla(pygame.K_RETURN)
        self.correr(5.0)
        self.assertEqual(self.j.estado, "marca")
        self.cam.puntos = persona()
        self.correr(reto.SEGUNDOS_MARCA + 0.2)
        self.assertEqual(self.j.estado, "cuenta")

    def test_escape_cancela_sin_guardar(self):
        self.tecla(pygame.K_RETURN)
        self.tecla(pygame.K_RETURN)
        self.llegar_al_juego()
        self.tecla(pygame.K_ESCAPE)
        self.assertEqual(self.j.estado, "menu")
        self.assertEqual(self.ranking.jugadas(), 0)

    def test_salir_del_menu_pide_escape_dos_veces(self):
        self.tecla(pygame.K_ESCAPE)
        self.assertFalse(self.j.salir)
        self.tecla(pygame.K_ESCAPE)
        self.assertTrue(self.j.salir)

    def test_el_apodo_tiene_limite_y_vacio_usa_uno_por_defecto(self):
        self.tecla(pygame.K_RETURN)
        self.escribir("x" * 40)
        self.assertEqual(len(self.j.apodo), reto.MAX_APODO)
        for _ in range(reto.MAX_APODO):
            self.tecla(pygame.K_BACKSPACE)
        self.tecla(pygame.K_RETURN)
        self.assertEqual(self.j.jugador["apodo"], "Jugador 1")

    def test_las_opciones_del_menu_cambian(self):
        self.tecla(pygame.K_d)
        self.assertEqual(self.j.nivel, "Difícil")
        self.tecla(pygame.K_z)
        self.assertEqual(self.j.entrenador, "zen")
        self.assertIn("hola", self.audio.dicho)
        self.tecla(pygame.K_p)
        self.assertTrue(self.j.privacidad)
        self.correr(0.2)

    def test_la_tecla_c_cambia_de_camara(self):
        self.tecla(pygame.K_c)
        self.assertEqual(self.cam.cambios, 1)
        self.correr(0.1)

    def test_se_dibuja_sin_imagen_de_camara(self):
        self.cam.rgb = None
        self.cam.error = "No se pudo abrir la cámara 0"
        self.correr(0.2)


if __name__ == "__main__":
    unittest.main()
