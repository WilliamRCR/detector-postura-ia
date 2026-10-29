"""Tests de las rutas de app_web, sin encender la camara.

app_web hace trabajo al importarse (carga el modelo, crea el historial, lee
config.json), asi que el modulo se importa desde un directorio temporal y se
usa el test client de Flask: el hilo de captura nunca arranca.
"""
import os
import sys
import tempfile
import unittest

_tmpdir = None
_cwd = None
app_web = None


def setUpModule():
    global _tmpdir, _cwd, app_web
    _cwd = os.getcwd()
    _tmpdir = tempfile.TemporaryDirectory()
    os.chdir(_tmpdir.name)
    # Sin voz ni notificaciones: los tests no deberian hablar en voz alta.
    with open("config.json", "w", encoding="utf-8") as f:
        f.write('{"voice_enabled": false, "desktop_notifications": false}')
    sys.path.insert(0, _cwd)
    import app_web as module

    app_web = module
    app_web.app.config["TESTING"] = True


def tearDownModule():
    os.chdir(_cwd)
    try:
        _tmpdir.cleanup()
    except OSError:
        # En Windows el .db puede seguir tomado un instante; no es un fallo.
        pass


class WebCase(unittest.TestCase):
    def setUp(self):
        self.client = app_web.app.test_client()


class TestPagina(WebCase):
    def test_la_plantilla_renderiza(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        for texto in ("Detector", "Historial", "Ajustes", "btnPrivacy", "weekChart"):
            self.assertIn(texto, html)


class TestStatus(WebCase):
    def test_devuelve_las_claves_esperadas(self):
        data = self.client.get("/api/status").get_json()
        for key in ("running", "testing_started", "status", "feedback", "color",
                    "context", "session_info", "camera_error", "voice_on",
                    "privacy", "break_in"):
            self.assertIn(key, data)


class TestConfig(WebCase):
    def test_get_devuelve_la_configuracion(self):
        data = self.client.get("/api/config").get_json()
        self.assertTrue(data["ok"])
        self.assertIn("break_minutes", data["config"])

    def test_post_guarda_y_aplica_en_caliente(self):
        r = self.client.post("/api/config", json={"break_minutes": 42, "alert_after_seconds": 9.0})
        cfg = r.get_json()["config"]
        self.assertEqual(cfg["break_minutes"], 42)
        self.assertEqual(cfg["alert_after_seconds"], 9.0)
        self.assertEqual(app_web.coach._timer.min_seconds, 9.0)
        self.assertEqual(app_web.breaks.interval_minutes, 42)

    def test_recorta_valores_fuera_de_rango(self):
        cfg = self.client.post("/api/config", json={"break_minutes": 9999, "camera_index": -3}).get_json()["config"]
        self.assertEqual(cfg["break_minutes"], 240)
        self.assertEqual(cfg["camera_index"], 0)

    def test_cambiar_un_umbral_apaga_calibrated(self):
        self.client.post("/api/config", json={"calibrated": True})
        cfg = self.client.post("/api/config", json={"shoulder_tilt_max": 0.3}).get_json()["config"]
        self.assertFalse(cfg["calibrated"])

    def test_avisa_que_la_camara_necesita_reinicio(self):
        aviso = self.client.post("/api/config", json={"camera_index": 1}).get_json()["aviso"]
        self.assertIn("reinicia", aviso.lower())
        self.client.post("/api/config", json={"camera_index": 0})


class TestPrivacidad(WebCase):
    def test_alterna_y_se_refleja_en_el_estado(self):
        antes = self.client.get("/api/status").get_json()["privacy"]
        despues = self.client.post("/api/privacy").get_json()["privacy"]
        self.assertNotEqual(antes, despues)
        self.assertEqual(self.client.get("/api/status").get_json()["privacy"], despues)
        self.client.post("/api/privacy")  # vuelve a dejarlo como estaba


class TestHistorial(WebCase):
    def setUp(self):
        super().setUp()
        app_web.history.clear()

    def test_semana_vacia_trae_siete_dias(self):
        data = self.client.get("/api/history").get_json()
        self.assertEqual(len(data["week"]), 7)
        self.assertEqual(data["totals"]["sesiones"], 0)

    def test_cuenta_una_sesion_guardada(self):
        app_web.history.save_session(minutos=25, porcentaje=72.0,
                                     problema_comun="Endereza los hombros")
        data = self.client.get("/api/history").get_json()
        self.assertEqual(data["totals"]["sesiones"], 1)
        self.assertEqual(data["totals"]["problema_comun"], "Endereza los hombros")
        self.assertEqual(len(data["recent"]), 1)


class TestControl(WebCase):
    def test_start_sin_camara_devuelve_error_claro(self):
        data = self.client.post("/api/start").get_json()
        self.assertFalse(data["ok"])
        self.assertTrue(data["error"])

    def test_stop_siempre_responde_ok(self):
        self.assertTrue(self.client.post("/api/stop").get_json()["ok"])

    def test_voice_alterna(self):
        data = self.client.post("/api/voice").get_json()
        self.assertTrue(data["ok"])
        self.assertIn("voice_on", data)


if __name__ == "__main__":
    unittest.main()
