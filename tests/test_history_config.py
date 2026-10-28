"""Tests del historial SQLite, la configuracion y las estadisticas de sesion."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path

from config import DEFAULTS, AppConfig
from history import SessionHistory
from posture_metrics import PostureThresholds, SessionStats


class TempFileCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()


class TestSessionHistory(TempFileCase):
    def nueva(self):
        return SessionHistory(self.tmp / "historial.db")

    def test_se_crea_la_base(self):
        h = self.nueva()
        self.assertTrue(h.exists())

    def test_guarda_y_lee_una_sesion(self):
        h = self.nueva()
        rowid = h.save_session(minutos=12.5, porcentaje=81.0, problema_comun="Endereza los hombros")
        self.assertIsNotNone(rowid)
        recientes = h.recent_sessions()
        self.assertEqual(len(recientes), 1)
        self.assertAlmostEqual(recientes[0]["minutos"], 12.5)
        self.assertEqual(recientes[0]["problema_comun"], "Endereza los hombros")

    def test_descarta_sesiones_demasiado_cortas(self):
        h = self.nueva()
        self.assertIsNone(h.save_session(minutos=0.1, porcentaje=90.0))
        self.assertEqual(h.recent_sessions(), [])

    def test_la_semana_siempre_trae_siete_dias(self):
        h = self.nueva()
        semana = h.weekly_summary(days=7)
        self.assertEqual(len(semana), 7)
        self.assertTrue(all(day["minutos"] == 0 for day in semana))

    def test_los_dias_sin_uso_quedan_en_cero(self):
        hoy = date(2026, 10, 27)
        h = self.nueva()
        h.save_session(minutos=20, porcentaje=75.0, inicio=datetime(2026, 10, 27, 9, 0))
        semana = h.weekly_summary(days=7, today=hoy)
        self.assertEqual(semana[-1]["minutos"], 20.0)
        self.assertEqual(semana[-1]["porcentaje"], 75.0)
        self.assertEqual(semana[0]["minutos"], 0.0)

    def test_el_porcentaje_del_dia_se_pondera_por_minutos(self):
        """Una sesion de 2 min no puede pesar lo mismo que una de una hora."""
        hoy = date(2026, 10, 27)
        cuando = datetime(2026, 10, 27, 10, 0)
        h = self.nueva()
        h.save_session(minutos=60, porcentaje=90.0, inicio=cuando)
        h.save_session(minutos=2, porcentaje=10.0, inicio=cuando)
        semana = h.weekly_summary(days=7, today=hoy)
        # Promedio simple seria 50%; ponderado ronda 87%.
        self.assertGreater(semana[-1]["porcentaje"], 85.0)

    def test_totales_de_la_semana(self):
        hoy = date(2026, 10, 27)
        h = self.nueva()
        h.save_session(minutos=30, porcentaje=80.0, problema_comun="Endereza los hombros",
                       inicio=datetime(2026, 10, 26, 9, 0))
        h.save_session(minutos=30, porcentaje=60.0, problema_comun="Endereza los hombros",
                       inicio=datetime(2026, 10, 27, 9, 0))
        totals = h.totals(days=7, today=hoy)
        self.assertEqual(totals["sesiones"], 2)
        self.assertAlmostEqual(totals["minutos"], 60.0)
        self.assertAlmostEqual(totals["porcentaje"], 70.0)
        self.assertEqual(totals["problema_comun"], "Endereza los hombros")

    def test_totales_vacios(self):
        totals = self.nueva().totals(days=7)
        self.assertEqual(totals["sesiones"], 0)
        self.assertIsNone(totals["problema_comun"])

    def test_no_cuenta_sesiones_fuera_de_la_ventana(self):
        hoy = date(2026, 10, 27)
        h = self.nueva()
        h.save_session(minutos=40, porcentaje=99.0, inicio=datetime(2026, 9, 1, 9, 0))
        self.assertEqual(h.totals(days=7, today=hoy)["sesiones"], 0)

    def test_guarda_desde_session_stats(self):
        h = self.nueva()
        stats = SessionStats()
        stats.start()
        stats._start_time -= 600  # simula 10 minutos de uso
        for _ in range(8):
            stats.record(True, "Buena postura")
        for _ in range(2):
            stats.record(False, "Endereza los hombros")
        self.assertIsNotNone(h.save_stats(stats))
        fila = h.recent_sessions()[0]
        self.assertAlmostEqual(fila["porcentaje"], 80.0, places=1)
        self.assertEqual(fila["problema_comun"], "Endereza los hombros")

    def test_no_guarda_una_sesion_vacia(self):
        h = self.nueva()
        self.assertIsNone(h.save_stats(SessionStats()))

    def test_clear_borra_todo(self):
        h = self.nueva()
        h.save_session(minutos=10, porcentaje=50.0)
        h.clear()
        self.assertEqual(h.recent_sessions(), [])


class TestSessionStats(unittest.TestCase):
    def test_porcentaje_correcto(self):
        stats = SessionStats()
        stats.start()
        for _ in range(3):
            stats.record(True)
        stats.record(False, "Endereza los hombros")
        self.assertAlmostEqual(stats.percent_correct, 75.0)
        self.assertEqual(stats.samples, 4)

    def test_none_no_cuenta(self):
        stats = SessionStats()
        stats.start()
        stats.record(None)
        self.assertEqual(stats.samples, 0)

    def test_problema_mas_comun(self):
        stats = SessionStats()
        stats.start()
        stats.record(False, "Endereza los hombros")
        stats.record(False, "Endereza los hombros")
        stats.record(False, "Levanta un poco la cabeza")
        self.assertEqual(stats.common_issue, "Endereza los hombros")

    def test_buena_postura_no_cuenta_como_problema(self):
        stats = SessionStats()
        stats.start()
        stats.record(False, "Buena postura")
        self.assertIsNone(stats.common_issue)

    def test_sin_muestras_no_hay_texto(self):
        self.assertEqual(SessionStats().summary_text(), "")


class TestAppConfig(TempFileCase):
    def nueva(self):
        return AppConfig(path=self.tmp / "config.json")

    def test_valores_por_defecto(self):
        cfg = self.nueva()
        self.assertEqual(cfg.camera_index, DEFAULTS["camera_index"])
        self.assertTrue(cfg.voice_enabled)

    def test_guardar_y_cargar(self):
        path = self.tmp / "config.json"
        cfg = AppConfig(path=path)
        cfg.update({"camera_index": 2, "break_minutes": 40})
        otra = AppConfig.load(path)
        self.assertEqual(otra.camera_index, 2)
        self.assertEqual(otra.break_minutes, 40)

    def test_ignora_claves_desconocidas(self):
        cfg = self.nueva()
        cfg.update({"no_existe": 1, "camera_index": 1})
        self.assertEqual(cfg.camera_index, 1)
        self.assertNotIn("no_existe", cfg.as_dict())

    def test_recorta_valores_fuera_de_rango(self):
        cfg = self.nueva()
        cfg.update({"camera_index": -5, "voice_rate": 9999, "break_minutes": 1})
        self.assertEqual(cfg.camera_index, 0)
        self.assertEqual(cfg.voice_rate, 400)
        self.assertEqual(cfg.break_minutes, 5)

    def test_convierte_tipos(self):
        cfg = self.nueva()
        cfg.update({"camera_index": "2", "voice_enabled": "false", "alert_after_seconds": "7.5"})
        self.assertEqual(cfg.camera_index, 2)
        self.assertFalse(cfg.voice_enabled)
        self.assertAlmostEqual(cfg.alert_after_seconds, 7.5)

    def test_valor_basura_cae_al_default(self):
        cfg = self.nueva()
        cfg.update({"alert_after_seconds": "no-es-un-numero"})
        self.assertEqual(cfg.alert_after_seconds, DEFAULTS["alert_after_seconds"])

    def test_config_json_corrupto_no_rompe(self):
        path = self.tmp / "config.json"
        path.write_text("{ esto no es json", encoding="utf-8")
        with redirect_stdout(io.StringIO()) as salida:
            cfg = AppConfig.load(path)
        self.assertEqual(cfg.camera_index, DEFAULTS["camera_index"])
        self.assertIn("ilegible", salida.getvalue())

    def test_archivo_inexistente_usa_defaults(self):
        cfg = AppConfig.load(self.tmp / "no-existe.json")
        self.assertEqual(cfg.as_dict(), DEFAULTS)

    def test_umbrales_van_y_vuelven(self):
        cfg = self.nueva()
        nuevos = PostureThresholds(shoulder_tilt_max=0.15, forward_head_max=0.40,
                                   neck_compression_min=0.30)
        cfg.set_thresholds(nuevos)
        self.assertEqual(cfg.thresholds(), nuevos)
        self.assertTrue(cfg.calibrated)

    def test_reset_vuelve_a_los_defaults(self):
        cfg = self.nueva()
        cfg.update({"camera_index": 3})
        cfg.reset()
        self.assertEqual(cfg.camera_index, DEFAULTS["camera_index"])

    def test_el_archivo_guardado_es_json_valido(self):
        path = self.tmp / "config.json"
        cfg = AppConfig(path=path)
        cfg.update({"break_minutes": 45})
        datos = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(datos["break_minutes"], 45)


if __name__ == "__main__":
    unittest.main()
