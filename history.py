"""Historial persistente de sesiones en SQLite.

Antes las estadisticas de sesion se perdian al cerrar la app. Aqui cada sesion
queda guardada (fecha, minutos, % correcta, problema mas comun) para poder ver
la evolucion de la semana.

Solo usa sqlite3 de la libreria estandar: no agrega dependencias.
"""
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

DB_FILE = "historial_postura.db"

# Sesiones mas cortas que esto son ruido (abrir y cerrar la app) y no se guardan.
MIN_SESSION_MINUTES = 0.5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sesiones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha           TEXT NOT NULL,   -- YYYY-MM-DD (dia local)
    inicio          TEXT NOT NULL,   -- ISO 8601
    minutos         REAL NOT NULL,
    porcentaje      REAL NOT NULL,
    problema_comun  TEXT,
    muestras        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sesiones_fecha ON sesiones (fecha);
"""

DAY_NAMES = ("Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom")


class SessionHistory:
    """Guarda y consulta el historial. Cada metodo abre y cierra su conexion,
    asi se puede usar desde varios hilos (GUI + hilo de camara) sin lios."""

    def __init__(self, path=DB_FILE):
        self.path = str(path)
        self._ensure()

    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self):
        """Abre, confirma y CIERRA la conexion.

        `with sqlite3.connect(...)` confirma la transaccion pero deja la
        conexion abierta: en Windows eso mantiene el archivo bloqueado y va
        acumulando conexiones sesion tras sesion.
        """
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure(self):
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            print(f"No pude preparar el historial ({exc}); sigo sin guardar.")

    # ------------------------------------------------------------------
    def save_session(self, minutos, porcentaje, problema_comun=None, muestras=0, inicio=None):
        """Guarda una sesion. Devuelve el id, o None si no se guardo."""
        if minutos is None or minutos < MIN_SESSION_MINUTES:
            return None
        started = inicio or datetime.now()
        if isinstance(started, (int, float)):
            started = datetime.fromtimestamp(started)
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO sesiones (fecha, inicio, minutos, porcentaje, problema_comun, muestras)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        started.strftime("%Y-%m-%d"),
                        started.isoformat(timespec="seconds"),
                        float(minutos),
                        float(porcentaje),
                        problema_comun,
                        int(muestras),
                    ),
                )
                return cur.lastrowid
        except sqlite3.Error as exc:
            print(f"No pude guardar la sesion en el historial ({exc}).")
            return None

    def save_stats(self, stats):
        """Atajo para guardar un SessionStats de posture_metrics."""
        if stats is None or stats.samples == 0:
            return None
        return self.save_session(
            minutos=stats.elapsed_minutes,
            porcentaje=stats.percent_correct,
            problema_comun=stats.common_issue,
            muestras=stats.samples,
            inicio=stats.started_at,
        )

    # ------------------------------------------------------------------
    def recent_sessions(self, limit=20):
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM sesiones ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def weekly_summary(self, days=7, today=None):
        """Una fila por dia de los ultimos `days` dias, incluidos los dias sin
        uso (en cero), para poder dibujar la grafica sin huecos."""
        today = today or date.today()
        start = today - timedelta(days=days - 1)

        totals = {}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT fecha,"
                    "       SUM(minutos) AS minutos,"
                    "       SUM(porcentaje * minutos) AS pond,"
                    "       SUM(minutos) AS peso,"
                    "       COUNT(*) AS sesiones"
                    "  FROM sesiones WHERE fecha >= ? AND fecha <= ?"
                    " GROUP BY fecha",
                    (start.isoformat(), today.isoformat()),
                ).fetchall()
                for row in rows:
                    peso = row["peso"] or 0.0
                    totals[row["fecha"]] = {
                        "minutos": row["minutos"] or 0.0,
                        # Promedio ponderado por minutos: una sesion de 2 min no
                        # pesa igual que una de una hora.
                        "porcentaje": (row["pond"] / peso) if peso else 0.0,
                        "sesiones": row["sesiones"],
                    }
        except sqlite3.Error:
            totals = {}

        summary = []
        for offset in range(days):
            day = start + timedelta(days=offset)
            key = day.isoformat()
            data = totals.get(key, {"minutos": 0.0, "porcentaje": 0.0, "sesiones": 0})
            summary.append(
                {
                    "fecha": key,
                    "dia": DAY_NAMES[day.weekday()],
                    "minutos": round(data["minutos"], 1),
                    "porcentaje": round(data["porcentaje"], 1),
                    "sesiones": data["sesiones"],
                }
            )
        return summary

    def totals(self, days=7, today=None):
        """Resumen agregado de la semana, para el texto que acompana la grafica."""
        week = self.weekly_summary(days=days, today=today)
        minutos = sum(day["minutos"] for day in week)
        if minutos <= 0:
            return {"minutos": 0.0, "porcentaje": 0.0, "sesiones": 0, "problema_comun": None}

        ponderado = sum(day["porcentaje"] * day["minutos"] for day in week) / minutos
        sesiones = sum(day["sesiones"] for day in week)

        problema = None
        try:
            end = today or date.today()
            start = end - timedelta(days=days - 1)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT problema_comun, COUNT(*) AS n FROM sesiones"
                    " WHERE fecha >= ? AND fecha <= ? AND problema_comun IS NOT NULL"
                    " GROUP BY problema_comun ORDER BY n DESC LIMIT 1",
                    (start.isoformat(), end.isoformat()),
                ).fetchone()
                if row:
                    problema = row["problema_comun"]
        except sqlite3.Error:
            problema = None

        return {
            "minutos": round(minutos, 1),
            "porcentaje": round(ponderado, 1),
            "sesiones": sesiones,
            "problema_comun": problema,
        }

    # ------------------------------------------------------------------
    def clear(self):
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM sesiones")
            return True
        except sqlite3.Error:
            return False

    def exists(self):
        return Path(self.path).exists()
