"""Aplicacion de escritorio (Tkinter).

Incluye evaluacion inicial de 30s con calibracion personal, avisos por tiempo
sostenido, recordatorios de pausa, modo privacidad, historial persistente,
configuracion editable y bandeja del sistema.
"""
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import core
from assessment import PostureAssessment
from config import AppConfig
from history import SessionHistory
from posture_metrics import (
    CONTEXT_LABELS,
    SessionStats,
    detect_context,
    evaluate_aspects,
    summarize_aspects,
)
from voice_feedback import BreakReminder, DesktopNotifier, PostureVoiceCoach, VoiceFeedback

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
ACCENT_COLOR = "#0f3460"
GREEN = "#2ecc71"
RED = "#e74c3c"
GRAY = "#555577"
TEXT_COLOR = "#e0e0e0"

VIDEO_MAX_W, VIDEO_MAX_H = 640, 480
VIDEO_MIN_H = 240

# Alto aproximado de todo lo que NO es el video (encabezado, pestanas, estado,
# feedback, botones y pie). Se reserva para que los botones siempre entren en
# pantalla: en monitores de 768 px de alto, un video fijo de 480 px dejaba
# INICIAR fuera de la ventana.
CHROME_HEIGHT = 370

# Barra de titulo + barra de tareas de Windows: espacio que la ventana nunca
# puede ocupar aunque la pantalla lo reporte como disponible.
SCREEN_MARGIN = 90


class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Postura IA")
        self.root.configure(bg=BG_COLOR)
        # Redimensionable a proposito: si la pantalla es justa, el usuario
        # siempre puede estirar la ventana para ver los botones.
        self.root.resizable(True, True)
        self.video_w, self.video_h = self._video_size()

        self.cfg = AppConfig.load()

        self.running = False
        self.testing_started = False
        self.cap = None
        self.model_error = None
        self._tray = None

        import mediapipe as mp

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = core.new_pose()

        try:
            self.predictor = core.PosturePredictor.load(thresholds=self.cfg.thresholds())
            self.model_loaded = True
        except core.ModelError as exc:
            self.predictor = None
            self.model_loaded = False
            self.model_error = str(exc)

        self.voice = VoiceFeedback(
            enabled=self.cfg.voice_enabled,
            rate=self.cfg.voice_rate,
            cooldown=self.cfg.voice_cooldown,
        )
        self.notifier = DesktopNotifier(enabled=self.cfg.desktop_notifications)
        self.coach = PostureVoiceCoach(
            self.voice,
            min_bad_seconds=self.cfg.alert_after_seconds,
            repeat_seconds=self.cfg.alert_repeat_seconds,
            notifier=self.notifier,
        )
        self.breaks = BreakReminder(
            interval_minutes=self.cfg.break_minutes, enabled=self.cfg.break_reminders
        )
        self.history = SessionHistory() if self.cfg.save_history else None

        self.voice_var = tk.BooleanVar(value=self.voice.enabled)
        self.privacy_var = tk.BooleanVar(value=self.cfg.privacy_mode)

        self.assessment = PostureAssessment(duration=self.cfg.assessment_seconds)
        self.last_verdict = None
        self.verdict_hold_until = 0.0
        self.session = SessionStats()

        self._build_ui()
        self._fit_to_screen()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_preview()
        self._setup_tray()

    # ------------------------------------------------------------------ UI
    def _video_size(self):
        """Tamano inicial del video, antes de medir la interfaz real."""
        try:
            disponible = self.root.winfo_screenheight() - CHROME_HEIGHT
        except tk.TclError:
            disponible = VIDEO_MAX_H
        alto = max(VIDEO_MIN_H, min(VIDEO_MAX_H, disponible))
        return int(alto * 4 / 3), int(alto)

    def _show_placeholder(self):
        """Pone una imagen negra del tamano exacto del video.

        Importante: un tk.Label SIN imagen mide width/height en caracteres y
        lineas, no en pixeles. Con height=480 pedia 480 lineas de texto y la
        ventana calculaba un alto absurdo (7000+ px) antes del primer frame.
        """
        img = ImageTk.PhotoImage(Image.new("RGB", (self.video_w, self.video_h), "black"))
        self.video_label.imgtk = img
        self.video_label.config(image=img)

    def _set_video_height(self, alto):
        self.video_w, self.video_h = int(alto * 4 / 3), int(alto)
        self._show_placeholder()
        self.root.update_idletasks()

    def _fit_to_screen(self):
        """Busca el video mas grande con el que la ventana entre en pantalla.

        Se mide la interfaz ya construida en vez de estimarla: el alto del
        encabezado, las pestanas y los botones cambia con la fuente y el tema
        de Windows, y una estimacion fija dejaba INICIAR fuera de cuadro.

        Arranca del tamano maximo y achica solo lo que sobra, repitiendo
        porque cada cambio de tamano reacomoda el resto de la ventana.
        """
        try:
            pantalla = self.root.winfo_screenheight()
        except tk.TclError:
            return
        usable = pantalla - SCREEN_MARGIN

        self._set_video_height(VIDEO_MAX_H)
        for _ in range(5):
            sobra = self.root.winfo_reqheight() - usable
            if sobra <= 0:
                break
            nuevo = max(VIDEO_MIN_H, self.video_h - sobra)
            if nuevo == self.video_h:
                break  # ya esta en el minimo: no se puede achicar mas
            self._set_video_height(nuevo)

        # El usuario puede achicar la ventana, pero no hasta esconder los botones.
        resto = self.root.winfo_reqheight() - self.video_h
        self.root.minsize(560, min(usable, resto + VIDEO_MIN_H))

    def _build_ui(self):
        header = tk.Frame(self.root, bg=ACCENT_COLOR, pady=14)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="DETECTOR DE POSTURA  IA",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg=ACCENT_COLOR,
        ).pack()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_detector = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_history = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_settings = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self.tab_detector, text="  Detector  ")
        self.notebook.add(self.tab_history, text="  Historial  ")
        self.notebook.add(self.tab_settings, text="  Ajustes  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_detector_tab()
        self._build_history_tab()
        self._build_settings_tab()

    def _build_detector_tab(self):
        parent = self.tab_detector

        video_frame = tk.Frame(parent, bg=BG_COLOR, pady=8)
        video_frame.pack()
        self.video_label = tk.Label(video_frame, bg="#000000", bd=0)
        self.video_label.pack()
        self._show_placeholder()

        self.status_var = tk.StringVar(value="Estado: Sin detectar")
        self.status_label = tk.Label(
            parent,
            textvariable=self.status_var,
            font=("Segoe UI", 15, "bold"),
            fg="white",
            bg=GRAY,
            padx=30,
            pady=12,
            width=36,
            anchor="center",
        )
        self.status_label.pack(pady=(0, 6))

        self.feedback_var = tk.StringVar(value="")
        tk.Label(
            parent,
            textvariable=self.feedback_var,
            font=("Segoe UI", 12),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            pady=4,
            justify=tk.LEFT,
        ).pack()

        self.info_var = tk.StringVar(value="")
        tk.Label(
            parent,
            textvariable=self.info_var,
            font=("Segoe UI", 10),
            fg="#8888aa",
            bg=BG_COLOR,
            pady=2,
        ).pack()

        btn_frame = tk.Frame(parent, bg=BG_COLOR, pady=10)
        btn_frame.pack()

        self.start_btn = tk.Button(
            btn_frame,
            text="▶   INICIAR",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg=GREEN,
            activebackground="#27ae60",
            activeforeground="white",
            padx=24,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._start,
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(
            btn_frame,
            text="■   DETENER",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg=RED,
            activebackground="#c0392b",
            activeforeground="white",
            padx=24,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._stop,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        self.voice_check = tk.Checkbutton(
            btn_frame,
            text="🔊 Voz",
            variable=self.voice_var,
            command=self._toggle_voice,
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            selectcolor=ACCENT_COLOR,
            activebackground=BG_COLOR,
            activeforeground=TEXT_COLOR,
            state=tk.NORMAL if self.voice.available else tk.DISABLED,
        )
        self.voice_check.pack(side=tk.LEFT, padx=10)

        self.privacy_check = tk.Checkbutton(
            btn_frame,
            text="🕶 Privacidad",
            variable=self.privacy_var,
            command=self._toggle_privacy,
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            selectcolor=ACCENT_COLOR,
            activebackground=BG_COLOR,
            activeforeground=TEXT_COLOR,
        )
        self.privacy_check.pack(side=tk.LEFT, padx=10)

        if not self.model_loaded:
            tk.Label(
                parent,
                text=self.model_error or "Modelo no encontrado. Ejecuta train.py primero.",
                fg=RED,
                bg=BG_COLOR,
                font=("Segoe UI", 10),
                pady=6,
                wraplength=600,
            ).pack()

        tk.Label(
            parent,
            text="DETENER vuelve a la vista previa  ·  cerrar minimiza a la bandeja",
            font=("Segoe UI", 9),
            fg="#666688",
            bg=BG_COLOR,
            pady=6,
        ).pack()

    # ---------------------------------------------------------- historial
    def _build_history_tab(self):
        parent = self.tab_history

        tk.Label(
            parent,
            text="Últimos 7 días",
            font=("Segoe UI", 14, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            pady=12,
        ).pack()

        self.history_canvas = tk.Canvas(
            parent, width=620, height=260, bg=PANEL_COLOR, highlightthickness=0
        )
        self.history_canvas.pack(pady=8)

        self.history_summary = tk.StringVar(value="")
        tk.Label(
            parent,
            textvariable=self.history_summary,
            font=("Segoe UI", 11),
            fg="#8888aa",
            bg=BG_COLOR,
            pady=10,
            justify=tk.CENTER,
        ).pack()

        tk.Button(
            parent,
            text="Actualizar",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=ACCENT_COLOR,
            relief=tk.FLAT,
            padx=18,
            pady=6,
            cursor="hand2",
            command=self._refresh_history,
        ).pack(pady=(0, 14))

    def _refresh_history(self):
        canvas = self.history_canvas
        canvas.delete("all")

        if self.history is None:
            canvas.create_text(
                310, 130, text="Historial desactivado en Ajustes", fill="#8888aa",
                font=("Segoe UI", 11),
            )
            self.history_summary.set("")
            return

        week = self.history.weekly_summary(days=7)
        totals = self.history.totals(days=7)

        width, height = 620, 260
        margin_left, margin_bottom, margin_top = 46, 44, 24
        plot_h = height - margin_bottom - margin_top
        slot = (width - margin_left - 20) / len(week)
        bar_w = min(46, slot * 0.55)

        # Rejilla 0-100%
        for pct in (0, 25, 50, 75, 100):
            y = height - margin_bottom - (pct / 100.0) * plot_h
            canvas.create_line(margin_left, y, width - 12, y, fill="#24244a")
            canvas.create_text(margin_left - 10, y, text=f"{pct}%", fill="#6666aa",
                               font=("Segoe UI", 8), anchor="e")

        for i, day in enumerate(week):
            cx = margin_left + slot * (i + 0.5)
            pct = day["porcentaje"]
            bar_h = (pct / 100.0) * plot_h
            y0 = height - margin_bottom - bar_h
            y1 = height - margin_bottom

            if day["minutos"] <= 0:
                canvas.create_rectangle(cx - bar_w / 2, y1 - 3, cx + bar_w / 2, y1,
                                        fill="#2a2a4e", outline="")
            else:
                color = GREEN if pct >= 70 else ("#f39c12" if pct >= 45 else RED)
                canvas.create_rectangle(cx - bar_w / 2, y0, cx + bar_w / 2, y1,
                                        fill=color, outline="")
                canvas.create_text(cx, y0 - 10, text=f"{pct:.0f}%", fill=TEXT_COLOR,
                                   font=("Segoe UI", 9, "bold"))
                canvas.create_text(cx, y1 + 26, text=f"{day['minutos']:.0f} min",
                                   fill="#6666aa", font=("Segoe UI", 8))

            canvas.create_text(cx, y1 + 12, text=day["dia"], fill="#9999cc",
                               font=("Segoe UI", 9))

        if totals["sesiones"] == 0:
            self.history_summary.set("Todavía no hay sesiones guardadas. Usa el detector y cierra la app para registrar una.")
        else:
            partes = [
                f"{totals['sesiones']} sesiones",
                f"{totals['minutos']:.0f} min en total",
                f"{totals['porcentaje']:.0f}% correcta",
            ]
            texto = "   ·   ".join(partes)
            if totals["problema_comun"]:
                texto += f"\nProblema más frecuente: {totals['problema_comun']}"
            self.history_summary.set(texto)

    # ------------------------------------------------------------ ajustes
    def _scrollable(self, parent):
        """Devuelve un frame interno con barra de scroll vertical.

        El Notebook adopta el alto de su pestana mas alta. Sin scroll, los ~18
        controles de Ajustes estiraban TODA la ventana y dejaban los botones
        del Detector fuera de la pantalla.
        """
        canvas = tk.Canvas(parent, bg=BG_COLOR, highlightthickness=0)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG_COLOR)

        inner.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        ventana = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(ventana, width=e.width))
        canvas.configure(yscrollcommand=bar.set)

        bar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def rueda(event):
            # Solo si el puntero esta sobre esta pestana. No se usa <Leave> del
            # canvas: Tk lo dispara al entrar a cualquier control hijo, y como
            # los hijos cubren todo el canvas la rueda dejaba de funcionar.
            sobre = self.root.winfo_containing(event.x_root, event.y_root)
            while sobre is not None and sobre is not parent:
                sobre = sobre.master
            if sobre is parent:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        canvas.bind_all("<MouseWheel>", rueda, add="+")
        return inner

    def _build_settings_tab(self):
        contenedor = self._scrollable(self.tab_settings)
        wrap = tk.Frame(contenedor, bg=BG_COLOR, padx=28, pady=16)
        wrap.pack(fill=tk.BOTH, expand=True)

        self.setting_vars = {}

        def row(label, key, kind="float", width=8, hint=""):
            line = tk.Frame(wrap, bg=BG_COLOR)
            line.pack(fill=tk.X, pady=4)
            tk.Label(line, text=label, font=("Segoe UI", 10), fg=TEXT_COLOR, bg=BG_COLOR,
                     width=30, anchor="w").pack(side=tk.LEFT)
            if kind == "bool":
                var = tk.BooleanVar(value=bool(self.cfg[key]))
                tk.Checkbutton(line, variable=var, bg=BG_COLOR, selectcolor=ACCENT_COLOR,
                               activebackground=BG_COLOR).pack(side=tk.LEFT)
            else:
                var = tk.StringVar(value=str(self.cfg[key]))
                tk.Entry(line, textvariable=var, width=width, font=("Segoe UI", 10),
                         bg=PANEL_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                         relief=tk.FLAT).pack(side=tk.LEFT)
            if hint:
                tk.Label(line, text=hint, font=("Segoe UI", 8), fg="#666688",
                         bg=BG_COLOR).pack(side=tk.LEFT, padx=8)
            self.setting_vars[key] = (var, kind)

        def title(text):
            tk.Label(wrap, text=text, font=("Segoe UI", 11, "bold"), fg="#9999cc",
                     bg=BG_COLOR, pady=6, anchor="w").pack(fill=tk.X)

        title("Cámara")
        row("Número de cámara", "camera_index", "int", hint="0 = la primera")
        row("Espejar imagen", "flip_camera", "bool")

        title("Voz y avisos")
        row("Voz activada", "voice_enabled", "bool")
        row("Velocidad de la voz", "voice_rate", "int", hint="80 a 400")
        row("Avisar tras (segundos)", "alert_after_seconds", "float", hint="mala postura sostenida")
        row("Repetir aviso cada (s)", "alert_repeat_seconds", "float")
        row("Notificaciones de Windows", "desktop_notifications", "bool")

        title("Pausas")
        row("Recordar pausas", "break_reminders", "bool")
        row("Cada (minutos)", "break_minutes", "int", hint="30 a 45 recomendado")

        title("Umbrales de postura")
        row("Inclinación de hombros máx.", "shoulder_tilt_max", "float")
        row("Cabeza adelantada máx.", "forward_head_max", "float")
        row("Altura de cabeza mín.", "neck_compression_min", "float")

        title("Otros")
        row("Duración evaluación (s)", "assessment_seconds", "float")
        row("Modo privacidad", "privacy_mode", "bool")
        row("Guardar historial", "save_history", "bool")

        self.settings_msg = tk.StringVar(value="")
        tk.Label(wrap, textvariable=self.settings_msg, font=("Segoe UI", 9),
                 fg=GREEN, bg=BG_COLOR, pady=6).pack()

        buttons = tk.Frame(wrap, bg=BG_COLOR, pady=8)
        buttons.pack()
        tk.Button(buttons, text="Guardar", font=("Segoe UI", 10, "bold"), fg="white",
                  bg=GREEN, relief=tk.FLAT, padx=20, pady=7, cursor="hand2",
                  command=self._save_settings).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="Restaurar valores por defecto", font=("Segoe UI", 10),
                  fg="white", bg=ACCENT_COLOR, relief=tk.FLAT, padx=20, pady=7,
                  cursor="hand2", command=self._reset_settings).pack(side=tk.LEFT, padx=8)

    def _collect_settings(self):
        values = {}
        for key, (var, kind) in self.setting_vars.items():
            raw = var.get()
            if kind == "bool":
                values[key] = bool(raw)
            else:
                text = str(raw).strip().replace(",", ".")
                if not text:
                    continue
                try:
                    values[key] = float(text)
                except ValueError:
                    return None, f"«{text}» no es un número válido."
        return values, None

    def _save_settings(self):
        values, error = self._collect_settings()
        if error:
            self.settings_msg.set(error)
            return
        # Un cambio manual de umbrales deja de ser una calibracion automatica.
        if any(self.cfg[key] != values.get(key, self.cfg[key]) for key in
               ("shoulder_tilt_max", "forward_head_max", "neck_compression_min")):
            values["calibrated"] = False
        camara_cambio = int(values.get("camera_index", self.cfg.camera_index)) != self.cfg.camera_index
        self.cfg.update(values)
        self._apply_config()
        self._refresh_settings_fields()
        self.settings_msg.set(
            "Guardado. El cambio de cámara se aplica al reiniciar la app."
            if camara_cambio
            else "Guardado en config.json."
        )
        self.root.after(4000, lambda: self.settings_msg.set(""))

    def _reset_settings(self):
        self.cfg.reset()
        self._apply_config()
        self._refresh_settings_fields()
        self.settings_msg.set("Valores por defecto restaurados.")
        self.root.after(4000, lambda: self.settings_msg.set(""))

    def _refresh_settings_fields(self):
        for key, (var, kind) in self.setting_vars.items():
            if kind == "bool":
                var.set(bool(self.cfg[key]))
            else:
                var.set(str(self.cfg[key]))

    def _apply_config(self):
        """Aplica la configuracion en caliente, sin reiniciar la app."""
        if self.predictor is not None:
            self.predictor.thresholds = self.cfg.thresholds()
        self.voice.set_enabled(self.cfg.voice_enabled)
        self.voice.set_rate(self.cfg.voice_rate)
        self.voice_var.set(self.voice.enabled)
        self.notifier.set_enabled(self.cfg.desktop_notifications)
        self.coach.configure(
            min_bad_seconds=self.cfg.alert_after_seconds,
            repeat_seconds=self.cfg.alert_repeat_seconds,
        )
        self.breaks.interval_minutes = self.cfg.break_minutes
        self.breaks.enabled = self.cfg.break_reminders
        self.assessment.duration = self.cfg.assessment_seconds
        self.privacy_var.set(self.cfg.privacy_mode)
        if self.cfg.save_history and self.history is None:
            self.history = SessionHistory()
        elif not self.cfg.save_history:
            self.history = None

    def _on_tab_changed(self, _event=None):
        if self.notebook.index(self.notebook.select()) == 1:
            self._refresh_history()

    # -------------------------------------------------------------- bandeja
    def _setup_tray(self):
        """Icono en la bandeja del sistema, para seguir avisando con la ventana
        cerrada. pystray es opcional: sin el, cerrar la ventana cierra la app."""
        try:
            import pystray
            from PIL import ImageDraw
        except Exception:
            self._tray = None
            return

        image = Image.new("RGB", (64, 64), ACCENT_COLOR)
        draw = ImageDraw.Draw(image)
        draw.ellipse((18, 8, 46, 36), fill=GREEN)        # cabeza
        draw.rectangle((26, 36, 38, 58), fill=GREEN)     # torso
        menu = pystray.Menu(
            pystray.MenuItem("Mostrar ventana", self._tray_show, default=True),
            pystray.MenuItem("Salir", self._tray_quit),
        )
        self._tray = pystray.Icon("postura", image, "Detector de Postura IA", menu)
        try:
            self._tray.run_detached()
        except Exception:
            # Algunos entornos no permiten el hilo detached; sin bandeja la app
            # funciona igual.
            self._tray = None

    def _tray_show(self, *_args):
        self.root.after(0, self._restore_window)

    def _restore_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _tray_quit(self, *_args):
        self.root.after(0, self._quit)

    # ------------------------------------------------------------------
    def _toggle_voice(self):
        self.voice.set_enabled(self.voice_var.get())
        self.cfg.update({"voice_enabled": self.voice.enabled})

    def _toggle_privacy(self):
        self.cfg.update({"privacy_mode": bool(self.privacy_var.get())})

    # ------------------------------------------------------------------
    def _start_preview(self):
        """Enciende la camara al abrir la app (antes de presionar INICIAR) y
        saluda al usuario, para que pueda ubicarse frente a la camara."""
        if not self.model_loaded:
            self.status_var.set("Estado: modelo no disponible")
            self.start_btn.config(state=tk.DISABLED)
            return
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            self.status_var.set(f"Error: no se pudo abrir la cámara {self.cfg.camera_index}")
            self.start_btn.config(state=tk.DISABLED)
            return
        self.voice.say_now(
            "Bienvenido al detector de postura. Ubícate frente a la cámara y "
            "presiona Iniciar cuando estés listo para la evaluación de treinta segundos."
        )
        self.running = True
        self._update_frame()

    def _start(self):
        if not self.model_loaded or self.cap is None:
            return
        self.predictor.reset()
        self.coach.reset()
        self.last_verdict = None
        self.verdict_hold_until = 0.0
        self.assessment.duration = self.cfg.assessment_seconds
        self.assessment.start()
        self.session.start()
        self.breaks.start()
        self.testing_started = True
        self.voice.say_now(
            "Mantén tu postura habitual, sentado o de pie, durante los próximos "
            f"{int(self.cfg.assessment_seconds)} segundos. Voy a evaluarla."
        )
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def _stop(self):
        # Vuelve a la vista previa; la camara sigue encendida.
        self._save_session()
        self.testing_started = False
        self.predictor.reset()
        self.coach.reset()
        self.feedback_var.set("")
        self.info_var.set("")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _save_session(self):
        if self.history is None or self.session.samples == 0:
            return
        saved = self.history.save_stats(self.session)
        self.session.reset()
        if saved:
            self._refresh_history()

    # ------------------------------------------------------------------
    def _apply_calibration(self, result):
        """Usa la evaluacion de 30s como postura de referencia y ajusta los
        umbrales a esa persona."""
        thresholds = result.get("thresholds")
        if thresholds is None:
            return result.get("calibration_note", "")
        self.cfg.set_thresholds(thresholds)
        self.predictor.thresholds = thresholds
        self._refresh_settings_fields()
        return result.get("calibration_note", "")

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            self.testing_started = False
            self.cap.release()
            self.cap = None
            self.status_var.set("Error: se perdió la conexión con la cámara")
            self.status_label.config(bg=RED)
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            return

        if self.cfg.flip_camera:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        status_text = "Estado: Sin detectar"
        feedback_text = ""
        bg_color = GRAY
        dot_color, line_color = (150, 150, 150), (100, 100, 100)
        context_text = ""

        if not self.testing_started:
            # --- Fase 0: vista previa, esperando a que presiones INICIAR ---
            status_text = "Estado:  Presiona INICIAR cuando estés listo"
            bg_color = ACCENT_COLOR
            feedback_text = "Ubícate frente a la cámara, sentado o de pie, como prefieras"

            if results.pose_landmarks:
                dot_color, line_color = (150, 150, 220), (110, 110, 180)
            else:
                self.predictor.reset()

        elif self.assessment.active:
            # --- Fase 1: evaluacion inicial, sin corregir en vivo ---
            remaining = int(self.assessment.remaining) + 1
            status_text = f"Estado:  EVALUANDO...  {remaining}s"
            bg_color = ACCENT_COLOR
            feedback_text = "Mantén tu postura habitual, sentado o de pie"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                is_correct, raw_feedback = self.predictor.predict(landmarks)
                context = detect_context(landmarks)
                context_text = CONTEXT_LABELS[context]
                aspects = evaluate_aspects(landmarks, self.cfg.thresholds())
                metrics = core.metrics_from_keypoints(core.keypoints_from_landmarks(landmarks))
                self.assessment.record(is_correct, raw_feedback, context, aspects, metrics)
                self.session.record(is_correct, raw_feedback)
                dot_color, line_color = (241, 196, 15), (243, 156, 18)
            else:
                self.predictor.reset()

            if self.assessment.time_up:
                result = self.assessment.finish(base_thresholds=self.cfg.thresholds())
                nota = self._apply_calibration(result)
                result["calibration_note"] = nota
                self.last_verdict = result
                self.verdict_hold_until = time.time() + 10.0
                self.voice.say_now(result["verdict"])

        elif time.time() < self.verdict_hold_until and self.last_verdict is not None:
            # --- Fase 2: mostrar el veredicto + checklist de aspectos ---
            result = self.last_verdict
            status_text = "Estado:  BUENA POSTURA  ✓" if result["good"] else "Estado:  A MEJORAR  ✗"
            bg_color = GREEN if result["good"] else RED
            lineas = [f"{result['percent_correct']:.0f}% correcta en la evaluación"]
            lineas += summarize_aspects(result["aspect_percents"])
            if result.get("calibration_note"):
                lineas.append(result["calibration_note"])
            feedback_text = "\n".join(lineas)
            dot_color, line_color = (46, 204, 113), (39, 174, 96)

        elif results.pose_landmarks:
            # --- Fase 3: coaching continuo (comportamiento normal) ---
            landmarks = results.pose_landmarks.landmark
            is_correct, feedback_text = self.predictor.predict(landmarks)
            context_text = CONTEXT_LABELS[detect_context(landmarks)]
            self.session.record(is_correct, feedback_text)

            if is_correct is None:
                status_text = "Estado:  Ponte de frente a la cámara"
                bg_color = ACCENT_COLOR
            elif is_correct:
                status_text = "Estado:  CORRECTA  ✓"
                bg_color = GREEN
                dot_color, line_color = (46, 204, 113), (39, 174, 96)
            else:
                status_text = "Estado:  INCORRECTA  ✗"
                bg_color = RED
                dot_color, line_color = (231, 76, 60), (192, 57, 43)
                segundos = self.coach.bad_seconds
                if 0 < segundos < self.cfg.alert_after_seconds:
                    status_text += f"   ({segundos:.0f}s)"

            self.coach.update(is_correct, feedback_text)

        else:
            self.predictor.reset()
            self.coach.reset()

        # --- Recordatorio de pausas (independiente de la postura) ---
        if self.testing_started and self.breaks.due():
            mensaje = self.breaks.message()
            self.voice.say_now(mensaje)
            self.notifier.notify("Hora de una pausa", mensaje, force=True)
            feedback_text = mensaje

        if self.privacy_var.get():
            # Modo privacidad: solo el esqueleto sobre fondo negro.
            frame = np.zeros_like(frame)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=dot_color[::-1], thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=line_color[::-1], thickness=2),
            )
        elif self.privacy_var.get():
            cv2.putText(frame, "Sin deteccion", (20, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 160), 2)

        self.status_var.set(status_text)
        self.status_label.config(bg=bg_color)
        self.feedback_var.set(feedback_text)

        pie = [context_text, self.session.summary_text()]
        if self.testing_started and self.breaks.enabled:
            pie.append(f"Pausa en {self.breaks.minutes_until_next:.0f} min")
        self.info_var.set("   ·   ".join(filter(None, pie)))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((self.video_w, self.video_h), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(15, self._update_frame)

    # ------------------------------------------------------------------
    def _on_close(self):
        """Con bandeja disponible, la X minimiza y el detector sigue avisando.
        Sin bandeja, cierra la app."""
        if self._tray is not None:
            self.root.withdraw()
            self.notifier.notify(
                "Sigo trabajando",
                "El detector quedó en la bandeja del sistema. Doble clic para volver.",
            )
            return
        self._quit()

    def _quit(self):
        self.running = False
        self._save_session()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._tray is not None:
            try:
                self._tray.stop()
            except Exception:
                pass
            self._tray = None
        self.voice.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PostureApp(root)
    root.mainloop()
