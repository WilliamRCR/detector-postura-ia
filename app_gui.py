import time

import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageTk

from assessment import PostureAssessment
from posture_metrics import (
    CONTEXT_LABELS,
    SessionStats,
    detect_context,
    evaluate_aspects,
    evaluate_posture,
    summarize_aspects,
)
from smoothing import MajorityVoteSmoother
from voice_feedback import VoiceFeedback, PostureVoiceCoach

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
ACCENT_COLOR = "#0f3460"
GREEN = "#2ecc71"
RED = "#e74c3c"
GRAY = "#555577"
TEXT_COLOR = "#e0e0e0"


class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Postura IA")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.running = False
        self.testing_started = False
        self.cap = None

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            self.model = joblib.load("modelo_postura.pkl")
            self.scaler = joblib.load("scaler.pkl")
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False

        self.status_smoother = MajorityVoteSmoother(window=7)
        self.feedback_smoother = MajorityVoteSmoother(window=5)
        self.voice = VoiceFeedback()
        self.coach = PostureVoiceCoach(self.voice)
        self.voice_var = tk.BooleanVar(value=self.voice.enabled)

        self.assessment = PostureAssessment(duration=30.0)
        self.last_verdict = None
        self.verdict_hold_until = 0.0
        self.session = SessionStats()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_preview()

    def _build_ui(self):
        # --- Header ---
        header = tk.Frame(self.root, bg=ACCENT_COLOR, pady=14)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="DETECTOR DE POSTURA  IA",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg=ACCENT_COLOR,
        ).pack()

        # --- Video ---
        video_frame = tk.Frame(self.root, bg=BG_COLOR, pady=12)
        video_frame.pack()
        self.video_label = tk.Label(
            video_frame, bg="#000000", width=640, height=480
        )
        self.video_label.pack()

        # --- Estado ---
        self.status_var = tk.StringVar(value="Estado: Sin detectar")
        self.status_label = tk.Label(
            self.root,
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

        # --- Feedback ---
        self.feedback_var = tk.StringVar(value="")
        tk.Label(
            self.root,
            textvariable=self.feedback_var,
            font=("Segoe UI", 12),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            pady=4,
            justify=tk.LEFT,
        ).pack()

        # --- Contexto + estadisticas de sesion ---
        self.info_var = tk.StringVar(value="")
        tk.Label(
            self.root,
            textvariable=self.info_var,
            font=("Segoe UI", 10),
            fg="#8888aa",
            bg=BG_COLOR,
            pady=2,
        ).pack()

        # --- Botones ---
        btn_frame = tk.Frame(self.root, bg=BG_COLOR, pady=14)
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
        self.start_btn.pack(side=tk.LEFT, padx=12)

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
        self.stop_btn.pack(side=tk.LEFT, padx=12)

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
            state=tk.NORMAL if self.voice.enabled else tk.DISABLED,
        )
        self.voice_check.pack(side=tk.LEFT, padx=12)

        if not self.model_loaded:
            tk.Label(
                self.root,
                text="Modelo no encontrado. Ejecuta train.py primero.",
                fg=RED,
                bg=BG_COLOR,
                font=("Segoe UI", 10),
                pady=6,
            ).pack()

        # --- Footer ---
        tk.Label(
            self.root,
            text="MediaPipe + scikit-learn  |  DETENER vuelve a la vista previa, cierra la ventana para apagar la cámara",
            font=("Segoe UI", 9),
            fg="#666688",
            bg=BG_COLOR,
            pady=8,
        ).pack()

    # ------------------------------------------------------------------
    def _get_keypoints(self, landmarks):
        kp = []
        for lm in landmarks:
            kp.extend([lm.x, lm.y, lm.z])
        return np.array(kp)

    def _predict(self, landmarks):
        kp = self._get_keypoints(landmarks)
        kp_scaled = self.scaler.transform([kp])
        pred = self.model.predict(kp_scaled)[0]
        is_correct = self.status_smoother.update(pred) == 1
        feedback_text = self.feedback_smoother.update(evaluate_posture(landmarks))
        return is_correct, feedback_text

    # ------------------------------------------------------------------
    def _toggle_voice(self):
        self.voice.set_enabled(self.voice_var.get())

    # ------------------------------------------------------------------
    def _start_preview(self):
        """Enciende la camara al abrir la app (antes de presionar INICIAR) y
        saluda al usuario, para que pueda ubicarse frente a la camara."""
        if not self.model_loaded:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Error: no se pudo abrir la cámara")
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
        self.status_smoother.reset()
        self.feedback_smoother.reset()
        self.coach.reset()
        self.last_verdict = None
        self.verdict_hold_until = 0.0
        self.assessment.start()
        self.session.start()
        self.testing_started = True
        self.voice.say_now(
            "Mantén tu postura habitual, sentado o de pie, durante los próximos "
            "treinta segundos. Voy a evaluarla."
        )
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def _stop(self):
        # Vuelve a la vista previa; la camara sigue encendida.
        self.testing_started = False
        self.status_smoother.reset()
        self.feedback_smoother.reset()
        self.feedback_var.set("")
        self.info_var.set("")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

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
                self.status_smoother.reset()
                self.feedback_smoother.reset()

        elif self.assessment.active:
            # --- Fase 1: evaluacion inicial de 30s, sin corregir en vivo ---
            remaining = int(self.assessment.remaining) + 1
            status_text = f"Estado:  EVALUANDO...  {remaining}s"
            bg_color = ACCENT_COLOR
            feedback_text = "Mantén tu postura habitual, sentado o de pie"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                is_correct, raw_feedback = self._predict(landmarks)
                context = detect_context(landmarks)
                context_text = CONTEXT_LABELS[context]
                aspects = evaluate_aspects(landmarks)
                self.assessment.record(is_correct, raw_feedback, context, aspects)
                self.session.record(is_correct)
                dot_color, line_color = (241, 196, 15), (243, 156, 18)
            else:
                self.status_smoother.reset()
                self.feedback_smoother.reset()

            if self.assessment.time_up:
                result = self.assessment.finish()
                self.last_verdict = result
                self.verdict_hold_until = time.time() + 10.0
                self.voice.say_now(result["verdict"])

        elif time.time() < self.verdict_hold_until and self.last_verdict is not None:
            # --- Fase 2: mostrar el veredicto + checklist de aspectos ---
            result = self.last_verdict
            status_text = "Estado:  BUENA POSTURA  ✓" if result["good"] else "Estado:  A MEJORAR  ✗"
            bg_color = GREEN if result["good"] else RED
            checklist = summarize_aspects(result["aspect_percents"])
            feedback_text = "\n".join([f"{result['percent_correct']:.0f}% correcta en la evaluación"] + checklist)
            dot_color, line_color = (46, 204, 113), (39, 174, 96)

        elif results.pose_landmarks:
            # --- Fase 3: coaching continuo (comportamiento normal) ---
            landmarks = results.pose_landmarks.landmark
            is_correct, feedback_text = self._predict(landmarks)
            context_text = CONTEXT_LABELS[detect_context(landmarks)]
            self.session.record(is_correct)

            if is_correct:
                status_text = "Estado:  CORRECTA  ✓"
                bg_color = GREEN
                dot_color, line_color = (46, 204, 113), (39, 174, 96)
            else:
                status_text = "Estado:  INCORRECTA  ✗"
                bg_color = RED
                dot_color, line_color = (231, 76, 60), (192, 57, 43)

            self.coach.update(is_correct, feedback_text)

        else:
            self.status_smoother.reset()
            self.feedback_smoother.reset()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=dot_color[::-1], thickness=2, circle_radius=4
                ),
                mp_drawing.DrawingSpec(
                    color=line_color[::-1], thickness=2
                ),
            )

        # Actualizar labels
        self.status_var.set(status_text)
        self.status_label.config(bg=bg_color)
        self.feedback_var.set(feedback_text)
        self.info_var.set("   ·   ".join(filter(None, [context_text, self.session.summary_text()])))

        # Mostrar frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((640, 480), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(15, self._update_frame)

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.voice.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PostureApp(root)
    root.mainloop()
