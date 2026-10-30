"""Version web (Flask): mismo detector, en el navegador.

Incluye evaluacion inicial con calibracion personal, avisos por tiempo
sostenido, recordatorios de pausa, modo privacidad, ajustes editables y la
grafica semanal del historial.
"""
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import core
from assessment import PostureAssessment, verdict_lines
from config import ANCHOR_KEY, AppConfig
from history import SessionHistory
from posture_metrics import SessionStats, detect_context, evaluate_aspects, scene_label
from voice_feedback import BreakReminder, DesktopNotifier, PostureVoiceCoach, VoiceFeedback

app = Flask(__name__)

cfg = AppConfig.load()

try:
    predictor = core.PosturePredictor.load(thresholds=cfg.thresholds())
    MODEL_OK = True
    MODEL_ERROR = ""
except core.ModelError as exc:
    predictor = None
    MODEL_OK = False
    MODEL_ERROR = str(exc)

pose = core.new_pose()

voice = VoiceFeedback(enabled=cfg.voice_enabled, rate=cfg.voice_rate, cooldown=cfg.voice_cooldown)
notifier = DesktopNotifier(enabled=cfg.desktop_notifications)
coach = PostureVoiceCoach(
    voice,
    min_bad_seconds=cfg.alert_after_seconds,
    repeat_seconds=cfg.alert_repeat_seconds,
    notifier=notifier,
)
breaks = BreakReminder(interval_minutes=cfg.break_minutes, enabled=cfg.break_reminders)
session = SessionStats()
assessment = PostureAssessment(duration=cfg.assessment_seconds)
history = SessionHistory()

# Estado compartido entre hilos
state = {
    "running": False,          # camara activa (vista previa)
    "testing_started": False,  # evaluacion/coaching en curso (tras INICIAR)
    "status": "Sin detectar",
    "feedback": "",
    "color": "gray",
    "context": "",
    "session_info": "",
    "camera_error": "",
    "voice_available": voice.available,
    "voice_on": voice.enabled,
    "privacy": cfg.privacy_mode,
    "break_in": None,
    "calibration_note": "",
}
state_lock = threading.Lock()

cap = None
capture_thread = None
latest_frame = None
frame_lock = threading.Lock()


def _apply_config():
    """Aplica config.json en caliente a los objetos ya creados."""
    if predictor is not None:
        predictor.thresholds = cfg.thresholds()
    voice.set_enabled(cfg.voice_enabled)
    voice.set_rate(cfg.voice_rate)
    notifier.set_enabled(cfg.desktop_notifications)
    coach.configure(
        min_bad_seconds=cfg.alert_after_seconds,
        repeat_seconds=cfg.alert_repeat_seconds,
    )
    breaks.interval_minutes = cfg.break_minutes
    breaks.enabled = cfg.break_reminders
    assessment.duration = cfg.assessment_seconds
    with state_lock:
        state["voice_on"] = voice.enabled
        state["privacy"] = cfg.privacy_mode


def _save_session():
    if not cfg.save_history or session.samples == 0:
        return
    history.save_stats(session)
    session.reset()


def capture_loop():
    """Corre continuamente desde que arranca el servidor: la camara queda
    activa (vista previa) aunque nadie haya presionado INICIAR todavia."""
    global cap, latest_frame
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        with state_lock:
            state["running"] = False
            state["camera_error"] = f"No se pudo abrir la cámara {cfg.camera_index}"
        return

    last_verdict = None
    verdict_hold_until = 0.0

    voice.say_now(
        "Bienvenido al detector de postura. Ubícate frente a la cámara y presiona "
        f"Iniciar cuando estés listo para la evaluación de {int(cfg.assessment_seconds)} segundos."
    )

    while True:
        with state_lock:
            running = state["running"]
            testing_started = state["testing_started"]
            privacy = state["privacy"]
        if not running:
            break

        ret, frame = cap.read()
        if not ret:
            with state_lock:
                state["running"] = False
                state["camera_error"] = "Se perdió la conexión con la cámara"
            break

        if cfg.flip_camera:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "Sin detectar"
        feedback = ""
        color = "gray"
        dot, line = (150, 150, 150), (100, 100, 100)
        context_text = ""
        calibration_note = ""

        if not MODEL_OK:
            status = "Modelo no disponible"
            feedback = MODEL_ERROR

        elif not testing_started:
            # --- Fase 0: vista previa, esperando a que presionen INICIAR ---
            status = "Presiona INICIAR cuando estés listo"
            color = "blue"
            feedback = "Ubícate frente a la cámara, sentado o de pie, como prefieras"

            if results.pose_landmarks:
                dot, line = (150, 150, 220), (110, 110, 180)
            else:
                predictor.reset()

        elif assessment.active:
            # --- Fase 1: evaluacion inicial, sin corregir en vivo ---
            remaining = int(assessment.remaining) + 1
            status = f"EVALUANDO... {remaining}s"
            color = "blue"
            feedback = "Mantén tu postura habitual, sentado o de pie"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                is_correct, raw_feedback = predictor.predict(landmarks)
                context = detect_context(landmarks)
                context_text = scene_label(landmarks)
                aspects = evaluate_aspects(landmarks, cfg.thresholds())
                metrics = core.metrics_from_keypoints(core.keypoints_from_landmarks(landmarks))
                assessment.record(is_correct, raw_feedback, context, aspects, metrics)
                session.record(is_correct, raw_feedback)
                dot, line = (241, 196, 15), (243, 156, 18)
                if is_correct is None:
                    feedback = "No te puedo medir: ubícate de frente o de perfil"
            else:
                predictor.reset()

            if assessment.time_up:
                result = assessment.finish(
                    base_thresholds=cfg.thresholds(), anchor_thresholds=cfg.anchor_thresholds()
                )
                thresholds = result.get("thresholds")
                if thresholds is not None:
                    cfg.set_thresholds(thresholds)
                    predictor.thresholds = thresholds
                last_verdict = result
                verdict_hold_until = time.time() + 10.0
                voice.say_now(result["verdict"])

        elif time.time() < verdict_hold_until and last_verdict is not None:
            # --- Fase 2: mostrar el veredicto + checklist de aspectos ---
            result = last_verdict
            if not result["samples"]:
                status, color = "SIN DATOS", "blue"
            elif result["good"]:
                status, color = "BUENA POSTURA", "green"
            else:
                status, color = "A MEJORAR", "red"
            calibration_note = result.get("calibration_note", "")
            feedback = "\n".join(verdict_lines(result))
            dot, line = (46, 204, 113), (39, 174, 96)

        elif results.pose_landmarks:
            # --- Fase 3: coaching continuo (comportamiento normal) ---
            landmarks = results.pose_landmarks.landmark
            is_correct, feedback = predictor.predict(landmarks)
            context_text = scene_label(landmarks)
            session.record(is_correct, feedback)

            if is_correct is None:
                status = "Ubícate de frente o de perfil"
                color = "blue"
            elif is_correct:
                status = "CORRECTA"
                color = "green"
                dot, line = (46, 204, 113), (39, 174, 96)
            else:
                status = "INCORRECTA"
                color = "red"
                dot, line = (231, 76, 60), (192, 57, 43)
                segundos = coach.bad_seconds
                if 0 < segundos < cfg.alert_after_seconds:
                    status += f"  ({segundos:.0f}s)"

            coach.update(is_correct, feedback)

        else:
            predictor.reset()
            coach.reset()

        # --- Recordatorio de pausas (independiente de la postura) ---
        if testing_started and breaks.due():
            mensaje = breaks.message()
            voice.say_now(mensaje)
            notifier.notify("Hora de una pausa", mensaje, force=True)
            feedback = mensaje

        if privacy:
            # Modo privacidad: solo el esqueleto sobre fondo negro.
            frame = np.zeros_like(frame)

        if results.pose_landmarks:
            import mediapipe as mp

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=dot[::-1], thickness=2, circle_radius=4),
                mp.solutions.drawing_utils.DrawingSpec(color=line[::-1], thickness=2),
            )

        with state_lock:
            state["status"] = status
            state["feedback"] = feedback
            state["color"] = color
            state["context"] = context_text
            state["session_info"] = session.summary_text()
            state["break_in"] = (
                round(breaks.minutes_until_next) if testing_started and breaks.enabled else None
            )
            if calibration_note:
                state["calibration_note"] = calibration_note

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buffer.tobytes()

    cap.release()
    with frame_lock:
        latest_frame = None


def generate_frames():
    last_sent = None
    while True:
        with state_lock:
            running = state["running"]
        if not running:
            break
        with frame_lock:
            frame = latest_frame
        if frame is None or frame is last_sent:
            # Sin frame nuevo: una pausa corta evita quemar un nucleo entero
            # girando en vacio.
            time.sleep(0.01)
            continue
        last_sent = frame
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ok=MODEL_OK,
        model_error=MODEL_ERROR,
        voice_available=voice.available,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(
            running=state["running"],
            testing_started=state["testing_started"],
            status=state["status"],
            feedback=state["feedback"],
            color=state["color"],
            context=state["context"],
            session_info=state["session_info"],
            camera_error=state["camera_error"],
            voice_on=state["voice_on"],
            privacy=state["privacy"],
            break_in=state["break_in"],
        )


@app.route("/api/voice", methods=["POST"])
def api_voice():
    with state_lock:
        state["voice_on"] = not state["voice_on"]
        voice.set_enabled(state["voice_on"])
        voice_on = state["voice_on"]
    cfg.update({"voice_enabled": voice_on})
    return jsonify(ok=True, voice_on=voice_on)


@app.route("/api/privacy", methods=["POST"])
def api_privacy():
    with state_lock:
        state["privacy"] = not state["privacy"]
        privacy = state["privacy"]
    cfg.update({"privacy_mode": privacy})
    return jsonify(ok=True, privacy=privacy)


@app.route("/api/start", methods=["POST"])
def api_start():
    if not MODEL_OK:
        return jsonify(ok=False, error=MODEL_ERROR or "Modelo no encontrado")
    with state_lock:
        if not state["running"]:
            return jsonify(ok=False, error=state["camera_error"] or "La cámara no está disponible")
        if state["testing_started"]:
            return jsonify(ok=True)
    # Primero se prepara todo y recien despues se avisa al hilo de captura: al
    # reves, un frame caia en coaching continuo antes de empezar la evaluacion.
    predictor.reset()
    coach.reset()
    assessment.duration = cfg.assessment_seconds
    assessment.start()
    session.start()
    breaks.start()
    with state_lock:
        state["testing_started"] = True
        state["status"] = "Sin detectar"
        state["feedback"] = ""
        state["color"] = "gray"
        state["context"] = ""
        state["session_info"] = ""
        state["calibration_note"] = ""
    voice.say_now(
        "Mantén tu postura habitual, sentado o de pie, durante los próximos "
        f"{int(cfg.assessment_seconds)} segundos. Voy a evaluarla."
    )
    return jsonify(ok=True)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    # Primero se detiene el hilo de captura y luego se guarda: al reves podia
    # anotar un frame mas en la sesion ya guardada.
    with state_lock:
        state["testing_started"] = False
        state["status"] = "Sin detectar"
        state["feedback"] = ""
        state["color"] = "gray"
        state["context"] = ""
        state["session_info"] = ""
        state["break_in"] = None
    _save_session()
    if predictor is not None:
        predictor.reset()
    coach.reset()
    return jsonify(ok=True)


@app.route("/api/history")
def api_history():
    if not cfg.save_history:
        return jsonify(enabled=False, week=[], totals={})
    return jsonify(
        enabled=True,
        week=history.weekly_summary(days=7),
        totals=history.totals(days=7),
        recent=history.recent_sessions(limit=8),
    )


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "GET":
        return jsonify(ok=True, config=cfg.as_dict())

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}
    payload.pop(ANCHOR_KEY, None)  # esa referencia solo la escribe train.py
    # La pagina manda todos los campos en cada guardado, asi que se compara con
    # lo que hay: solo un cambio real de umbral deja de ser calibracion automatica.
    if cfg.thresholds_changed(payload):
        payload.setdefault("calibrated", False)
    camara_cambio = cfg.would_change("camera_index", payload.get("camera_index"))
    cfg.update(payload)
    _apply_config()
    aviso = "El cambio de cámara se aplica al reiniciar la app." if camara_cambio else ""
    return jsonify(ok=True, config=cfg.as_dict(), aviso=aviso)


if __name__ == "__main__":
    with state_lock:
        state["running"] = True
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    print("Abre tu navegador en: http://localhost:5000")
    try:
        app.run(debug=False, threaded=True)
    finally:
        with state_lock:
            state["running"] = False
        _save_session()
        voice.stop()
