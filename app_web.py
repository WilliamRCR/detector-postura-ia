import time

import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
from flask import Flask, Response, jsonify, render_template

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

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

try:
    model = joblib.load("modelo_postura.pkl")
    scaler = joblib.load("scaler.pkl")
    MODEL_OK = True
except FileNotFoundError:
    MODEL_OK = False

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

voice = VoiceFeedback()
coach = PostureVoiceCoach(voice)
session = SessionStats()
assessment = PostureAssessment(duration=30.0)

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
    "voice_available": voice.enabled,
    "voice_on": voice.enabled,
}
state_lock = threading.Lock()

cap = None
capture_thread = None
latest_frame = None
frame_lock = threading.Lock()


def get_keypoints(landmarks):
    kp = []
    for lm in landmarks:
        kp.extend([lm.x, lm.y, lm.z])
    return np.array(kp)


def predict(landmarks, status_smoother, feedback_smoother):
    kp = get_keypoints(landmarks)
    kp_scaled = scaler.transform([kp])
    pred = model.predict(kp_scaled)[0]
    is_correct = status_smoother.update(pred) == 1
    feedback_text = feedback_smoother.update(evaluate_posture(landmarks))
    return is_correct, feedback_text


def capture_loop():
    """Corre continuamente desde que arranca el servidor: la camara queda
    activa (vista previa) aunque nadie haya presionado INICIAR todavia."""
    global cap, latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with state_lock:
            state["running"] = False
            state["camera_error"] = "No se pudo abrir la cámara"
        return

    status_smoother = MajorityVoteSmoother(window=7)
    feedback_smoother = MajorityVoteSmoother(window=5)
    last_verdict = None
    verdict_hold_until = 0.0

    voice.say_now(
        "Bienvenido al detector de postura. Ubícate frente a la cámara y "
        "presiona Iniciar cuando estés listo para la evaluación de treinta segundos."
    )

    while True:
        with state_lock:
            running = state["running"]
            testing_started = state["testing_started"]
        if not running:
            break

        ret, frame = cap.read()
        if not ret:
            with state_lock:
                state["running"] = False
                state["camera_error"] = "Se perdió la conexión con la cámara"
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "Sin detectar"
        feedback = ""
        color = "gray"
        dot, line = (150, 150, 150), (100, 100, 100)
        context_text = ""

        if not MODEL_OK:
            pass

        elif not testing_started:
            # --- Fase 0: vista previa, esperando a que presionen INICIAR ---
            status = "Presiona INICIAR cuando estés listo"
            color = "blue"
            feedback = "Ubícate frente a la cámara, sentado o de pie, como prefieras"

            if results.pose_landmarks:
                dot, line = (150, 150, 220), (110, 110, 180)
            else:
                status_smoother.reset()
                feedback_smoother.reset()

        elif assessment.active:
            # --- Fase 1: evaluacion inicial de 30s, sin corregir en vivo ---
            remaining = int(assessment.remaining) + 1
            status = f"EVALUANDO... {remaining}s"
            color = "blue"
            feedback = "Mantén tu postura habitual, sentado o de pie"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                is_correct, raw_feedback = predict(landmarks, status_smoother, feedback_smoother)
                context = detect_context(landmarks)
                context_text = CONTEXT_LABELS[context]
                aspects = evaluate_aspects(landmarks)
                assessment.record(is_correct, raw_feedback, context, aspects)
                session.record(is_correct)
                dot, line = (241, 196, 15), (243, 156, 18)
            else:
                status_smoother.reset()
                feedback_smoother.reset()

            if assessment.time_up:
                result = assessment.finish()
                last_verdict = result
                verdict_hold_until = time.time() + 10.0
                voice.say_now(result["verdict"])

        elif time.time() < verdict_hold_until and last_verdict is not None:
            # --- Fase 2: mostrar el veredicto + checklist de aspectos ---
            result = last_verdict
            status = "BUENA POSTURA" if result["good"] else "A MEJORAR"
            color = "green" if result["good"] else "red"
            checklist = summarize_aspects(result["aspect_percents"])
            feedback = "\n".join([f"{result['percent_correct']:.0f}% correcta en la evaluación"] + checklist)
            dot, line = (46, 204, 113), (39, 174, 96)

        elif results.pose_landmarks:
            # --- Fase 3: coaching continuo (comportamiento normal) ---
            landmarks = results.pose_landmarks.landmark
            is_correct, feedback = predict(landmarks, status_smoother, feedback_smoother)
            context_text = CONTEXT_LABELS[detect_context(landmarks)]
            session.record(is_correct)

            if is_correct:
                status = "CORRECTA"
                color = "green"
                dot, line = (46, 204, 113), (39, 174, 96)
            else:
                status = "INCORRECTA"
                color = "red"
                dot, line = (231, 76, 60), (192, 57, 43)

            coach.update(is_correct, feedback)

        else:
            status_smoother.reset()
            feedback_smoother.reset()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=dot[::-1], thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=line[::-1], thickness=2),
            )

        with state_lock:
            state["status"] = status
            state["feedback"] = feedback
            state["color"] = color
            state["context"] = context_text
            state["session_info"] = session.summary_text()

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buffer.tobytes()

    cap.release()
    with frame_lock:
        latest_frame = None


def generate_frames():
    while True:
        with state_lock:
            running = state["running"]
        if not running:
            break
        with frame_lock:
            frame = latest_frame
        if frame is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html", model_ok=MODEL_OK, voice_available=voice.enabled)


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
        )


@app.route("/api/voice", methods=["POST"])
def api_voice():
    with state_lock:
        state["voice_on"] = not state["voice_on"]
        voice.set_enabled(state["voice_on"])
        voice_on = state["voice_on"]
    return jsonify(ok=True, voice_on=voice_on)


@app.route("/api/start", methods=["POST"])
def api_start():
    if not MODEL_OK:
        return jsonify(ok=False, error="Modelo no encontrado")
    with state_lock:
        if not state["running"]:
            return jsonify(ok=False, error=state["camera_error"] or "La cámara no está disponible")
        if state["testing_started"]:
            return jsonify(ok=True)
        state["testing_started"] = True
        state["status"] = "Sin detectar"
        state["feedback"] = ""
        state["color"] = "gray"
        state["context"] = ""
        state["session_info"] = ""
    coach.reset()
    assessment.start()
    session.start()
    voice.say_now(
        "Mantén tu postura habitual, sentado o de pie, durante los próximos "
        "treinta segundos. Voy a evaluarla."
    )
    return jsonify(ok=True)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state["testing_started"] = False
        state["status"] = "Sin detectar"
        state["feedback"] = ""
        state["color"] = "gray"
        state["context"] = ""
        state["session_info"] = ""
    return jsonify(ok=True)


if __name__ == "__main__":
    with state_lock:
        state["running"] = True
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    print("Abre tu navegador en: http://localhost:5000")
    app.run(debug=False, threaded=True)
