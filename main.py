"""Detector de postura en consola (ventana de OpenCV).

Es la version minima: para la experiencia completa (evaluacion de 30s,
calibracion, historial, modo privacidad) usa app_gui.py o app_web.py.

predict.py era un duplicado casi identico de este archivo y se elimino.
"""
import unicodedata

import cv2

import core
from config import AppConfig
from history import SessionHistory
from posture_metrics import CONTEXT_LABELS, SessionStats, detect_context
from voice_feedback import BreakReminder, DesktopNotifier, PostureVoiceCoach, VoiceFeedback


def _ascii(text):
    """Las fuentes de OpenCV no dibujan tildes ni la n con tilde (salen como
    '?'), asi que el texto de la ventana se pasa a ASCII."""
    plano = unicodedata.normalize("NFKD", text)
    return "".join(c for c in plano if not unicodedata.combining(c))


def main():
    cfg = AppConfig.load()

    try:
        predictor = core.PosturePredictor.load(thresholds=cfg.thresholds())
    except core.ModelError as exc:
        print(f"Error: {exc}")
        return

    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
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
    session.start()
    breaks.start()
    history = SessionHistory() if cfg.save_history else None

    print("=== DETECTOR DE POSTURA IA ===")
    print(f"Avisos tras {cfg.alert_after_seconds:.0f}s de mala postura sostenida.")
    print("Presiona 'q' para salir, 'p' para el modo privacidad.")

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la camara {cfg.camera_index}.")
        voice.stop()
        return

    privacy = cfg.privacy_mode

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cfg.flip_camera:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "Sin detectar"
        feedback = ""
        color = (100, 100, 100)
        context_text = ""

        if privacy:
            # Modo privacidad: se descarta el video y solo queda el esqueleto
            # sobre fondo negro.
            frame = frame * 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            is_correct, feedback = predictor.predict(landmarks)
            context_text = CONTEXT_LABELS[detect_context(landmarks)]
            session.record(is_correct, feedback)

            if is_correct is None:
                status = "Ponte de frente"
            elif is_correct:
                status = "CORRECTA"
                color = (0, 255, 0)
            else:
                status = "INCORRECTA"
                color = (0, 0, 255)

            coach.update(is_correct, feedback)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            predictor.reset()
            coach.reset()

        if breaks.due():
            mensaje = breaks.message()
            voice.say_now(mensaje)
            notifier.notify("Hora de una pausa", mensaje, force=True)
            print(f"[pausa] {mensaje}")

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 90), color, -1)
        cv2.putText(frame, f"Postura: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, _ascii(feedback), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        pie = [context_text, session.summary_text()]
        if privacy:
            pie.append("Privacidad ON")
        info_line = "  |  ".join(filter(None, pie))
        if info_line:
            cv2.putText(frame, _ascii(info_line), (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Detector de Postura IA", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            privacy = not privacy
            print(f"Modo privacidad: {'ON' if privacy else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    if history is not None:
        saved = history.save_stats(session)
        if saved:
            print(
                f"Sesion guardada en el historial: {session.elapsed_minutes:.1f} min, "
                f"{session.percent_correct:.0f}% correcta."
            )
    voice.stop()


if __name__ == "__main__":
    main()
