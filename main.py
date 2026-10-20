import cv2
import mediapipe as mp
import numpy as np
import joblib

from posture_metrics import CONTEXT_LABELS, SessionStats, detect_context, evaluate_posture
from smoothing import MajorityVoteSmoother
from voice_feedback import VoiceFeedback, PostureVoiceCoach

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cargar modelo IA
try:
    model = joblib.load("modelo_postura.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: Modelo no encontrado.")
    exit()

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_keypoints(landmarks):
    keypoints = []

    for lm in landmarks:
        keypoints.extend([lm.x, lm.y, lm.z])

    return np.array(keypoints)

status_smoother = MajorityVoteSmoother(window=7)
feedback_smoother = MajorityVoteSmoother(window=5)
voice = VoiceFeedback()
coach = PostureVoiceCoach(voice)
session = SessionStats()
session.start()

print("=== DETECTOR DE POSTURA IA ===")
print("Presiona 'q' para salir")

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    status = "Sin detectar"
    feedback = ""
    color = (100, 100, 100)
    context_text = ""

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # IA
        kp = get_keypoints(landmarks)
        kp_scaled = scaler.transform([kp])

        pred = model.predict(kp_scaled)[0]

        is_correct = status_smoother.update(pred) == 1
        feedback = feedback_smoother.update(evaluate_posture(landmarks))
        context_text = CONTEXT_LABELS[detect_context(landmarks)]
        session.record(is_correct)

        if is_correct:
            status = "CORRECTA"
            color = (0, 255, 0)
        else:
            status = "INCORRECTA"
            color = (0, 0, 255)

        coach.update(is_correct, feedback)

        # Dibujar esqueleto
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    else:
        status_smoother.reset()
        feedback_smoother.reset()

    h, w, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (w, 90), color, -1)

    cv2.putText(
        frame,
        f"Postura: {status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        feedback,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    info_line = "  |  ".join(filter(None, [context_text, session.summary_text()]))
    if info_line:
        cv2.putText(frame, info_line, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Detector de Postura IA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
voice.stop()