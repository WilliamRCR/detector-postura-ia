import cv2
import mediapipe as mp
import numpy as np
import joblib

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

def get_feedback(landmarks):
    """
    Feedback simple y estable.
    """

    nose = landmarks[0]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_ear = landmarks[7]
    right_ear = landmarks[8]

    # Promedios
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    ear_y = (left_ear.y + right_ear.y) / 2

    shoulder_level = abs(left_shoulder.y - right_shoulder.y)

    # Hombros desnivelados
    if shoulder_level > 0.05:
        return "Endereza los hombros"

    # Cabeza muy adelante
    if nose.z < left_shoulder.z - 0.15:
        return "Retrasa un poco la cabeza"

    # Cuello inclinado
    if abs(ear_y - shoulder_y) < 0.08:
        return "Levanta un poco la cabeza"

    return "Buena postura"

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

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # IA
        kp = get_keypoints(landmarks)
        kp_scaled = scaler.transform([kp])

        pred = model.predict(kp_scaled)[0]

        if pred == 1:
            status = "CORRECTA"
            color = (0, 255, 0)
        else:
            status = "INCORRECTA"
            color = (0, 0, 255)

        # Feedback
        feedback = get_feedback(landmarks)

        # Dibujar esqueleto
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

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

    cv2.imshow("Detector de Postura IA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()