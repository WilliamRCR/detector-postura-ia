import cv2
import mediapipe as mp
import numpy as np
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

try:
    model = joblib.load("modelo_postura.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: Modelo no encontrado. Ejecuta train.py primero.")
    exit()

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_keypoints(landmarks, w, h):
    keypoints = []
    for landmark in landmarks:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)

print("=== DETECTOR DE POSTURA (RED NEURONAL) ===")
print("Presiona 'q' para salir")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    status = "Sin detectar"
    color = (100, 100, 100)
    
    if results.pose_landmarks:
        kp = get_keypoints(results.pose_landmarks.landmark, w, h)
        kp_scaled = scaler.transform([kp])
        pred = model.predict(kp_scaled)[0]
        
        if pred == 1:
            status = "CORRECTA"
            color = (0, 255, 0)
        else:
            status = "INCORRECTA"
            color = (0, 0, 255)
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.rectangle(frame, (0, 0), (w, 70), color, -1)
    cv2.putText(frame, f"Postura: {status}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.imshow("Detector de Postura IA", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()