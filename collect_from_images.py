import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)

def get_keypoints(landmarks):
    keypoints = []

    for landmark in landmarks:
        keypoints.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(keypoints)

def process_folder(folder_path, label):
    X = []
    y = []

    files = os.listdir(folder_path)

    for file in files:

        image_path = os.path.join(folder_path, file)

        image = cv2.imread(image_path)

        if image is None:
            print(f"No se pudo leer: {file}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:

            kp = get_keypoints(results.pose_landmarks.landmark)

            if len(kp) == 99:
                X.append(kp)
                y.append(label)

                print(f"[OK] {file}")

            else:
                print(f"[ERROR] Keypoints inválidos: {file}")

        else:
            print(f"[SIN POSTURA] {file}")

    return X, y

print("=== RECOLECCION DESDE IMAGENES ===")

X_correcta, y_correcta = process_folder(
    "dataset/correcta",
    1
)

X_incorrecta, y_incorrecta = process_folder(
    "dataset/incorrecta",
    0
)

X = np.array(X_correcta + X_incorrecta)
y = np.array(y_correcta + y_incorrecta)

print("\nDataset final:")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.savez(
    "dataset_posturas.npz",
    X=X,
    y=y
)

print("\nDataset guardado correctamente")