"""Recolecta un dataset de posturas directamente desde la webcam, sin fotos manuales.

Graba tus landmarks de MediaPipe durante varios segundos mientras mantienes una
postura, los aumenta (espejo + ruido leve) y los agrega a dataset_posturas.npz.

Uso:
    python collect_data.py 1 20 --contexto sentado    # 1 = postura correcta
    python collect_data.py 2 20 --contexto de_pie     # 2 = postura incorrecta

Repite para cada combinacion de correcta/incorrecta x sentado/de pie: cuantas
mas variantes cubras, mejor generaliza el modelo.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

DATASET_FILE = "dataset_posturas.npz"
LOG_FILE = "dataset_sessions.json"
SAMPLE_INTERVAL = 0.4  # segundos entre muestras automaticas
COUNTDOWN = 3  # segundos de preparacion antes de empezar a grabar

# Pares de landmarks izquierda/derecha de MediaPipe Pose (33 puntos), para
# poder espejar una muestra intercambiando el lado correcto.
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def mirror_keypoints(kp):
    pts = kp.reshape(33, 3).copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    for left, right in MIRROR_PAIRS:
        pts[[left, right]] = pts[[right, left]]
    return pts.reshape(-1)


def jitter_keypoints(kp, std=0.01):
    return kp + np.random.normal(0, std, size=kp.shape).astype(kp.dtype)


def get_keypoints(landmarks):
    kp = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]
    return np.array(kp, dtype=np.float32)


def load_existing():
    if Path(DATASET_FILE).exists():
        data = np.load(DATASET_FILE)
        return list(data["X"]), list(data["y"])
    return [], []


def save_dataset(X, y):
    np.savez(DATASET_FILE, X=np.array(X), y=np.array(y))


def log_session(label_name, contexto, capturadas, guardadas):
    entry = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "etiqueta": label_name,
        "contexto": contexto,
        "muestras_capturadas": capturadas,
        "muestras_guardadas": guardadas,
    }
    log = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
    log.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Recolecta posturas desde la webcam.")
    parser.add_argument("label", type=int, choices=[1, 2], help="1 = postura correcta, 2 = postura incorrecta")
    parser.add_argument("duracion", type=int, nargs="?", default=20, help="segundos de grabacion (default 20)")
    parser.add_argument(
        "--contexto",
        choices=["sentado", "de_pie", "mixto"],
        default="mixto",
        help="solo para llevar registro de si cubriste sentado y de pie",
    )
    args = parser.parse_args()

    label_name = "CORRECTA" if args.label == 1 else "INCORRECTA"
    print(f"=== RECOLECCION DESDE WEBCAM: {label_name} ({args.contexto}) ===")
    print(f"Prepárate para mantener esa postura durante {args.duracion}s.")
    print("Presiona 'q' en cualquier momento para cancelar.")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    raw_samples = []
    start_time = None
    last_sample_at = 0.0
    program_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        now = time.time()
        elapsed_total = now - program_start

        if elapsed_total < COUNTDOWN:
            texto = f"Prepárate...  {COUNTDOWN - int(elapsed_total)}"
            color = (0, 165, 255)
        else:
            if start_time is None:
                start_time = now
            restante = args.duracion - (now - start_time)

            if restante <= 0:
                break

            if results.pose_landmarks and (now - last_sample_at) >= SAMPLE_INTERVAL:
                raw_samples.append(get_keypoints(results.pose_landmarks.landmark))
                last_sample_at = now

            texto = f"Grabando...  {restante:.0f}s  |  muestras: {len(raw_samples)}"
            color = (0, 255, 0)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, texto, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Recoleccion de Postura", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cancelado por el usuario.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if not raw_samples:
        print("No se capturo ninguna muestra (no se te detecto). Intenta de nuevo con mejor luz/encuadre.")
        return

    # Aumento de datos: cada muestra real aporta su espejo (lado contrario)
    # y una variante con ruido leve, multiplicando el dataset x3 sin grabar mas.
    X_new = []
    for kp in raw_samples:
        X_new.append(kp)
        X_new.append(mirror_keypoints(kp))
        X_new.append(jitter_keypoints(kp))

    label_value = 1 if args.label == 1 else 0
    y_new = [label_value] * len(X_new)

    X_all, y_all = load_existing()
    X_all.extend(X_new)
    y_all.extend(y_new)
    save_dataset(X_all, y_all)
    log_session(label_name, args.contexto, len(raw_samples), len(X_new))

    print(f"\nCapturadas {len(raw_samples)} muestras reales -> {len(X_new)} tras aumento (espejo + ruido).")
    print(f"Dataset total ahora: {len(X_all)} muestras.")
    print("Repite con las otras combinaciones (correcta/incorrecta x sentado/de_pie).")
    print("Cuando tengas suficientes, ejecuta: python train.py")


if __name__ == "__main__":
    main()
