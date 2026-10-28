"""Recolecta un dataset de posturas directamente desde la webcam, sin fotos manuales.

Esta es la forma recomendada de armar el dataset: graba tus landmarks de
MediaPipe de frente a la camara, igual que como se usa la app. Las fotos de
dataset/ son casi todas de perfil, asi que el modelo aprendia una vista y se
usaba en otra.

Dos diferencias respecto a la version anterior:

- Guarda las muestras crudas, sin espejo ni ruido. El aumento lo hace train.py
  despues de dividir, para que la precision de prueba sea real.
- Guarda a que sesion pertenece cada muestra (grupo), para que train.py pueda
  dividir por sesion y no mezclar frames casi identicos entre entrenamiento y
  prueba.

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
import numpy as np

import core
from config import AppConfig

LOG_FILE = "dataset_sessions.json"
SAMPLE_INTERVAL = 0.4  # segundos entre muestras automaticas
COUNTDOWN = 3  # segundos de preparacion antes de empezar a grabar


def load_existing():
    """Devuelve (X, y, groups) del dataset existente, o listas vacias."""
    path = Path(core.DATASET_FILE)
    if not path.exists():
        return [], [], []
    data = np.load(path, allow_pickle=False)
    X = list(data["X"])
    y = list(data["y"])
    if "groups" in data.files:
        groups = list(np.asarray(data["groups"]).astype(str))
    else:
        # Dataset viejo sin grupos: cada muestra queda en su propio grupo para
        # no fingir que vienen de la misma sesion.
        groups = [f"legado_{i}" for i in range(len(X))]
    return X, y, groups


def save_dataset(X, y, groups):
    np.savez(
        core.DATASET_FILE,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        groups=np.asarray(groups, dtype="U64"),
    )


def log_session(session_id, label_name, contexto, capturadas, medibles):
    entry = {
        "sesion": session_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "etiqueta": label_name,
        "contexto": contexto,
        "muestras_capturadas": capturadas,
        "muestras_frontales": medibles,
    }
    log = []
    path = Path(LOG_FILE)
    if path.exists():
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            log = []
    if not isinstance(log, list):
        log = []
    log.append(entry)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


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
    parser.add_argument("--camara", type=int, default=None, help="indice de camara (default: config.json)")
    args = parser.parse_args()

    cfg = AppConfig.load()
    camera_index = args.camara if args.camara is not None else cfg.camera_index

    label_name = "CORRECTA" if args.label == 1 else "INCORRECTA"
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{label_name.lower()}_{args.contexto}"

    print(f"=== RECOLECCION DESDE WEBCAM: {label_name} ({args.contexto}) ===")
    print(f"Prepárate para mantener esa postura durante {args.duracion}s, de frente a la cámara.")
    print("Presiona 'q' en cualquier momento para cancelar.")

    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = core.new_pose(static=False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la camara {camera_index}.")
        return

    raw_samples = []
    descartadas_perfil = 0
    start_time = None
    last_sample_at = 0.0
    program_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cfg.flip_camera:
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
                kp = core.keypoints_from_landmarks(results.pose_landmarks.landmark)
                # Solo guardamos muestras frontales: son las que coinciden con
                # como se usa la app de verdad.
                if core.is_frontal(kp):
                    raw_samples.append(kp)
                else:
                    descartadas_perfil += 1
                last_sample_at = now

            aviso = f"  |  de perfil: {descartadas_perfil}" if descartadas_perfil else ""
            texto = f"Grabando...  {restante:.0f}s  |  muestras: {len(raw_samples)}{aviso}"
            color = (0, 255, 0)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, texto, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Recoleccion de Postura", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cancelado por el usuario.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if not raw_samples:
        print(
            "No se guardo ninguna muestra frontal. Ubicate de frente a la camara, "
            "con los dos hombros visibles, y vuelve a intentar."
        )
        if descartadas_perfil:
            print(f"(Se descartaron {descartadas_perfil} muestras por estar de perfil.)")
        return

    label_value = 1 if args.label == 1 else 0
    X_all, y_all, groups_all = load_existing()
    X_all.extend(raw_samples)
    y_all.extend([label_value] * len(raw_samples))
    groups_all.extend([session_id] * len(raw_samples))
    save_dataset(X_all, y_all, groups_all)
    log_session(session_id, label_name, args.contexto, len(raw_samples) + descartadas_perfil, len(raw_samples))

    print(f"\nGuardadas {len(raw_samples)} muestras frontales (sesion {session_id}).")
    if descartadas_perfil:
        print(f"Descartadas {descartadas_perfil} por estar de perfil.")
    print(f"Dataset total ahora: {len(X_all)} muestras en {len(set(groups_all))} sesiones.")
    print("El espejo y el ruido los agrega train.py, despues de dividir.")
    print("Repite con las otras combinaciones (correcta/incorrecta x sentado/de_pie).")
    print("Cuando tengas al menos 4 sesiones, ejecuta: python train.py")


if __name__ == "__main__":
    main()
