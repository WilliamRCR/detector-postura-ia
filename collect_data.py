"""Recolecta un dataset de posturas directamente desde la webcam, sin fotos manuales.

Esta es la forma recomendada de armar el dataset: graba tus landmarks de
MediaPipe con la misma camara y encuadre con que se usa la app. El detector
trabaja en cuatro situaciones y hay que grabar las cuatro:

    sentado de frente   |   sentado de perfil
    de pie de frente    |   de pie de perfil

cada una con postura correcta e incorrecta (8 grabaciones). `--vista` dice en
que vista estas; la muestra solo se guarda si de verdad lo estas (si dices
perfil y estas de frente, se descarta y te lo avisa en pantalla).

- Guarda las muestras crudas, sin espejo ni ruido. El aumento lo hace train.py
  despues de dividir, para que la precision de prueba sea real.
- Guarda a que sesion pertenece cada muestra (grupo), la vista y el contexto,
  para que train.py divida por sesion y avise de que combinaciones faltan.

Uso:
    python collect_data.py 1 20 --vista frente --contexto sentado   # 1 = correcta
    python collect_data.py 2 20 --vista perfil --contexto de_pie    # 2 = incorrecta

Cuantas mas variantes cubras (distancia, luz, ropa, dias distintos), mejor
generaliza el modelo.
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
    """Devuelve (X, y, groups, views, contexts) del dataset existente, o listas
    vacias."""
    path = Path(core.DATASET_FILE)
    if not path.exists():
        return [], [], [], [], []
    data = np.load(path, allow_pickle=False)
    X = list(data["X"])
    y = list(data["y"])
    if "groups" in data.files:
        groups = list(np.asarray(data["groups"]).astype(str))
    else:
        # Dataset viejo sin grupos: cada muestra queda en su propio grupo para
        # no fingir que vienen de la misma sesion.
        groups = [f"legado_{i}" for i in range(len(X))]
    if "views" in data.files:
        views = list(np.asarray(data["views"]).astype(str))
    else:
        views = [core.view_of(kp) for kp in X]
    if "contexts" in data.files:
        contexts = list(np.asarray(data["contexts"]).astype(str))
    else:
        contexts = [core.context_from_group(g) for g in groups]
    return X, y, groups, views, contexts


def save_dataset(X, y, groups, views, contexts):
    np.savez(
        core.DATASET_FILE,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        groups=np.asarray(groups, dtype="U64"),
        views=np.asarray(views, dtype="U12"),
        contexts=np.asarray(contexts, dtype="U12"),
    )


def log_session(session_id, label_name, vista, contexto, capturadas, medibles):
    entry = {
        "sesion": session_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "etiqueta": label_name,
        "vista": vista,
        "contexto": contexto,
        "muestras_capturadas": capturadas,
        "muestras_validas": medibles,
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
        "--vista",
        choices=list(core.VIEWS),
        default=core.VIEW_FRONT,
        help="como estas frente a la camara: frente o perfil (default frente)",
    )
    parser.add_argument(
        "--contexto",
        choices=["sentado", "de_pie", "mixto"],
        default="mixto",
        help="si estas sentado o de pie (train.py avisa de las combinaciones que falten)",
    )
    parser.add_argument("--camara", type=int, default=None, help="indice de camara (default: config.json)")
    args = parser.parse_args()

    cfg = AppConfig.load()
    camera_index = args.camara if args.camara is not None else cfg.camera_index

    label_name = "CORRECTA" if args.label == 1 else "INCORRECTA"
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{label_name.lower()}_{args.vista}_{args.contexto}"

    print(f"=== RECOLECCION DESDE WEBCAM: {label_name} ({args.vista}, {args.contexto}) ===")
    if args.vista == core.VIEW_SIDE:
        print(f"Prepárate para mantener esa postura durante {args.duracion}s, DE LADO a la cámara (perfil completo).")
    else:
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
    descartadas_vista = 0
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
            texto = f"Preparate...  {COUNTDOWN - int(elapsed_total)}"  # sin tilde: OpenCV la dibuja como '?'
            color = (0, 165, 255)
        else:
            if start_time is None:
                start_time = now
            restante = args.duracion - (now - start_time)

            if restante <= 0:
                break

            if results.pose_landmarks and (now - last_sample_at) >= SAMPLE_INTERVAL:
                kp = core.keypoints_from_landmarks(results.pose_landmarks.landmark)
                # Solo se guardan las muestras de la vista pedida: asi una
                # grabacion "de perfil" no se contamina con frames de frente.
                if core.view_of(kp) == args.vista:
                    raw_samples.append(kp)
                else:
                    descartadas_vista += 1
                last_sample_at = now

            aviso = f"  |  fuera de {args.vista}: {descartadas_vista}" if descartadas_vista else ""
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
        if args.vista == core.VIEW_SIDE:
            print(
                "No se guardo ninguna muestra de perfil. Ubicate completamente de lado "
                "a la camara (la nariz y las orejas bien visibles) y vuelve a intentar."
            )
        else:
            print(
                "No se guardo ninguna muestra de frente. Ubicate de frente a la camara, "
                "con la cabeza y los dos hombros visibles, y vuelve a intentar."
            )
        if descartadas_vista:
            print(f"(Se descartaron {descartadas_vista} muestras que no estaban de {args.vista}.)")
        return

    label_value = 1 if args.label == 1 else 0
    X_all, y_all, groups_all, views_all, contexts_all = load_existing()
    X_all.extend(raw_samples)
    y_all.extend([label_value] * len(raw_samples))
    groups_all.extend([session_id] * len(raw_samples))
    views_all.extend([args.vista] * len(raw_samples))
    contexts_all.extend([args.contexto] * len(raw_samples))
    save_dataset(X_all, y_all, groups_all, views_all, contexts_all)
    log_session(
        session_id, label_name, args.vista, args.contexto,
        len(raw_samples) + descartadas_vista, len(raw_samples),
    )

    print(f"\nGuardadas {len(raw_samples)} muestras de {args.vista} (sesion {session_id}).")
    if descartadas_vista:
        print(f"Descartadas {descartadas_vista} que no estaban de {args.vista}.")
    print(f"Dataset total ahora: {len(X_all)} muestras en {len(set(groups_all))} sesiones.")
    print("El espejo y el ruido los agrega train.py, despues de dividir.")
    print("Repite con las otras combinaciones: correcta/incorrecta x sentado/de_pie x frente/perfil.")
    print("Cuando las tengas, ejecuta: python train.py (te dice cuales faltan)")


if __name__ == "__main__":
    main()
