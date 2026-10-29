"""Recolecta posturas desde las fotos de dataset/ (camino NO recomendado).

Las fotos de dataset/ son casi todas de perfil, pero la webcam y la app ven al
usuario de frente: entrenar con ellas hace que el modelo aprenda una vista y se
use en otra. Por eso ahora este script:

- descarta por defecto las fotos que no son frontales (--incluir-perfil las
  vuelve a incluir, bajo tu responsabilidad);
- agrupa por foto original, no por archivo: las 160 fotos "correctas" de
  Roboflow son solo 57 distintas con recortes repetidos, y sin agrupar esas
  copias caen a la vez en entrenamiento y prueba;
- escribe en un .npz aparte, para no pisar el dataset grabado con
  collect_data.py.

Uso:
    python collect_from_images.py                       # solo frontales
    python collect_from_images.py --incluir-perfil      # todo (no recomendado)
    python collect_from_images.py --salida mezcla.npz
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

import core

DEFAULT_OUTPUT = "dataset_imagenes.npz"
FOLDERS = (("dataset/correcta", 1), ("dataset/incorrecta", 0))

# Extensiones que OpenCV lee. Las .avif quedan fuera (por eso una foto del
# dataset nunca se leyo).
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Las copias de Roboflow se llaman <original>_png.rf.<hash>.jpg: el grupo es el
# nombre de la foto original, asi todas sus variantes van juntas al dividir.
ROBOFLOW_RE = re.compile(r"^(.*?)(?:_png)?\.rf\.[0-9a-f]+$", re.IGNORECASE)


def group_name(filename):
    """Nombre del grupo (foto original) para un archivo del dataset."""
    stem = Path(filename).stem
    match = ROBOFLOW_RE.match(stem)
    if match:
        return match.group(1)
    return stem


def process_folder(folder_path, label, pose, incluir_perfil=False, verbose=True):
    X, y, groups = [], [], []
    saltadas = {"ilegible": 0, "sin_postura": 0, "perfil": 0, "no_imagen": 0}

    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"No existe la carpeta {folder_path}, la salto.")
        return X, y, groups, saltadas

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_SUFFIXES:
            saltadas["no_imagen"] += 1
            if verbose:
                print(f"[FORMATO] OpenCV no lee {path.suffix}: {path.name}")
            continue

        image = cv2.imread(str(path))
        if image is None:
            saltadas["ilegible"] += 1
            if verbose:
                print(f"[ILEGIBLE] {path.name}")
            continue

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            saltadas["sin_postura"] += 1
            if verbose:
                print(f"[SIN POSTURA] {path.name}")
            continue

        kp = core.keypoints_from_landmarks(results.pose_landmarks.landmark)
        if kp.size != core.RAW_SIZE:
            saltadas["sin_postura"] += 1
            if verbose:
                print(f"[KEYPOINTS INVALIDOS] {path.name}")
            continue

        if not incluir_perfil and not core.is_frontal(kp):
            saltadas["perfil"] += 1
            if verbose:
                ratio = core.frontality(kp)
                detalle = f"{ratio:.2f}" if ratio is not None else "n/d"
                print(f"[DE PERFIL] {path.name}  (hombros/torso = {detalle})")
            continue

        X.append(kp)
        y.append(label)
        groups.append(f"{folder.name}:{group_name(path.name)}")
        if verbose:
            print(f"[OK] {path.name}")

    return X, y, groups, saltadas


def main():
    parser = argparse.ArgumentParser(description="Recolecta posturas desde las fotos de dataset/.")
    parser.add_argument(
        "--incluir-perfil",
        action="store_true",
        help="incluir fotos de perfil (NO recomendado: la app ve al usuario de frente)",
    )
    parser.add_argument("--salida", default=DEFAULT_OUTPUT, help=f"archivo .npz de salida (default {DEFAULT_OUTPUT})")
    parser.add_argument("--silencioso", action="store_true", help="no listar foto por foto")
    args = parser.parse_args()

    print("=== RECOLECCION DESDE IMAGENES ===")
    print(
        "Recordatorio: el camino recomendado es collect_data.py (webcam, de frente).\n"
        "Este script existe para inspeccionar las fotos, no para entrenar el modelo final.\n"
    )
    if args.incluir_perfil:
        print("--incluir-perfil activo: se incluiran fotos de perfil, que la app nunca vera.\n")

    pose = core.new_pose(static=True)

    X_all, y_all, groups_all = [], [], []
    for folder, label in FOLDERS:
        print(f"--- {folder} (etiqueta {label}) ---")
        X, y, groups, saltadas = process_folder(
            folder, label, pose, incluir_perfil=args.incluir_perfil, verbose=not args.silencioso
        )
        X_all.extend(X)
        y_all.extend(y)
        groups_all.extend(groups)
        print(
            f"  usadas {len(X)}  |  de perfil {saltadas['perfil']}  |  "
            f"sin postura {saltadas['sin_postura']}  |  ilegibles {saltadas['ilegible']}  |  "
            f"formato no soportado {saltadas['no_imagen']}"
        )
        print(f"  fotos originales distintas: {len(set(groups))}\n")

    if not X_all:
        print("No quedo ninguna muestra utilizable. Usa collect_data.py.")
        return

    np.savez(
        args.salida,
        X=np.asarray(X_all, dtype=np.float32),
        y=np.asarray(y_all, dtype=np.int64),
        groups=np.asarray(groups_all, dtype="U96"),
    )

    print("Dataset final:")
    print(f"  muestras: {len(X_all)}")
    print(f"  grupos (fotos originales): {len(set(groups_all))}")
    print(f"  correctas: {sum(1 for v in y_all if v == 1)}, incorrectas: {sum(1 for v in y_all if v == 0)}")
    print(f"\nGuardado en {args.salida}")
    print(
        f"Para entrenar con esto (no recomendado): python train.py --dataset {args.salida}\n"
        "El modelo que usa la app se entrena con el dataset de collect_data.py."
    )


if __name__ == "__main__":
    main()
