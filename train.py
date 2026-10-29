"""Entrena la red neuronal sobre el dataset recolectado.

Dos cosas que antes inflaban la precision y ya no:

1. La division entrenamiento/prueba se hace por grupo (foto original o sesion
   de grabacion), no por muestra. Las 160 fotos "correctas" de Roboflow eran
   solo 57 distintas con recortes repetidos: al dividir al azar, copias de la
   misma foto caian en ambos lados y el modelo aprobaba examenes que ya habia
   visto.
2. El espejo y el ruido se agregan despues de dividir y solo al lado de
   entrenamiento. Antes se agregaban antes, asi que la prueba contenia el
   espejo de muestras de entrenamiento.

Ademas se reporta matriz de confusion y precision/recall por clase, no solo
accuracy, que con clases desbalanceadas dice poco.
"""
import argparse

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import core

AUGMENT_JITTER_STD = 0.01
RANDOM_STATE = 42


def load_dataset(path=core.DATASET_FILE):
    """Carga X crudo (99 valores), y y los grupos.

    Si el .npz es de una version anterior y no trae grupos, cada muestra queda
    en su propio grupo y se avisa: en ese caso la division vuelve a ser por
    muestra y la precision puede salir inflada.
    """
    try:
        data = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise SystemExit(
            "Error: Faltan archivos de datos. Ejecuta collect_data.py primero."
        ) from None

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"]).astype(int)

    if "groups" in data.files:
        groups = np.asarray(data["groups"]).astype(str)
        has_groups = True
    else:
        groups = np.array([f"muestra_{i}" for i in range(len(X))])
        has_groups = False

    if X.ndim != 2 or X.shape[1] != core.RAW_SIZE:
        raise SystemExit(
            f"Error: el dataset tiene muestras de {X.shape[-1] if X.ndim == 2 else '?'} "
            f"valores y se esperaban {core.RAW_SIZE} crudos. Vuelve a recolectar datos."
        )
    if not (len(X) == len(y) == len(groups)):
        raise SystemExit("Error: X, y y groups tienen tamanos distintos.")

    return X, y, groups, has_groups


def augment(X_raw, y, groups, rng):
    """Duplica el set de entrenamiento con espejo + ruido leve.

    Se aplica sobre las muestras crudas (antes de normalizar) y solo al lado de
    entrenamiento, para que la prueba siga siendo datos que el modelo no vio.
    """
    extra_X, extra_y, extra_g = [], [], []
    for kp, label, group in zip(X_raw, y, groups):
        extra_X.append(core.mirror_keypoints(kp))
        extra_y.append(label)
        extra_g.append(group)
        extra_X.append(core.jitter_keypoints(kp, std=AUGMENT_JITTER_STD, rng=rng))
        extra_y.append(label)
        extra_g.append(group)
    if not extra_X:
        return X_raw, y, groups
    return (
        np.vstack([X_raw, np.vstack(extra_X)]),
        np.concatenate([y, np.array(extra_y, dtype=int)]),
        np.concatenate([groups, np.array(extra_g, dtype=str)]),
    )


def split_by_group(X, y, groups, test_size=0.25):
    """Division por grupo: ninguna copia de la misma foto/sesion aparece en
    entrenamiento y prueba a la vez."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        print(
            f"ADVERTENCIA: solo hay {len(unique_groups)} grupos distintos. "
            "Divido por muestra, asi que la precision puede salir inflada.\n"
            "            Graba mas sesiones con collect_data.py."
        )
        stratify = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
        return train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def describe(name, y):
    total = len(y)
    correctas = int(np.sum(y == 1))
    print(f"{name}: {total} muestras  (correctas {correctas}, incorrectas {total - correctas})")


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo de postura.")
    parser.add_argument("--dataset", default=core.DATASET_FILE, help="archivo .npz de entrada")
    parser.add_argument("--test-size", type=float, default=0.25, help="proporcion de prueba")
    parser.add_argument("--sin-aumento", action="store_true", help="no agregar espejo ni ruido")
    args = parser.parse_args()

    print("=== ENTRENAMIENTO DE RED NEURONAL ===")

    X_raw, y, groups, has_groups = load_dataset(args.dataset)
    describe("Dataset", y)
    print(f"Grupos distintos (fotos/sesiones): {len(np.unique(groups))}")
    if not has_groups:
        print(
            "ADVERTENCIA: este dataset es de una version anterior y no guarda a que\n"
            "            foto o sesion pertenece cada muestra. Vuelve a recolectar\n"
            "            con collect_data.py para una division honesta."
        )

    if len(X_raw) < 10:
        print(f"\nADVERTENCIA: solo hay {len(X_raw)} muestras. Necesitas mas datos.")
        print("Ejecuta: python collect_data.py 1 30")
        print("Ejecuta: python collect_data.py 2 30")
    if len(np.unique(y)) < 2:
        raise SystemExit(
            "Error: el dataset tiene una sola clase. Recolecta posturas correctas E incorrectas."
        )

    # --- 1. Dividir primero (por grupo), aumentar despues ---
    train_idx, test_idx = split_by_group(X_raw, y, groups, test_size=args.test_size)

    X_train_raw, y_train, g_train = X_raw[train_idx], y[train_idx], groups[train_idx]
    X_test_raw, y_test = X_raw[test_idx], y[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(
            "ADVERTENCIA: la division dejo una clase sola en entrenamiento o prueba.\n"
            "            Graba mas sesiones variadas para que el reparto sea parejo."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    if not args.sin_aumento:
        before = len(X_train_raw)
        X_train_raw, y_train, g_train = augment(X_train_raw, y_train, g_train, rng)
        print(f"\nAumento solo en entrenamiento: {before} -> {len(X_train_raw)} muestras.")

    # --- 2. Caracteristicas normalizadas ---
    X_train, keep_train = core.features_matrix(X_train_raw)
    X_test, keep_test = core.features_matrix(X_test_raw)
    y_train = y_train[keep_train]
    y_test = y_test[keep_test]

    descartadas = (len(X_train_raw) - len(keep_train)) + (len(X_test_raw) - len(keep_test))
    if descartadas:
        print(
            f"Descartadas {descartadas} muestras no medibles (de perfil o con deteccion mala)."
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit(
            "Error: no quedaron muestras utiles tras normalizar. Revisa que las "
            "grabaciones sean de frente a la camara."
        )

    describe("Entrenamiento", y_train)
    describe("Prueba       ", y_test)
    print(f"Caracteristicas por muestra: {X_train.shape[1]} (antes eran {core.RAW_SIZE} crudas)")

    # --- 3. Entrenar ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nEntrenando red neuronal...")
    usar_early_stopping = len(X_train) >= 40 and np.min(np.bincount(y_train)) >= 10
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=usar_early_stopping,
        validation_fraction=0.1 if usar_early_stopping else 0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # --- 4. Reportar de verdad ---
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)

    print(f"\nAccuracy entrenamiento: {train_acc:.2%}")
    print(f"Accuracy prueba:        {test_acc:.2%}")

    print("\nMatriz de confusion (prueba):")
    etiquetas = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_test, y_pred, labels=etiquetas)
    nombres = {0: "incorrecta", 1: "correcta"}
    ancho = max(len(nombres.get(e, str(e))) for e in etiquetas) + 2
    print(" " * (ancho + 10) + "predicho")
    print(" " * (ancho + 2) + "  ".join(f"{nombres.get(e, e):>10}" for e in etiquetas))
    for fila, real in zip(cm, etiquetas):
        print(f"real {nombres.get(real, real):<{ancho}}" + "  ".join(f"{v:>10}" for v in fila))

    print("\nPrecision y recall por clase (prueba):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=etiquetas,
            target_names=[nombres.get(e, str(e)) for e in etiquetas],
            zero_division=0,
        )
    )

    if train_acc - test_acc > 0.20:
        print(
            "ADVERTENCIA: la precision de entrenamiento supera mucho a la de prueba.\n"
            "            El modelo se esta memorizando: graba mas variedad de sesiones."
        )

    core.save_model(
        model,
        scaler,
        meta={
            "test_accuracy": round(float(test_acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_grupos": int(len(np.unique(groups))),
            "division_por_grupo": bool(has_groups),
        },
    )
    print(f"\nModelo guardado: {core.MODEL_FILE} (+ {core.SCALER_FILE}, {core.META_FILE})")


if __name__ == "__main__":
    main()
