"""Entrena la red neuronal sobre el dataset recolectado.

El detector se usa en cuatro situaciones: sentado de frente, sentado de perfil,
de pie de frente y de pie de perfil. Este script reporta cuales cubre tu
dataset y cuales faltan, y mide la precision de cada vista por separado.

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

La division sirve para MEDIR. El modelo que se guarda se vuelve a entrenar con
todas las sesiones: dejar fuera las de prueba le quitaba a la app situaciones
enteras (con 8 grabaciones, dos de las ocho combinaciones nunca se aprendian).
"""
import argparse
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import core
from config import AppConfig
from posture_metrics import personal_thresholds

AUGMENT_JITTER_REL = 0.03  # ruido = 3% del tamano de la muestra (hombros o cabeza)
RANDOM_STATE = 42

CONTEXTS = ("sentado", "de_pie")
UNKNOWN_CONTEXT = core.UNKNOWN_CONTEXT
# Cuantas muestras por combinacion y clase se consideran un minimo razonable.
MIN_PER_CELL = 20


def load_dataset(path=core.DATASET_FILE):
    """Carga X crudo (99 valores), y, grupos, vistas y contextos.

    Si el .npz es de una version anterior y no trae grupos, cada muestra queda
    en su propio grupo y se avisa: en ese caso la division vuelve a ser por
    muestra y la precision puede salir inflada. Si no trae vistas se calculan; si
    no trae contextos se leen del nombre de la sesion (o quedan "desconocido").
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

    if "views" in data.files:
        views = np.asarray(data["views"]).astype(str)
    else:
        views = np.array([core.view_of(kp) for kp in X], dtype=str)
    if "contexts" in data.files:
        contexts = np.asarray(data["contexts"]).astype(str)
    else:
        contexts = np.array([core.context_from_group(g) for g in groups], dtype=str)
    if not (len(views) == len(contexts) == len(X)):
        raise SystemExit("Error: views y contexts no coinciden con X.")

    return X, y, groups, has_groups, views, contexts


def augment(X_raw, y, groups, rng):
    """Duplica el set de entrenamiento con espejo + ruido leve.

    Se aplica sobre las muestras crudas (antes de normalizar) y solo al lado de
    entrenamiento, para que la prueba siga siendo datos que el modelo no vio.
    De perfil no se espeja: las muestras de perfil ya se normalizan mirando hacia
    el mismo lado, asi que el espejo seria una copia identica. El ruido es
    proporcional al tamano de la muestra: uno fijo borraba la senal de cabeza
    adelantada de perfil, donde la cabeza mide un tercio que el ancho de hombros.
    """
    extra_X, extra_y, extra_g = [], [], []
    for kp, label, group in zip(X_raw, y, groups):
        if core.view_of(kp) != core.VIEW_SIDE:
            extra_X.append(core.mirror_keypoints(kp))
            extra_y.append(label)
            extra_g.append(group)
        extra_X.append(core.jitter_relative(kp, rel=AUGMENT_JITTER_REL, rng=rng))
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

    # Un GroupShuffleSplit a secas puede dejar en la prueba grupos de una sola
    # clase (con pocas sesiones, la prueba era UNA sesion) y la precision de
    # prueba no significaba nada. Se reparte por clase: cada una aporta al menos
    # un grupo completo a la prueba.
    rng = np.random.default_rng(RANDOM_STATE)
    groups_by_class = {}
    for group in unique_groups:
        label = int(np.bincount(y[groups == group]).argmax())
        groups_by_class.setdefault(label, []).append(group)

    if all(len(members) >= 2 for members in groups_by_class.values()):
        test_groups = []
        for members in groups_by_class.values():
            n_test = min(len(members) - 1, max(1, round(test_size * len(members))))
            test_groups.extend(rng.choice(members, size=n_test, replace=False))
        in_test = np.isin(groups, test_groups)
        return np.flatnonzero(~in_test), np.flatnonzero(in_test)

    print(
        "ADVERTENCIA: alguna clase tiene una sola sesion/foto. Divido sin garantizar\n"
        "            que la prueba tenga las dos clases; graba mas sesiones."
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    return next(splitter.split(X, y, groups=groups))


def print_coverage(y, views, contexts, groups):
    """Cuantas muestras hay en cada una de las 4 situaciones (vista x contexto)
    y de que combinaciones faltan datos. Devuelve la lista de faltantes."""
    print("\nCobertura del dataset (muestras y sesiones de cada clase):")
    print(f"  {'':22s}{'correctas':>10s}{'incorrectas':>13s}{'ses. correctas':>16s}{'ses. incorrectas':>18s}")
    faltan = []
    for view in core.VIEWS:
        for context in CONTEXTS:
            mask = (views == view) & (contexts == context)
            good_mask = mask & (y == 1)
            bad_mask = mask & (y == 0)
            good = int(good_mask.sum())
            bad = int(bad_mask.sum())
            good_sessions = len(set(groups[good_mask]))
            bad_sessions = len(set(groups[bad_mask]))
            print(f"  {view + ' / ' + context:22s}{good:>10d}{bad:>13d}{good_sessions:>16d}{bad_sessions:>18d}")
            if good == 0 or bad == 0:
                faltan.append(f"{context} de {view}: sin muestras " + ("correctas" if good == 0 else "incorrectas"))
            elif min(good, bad) < MIN_PER_CELL or min(good_sessions, bad_sessions) < 2:
                # Por clase: una correcta + una incorrecta son 2 sesiones, pero
                # si la correcta queda en la prueba no hay con que medirla.
                faltan.append(f"{context} de {view}: pocas muestras o una sola sesion por clase")

    otras = {}
    for view in core.VIEWS:
        mask = (views == view) & ~np.isin(contexts, CONTEXTS)
        if mask.any():
            otras[view] = (int(np.sum(y[mask] == 1)), int(np.sum(y[mask] == 0)), len(set(groups[mask])))
    for view, (good, bad, sessions) in otras.items():
        print(f"  {view + ' / sin contexto':22s}{good:>10d}{bad:>13d}{sessions:>10d}   (fotos antiguas o mixto)")

    if faltan:
        print("\nADVERTENCIA: faltan datos en estas situaciones y el detector sera poco fiable ahi:")
        for texto in faltan:
            print(f"  - {texto}")
        print("  Graba cada una con 1-GRABAR-DATOS.bat (o collect_data.py --vista ... --contexto ...).")
    return faltan


def describe(name, y):
    total = len(y)
    correctas = int(np.sum(y == 1))
    print(f"{name}: {total} muestras  (correctas {correctas}, incorrectas {total - correctas})")


def save_personal_thresholds(X_raw, y, contexts):
    """Calcula los umbrales de las reglas de cada vista con TUS muestras
    "correcta" y los guarda en config.json. Solo usa grabaciones con contexto
    (las de collect_data.py): las fotos antiguas vienen de otras camaras y
    calibrarian para ellas, no para la tuya. Devuelve True si cambio algo."""
    propias = contexts != UNKNOWN_CONTEXT
    X_own, y_own = X_raw[propias], y[propias]
    correctas = [core.metrics_from_keypoints(kp) for kp in X_own[y_own == 1]]
    incorrectas = [core.metrics_from_keypoints(kp) for kp in X_own[y_own == 0]]

    cfg = AppConfig.load()
    thresholds = cfg.thresholds()
    cambio = False
    print("\nUmbrales de las reglas (con tus grabaciones):")
    for view in core.VIEWS:
        nuevos, nota = personal_thresholds(correctas, base=thresholds, view=view)
        print(f"  [{view}] {nota}")
        if nuevos is None:
            continue
        thresholds = nuevos
        cambio = True
        if view == core.VIEW_SIDE:
            print(
                f"      espalda <= {nuevos.side_trunk_lean_max:.1f} grados   cabeza adelantada <= "
                f"{nuevos.side_head_forward_max:.2f}   altura de cabeza >= {nuevos.side_neck_height_min:.2f}"
            )
        else:
            print(
                f"      hombros <= {nuevos.shoulder_tilt_max:.2f}   cabeza adelantada <= "
                f"{nuevos.forward_head_max:.2f}   altura de cabeza >= {nuevos.neck_compression_min:.2f}"
            )

    if not cambio:
        return False
    cfg.set_thresholds(thresholds, anchor=True)
    print("  (guardados en config.json)")

    def marca_mal(m):
        if m["view"] == core.VIEW_SIDE:
            return (
                m["trunk_lean"] > thresholds.side_trunk_lean_max
                or m["forward_head"] > thresholds.side_head_forward_max
                or m["neck_gap"] < thresholds.side_neck_height_min
            )
        return (
            m["shoulder_tilt"] > thresholds.shoulder_tilt_max
            or m["forward_head"] > thresholds.forward_head_max
            or m["neck_gap"] < thresholds.neck_compression_min
        )

    for view in core.VIEWS:
        for nombre, lote in (("correctas", correctas), ("incorrectas", incorrectas)):
            medibles = [m for m in lote if m and m["view"] == view]
            if medibles:
                share = sum(marca_mal(m) for m in medibles) / len(medibles)
                print(f"  [{view}] las reglas marcan como mala postura el {share:.0%} de tus {nombre}")
    return True


def fit_model(X_raw, y, groups, rng, aumentar=True):
    """Aumenta, normaliza y entrena. Devuelve (modelo, scaler, X_escalado, y)
    con lo que realmente se uso, o None si no quedo ninguna muestra medible."""
    if aumentar:
        X_raw, y, groups = augment(X_raw, y, groups, rng)
    X, keep = core.features_matrix(X_raw)
    y = y[keep]
    if len(X) == 0:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    usar_early_stopping = len(X) >= 40 and np.min(np.bincount(y)) >= 10
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=usar_early_stopping,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled, y)
    return model, scaler, X_scaled, y


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo de postura.")
    parser.add_argument("--dataset", default=core.DATASET_FILE, help="archivo .npz de entrada")
    parser.add_argument("--test-size", type=float, default=0.25, help="proporcion de prueba")
    parser.add_argument("--sin-aumento", action="store_true", help="no agregar espejo ni ruido")
    parser.add_argument(
        "--solo-vista",
        choices=list(core.VIEWS),
        help="entrenar solo con una vista (por defecto se usan las dos)",
    )
    parser.add_argument(
        "--solo-grabaciones",
        action="store_true",
        help="ignorar las fotos antiguas y usar solo lo que grabaste con collect_data.py",
    )
    parser.add_argument(
        "--sin-umbrales",
        action="store_true",
        help="no recalcular los umbrales de las reglas en config.json",
    )
    args = parser.parse_args()

    print("=== ENTRENAMIENTO DE RED NEURONAL ===")

    X_raw, y, groups, has_groups, views, contexts = load_dataset(args.dataset)
    describe("Dataset", y)

    if args.solo_grabaciones:
        mask = contexts != UNKNOWN_CONTEXT
        print(f"--solo-grabaciones: ignoro {int((~mask).sum())} muestras sin contexto (fotos antiguas).")
        X_raw, y, groups, views, contexts = X_raw[mask], y[mask], groups[mask], views[mask], contexts[mask]
    if args.solo_vista:
        mask = views == args.solo_vista
        print(f"--solo-vista {args.solo_vista}: ignoro {int((~mask).sum())} muestras de la otra vista.")
        X_raw, y, groups, views, contexts = X_raw[mask], y[mask], groups[mask], views[mask], contexts[mask]

    intermedias = int(np.sum(~np.isin(views, core.VIEWS)))
    if intermedias:
        print(
            f"Descartadas {intermedias} muestras ni de frente ni de perfil (3/4, lejos o "
            "deteccion mala): la app tampoco las mide."
        )
        mask = np.isin(views, core.VIEWS)
        X_raw, y, groups, views, contexts = X_raw[mask], y[mask], groups[mask], views[mask], contexts[mask]
    if len(X_raw) == 0:
        raise SystemExit(
            "Error: no queda ninguna muestra utilizable. Graba sesiones con "
            "collect_data.py (1-GRABAR-DATOS.bat)."
        )

    describe("Utilizables", y)
    print(f"Grupos distintos (fotos/sesiones): {len(np.unique(groups))}")
    print_coverage(y, views, contexts, groups)
    if not has_groups:
        print(
            "ADVERTENCIA: este dataset es de una version anterior y no guarda a que\n"
            "            foto o sesion pertenece cada muestra. Vuelve a recolectar\n"
            "            con collect_data.py para una division honesta."
        )

    if len(X_raw) < 10:
        print(f"\nADVERTENCIA: solo hay {len(X_raw)} muestras. Necesitas mas datos.")
    if len(np.unique(y)) < 2:
        raise SystemExit(
            "Error: el dataset tiene una sola clase. Recolecta posturas correctas E incorrectas."
        )

    # --- 1. Dividir primero (por grupo), aumentar despues ---
    train_idx, test_idx = split_by_group(X_raw, y, groups, test_size=args.test_size)

    X_train_raw, y_train, g_train = X_raw[train_idx], y[train_idx], groups[train_idx]
    X_test_raw, y_test = X_raw[test_idx], y[test_idx]
    v_test = views[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(
            "ADVERTENCIA: la division dejo una clase sola en entrenamiento o prueba.\n"
            "            Graba mas sesiones variadas para que el reparto sea parejo."
        )

    # --- 2. Caracteristicas normalizadas y red de evaluacion ---
    rng = np.random.default_rng(RANDOM_STATE)
    X_test, keep_test = core.features_matrix(X_test_raw)
    y_test = y_test[keep_test]
    v_test = v_test[keep_test]

    print("\nEntrenando red neuronal para medirla...")
    ajuste = fit_model(X_train_raw, y_train, g_train, rng, aumentar=not args.sin_aumento)
    if ajuste is None or len(X_test) == 0:
        raise SystemExit(
            "Error: no quedaron muestras utiles tras normalizar. Revisa que las "
            "grabaciones sean de frente o de perfil a la camara."
        )
    model, scaler, X_train_scaled, y_train = ajuste
    X_test_scaled = scaler.transform(X_test)

    describe("Entrenamiento", y_train)
    describe("Prueba       ", y_test)
    print(f"Caracteristicas por muestra: {X_train_scaled.shape[1]} (antes eran {core.RAW_SIZE} crudas)")

    # --- 3. Reportar de verdad ---
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)

    print(f"\nAccuracy entrenamiento: {train_acc:.2%}")
    print(f"Accuracy prueba:        {test_acc:.2%}")

    print("\nAccuracy de prueba por vista:")
    for view in core.VIEWS:
        mask = v_test == view
        if mask.any():
            print(f"  {view:7s}: {np.mean(y_pred[mask] == y_test[mask]):.2%}  ({int(mask.sum())} muestras)")
        else:
            print(f"  {view:7s}: sin muestras en la prueba")

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

    # --- 4. Modelo final con TODAS las sesiones ---
    print("\nEntrenando el modelo final con todas las sesiones (entrenamiento + prueba)...")
    final_model, final_scaler, X_final, y_final = fit_model(
        X_raw, y, groups, rng, aumentar=not args.sin_aumento
    )
    describe("Modelo final ", y_final)
    print("La precision de arriba mide sesiones que la red no vio; el modelo final si las aprende.")

    core.save_model(
        final_model,
        final_scaler,
        meta={
            "test_accuracy": round(float(test_acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "n_train": int(len(X_train_scaled)),
            "n_test": int(len(X_test)),
            "n_final": int(len(X_final)),
            "n_grupos": int(len(np.unique(groups))),
            "division_por_grupo": bool(has_groups),
            "muestras_por_vista": {k: int(v) for k, v in Counter(views.tolist()).items()},
        },
    )
    print(f"\nModelo guardado: {core.MODEL_FILE} (+ {core.SCALER_FILE}, {core.META_FILE})")

    if not args.sin_umbrales:
        save_personal_thresholds(X_raw, y, contexts)


if __name__ == "__main__":
    main()
