import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

print("=== ENTRENAMIENTO DE RED NEURONAL ===")

try:
    data = np.load("dataset_posturas.npz")
except FileNotFoundError:
    print("Error: Faltan archivos de datos. Ejecuta collect_data.py primero.")
    exit()

X = data["X"]
y = data["y"]

print(f"Total samples: {len(X)}")
print(f"Correctas: {sum(y == 1)}, Incorrectas: {sum(y == 0)}")

if len(X) < 10:
    print(f"\nADVERTENCIA: Solo hay {len(X)} muestras. Necesitas más datos.")
    print("Recolecta al menos 20-30 muestras de cada postura.")
    print("Ejecuta: python collect_data.py 1 30")
    print("Ejecuta: python collect_data.py 2 30")
    print("\nUsando validación simple sin estratificación...")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) != len(y):
        print("ERROR: X e y tienen tamaños distintos")
        exit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nEntrenando red neuronal...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=len(X) >= 20,
    validation_fraction=0.1 if len(X) >= 20 else 0.0,
    random_state=42
)

model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"Accuracy entrenamiento: {train_acc:.2%}")
print(f"Accuracy prueba: {test_acc:.2%}")

joblib.dump(model, "modelo_postura.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModelo guardado: modelo_postura.pkl")