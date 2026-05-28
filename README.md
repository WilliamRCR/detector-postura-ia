# Detector de Postura Corporal

Sistema de detección de postura corporal usando visión por computadora y Machine Learning.

## Requisitos Previos

### 1. Python 3.8 o superior
- **Windows**: Descarga de [python.org](https://www.python.org/downloads/) o desde la Microsoft Store
- **Mac**: `brew install python3` o descargarlo de python.org
- **Linux**: `sudo apt install python3` (Ubuntu/Debian) o `sudo yum install python3` (Fedora)

**IMPORTANTE**: Durante la instalación de Windows, marca la opción **"Add Python to PATH"**

### 2. Cámara web
- Necesitas una cámara integrada o externa conectada

---

## Instalación

### Paso 1: Descargar el proyecto

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd proyecto-ia
```

O si ya tienes el proyecto descargado, simplemente abre la carpeta.

### Paso 2: Crear un entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso del Proyecto

### 1. Recopilar datos de postura CORRECTA
```bash
python collect_data.py 1 10
```
- Presiona `s` para guardar cada muestra
- Presiona `q` para salir antes de tiempo
- El número 10 indica cuántas muestras quieres capturar (por defecto)

### 2. Recopilar datos de postura INCORRECTA
```bash
python collect_data.py 2 10
```
- Igual que el anterior pero para postura incorrecta
- Presiona `s` para cada muestra, `q` para salir

### 3. Entrenar el modelo
```bash
python train.py
```
- Generará los archivos `modelo_postura.pkl` y `scaler.pkl`

### 4. Usar el detector
```bash
python predict.py
```
- Muestra tu cámara y detecta tu postura en tiempo real
- Presiona `q` para salir

### 5. Versión alternativa (Reglas Heurísticas)
```bash
python main.py
```
- Detecta postura usando reglas geométricas (no necesita ML)
- Funciona sin entrenamiento previo

---
## Solución de Problemas Comunes

### ❌ "pip no se reconoce como comando interno o externo"
**Solución**: 
- Windows: Busca "Python" en el menú inicio → "Modify Python" → repair installation y marca "Add to PATH"
- O usa: `py -m pip install -r requirements.txt`

### ❌ "ModuleNotFoundError: No module named 'cv2'"
**Solución**: 
```bash
pip install opencv-python
```

### ❌ "ModuleNotFoundError: No module named 'mediapipe'"
**Solución**:
```bash
pip install mediapipe
```

### ❌ Error de cámara no encontrada
**Solución**:
- Verifica que otra aplicación no esté usando la cámara
- En `predict.py`, `collect_data.py`, busca la línea `cap = cv2.VideoCapture(0)` y cambia el `0` a `1`, `2`, etc. según tu cámara

### ❌ "ImportError: DLL load failed" (Windows)
**Solución**:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### ❌ Error con numpy, scikit-learn o incompatibilidad de versiones
**Síntoma**:  
Errores como "_ARRAY_API not found" o "numpy.core._multiarray_umath failed to import"

**Causa**:  
TensorFlow y MediaPipe necesitan NumPy 1.x (no 2.x)

**Solución**:
```bash
pip install numpy==1.26.4 scikit-learn==1.3.2 joblib
pip install tensorflow==2.15.0 mediapipe==0.10.9
```

Si ya tienes los datos recolectados pero el training falla por muy pocas muestras:
```bash
python collect_data.py 1 30    # Recolecta 30 muestras correctas
python collect_data.py 2 30    # Recolecta 30 muestras incorrectas  
python train.py                # Entrena el modelo
```

### ❌ La cámara se ve muy lenta o Lag
**Solución**: Cierra otras aplicaciones que usen la cámara o reduzca la resolución en el código cambiando:
```python
cap.set(3, 640)  # ancho
cap.set(4, 480)  # alto
```

---

## Estructura de Archivos

```
proyecto-ia/
├── main.py           # Punto de entrada
├── collect_data.py   # Recolectar datos de postura
├── train.py          # Entrenar el modelo
├── predict.py        # Usar el detector
├── requirements.txt  # Dependencias del proyecto
├── data_correcta.npz # Datos recopilados (se crea al ejecutar)
├── data_incorrecta.npz
├── modelo_postura.pkl# Modelo entrenado (se crea al ejecutar)
└── scaler.pkl        # Escalador (se crea al ejecutar)
```

---

## Notas

- Para mejores resultados, usa buena iluminación
- Mantén una distancia adecuada de la cámara (1-2 metros)
- El modelo funciona mejor cuando se entrena con datos de la misma cámara
- Si tus amigos tienen problemas, sugiere que actualicen sus drivers de cámara