import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from datos import cargar_datos

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# cargamos los datos ya preparados
games = cargar_datos()

# creamos carpetas para guardar gráficos y modelos
os.makedirs("graficos_dl", exist_ok=True)
os.makedirs("modelos_dl", exist_ok=True)

print("\n" + "=" * 60)
print("INICIANDO MODELO DE DEEP LEARNING (TENSORFLOW / KERAS)")
print("=" * 60)

# ---------------------------------------------------------
# 1. PREPARACIÓN DE LOS DATOS
# ---------------------------------------------------------

dl_data = games.copy()

# Variable rated numérica
dl_data["rated_num"] = dl_data["rated"].map({True: 1, False: 0})

# Variables predictoras para la red neuronal
features = ["opening_eco", "white_rating", "black_rating", "rating_diff", "rated_num"]
target = "winner_num"

# Eliminamos posibles nulos por seguridad
dl_data = dl_data.dropna(subset=features + [target]).copy()

X = dl_data[features]
y = dl_data[target]

print("\n1. Datos preparados correctamente.")
print("Shape de X:", X.shape)
print("Shape de y:", y.shape)
print("\nDistribución de la variable objetivo:")
print(y.value_counts(normalize=True))

# ---------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 3. PREPROCESAMIENTO PARA KERAS
# ---------------------------------------------------------

categorical_features = ["opening_eco"]
numeric_features = ["white_rating", "black_rating", "rating_diff", "rated_num"]

preprocessor_dl = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

X_train_nn = preprocessor_dl.fit_transform(X_train)
X_test_nn = preprocessor_dl.transform(X_test)

# Las redes neuronales necesitan arrays densos
if hasattr(X_train_nn, "toarray"):
    X_train_nn = X_train_nn.toarray()

if hasattr(X_test_nn, "toarray"):
    X_test_nn = X_test_nn.toarray()

y_train_nn = np.array(y_train)
y_test_nn = np.array(y_test)

input_dim = X_train_nn.shape[1]

print("\n2. Datos transformados para Keras.")
print("Shape de X_train_nn:", X_train_nn.shape)
print("Shape de X_test_nn :", X_test_nn.shape)

# ---------------------------------------------------------
# 4. DEFINICIÓN DE LA RED NEURONAL
# ---------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

print("\n3. Construyendo la arquitectura de la red...")

dl_model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation="relu"),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

dl_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("\nResumen del modelo:")
dl_model.summary()

# ---------------------------------------------------------
# 5. ENTRENAMIENTO
# ---------------------------------------------------------

print("\n4. Entrenando la red neuronal...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = dl_model.fit(
    X_train_nn,
    y_train_nn,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------------
# 6. EVALUACIÓN FINAL
# ---------------------------------------------------------

print("\n5. Evaluando el modelo final...")

# Probabilidades de que ganen blancas
y_prob_dl = dl_model.predict(X_test_nn, verbose=0).flatten()

# Predicción binaria con umbral 0.5
y_pred_dl = (y_prob_dl >= 0.5).astype(int)

# Métricas
acc_dl = accuracy_score(y_test_nn, y_pred_dl)
prec_dl = precision_score(y_test_nn, y_pred_dl, zero_division=0)
rec_dl = recall_score(y_test_nn, y_pred_dl, zero_division=0)
f1_dl = f1_score(y_test_nn, y_pred_dl, zero_division=0)
roc_auc_dl = roc_auc_score(y_test_nn, y_prob_dl)

print("\n" + "=" * 40)
print("RESULTADOS - RED NEURONAL")
print("=" * 40)
print(f"Accuracy : {acc_dl:.4f}")
print(f"Precision: {prec_dl:.4f}")
print(f"Recall   : {rec_dl:.4f}")
print(f"F1-score : {f1_dl:.4f}")
print(f"ROC-AUC  : {roc_auc_dl:.4f}")

# ---------------------------------------------------------
# 7. MATRIZ DE CONFUSIÓN
# ---------------------------------------------------------

cm_dl = confusion_matrix(y_test_nn, y_pred_dl)
disp_dl = ConfusionMatrixDisplay(
    confusion_matrix=cm_dl,
    display_labels=["black", "white"]
)

disp_dl.plot(cmap="Blues")
plt.title("Matriz de confusión - Red Neuronal")
plt.tight_layout()
plt.savefig("graficos_dl/matriz_confusion_red_neuronal.png")
plt.close()

# ---------------------------------------------------------
# 8. CURVA ROC
# ---------------------------------------------------------

fpr_dl, tpr_dl, _ = roc_curve(y_test_nn, y_prob_dl)

plt.figure(figsize=(6, 4))
plt.plot(fpr_dl, tpr_dl, label=f"Red Neuronal (AUC = {roc_auc_dl:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Red Neuronal")
plt.legend()
plt.tight_layout()
plt.savefig("graficos_dl/roc_red_neuronal.png")
plt.close()

# ---------------------------------------------------------
# 9. GRÁFICOS DE ENTRENAMIENTO
# ---------------------------------------------------------

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Evolución del error (loss)")
plt.xlabel("Épocas")
plt.ylabel("Binary Crossentropy")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Evolución de la accuracy")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("graficos_dl/evolucion_entrenamiento.png")
plt.close()

# ---------------------------------------------------------
# 10. GUARDADO DEL MODELO Y DEL PREPROCESADOR
# ---------------------------------------------------------

dl_model.save("modelos_dl/chess_dl_model.keras")
joblib.dump(preprocessor_dl, "modelos_dl/chess_dl_preprocessor.pkl")

print("\n6. Archivos guardados correctamente:")
print("- Modelo: modelos_dl/chess_dl_model.keras")
print("- Preprocesador: modelos_dl/chess_dl_preprocessor.pkl")

# ---------------------------------------------------------
# 11. FUNCIÓN DE PREDICCIÓN PARA LA APP
# ---------------------------------------------------------

def predict_white_win_probs_dl(
    opening_eco,
    white_rating,
    rated=True,
    black_ratings=(1000, 1500, 2000),
    model=dl_model,
    preprocessor=preprocessor_dl
):
    """
    Devuelve la probabilidad de victoria de las blancas
    frente a distintos ratings posibles de negras.
    """

    input_df = pd.DataFrame({
        "opening_eco": [opening_eco] * len(black_ratings),
        "white_rating": [white_rating] * len(black_ratings),
        "black_rating": list(black_ratings),
        "rating_diff": [white_rating - br for br in black_ratings],
        "rated_num": [1 if rated else 0] * len(black_ratings)
    })

    X_input = preprocessor.transform(input_df)

    if hasattr(X_input, "toarray"):
        X_input = X_input.toarray()

    probs = model.predict(X_input, verbose=0).flatten()

    output = pd.DataFrame({
        "black_rating": list(black_ratings),
        "prob_white_win_red_neuronal": probs
    })

    return output

# ---------------------------------------------------------
# 12. PRUEBAS DE FUNCIONAMIENTO
# ---------------------------------------------------------

print("\n7. Pruebas finales de la función predictora...")

prueba_1 = predict_white_win_probs_dl(
    opening_eco="B12",
    white_rating=2000,
    rated=True,
    black_ratings=(1200, 1600, 2000)
)

print("\nPrueba 1:")
print(prueba_1)

prueba_2 = predict_white_win_probs_dl(
    opening_eco="A00",
    white_rating=1400,
    rated=False,
    black_ratings=(1000, 1400, 1800)
)

print("\nPrueba 2:")
print(prueba_2)