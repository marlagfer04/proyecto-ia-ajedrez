import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from datos import cargar_datos

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

# cargamos los datos ya preparados
games = cargar_datos()

# creamos carpeta para guardar gráficos y modelos
os.makedirs("graficos_ml_reglog", exist_ok=True)
os.makedirs("graficos_ml_randomforest", exist_ok=True)
os.makedirs("modelos_ml", exist_ok=True)



# =========================================================
# MODELIZACIÓN (ML)
# =========================================================

# ---------------------------------------------------------
# 1. PREPARACIÓN DE LOS DATOS PARA ML
# ---------------------------------------------------------

# Trabajamos sobre una copia para no tocar el resto
ml_data = games.copy()

# Variable rated a numérica
ml_data["rated_num"] = ml_data["rated"].map({True: 1, False: 0})

# Comprobamos si hay nulos en las variables que vamos a usar
# features: variables predictoras
# elegimos estas variables porque son relevantes para saber cómo puede desarrollarse 
# la partida y no usan el resultado final para predecir el desenlace de la partida
features = ["opening_eco", "white_level_num", "black_level_num", "rated_num"]
# target: variable objetivo
target = "winner_num"


# vemos que no hay nulos para asegurarnos de que el modelo funciona
print("Nulos en variables de modelización:")
print(ml_data[features + [target]].isna().sum())

# Eliminamos filas con posibles nulos por seguridad
ml_data = ml_data.dropna(subset=features + [target])

# X e y
X = ml_data[features]
y = ml_data[target]

print("\nShape de X:", X.shape)
print("Shape de y:", y.shape)
print("\nDistribución de la variable objetivo:")
print(y.value_counts(normalize=True))

# ---------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    # semilla, usamos este valor porque es el que usó el profesor
    random_state=42,
    # si no ponemos el stratify, podemos llegar a un modelo poco representativo
    stratify=y
)

# ---------------------------------------------------------
# 3. PREPROCESAMIENTO
# ---------------------------------------------------------
# opening_eco es categórica -> OneHotEncoder
# niveles y rated_num ya son numéricas -> passthrough

# Separamos categóricas de numéricas porque se transforman distinto
categorical_features = ["opening_eco"]
numeric_features = ["white_level_num", "black_level_num", "rated_num"]

#IDEA DE LA TRANSFORMACIÓN: hace vector de 0's y 1's para los opening_eco, tantas entradas como aperturas registradas y sólo hay 1 en la que corresponda a esa partida, luego se añaden las demás variables al vector que SÍ son NUMÉRICAS.

#Ejemplo:
# Fila 1 → ["A00", 1, 0, 1]
# Fila 2 → ["B12", 2, 1, 0]

# Fila 1 → [1,0,0, 1,0,1]
# Fila 2 → [0,1,0, 2,1,0]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# ---------------------------------------------------------
# 4. DEFINICIÓN DE MODELOS
# ---------------------------------------------------------

# PROBAMOS 2 MODELOS
# usamos el pipeline y le damos la tranformacion y el modelo

log_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42
    ))
])

models = {
    "Logistic_Regression": log_reg,
    "Random_Forest": rf
}

# ---------------------------------------------------------
# 5. ENTRENAMIENTO Y EVALUACIÓN
# ---------------------------------------------------------

results = {}

for name, model in models.items():

    if name == "Logistic_Regression":
        carpeta_graficos = "graficos_ml_reglog"
    else:
        carpeta_graficos = "graficos_ml_randomforest"
    print(f"\n==============================")
    print(f"Entrenando modelo: {name}")
    print(f"==============================")

    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] #nos quedamos con la probabilidad de la clase 1, que son blancas (las que queremos).

    # Métricas

    #ACC: proporción total de aciertos
    acc = accuracy_score(y_test, y_pred)

    #Precisión: de las partidas que el modelo dijo que ganaban blancas, cuántas realmente ganaron blancas
    prec = precision_score(y_test, y_pred, zero_division=0)

    #Recall: de todas las partidas que realmente ganaron blancas, cuántas detectó el modelo.
    rec = recall_score(y_test, y_pred, zero_division=0)

    #F1: media armónica entre prec y rec, cuando no queremos fijarnos solamente en una de las dos.
    f1 = f1_score(y_test, y_pred, zero_division=0)

    #Roc_auc: miden lo bien que el modelo separa las clases (winner: blancas o winner: negras), es mejor cuanto más cercano sea a 1.
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # Matriz de confusión: para ver en qué se equivoca el modelo, no sólo cuanto acierta
    # Para ver:
    #fp, vp, vn, fn
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["black", "white"])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de confusión - {name}")
    plt.tight_layout()
    plt.savefig(f"{carpeta_graficos}/matriz_confusion_{name}.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{carpeta_graficos}/roc_{name}.png")
    plt.close()

# ---------------------------------------------------------
# 6. TABLA RESUMEN DE RESULTADOS
# ---------------------------------------------------------

results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)

print("\nResumen de métricas:")
print(results_df)

# ---------------------------------------------------------
# 7. GUARDAR AMBOS MODELOS
# ---------------------------------------------------------

results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)

print("\nResumen de métricas:")
print(results_df)

# Guardamos métricas
results_df.to_csv("modelos_ml/model_metrics.csv")

# Guardamos ambos modelos entrenados
joblib.dump(models["Logistic_Regression"], "modelos_ml/logistic_regression_model.pkl")
joblib.dump(models["Random_Forest"], "modelos_ml/random_forest_model.pkl")

print("\nModelos guardados correctamente:")
print("Regresión logística: modelos_ml/logistic_regression_model.pkl")
print("Random Forest: modelos_ml/random_forest_model.pkl")
print("Métricas: modelos_ml/model_metrics.csv")

# ---------------------------------------------------------
# 8. FUNCIONES DE PREDICCIÓN PARA LA APP
# ---------------------------------------------------------

modelo_reglog = models["Logistic_Regression"]
modelo_randomforest = models["Random_Forest"]


def predict_white_win_probs_RegLog(opening_eco, white_level, rated=True, model=modelo_reglog):
    level_map = {
        "principiante": 0,
        "avanzado": 1,
        "experto": 2
    }

    black_levels = ["principiante", "avanzado", "experto"]

    input_df = pd.DataFrame({
        "opening_eco": [opening_eco] * 3,
        "white_level_num": [level_map[white_level]] * 3,
        "black_level_num": [level_map[level] for level in black_levels],
        "rated_num": [1 if rated else 0] * 3
    })

    probs = model.predict_proba(input_df)[:, 1]

    return pd.DataFrame({
        "black_level": black_levels,
        "prob_white_win": probs
    })


def predict_white_win_probs_RandomForest(opening_eco, white_level, rated=True, model=modelo_randomforest):
    level_map = {
        "principiante": 0,
        "avanzado": 1,
        "experto": 2
    }

    black_levels = ["principiante", "avanzado", "experto"]

    input_df = pd.DataFrame({
        "opening_eco": [opening_eco] * 3,
        "white_level_num": [level_map[white_level]] * 3,
        "black_level_num": [level_map[level] for level in black_levels],
        "rated_num": [1 if rated else 0] * 3
    })

    probs = model.predict_proba(input_df)[:, 1]

    return pd.DataFrame({
        "black_level": black_levels,
        "prob_white_win": probs
    })

# ---------------------------------------------------------
# 9. EJEMPLOS DE USO
# ---------------------------------------------------------

print("\n==============================")
print("EJEMPLOS REGRESIÓN LOGÍSTICA")
print("==============================")

ejemplo_reglog_1 = predict_white_win_probs_RegLog(
    opening_eco="A00",
    white_level="avanzado",
    rated=True
)
print("\nRegLog - A00 avanzado rated:")
print(ejemplo_reglog_1)

ejemplo_reglog_2 = predict_white_win_probs_RegLog(
    opening_eco="B12",
    white_level="experto",
    rated=True
)
print("\nRegLog - B12 experto rated:")
print(ejemplo_reglog_2)


print("\n==============================")
print("EJEMPLOS RANDOM FOREST")
print("==============================")

ejemplo_rf_1 = predict_white_win_probs_RandomForest(
    opening_eco="A00",
    white_level="avanzado",
    rated=True
)
print("\nRandom Forest - A00 avanzado rated:")
print(ejemplo_rf_1)

ejemplo_rf_2 = predict_white_win_probs_RandomForest(
    opening_eco="B12",
    white_level="experto",
    rated=True
)
print("\nRandom Forest - B12 experto rated:")
print(ejemplo_rf_2)