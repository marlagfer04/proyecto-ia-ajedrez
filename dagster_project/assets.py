import os
import pandas as pd
import dagster as dg
import matplotlib.pyplot as plt
import joblib

from datos import cargar_datos

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


# =========================================================
# ASSET 1: DATOS BRUTOS
# =========================================================

@dg.asset(group_name="datos")
def raw_games() -> dg.MaterializeResult:
    raw_games_df = pd.read_csv("archive/games.csv")

    return dg.MaterializeResult(
        value=raw_games_df,
        metadata={
            "num_filas": len(raw_games_df),
            "num_columnas": raw_games_df.shape[1]
        }
    )


# =========================================================
# ASSET 2: DATOS LIMPIOS
# =========================================================

@dg.asset(group_name="datos")
def games_limpio(raw_games: pd.DataFrame) -> dg.MaterializeResult:
    # reutilizamos vuestra lógica, pero aquí usamos directamente cargar_datos()
    games = cargar_datos()

    return dg.MaterializeResult(
        value=games,
        metadata={
            "num_filas": len(games),
            "num_columnas": games.shape[1],
            "columnas": list(games.columns)
        }
    )


# =========================================================
# CHECKS DE CALIDAD
# =========================================================

@dg.asset_check(asset=games_limpio)
def check_games_no_vacio(games_limpio: pd.DataFrame) -> dg.AssetCheckResult:
    passed = len(games_limpio) > 0
    return dg.AssetCheckResult(
        passed=passed,
        metadata={"num_filas": len(games_limpio)}
    )


@dg.asset_check(asset=games_limpio)
def check_games_sin_nulos_clave(games_limpio: pd.DataFrame) -> dg.AssetCheckResult:
    columnas_clave = ["opening_eco", "winner_num", "white_level_num", "black_level_num"]
    nulos = games_limpio[columnas_clave].isna().sum().to_dict()
    passed = all(v == 0 for v in nulos.values())

    return dg.AssetCheckResult(
        passed=passed,
        metadata={"nulos_por_columna": nulos}
    )


# =========================================================
# ASSET 3: TABLA DE PROBABILIDADES
# =========================================================

@dg.asset(group_name="descriptiva")
def tabla_probabilidades(games_limpio: pd.DataFrame) -> dg.MaterializeResult:
    prob_table = (
        games_limpio.groupby(["opening_eco", "white_level", "black_level"])["winner"]
        .apply(lambda x: (x == "white").mean())
        .reset_index(name="prob_white_win")
    )

    return dg.MaterializeResult(
        value=prob_table,
        metadata={
            "num_filas": len(prob_table),
            "preview": dg.MetadataValue.md(prob_table.head(10).to_markdown(index=False))
        }
    )


# =========================================================
# ASSET 4: GRÁFICOS DESCRIPTIVOS
# =========================================================

@dg.asset(group_name="graficos")
def graficos_descriptivos(games_limpio: pd.DataFrame) -> dg.MaterializeResult:
    os.makedirs("graficos_dagster", exist_ok=True)

    # 1. Distribución ratings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(games_limpio["white_rating"], bins=30)
    ax1.set_title("Distribución del rating de las blancas")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Número de partidas")

    ax2.hist(games_limpio["black_rating"], bins=30)
    ax2.set_title("Distribución del rating de las negras")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Número de partidas")

    plt.tight_layout()
    ruta1 = "graficos_dagster/distribucion_ratings.png"
    plt.savefig(ruta1)
    plt.close()

    # 2. Histograma rating_diff
    plt.figure(figsize=(8, 4))
    plt.hist(games_limpio["rating_diff"], bins=30)
    plt.title("Distribución de la diferencia del rating")
    plt.xlabel("Diferencia")
    plt.ylabel("Número de partidas")
    plt.tight_layout()
    ruta2 = "graficos_dagster/histograma_rating_diff.png"
    plt.savefig(ruta2)
    plt.close()

    # 3. Victorias
    plt.figure(figsize=(6, 4))
    games_limpio["winner"].value_counts().plot(kind="bar")
    plt.title("Número de victorias por color")
    plt.xlabel("Ganador")
    plt.ylabel("Número de partidas")
    plt.tight_layout()
    ruta3 = "graficos_dagster/victorias.png"
    plt.savefig(ruta3)
    plt.close()

    rutas = [ruta1, ruta2, ruta3]

    return dg.MaterializeResult(
        metadata={
            "num_graficos": len(rutas),
            "graficos": dg.MetadataValue.json(rutas)
        }
    )


# =========================================================
# ASSET 5: MODELO ML
# =========================================================

@dg.asset(group_name="ml")
def modelo_ml(games_limpio: pd.DataFrame) -> dg.MaterializeResult:
    os.makedirs("modelos_ml", exist_ok=True)

    ml_data = games_limpio.copy()
    ml_data["rated_num"] = ml_data["rated"].map({True: 1, False: 0})

    features = ["opening_eco", "white_level_num", "black_level_num", "rated_num"]
    target = "winner_num"

    ml_data = ml_data.dropna(subset=features + [target]).copy()

    X = ml_data[features]
    y = ml_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    categorical_features = ["opening_eco"]
    numeric_features = ["white_level_num", "black_level_num", "rated_num"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

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

    resultados = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        resultados[name] = f1

    best_model_name = max(resultados, key=resultados.get)
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    ruta_modelo = "modelos_ml/best_chess_model.pkl"
    joblib.dump(best_model, ruta_modelo)

    resultados_df = pd.DataFrame.from_dict(resultados, orient="index", columns=["f1"])
    ruta_metricas = "modelos_ml/model_metrics_dagster.csv"
    resultados_df.to_csv(ruta_metricas)

    return dg.MaterializeResult(
        metadata={
            "mejor_modelo": best_model_name,
            "ruta_modelo": ruta_modelo,
            "ruta_metricas": ruta_metricas,
            "resultados": dg.MetadataValue.md(resultados_df.to_markdown())
        }
    )