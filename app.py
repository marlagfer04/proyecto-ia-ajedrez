import os
import joblib
import pandas as pd
import streamlit as st

from datos import cargar_datos
from tensorflow.keras.models import load_model


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Predicción de ajedrez",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CARGA DE DATOS Y MODELOS
# =========================================================

@st.cache_data
def cargar_dataset():
    return cargar_datos()


@st.cache_resource
def cargar_modelo_reglog():
    ruta_modelo = "modelos_ml/logistic_regression_model.pkl"
    if os.path.exists(ruta_modelo):
        return joblib.load(ruta_modelo)
    return None


@st.cache_resource
def cargar_modelo_randomforest():
    ruta_modelo = "modelos_ml/random_forest_model.pkl"
    if os.path.exists(ruta_modelo):
        return joblib.load(ruta_modelo)
    return None


@st.cache_resource
def cargar_modelo_dl():
    ruta_modelo = "modelos_dl/chess_dl_model.keras"
    if os.path.exists(ruta_modelo):
        return load_model(ruta_modelo)
    return None


@st.cache_resource
def cargar_preprocessor_dl():
    ruta_preprocessor = "modelos_dl/chess_dl_preprocessor.pkl"
    if os.path.exists(ruta_preprocessor):
        return joblib.load(ruta_preprocessor)
    return None


games = cargar_dataset()
modelo_reglog = cargar_modelo_reglog()
modelo_randomforest = cargar_modelo_randomforest()
modelo_dl = cargar_modelo_dl()
preprocessor_dl = cargar_preprocessor_dl()


# =========================================================
# TABLA DE APERTURAS
# =========================================================

openings_df = (
    games[["opening_eco", "opening_name"]]
    .drop_duplicates()
    .sort_values("opening_name")
    .reset_index(drop=True)
)

openings_df["label"] = openings_df["opening_name"] + " (" + openings_df["opening_eco"] + ")"

label_to_eco = dict(zip(openings_df["label"], openings_df["opening_eco"]))
eco_to_name = dict(zip(openings_df["opening_eco"], openings_df["opening_name"]))


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def predict_white_win_probs_ml(opening_eco, white_level, rated=True, model=None):
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
        "Nivel negras": black_levels,
        "Probabilidad de victoria de blancas (%)": (probs * 100).round(2)
    })


def predict_white_win_probs_dl(
    opening_eco,
    white_rating,
    rated=True,
    black_ratings=(1000, 1500, 2000),
    model=None,
    preprocessor=None
):
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

    return pd.DataFrame({
        "Rating negras": list(black_ratings),
        "Probabilidad de victoria de blancas (%)": (probs * 100).round(2)
    })


def obtener_top3_aperturas_por_nivel(white_level):
    top_openings_by_level = (
        games.groupby(["white_level", "opening_eco", "opening_name"])["winner"]
        .agg(
            n_games="size",
            prob_white_win=lambda x: (x == "white").mean()
        )
        .reset_index()
    )

    top_openings_by_level = top_openings_by_level[top_openings_by_level["n_games"] > 20]

    top3 = (
        top_openings_by_level[top_openings_by_level["white_level"] == white_level]
        .sort_values(["prob_white_win", "n_games"], ascending=[False, False])
        .head(3)
        .reset_index(drop=True)
    )

    top3["prob_white_win"] = (top3["prob_white_win"] * 100).round(2)

    return top3


def mostrar_galeria_graficos(carpeta, key, pequeno=True):
    if not os.path.exists(carpeta):
        st.warning(f"No se ha encontrado la carpeta: {carpeta}")
        return

    imagenes = sorted([
        f for f in os.listdir(carpeta)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(imagenes) == 0:
        st.warning("No hay gráficos disponibles en esta carpeta.")
        return

    grafico_elegido = st.selectbox(
        "Selecciona el gráfico que quieres visualizar",
        imagenes,
        key=f"selectbox_{key}"
    )

    ruta = os.path.join(carpeta, grafico_elegido)

    ampliar = st.checkbox(
        "Ampliar gráfico",
        value=False,
        key=f"ampliar_{key}"
    )

    if pequeno and not ampliar:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(ruta, use_container_width=True)
    else:
        st.image(ruta, use_container_width=True)


def mostrar_prediccion_personalizada(modelo_elegido, rated):
    apertura_label = st.selectbox(
        "Selecciona una apertura",
        openings_df["label"],
        key="apertura_personalizada"
    )

    if modelo_elegido in ["Regresión logística", "Random Forest"]:
        white_level = st.selectbox(
            "Selecciona el nivel de las blancas",
            ["principiante", "avanzado", "experto"],
            key="nivel_personalizado"
        )

        if st.button("Predecir con apertura personalizada"):
            opening_eco = label_to_eco[apertura_label]
            opening_name = eco_to_name[opening_eco]

            if modelo_elegido == "Regresión logística":
                modelo_seleccionado = modelo_reglog
            else:
                modelo_seleccionado = modelo_randomforest

            if modelo_seleccionado is None:
                st.error(f"No se ha encontrado el modelo de {modelo_elegido} en la carpeta modelos_ml.")
            else:
                resultado = predict_white_win_probs_ml(
                    opening_eco=opening_eco,
                    white_level=white_level,
                    rated=rated,
                    model=modelo_seleccionado
        )
                st.subheader("Resultado de la predicción personalizada")
                st.write(f"*Apertura seleccionada:* {opening_name} ({opening_eco})")
                st.write(f"*Nivel de blancas:* {white_level}")
                st.write(f"*Tipo de partida:* {'Rated' if rated else 'No rated'}")
                st.dataframe(resultado, hide_index=True, use_container_width=True)

                st.bar_chart(
                    resultado.set_index("Nivel negras")[
                        "Probabilidad de victoria de blancas (%)"
                    ]
                )

    else:
        white_rating = st.number_input(
            "Introduce el rating de las blancas",
            min_value=0,
            max_value=3000,
            value=1500,
            step=50,
            key="rating_personalizado"
        )

        if st.button("Predecir con apertura personalizada"):
            opening_eco = label_to_eco[apertura_label]
            opening_name = eco_to_name[opening_eco]

            if modelo_dl is None or preprocessor_dl is None:
                st.error("No se ha encontrado el modelo o el preprocesador de Deep Learning.")
            else:
                resultado = predict_white_win_probs_dl(
                    opening_eco=opening_eco,
                    white_rating=white_rating,
                    rated=rated,
                    black_ratings=(1000, 1500, 2000),
                    model=modelo_dl,
                    preprocessor=preprocessor_dl
                )

                st.subheader("Resultado de la predicción personalizada")
                st.write(f"*Apertura seleccionada:* {opening_name} ({opening_eco})")
                st.write(f"*Rating de blancas:* {white_rating}")
                st.write(f"*Tipo de partida:* {'Rated' if rated else 'No rated'}")
                st.dataframe(resultado, hide_index=True, use_container_width=True)

                st.bar_chart(
                    resultado.set_index("Rating negras")[
                        "Probabilidad de victoria de blancas (%)"
                    ]
                )


# =========================================================
# MENÚ LATERAL
# =========================================================

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
    overflow: hidden !important;
    background-color: #11131c !important;
    background-image:
        linear-gradient(45deg, rgba(255,255,255,0.05) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,0.05) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.05) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.05) 75%);
    background-size: 40px 40px;
    background-position: 0 0, 0 20px, 20px -20px, -20px 0px;
}

    section[data-testid="stSidebar"] > div {
        overflow: hidden !important;
        height: 100vh !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #e6e9f0 !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.65rem 0.75rem !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        width: 100% !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;
        transform: none !important;
    }

    .sidebar-spacer {
        height: calc(100vh - 550px);
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("♟️ MENÚ")

    if "pagina" not in st.session_state:
        st.session_state["pagina"] = "Inicio"

    if st.button("🏠 Inicio"):
        st.session_state["pagina"] = "Inicio"

    if st.button("🚀 Motivación"):
        st.session_state["pagina"] = "Motivación"

    if st.button("📊 Resumen"):
        st.session_state["pagina"] = "Resumen"

    if st.button("🏆 Recomendación y predicción"):
        st.session_state["pagina"] = "Recomendación y predicción"


    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    st.markdown("---")


    if st.button("👥 ¿Quiénes somos?"):
        st.session_state["pagina"] = "¿Quiénes somos?"

pagina = st.session_state["pagina"]

# =========================================================
# PÁGINA 0: INICIO
# =========================================================

if pagina == "Inicio":
    st.markdown(
        """
        <div style="text-align: center; padding-top: 80px;">
            <h1 style="font-size: 3.2rem;">♟️ Bienvenido a nuestra aplicación de ajedrez</h1>
            <p style="font-size: 1.3rem; color: gray;">
                Explora el mundo del ajedrez a través de datos, estrategia y predicción.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="text-align: center;">
                <h3>📊 Análisis</h3>
                <p>Descubre patrones en miles de partidas reales.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <h3>🤖 Predicción</h3>
                <p>Utiliza modelos de Machine Learning y Deep Learning.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style="text-align: center;">
                <h3>♟️ Estrategia</h3>
                <p>Encuentra las mejores aperturas para tu nivel.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center; font-size: 1.1rem;">
            "El ajedrez es el gimnasio de la mente" – Blaise Pascal
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# PÁGINA 1: MOTIVACIÓN
# =========================================================

if pagina == "Motivación":
    st.title("🚀 Motivación del proyecto")

    st.markdown(
        """
        Esta aplicación forma parte de un proyecto de análisis de datos y predicción aplicado
        a partidas de ajedrez.

        El proyecto combina análisis descriptivo, visualización de datos, modelos clásicos
        de Machine Learning, Deep Learning con TensorFlow/Keras y orquestación mediante Dagster.

        ### Objetivo del proyecto

        El objetivo principal es estimar la probabilidad de victoria de las blancas
        a partir de variables como la apertura, el nivel o rating de los jugadores,
        el tipo de partida y el modelo predictivo seleccionado.

        ### Tecnologías utilizadas

        <div style="margin-left: 20px">

        *Lenguaje y entorno*
        - Python  

        *Análisis y manipulación de datos*
        - Pandas  
        - NumPy  

        *Visualización*
        - Matplotlib  

        *Machine Learning*
        - Scikit-learn  
        - Joblib  

        *Deep Learning*
        - TensorFlow / Keras  

        *Aplicación y despliegue*
        - Streamlit  

        *Orquestación de datos*
        - Dagster  

        </div>  
        """,unsafe_allow_html=True
    )

# =========================================================
# PÁGINA 2: RESUMEN
# =========================================================

if pagina == "Resumen":
    st.title("♟️ Exploración y resumen del dataset")

    st.subheader("Resumen del dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Número de partidas", games.shape[0])

    with col2:
        st.metric("Número de variables", games.shape[1])

    with col3:
        st.metric("Aperturas distintas", games["opening_eco"].nunique())

    st.subheader("Muestra de los datos")
    st.dataframe(games.head(10), use_container_width=True)

    st.subheader("Gráficos descriptivos del dataset")

    st.markdown(
        """
        Estos gráficos proceden directamente del análisis descriptivo del dataset,
        antes de aplicar los modelos predictivos.
        """
    )

    mostrar_galeria_graficos(
        carpeta="graficos_descriptiva",
        key="descriptiva",
        pequeno=True
    )


# =========================================================
# PÁGINA 3: RECOMENDACIÓN Y PREDICCIÓN
# =========================================================

elif pagina == "Recomendación y predicción":
    st.title("🏆 Recomendación de aperturas y predicción personalizada")

    st.markdown(
        """
        En esta sección puedes obtener primero una recomendación de aperturas para tu nivel
        y después realizar una predicción personalizada con la apertura que prefieras.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # PARTE 1: RECOMENDADOR
    # -----------------------------------------------------

    st.header("1. Recomendador de aperturas")

    white_level_recomendador = st.selectbox(
        "Selecciona el nivel de las blancas para recibir recomendaciones",
        ["principiante", "avanzado", "experto"],
        key="nivel_recomendador"
    )

    if st.button("Recomendar aperturas"):
        top3 = obtener_top3_aperturas_por_nivel(white_level_recomendador)

        if top3.empty:
            st.warning("No hay aperturas suficientes para este nivel.")
        else:
            st.subheader("Top 3 aperturas recomendadas")

            tabla_mostrar = top3[[
                "opening_eco",
                "opening_name",
                "n_games",
                "prob_white_win"
            ]].rename(columns={
                "opening_eco": "Código ECO",
                "opening_name": "Apertura",
                "n_games": "Número de partidas",
                "prob_white_win": "Winrate blancas (%)"
            })

            st.dataframe(tabla_mostrar, hide_index=True, use_container_width=True)

            st.info(
                "Estas aperturas son las que presentan mayor porcentaje histórico "
                "de victoria de las blancas para el nivel seleccionado, considerando "
                "solo aperturas con más de 20 partidas registradas."
            )

    st.divider()

    # -----------------------------------------------------
    # PARTE 2: PREDICCIÓN PERSONALIZADA
    # -----------------------------------------------------

    st.header("2. Predicción personalizada")

    modelo_elegido = st.selectbox(
        "Selecciona el modelo que quieres utilizar",
        ["Regresión logística", "Random Forest", "Deep Learning"],
        key="modelo_prediccion"
    )

    rated = st.radio(
        "Tipo de partida",
        options=[True, False],
        format_func=lambda x: "Rated" if x else "No rated",
        horizontal=True,
        key="rated_prediccion"
    )

    st.markdown(
        """
        Puedes usar una de las aperturas recomendadas o seleccionar cualquier otra apertura
        disponible en el dataset.
        """
    )

    mostrar_prediccion_personalizada(modelo_elegido, rated)

    st.divider()

    # -----------------------------------------------------
    # PARTE 3: GRÁFICOS DEL MODELO SELECCIONADO
    # -----------------------------------------------------

    st.header("3. Gráficos asociados al modelo seleccionado")

    if "mostrar_graficos_modelo" not in st.session_state:
        st.session_state["mostrar_graficos_modelo"] = False

    if st.button("Ver gráficos del modelo seleccionado"):
        st.session_state["mostrar_graficos_modelo"] = True

    if st.session_state["mostrar_graficos_modelo"]:
        if modelo_elegido == "Regresión logística":
            st.subheader("Gráficos del modelo de Regresión Logística")
            mostrar_galeria_graficos(
                carpeta="graficos_ml_reglog",
                key="ml_reglog",
                pequeno=True
            )

        elif modelo_elegido == "Random Forest":
            st.subheader("Gráficos del modelo Random Forest")
            mostrar_galeria_graficos(
                carpeta="graficos_ml_randomforest",
                key="ml_randomforest",
                pequeno=True
            )

        else:
            st.subheader("Gráficos del modelo de Deep Learning")
            mostrar_galeria_graficos(
                carpeta="graficos_dl",
                key="dl",
                pequeno=True
            )

# =========================================================
# PÁGINA 3: QUIÉNES SOMOS
# =========================================================

elif pagina == "¿Quiénes somos?":
    st.title("👥 ¿Quiénes somos?")

    st.markdown(
        """
        Esta aplicación ha sido desarrollada como parte de un proyecto académico
        de la asignatura optativa del quinto curso, Inteligencia Artificial y Estadística, del DGME de la Universidad de Sevilla

        ### Equipo

        - Marta Laguna Fernández (marlagfer@alum.us.es)
        - Jimena Toro Trigo (jimtortri@alum.us.es)

        ### Sobre nosotras

        Somos estudiantes del Doble Grado en Matemáticas y Estadística, con interés en el análisis de datos, la modelización predictiva y la aplicación de técnicas de inteligencia artificial a problemas reales.

        Elegimos el ajedrez como contexto del proyecto porque combina de forma natural estrategia, toma de decisiones y datos estructurados, lo que lo convierte en un entorno especialmente interesante para aplicar técnicas de análisis y predicción.

        Además, nos ha permitido trabajar con un problema intuitivo y fácilmente interpretable, donde los resultados pueden entenderse tanto desde un punto de vista técnico como desde la experiencia del usuario, facilitando así la conexión entre el modelo y su aplicación práctica.
        """
    )