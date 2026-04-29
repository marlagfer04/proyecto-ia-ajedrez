# ♟️ Predicción de victoria en partidas de ajedrez

**Trabajo Final — Asignatura de Inteligencia Artificial y Estadística**  
Curso 2025–2026

**Autoras:**

- Marta Laguna Fernández
- Jimena Toro Trigo

---

## 1. Descripción del proyecto

Este proyecto desarrolla una aplicación interactiva para analizar partidas de ajedrez y predecir la probabilidad de victoria de las piezas blancas.

A partir de un conjunto de datos de partidas reales, se realiza un proceso completo de:

- carga y limpieza de datos;
- análisis descriptivo;
- visualización de patrones;
- entrenamiento de modelos de Machine Learning;
- entrenamiento de un modelo de Deep Learning;
- comparación de resultados;
- orquestación mediante Dagster;
- despliegue de una aplicación interactiva con Streamlit.

La aplicación incorpora dos funcionalidades principales:

- **Recomendación de aperturas:** basada en un análisis descriptivo del dataset, que permite identificar las aperturas más eficaces según el nivel del jugador.
- **Predicción personalizada:** el usuario selecciona una apertura, su nivel o rating (ELO) y el tipo de partida, obteniendo una estimación de la probabilidad de victoria de las blancas mediante distintos modelos predictivos.

De este modo, el proyecto combina técnicas de análisis de datos, modelización predictiva y desarrollo de aplicaciones interactivas.

---

## 2. Problema abordado

El ajedrez es un juego estratégico en el que influyen múltiples factores: el nivel de los jugadores, la apertura elegida, el tipo de partida y la diferencia de rating.

El objetivo principal del proyecto es responder a la siguiente pregunta:

> ¿Es posible estimar la probabilidad de victoria de las blancas a partir de variables conocidas antes o al inicio de la partida?

Por este motivo, se eliminan aquellas variables que contienen información posterior al resultado de la partida.

Para ello, se utilizan variables como:

- apertura de la partida (`opening_eco`);
- rating de blancas;
- rating de negras;
- diferencia de rating (`rating_diff`);
- tipo de partida (`rated`);
- nivel del jugador: principiante, avanzado o experto.

---

## 3. Estructura del proyecto

El proyecto está organizado en distintos módulos:

- `datos.py` → carga y preprocesamiento de datos
- `descriptiva.py` → análisis exploratorio
- `graficos.py` → generación de visualizaciones
- `ml.py` → modelos de Machine Learning
- `deep_learning.py` → modelo de red neuronal (tensorflow)
- `dagster_project` → carpeta donde se encuentran los ficheros `__init__.py`,`assets.py` y `definitions.py`, necesarios para la orquestación de los datos mediante Dagster
- `app.py` → aplicación interactiva
- `main.py` → ejecución completa del pipeline

---

## 4. Entorno de ejecución

Para crear el entorno virtual, ejecutar en terminal:

```
uv venv --python 3.11.14
```

```
uv init
```

```
uv sync
```

Para activar el entorno:

```
.venv\Scripts\activate
```

---

## 5. Instalación de dependencias

En caso de no tener pip instalado:

```
python -m ensurepip --upgrade
```

```
python -m pip --version
```

```
python -m pip install --upgrade pip
```

Si ya lo tenemos, escribimos por consola, instalando manualmente las dependencias:

```
pip install pandas
pip install matplotlib
pip install seaborn
pip install joblib
pip install scikit-learn
pip install dagster
pip install dagster-webserver
pip install tensorflow
pip install streamlit
```

---

## 6. Ejecución del proyecto

Para ejecutar todo el flujo completo del proyecto:

```
python main.py
```

Este script ejecuta:

- análisis descriptivo
- generación de gráficos
- entrenamiento de modelos ML
- entrenamiento del modelo DL

---

## 7. Dagster

Para acceder a Dagster, escribimos en la terminal:

```
python -m dagster dev -m dagster_project.definitions
```

En la UI de Dagster, accedemos a _Lineage_ y pinchamos en _Materialize all_

---

## 8. Aplicación interactiva

La aplicación ha sido desarrollada con **Streamlit** y permite:

- explorar información general del dataset;
- visualizar gráficos descriptivos;
- seleccionar un modelo predictivo;
- introducir valores mediante menús desplegables y controles interactivos;
- obtener predicciones personalizadas;
- consultar gráficos asociados a cada modelo.

Para lanzar la aplicación:

```
streamlit run app.py
```

---

## 9. Reproducibilidad

A modo de resumen, el proyecto puede reproducirse completamente siguiendo estos pasos (previamente instaladas las dependencias necesarias):

```
uv venv --python 3.11.14
```

```
uv sync
```

```
python main.py
```

```
python -m dagster dev -m dagster_project.definitions
```

```
streamlit run app.py
```

---

## 10. Enlace a la aplicación

La aplicación está disponible en:

```
https://proyecto-ia-ajedrez.streamlit.app/
```

En caso de que la aplicación no esté activa, puede ejecutarse localmente siguiendo las instrucciones de este README.

---

## 11. Observaciones

- El dataset utilizado se encuentra en `archive/games.csv`
- Los modelos entrenados se guardan en las carpetas correspondientes
- Los gráficos generados se almacenan automáticamente en carpetas en función de su naturaleza
- Se consideró el uso de la herramienta `Dask` para el procesamiento de datos. Sin embargo, dado que el dataset cuenta con aproximadamente 20.000 observaciones y siguiendo las recomendaciones de nuestros profesores, se concluyó que su uso no resulta necesario en este caso. Este tipo de herramientas es especialmente útil en escenarios con volúmenes de datos mucho mayores (del orden de millones de registros)
