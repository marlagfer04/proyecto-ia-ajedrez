import pandas as pd
from datos import cargar_datos
import matplotlib.pyplot as plt
import os
import seaborn as sns

# volvemos a cargar los datos
games = cargar_datos()

# crea la carpeta donde guardaremos los gráficos
os.makedirs("graficos_descriptiva", exist_ok=True)

#Realizamos gráficos de interés 
#1) Distribución de ratings para ambos (white & black)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
# white
ax1.hist(games["white_rating"],bins=30)
ax1.set_title("Distribución del rating de las blancas")
ax1.set_xlabel("Rating")
ax1.set_ylabel("Número de partidas")
# black
ax2.hist(games["black_rating"],bins=30)
ax2.set_title("Distribución del rating de las negras")
ax2.set_xlabel("Rating")
ax2.set_ylabel("Número de partidas")

plt.tight_layout()
plt.savefig("graficos_descriptiva/distribucion_ratings.png")
plt.close()

#2) Histograma del rating_diff
plt.hist(games["rating_diff"],bins=30)
plt.title("Distribución de la diferencia del rating")
plt.xlabel("Diferencia")
plt.ylabel("Número de partidas")

plt.savefig("graficos_descriptiva/histograma_rating_diff.png")
plt.close()

# 3) Victorias de las blancas vs las negras
games["winner"].value_counts().plot(kind="bar")
plt.title("Número de victorias por color")
plt.xlabel("Ganador")
plt.ylabel("Número de partidas")

plt.savefig("graficos_descriptiva/victorias.png")
plt.close()

# 4) Winrate según si la partida es amistosa o profesional
pd.crosstab(games["rated"],games["winner"]).plot(kind="bar")
plt.title("Resultado según si la partida es amistosa o profesional")
plt.xlabel("Rated")
plt.ylabel("Número de partidas")

plt.savefig("graficos_descriptiva/rated_vs_winner.png")
plt.close()

# 5) Top 10 aperturas
top_openings = games["opening_name"].value_counts().head(10)
top_openings.plot(kind="bar")
plt.title("Top 10 aperturas más jugadas")
plt.xlabel("Apertura")
plt.ylabel("Número de partidas")

plt.savefig("graficos_descriptiva/top_aperturas.png")
plt.close()

# 6) Boxplot de ratings (blancas vs negras)
# creamos el formato largo para el boxplot

    # vamos ahora a crear una tabla: 
    # white_rating    black_rating
    #    ...               ...

    # ahora cambiamos el formato wide_to_long
    #    color    rating
    #     ...       ...

ratings_melt = (
    games[["game_id", "white_rating", "black_rating"]]
    .set_index("game_id")
    .stack()
    .reset_index()
)

ratings_melt.columns = ["game_id", "color", "rating"]

plt.figure(figsize=(6, 4))
sns.boxplot(data=ratings_melt, x="color", y="rating")

plt.title("Comparación de ratings entre blancas y negras")
plt.xlabel("Color")
plt.ylabel("Rating")

plt.savefig("graficos_descriptiva/boxplot_ratings.png")
plt.close()

# 7) Probabilidad de victoria según rating_diff
# trabajamos sobre copia para no modificar el dataset original
# para no tener que preocuparnos de eliminar esta variable
games_aux = games.copy()

games_aux["rating_bin"] = pd.cut(games_aux["rating_diff"], bins=20)

prob_by_rating = (
    games_aux.groupby("rating_bin")["winner"]
    .apply(lambda x: (x == "white").mean())
    .reset_index(name="prob_white_win")
)

prob_by_rating["rating_mid"] = prob_by_rating["rating_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(8, 4))
plt.plot(prob_by_rating["rating_mid"], prob_by_rating["prob_white_win"])

plt.xlabel("Diferencia de rating")
plt.ylabel("Probabilidad de ganar blancas")
plt.title("Probabilidad real de victoria de blancas según rating_diff")

plt.tight_layout()
plt.savefig("graficos_descriptiva/probabilidad_rating_diff.png")
plt.close()