from datos import cargar_datos

# cargamos los datos manipulados
games = cargar_datos()

# vamos a explorar el DataFrame de nuevo para ver cómo ha quedado
print("Primeras filas del dataset:")
print(games.head())
print("\nInformación general:")
print(games.info())
print("\nDimensión del DataFrame:")
print(games.shape)
print("\nResumen estadístico de variables numéricas:")
print(games.describe())
print("\nResumen incluyendo variables categóricas:")
print(games.describe(include="all"))

prob_table = (
    games.groupby(["opening_eco", "white_level", "black_level"])["winner"]
    .apply(lambda x: (x == "white").mean())
    .reset_index(name="prob_white_win")
)
print("\nTabla de probabilidades de victoria de blancas:")
print(prob_table.head(10))

prob_table_fiable = (
    games.groupby(["opening_eco", "white_level", "black_level"])["winner"]
    .agg(
        n_games="size",
        prob_white_win=lambda x: (x == "white").mean()
    )
    .reset_index()
)
print("\nTabla de probabilidades con número de partidas:")
print(prob_table_fiable.head(10))

# top aperturas por nivel
top_openings_by_level = (
    games.groupby(["white_level", "opening_eco"])["winner"]
    .agg(
        n_games="size",
        prob_white_win=lambda x: (x == "white").mean()
    )
    .reset_index()
)

# queremos quedarnos con aquellas aperturas para las que se hayan jugado más de 20
# partidas para así obtener unos resultados más fiables
top_openings_by_level = top_openings_by_level[top_openings_by_level["n_games"] > 20]

top3_openings_by_level = (
    top_openings_by_level
    .sort_values(["white_level", "prob_white_win", "n_games"],
                ascending=[True, False, False])
    .groupby("white_level")
    .head(3)
    .reset_index(drop=True)
)

print("\nTop 3 aperturas por nivel:")
print(top3_openings_by_level)
