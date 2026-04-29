import pandas as pd

# en este script creamos funciones para tener disponibles los datos en 
# todos los script sin tener que repetir el código



def cargar_datos():
    # cargamos datos brutos
    raw_games = pd.read_csv("archive/games.csv")

    # eliminamos las columnas que no nos interesan
    # id: no tiene un significado relevante en nuestro estudio, era de 0:nrow(datas)
    # created_at: el momento en el que se crea la partida no afecta al ganador
    # turns: número total de movimientos en la partida. No nos interesa porque es una variable 
    #        que se calcula una vez se ha jugado toda la partida
    # moves: toda la secuencia de movimientos de la partida, análogo a turns
    # increment_code: tiempo de juego. No le podemos sacar partido
    # last_move_at: último movimiento, análogo a turns
    # victory_status: implica conocer el final de la partida. De nuevo, no nos interesa
    # white_id & black_id: simple numeración, no nos sirve para nuestro estudio
    # opening_ply: número de movimientos de la apertura hasta ser identficada. No nos interesa porque
    #              nuestro estudio no se basa en identificar la apertura, el usuario elige una apertura
    #              concreta, la sabe de antemano
    games=raw_games.drop(columns=["id","created_at","turns","moves",
                                    "increment_code","last_move_at",
                                    "victory_status","white_id",
                                    "black_id","opening_ply"])
    
    # creamos la nueva variable llamada: rating_diff
    # recuerda que rating es el nivel (ELO) de cada jugador
    # si rating_diff > 0 ===> rating blancas > rating de negras
    # si rating_diff < 0 ===> rating blancas < rating de negras 
    games["rating_diff"]=games["white_rating"]-games["black_rating"]
    
    # filtramos los datos para quedarnos con aquellos que no sean empates
    games = games[games["winner"]!="draw"]

    # tabla resumen por apertura
    opening_stats = (
        games.groupby("opening_eco")
        .agg(
            n_games=("winner", "size"),
            white_win_rate=("winner", lambda x: (x == "white").mean())
        )
        .reset_index()
    )

    # unimos la tabla resumen al dataset principal
    games = games.drop(columns=["n_games", "white_win_rate"], errors="ignore")

    games = games.merge(
        opening_stats,
        on="opening_eco",
        how="left",
        validate="many_to_one"
    )

    # creamos el id, para identificar a que partida corresponde
    if "game_id" not in games.columns:
        games.insert(0, "game_id", games.index)

    # añadimos variable objetivo numérica
    games["winner_num"] = games["winner"].map({"white": 1, "black": 0})

    # creamos niveles
    games["white_level"] = pd.cut(
        games["white_rating"],
        bins=[0, 1200, 1800, 3000],
        labels=["principiante", "avanzado", "experto"]
    )

    games["black_level"] = pd.cut(
        games["black_rating"],
        bins=[0, 1200, 1800, 3000],
        labels=["principiante", "avanzado", "experto"]
    )

    # codificamos niveles
    level_map = {
        "principiante": 0,
        "avanzado": 1,
        "experto": 2
    }

    games["white_level_num"] = games["white_level"].map(level_map)
    games["black_level_num"] = games["black_level"].map(level_map)

    return games

