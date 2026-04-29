from dagster import Definitions

from .assets import (
    raw_games,
    games_limpio,
    tabla_probabilidades,
    graficos_descriptivos,
    modelo_ml,
    check_games_no_vacio,
    check_games_sin_nulos_clave,
)

defs = Definitions(
    assets=[
        raw_games,
        games_limpio,
        tabla_probabilidades,
        graficos_descriptivos,
        modelo_ml,
    ],
    asset_checks=[
        check_games_no_vacio,
        check_games_sin_nulos_clave,
    ],
)
