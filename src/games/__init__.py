"""Games package for PSRO implementations."""

from games.game import (
    Game,
    run_self_play,
    run_PSRO_uniform,
    run_PSRO_uniform_weaker,
    run_PSRO_uniform_stronger,
)

__all__ = [
    "Game",
    "contract",
    "run_self_play",
    "run_PSRO_uniform",
    "run_PSRO_uniform_weaker",
    "run_PSRO_uniform_stronger",
]

