import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
from typing import List, Optional, Tuple
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
import concurrent.futures

# utils / helpers
from games.utils import (
    _compute_pairs_to_evaluate,
    _compute_payoffs_serial,
    _compute_payoffs_parallel
)



class Game(ABC):

    def play(self, u, v) -> float:
        pass

    def improve(self, u, v, **kwargs):
        """
        Takes in agents u and v. Returns a policy u_new that improves on u
        when playing against v.
        """
        pass

    def improve_from_transcripts(self, u, transcripts, **kwargs):
        """
        Takes in an agent u and a set of game transcripts which may be in a format
        specified by the game. Assumes that u is the first agent in all transcripts.

        Transcripts should be a list of representations of various games that need-not 
        be from the same opponent. In LLMGames this will be fed into the optimizer llm.
        """
        pass

def compute_empirical_payoff_matrix(
    population: list,
    game: Game,
    antisymmetric: bool = True,
    n_games: int = 1,
    parallel: bool = False,
    cached_payoff_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes the empirical payoff matrix by taking in a population and playing
    each agent against eachother for n_games and fillin in the corresponding matrix entry

    Games can be run serially or in parllel.
    """
    n = len(population)
    
    if cached_payoff_matrix is not None:
        m = cached_payoff_matrix.shape[0]
        payoff_matrix = np.pad(cached_payoff_matrix, ((0, n - m), (0, n - m)), mode='constant')
    else:
        m = 0
        payoff_matrix = np.zeros((n, n), dtype=float)
    
    # Determine which games to compute
    agent_indices_to_eval = list(range(m, n)) # Only compute payoffs for new agents (indices m to n-1)
    pairs = _compute_pairs_to_evaluate(agent_indices_to_eval, n, antisymmetric)
    
    # Rollout games, compute payoffs
    with tqdm(
        total=len(pairs), 
        desc=f"Computing Payoff Matrix (indices {m} to {n-1}) ({'parallel' if parallel else 'serial'})" 
    ) as pbar:
        if parallel:
            _compute_payoffs_parallel(pairs, population, game, antisymmetric, n_games, payoff_matrix, pbar)
        else:
            _compute_payoffs_serial(pairs, population, game, antisymmetric, n_games, payoff_matrix, pbar)
    
    # Ensure diagonal is zero
    for i in range(n):
        payoff_matrix[i, i] = 0.0
    
    return payoff_matrix


# ---- Population Distribution Helpers ----

def get_self_play_distribution(agent_idx, payoff_matrix):
    n = payoff_matrix.shape[0]
    dist = np.zeros(n)
    dist[agent_idx] = 1.0
    return dist

def get_uniform_distribution(agent_idx, payoff_matrix):
    n = payoff_matrix.shape[0]
    dist = np.ones(n) / n
    return dist

def get_uniform_weaker_distribution(agent_idx, payoff_matrix):
    n = payoff_matrix.shape[0]
    weaker = [i for i in range(n) if i != agent_idx and payoff_matrix[agent_idx, i] > 0]
    dist = np.zeros(n)
    if weaker:
        for idx in weaker:
            dist[idx] = 1.0 / len(weaker)
    else:
        dist[agent_idx] = 1.0
    return dist

def get_uniform_stronger_distribution(agent_idx, payoff_matrix):
    n = payoff_matrix.shape[0]
    stronger = [i for i in range(n) if i != agent_idx and payoff_matrix[agent_idx, i] < 0]
    dist = np.zeros(n)
    if stronger:
        for idx in stronger:
            dist[idx] = 1.0 / len(stronger)
    else:
        dist[agent_idx] = 1.0
    return dist

# ---- PSRO Update Rules ----

def run_PSRO(agent_idx: int, population: list, game: Game, payoff_matrix: np.ndarray, opp_distribution_fn, **kwargs):
    """
    Samples a single opponent according to the provided population distribution fn.
    """
    del kwargs['n_games']
    dist = opp_distribution_fn(agent_idx, payoff_matrix)
    rand_idx = np.random.choice(len(population), p=dist)
    return game.improve(population[agent_idx], population[rand_idx], **kwargs)

def run_PSRO_from_transcripts(agent_idx: int, population: list, game: Game, payoff_matrix, n_games: int, opp_distribution_fn, **kwargs):
    """
    Samples n_games opponents according to the provided population distribution fn and
    collects transcripts.
    """
    dist = opp_distribution_fn(agent_idx, payoff_matrix)
    transcripts = []
    for _ in trange(n_games):
        rand_idx = np.random.choice(len(population), p=dist)
        u = population[agent_idx]
        v = population[rand_idx]
        transcripts.append(game.play(u, v, return_transcript=True))
    return game.improve_from_transcripts(population[agent_idx], transcripts)

## -------------------------------------------------------------------- ##
"""
NOTE(Faraz): The following functions accomodate the old interface by composing 
run_PSRO with a specific opp_distribution_fn to recreate the old interface 
such as to not break old code 
"""
def run_self_play(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", **kwargs):
    return run_PSRO(agent_idx, population, game, payoff_matrix, opp_distribution_fn=get_self_play_distribution, **kwargs)

def run_PSRO_uniform(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", **kwargs):
    return run_PSRO(agent_idx, population, game, payoff_matrix, opp_distribution_fn=get_uniform_distribution, **kwargs)

def run_PSRO_uniform_weaker(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", **kwargs):
    return run_PSRO(agent_idx, population, game, payoff_matrix, opp_distribution_fn=get_uniform_weaker_distribution, **kwargs)

def run_PSRO_uniform_stronger(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", **kwargs):
    return run_PSRO(agent_idx, population, game, payoff_matrix, opp_distribution_fn=get_uniform_stronger_distribution, **kwargs)

## -------------------------------------------------------------------- ##

def run_self_play_from_transcripts(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", n_games: int, **kwargs):
    return run_PSRO_from_transcripts(agent_idx, population, game, payoff_matrix, n_games, opp_distribution_fn=get_self_play_distribution, **kwargs)

def run_PSRO_uniform_from_transcripts(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", n_games: int, **kwargs):
    return run_PSRO_from_transcripts(agent_idx, population, game, payoff_matrix, n_games, opp_distribution_fn=get_uniform_distribution, **kwargs)

def run_PSRO_uniform_weaker_from_transcripts(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", n_games: int, **kwargs):
    return run_PSRO_from_transcripts(agent_idx, population, game, payoff_matrix, n_games, opp_distribution_fn=get_uniform_weaker_distribution, **kwargs)

def run_PSRO_uniform_stronger_from_transcripts(agent_idx: int, population: list, game: "Game", payoff_matrix: "np.ndarray", n_games: int, **kwargs):
    return run_PSRO_from_transcripts(agent_idx, population, game, payoff_matrix, n_games, opp_distribution_fn=get_uniform_stronger_distribution, **kwargs)

## -------------------------------------------------------------------- ##

def train_population_from_last_agent(initial_population, update_rule, game: Game, *, n_iters: int, n_games_per_empirical_payoff: int, **kwargs):
    import numpy as np
    population = list(initial_population)
    payoff_matrix = None

    for _ in range(n_iters):
        # Recompute payoff matrix only when the population has changed (agent added)
        payoff_matrix = compute_empirical_payoff_matrix(population, game, parallel=True, cached_payoff_matrix=payoff_matrix, n_games=n_games_per_empirical_payoff)
        new_agent = update_rule(
            agent_idx = -1, 
            population = population, 
            game = game,
            payoff_matrix = payoff_matrix,
            n_games=10,
            **kwargs
        )
        population.append(new_agent)

    return population, payoff_matrix

def train_population_from_random_agent(initial_population, update_rule, game: Game, *, n_iters: int, n_games_per_empirical_payoff: int, **kwargs):
    import numpy as np
    population = list(initial_population)
    payoff_matrix = None

    for _ in range(n_iters):
        # Recompute payoff matrix only when the population has changed (agent added)
        payoff_matrix = compute_empirical_payoff_matrix(population, game, parallel=True, cached_payoff_matrix=payoff_matrix, n_games=n_games_per_empirical_payoff)
        new_agent = update_rule(
            agent_idx = np.random.randint(0, len(population)), 
            population = population, 
            game = game,
            payoff_matrix = payoff_matrix,
            n_games=10,
            **kwargs
        )
        population.append(new_agent)

    return population, payoff_matrix


if __name__ == "__main__":


    from games.disc.disc_game import DiscGame, get_RPS_triangle
    from games.disc.disc_game_vis import plot_image

    game = DiscGame()
    initial_population = get_RPS_triangle()
    # final_population, payoff_matrix = train_population_from_last_agent(initial_population, run_self_play, game, n_iters=4, n_games_per_empirical_payoff=3)
    # final_population, payoff_matrix = train_population_from_last_agent(initial_population, run_PSRO_uniform_weaker, game, n_iters=80, n_games_per_empirical_payoff=4, learning_rate=.1)
    final_population, payoff_matrix = train_population_from_random_agent(initial_population, run_PSRO_uniform_weaker, game, n_iters=80, n_games_per_empirical_payoff=4, learning_rate=.5)


    # from games.llms.llm2 import LLMRockPaperScissors, rock_prompt, paper_scissors_prompt, example_population 
    # from games.disc.disc_game import rps_to_disc

    # game = LLMRockPaperScissors()
    # # Initial population: rock, paper-or-scissors, fully random
    # # initial_population = [rock_prompt, paper_scissors_prompt]
    # # final_population, payoff_matrix = train_population_from_last_agent(example_population, run_self_play, game, n_iters=4, n_games_per_empirical_payoff=5)
    # final_population, payoff_matrix = train_population_from_last_agent(example_population, run_PSRO_uniform_weaker_from_transcripts, game, n_iters=4, n_games_per_empirical_payoff=5)

    # A = compute_empirical_payoff_matrix_serial(example_population, game, n_games=10)
    # print(A)
    # A = compute_empirical_payoff_matrix_parallel(example_population, game, n_games=10)
    # print(A)


    print("Final population strategy prompts:")
    for idx, agent in enumerate(final_population):
        print(f"\nAgent {idx + 1}:\n{'-'*40}\n{agent}\n{'-'*40}")


    # final_population = [rps_to_disc(empirical_rps_distribution(u, n_games=10)) for u in final_population]
    # Plot the final population using plot_image
    # img = plot_image(final_population)
    # img.show()

