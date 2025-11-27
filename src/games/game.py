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

def train_population_from_last_agent(initial_population, update_rule, game: Game, *, n_iters: int, n_games_per_empirical_payoff: int, parallel: bool = True, **kwargs):
    import numpy as np
    population = list(initial_population)
    payoff_matrix = None

    for _ in range(n_iters):
        # Recompute payoff matrix only when the population has changed (agent added)
        payoff_matrix = compute_empirical_payoff_matrix(population, game, parallel=parallel, cached_payoff_matrix=payoff_matrix, n_games=n_games_per_empirical_payoff)
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

def train_population_from_random_agent(initial_population, update_rule, game: Game, *, n_iters: int, n_games_per_empirical_payoff: int, parallel: bool = True, **kwargs):
    import numpy as np
    population = list(initial_population)
    payoff_matrix = None

    for _ in range(n_iters):
        # Recompute payoff matrix only when the population has changed (agent added)
        payoff_matrix = compute_empirical_payoff_matrix(population, game, parallel=parallel, cached_payoff_matrix=payoff_matrix, n_games=n_games_per_empirical_payoff)
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

    from games.llms.multiturn_rps import LLMRockPaperScissors as MTRPS_LLMRockPaperScissors
    from games.llms.multiturn_rps import example_population as mt_example_population
    from games.egs import EmpiricalGS

    mt_game = MTRPS_LLMRockPaperScissors(n_games=7, inform_game_count = True)

    # Run with uniform_weaker_from_transcripts
    print("=" * 60)
    print("Running PSRO with uniform_weaker_from_transcripts")
    print("=" * 60)
    final_population_weaker, payoff_matrix_weaker = train_population_from_last_agent(
        mt_example_population,
        run_PSRO_uniform_weaker_from_transcripts,
        mt_game,
        n_iters=6,
        n_games_per_empirical_payoff=5,
        # parallel=False
    )

    # Run with uniform_stronger_from_transcripts
    print("\n" + "=" * 60)
    print("Running PSRO with uniform_stronger_from_transcripts")
    print("=" * 60)
    final_population_stronger, payoff_matrix_stronger = train_population_from_last_agent(
        mt_example_population,
        run_PSRO_uniform_stronger_from_transcripts,
        mt_game,
        n_iters=6,
        n_games_per_empirical_payoff=5
    )

    # Compute embedding convex hull areas
    print("\n" + "=" * 60)
    print("Computing embedding convex hull areas")
    print("=" * 60)
    
    # For weaker distribution
    egs_weaker = EmpiricalGS(payoff_matrix_weaker)
    embeddings_weaker = egs_weaker.schur_embeddings()
    hull_area_weaker = egs_weaker._embedding_convex_hull_area(embeddings_weaker)
    
    # For stronger distribution
    egs_stronger = EmpiricalGS(payoff_matrix_stronger)
    embeddings_stronger = egs_stronger.schur_embeddings()
    hull_area_stronger = egs_stronger._embedding_convex_hull_area(embeddings_stronger)
    
    # Print results
    print(f"\nConvex Hull Area (uniform_weaker_from_transcripts): {hull_area_weaker:.6f}")
    print(f"Convex Hull Area (uniform_stronger_from_transcripts): {hull_area_stronger:.6f}")
    print(f"\nRelative difference: {abs(hull_area_weaker - hull_area_stronger) / max(hull_area_weaker, hull_area_stronger, 1e-10) * 100:.2f}%")
    
    if hull_area_weaker > hull_area_stronger:
        print(f"uniform_weaker_from_transcripts has {hull_area_weaker / hull_area_stronger:.2f}x larger convex hull area")
    elif hull_area_stronger > hull_area_weaker:
        print(f"uniform_stronger_from_transcripts has {hull_area_stronger / hull_area_weaker:.2f}x larger convex hull area")
    else:
        print("Both methods have the same convex hull area")


    breakpoint()

