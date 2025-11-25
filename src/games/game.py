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


def run_self_play(agent_idx: int, population: list, game: Game, payoff_matrix: np.ndarray, **kwargs):
    """
    Simply play agent_idx against itself
    """
    return game.improve(population[agent_idx], population[agent_idx], **kwargs)


def run_PSRO_uniform(agent_idx: int, population: list, game: Game, payoff_matrix: np.ndarray, **kwargs):
    """
    This algorithm samples opponents uniformly from all agents
    """
    rand_idx = np.random.randint(len(population))
    return game.improve(population[agent_idx], population[rand_idx], **kwargs)

def run_PSRO_uniform_from_transcripts(agent_idx: int, population:list, game: Game, payoff_matrix, n_games: int, *kwargs):
    """
    This algorithm samples opponents uniformly from all agents and plays n_games before updating
    """
    transcripts = []
    for _ in trange(n_games):
        rand_idx = np.random.randint(len(population))
        u = population[agent_idx]
        v = population[rand_idx]
        transcripts.append(game.play(u, v, return_transcript=True))
    return game.improve_from_transcripts(population[agent_idx], transcripts)

def run_PSRO_uniform_weaker(agent_idx: int, population: list, game: Game, payoff_matrix, **kwargs):
    """
    This algorithm samples opponents uniformly from the weaker agents
    """
    # Weaker = population[i] that agent_idx has positive payoff against
    weaker_indices = [i for i in range(len(population)) if i != agent_idx and payoff_matrix[agent_idx, i] > 0]

    rand_weaker_idx = np.random.choice(weaker_indices) if weaker_indices else agent_idx

    return game.improve(population[agent_idx], population[rand_weaker_idx], **kwargs)

def run_PSRO_uniform_stronger(agent_idx: int, population: list, game: Game, payoff_matrix, **kwargs):
    """
    This algorithm samples opponents uniformly from the stronger agents
    """
    # Stronger = population[i] that agent_idx has negative payoff against
    stronger_indices = [i for i in range(len(population)) if i != agent_idx and payoff_matrix[agent_idx, i] < 0]

    rand_stronger_idx = np.random.choice(stronger_indices) if stronger_indices else agent_idx

    return game.improve(population[agent_idx], population[rand_stronger_idx], **kwargs)

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

    return population


if __name__ == "__main__":


    from games.disc.disc_game import DiscGame, get_RPS_triangle
    from games.disc.disc_game_vis import plot_image

    # game = DiscGame()
    # initial_population = get_RPS_triangle()
    # # final_population = train_population_from_last_agent(initial_population, run_self_play, game, n_iters=4, n_games_per_empirical_payoff=3)
    # final_population = train_population_from_last_agent(initial_population, run_PSRO_uniform_stronger, game, n_iters=8, n_games_per_empirical_payoff=4, learning_rate=.1)


    from games.llms.llm2 import LLMRockPaperScissors, rock_prompt, paper_scissors_prompt, example_population 
    from games.disc.disc_game import rps_to_disc

    game = LLMRockPaperScissors()
    # Initial population: rock, paper-or-scissors, fully random
    # initial_population = [rock_prompt, paper_scissors_prompt]
    # final_population = train_population_from_last_agent(example_population, run_self_play, game, n_iters=4, n_games_per_empirical_payoff=5)
    final_population = train_population_from_last_agent(example_population, run_PSRO_uniform_from_transcripts, game, n_iters=4, n_games_per_empirical_payoff=5)

    # A = compute_empirical_payoff_matrix_serial(example_population, game, n_games=10)
    # print(A)
    # A = compute_empirical_payoff_matrix_parallel(example_population, game, n_games=10)
    # print(A)


    print("Final population strategy prompts:")
    for idx, agent in enumerate(final_population):
        print(f"\nAgent {idx + 1}:\n{'-'*40}\n{agent}\n{'-'*40}")


    # final_population = [rps_to_disc(empirical_rps_distribution(u, n_games=10)) for u in final_population]
    # # Plot the final population using plot_image
    # img = plot_image(final_population)
    # img.show()

