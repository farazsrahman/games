import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
from typing import List, Optional, Tuple
from tqdm import tqdm, trange
from abc import ABC, abstractmethod


class Game(ABC):

    def play(self, u, v) -> float:
        pass

    def improve(self, u, v, **kwargs):
        """
        Takes in agents u and v. Returns a policy u_new that improves on u
        when playing against v.
        """
        pass


def contract(v: np.ndarray):
    """Helper function to keep player inside unit sphere."""
    norm_v = np.linalg.norm(v)
    if norm_v >= 1:
        v = v / norm_v
    return v

def run_self_play(agent_idx: int, population: list, game: Game, **kwargs):
    """
    Simply play agent_idx against itself
    """
    return game.improve(population[agent_idx], population[agent_idx], **kwargs)


def run_PSRO_uniform(agent_idx: int, population: list, game: Game, **kwargs):
    """
    This algorithm samples uniformly from the agents
    when deciding how to improve an agent.
    """
    rand_idx = np.random.randint(len(population))
    return game.improve(population[agent_idx], population[rand_idx], **kwargs)

def run_PSRO_uniform_weaker(agent_idx: int, population: list, game: Game, **kwargs):
    """
    This algorithm samples uniformly from the weaker agents
    when deciding how to improve an agent.
    """
    # Have agent_idx play everyone else, save indices of those agent_idx beats
    weaker_indices = [i for i in range(len(population)) if i != agent_idx and game.play(population[agent_idx], population[i]) > 0]

    # sample from weaker opponents or fall back to self-play
    rand_weaker_idx = np.random.choice(weaker_indices) if weaker_indices else agent_idx

    return game.improve(population[agent_idx], population[rand_weaker_idx], **kwargs)

def run_PSRO_uniform_stronger(agent_idx: int, population: list, game: Game, **kwargs):
    """
    This algorithm samples uniformly from the stronger agents
    when deciding how to improve an agent.
    """
    # Have agent_idx play everyone else, save indices of those agent_idx beats
    stronger_indices = [i for i in range(len(population)) if i != agent_idx and game.play(population[agent_idx], population[i]) < 0]

    # sample from defeated opponents or fall back to self-play
    rand_stronger_idx = np.random.choice(stronger_indices) if stronger_indices else agent_idx

    return game.improve(population[agent_idx], population[rand_stronger_idx], **kwargs)

def train_population_from_last_agent(initial_population, update_rule, game: Game, n_iters: int, **kwargs):
    population = initial_population

    for _ in range(n_iters):
        new_agent = update_rule(
            agent_idx = -1, 
            population = population, 
            game = game,
            **kwargs
        )
        population.append(new_agent)

    return population

if __name__ == "__main__":


    from games.disc.disc_game import DiscGame, get_RPS_triangle
    from games.disc.disc_game_vis import plot_image

    # game = DiscGame()
    # initial_population = game.get_population()
    # # final_population = train_population_from_last_agent(initial_population, run_self_play, game, 10)
    # final_population = train_population_from_last_agent(initial_population, run_PSRO_uniform_stronger, game, 20, learning_rate=.1)


    from games.llms.llm2 import LLMRockPaperScissors, rock_prompt, paper_scissors_prompt, empirical_rps_distribution
    from games.disc.disc_game import rps_to_disc

    game = LLMRockPaperScissors()
    # Initial population: rock, paper-or-scissors, fully random
    initial_population = [rock_prompt, paper_scissors_prompt]
    # final_population = train_population_from_last_agent(initial_population, run_self_play, game, 10)
    final_population = train_population_from_last_agent(initial_population, run_PSRO_uniform_weaker, game, 10)

    # final_population = [rps_to_disc(empirical_rps_distribution(u, n_games=10)) for u in final_population]



    # # Plot the final population using plot_image
    # img = plot_image(final_population)
    # img.show()

