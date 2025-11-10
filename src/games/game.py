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

def run_self_play(agent_idx: int, population: list, game: Game):
    """
    Simply play agent_idx against itself
    """
    return game.improve(population[agent_idx], population[agent_idx])


def run_PSRO_uniform(agent_idx: int, population: list, game: Game):
    """
    This algorithm samples uniformly from the agents
    when deciding how to improve an agent.
    """
    rand_idx = np.random.randint(len(population))
    return game.improve(population[agent_idx], population[rand_idx])

def run_PSRO_uniform_weaker(agent_idx: int, population: list, game: Game):
    """
    This algorithm samples uniformly from the weaker agents
    when deciding how to improve an agent.
    """
    # Have agent_idx play everyone else, save indices of those agent_idx beats
    weaker_indices = [i for i in range(len(population)) if i != agent_idx and game.play(population[agent_idx], population[i]) > 0]

    # sample from weaker opponents or fall back to self-play
    rand_weaker_idx = np.random.choice(weaker_indices) if weaker_indices else agent_idx

    return game.improve(population[agent_idx], population[rand_weaker_idx])

def run_PSRO_uniform_stronger(agent_idx: int, population: list, game: Game):
    """
    This algorithm samples uniformly from the stronger agents
    when deciding how to improve an agent.
    """
    # Have agent_idx play everyone else, save indices of those agent_idx beats
    stronger_indices = [i for i in range(len(population)) if i != agent_idx and game.play(population[agent_idx], population[i]) < 0]

    # sample from defeated opponents or fall back to self-play
    rand_stronger_idx = np.random.choice(stronger_indices) if stronger_indices else agent_idx

    return game.improve(population[agent_idx], population[rand_stronger_idx])
