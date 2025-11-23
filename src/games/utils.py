import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
from typing import List, Optional, Tuple
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
import concurrent.futures


# COMMENTED OUT GAME CLASS FOR REFERENCE
# THE REAL ONE IS DEFINED in src/games/game.py

# class "Game"(ABC):

#     def play(self, u, v) -> float:
#         pass

#     def improve(self, u, v, **kwargs):
#         pass

# ------ EMPIRICAL PAYOFF MATRIX COMPUTATION HELPERS -----------

def _compute_pairs_to_evaluate(agent_indices: list, n: int, antisymmetric: bool) -> list:
    """
    Helper function to determine which pairs of agents need to be evaluated.
    
    Args:
        agent_indices: List of agent indices to compute payoffs for.
        n: Total number of agents in the population.
        antisymmetric: If True, treats the game as antisymmetric (only compute i < j).
    
    Returns:
        List of (i, j) tuples representing pairs to evaluate.
    """
    if antisymmetric:
        pairs = []
        for i in range(n):
            for j in agent_indices:
                if i < j:
                    pairs.append((i, j))
        for i in agent_indices:
            for j in range(i + 1, n):
                pairs.append((i, j))
        pairs = list(set(pairs))
    else:
        pairs = []
        for i in range(n):
            for j in agent_indices:
                if i != j:
                    pairs.append((i, j))
    return pairs


def _compute_payoffs_serial(
    pairs: list,
    population: list,
    game: "Game",
    antisymmetric: bool,
    n_games: int,
    payoff_matrix: np.ndarray,
    pbar: Optional[tqdm] = None
) -> np.ndarray:
    """
    Helper function to compute payoffs serially for specified pairs.
    
    Args:
        pairs: List of (i, j) tuples representing agent pairs to evaluate.
        population: List of agents.
        game: "Game" instance with .play(u, v)
        antisymmetric: If True, treats the game as antisymmetric.
        n_games: Number of times to play each matchup to compute average payoff.
        payoff_matrix: The payoff matrix to update (modified in place).
        pbar: Optional progress bar to update.
    
    Returns:
        The updated payoff_matrix.
    """
    # Compute payoffs serially
    for i, j in pairs:
        if i == j:
            avg_payoff = 0.0
        else:
            sum_payoff = 0.0
            for _ in range(n_games):
                sum_payoff += game.play(population[i], population[j])
            avg_payoff = sum_payoff / n_games
        
        payoff_matrix[i, j] = avg_payoff
        if antisymmetric and i != j:
            payoff_matrix[j, i] = -avg_payoff
        
        if pbar is not None:
            pbar.update(1)
    
    return payoff_matrix


def _compute_payoffs_parallel(
    pairs: list,
    population: list,
    game: "Game",
    antisymmetric: bool,
    n_games: int,
    payoff_matrix: np.ndarray,
    pbar: Optional[tqdm] = None
) -> np.ndarray:
    """
    Helper function to compute payoffs in parallel for specified pairs.
    
    Args:
        pairs: List of (i, j) tuples representing agent pairs to evaluate.
        population: List of agents.
        game: "Game" instance with .play(u, v)
        antisymmetric: If True, treats the game as antisymmetric.
        n_games: Number of times to play each matchup to compute average payoff.
        payoff_matrix: The payoff matrix to update (modified in place).
        pbar: Optional progress bar to update.
    
    Returns:
        The updated payoff_matrix.
    """
    def play_pair(pair):
        i, j = pair
        if i == j:
            return (i, j, 0.0)
        sum_payoff = 0.0
        for _ in range(n_games):
            sum_payoff += game.play(population[i], population[j])
        avg_payoff = sum_payoff / n_games
        return (i, j, avg_payoff)
    
    # Compute payoffs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(play_pair, pair) for pair in pairs]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if pbar is not None:
                pbar.update(1)
    
    # Update payoff matrix with results
    for i, j, avg_payoff in results:
        payoff_matrix[i, j] = avg_payoff
        if antisymmetric and i != j:
            payoff_matrix[j, i] = -avg_payoff
    
    return payoff_matrix
