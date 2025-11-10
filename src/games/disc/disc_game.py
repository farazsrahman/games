import numpy as np
import os
from tqdm import trange
from games.game import Game, contract, run_PSRO_uniform_weaker, run_PSRO_uniform_stronger
from games.disc.disc_game_vis import gif_from_population

"""
This file introduces DiscGame as discussed in Balduzzi et. al. (https://arxiv.org/pdf/1901.08106)
Running the demo with PSRO_uniform_stronger will result in all players converging to the same distribution.
Running the demo with PSRO_uniform_weaker will result in a diverse distribution.
"""

class DiscGame(Game):
    def __init__(self):
        self.A = np.array([[0, -1], [1, 0]])

    def play(self, u, v) -> float:
        return float(u.T @ self.A @ v > 0) * 2 - 1

    def improve(self, u, v, *, learning_rate: float = 0.1) -> float:
        descent_dir = self.A @ v
        u_new = u + learning_rate * descent_dir
        u_new = contract(u_new)
        return u_new


def get_RPS_triangle():
    """Initialize agents in Rock-Paper-Scissors triangle formation."""
    return [
        np.array([0.5, 0]),
        np.array([-np.sqrt(1)/4,  np.sqrt(3)/4]),
        np.array([-np.sqrt(1)/4, -np.sqrt(3)/4])
    ]


def demo_disc_game(improvement_function, gif_file_name="demo_PSRO_disc_game"):
    """Run a demo of the disc game with a given improvement function."""
    
    # Ensure demos/disc directory exists
    os.makedirs("demos/disc", exist_ok=True)

    game = DiscGame()
    learning_rate = 0.01
    num_iterations = 500

    rock, paper, scissors = [agent.copy() for agent in get_RPS_triangle()]
    print("=" * 70)
    print("Initial states:")
    print("=" * 70)
    print(f"  Rock:     {rock}")
    print(f"  Paper:    {paper}")
    print(f"  Scissors: {scissors}")

    agents_history = [[rock.copy(), paper.copy(), scissors.copy()]]

    for iteration in trange(num_iterations, desc="Iterations"):
        population  = [rock, paper, scissors] 
        rock        = improvement_function(0, population, game)
        paper       = improvement_function(1, population, game)
        scissors    = improvement_function(2, population, game)
        agents_history.append([rock.copy(), paper.copy(), scissors.copy()])

    # Create GIF
    gif_path = gif_from_population(
        np.array(agents_history),
        path="demos/disc/" + gif_file_name + ".gif",
        fps=20,
        stride=1,
        dpi=120,
        unit_circle=True,
        normalize_difference_vector = .3
    )
    print(f"\nSaved GIF: {gif_path}")

    # Print final states
    print("\n" + "=" * 70)
    print("Final states:")
    print("=" * 70)
    print(f"  Rock:     {rock}")
    print(f"  Paper:    {paper}")
    print(f"  Scissors: {scissors}")


if __name__ == "__main__":
    print("Running Disc Game Demo with PSRO_uniform_weaker...")
    demo_disc_game(run_PSRO_uniform_weaker, "demo_PSRO_u_weaker")
    
    print("\n" + "=" * 70)
    print("Running Disc Game Demo with PSRO_uniform_stronger...")
    print("=" * 70 + "\n")
    demo_disc_game(run_PSRO_uniform_stronger, "demo_PSRO_u_stronger")
    
    print("\n" + "=" * 70)
    print("All demos completed! Check demos/disc/ for the generated GIFs.")
    print("=" * 70)
