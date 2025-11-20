import numpy as np
import os
from tqdm import trange
from games.game import Game, contract, run_PSRO_uniform_weaker, run_PSRO_uniform_stronger, create_population
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
    
    def create_agent(self, seed: int = None) -> np.ndarray:
        """Create a random agent on the unit circle."""
        if seed is not None:
            np.random.seed(seed)
        # Sample uniformly on unit circle
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)])


def get_RPS_triangle():
    """Initialize agents in Rock-Paper-Scissors triangle formation."""
    return [
        np.array([0.5, 0]),
        np.array([-1/4,  np.sqrt(3)/4]),
        np.array([-1/4, -np.sqrt(3)/4])
    ]


def demo_disc_game(improvement_function, gif_file_name="demo_PSRO_disc_game", num_agents: int = 3):
    """Run a demo of the disc game with a given improvement function."""
    
    # Ensure demos/disc directory exists
    os.makedirs("demos/disc", exist_ok=True)

    game = DiscGame()
    learning_rate = 0.01
    num_iterations = 500

    # Use RPS triangle for 3 agents, otherwise create random agents
    if num_agents == 3:
        population = [agent.copy() for agent in get_RPS_triangle()]
        print("=" * 70)
        print("Initial states (RPS triangle):")
        print("=" * 70)
        print(f"  Agent 0: {population[0]}")
        print(f"  Agent 1: {population[1]}")
        print(f"  Agent 2: {population[2]}")
    else:
        # Create random agents on unit circle
        population = create_population(game, num_agents, seed=42)
        print("=" * 70)
        print(f"Initial states ({num_agents} agents):")
        print("=" * 70)
        for i, agent in enumerate(population[:5], 1):  # Show first 5
            print(f"  Agent {i-1}: {agent}")

    agents_history = [[agent.copy() for agent in population]]

    for iteration in trange(num_iterations, desc="Iterations"):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            improved_agent = improvement_function(agent_idx, population, game)
            new_population.append(improved_agent)
        population = new_population
        agents_history.append([agent.copy() for agent in population])

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
    for i, agent in enumerate(population[:5], 1):  # Show first 5
        print(f"  Agent {i-1}: {agent}")


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
