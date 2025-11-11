"""
Game-specific runner functions for Streamlit app.
These functions wrap the game demos to work with Streamlit's execution model.
"""
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from games.disc.disc_game import DiscGame, get_RPS_triangle, demo_disc_game
from games.disc.disc_game_vis import gif_from_population
from games.blotto.blotto import BlottoGame, LogitAgent
from games.blotto.differentiable_lotto import DifferentiableLotto
from games.blotto.differentiable_lotto_vis import gif_from_matchups
from games.game import run_PSRO_uniform_weaker, run_PSRO_uniform_stronger, run_PSRO_uniform


def run_disc_game_demo(
    improvement_type: str = "weaker",
    num_iterations: int = 500,
    learning_rate: float = 0.01,
    fps: int = 20,
    dpi: int = 120
) -> Dict[str, Any]:
    """
    Run Disc Game demo.
    
    Args:
        improvement_type: "weaker" or "stronger"
        num_iterations: Number of training iterations
        learning_rate: Learning rate for improvement
        fps: Frames per second for GIF
        dpi: DPI for visualization
    
    Returns:
        Dictionary with results
    """
    os.makedirs("demos/disc", exist_ok=True)
    
    improvement_func = run_PSRO_uniform_weaker if improvement_type == "weaker" else run_PSRO_uniform_stronger
    gif_name = f"demo_PSRO_u_{improvement_type}"
    
    game = DiscGame()
    rock, paper, scissors = [agent.copy() for agent in get_RPS_triangle()]
    
    # Store initial states
    initial_rock = rock.copy()
    initial_paper = paper.copy()
    initial_scissors = scissors.copy()
    
    agents_history = [[rock.copy(), paper.copy(), scissors.copy()]]
    
    # Run simulation
    for iteration in range(num_iterations):
        population = [rock, paper, scissors]
        rock = improvement_func(0, population, game)
        paper = improvement_func(1, population, game)
        scissors = improvement_func(2, population, game)
        agents_history.append([rock.copy(), paper.copy(), scissors.copy()])
    
    # Create GIF
    gif_path = gif_from_population(
        np.array(agents_history),
        path=f"demos/disc/{gif_name}.gif",
        fps=fps,
        stride=1,
        dpi=dpi,
        unit_circle=True,
        normalize_difference_vector=0.3
    )
    
    return {
        "gif_path": gif_path,
        "initial_states": {
            "rock": initial_rock.tolist(),
            "paper": initial_paper.tolist(),
            "scissors": initial_scissors.tolist()
        },
        "final_states": {
            "rock": agents_history[-1][0].tolist(),
            "paper": agents_history[-1][1].tolist(),
            "scissors": agents_history[-1][2].tolist()
        },
        "num_iterations": num_iterations
    }


def run_blotto_game_demo(
    improvement_type: str = "uniform",
    num_iterations: int = 1000,
    n_rounds: int = 1000,
    n_battlefields: int = 3,
    budget: int = 10
) -> Dict[str, Any]:
    """
    Run Blotto Game demo with PSRO variants.
    
    Args:
        improvement_type: "uniform", "weaker", or "stronger"
        num_iterations: Number of training iterations
        n_rounds: Number of rounds per evaluation
        n_battlefields: Number of battlefields
        budget: Total budget
    
    Returns:
        Dictionary with results
    """
    os.makedirs("demos/blotto", exist_ok=True)
    
    improvement_funcs = {
        "uniform": run_PSRO_uniform,
        "weaker": run_PSRO_uniform_weaker,
        "stronger": run_PSRO_uniform_stronger
    }
    improvement_func = improvement_funcs.get(improvement_type, run_PSRO_uniform)
    plot_name = f"blotto_PSRO_{improvement_type}"
    
    import copy
    
    game = BlottoGame()
    # Create a population of 3 agents
    agent_1 = LogitAgent(n_battlefields, budget)
    agent_2 = LogitAgent(n_battlefields, budget)
    agent_3 = LogitAgent(n_battlefields, budget)
    
    # Track win rates for each agent pair
    values_12 = []
    values_13 = []
    values_23 = []
    
    # Track agent history for GIF generation
    agents_history = [[copy.deepcopy(agent_1), copy.deepcopy(agent_2), copy.deepcopy(agent_3)]]
    
    for i in range(num_iterations):
        population = [agent_1, agent_2, agent_3]
        
        # Improve each agent using the PSRO strategy
        agent_1 = improvement_func(0, population, game)
        agent_2 = improvement_func(1, population, game)
        agent_3 = improvement_func(2, population, game)
        
        # Store agent states for GIF
        agents_history.append([copy.deepcopy(agent_1), copy.deepcopy(agent_2), copy.deepcopy(agent_3)])
        
        # Evaluate win rates
        val_12 = game.play(agent_1, agent_2, n_rounds=n_rounds)
        val_13 = game.play(agent_1, agent_3, n_rounds=n_rounds)
        val_23 = game.play(agent_2, agent_3, n_rounds=n_rounds)
        
        values_12.append(val_12)
        values_13.append(val_13)
        values_23.append(val_23)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(values_12, label='Agent 1 vs Agent 2', alpha=0.7)
    plt.plot(values_13, label='Agent 1 vs Agent 3', alpha=0.7)
    plt.plot(values_23, label='Agent 2 vs Agent 3', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate')
    plt.title(f'Blotto Game: Win Rate Over Time ({improvement_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"demos/blotto/{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Generate GIF visualizations
    gif_path_pop = None
    gif_path_match = None
    try:
        from games.blotto.blotto_vis import gif_from_population, gif_from_matchups
        
        # GIF showing expected allocations and entropy
        gif_path_pop = gif_from_population(
            agents_history,
            path=f"demos/blotto/{plot_name}_population.gif",
            fps=20,
            stride=max(1, num_iterations // 200),  # Limit to ~200 frames
            dpi=120,
            show_entropy=True
        )
        
        # GIF showing win rates over time
        gif_path_match = gif_from_matchups(
            game,
            agents_history,
            path=f"demos/blotto/{plot_name}_matchups.gif",
            fps=20,
            stride=max(1, num_iterations // 200),
            dpi=120,
            n_rounds=500  # Use fewer rounds for faster computation
        )
    except Exception as e:
        print(f"Could not generate GIFs: {e}")
    
    return {
        "plot_path": plot_path,
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "values_12": values_12,
        "values_13": values_13,
        "values_23": values_23,
        "final_values": {
            "agent_1_vs_2": values_12[-1] if values_12 else 0.0,
            "agent_1_vs_3": values_13[-1] if values_13 else 0.0,
            "agent_2_vs_3": values_23[-1] if values_23 else 0.0
        },
        "num_iterations": num_iterations
    }


def run_differentiable_lotto_demo(
    improvement_type: str = "weaker",
    num_iterations: int = 100,
    num_customers: int = 9,
    num_servers: int = 3,
    optimize_server_positions: bool = True,
    enforce_width_constraint: bool = True,
    width_penalty_lambda: float = 1.0,
    fps: int = 20,
    dpi: int = 120
) -> Dict[str, Any]:
    """
    Run Differentiable Lotto demo.
    
    Args:
        improvement_type: "weaker", "stronger", or "uniform"
        num_iterations: Number of training iterations
        num_customers: Number of customers
        num_servers: Number of servers per agent
        optimize_server_positions: Whether to optimize server positions
        enforce_width_constraint: Whether to enforce width constraint
        width_penalty_lambda: Penalty coefficient for width constraint
        fps: Frames per second for GIF
        dpi: DPI for visualization
    
    Returns:
        Dictionary with results
    """
    os.makedirs("demos/blotto", exist_ok=True)
    
    improvement_funcs = {
        "weaker": run_PSRO_uniform_weaker,
        "stronger": run_PSRO_uniform_stronger,
        "uniform": run_PSRO_uniform
    }
    improvement_func = improvement_funcs.get(improvement_type, run_PSRO_uniform_weaker)
    gif_name = f"demo_PSRO_u_{improvement_type}"
    
    game = DifferentiableLotto(
        num_customers=num_customers,
        num_servers=num_servers,
        customer_scale=1.0,
        seed=42,
        optimize_server_positions=optimize_server_positions,
        enforce_width_constraint=enforce_width_constraint,
        width_penalty_lambda=width_penalty_lambda
    )
    
    agent1 = game.create_random_agent()
    agent2 = game.create_random_agent()
    agent3 = game.create_random_agent()
    
    initial_payoffs = [
        game.play(agent1, agent2),
        game.play(agent1, agent3),
        game.play(agent2, agent3)
    ]
    
    agents_history = [[agent1, agent2, agent3]]
    for _ in range(num_iterations):
        population = [agent1, agent2, agent3]
        agent1 = improvement_func(0, population, game)
        agent2 = improvement_func(1, population, game)
        agent3 = improvement_func(2, population, game)
        agents_history.append([agent1, agent2, agent3])
    
    final_payoffs = [
        game.play(agent1, agent2),
        game.play(agent1, agent3),
        game.play(agent2, agent3)
    ]
    
    # Create visualization
    gif_path = None
    try:
        gif_path = gif_from_matchups(
            game, agents_history,
            path=f"demos/blotto/{gif_name}.gif",
            fps=fps, stride=1, dpi=dpi,
            show_customers=True,
            show_gradients=True,
            gradient_scale=0.3
        )
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    return {
        "gif_path": gif_path,
        "initial_payoffs": initial_payoffs,
        "final_payoffs": final_payoffs,
        "initial_widths": [
            game._compute_width(agent1[0], agent1[1]),
            game._compute_width(agent2[0], agent2[1]),
            game._compute_width(agent3[0], agent3[1])
        ],
        "final_widths": [
            game._compute_width(agents_history[-1][0][0], agents_history[-1][0][1]),
            game._compute_width(agents_history[-1][1][0], agents_history[-1][1][1]),
            game._compute_width(agents_history[-1][2][0], agents_history[-1][2][1])
        ],
        "num_iterations": num_iterations
    }

