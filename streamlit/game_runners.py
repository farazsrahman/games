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
from games.differentiable_lotto.differentiable_lotto import DifferentiableLotto
from games.differentiable_lotto.differentiable_lotto_vis import gif_from_matchups
from games.penneys.penneys import PennysGame, PennysAgent, demo_penneys_game
from games.penneys.penneys_vis import gif_from_population as penneys_gif_from_population, gif_from_matchups as penneys_gif_from_matchups
from games.game import run_PSRO_uniform_weaker, run_PSRO_uniform_stronger, run_PSRO_uniform, create_population


def run_disc_game_demo(
    improvement_type: str = "uniform",
    num_iterations: int = 500,
    learning_rate: float = 0.01,
    num_agents: int = 3,
    fps: int = 20,
    dpi: int = 120
) -> Dict[str, Any]:
    """
    Run Disc Game demo with PSRO variants and multiple agents.
    
    Args:
        improvement_type: "uniform", "weaker", or "stronger"
        num_iterations: Number of training iterations
        learning_rate: Learning rate for improvement
        num_agents: Number of agents in the population
        fps: Frames per second for GIF
        dpi: DPI for visualization
    
    Returns:
        Dictionary with results, including GIF, training plot, and EGS visualizations.
    """
    os.makedirs("demos/disc", exist_ok=True)
    
    improvement_funcs = {
        "uniform": run_PSRO_uniform,
        "weaker": run_PSRO_uniform_weaker,
        "stronger": run_PSRO_uniform_stronger
    }
    improvement_func = improvement_funcs.get(improvement_type, run_PSRO_uniform_weaker)
    plot_name = f"disc_PSRO_{improvement_type}"
    
    game = DiscGame()
    
    # Initialize population: for N=3, use RPS triangle; otherwise sample random points on the unit disc
    if num_agents == 3:
        population = [agent.copy() for agent in get_RPS_triangle()]
    else:
        rng = np.random.RandomState(42)
        population = []
        for _ in range(num_agents):
            # Sample from standard normal and project to unit disc
            v = rng.normal(size=2)
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                v = v / norm
            population.append(v)
    
    import copy
    
    # Track win rates for all agent pairs (values in [0, 1])
    win_rate_history: Dict[Tuple[int, int], list] = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            win_rate_history[(i, j)] = []
    
    # Track agent history for GIF generation
    agents_history = [[copy.deepcopy(agent) for agent in population]]
    
    # Initial payoffs (raw)
    initial_payoffs: Dict[Tuple[int, int], float] = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            payoff = game.play(population[i], population[j])
            initial_payoffs[(i, j)] = payoff
    
    for _ in range(num_iterations):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            new_agent = improvement_func(agent_idx, population, game, learning_rate=learning_rate)
            new_population.append(new_agent)
        population = new_population
        
        # Store agent states for GIF
        agents_history.append([copy.deepcopy(agent) for agent in population])
        
        # Evaluate "win rates" for all pairs (map payoffs -1/1 to [0, 1])
        for i_idx in range(num_agents):
            for j_idx in range(i_idx + 1, num_agents):
                payoff = game.play(population[i_idx], population[j_idx])
                win_rate = (payoff + 1.0) / 2.0
                win_rate_history[(i_idx, j_idx)].append(win_rate)
    
    # Final payoffs (raw)
    final_payoffs: Dict[Tuple[int, int], float] = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            payoff = game.play(population[i], population[j])
            final_payoffs[(i, j)] = payoff
    
    # Create training plot
    plt.figure(figsize=(12, 6))
    for (i, j), values in win_rate_history.items():
        plt.plot(values, label=f'Agent {i+1} vs Agent {j+1}', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (mapped from payoff)')
    plt.title(f'Disc Game: Win Rate Over Time ({improvement_type}, {num_agents} agents)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"demos/disc/{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Create population GIF
    gif_path_pop = None
    try:
        gif_path_pop = gif_from_population(
            np.array(agents_history),
            path=f"demos/disc/{plot_name}_population.gif",
            fps=fps,
            stride=max(1, num_iterations // 200),
            dpi=dpi,
            unit_circle=True,
            normalize_difference_vector=0.3
        )
    except Exception as e:
        print(f"Could not generate population GIF: {e}")
    
    # Generate EGS visualizations
    egs_visualization_paths: Dict[str, str] = {}
    try:
        from games.disc.disc_game_vis import plot_all_egs_visualizations
        
        egs_visualization_paths = plot_all_egs_visualizations(
            game,
            population,
            output_dir="demos/disc",
            base_name=f"{plot_name}_egs",
            n_rounds=1000,
            dpi=150
        )
        if egs_visualization_paths:
            print(f"Generated {len(egs_visualization_paths)} Disc EGS visualizations: {list(egs_visualization_paths.keys())}")
        else:
            print("Warning: No Disc EGS visualizations were generated")
    except Exception as e:
        import traceback
        print(f"Could not generate Disc EGS visualization plots: {e}")
        traceback.print_exc()
    
    # Build final_values dict for backward compatibility (use last win rate)
    final_values = {}
    for (i, j), values in win_rate_history.items():
        if values:
            final_values[f'agent_{i+1}_vs_{j+1}'] = values[-1]
    
    return {
        "plot_path": plot_path,
        "gif_path_population": gif_path_pop,
        "egs_visualization_paths": egs_visualization_paths,
        "win_rate_history": win_rate_history,
        "final_values": final_values,
        "initial_payoffs": initial_payoffs,
        "final_payoffs": final_payoffs,
        "num_iterations": num_iterations,
        "num_agents": num_agents,
    }


def run_blotto_game_demo(
    improvement_type: str = "uniform",
    num_iterations: int = 1000,
    n_rounds: int = 1000,
    n_battlefields: int = 3,
    budget: int = 10,
    num_agents: int = 3
) -> Dict[str, Any]:
    """
    Run Blotto Game demo with PSRO variants.
    
    Args:
        improvement_type: "uniform", "weaker", or "stronger"
        num_iterations: Number of training iterations
        n_rounds: Number of rounds per evaluation
        n_battlefields: Number of battlefields
        budget: Total budget
        num_agents: Number of agents in the population (default: 3)
    
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
    
    # Create population of N agents using create_population helper
    def agent_factory(**kwargs):
        return LogitAgent(n_battlefields=kwargs.get('n_battlefields', n_battlefields),
                         budget=kwargs.get('budget', budget))
    
    population = create_population(
        game=game,
        num_agents=num_agents,
        seed=42,
        agent_factory=agent_factory,
        n_battlefields=n_battlefields,
        budget=budget
    )
    
    # Track win rates for all agent pairs
    # Store as dict: {(i, j): [values over time]}
    win_rate_history = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            win_rate_history[(i, j)] = []
    
    # Track agent history for GIF generation
    agents_history = [[copy.deepcopy(agent) for agent in population]]
    
    for i in range(num_iterations):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            new_agent = improvement_func(agent_idx, population, game)
            new_population.append(new_agent)
        population = new_population
        
        # Store agent states for GIF
        agents_history.append([copy.deepcopy(agent) for agent in population])
        
        # Evaluate win rates for all pairs
        for i_idx in range(num_agents):
            for j_idx in range(i_idx + 1, num_agents):
                val = game.play(population[i_idx], population[j_idx], n_rounds=n_rounds)
                win_rate_history[(i_idx, j_idx)].append(val)
    
    # Create plot with all pairs
    plt.figure(figsize=(12, 6))
    for (i, j), values in win_rate_history.items():
        plt.plot(values, label=f'Agent {i+1} vs Agent {j+1}', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate')
    plt.title(f'Blotto Game: Win Rate Over Time ({improvement_type}, {num_agents} agents)')
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
    
    # Generate gamescape matrix and 2D embeddings visualizations
    # Generate all EGS visualizations (matrix + PCA, Schur, SVD, t-SNE)
    egs_visualization_paths = {}
    try:
        from games.blotto.blotto_vis import plot_all_egs_visualizations
        
        # Generate all EGS visualizations
        egs_visualization_paths = plot_all_egs_visualizations(
            game,
            population,
            output_dir="demos/blotto",
            base_name=f"{plot_name}_egs",
            n_rounds=1000,
            dpi=150
        )
        if egs_visualization_paths:
            print(f"Generated {len(egs_visualization_paths)} EGS visualizations: {list(egs_visualization_paths.keys())}")
        else:
            print("Warning: No EGS visualizations were generated")
    except Exception as e:
        import traceback
        print(f"Could not generate EGS visualization plots: {e}")
        traceback.print_exc()
    
    # Build final_values dict for backward compatibility
    final_values = {}
    for (i, j), values in win_rate_history.items():
        if values:
            final_values[f'agent_{i+1}_vs_{j+1}'] = values[-1]
    
    return {
        "plot_path": plot_path,
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "egs_visualization_paths": egs_visualization_paths,  # Dict of all EGS visualizations
        "win_rate_history": win_rate_history,
        "final_values": final_values,
        "num_iterations": num_iterations,
        "num_agents": num_agents
    }


def run_differentiable_lotto_demo(
    improvement_type: str = "weaker",
    num_iterations: int = 100,
    num_customers: int = 9,
    num_servers: int = 3,
    num_agents: int = 3,
    optimize_server_positions: bool = True,
    enforce_width_constraint: bool = True,
    width_penalty_lambda: float = 1.0,
    fps: int = 20,
    dpi: int = 120,
    n_rounds: int = 1000
) -> Dict[str, Any]:
    """
    Run Differentiable Lotto demo.
    
    Args:
        improvement_type: "weaker", "stronger", or "uniform"
        num_iterations: Number of training iterations
        num_customers: Number of customers
        num_servers: Number of servers per agent
        num_agents: Number of agents in the population
        optimize_server_positions: Whether to optimize server positions
        enforce_width_constraint: Whether to enforce width constraint
        width_penalty_lambda: Penalty coefficient for width constraint
        fps: Frames per second for GIF
        dpi: DPI for visualization
        n_rounds: Number of rounds per evaluation (for win rate computation)
    
    Returns:
        Dictionary with results
    """
    os.makedirs("demos/differentiable_lotto", exist_ok=True)
    
    improvement_funcs = {
        "weaker": run_PSRO_uniform_weaker,
        "stronger": run_PSRO_uniform_stronger,
        "uniform": run_PSRO_uniform
    }
    improvement_func = improvement_funcs.get(improvement_type, run_PSRO_uniform_weaker)
    plot_name = f"diff_lotto_PSRO_{improvement_type}"
    
    game = DifferentiableLotto(
        num_customers=num_customers,
        num_servers=num_servers,
        customer_scale=1.0,
        seed=42,
        optimize_server_positions=optimize_server_positions,
        enforce_width_constraint=enforce_width_constraint,
        width_penalty_lambda=width_penalty_lambda
    )
    
    # Create population of N agents
    import copy
    population = [game.create_random_agent() for _ in range(num_agents)]
    
    # Track win rates for all agent pairs
    win_rate_history = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            win_rate_history[(i, j)] = []
    
    # Track agent history for GIF generation
    agents_history = [[copy.deepcopy(agent) for agent in population]]
    
    # Compute initial payoffs
    initial_payoffs = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            payoff = game.play(population[i], population[j])
            initial_payoffs[(i, j)] = payoff
    
    for i in range(num_iterations):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            new_agent = improvement_func(agent_idx, population, game)
            new_population.append(new_agent)
        population = new_population
        
        # Store agent states for GIF
        agents_history.append([copy.deepcopy(agent) for agent in population])
        
        # Evaluate payoffs for all pairs
        for i_idx in range(num_agents):
            for j_idx in range(i_idx + 1, num_agents):
                payoff = game.play(population[i_idx], population[j_idx])
                # Convert payoff to win rate (normalize to [0, 1])
                # For differentiable lotto, payoffs can be negative/positive
                # We'll use a simple normalization: sigmoid transformation
                win_rate = 1.0 / (1.0 + np.exp(-payoff / 2.0))
                win_rate_history[(i_idx, j_idx)].append(win_rate)
    
    # Compute final payoffs
    final_payoffs = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            payoff = game.play(population[i], population[j])
            final_payoffs[(i, j)] = payoff
    
    # Create training plot
    plt.figure(figsize=(12, 6))
    for (i, j), values in win_rate_history.items():
        plt.plot(values, label=f'Agent {i+1} vs Agent {j+1}', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate')
    plt.title(f'Differentiable Lotto: Win Rate Over Time ({improvement_type}, {num_agents} agents)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"demos/differentiable_lotto/{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Generate GIF visualizations
    gif_path_pop = None
    gif_path_match = None
    try:
        from games.differentiable_lotto.differentiable_lotto_vis import gif_from_population, gif_from_matchups
        
        # GIF showing population evolution
        gif_path_pop = gif_from_population(
            game,
            agents_history,
            path=f"demos/differentiable_lotto/{plot_name}_population.gif",
            fps=fps,
            stride=max(1, num_iterations // 200),  # Limit to ~200 frames
            dpi=dpi,
            show_customers=True,
            show_gradients=True,
            gradient_scale=0.3
        )
        
        # GIF showing matchups over time
        gif_path_match = gif_from_matchups(
            game,
            agents_history,
            path=f"demos/differentiable_lotto/{plot_name}_matchups.gif",
            fps=fps,
            stride=max(1, num_iterations // 200),
            dpi=dpi,
            show_customers=True,
            show_gradients=True,
            gradient_scale=0.3
        )
    except Exception as e:
        print(f"Could not generate GIFs: {e}")
    
    # Generate EGS visualizations (matrix + PCA, Schur, SVD, t-SNE)
    egs_visualization_paths = {}
    try:
        from games.differentiable_lotto.differentiable_lotto_vis import plot_all_egs_visualizations
        
        # Generate all EGS visualizations
        egs_visualization_paths = plot_all_egs_visualizations(
            game,
            population,
            output_dir=f"demos/differentiable_lotto",
            base_name=f"{plot_name}_egs",
            n_rounds=n_rounds,
            dpi=150
        )
        if egs_visualization_paths:
            print(f"Generated {len(egs_visualization_paths)} EGS visualizations: {list(egs_visualization_paths.keys())}")
        else:
            print("Warning: No EGS visualizations were generated")
    except Exception as e:
        import traceback
        print(f"Could not generate EGS visualization plots: {e}")
        traceback.print_exc()
    
    # Build final_values dict for backward compatibility
    final_values = {}
    for (i, j), values in win_rate_history.items():
        if values:
            final_values[f'agent_{i+1}_vs_{j+1}'] = values[-1]
    
    return {
        "plot_path": plot_path,
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "egs_visualization_paths": egs_visualization_paths,  # Dict of all EGS visualizations
        "win_rate_history": win_rate_history,
        "final_values": final_values,
        "initial_payoffs": initial_payoffs,
        "final_payoffs": final_payoffs,
        "initial_widths": [
            game._compute_width(agent[0], agent[1]) for agent in agents_history[0]
        ],
        "final_widths": [
            game._compute_width(agent[0], agent[1]) for agent in agents_history[-1]
        ],
        "num_iterations": num_iterations,
        "num_agents": num_agents
    }


def run_penneys_game_demo(
    improvement_type: str = "uniform",
    num_iterations: int = 500,
    sequence_length: int = 3,
    n_rounds: int = 500,
    learning_rate: float = 0.1,
    fps: int = 20,
    dpi: int = 120
) -> Dict[str, Any]:
    """
    Run Penney's Game demo.
    
    Args:
        improvement_type: "weaker", "stronger", or "uniform"
        num_iterations: Number of training iterations
        sequence_length: Length of H/T sequences (default 3)
        n_rounds: Number of rounds for evaluation
        learning_rate: Learning rate for improvement
        fps: Frames per second for GIF
        dpi: DPI for visualization
    
    Returns:
        Dictionary with results
    """
    os.makedirs("demos/penneys", exist_ok=True)
    
    improvement_funcs = {
        "weaker": run_PSRO_uniform_weaker,
        "stronger": run_PSRO_uniform_stronger,
        "uniform": run_PSRO_uniform
    }
    improvement_func = improvement_funcs.get(improvement_type, run_PSRO_uniform)
    plot_name = f"penneys_PSRO_{improvement_type}"
    
    game = PennysGame(sequence_length=sequence_length)
    
    # Create a population of 3 agents
    agent_1 = PennysAgent(sequence_length=sequence_length, seed=42)
    agent_2 = PennysAgent(sequence_length=sequence_length, seed=43)
    agent_3 = PennysAgent(sequence_length=sequence_length, seed=44)
    
    # Initialize with slightly different distributions
    agent_1.logits = np.random.RandomState(42).randn(2 ** sequence_length)
    agent_2.logits = np.random.RandomState(43).randn(2 ** sequence_length)
    agent_3.logits = np.random.RandomState(44).randn(2 ** sequence_length)
    
    # Track win rates
    values_12 = []
    values_13 = []
    values_23 = []
    
    # Track agent history
    agents_history = [[agent_1.copy(), agent_2.copy(), agent_3.copy()]]
    
    from tqdm import trange
    
    for i in trange(num_iterations, desc="Training iterations"):
        population = [agent_1, agent_2, agent_3]
        
        # Improve each agent
        agent_1 = improvement_func(0, population, game)
        agent_2 = improvement_func(1, population, game)
        agent_3 = improvement_func(2, population, game)
        
        # Store agent states
        agents_history.append([agent_1.copy(), agent_2.copy(), agent_3.copy()])
        
        # Evaluate win rates
        val_12 = game.play(agent_1, agent_2, n_rounds=n_rounds)
        val_13 = game.play(agent_1, agent_3, n_rounds=n_rounds)
        val_23 = game.play(agent_2, agent_3, n_rounds=n_rounds)
        
        values_12.append(val_12)
        values_13.append(val_13)
        values_23.append(val_23)
    
    # Create static plot
    plt.figure(figsize=(12, 6))
    plt.plot(values_12, label='Agent 1 vs Agent 2', alpha=0.7)
    plt.plot(values_13, label='Agent 1 vs Agent 3', alpha=0.7)
    plt.plot(values_23, label='Agent 2 vs Agent 3', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (Agent i)')
    plt.title(f"Penney's Game: Win Rate Over Time ({improvement_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = f"demos/penneys/{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Generate GIFs
    gif_path_pop = None
    gif_path_match = None
    try:
        gif_path_pop = penneys_gif_from_population(
            agents_history,
            path=f"demos/penneys/{plot_name}_population.gif",
            fps=fps,
            stride=max(1, num_iterations // 200),
            dpi=dpi,
            show_entropy=True
        )
        
        gif_path_match = penneys_gif_from_matchups(
            game,
            agents_history,
            path=f"demos/penneys/{plot_name}_matchups.gif",
            fps=fps,
            stride=max(1, num_iterations // 200),
            dpi=dpi,
            n_rounds=500
        )
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    return {
        "plot_path": plot_path,
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "final_values": {
            "agent_1_vs_2": values_12[-1],
            "agent_1_vs_3": values_13[-1],
            "agent_2_vs_3": values_23[-1]
        },
        "num_iterations": num_iterations
    }

