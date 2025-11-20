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
from games.penneys.penneys import PennysGame, PennysAgent, demo_penneys_game
from games.penneys.penneys_vis import gif_from_population as penneys_gif_from_population, gif_from_matchups as penneys_gif_from_matchups
from games.game import run_PSRO_uniform_weaker, run_PSRO_uniform_stronger, run_PSRO_uniform, create_population


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
        num_agents: Number of agents in the population
    
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
    # Create a population of N agents
    population = create_population(game, num_agents, seed=42, n_battlefields=n_battlefields, budget=budget)
    
    # Track win rates for all agent pairs
    win_rates = {f"{i}vs{j}": [] for i in range(num_agents) for j in range(i+1, num_agents)}
    
    # Track agent history for GIF generation
    agents_history = [[copy.deepcopy(agent) for agent in population]]
    
    # Track entropy over time for diversity plot
    entropies_history = []
    
    for i in range(num_iterations):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            improved_agent = improvement_func(agent_idx, population, game)
            new_population.append(improved_agent)
        population = new_population
        
        # Store agent states for GIF
        agents_history.append([copy.deepcopy(agent) for agent in population])
        
        # Track entropy for diversity plot
        try:
            from games.blotto.blotto_vis import get_entropy
            current_entropies = [get_entropy(agent) for agent in population]
            entropies_history.append(current_entropies)
        except:
            pass
        
        # Evaluate win rates for all pairs
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                val = game.play(population[i], population[j], n_rounds=n_rounds)
                win_rates[f"{i}vs{j}"].append(val)
    
    # Create win rate plot
    plt.figure(figsize=(12, 6))
    for pair_key, values in win_rates.items():
        i, j = map(int, pair_key.split('vs'))
        plt.plot(values, label=f'Agent {i} vs Agent {j}', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate')
    plt.title(f'Blotto Game: Win Rate Over Time ({improvement_type}, {num_agents} agents)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"demos/blotto/{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Create expected allocation per battlefield plot
    allocation_plot_path = None
    try:
        from games.blotto.blotto_vis import get_expected_allocation
        
        plt.figure(figsize=(12, 6))
        battlefields = ['Battlefield 1', 'Battlefield 2', 'Battlefield 3']
        x = np.arange(len(battlefields))
        width = 0.8 / num_agents
        agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
        
        # Get final expected allocations for all agents
        for i in range(num_agents):
            offset = (i - (num_agents - 1) / 2) * width
            expected = get_expected_allocation(population[i])
            bars = plt.bar(x + offset, expected, width, 
                          label=f'Agent {i}', color=agent_colors[i], alpha=0.8)
            
            # Add value labels on bars
            for j, val in enumerate(expected):
                plt.text(j + offset, val + 0.2, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Battlefield', fontsize=12)
        plt.ylabel('Expected Troops', fontsize=12)
        plt.title(f'Expected Allocation per Battlefield (Final, {improvement_type}, {num_agents} agents)', fontsize=14)
        plt.xticks(x, battlefields)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 10)
        plt.tight_layout()
        
        allocation_plot_path = f"demos/blotto/{plot_name}_allocations.png"
        plt.savefig(allocation_plot_path)
        plt.close()
    except Exception as e:
        print(f"Could not generate allocation plot: {e}")
        allocation_plot_path = None
    
    # Create policy diversity (entropy) over time plot
    entropy_plot_path = None
    if entropies_history:
        try:
            from games.blotto.blotto_vis import get_entropy
            
            entropies_array = np.array(entropies_history)  # Shape: (iterations, num_agents)
            iterations = np.arange(len(entropies_history))
            agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
            
            plt.figure(figsize=(12, 6))
            for i in range(num_agents):
                plt.plot(iterations, entropies_array[:, i], 
                        label=f'Agent {i}', color=agent_colors[i], linewidth=2, alpha=0.8)
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Policy Entropy', fontsize=12)
            plt.title(f'Policy Diversity Over Time ({improvement_type}, {num_agents} agents)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            entropy_plot_path = f"demos/blotto/{plot_name}_entropy.png"
            plt.savefig(entropy_plot_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate entropy plot: {e}")
            entropy_plot_path = None
    
    # Create payoff matrix (final win rates between all pairs)
    payoff_matrix_path = None
    try:
        # Build symmetric payoff matrix
        payoff_matrix = np.zeros((num_agents, num_agents))
        for pair_key, values in win_rates.items():
            i, j = map(int, pair_key.split('vs'))
            final_rate = values[-1] if values else 0.5
            payoff_matrix[i, j] = final_rate
            payoff_matrix[j, i] = 1.0 - final_rate  # Symmetric
        
        # Set diagonal to 0.5 (agents vs themselves)
        np.fill_diagonal(payoff_matrix, 0.5)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(payoff_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, label='Win Rate (Agent i)')
        
        # Add text annotations
        for i in range(num_agents):
            for j in range(num_agents):
                text = plt.text(j, i, f'{payoff_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.xlabel('Agent j', fontsize=12)
        plt.ylabel('Agent i', fontsize=12)
        plt.title(f'Final Payoff Matrix ({improvement_type}, {num_agents} agents)', fontsize=14)
        plt.xticks(range(num_agents), [f'Agent {i}' for i in range(num_agents)])
        plt.yticks(range(num_agents), [f'Agent {i}' for i in range(num_agents)])
        plt.tight_layout()
        
        payoff_matrix_path = f"demos/blotto/{plot_name}_payoff_matrix.png"
        plt.savefig(payoff_matrix_path)
        plt.close()
    except Exception as e:
        print(f"Could not generate payoff matrix: {e}")
        payoff_matrix_path = None
    
    # Generate Empirical Gamescape Matrix and 2D Embeddings
    gamescape_matrix_path = None
    embeddings_2d_path = None
    try:
        from games.blotto.blotto_vis import plot_gamescape_matrix, plot_2d_embeddings
        
        # Empirical Gamescape Matrix
        gamescape_matrix_path = plot_gamescape_matrix(
            game,
            population,
            path=f"demos/blotto/{plot_name}_gamescape_matrix.png",
            n_rounds=n_rounds,
            dpi=120
        )
        
        # 2D Embeddings
        embeddings_2d_path = plot_2d_embeddings(
            game,
            population,
            path=f"demos/blotto/{plot_name}_2d_embeddings.png",
            n_rounds=n_rounds,
            dpi=120,
            use_probabilities=True
        )
    except Exception as e:
        print(f"Could not generate gamescape/embeddings plots: {e}")
        gamescape_matrix_path = None
        embeddings_2d_path = None
    
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
    
    # Build final values dict for all pairs
    final_values = {}
    for pair_key, values in win_rates.items():
        i, j = map(int, pair_key.split('vs'))
        final_values[f"agent_{i}_vs_{j}"] = values[-1] if values else 0.0
    
    return {
        "plot_path": plot_path,  # Win rate over time
        "allocation_plot_path": allocation_plot_path,  # Expected allocation per battlefield (final)
        "entropy_plot_path": entropy_plot_path,  # Policy diversity over time
        "payoff_matrix_path": payoff_matrix_path,  # Final payoff matrix
        "gamescape_matrix_path": gamescape_matrix_path,  # Empirical Gamescape Matrix
        "embeddings_2d_path": embeddings_2d_path,  # 2D Embeddings
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "win_rates": win_rates,
        "final_values": final_values,
        "num_iterations": num_iterations,
        "num_agents": num_agents
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


def run_penneys_game_demo(
    improvement_type: str = "uniform",
    num_iterations: int = 500,
    sequence_length: int = 3,
    n_rounds: int = 500,
    learning_rate: float = 0.1,
    fps: int = 20,
    dpi: int = 120,
    num_agents: int = 3
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
        num_agents: Number of agents in the population
    
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
    
    import copy
    
    game = PennysGame(sequence_length=sequence_length)
    
    # Create a population of N agents
    population = create_population(game, num_agents, seed=42)
    
    # Track win rates for all agent pairs
    win_rates = {f"{i}vs{j}": [] for i in range(num_agents) for j in range(i+1, num_agents)}
    
    # Track agent history for GIF generation
    agents_history = [[copy.deepcopy(agent) for agent in population]]
    
    # Track entropy over time for diversity plot
    entropies_history = []
    
    for i in range(num_iterations):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            improved_agent = improvement_func(agent_idx, population, game)
            new_population.append(improved_agent)
        population = new_population
        
        # Store agent states for GIF
        agents_history.append([copy.deepcopy(agent) for agent in population])
        
        # Track entropy for diversity plot
        try:
            from games.penneys.penneys_vis import get_entropy
            current_entropies = [get_entropy(agent) for agent in population]
            entropies_history.append(current_entropies)
        except:
            pass
        
        # Evaluate win rates for all pairs
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                val = game.play(population[i], population[j], n_rounds=n_rounds)
                win_rates[f"{i}vs{j}"].append(val)
    
    # Create win rate plot
    plt.figure(figsize=(12, 6))
    for pair_key, values in win_rates.items():
        i, j = map(int, pair_key.split('vs'))
        plt.plot(values, label=f'Agent {i} vs Agent {j}', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (Agent i)')
    plt.title(f"Penney's Game: Win Rate Over Time ({improvement_type}, {num_agents} agents)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Tie')
    plt.tight_layout()
    
    plot_path = f"demos/penneys/{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Create sequence probability distribution plot (final state)
    sequence_plot_path = None
    try:
        from games.penneys.penneys_vis import get_entropy
        
        sequences = population[0].get_all_sequences()
        num_sequences = len(sequences)
        x = np.arange(num_sequences)
        width = 0.8 / num_agents
        agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
        
        plt.figure(figsize=(14, 6))
        for i in range(num_agents):
            offset = (i - (num_agents - 1) / 2) * width
            probs = population[i].get_probabilities()
            bars = plt.bar(x + offset, probs, width, 
                          label=f'Agent {i}', color=agent_colors[i], alpha=0.8)
            
            # Add value labels on bars
            for j, val in enumerate(probs):
                if val > 0.05:  # Only label if significant
                    plt.text(j + offset, val + 0.02, f'{val:.2f}', 
                            ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Sequence', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f"Probability Distribution Over Sequences (Final, {improvement_type}, {num_agents} agents)", fontsize=14)
        plt.xticks(x, sequences, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        sequence_plot_path = f"demos/penneys/{plot_name}_sequences.png"
        plt.savefig(sequence_plot_path)
        plt.close()
    except Exception as e:
        print(f"Could not generate sequence plot: {e}")
        sequence_plot_path = None
    
    # Create policy diversity (entropy) over time plot
    entropy_plot_path = None
    if entropies_history:
        try:
            from games.penneys.penneys_vis import get_entropy
            
            entropies_array = np.array(entropies_history)  # Shape: (iterations, num_agents)
            iterations = np.arange(len(entropies_history))
            agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
            
            plt.figure(figsize=(12, 6))
            for i in range(num_agents):
                plt.plot(iterations, entropies_array[:, i], 
                        label=f'Agent {i}', color=agent_colors[i], linewidth=2, alpha=0.8)
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Policy Entropy', fontsize=12)
            plt.title(f'Policy Diversity Over Time ({improvement_type}, {num_agents} agents)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            entropy_plot_path = f"demos/penneys/{plot_name}_entropy.png"
            plt.savefig(entropy_plot_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate entropy plot: {e}")
            entropy_plot_path = None
    
    # Create payoff matrix (final win rates between all pairs)
    payoff_matrix_path = None
    try:
        # Build symmetric payoff matrix
        payoff_matrix = np.zeros((num_agents, num_agents))
        for pair_key, values in win_rates.items():
            i, j = map(int, pair_key.split('vs'))
            final_rate = values[-1] if values else 0.0
            payoff_matrix[i, j] = final_rate
            payoff_matrix[j, i] = -final_rate  # Symmetric (Penney's is zero-sum)
        
        # Set diagonal to 0 (agents vs themselves)
        np.fill_diagonal(payoff_matrix, 0.0)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(payoff_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label='Win Rate (Agent i)')
        
        # Add text annotations
        for i in range(num_agents):
            for j in range(num_agents):
                text = plt.text(j, i, f'{payoff_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.xlabel('Agent j', fontsize=12)
        plt.ylabel('Agent i', fontsize=12)
        plt.title(f'Final Payoff Matrix ({improvement_type}, {num_agents} agents)', fontsize=14)
        plt.xticks(range(num_agents), [f'Agent {i}' for i in range(num_agents)])
        plt.yticks(range(num_agents), [f'Agent {i}' for i in range(num_agents)])
        plt.tight_layout()
        
        payoff_matrix_path = f"demos/penneys/{plot_name}_payoff_matrix.png"
        plt.savefig(payoff_matrix_path)
        plt.close()
    except Exception as e:
        print(f"Could not generate payoff matrix: {e}")
        payoff_matrix_path = None
    
    # Generate Empirical Gamescape Matrix and 2D Embeddings
    gamescape_matrix_path = None
    embeddings_2d_path = None
    try:
        from games.penneys.penneys_vis import plot_gamescape_matrix, plot_2d_embeddings
        
        # Empirical Gamescape Matrix
        gamescape_matrix_path = plot_gamescape_matrix(
            game,
            population,
            path=f"demos/penneys/{plot_name}_gamescape_matrix.png",
            n_rounds=n_rounds,
            dpi=120
        )
        
        # 2D Embeddings
        embeddings_2d_path = plot_2d_embeddings(
            game,
            population,
            path=f"demos/penneys/{plot_name}_2d_embeddings.png",
            n_rounds=n_rounds,
            dpi=120,
            use_probabilities=True
        )
    except Exception as e:
        print(f"Could not generate gamescape/embeddings plots: {e}")
        gamescape_matrix_path = None
        embeddings_2d_path = None
    
    # Generate GIF visualizations
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
        print(f"Could not generate GIFs: {e}")
    
    # Build final values dict for all pairs
    final_values = {}
    for pair_key, values in win_rates.items():
        i, j = map(int, pair_key.split('vs'))
        final_values[f"agent_{i}_vs_{j}"] = values[-1] if values else 0.0

    return {
        "plot_path": plot_path,  # Win rate over time
        "sequence_plot_path": sequence_plot_path,  # Sequence probability distribution (final)
        "entropy_plot_path": entropy_plot_path,  # Policy diversity over time
        "payoff_matrix_path": payoff_matrix_path,  # Final payoff matrix
        "gamescape_matrix_path": gamescape_matrix_path,  # Empirical Gamescape Matrix
        "embeddings_2d_path": embeddings_2d_path,  # 2D Embeddings
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "win_rates": win_rates,
        "final_values": final_values,
        "num_iterations": num_iterations,
        "num_agents": num_agents
    }

