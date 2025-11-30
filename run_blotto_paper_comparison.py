#!/usr/bin/env python3
"""
Run Blotto game with paper-matching configuration (40 agents).

This script runs all three PSRO variants (uniform, weaker, stronger) with:
- 40 agents (as in the paper)
- 3 battlefields, budget 10
- 1000 iterations
- 1000 evaluation rounds

Outputs are saved to demos/blotto_paper_comparison/
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from games.blotto.blotto import BlottoGame, LogitAgent
from games.game import (
    run_PSRO_uniform,
    run_PSRO_uniform_weaker,
    run_PSRO_uniform_stronger,
    create_population
)
from games.blotto.blotto_vis import (
    gif_from_population,
    gif_from_matchups,
    plot_all_egs_visualizations
)


def run_blotto_paper_comparison(
    improvement_type: str = "uniform",
    num_iterations: int = 1000,
    n_rounds: int = 1000,
    n_battlefields: int = 3,
    budget: int = 10,
    num_agents: int = 40,
    output_dir: str = "demos/blotto_paper_comparison"
) -> dict:
    """
    Run Blotto Game with paper-matching configuration.
    
    Args:
        improvement_type: "uniform", "weaker", or "stronger"
        num_iterations: Number of training iterations
        n_rounds: Number of rounds per evaluation
        n_battlefields: Number of battlefields
        budget: Total budget
        num_agents: Number of agents (40 for paper-matching)
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    improvement_funcs = {
        "uniform": run_PSRO_uniform,
        "weaker": run_PSRO_uniform_weaker,
        "stronger": run_PSRO_uniform_stronger
    }
    improvement_func = improvement_funcs.get(improvement_type, run_PSRO_uniform)
    plot_name = f"blotto_PSRO_{improvement_type}"
    
    game = BlottoGame()
    
    # Create population of N agents using create_population helper
    def agent_factory(**kwargs):
        return LogitAgent(n_battlefields=kwargs.get('n_battlefields', n_battlefields),
                         budget=kwargs.get('budget', budget))
    
    print(f"Creating population of {num_agents} agents...")
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
    
    print(f"Running {num_iterations} iterations with {improvement_type} PSRO...")
    print(f"Total matchups per iteration: {num_agents * (num_agents - 1) // 2}")
    
    for i in tqdm(range(num_iterations), desc=f"Training ({improvement_type})"):
        # Improve each agent using the PSRO strategy
        new_population = []
        for agent_idx in range(num_agents):
            new_agent = improvement_func(agent_idx, population, game)
            new_population.append(new_agent)
        population = new_population
        
        # Store agent states for GIF (subsample to save memory)
        if i % max(1, num_iterations // 200) == 0 or i == num_iterations - 1:
            agents_history.append([copy.deepcopy(agent) for agent in population])
        
        # Evaluate win rates for all pairs (subsample for performance)
        if i % 10 == 0 or i == num_iterations - 1:  # Evaluate every 10 iterations
            for i_idx in range(num_agents):
                for j_idx in range(i_idx + 1, num_agents):
                    val = game.play(population[i_idx], population[j_idx], n_rounds=n_rounds)
                    win_rate_history[(i_idx, j_idx)].append(val)
    
    # Create plot with win rate evolution
    # For 40 agents, we have 780 pairs - too many to plot individually
    # Instead, plot statistics: mean, min, max win rates
    print("Generating win rate plot...")
    plt.figure(figsize=(14, 8))
    
    # Compute statistics across all pairs
    all_values = []
    for (i, j), values in win_rate_history.items():
        all_values.append(values)
    
    if all_values:
        # Pad all to same length
        max_len = max(len(v) for v in all_values)
        padded = []
        for v in all_values:
            padded.append(v + [v[-1]] * (max_len - len(v)) if len(v) < max_len else v)
        
        all_values_array = np.array(padded)
        mean_win_rate = np.mean(all_values_array, axis=0)
        std_win_rate = np.std(all_values_array, axis=0)
        min_win_rate = np.min(all_values_array, axis=0)
        max_win_rate = np.max(all_values_array, axis=0)
        
        iterations = np.arange(0, num_iterations + 1, 10)
        if len(iterations) > len(mean_win_rate):
            iterations = iterations[:len(mean_win_rate)]
        elif len(iterations) < len(mean_win_rate):
            iterations = np.linspace(0, num_iterations, len(mean_win_rate))
        
        plt.plot(iterations, mean_win_rate, label='Mean Win Rate', linewidth=2, color='blue')
        plt.fill_between(iterations, mean_win_rate - std_win_rate, mean_win_rate + std_win_rate,
                        alpha=0.3, color='blue', label='±1 Std Dev')
        plt.plot(iterations, min_win_rate, '--', label='Min Win Rate', alpha=0.7, color='red')
        plt.plot(iterations, max_win_rate, '--', label='Max Win Rate', alpha=0.7, color='green')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.title(f'Blotto Game: Win Rate Statistics Over Time\n({improvement_type}, {num_agents} agents, {n_battlefields} battlefields, budget {budget})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Tie')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{plot_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    # Generate GIF visualizations
    print("Generating GIF visualizations...")
    gif_path_pop = None
    gif_path_match = None
    try:
        # GIF showing expected allocations and entropy
        gif_path_pop = gif_from_population(
            agents_history,
            path=os.path.join(output_dir, f"{plot_name}_population.gif"),
            fps=20,
            stride=max(1, len(agents_history) // 200),  # Limit to ~200 frames
            dpi=120,
            show_entropy=True
        )
        print(f"Saved population GIF: {gif_path_pop}")
        
        # GIF showing win rates over time
        gif_path_match = gif_from_matchups(
            game,
            agents_history,
            path=os.path.join(output_dir, f"{plot_name}_matchups.gif"),
            fps=20,
            stride=max(1, len(agents_history) // 200),
            dpi=120,
            n_rounds=500  # Use fewer rounds for faster computation
        )
        print(f"Saved matchups GIF: {gif_path_match}")
    except Exception as e:
        print(f"Could not generate GIFs: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate all EGS visualizations (matrix + PCA, Schur, SVD, t-SNE)
    print("Generating EGS visualizations...")
    egs_visualization_paths = {}
    try:
        # Generate all EGS visualizations
        egs_visualization_paths = plot_all_egs_visualizations(
            game,
            population,
            output_dir=output_dir,
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
    
    # Build final_values dict
    final_values = {}
    for (i, j), values in win_rate_history.items():
        if values:
            final_values[f'agent_{i+1}_vs_{j+1}'] = values[-1]
    
    return {
        "plot_path": plot_path,
        "gif_path_population": gif_path_pop,
        "gif_path_matchups": gif_path_match,
        "egs_visualization_paths": egs_visualization_paths,
        "win_rate_history": win_rate_history,
        "final_values": final_values,
        "num_iterations": num_iterations,
        "num_agents": num_agents
    }


def main():
    """Run all three PSRO variants with paper-matching configuration."""
    print("=" * 80)
    print("BLOTTO GAME - PAPER COMPARISON CONFIGURATION")
    print("=" * 80)
    print("Configuration:")
    print("  - Number of Agents: 40")
    print("  - Battlefields: 3")
    print("  - Budget: 10")
    print("  - Iterations: 1000")
    print("  - Evaluation Rounds: 1000")
    print("=" * 80)
    print()
    
    output_dir = "demos/blotto_paper_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    variants = ["uniform", "weaker", "stronger"]
    results = {}
    
    for variant in variants:
        print(f"\n{'=' * 80}")
        print(f"Running {variant.upper()} variant...")
        print(f"{'=' * 80}\n")
        
        try:
            result = run_blotto_paper_comparison(
                improvement_type=variant,
                num_iterations=1000,
                n_rounds=1000,
                n_battlefields=3,
                budget=10,
                num_agents=40,
                output_dir=output_dir
            )
            results[variant] = result
            print(f"\n✓ {variant} variant completed successfully!")
        except Exception as e:
            print(f"\n✗ {variant} variant failed: {e}")
            import traceback
            traceback.print_exc()
            results[variant] = None
    
    print("\n" + "=" * 80)
    print("ALL VARIANTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    for variant in variants:
        if results.get(variant):
            print(f"\n  {variant}:")
            print(f"    - Plot: {results[variant]['plot_path']}")
            if results[variant].get('gif_path_population'):
                print(f"    - Population GIF: {results[variant]['gif_path_population']}")
            if results[variant].get('gif_path_matchups'):
                print(f"    - Matchups GIF: {results[variant]['gif_path_matchups']}")
            if results[variant].get('egs_visualization_paths'):
                print(f"    - EGS visualizations: {len(results[variant]['egs_visualization_paths'])} files")
                for method, path in results[variant]['egs_visualization_paths'].items():
                    print(f"      * {method}: {path}")


if __name__ == "__main__":
    main()

