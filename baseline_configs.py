#!/usr/bin/env python3
"""
Baseline configurations for running experiments on the games:
1. Colonel Blotto
2. Differentiable Lotto  
3. Disc Game
4. Penneys Game

These configurations are designed to be:
- Fast enough to run quickly for baseline comparisons
- Comprehensive enough to show meaningful results
- Standard enough to be reproducible

Usage:
    python baseline_configs.py
    # Or import and use in your own scripts
    from baseline_configs import BLOTTO_BASELINE, DIFF_LOTTO_BASELINE, DISC_BASELINE
"""

# ============================================================================
# COLONEL BLOTTO BASELINE CONFIGURATIONS
# ============================================================================

BLOTTO_BASELINE = {
    "name": "blotto_baseline",
    "improvement_type": "uniform",  # Options: "uniform", "weaker", "stronger"
    "num_iterations": 500,          # Reduced from 1000 for faster baseline
    "n_rounds": 1000,               # Evaluation rounds per iteration
    "n_battlefields": 3,            # Number of battlefields (action space size: comb(10+3-1, 3-1) = 66)
    "budget": 10,                    # Total budget to allocate
    "num_agents": 3,                 # Number of agents in population
}

BLOTTO_BASELINE_SMALL = {
    "name": "blotto_baseline_small",
    "improvement_type": "uniform",
    "num_iterations": 300,           # Even faster for quick tests
    "n_rounds": 500,                # Fewer evaluation rounds
    "n_battlefields": 3,
    "budget": 8,                     # Smaller budget = smaller action space (comb(8+3-1, 3-1) = 45)
    "num_agents": 3,
}

BLOTTO_BASELINE_LARGE = {
    "name": "blotto_baseline_large",
    "improvement_type": "uniform",
    "num_iterations": 1000,          # Full training
    "n_rounds": 2000,               # More evaluation rounds for stability
    "n_battlefields": 4,             # More battlefields (action space: comb(10+4-1, 4-1) = 286)
    "budget": 10,
    "num_agents": 3,
}

# All PSRO variants for comparison
BLOTTO_ALL_VARIANTS = [
    {**BLOTTO_BASELINE, "improvement_type": "uniform", "name": "blotto_uniform"},
    {**BLOTTO_BASELINE, "improvement_type": "weaker", "name": "blotto_weaker"},
    {**BLOTTO_BASELINE, "improvement_type": "stronger", "name": "blotto_stronger"},
]


# ============================================================================
# DIFFERENTIABLE LOTTO BASELINE CONFIGURATIONS
# ============================================================================

DIFF_LOTTO_BASELINE = {
    "name": "diff_lotto_baseline",
    "improvement_type": "weaker",    # Default for differentiable lotto
    "num_iterations": 100,           # Standard for continuous games
    "num_customers": 9,               # As in paper experiments
    "num_servers": 3,                 # Number of servers per agent
    "num_agents": 3,                  # Number of agents in population
    "optimize_server_positions": True, # Whether to optimize server positions
    "enforce_width_constraint": True,  # Enforce width constraint
    "width_penalty_lambda": 1.0,      # Penalty coefficient for width constraint
    "n_rounds": 1000,                 # Evaluation rounds (for win rate computation)
    "fps": 20,                        # GIF frames per second
    "dpi": 120,                       # GIF resolution
}

DIFF_LOTTO_BASELINE_SMALL = {
    "name": "diff_lotto_baseline_small",
    "improvement_type": "weaker",
    "num_iterations": 50,             # Faster for quick tests
    "num_customers": 6,                # Fewer customers
    "num_servers": 2,                  # Fewer servers
    "num_agents": 3,
    "optimize_server_positions": True,
    "enforce_width_constraint": True,
    "width_penalty_lambda": 1.0,
    "n_rounds": 500,
    "fps": 20,
    "dpi": 120,
}

DIFF_LOTTO_BASELINE_LARGE = {
    "name": "diff_lotto_baseline_large",
    "improvement_type": "weaker",
    "num_iterations": 200,            # More iterations for convergence
    "num_customers": 12,               # More customers
    "num_servers": 4,                  # More servers
    "num_agents": 3,
    "optimize_server_positions": True,
    "enforce_width_constraint": True,
    "width_penalty_lambda": 1.0,
    "n_rounds": 2000,
    "fps": 20,
    "dpi": 120,
}

# All PSRO variants for comparison
DIFF_LOTTO_ALL_VARIANTS = [
    {**DIFF_LOTTO_BASELINE, "improvement_type": "uniform", "name": "diff_lotto_uniform"},
    {**DIFF_LOTTO_BASELINE, "improvement_type": "weaker", "name": "diff_lotto_weaker"},
    {**DIFF_LOTTO_BASELINE, "improvement_type": "stronger", "name": "diff_lotto_stronger"},
]


# ============================================================================
# DISC GAME BASELINE CONFIGURATIONS
# ============================================================================

DISC_BASELINE = {
    "name": "disc_baseline",
    "improvement_type": "uniform",   # Options: "uniform", "weaker", "stronger"
    "num_iterations": 500,           # Standard for disc game
    "learning_rate": 0.01,            # Learning rate for improvement
    "num_agents": 3,                  # Number of agents (3 = RPS triangle initialization)
    "fps": 20,                        # GIF frames per second
    "dpi": 120,                       # GIF resolution
}

DISC_BASELINE_SMALL = {
    "name": "disc_baseline_small",
    "improvement_type": "uniform",
    "num_iterations": 200,            # Faster for quick tests
    "learning_rate": 0.01,
    "num_agents": 3,
    "fps": 20,
    "dpi": 120,
}

DISC_BASELINE_LARGE = {
    "name": "disc_baseline_large",
    "improvement_type": "uniform",
    "num_iterations": 1000,           # More iterations to see convergence
    "learning_rate": 0.01,
    "num_agents": 3,
    "fps": 20,
    "dpi": 120,
}

# All PSRO variants for comparison
DISC_ALL_VARIANTS = [
    {**DISC_BASELINE, "improvement_type": "uniform", "name": "disc_uniform"},
    {**DISC_BASELINE, "improvement_type": "weaker", "name": "disc_weaker"},
    {**DISC_BASELINE, "improvement_type": "stronger", "name": "disc_stronger"},
]


# ============================================================================
# PENNEYS GAME BASELINE CONFIGURATIONS
# ============================================================================

PENNEYS_BASELINE = {
    "name": "penneys_baseline",
    "improvement_type": "uniform",  # Options: "uniform", "weaker", "stronger"
    "num_iterations": 500,          # Standard for Penneys game
    "sequence_length": 3,           # Length of H/T sequences (2^3 = 8 possible sequences)
    "n_rounds": 500,                # Evaluation rounds per iteration
    "learning_rate": 0.1,          # Learning rate for improvement
    "num_agents": 3,                 # Number of agents in population (default: 3)
    "fps": 20,                      # GIF frames per second
    "dpi": 120,                     # GIF resolution
}

PENNEYS_BASELINE_SMALL = {
    "name": "penneys_baseline_small",
    "improvement_type": "uniform",
    "num_iterations": 300,           # Faster for quick tests
    "sequence_length": 3,
    "n_rounds": 300,
    "learning_rate": 0.1,
    "num_agents": 3,
    "fps": 20,
    "dpi": 120,
}

PENNEYS_BASELINE_LARGE = {
    "name": "penneys_baseline_large",
    "improvement_type": "uniform",
    "num_iterations": 1000,          # Full training
    "sequence_length": 3,
    "n_rounds": 1000,                # More evaluation rounds for stability
    "learning_rate": 0.1,
    "num_agents": 3,
    "fps": 20,
    "dpi": 120,
}

# All PSRO variants for comparison
PENNEYS_ALL_VARIANTS = [
    {**PENNEYS_BASELINE, "improvement_type": "uniform", "name": "penneys_uniform"},
    {**PENNEYS_BASELINE, "improvement_type": "weaker", "name": "penneys_weaker"},
    {**PENNEYS_BASELINE, "improvement_type": "stronger", "name": "penneys_stronger"},
]


# ============================================================================
# COMPLETE BASELINE SUITE
# ============================================================================

# Recommended baseline configurations for class project
BASELINE_SUITE = {
    "blotto": BLOTTO_BASELINE,
    "diff_lotto": DIFF_LOTTO_BASELINE,
    "disc": DISC_BASELINE,
    "penneys": PENNEYS_BASELINE,
}

# All variants for comprehensive comparison
COMPLETE_BASELINE_SUITE = {
    "blotto": BLOTTO_ALL_VARIANTS,
    "diff_lotto": DIFF_LOTTO_ALL_VARIANTS,
    "disc": DISC_ALL_VARIANTS,
    "penneys": PENNEYS_ALL_VARIANTS,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_baseline_configs():
    """Print all baseline configurations in a readable format."""
    print("=" * 80)
    print("BASELINE CONFIGURATIONS FOR CLASS PROJECT")
    print("=" * 80)
    
    print("\n1. COLONEL BLOTTO")
    print("-" * 80)
    print(f"  Baseline: {BLOTTO_BASELINE}")
    print(f"  Small:    {BLOTTO_BASELINE_SMALL}")
    print(f"  Large:    {BLOTTO_BASELINE_LARGE}")
    
    print("\n2. DIFFERENTIABLE LOTTO")
    print("-" * 80)
    print(f"  Baseline: {DIFF_LOTTO_BASELINE}")
    print(f"  Small:    {DIFF_LOTTO_BASELINE_SMALL}")
    print(f"  Large:    {DIFF_LOTTO_BASELINE_LARGE}")
    
    print("\n3. DISC GAME")
    print("-" * 80)
    print(f"  Baseline: {DISC_BASELINE}")
    print(f"  Small:    {DISC_BASELINE_SMALL}")
    print(f"  Large:    {DISC_BASELINE_LARGE}")
    
    print("\n4. PENNEYS GAME")
    print("-" * 80)
    print(f"  Baseline: {PENNEYS_BASELINE}")
    print(f"  Small:    {PENNEYS_BASELINE_SMALL}")
    print(f"  Large:    {PENNEYS_BASELINE_LARGE}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED BASELINE RUNS:")
    print("=" * 80)
    print("\nFor quick baseline comparison, run all games with baseline configs:")
    print("  - blotto: uniform, weaker, stronger (500 iterations)")
    print("  - diff_lotto: uniform, weaker, stronger (100 iterations)")
    print("  - disc: uniform, weaker, stronger (500 iterations)")
    print("  - penneys: uniform, weaker, stronger (500 iterations)")
    print("\nThis gives you 12 total runs to compare PSRO variants across all games.")


if __name__ == "__main__":
    print_baseline_configs()

