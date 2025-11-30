"""
Visualization functions for the discrete Colonel Blotto game.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from typing import List, Optional, Dict
from tqdm import tqdm

from games.blotto.blotto import BlottoAgent, index_to_allocation
from games.egs import EmpiricalGS, visualize_egs_matrix_and_embeddings


def get_agent_probabilities(agent: BlottoAgent) -> np.ndarray:
    """Convert agent logits to probability distribution over allocations."""
    exp_logits = np.exp(agent.logits - np.max(agent.logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs


def get_expected_allocation(agent: BlottoAgent) -> np.ndarray:
    """Compute expected allocation (expected troops per battlefield)."""
    probs = get_agent_probabilities(agent)
    expected = np.zeros(agent.n_battlefields)
    
    for idx in range(len(probs)):
        alloc = index_to_allocation(idx, agent.budget, agent.n_battlefields)
        expected += probs[idx] * np.array(alloc)
    
    return expected


def get_entropy(agent: BlottoAgent) -> float:
    """Compute entropy of the agent's policy distribution."""
    probs = get_agent_probabilities(agent)
    probs = probs[probs > 1e-10]  # Remove zeros for log
    return -np.sum(probs * np.log(probs))


def gif_from_population(
    agents_history: List[List[BlottoAgent]],
    path: str = "blotto_population.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    agent_colors: Optional[List] = None,
    show_entropy: bool = True,
) -> str:
    """
    Create a GIF from a population of Blotto agents evolving over time.
    
    Shows:
    - Expected allocation per battlefield for each agent (bar chart)
    - Policy entropy (diversity measure) over time
    - Win rates between agents
    
    Args:
        agents_history: List of lists, where each inner list contains agents at a timestep.
                       Shape: (T, N) where T is timesteps and N is number of agents.
        path: output GIF path.
        fps: frames per second of the GIF.
        stride: render every k-th frame.
        dpi: figure DPI.
        agent_colors: optional list of colors for each agent.
        show_entropy: whether to show entropy plot.
        
    Returns:
        The path to the saved GIF.
    """
    T = len(agents_history)
    if T == 0:
        raise ValueError("agents_history is empty")
    
    N = len(agents_history[0])
    if N == 0:
        raise ValueError("No agents in history")
    
    if agent_colors is None:
        agent_colors = plt.cm.tab10(np.linspace(0, 1, N))
    
    # Subsample frames
    idx = np.arange(0, T, stride)
    agents_s = [agents_history[i] for i in idx]
    
    # Compute expected allocations and entropies for all timesteps
    expected_allocs = []
    entropies = []
    for agents in agents_s:
        expected_allocs.append([get_expected_allocation(agent) for agent in agents])
        entropies.append([get_entropy(agent) for agent in agents])
    
    expected_allocs = np.array(expected_allocs)  # Shape: (T', N, 3)
    entropies = np.array(entropies)  # Shape: (T', N)
    
    # Compute win rates over time (if we have a game instance)
    # For now, we'll just show allocations and entropy
    
    # Create figure with subplots
    if show_entropy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=dpi)
        ax2 = None
    
    # Set up bar chart - dynamically determine number of battlefields from first agent
    if len(agents_history[0]) > 0:
        n_battlefields = agents_history[0][0].n_battlefields
        budget = agents_history[0][0].budget
    else:
        n_battlefields = 3
        budget = 10
    
    battlefields = [f'Battlefield {i+1}' for i in range(n_battlefields)]
    x = np.arange(len(battlefields))
    width = 0.8 / N  # Width of bars, divided by number of agents
    
    def animate(frame):
        ax1.clear()
        if ax2 is not None:
            ax2.clear()
        
        # Current timestep
        t = frame
        
        # Ensure t is within bounds
        if t >= len(agents_s) or t >= len(expected_allocs):
            return []
        
        # Plot expected allocations as grouped bar chart
        bars = []
        for i in range(N):
            # Calculate offset to center bars around each battlefield
            offset = (i - (N - 1) / 2) * width
            expected = expected_allocs[t, i]
            bar = ax1.bar(x + offset, expected, width, 
                         label=f'Agent {i+1}', color=agent_colors[i], alpha=0.8)
            bars.extend(bar)
        
        ax1.set_xlabel('Battlefield', fontsize=12)
        ax1.set_ylabel('Expected Troops', fontsize=12)
        ax1.set_title(f'Expected Allocation per Battlefield (Iteration {idx[t]})', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(battlefields)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        # Set y-axis limit based on budget (with generous padding)
        ax1.set_ylim(0, budget * 1.2)
        
        # Add value labels on bars
        texts = []
        for i in range(N):
            offset = (i - (N - 1) / 2) * width
            expected = expected_allocs[t, i]
            for j, val in enumerate(expected):
                text = ax1.text(j + offset, val + 0.2, f'{val:.1f}', 
                               ha='center', va='bottom', fontsize=9)
                texts.append(text)
        
        # Plot entropy over time
        artists = bars + texts
        if ax2 is not None:
            # Show entropy history up to current frame
            t_range = min(t + 1, len(entropies))
            lines = []
            scatters = []
            if t_range > 0:
                for i in range(N):
                    line, = ax2.plot(idx[:t_range], entropies[:t_range, i], 
                                    label=f'Agent {i+1}', color=agent_colors[i], linewidth=2)
                    lines.append(line)
                
                # Mark current point
                for i in range(N):
                    scatter = ax2.scatter([idx[t]], [entropies[t, i]], 
                                         color=agent_colors[i], s=100, zorder=5)
                    scatters.append(scatter)
            
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Policy Entropy', fontsize=12)
            ax2.set_title('Policy Diversity Over Time', fontsize=14)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            # Let matplotlib auto-scale axes instead of setting restrictive limits
            if len(idx) > 0:
                ax2.set_xlim(0, idx[-1] * 1.05 if idx[-1] > 0 else 1)
            if len(entropies) > 0:
                max_entropy = entropies[:t_range].max() if t_range > 0 else entropies.max()
                ax2.set_ylim(0, max(max_entropy * 1.2, 1.1))
            
            artists.extend(lines)
            artists.extend(scatters)
        
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        return artists
    
    # Create animation
    num_frames = len(agents_s)
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=1000/fps, repeat=True)
    
    # Save as GIF
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    
    return path


def gif_from_matchups(
    game,
    agents_history: List[List[BlottoAgent]],
    path: str = "blotto_matchups.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    n_rounds: int = 1000,
) -> str:
    """
    Create a GIF showing matchups between agents over time.
    
    Shows win rates between all agent pairs as they evolve.
    
    Args:
        game: BlottoGame instance
        agents_history: List of lists of agents at each timestep
        path: output GIF path
        fps: frames per second
        stride: render every k-th frame
        dpi: figure DPI
        n_rounds: number of rounds to evaluate win rates
        
    Returns:
        The path to the saved GIF.
    """
    T = len(agents_history)
    if T == 0:
        raise ValueError("agents_history is empty")
    
    N = len(agents_history[0])
    if N < 2:
        raise ValueError("Need at least 2 agents for matchups")
    
    # Subsample frames
    idx = np.arange(0, T, stride)
    agents_s = [agents_history[i] for i in idx]
    
    # Compute win rates for all timesteps
    win_rates = []  # List of dicts: {('0', '1'): rate, ...}
    
    for agents in tqdm(agents_s, desc="Computing win rates"):
        rates = {}
        for i in range(N):
            for j in range(i + 1, N):
                rate = game.play(agents[i], agents[j], n_rounds=n_rounds)
                rates[(i, j)] = rate
        win_rates.append(rates)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    # Get all matchup pairs
    matchups = list(win_rates[0].keys())
    matchup_labels = [f'Agent {i+1} vs Agent {j+1}' for i, j in matchups]
    
    # Colors for each matchup
    matchup_colors = plt.cm.tab10(np.linspace(0, 1, len(matchups)))
    
    def animate(frame):
        ax.clear()
        
        # Current timestep
        t = frame
        
        # Plot win rate history up to current frame
        for idx_matchup, (i, j) in enumerate(matchups):
            history = [win_rates[k][(i, j)] for k in range(min(t + 1, len(win_rates)))]
            ax.plot(idx[:len(history)], history, 
                   label=matchup_labels[idx_matchup], 
                   color=matchup_colors[idx_matchup], linewidth=2)
            
            # Mark current point
            if t < len(win_rates):
                ax.scatter([idx[t]], [win_rates[t][(i, j)]], 
                          color=matchup_colors[idx_matchup], s=100, zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Win Rate (Agent i)', fontsize=12)
        ax.set_title(f'Win Rates Between Agents (Iteration {idx[t]})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Use more flexible limits
        if len(idx) > 0 and idx[-1] > 0:
            ax.set_xlim(-0.05 * idx[-1], idx[-1] * 1.05)
        else:
            ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)  # Allow slight overflow for visibility
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Tie')
        
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # Create animation
    num_frames = len(agents_s)
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=1000/fps, repeat=True)
    
    # Save as GIF
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    
    return path


def plot_gamescape_matrix(
    game,
    agents: List[BlottoAgent],
    output_path: str,
    n_rounds: int = 1000,
    dpi: int = 120
) -> str:
    """
    Create a heatmap showing payoffs between all agent pairs (gamescape matrix).
    
    Args:
        game: BlottoGame instance
        agents: List of agents
        output_path: Path to save the plot
        n_rounds: Number of rounds per evaluation
        dpi: Figure DPI
        
    Returns:
        Path to saved plot
    """
    N = len(agents)
    if N < 2:
        raise ValueError("Need at least 2 agents for gamescape matrix")
    
    # Compute payoff matrix
    payoff_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                # Agent vs itself = tie (0.5)
                payoff_matrix[i, j] = 0.5
            else:
                # Compute payoff for agent i vs agent j
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, N * 0.8), max(6, N * 0.7)), dpi=dpi)
    
    # Use diverging colormap centered at 0.5
    # Shift values so 0.5 is at center: map [0, 1] -> [-0.5, 0.5] -> [0, 1] for colormap
    shifted_matrix = payoff_matrix - 0.5  # Center at 0
    im = ax.imshow(shifted_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=-0.5, vmax=0.5, interpolation='nearest')
    
    # Add text annotations
    for i in range(N):
        for j in range(N):
            text_color = 'white' if abs(shifted_matrix[i, j]) > 0.15 else 'black'
            ax.text(j, i, f'{payoff_matrix[i, j]:.3f}',
                   ha='center', va='center', color=text_color, fontsize=10)
    
    # Set labels
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f'Agent {i+1}' for i in range(N)])
    ax.set_yticklabels([f'Agent {i+1}' for i in range(N)])
    ax.set_xlabel('Opponent', fontsize=12)
    ax.set_ylabel('Agent', fontsize=12)
    ax.set_title('Gamescape Matrix: Payoff Between Agent Pairs', fontsize=14, pad=20)
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Payoff (Agent i)', rotation=270, labelpad=20, fontsize=11)
    # Set custom tick labels for colorbar
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    
    plt.subplots_adjust(left=0.1, right=0.92, top=0.95, bottom=0.1)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_all_egs_visualizations(
    game,
    agents: List[BlottoAgent],
    output_dir: str,
    base_name: str = "egs",
    n_rounds: int = 1000,
    dpi: int = 150
) -> Dict[str, str]:
    """
    Generate all EGS visualizations: matrix + PCA, Schur, SVD, and t-SNE embeddings.
    
    Args:
        game: BlottoGame instance
        agents: List of agents
        output_dir: Directory to save the plots
        base_name: Base name for output files
        n_rounds: Number of rounds per evaluation
        dpi: Figure DPI
        
    Returns:
        Dictionary mapping embedding method names to file paths
    """
    import os
    from pathlib import Path
    
    N = len(agents)
    if N < 2:
        raise ValueError("Need at least 2 agents for 2D embeddings")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute payoff matrix (win rates in [0, 1])
    payoff_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                payoff_matrix[i, j] = 0.5  # Agent vs itself = tie
            else:
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    # Convert to antisymmetric EGS matrix (centered at 0, zero-sum)
    egs_matrix = payoff_matrix - 0.5  # Center at 0
    egs_matrix = (egs_matrix - egs_matrix.T) / 2  # Make antisymmetric
    
    # Create EmpiricalGS instance
    try:
        gamescape = EmpiricalGS(egs_matrix)
    except AssertionError:
        # Fallback: create a minimal valid EGS matrix
        egs_matrix_fallback = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                val = payoff_matrix[i, j] - 0.5
                egs_matrix_fallback[i, j] = val
                egs_matrix_fallback[j, i] = -val
        gamescape = EmpiricalGS(egs_matrix_fallback)
    
    # Generate all embedding methods
    methods = ["PCA", "SVD", "schur", "tSNE"]
    output_paths = {}
    
    for method in methods:
        try:
            # Get embeddings based on method
            if method.lower() == "pca":
                coords_2d = gamescape.PCA_embeddings()
            elif method.lower() == "svd":
                coords_2d = gamescape.SVD_embeddings()
            elif method.lower() == "schur":
                coords_2d = gamescape.schur_embeddings()
            elif method.lower() == "tsne":
                coords_2d = gamescape.tSNE_embeddings()
            else:
                continue
            
            # Create output path (use absolute path to avoid issues)
            output_path = os.path.abspath(os.path.join(output_dir, f"{base_name}_{method.lower()}.png"))
            
            # Generate visualization
            visualize_egs_matrix_and_embeddings(gamescape, coords_2d, save_path=output_path, dpi=dpi)
            output_paths[method] = output_path
            
        except Exception as e:
            print(f"Warning: Could not generate {method} visualization: {e}")
            continue
    
    return output_paths


def plot_2d_embeddings(
    game,
    agents: List[BlottoAgent],
    output_path: str,
    n_rounds: int = 1000,
    dpi: int = 120,
    embedding_method: str = "PCA",
    show_convex_hull: bool = False
) -> str:
    """
    Create a combined visualization of the gamescape matrix and 2D embeddings using Empirical Gamescape methods.
    Uses the payoff matrix to embed agents based on their game-theoretic relationships.
    This function uses visualize_egs_matrix_and_embeddings from egs.py for the visualization.
    
    Args:
        game: BlottoGame instance
        agents: List of agents
        output_path: Path to save the plot
        n_rounds: Number of rounds per evaluation
        dpi: Figure DPI
        embedding_method: Method to use ("PCA", "SVD", "schur", or "tSNE")
        show_convex_hull: Whether to draw convex hull around points (always shown if >= 3 agents)
        
    Returns:
        Path to saved plot
    """
    N = len(agents)
    if N < 2:
        raise ValueError("Need at least 2 agents for 2D embeddings")
    
    # Compute payoff matrix (win rates in [0, 1])
    payoff_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                payoff_matrix[i, j] = 0.5  # Agent vs itself = tie
            else:
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    # Convert to antisymmetric EGS matrix (centered at 0, zero-sum)
    # EGS matrix: egs[i,j] = payoff[i,j] - 0.5, then make antisymmetric
    egs_matrix = payoff_matrix - 0.5  # Center at 0
    egs_matrix = (egs_matrix - egs_matrix.T) / 2  # Make antisymmetric
    
    # Create EmpiricalGS instance and get embeddings
    try:
        gamescape = EmpiricalGS(egs_matrix)
        
        # Get embeddings based on method
        if embedding_method.lower() == "pca":
            coords_2d = gamescape.PCA_embeddings()
        elif embedding_method.lower() == "svd":
            coords_2d = gamescape.SVD_embeddings()
        elif embedding_method.lower() == "schur":
            coords_2d = gamescape.schur_embeddings()
        elif embedding_method.lower() == "tsne":
            coords_2d = gamescape.tSNE_embeddings()
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}. Use 'PCA', 'SVD', 'schur', or 'tSNE'")
    except (AssertionError, ValueError) as e:
        # If matrix validation fails, fall back to simple PCA on payoff matrix
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(payoff_matrix)
        embedding_method = "PCA (fallback)"
        # Try to recreate gamescape with fallback matrix
        try:
            egs_matrix_fallback = payoff_matrix - 0.5
            egs_matrix_fallback = (egs_matrix_fallback - egs_matrix_fallback.T) / 2
            gamescape = EmpiricalGS(egs_matrix_fallback)
        except AssertionError:
            # If even the fallback fails, create a minimal valid EGS matrix
            egs_matrix_fallback = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1, N):
                    val = payoff_matrix[i, j] - 0.5
                    egs_matrix_fallback[i, j] = val
                    egs_matrix_fallback[j, i] = -val
            gamescape = EmpiricalGS(egs_matrix_fallback)
    
    # Use the visualize_egs_matrix_and_embeddings function from egs.py
    # This creates the side-by-side visualization with matrix and embeddings
    visualize_egs_matrix_and_embeddings(gamescape, coords_2d, save_path=output_path, dpi=dpi)
    
    return output_path
