"""
Visualization functions for the discrete Colonel Blotto game.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from typing import List, Optional
from tqdm import tqdm

from games.blotto.blotto import BlottoAgent, index_to_allocation


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
    
    # Set up bar chart
    battlefields = ['Battlefield 1', 'Battlefield 2', 'Battlefield 3']
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
        ax1.set_ylim(0, 10)
        
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
            if len(idx) > 0:
                ax2.set_xlim(-0.5, idx[-1] + 0.5)
            if len(entropies) > 0:
                max_entropy = entropies[:t_range].max() if t_range > 0 else entropies.max()
                ax2.set_ylim(0, max(max_entropy * 1.1, 1))
            
            artists.extend(lines)
            artists.extend(scatters)
        
        plt.tight_layout()
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
        ax.set_xlim(0, idx[-1] if len(idx) > 0 else 1)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Tie')
        
        plt.tight_layout()
    
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
    path: str = "blotto_gamescape_matrix.png",
    n_rounds: int = 1000,
    dpi: int = 120,
) -> str:
    """
    Create an Empirical Gamescape Matrix showing payoffs between all agent pairs.
    
    Args:
        game: BlottoGame instance
        agents: List of agents
        path: output path
        n_rounds: number of rounds to evaluate payoffs
        dpi: figure DPI
        
    Returns:
        Path to saved plot
    """
    N = len(agents)
    payoff_matrix = np.zeros((N, N))
    
    # Compute payoffs for all pairs
    for i in range(N):
        for j in range(N):
            if i == j:
                payoff_matrix[i, j] = 0.5  # Agent vs itself
            else:
                # Payoff for agent i against agent j
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    
    # Use diverging colormap (green for positive, red for negative)
    # Shift payoffs so 0.5 (tie) is at center
    shifted_matrix = payoff_matrix - 0.5
    im = ax.imshow(shifted_matrix, cmap='RdYlGn', vmin=-0.5, vmax=0.5, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Payoff (shifted)')
    cbar.set_ticks([-0.5, 0, 0.5])
    cbar.set_ticklabels(['0.0 (Loss)', '0.5 (Tie)', '1.0 (Win)'])
    
    # Add text annotations
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i, f'{payoff_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(shifted_matrix[i, j]) > 0.25 else "black",
                          fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Agent j', fontsize=12)
    ax.set_ylabel('Agent i', fontsize=12)
    ax.set_title('Empirical Gamescape Matrix (Green: Positive, Red: Negative)', fontsize=14)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f'{i}' for i in range(N)])
    ax.set_yticklabels([f'{i}' for i in range(N)])
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    return path


def plot_2d_embeddings(
    game,
    agents: List[BlottoAgent],
    path: str = "blotto_2d_embeddings.png",
    n_rounds: int = 1000,
    dpi: int = 120,
    use_probabilities: bool = True,
) -> str:
    """
    Create 2D embeddings of agents using PCA, colored by row average payoff.
    
    Args:
        game: BlottoGame instance
        agents: List of agents
        path: output path
        n_rounds: number of rounds to evaluate payoffs
        dpi: figure DPI
        use_probabilities: If True, use probability distribution; if False, use logits
        
    Returns:
        Path to saved plot
    """
    N = len(agents)
    
    # Extract policy representations (66-dimensional)
    policies = []
    for agent in agents:
        if use_probabilities:
            policy = get_agent_probabilities(agent)
        else:
            # Use logits (normalized)
            policy = agent.logits - np.max(agent.logits)
            policy = np.exp(policy)
            policy = policy / np.sum(policy)
        policies.append(policy)
    
    policies = np.array(policies)  # Shape: (N, 66)
    
    # Compute PCA to reduce to 2D
    # Center the data
    mean_policy = np.mean(policies, axis=0)
    centered_policies = policies - mean_policy
    
    # Compute SVD for PCA
    U, s, Vt = np.linalg.svd(centered_policies, full_matrices=False)
    
    # Project to 2D using first 2 principal components
    embeddings_2d = U[:, :2] @ np.diag(s[:2])
    
    # Compute row average payoffs (average win rate for each agent)
    payoff_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                payoff_matrix[i, j] = 0.5
            else:
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    row_averages = np.mean(payoff_matrix, axis=1)  # Average payoff for each agent
    
    # Compute convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(embeddings_2d)
        hull_area = hull.volume  # In 2D, volume is area
        hull_points = embeddings_2d[hull.vertices]
    except ImportError:
        # Fallback: compute convex hull manually if scipy not available
        hull_area = 0.0
        hull_points = None
        # Simple convex hull approximation
        if N >= 3:
            # Find extreme points
            min_x_idx = np.argmin(embeddings_2d[:, 0])
            max_x_idx = np.argmax(embeddings_2d[:, 0])
            min_y_idx = np.argmin(embeddings_2d[:, 1])
            max_y_idx = np.argmax(embeddings_2d[:, 1])
            extreme_indices = [min_x_idx, max_x_idx, min_y_idx, max_y_idx]
            hull_points = embeddings_2d[extreme_indices]
            # Approximate area as bounding box area
            hull_area = (embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()) * \
                       (embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min())
    except:
        hull_area = 0.0
        hull_points = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    
    # Plot convex hull if available
    if hull_points is not None and len(hull_points) > 2:
        from matplotlib.patches import Polygon
        hull_polygon = Polygon(hull_points, closed=True, fill=True, 
                              alpha=0.2, facecolor='lightblue', edgecolor='blue', linewidth=1.5)
        ax.add_patch(hull_polygon)
    
    # Plot embeddings colored by row average payoff
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=row_averages, cmap='viridis', s=100, 
                        edgecolors='black', linewidths=1.5, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Row Average Payoff')
    
    # Add labels for each agent
    for i in range(N):
        ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], f'{i}',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white' if row_averages[i] < 0.5 else 'black', zorder=6)
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(f'2D Embeddings (Colored by Row Average) Convex Hull Area: {hull_area:.4f}', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    return path
