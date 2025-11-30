"""
Visualization functions for Penney's Game.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from typing import List, Optional, Dict
from tqdm import tqdm

from games.penneys.penneys import PennysAgent
from games.egs import EmpiricalGS, visualize_egs_matrix_and_embeddings


def get_entropy(agent: PennysAgent) -> float:
    """Compute entropy of the agent's probability distribution."""
    probs = agent.get_probabilities()
    probs = probs[probs > 1e-10]  # Remove zeros for log
    return -np.sum(probs * np.log(probs))


def gif_from_population(
    agents_history: List[List[PennysAgent]],
    path: str = "penneys_population.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    agent_colors: Optional[List] = None,
    show_entropy: bool = True,
) -> str:
    """
    Create a GIF from a population of Penney's Game agents evolving over time.
    
    Shows:
    - Probability distribution over sequences for each agent (bar chart)
    - Policy entropy (diversity measure) over time
    
    Args:
        agents_history: List of lists, where each inner list contains agents at a timestep.
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
    
    # Get sequence labels (same for all agents)
    sequences = agents_s[0][0].get_all_sequences()
    num_sequences = len(sequences)
    
    # Compute probabilities and entropies for all timesteps
    probabilities = []
    entropies = []
    for agents in agents_s:
        probabilities.append([agent.get_probabilities() for agent in agents])
        entropies.append([get_entropy(agent) for agent in agents])
    
    probabilities = np.array(probabilities)  # Shape: (T', N, num_sequences)
    entropies = np.array(entropies)  # Shape: (T', N)
    
    # Create figure with subplots
    if show_entropy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=dpi)
        ax2 = None
    
    # Set up bar chart
    x = np.arange(num_sequences)
    width = 0.8 / N  # Width of bars, divided by number of agents
    
    def animate(frame):
        ax1.clear()
        if ax2 is not None:
            ax2.clear()
        
        # Current timestep
        t = frame
        
        # Ensure t is within bounds
        if t >= len(agents_s) or t >= len(probabilities):
            return []
        
        # Plot probability distributions as grouped bar chart
        bars = []
        for i in range(N):
            # Calculate offset to center bars around each sequence
            offset = (i - (N - 1) / 2) * width
            probs = probabilities[t, i]
            bar = ax1.bar(x + offset, probs, width, 
                         label=f'Agent {i+1}', color=agent_colors[i], alpha=0.8)
            bars.extend(bar)
        
        ax1.set_xlabel('Sequence', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title(f"Probability Distribution Over Sequences (Iteration {idx[t]})", fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sequences, rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        texts = []
        for i in range(N):
            offset = (i - (N - 1) / 2) * width
            probs = probabilities[t, i]
            for j, val in enumerate(probs):
                if val > 0.05:  # Only label if significant
                    text = ax1.text(j + offset, val + 0.02, f'{val:.2f}', 
                                   ha='center', va='bottom', fontsize=8)
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
                if t < len(entropies):
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
        # Return empty list - return value is ignored when blit=False
        return []
    
    # Create animation
    num_frames = len(agents_s)
    if num_frames < 1:
        raise ValueError(f"Not enough frames for animation: {num_frames}")
    
    # If only one frame, create a static image instead
    if num_frames == 1:
        animate(0)
        static_path = path.replace('.gif', '.png')
        plt.savefig(static_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return static_path
    
    # Manually render each frame and save as images, then combine into GIF
    # This approach works better when using ax.clear() which can cause issues with PillowWriter
    frames = []
    
    # Ensure we have a non-interactive backend for rendering
    import matplotlib
    backend = matplotlib.get_backend()
    if backend.lower() == 'agg':
        # Agg backend works well for this
        pass
    else:
        # Try to set agg backend if not already set
        try:
            matplotlib.use('Agg', force=False)
        except:
            pass
    
    for frame_idx in range(num_frames):
        # Call animate to draw the frame
        animate(frame_idx)
        # Render the figure to a buffer - ensure it's fully drawn
        fig.canvas.draw()
        fig.canvas.flush_events()  # Ensure rendering is complete
        
        # Get the buffer - use print_to_buffer for more reliable capture
        try:
            # Method 1: Use print_to_buffer (most reliable)
            buf = fig.canvas.print_to_buffer()[0]
            w, h = fig.canvas.get_width_height()
            # print_to_buffer returns RGBA, convert to RGB
            buf = buf.reshape((h, w, 4))
            buf_rgb = buf[:, :, :3]  # Take only RGB channels
            frames.append(Image.fromarray(buf_rgb))
        except Exception:
            try:
                # Method 2: Use tostring_rgb()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                buf = buf.reshape((h, w, 3))
                frames.append(Image.fromarray(buf))
            except Exception:
                # Method 3: Use buffer_rgba as last resort
                buf = np.asarray(fig.canvas.buffer_rgba())
                buf = buf[:, :, :3]  # Convert RGBA to RGB
                frames.append(Image.fromarray(buf))
    
    # Save as animated GIF using PIL
    if len(frames) == 0:
        raise ValueError(f"No frames generated! num_frames={num_frames}, T={T}, stride={stride}")
    
    if len(frames) == 1:
        # Only one frame - save as static image
        static_path = path.replace('.gif', '.png')
        frames[0].save(static_path)
        plt.close(fig)
        return static_path
    
    # Calculate duration in milliseconds
    duration = int(1000 / fps)
    
    # Save as animated GIF
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        format='GIF'
    )
    
    plt.close(fig)
    return path


def gif_from_matchups(
    game,
    agents_history: List[List[PennysAgent]],
    path: str = "penneys_matchups.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    n_rounds: int = 500,
    agent_colors: Optional[List] = None,
    show_labels: bool = True,
) -> str:
    """
    Create a GIF showing matchups between agents over time.
    
    Shows win rates between all agent pairs as they evolve.
    
    Args:
        game: PennysGame instance
        agents_history: List of lists of agents at each timestep
        path: output GIF path
        fps: frames per second
        stride: render every k-th frame
        dpi: figure DPI
        n_rounds: number of rounds to evaluate win rates
        agent_colors: optional list of colors for each agent
        show_labels: whether to show labels
        
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
    win_rates = []  # List of dicts: {(0, 1): rate, ...}
    
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
    if agent_colors is None:
        agent_colors = plt.cm.tab10(np.linspace(0, 1, N))
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
        ax.set_title(f"Win Rates Between Agents (Iteration {idx[t]})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, idx[-1] if len(idx) > 0 else 1)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Tie')
        
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
    agents: List[PennysAgent],
    output_path: str,
    n_rounds: int = 1000,
    dpi: int = 120
) -> str:
    """
    Create a heatmap showing payoffs between all agent pairs (gamescape matrix).
    
    Args:
        game: PennysGame instance
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
                # Agent vs itself = tie (0.0 for Penneys, which uses [-1, 1] range)
                payoff_matrix[i, j] = 0.0
            else:
                # Compute payoff for agent i vs agent j
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, N * 0.8), max(6, N * 0.7)), dpi=dpi)
    
    # Use diverging colormap centered at 0
    # Penneys game uses [-1, 1] range, so center at 0
    vmax = np.abs(payoff_matrix).max()
    if vmax < 1e-10:
        vmax = 1.0  # Avoid division by zero
    im = ax.imshow(payoff_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    # Add text annotations
    for i in range(N):
        for j in range(N):
            text_color = 'white' if abs(payoff_matrix[i, j]) > vmax * 0.3 else 'black'
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
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Payoff (Agent i)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_all_egs_visualizations(
    game,
    agents: List[PennysAgent],
    output_dir: str,
    base_name: str = "egs",
    n_rounds: int = 1000,
    dpi: int = 150
) -> Dict[str, str]:
    """
    Generate all EGS visualizations: matrix + PCA, Schur, SVD, and t-SNE embeddings.
    
    Args:
        game: PennysGame instance
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
    
    # Compute payoff matrix (payoffs in [-1, 1] for Penneys)
    payoff_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                payoff_matrix[i, j] = 0.0  # Agent vs itself = tie
            else:
                payoff = game.play(agents[i], agents[j], n_rounds=n_rounds)
                payoff_matrix[i, j] = payoff
    
    # Convert to antisymmetric EGS matrix (already centered at 0, zero-sum)
    # For Penneys, payoffs are already in [-1, 1] range centered at 0
    egs_matrix = (payoff_matrix - payoff_matrix.T) / 2  # Make antisymmetric
    
    # Create EmpiricalGS instance
    try:
        gamescape = EmpiricalGS(egs_matrix)
    except AssertionError:
        # Fallback: create a minimal valid EGS matrix
        egs_matrix_fallback = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                val = payoff_matrix[i, j]
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

