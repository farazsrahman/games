# Visualization functions for differentiable_lotto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
import io
from typing import List, Optional, Tuple


def plot_colonel_blotto_state(
    game,
    agents: List[Tuple[np.ndarray, np.ndarray]],
    agent_colors: Optional[List] = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 120,
    show_customers: bool = True,
    show_assignments: bool = True,
    assignment_alpha: float = 0.3,
) -> Image.Image:
    """
    Plot a single state of the Differentiable Lotto game showing customers, servers, and assignments.
    
    Args:
        game: DifferentiableLotto instance
        agents: List of agents, each as (p, v) tuple
        agent_colors: Optional list of colors for each agent
        figsize: Figure size
        dpi: DPI
        show_customers: Whether to show customer points
        show_assignments: Whether to show soft assignments as lines
        assignment_alpha: Transparency for assignment lines
        
    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Determine plot limits from customers and servers
    all_points = [game.customers]
    for p, v in agents:
        all_points.append(v)
    all_points = np.concatenate(all_points, axis=0)
    
    if all_points.size > 0:
        max_extent = np.abs(all_points).max()
        pad = 0.2 * max_extent
        lim = max_extent + pad
    else:
        lim = 3.0
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    
    # Set up colors for agents
    num_agents = len(agents)
    if agent_colors is None:
        cmap = cm.get_cmap('tab10')
        agent_colors = [cmap(i % 10) for i in range(num_agents)]
    elif len(agent_colors) != num_agents:
        cmap = cm.get_cmap('tab10')
        agent_colors = [agent_colors[i % len(agent_colors)] if agent_colors else cmap(i % 10) 
                       for i in range(num_agents)]
    
    # Plot customers
    if show_customers:
        ax.scatter(game.customers[:, 0], game.customers[:, 1], 
                  c='black', s=20, alpha=0.5, marker='o', 
                  label='Customers', zorder=1)
    
    # Plot servers and assignments for each agent
    for agent_idx, (p, v) in enumerate(agents):
        color = agent_colors[agent_idx]
        
        # Plot servers with size proportional to mass
        server_sizes = p * 500  # Scale mass for visibility
        ax.scatter(v[:, 0], v[:, 1], c=[color], s=server_sizes, 
                  alpha=0.8, edgecolors='black', linewidths=1.5,
                  label=f'Agent {agent_idx} servers', zorder=3)
        
        # Show soft assignments
        if show_assignments:
            for i, customer in enumerate(game.customers):
                # Compute soft assignments for this customer
                # For simplicity, show assignments to the first agent's servers
                # In a real game, we'd need both agents, but for visualization
                # we'll just show assignments to this agent's servers
                dists = np.sum((customer - v) ** 2, axis=1)
                softmax_dists = np.exp(-dists) / np.sum(np.exp(-dists))
                
                # Draw lines from customer to servers, weighted by assignment
                for j in range(game.k):
                    if softmax_dists[j] > 0.1:  # Only show significant assignments
                        ax.plot([customer[0], v[j, 0]], 
                               [customer[1], v[j, 1]],
                               color=color, alpha=assignment_alpha * softmax_dists[j],
                               linewidth=1, zorder=2)
    
    ax.set_title(f"Colonel Blotto Game State ({num_agents} agents, {game.c} customers, {game.k} servers)")
    if num_agents <= 5:
        ax.legend(loc="upper left", fontsize=8)
    
    # Render to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    
    buf.seek(0)
    img = Image.open(buf).convert("RGBA").copy()
    buf.close()
    return img


def gif_from_matchups(
    game,
    agents_history: List[List[Tuple[np.ndarray, np.ndarray]]],
    path: str = "colonel_blotto_matchups.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    show_customers: bool = True,
    show_gradients: bool = True,
    gradient_scale: float = 0.3,
) -> str:
    """
    Create a GIF showing separate plots for each matchup, indicating which agent is winning.
    
    Args:
        game: DifferentiableLotto instance
        agents_history: List of lists, where each inner list contains agents at a timestep.
                       Shape: (T, N) where T is timesteps and N is number of agents.
                       Each agent is a (p, v) tuple.
        path: output GIF path.
        fps: frames per second of the GIF.
        stride: render every k-th frame.
        dpi: figure DPI.
        show_customers: whether to show customer points.
        
    Returns:
        The path to the saved GIF.
    """
    T = len(agents_history)
    if T == 0:
        raise ValueError("agents_history is empty")
    
    N = len(agents_history[0])
    if N < 2:
        raise ValueError("Need at least 2 agents for matchups")
    
    # Create all pairwise matchups
    matchups = []
    for i in range(N):
        for j in range(i + 1, N):
            matchups.append((i, j))
    
    if len(matchups) == 0:
        raise ValueError("No matchups to visualize")
    
    lim = 2.0  # Fixed limits for width=1 constrained servers
    
    idx = np.arange(0, T, stride)
    agents_s = [agents_history[i] for i in idx]
    
    # Create subplots: one row per matchup
    n_matchups = len(matchups)
    fig, axes = plt.subplots(n_matchups, 1, figsize=(10, 4 * n_matchups), dpi=dpi)
    if n_matchups == 1:
        axes = [axes]
    
    # Set up colors
    default_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    # Initialize plots for each matchup
    scatters_list = []  # List of lists: [matchup][agent][server]
    quivers_list = []   # List of lists: [matchup][agent][server] for gradient arrows
    titles = []
    
    for matchup_idx, (i, j) in enumerate(matchups):
        ax = axes[matchup_idx]
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="lightgray", lw=0.5, alpha=0.3, linestyle='--')
        ax.axvline(0, color="lightgray", lw=0.5, alpha=0.3, linestyle='--')
        
        # Draw square boundary
        square = plt.Rectangle((-1, -1), 2, 2, fill=False, 
                              edgecolor='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.add_patch(square)
        
        # Plot customers
        if show_customers:
            ax.scatter(game.customers[:, 0], game.customers[:, 1],
                     c='black', s=20, alpha=0.4, marker='o', zorder=1)
        
        # Initialize scatters and quivers for both agents
        matchup_scatters = []
        matchup_quivers = []
        for agent_idx, color in [(i, default_colors[i % len(default_colors)]), 
                                  (j, default_colors[j % len(default_colors)])]:
            p0, v0 = agents_s[0][agent_idx]
            agent_scatters = []
            agent_quivers = []
            for server_idx in range(game.k):
                server_size = max(80, p0[server_idx] * 600)
                sc = ax.scatter([v0[server_idx, 0]], [v0[server_idx, 1]], 
                               c=[color], s=server_size, 
                               alpha=0.7, edgecolors='black', linewidths=1.5,
                               zorder=3, marker='o')
                agent_scatters.append(sc)
                
                # Initialize quiver for gradient (will be updated in frame function)
                if show_gradients:
                    q = ax.quiver([v0[server_idx, 0]], [v0[server_idx, 1]], 
                                [0], [0],  # Will be updated with actual gradients
                                angles='xy', scale_units='xy', scale=1,
                                color=color, width=0.003, alpha=0.6, zorder=2)
                    agent_quivers.append(q)
                else:
                    agent_quivers.append(None)
            matchup_scatters.append(agent_scatters)
            matchup_quivers.append(agent_quivers)
        
        scatters_list.append(matchup_scatters)
        quivers_list.append(matchup_quivers)
        title = ax.set_title("", fontsize=11, pad=10)
        titles.append(title)
    
    def frame(frame_idx):
        agents = agents_s[frame_idx]
        all_artists = []
        
        for matchup_idx, (i, j) in enumerate(matchups):
            ax = axes[matchup_idx]
            agent_i, agent_j = agents[i], agents[j]
            
            # Compute payoff to determine winner
            payoff_ij = game.play(agent_i, agent_j)
            
            # Update scatters and gradients for both agents
            for agent_list_idx, (agent_idx, agent) in enumerate([(i, agent_i), (j, agent_j)]):
                p, v = agent
                
                # Compute gradients if showing them
                if show_gradients:
                    # Compute gradient for this agent against the opponent
                    grad_p, grad_v = game._compute_gradient(agent, agent_j if agent_idx == i else agent_i)
                    # Normalize gradient for visualization
                    grad_v_norm = grad_v.copy()
                    grad_v_magnitude = np.linalg.norm(grad_v, axis=1, keepdims=True)
                    grad_v_magnitude[grad_v_magnitude < 1e-8] = 1.0  # Avoid division by zero
                    grad_v_norm = grad_v_norm / grad_v_magnitude * gradient_scale
                
                for server_idx in range(game.k):
                    server_size = max(80, p[server_idx] * 600)
                    scatters_list[matchup_idx][agent_list_idx][server_idx].set_offsets(
                        [[v[server_idx, 0], v[server_idx, 1]]])
                    scatters_list[matchup_idx][agent_list_idx][server_idx].set_sizes([server_size])
                    
                    # Update gradient arrows
                    if show_gradients and quivers_list[matchup_idx][agent_list_idx][server_idx] is not None:
                        q = quivers_list[matchup_idx][agent_list_idx][server_idx]
                        q.set_offsets(np.c_[[v[server_idx, 0]], [v[server_idx, 1]]])
                        q.set_UVC([grad_v_norm[server_idx, 0]], [grad_v_norm[server_idx, 1]])
            
            # Update title with matchup info and winner
            if payoff_ij > 0:
                winner = f"Agent {i} wins"
                winner_color = default_colors[i % len(default_colors)]
            elif payoff_ij < 0:
                winner = f"Agent {j} wins"
                winner_color = default_colors[j % len(default_colors)]
            else:
                winner = "Tie"
                winner_color = "gray"
            
            title_text = f"Differentiable Colonel Blotto: Agent {i} vs Agent {j} | Payoff: {payoff_ij:.4f} | {winner}"
            titles[matchup_idx].set_text(title_text)
            titles[matchup_idx].set_color(winner_color)
            
            all_artists.extend([item for sublist in scatters_list[matchup_idx] for item in sublist])
            if show_gradients:
                all_artists.extend([q for sublist in quivers_list[matchup_idx] 
                                   for q in sublist if q is not None])
            all_artists.append(titles[matchup_idx])
        
        # Add timestep info
        timestep_text = f"Timestep {idx[frame_idx]}/{T-1}"
        fig.suptitle(timestep_text, fontsize=12, y=0.995)
        all_artists.append(fig._suptitle)
        
        return tuple(all_artists)
    
    anim = animation.FuncAnimation(
        fig, frame, frames=len(agents_s), interval=1000.0 / fps, blit=False
    )
    
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    return path


def gif_from_population(
    game,
    agents_history: List[List[Tuple[np.ndarray, np.ndarray]]],
    path: str = "colonel_blotto_population.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    show_customers: bool = True,
    show_assignments: bool = False,
    agent_colors: Optional[List] = None,
    show_labels: bool = True,
    show_gradients: bool = True,
    gradient_scale: float = 0.3,
) -> str:
    """
    Create a GIF from a population of Colonel Blotto agents evolving over time.
    Visualization matches the paper's style: shows customers and server positions clearly.
    
    Args:
        game: DifferentiableLotto instance
        agents_history: List of lists, where each inner list contains agents at a timestep.
                       Shape: (T, N) where T is timesteps and N is number of agents.
                       Each agent is a (p, v) tuple.
        path: output GIF path.
        fps: frames per second of the GIF.
        stride: render every k-th frame.
        dpi: figure DPI.
        show_customers: whether to show customer points.
        show_assignments: whether to show soft assignments (can be slow).
        agent_colors: optional list of colors for each agent.
        show_labels: whether to show legend.
        
    Returns:
        The path to the saved GIF.
    """
    T = len(agents_history)
    if T == 0:
        raise ValueError("agents_history is empty")
    
    N = len(agents_history[0])
    if N == 0:
        raise ValueError("No agents in history")
    
    # Determine plot limits - with width=1 constraint, servers should stay reasonably bounded
    # Use fixed limits based on expected bounds (width=1 means servers are within ~2-3 units typically)
    # Paper uses customers in [-1, 1]², so we'll use that as base with some padding
    lim = 2.0  # Should be sufficient for width=1 constrained servers
    
    idx = np.arange(0, T, stride)
    agents_s = [agents_history[i] for i in idx]
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="lightgray", lw=0.5, alpha=0.3, linestyle='--')
    ax.axvline(0, color="lightgray", lw=0.5, alpha=0.3, linestyle='--')
    
    # Draw square boundary to show [-1, 1]² region
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, 
                          edgecolor='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.add_patch(square)
    
    # Set up colors - use distinct colors for each agent
    if agent_colors is None:
        # Use distinct colors: blue, red, green, orange, purple
        default_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        agent_colors = [default_colors[i % len(default_colors)] for i in range(N)]
    elif len(agent_colors) != N:
        cmap = cm.get_cmap('tab10')
        agent_colors = [agent_colors[i % len(agent_colors)] if agent_colors else cmap(i % 10) 
                       for i in range(N)]
    
    # Plot customers (static) - shown as small black dots
    if show_customers:
        customer_scatter = ax.scatter(game.customers[:, 0], game.customers[:, 1],
                                     c='black', s=30, alpha=0.6, marker='o',
                                     zorder=1, label='Customers', edgecolors='none')
    
    # Initialize scatter plots and quivers for servers of each agent
    # Each agent has k servers, so we need k scatter points per agent
    scatters = []  # List of lists: [agent][server]
    quivers = []   # List of lists: [agent][server] for gradient arrows
    for agent_idx in range(N):
        p0, v0 = agents_s[0][agent_idx]
        agent_scatters = []
        agent_quivers = []
        for server_idx in range(game.k):
            # Size proportional to mass, with minimum size for visibility
            server_size = max(100, p0[server_idx] * 800)
            sc = ax.scatter([v0[server_idx, 0]], [v0[server_idx, 1]], 
                           c=[agent_colors[agent_idx]], s=server_size, 
                           alpha=0.7, edgecolors='black', linewidths=1.5,
                           zorder=3, marker='o')
            agent_scatters.append(sc)
            
            # Initialize quiver for gradient (will be updated in frame function)
            if show_gradients:
                q = ax.quiver([v0[server_idx, 0]], [v0[server_idx, 1]], 
                            [0], [0],  # Will be updated with actual gradients
                            angles='xy', scale_units='xy', scale=1,
                            color=agent_colors[agent_idx], width=0.003, 
                            alpha=0.6, zorder=2)
                agent_quivers.append(q)
            else:
                agent_quivers.append(None)
        scatters.append(agent_scatters)
        quivers.append(agent_quivers)
    
    title = ax.set_title("", fontsize=12)
    if show_labels and N <= 5:
        # Create legend entries for each agent
        legend_elements = []
        if show_customers:
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='black', markersize=8, 
                                        label='Customers', markeredgecolor='none'))
        for agent_idx in range(N):
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=agent_colors[agent_idx], 
                                        markersize=10, label=f'Agent {agent_idx} servers',
                                        markeredgecolor='black', markeredgewidth=1.5))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)
    
    def frame(i):
        agents = agents_s[i]
        
        # Update all agents' servers and gradients
        for agent_idx in range(N):
            p, v = agents[agent_idx]
            
            # Compute gradients if showing them
            # For population view, compute gradient against a representative opponent
            # (e.g., average opponent or first other agent)
            if show_gradients:
                # Use first other agent as opponent (or self if only one agent)
                opponent_idx = (agent_idx + 1) % N if N > 1 else agent_idx
                opponent = agents[opponent_idx]
                grad_p, grad_v = game._compute_gradient(agents[agent_idx], opponent)
                # Normalize gradient for visualization
                grad_v_norm = grad_v.copy()
                grad_v_magnitude = np.linalg.norm(grad_v, axis=1, keepdims=True)
                grad_v_magnitude[grad_v_magnitude < 1e-8] = 1.0  # Avoid division by zero
                grad_v_norm = grad_v_norm / grad_v_magnitude * gradient_scale
            
            for server_idx in range(game.k):
                server_size = max(100, p[server_idx] * 800)
                scatters[agent_idx][server_idx].set_offsets([[v[server_idx, 0], v[server_idx, 1]]])
                scatters[agent_idx][server_idx].set_sizes([server_size])
                
                # Update gradient arrows
                if show_gradients and quivers[agent_idx][server_idx] is not None:
                    q = quivers[agent_idx][server_idx]
                    q.set_offsets(np.c_[[v[server_idx, 0]], [v[server_idx, 1]]])
                    q.set_UVC([grad_v_norm[server_idx, 0]], [grad_v_norm[server_idx, 1]])
        
        title.set_text(f"Timestep {idx[i]}/{T-1}")
        artists = [item for sublist in scatters for item in sublist] + [title]
        if show_gradients:
            artists.extend([q for sublist in quivers for q in sublist if q is not None])
        if show_customers:
            artists.append(customer_scatter)
        return tuple(artists)
    
    anim = animation.FuncAnimation(
        fig, frame, frames=len(agents_s), interval=1000.0 / fps, blit=True
    )
    
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    return path


def plot_agent_comparison(
    game,
    agent1: Tuple[np.ndarray, np.ndarray],
    agent2: Tuple[np.ndarray, np.ndarray],
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 120,
) -> Image.Image:
    """
    Plot a side-by-side comparison of two agents.
    
    Args:
        game: DifferentiableLotto instance
        agent1: First agent (p, v)
        agent2: Second agent (p, v)
        figsize: Figure size
        dpi: DPI
        
    Returns:
        PIL Image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Determine plot limits
    all_points = np.concatenate([game.customers, agent1[1], agent2[1]], axis=0)
    max_extent = np.abs(all_points).max()
    pad = 0.2 * max_extent
    lim = max_extent + pad
    
    for ax, (p, v), label, color in [(ax1, agent1, "Agent 1", "blue"), 
                                      (ax2, agent2, "Agent 2", "orange")]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        
        # Plot customers
        ax.scatter(game.customers[:, 0], game.customers[:, 1],
                  c='black', s=15, alpha=0.4, marker='o', zorder=1)
        
        # Plot servers with size proportional to mass
        server_sizes = p * 500
        ax.scatter(v[:, 0], v[:, 1], c=color, s=server_sizes,
                  alpha=0.8, edgecolors='black', linewidths=1.5,
                  zorder=3)
        
        # Show mass distribution as text
        for j in range(game.k):
            ax.text(v[j, 0], v[j, 1], f'{p[j]:.2f}',
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_title(f"{label}\nMass: {p}, Width: {game._compute_width(p, v):.3f}")
    
    fig.suptitle("Agent Comparison", fontsize=14, fontweight='bold')
    
    # Render to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    
    buf.seek(0)
    img = Image.open(buf).convert("RGBA").copy()
    buf.close()
    return img
