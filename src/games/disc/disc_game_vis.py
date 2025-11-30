# Visualization functions for disc_game
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
import io
from typing import List, Optional, Tuple, Dict

from games.egs import EmpiricalGS, visualize_egs_matrix_and_embeddings

def plot_image(
    vs: List[np.ndarray],
    dvs: Optional[List[np.ndarray]] = None,
    figsize: Tuple[float, float] = (4, 4),
    dpi: int = 120,
) -> Image.Image:
    """
    Plot a list of points `vs` and, optionally, a list of vectors `dvs` at each point.
    The points are colored with a gradient to show ordering (earliest to latest).
    Returns a lightweight PIL.Image (not a Matplotlib Figure).
    
    Args:
        vs: List of 2D points (np.ndarray with shape (2,)).
        dvs: Optional list of 2D vectors (same length as vs or None). If provided, draws arrow from vs[i] to vs[i]+dvs[i].
        figsize: Figure size.
        dpi: DPI.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")

    # Axes & unit circle
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.add_patch(plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--", alpha=0.7))

    vs_array = np.array(vs)
    N = len(vs_array)

    # Gradient coloring
    cmap = plt.get_cmap('viridis', N)
    norm = mcolors.Normalize(vmin=0, vmax=N-1)
    colors = [cmap(norm(i)) for i in range(N)]

    # Scatter with gradient color; create a colorbar
    sc = ax.scatter(vs_array[:, 0], vs_array[:, 1], color=colors, s=60, zorder=4, label='points (earlier → later)')

    # Plot descent direction arrows, if provided
    if dvs is not None:
        dvs_array = np.array(dvs)
        # If some entries might be None, handle that
        for idx, (v, dv) in enumerate(zip(vs_array, dvs_array)):
            if dv is not None:
                ax.quiver(v[0], v[1], dv[0], dv[1],
                          angles='xy', scale_units='xy', scale=1,
                          color=colors[idx], width=0.005, alpha=0.8, zorder=5)

    # Add colorbar for the point index gradient
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for colorbar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Point Index (earlier → later)')

    ax.set_title(f"{len(vs)} points" if len(vs) != 2 else f"v = {vs[0]}, w = {vs[1]}")
    ax.legend(loc="upper left")

    # Render to PNG in-memory, then close fig to free memory
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGBA").copy()  # copy() to detach from buffer
    buf.close()
    return img


def plot_timeseries(values: List[float], title: str = "Time Series", ylabel: str = "Value"):
    """
    Plot a simple time series of values.
    Args:
        values: List or array of numeric values.
        title: Plot title.
        ylabel: Label for the y-axis.
    """
    values = np.array(values)
    plt.figure(figsize=(6, 3))
    plt.plot(values, lw=1.8, color="blue")
    plt.title(title)
    plt.xlabel("timestep")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_positive_negative_ratio(values: List[float], title: str = "Positive vs. Negative Frequency"):
    """
    Plot the percentage of times the series is positive vs. negative (and zero if any).

    Args:
        values: List or array of numeric values.
        title: Title for the plot.
    """
    values = np.array(values)
    n = len(values)
    if n == 0:
        raise ValueError("Input series is empty.")

    pos_frac = np.sum(values > 0) / n * 100
    neg_frac = np.sum(values < 0) / n * 100
    zero_frac = np.sum(values == 0) / n * 100

    labels = ["Positive", "Negative"]
    fracs = [pos_frac, neg_frac]
    colors = ["green", "red"]

    # include zero only if it exists
    if zero_frac > 0:
        labels.append("Zero")
        fracs.append(zero_frac)
        colors.append("gray")

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        fracs,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"color": "w"},
    )
    ax.set_title(title)
    plt.show()

    return {"positive_%": pos_frac, "negative_%": neg_frac, "zero_%": zero_frac}


def images_to_gif(images: List[Image.Image], path: str = "animation.gif", duration: int = 120, loop: int = 0) -> str:
    """
    Turn a list of PIL.Image into a GIF. Converts frames to an adaptive palette
    for smaller file size.
    """
    if not images:
        raise ValueError("No images provided.")

    # Convert to palette mode for GIF
    frames = [im.convert("P", palette=Image.ADAPTIVE) for im in images]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    return path


def sample_unit_hypersphere(num_samples, dim=2):
    """
    Samples points uniformly from inside the unit hypersphere in given dimension.

    Args:
        num_samples (int): The number of points to sample.
        dim (int): The dimension of the hypersphere (default is 2).

    Returns:
        numpy.ndarray: An array of shape (num_samples, dim) containing 
                       the coordinates of the sampled points.
    """
    # Step 1: Sample points from standard normal distribution for direction
    X = np.random.normal(size=(num_samples, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # Normalize to get directions

    # Step 2: Sample radius so that points are uniform in the volume
    U = np.random.uniform(0, 1, size=(num_samples, 1))
    R = U**(1.0/dim)  # Proper scaling for uniform volume in d dimensions

    X *= R  # Scale directions by radii
    return X


def gif_from_states(
    vs,
    ws,
    dvs=None,
    dws=None,
    A=None,                  # if provided, will show v^T A w in the title
    path="descent.gif",
    fps=20,
    stride=1,                # render every k-th frame
    dpi=120,
    unit_circle=True,
    tight=True,
):
    """
    Create a GIF from sequences of (v, w) and optional (dv, dw) using a single
    Matplotlib figure with in-place artist updates (no per-frame images).

    Args:
        vs, ws: array-like of shape (T, 2). T timesteps of vectors.
        dvs, dws: optional array-like of shape (T, 2) for descent directions.
                  If provided, dv is drawn starting at v, and dw at w.
        A: optional 2x2 ndarray. If provided, title shows v^T A w each frame.
        path: output GIF path.
        fps: frames per second of the GIF.
        stride: render every k-th frame (e.g., 5 renders 0,5,10,...).
        dpi: figure DPI.
        unit_circle: draw unit circle.
        tight: call tight_layout() once.
    Returns:
        The path to the saved GIF.
    """
    vs = np.asarray(vs, dtype=float)
    ws = np.asarray(ws, dtype=float)
    assert vs.shape == ws.shape and vs.ndim == 2 and vs.shape[1] == 2, "vs/ws must be (T,2)"

    T = vs.shape[0]
    idx = np.arange(0, T, stride)
    vs_s, ws_s = vs[idx], ws[idx]
    dvs_s = np.asarray(dvs, dtype=float)[idx] if dvs is not None else None
    dws_s = np.asarray(dws, dtype=float)[idx] if dws is not None else None

    # Determine plot limits from data (fallback to [-1.2, 1.2] if too small)
    max_extent = 1.0
    all_pts = np.concatenate([vs, ws], axis=0)
    if all_pts.size > 0:
        max_extent = max(max(1.0, np.abs(all_pts).max()), 1.0)
    pad = 0.2 * max_extent
    lim = max_extent + pad

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, lw=0.5)
    ax.axvline(0, lw=0.5)
    if unit_circle:
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, linestyle="--", alpha=0.7))

    # Plot handle initialization for points and black quivers
    v0, w0 = vs_s[0], ws_s[0]

    sc_v = ax.scatter([v0[0]], [v0[1]], c='blue', s=60, zorder=4, label='v')
    sc_w = ax.scatter([w0[0]], [w0[1]], c='orange', s=60, zorder=4, label='w')

    qdv = qdw = None
    if dvs_s is not None:
        dv0 = dvs_s[0]
        qdv = ax.quiver([v0[0]], [v0[1]], [dv0[0]], [dv0[1]],
                        angles="xy", scale_units="xy", scale=1, color="black", linewidth=1.4)
    if dws_s is not None:
        dw0 = dws_s[0]
        qdw = ax.quiver([w0[0]], [w0[1]], [dw0[0]], [dw0[1]],
                        angles="xy", scale_units="xy", scale=1, color="black", linewidth=1.4)

    title = ax.set_title("")
    if tight:
        fig.tight_layout()

    # Convenience setters for Quiver artists and scatter points
    def set_scatter(sc, x, y):
        sc.set_offsets([[x, y]])

    def set_quiver_UV(q, u, v):
        # Update vector components
        q.set_UVC([u], [v])

    def set_quiver_pos(q, x, y):
        # Update arrow base (starting point)
        q.set_offsets(np.c_[ [x], [y] ])

    def frame(i):
        v, w = vs_s[i], ws_s[i]
        # Update points for v and w
        set_scatter(sc_v, v[0], v[1])
        set_scatter(sc_w, w[0], w[1])

        artists = [sc_v, sc_w, title]
        # Update descent directions if provided (anchored at v / w)
        if qdv is not None:
            dv = dvs_s[i]
            set_quiver_pos(qdv, v[0], v[1])
            set_quiver_UV(qdv, dv[0], dv[1])
            artists.append(qdv)
        if qdw is not None:
            dw = dws_s[i]
            set_quiver_pos(qdw, w[0], w[1])
            set_quiver_UV(qdw, dw[0], dw[1])
            artists.append(qdw)

        if A is not None:
            title.set_text(f"vᵀAw = {float(v.T @ A @ w):.3f}")
        else:
            title.set_text("")
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig, frame, frames=len(vs_s), interval=1000.0 / fps, blit=True
    )

    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    return path


def plot_all_egs_visualizations(
    game,
    agents: List[np.ndarray],
    output_dir: str,
    base_name: str = "disc_egs",
    n_rounds: int = 1000,
    dpi: int = 150
) -> Dict[str, str]:
    """
    Generate all EGS visualizations: matrix + PCA, Schur, SVD, and t-SNE embeddings
    for the Disc Game.

    Args:
        game: DiscGame instance
        agents: List of 2D agent vectors (points on the disc)
        output_dir: Directory to save the plots
        base_name: Base name for output files
        n_rounds: Unused for DiscGame (kept for API compatibility)
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

    # Compute payoff matrix (values in [-1, 1] from DiscGame.play)
    # Note: DiscGame.play returns u^T A v > 0 ? 1 : -1, which is already antisymmetric
    # since u^T A v = -v^T A u (A is antisymmetric)
    payoff_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                payoff_matrix[i, j] = 0.0  # self-play treated as tie
            else:
                payoff = game.play(agents[i], agents[j])
                payoff_matrix[i, j] = payoff

    # Convert to "win rate" style matrix in [0, 1] for consistency with other games
    winrate_matrix = (payoff_matrix + 1.0) / 2.0

    # Convert to antisymmetric EGS matrix (centered at 0, zero-sum)
    # Since winrate[i,j] + winrate[j,i] = 1, we have:
    # egs[i,j] = winrate[i,j] - 0.5, which gives egs[i,j] + egs[j,i] = 0 (antisymmetric)
    # So we can directly construct the antisymmetric matrix
    egs_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            # Convert winrate to centered value: winrate - 0.5 is in [-0.5, 0.5]
            val = winrate_matrix[i, j] - 0.5
            egs_matrix[i, j] = val
            egs_matrix[j, i] = -val  # Ensure antisymmetry

    # Create EmpiricalGS instance
    try:
        gamescape = EmpiricalGS(egs_matrix)
    except AssertionError as e:
        # If assertion fails, print debug info and re-raise
        print(f"Warning: EGS matrix assertion failed: {e}")
        print(f"Matrix shape: {egs_matrix.shape}")
        print(f"Is antisymmetric? {np.allclose(egs_matrix, -egs_matrix.T)}")
        print(f"Matrix:\n{egs_matrix}")
        raise

    # Generate all embedding methods
    methods = ["PCA", "SVD", "schur", "tSNE"]
    output_paths: Dict[str, str] = {}

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



def gif_from_population(
    us: np.ndarray,
    path: str = "population.gif",
    fps: int = 20,
    stride: int = 1,
    dpi: int = 120,
    unit_circle: bool = True,
    tight: bool = True,
    agent_colors: Optional[List] = None,
    show_labels: bool = True,
    normalize_difference_vector: Optional[float] = None,
) -> str:
    """
    Create a GIF from a population of agents evolving over time.

    Args:
        us: array-like of shape (T, N, 2). T timesteps, N agents, 2D coordinates.
        path: output GIF path.
        fps: frames per second of the GIF.
        stride: render every k-th frame (e.g., 5 renders 0,5,10,...).
        dpi: figure DPI.
        unit_circle: draw unit circle.
        tight: call tight_layout() once.
        agent_colors: optional list of N colors (one per agent). If None, uses default colormap.
        show_labels: whether to show legend with agent labels.
        normalize_difference_vector: if not None, rescales small difference vectors to this magnitude for visibility.
    Returns:
        The path to the saved GIF.
    """
    us = np.asarray(us, dtype=float)
    assert us.ndim == 3 and us.shape[2] == 2, "us must be (T, N, 2)"

    T, N = us.shape[0], us.shape[1]

    # Always compute dus as first-order difference
    dus = np.diff(us, axis=0)  # Shape: (T-1, N, 2)
    # Pad with zeros for the last timestep to match us shape
    dus = np.concatenate([dus, np.zeros((1, N, 2))], axis=0)  # Shape: (T, N, 2)

    # If normalizing difference vectors, rescale each vector to the desired magnitude (if it's nonzero)
    if normalize_difference_vector is not None:
        mag = float(normalize_difference_vector)
        norm_dus = np.linalg.norm(dus, axis=2, keepdims=True)
        # To avoid division by zero, only scale nonzero vectors
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.ones_like(norm_dus)
            nonzero = norm_dus > 1e-8
            scale[nonzero] = mag / norm_dus[nonzero]
            dus = dus * scale

    assert dus.shape == us.shape, f"dus shape {dus.shape} must match us shape {us.shape}"

    idx = np.arange(0, T, stride)
    us_s = us[idx]
    dus_s = dus[idx]

    # Determine plot limits from data
    max_extent = 1.0
    if us.size > 0:
        max_extent = max(max(1.0, np.abs(us).max()), 1.0)
    pad = 0.2 * max_extent
    lim = max_extent + pad

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, lw=0.5, color="gray", alpha=0.5)
    ax.axvline(0, lw=0.5, color="gray", alpha=0.5)
    if unit_circle:
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, linestyle="--", alpha=0.7, color="gray"))

    # Set up colors for agents
    if agent_colors is None:
        cmap = cm.get_cmap('tab10')
        agent_colors = [cmap(i % 10) for i in range(N)]
    elif len(agent_colors) != N:
        # If not enough colors provided, cycle through available ones
        cmap = cm.get_cmap('tab10')
        agent_colors = [agent_colors[i % len(agent_colors)] if agent_colors else cmap(i % 10) 
                       for i in range(N)]

    # Initialize scatter plots for each agent
    scatters = []
    for i in range(N):
        u0 = us_s[0, i]
        sc = ax.scatter([u0[0]], [u0[1]], c=[agent_colors[i]], s=80,
                       zorder=4, label=f'Agent {i}', alpha=0.8)
        scatters.append(sc)

    # Initialize quivers for descent directions
    quivers = []
    for i in range(N):
        u0 = us_s[0, i]
        du0 = dus_s[0, i]
        q = ax.quiver([u0[0]], [u0[1]], [du0[0]], [du0[1]],
                     angles="xy", scale_units="xy", scale=1,
                     color=agent_colors[i], linewidth=1.2, alpha=0.7,
                     width=0.004, zorder=3)
        quivers.append(q)

    title = ax.set_title("")
    if show_labels and N <= 10:  # Only show legend if not too many agents
        ax.legend(loc="upper left", fontsize=8)
    if tight:
        fig.tight_layout()

    def frame(i):
        # Update all agents
        for j in range(N):
            u = us_s[i, j]
            scatters[j].set_offsets([[u[0], u[1]]])

        # Update descent directions
        for j in range(N):
            u = us_s[i, j]
            du = dus_s[i, j]
            quivers[j].set_offsets(np.c_[[u[0]], [u[1]]])
            quivers[j].set_UVC([du[0]], [du[1]])

        # Update title (no statistics)
        title.set_text(f"t={idx[i]}")

        artists = scatters + quivers + [title]
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig, frame, frames=len(us_s), interval=1000.0 / fps, blit=True
    )

    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
    return path

