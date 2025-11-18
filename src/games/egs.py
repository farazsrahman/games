"""
Empirical Gamescape utility class for analyzing empirical gamescape matrices
as described in "Open-ended Learning in Symmetric Zero-sum Games" (arXiv:1901.08106).
"""
import numpy as np
from scipy.linalg import schur
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


class EmpiricalGS:
    
    def __init__(self, egs_matrix: np.ndarray):
        # Assert that the egs matrix is square, antisymmetric and 2d
        assert egs_matrix.ndim == 2, "Matrix must be 2D"
        assert egs_matrix.shape[0] == egs_matrix.shape[1], "Matrix must be square"
        assert np.allclose(egs_matrix, -egs_matrix.T), "Matrix must be antisymmetric"
        
        self.egs_matrix = egs_matrix
    
    def schur_embeddings(self) -> np.ndarray:
        T, Z = schur(self.egs_matrix)
        # Extract first two columns of Q (which is Z in scipy's schur)
        embeddings = Z[:, :2]
        return embeddings.real if np.iscomplexobj(embeddings) else embeddings
    
    def PCA_embeddings(self) -> np.ndarray:
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(self.egs_matrix)
        return embeddings
    
    def SVD_embeddings(self) -> np.ndarray:
        svd = TruncatedSVD(n_components=2)
        embeddings = svd.fit_transform(self.egs_matrix)
        return embeddings
    
    def tSNE_embeddings(self) -> np.ndarray:
        n_samples = self.egs_matrix.shape[0]
        # Perplexity must be less than n_samples
        perplexity = min(9, max(1, n_samples - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings = tsne.fit_transform(self.egs_matrix)
        return embeddings
    
    def _embedding_convex_hull_area(self, embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 3:
            return 0.0 
        hull = ConvexHull(embeddings)
        return hull.volume


def visualize_egs_matrix_and_embeddings(
    egs: EmpiricalGS,
    embeddings: np.ndarray,
    save_path: str = None
):
    """
    Visualize empirical gamescape matrix and embeddings side-by-side.
    
    Args:
        egs: EmpiricalGS instance
        embeddings: numpy array of shape (N, 2) containing embedding points
        save_path: optional path to save the figure. If None, displays the plot.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Matrix heatmap with green (positive) and red (negative)
    matrix = egs.egs_matrix
    # Create a diverging colormap: red for negative, green for positive
    # Center colormap at 0
    vmax = np.abs(matrix).max()
    im = ax1.imshow(matrix, cmap='RdYlGn', aspect='auto', 
                    vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax1.set_title('Empirical Gamescape Matrix\n(Green: Positive, Red: Negative)')
    ax1.set_xlabel('Agent j')
    ax1.set_ylabel('Agent i')
    cbar1 = fig.colorbar(im, ax=ax1)
    cbar1.set_label('Payoff')
    
    # Right plot: 2D embeddings colored by row average with convex hull
    row_averages = np.mean(egs.egs_matrix, axis=1)
    
    # Compute convex hull and area
    hull_area = egs._embedding_convex_hull_area(embeddings)
    
    # Draw convex hull if we have enough points
    if embeddings.shape[0] >= 3:
        hull = ConvexHull(embeddings)
        # Get hull vertices in order
        hull_vertices = embeddings[hull.vertices]
        # Create polygon for shading
        hull_polygon = Polygon(hull_vertices, closed=True, 
                              facecolor='lightblue', edgecolor='steelblue', 
                              linewidth=2, alpha=0.3, zorder=0)
        ax2.add_patch(hull_polygon)
        # Draw hull edges
        for simplex in hull.simplices:
            ax2.plot(embeddings[simplex, 0], embeddings[simplex, 1], 
                    'steelblue', linewidth=1.5, alpha=0.6, zorder=1)
    
    scatter = ax2.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=row_averages, cmap='viridis', s=100, alpha=0.8, zorder=2)
    ax2.set_title(f'2D Embeddings\n(Colored by Row Average)\nConvex Hull Area: {hull_area:.4f}')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    cbar2 = fig.colorbar(scatter, ax=ax2)
    cbar2.set_label('Row Average Payoff')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

if __name__ == "__main__":

    N = 20


    r = np.sort(np.random.randn(N))[::-1]  # shape (20,), sorted from greatest to least
    P_trans = r[:, None] - r[None, :]   # shape (20, 20)

    P_cycle = np.zeros((N, N))
    cycle_len = 5
    for i in range(N):
        j_next = (i + 1) % cycle_len  # player i beats j_next
        P_cycle[i, j_next] = 1
        P_cycle[j_next, i] = -1         # antisymmetry: j_next loses to i


    E       = np.random.randn(N, N)
    P_rand  = (E - E.T)/2

    th   = 0.7

    P = (1.0 - th) * P_rand + th * P_cycle


    gamescape = EmpiricalGS(P)
    
    visualize_egs_matrix_and_embeddings(gamescape, gamescape.schur_embeddings())
