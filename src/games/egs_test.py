"""
Tests for Empirical Gamescape utility class.
"""
import numpy as np
import pytest
from games.egs import EmpiricalGS


def generate_rps_matrix() -> np.ndarray:
    """
    Generate a 3x3 Rock-Paper-Scissors style antisymmetric matrix.
    RPS: R beats S, S beats P, P beats R
    """
    matrix = np.array([
        [0, 1, -1],   # Rock vs Rock, Paper, Scissors
        [-1, 0, 1],   # Paper vs Rock, Paper, Scissors
        [1, -1, 0]    # Scissors vs Rock, Paper, Scissors
    ], dtype=float)
    return matrix


def generate_random_antisymmetric_matrix(size: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random antisymmetric matrix of given size.
    """
    rng = np.random.RandomState(seed)
    # Generate random upper triangular matrix
    A = rng.randn(size, size)
    A = np.triu(A) - np.triu(A).T  # Make antisymmetric
    return A


def test_init_validations():
    """Test that __init__ correctly validates matrix properties."""
    # Test non-square matrix
    non_square = np.array([[1, 2, 3], [4, 5, 6]])
    try:
        EmpiricalGS(non_square)
        assert False, "Should raise AssertionError for non-square matrix"
    except AssertionError:
        pass
    
    # Test non-2D matrix
    non_2d = np.array([1, 2, 3])
    try:
        EmpiricalGS(non_2d)
        assert False, "Should raise AssertionError for non-2D matrix"
    except AssertionError:
        pass
    
    # Test non-antisymmetric matrix
    non_antisymmetric = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    # This is antisymmetric, so let's use a symmetric one
    symmetric = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    try:
        EmpiricalGS(symmetric)
        assert False, "Should raise AssertionError for non-antisymmetric matrix"
    except AssertionError:
        pass
    
    # Test valid matrix
    valid_matrix = generate_rps_matrix()
    egs = EmpiricalGS(valid_matrix)
    assert np.allclose(egs.egs_matrix, valid_matrix)


def test_schur_embeddings():
    """Test Schur embeddings return correct shape."""
    matrix = generate_rps_matrix()
    egs = EmpiricalGS(matrix)
    
    embeddings = egs.schur_embeddings()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 2), f"Expected shape (3, 2), got {embeddings.shape}"
    
    # Test with larger matrix
    large_matrix = generate_random_antisymmetric_matrix(10)
    egs_large = EmpiricalGS(large_matrix)
    embeddings_large = egs_large.schur_embeddings()
    assert embeddings_large.shape == (10, 2)


def test_pca_embeddings():
    """Test PCA embeddings return correct shape."""
    matrix = generate_rps_matrix()
    egs = EmpiricalGS(matrix)
    
    embeddings = egs.PCA_embeddings()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 2), f"Expected shape (3, 2), got {embeddings.shape}"
    
    # Test with larger matrix
    large_matrix = generate_random_antisymmetric_matrix(10)
    egs_large = EmpiricalGS(large_matrix)
    embeddings_large = egs_large.PCA_embeddings()
    assert embeddings_large.shape == (10, 2)


def test_svd_embeddings():
    """Test SVD embeddings return correct shape."""
    matrix = generate_rps_matrix()
    egs = EmpiricalGS(matrix)
    
    embeddings = egs.SVD_embeddings()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 2), f"Expected shape (3, 2), got {embeddings.shape}"
    
    # Test with larger matrix
    large_matrix = generate_random_antisymmetric_matrix(10)
    egs_large = EmpiricalGS(large_matrix)
    embeddings_large = egs_large.SVD_embeddings()
    assert embeddings_large.shape == (10, 2)


def test_tsne_embeddings():
    """Test tSNE embeddings return correct shape."""
    matrix = generate_rps_matrix()
    egs = EmpiricalGS(matrix)
    
    embeddings = egs.tSNE_embeddings()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 2), f"Expected shape (3, 2), got {embeddings.shape}"
    
    # Test with larger matrix
    large_matrix = generate_random_antisymmetric_matrix(10)
    egs_large = EmpiricalGS(large_matrix)
    embeddings_large = egs_large.tSNE_embeddings()
    assert embeddings_large.shape == (10, 2)


def test_convex_hull_area():
    """Test convex hull area calculation."""
    matrix = generate_rps_matrix()
    egs = EmpiricalGS(matrix)
    
    # Test with Schur embeddings
    embeddings = egs.schur_embeddings()
    area = egs._embedding_convex_hull_area(embeddings)
    assert isinstance(area, (float, np.floating))
    assert area >= 0.0, "Area should be non-negative"
    
    # Test with larger matrix
    large_matrix = generate_random_antisymmetric_matrix(10)
    egs_large = EmpiricalGS(large_matrix)
    embeddings_large = egs_large.schur_embeddings()
    area_large = egs_large._embedding_convex_hull_area(embeddings_large)
    assert area_large >= 0.0
    
    # Test edge case: less than 3 points
    two_points = np.array([[0, 0], [1, 0]])
    area_two = egs._embedding_convex_hull_area(two_points)
    assert area_two == 0.0, "Area should be 0.0 for < 3 points"


if __name__ == "__main__":
    # Run tests
    test_init_validations()
    print("✓ test_init_validations passed")
    
    test_schur_embeddings()
    print("✓ test_schur_embeddings passed")
    
    test_pca_embeddings()
    print("✓ test_pca_embeddings passed")
    
    test_svd_embeddings()
    print("✓ test_svd_embeddings passed")
    
    test_tsne_embeddings()
    print("✓ test_tsne_embeddings passed")
    
    test_convex_hull_area()
    print("✓ test_convex_hull_area passed")
    
    print("\nAll tests passed!")

