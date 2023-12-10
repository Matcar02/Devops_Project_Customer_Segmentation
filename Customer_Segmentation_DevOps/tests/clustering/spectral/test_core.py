import pytest
from unittest.mock import patch
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sys, os, logging

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)
from src.clustering.spectral.core import choose_spectral, spectral_clustering


@pytest.fixture
def sample_data():
    """
    Generate sample data for testing.
    """
    X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
    return X


@patch('builtins.input', side_effect=[4, 'nearest_neighbors'])
def test_choose_spectral(mock_input, caplog):
    """
    Test the choose_spectral function.
    """
    caplog.set_level(logging.INFO)

    n_neighbors, affinity = choose_spectral()

    assert n_neighbors == 4, "Should return the correct number of neighbors"
    assert affinity == 'nearest_neighbors', "Should return the correct affinity"
    assert 'Starting Spectral Clustering selection' in caplog.text

    assert 'Invalid affinity choice' not in caplog.text


def test_spectral_clustering(sample_data, caplog):
    caplog.set_level(logging.INFO)
    nclusters = 3
    affinity = 'nearest_neighbors'
    neighbors = 5

    labels, sil_score = spectral_clustering(sample_data, nclusters, affinity, neighbors)

    assert len(set(labels)) == nclusters, "Should produce the correct number of clusters"
    assert isinstance(sil_score, float), "Silhouette score should be a float"
    assert 0 <= sil_score <= 1, "Silhouette score should be between 0 and 1"
    assert 'Starting Spectral Clustering' in caplog.text
    assert 'Spectral Clustering completed' in caplog.text
