import pytest
import pandas as pd
from sklearn.datasets import make_blobs
import sys, os, logging
from src.clustering.agglomerative.core import agglomerative_clustering


script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)


@pytest.fixture
def sample_data():
    """
    Generate sample data for testing agglomerative clustering.
    """
    X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
    rfmcopy = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary value'])
    n_clustersagg = 3
    return X, rfmcopy, n_clustersagg


def test_agglomerative_clustering(sample_data, caplog):
    """
    Test agglomerative clustering function.
    """
    X, rfmcopy, n_clustersagg = sample_data
    caplog.set_level(logging.INFO)
    y_hc = agglomerative_clustering(X, rfmcopy, n_clustersagg)

    assert len(y_hc) == len(X)
    assert set(y_hc) == set([0, 1, 2])

    assert 'Starting Agglomerative Clustering' in caplog.text
    assert 'Clustering completed' in caplog.text
    assert 'Agglomerative Clustering completed' in caplog.text

