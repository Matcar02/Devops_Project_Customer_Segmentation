import pytest
import pandas as pd
from sklearn.datasets import make_blobs
import sys, os, logging
from unittest.mock import patch 


script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)
from src.clustering.kmeans.core import clustering, choose


@pytest.fixture
def sample_data():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
    rfmcopy = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary value'])
    return X, rfmcopy

def test_clustering(sample_data, caplog):
    X, rfmcopy = sample_data
    caplog.set_level(logging.INFO)
    clusters1 = 3
    algorithm1 = 'lloyd'
    rand_state = 42
    y_kmeans = clustering(clusters1, algorithm1, rand_state, X, rfmcopy)

    assert 'kmeans_cluster' in y_kmeans.columns, "Output dataframe should have 'kmeans_cluster' column"
    assert len(y_kmeans) == len(X), "Output dataframe should have the same length as input data"
    assert set(y_kmeans['kmeans_cluster'].unique()) == set(range(clusters1)), "There should be `clusters1` unique clusters"

    assert 'Starting clustering' in caplog.text, "Log should contain 'Starting clustering'"
    assert 'Clustering completed' in caplog.text, "Log should contain 'Clustering completed'"

@patch('builtins.input', side_effect=[3, 'lloyd', 42])
def test_choose(mock_input, sample_data, caplog):
    X, rfmcopy = sample_data
    caplog.set_level(logging.INFO)
    result, selected_clusters = choose(rfmcopy, X)

    assert 'Starting cluster selection' in caplog.text, "Log should contain 'Starting cluster selection'"
    assert 'Cluster selection completed' in caplog.text, "Log should contain 'Cluster selection completed'"
    assert selected_clusters == 3, "The selected number of clusters should be 3"
    assert 'kmeans_cluster' in result.columns, "Output dataframe should have 'kmeans_cluster' column"
