import pytest
from unittest.mock import patch
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import logging
import sys
import os 


current_path = os.getcwd()
src_dir = os.path.join(current_path, '..', '..', '..')
sys.path.append(src_dir)
from src.clustering.kmeans.analysis import elbow_method, get_best_kmeans_params, silhouette_score_f


# Fixture for generating the RFM dataset.
@pytest.fixture
def rfm_dataset():
    X, _ = make_blobs(n_samples=100, centers=5, n_features=3, random_state=42)
    rfm_dataset = pd.DataFrame(X, columns=['Recency', 'Monetary value', 'Frequency'])
    return rfm_dataset

# Fixture for generating cluster labels.
@pytest.fixture
def cluster_labels():
    return [0, 1, 2, 3, 4] * 20


@patch('matplotlib.pyplot.show')
def test_elbow_method(mock_show, rfm_dataset, caplog):
    """
    Test case for the elbow_method function.
    """
    caplog.set_level(logging.INFO)
    X, features = elbow_method(rfm_dataset)

    assert mock_show.called, "plt.show() should be called"
    assert 'Starting Elbow Method' in caplog.text
    assert 'Elbow Method completed' in caplog.text
    assert X.shape == (100, 3), "The function should return the correct shape of X"
    assert set(features) == {'Recency', 'Monetary value', 'Frequency'}, "Features should match"


def test_get_best_kmeans_params(rfm_dataset, caplog):
    """
    Test case for the get_best_kmeans_params function.
    """
    caplog.set_level(logging.INFO)
    features = ['Recency', 'Monetary value', 'Frequency']
    best_params = get_best_kmeans_params(rfm_dataset[features])

    assert 'Starting GridSearchCV for KMeans parameters' in caplog.text
    assert 'GridSearchCV for KMeans parameters completed' in caplog.text
    assert isinstance(best_params, dict), "The function should return a dictionary of best parameters"


@patch('src.clustering.kmeans.analysis.silhouette_score')
def test_silhouette_score_f(mock_silhouette_score, rfm_dataset, cluster_labels, caplog):
    """
    Test case for the silhouette_score_f function.
    """
    caplog.set_level(logging.INFO)
    features = ['Recency', 'Monetary value', 'Frequency']
    method = 'kmeans'
    rfm_dataset[method] = cluster_labels
    mock_silhouette_score.return_value = 0.5 

    silscores, silsc = silhouette_score_f(rfm_dataset[features], rfm_dataset, method)

    mock_silhouette_score.assert_called_once()
    assert 'Calculating Silhouette Score for kmeans' in caplog.text
    assert silscores[method] == 0.5, "The silhouette scores should match the mocked value"
    assert silsc == 0.5, "The silhouette score should match the mocked value"