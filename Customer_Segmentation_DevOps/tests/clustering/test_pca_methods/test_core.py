import pytest
from unittest.mock import patch
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import logging
import numpy as np
import sys, os, logging

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)

from src.clustering.pca_methods.core import pca_kmeans, pca_components


@pytest.fixture
def sc_features_and_scores():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
    sc_features = pd.DataFrame(X, columns=[f'feature{i}' for i in range(X.shape[1])])
    
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X)

    return sc_features, scores

@pytest.fixture
def rfm_dataset():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
    rfmcopy = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary'])
    return rfmcopy

@patch('matplotlib.pyplot.show')
def test_pca_kmeans(mock_show, sc_features_and_scores, caplog):
    sc_features, scores = sc_features_and_scores
    nclusterspca = 3
    caplog.set_level(logging.INFO)

    segmkmeans, kmeanspca = pca_kmeans(sc_features, scores, nclusterspca)

    assert mock_show.called, "plt.show() should be called"
    assert 'Starting PCA and K-Means clustering' in caplog.text
    assert 'PCA and K-Means clustering completed' in caplog.text
    assert 'kmeansclusters' in segmkmeans.columns, "segmkmeans should have a 'kmeansclusters' column"
    assert len(segmkmeans) == len(sc_features), "segmkmeans should have the same number of rows as sc_features"

@patch('matplotlib.pyplot.show')
def test_pca_kmeans(mock_show, sc_features_and_scores, caplog):
    sc_features, scores = sc_features_and_scores
    nclusterspca = 3
    caplog.set_level(logging.INFO)

    segmkmeans, kmeanspca = pca_kmeans(sc_features, scores, nclusterspca)


    assert 'Starting PCA and K-Means clustering' in caplog.text
    assert 'PCA and K-Means clustering completed' in caplog.text
    assert 'kmeansclusters' in segmkmeans.columns, "segmkmeans should have a 'kmeansclusters' column"
    assert len(segmkmeans) == len(sc_features), "segmkmeans should have the same number of rows as sc_features"
