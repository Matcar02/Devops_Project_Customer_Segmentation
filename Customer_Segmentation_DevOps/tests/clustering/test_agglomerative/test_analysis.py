import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from unittest.mock import MagicMock, patch
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)

from src.clustering.agglomerative.analysis import dendrogram

def test_dendrogram(caplog):
    caplog.set_level(logging.INFO)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    with patch('matplotlib.pyplot.show') as mock_show:
        dendrogram(X)

    mock_show.assert_called_once()

    assert 'Starting Dendrogram generation' in caplog.text
    assert 'Dendrogram generation completed' in caplog.text