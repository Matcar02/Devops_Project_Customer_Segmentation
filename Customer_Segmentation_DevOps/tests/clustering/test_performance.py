import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys, os, logging

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.clustering.performance import silhouette_score_df


@pytest.fixture
def silscores_fixture():
    data = np.random.rand(10, 4) 
    columns = ['sil_score1', 'sil_score2', 'sil_score3', 'sil_score4']
    return pd.DataFrame(data, columns=columns)

@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
@patch('pandas.DataFrame.to_csv')
@patch('seaborn.displot')
@patch('matplotlib.pyplot.show')
def test_silhouette_score_df(mock_show, mock_displot, mock_to_csv, mock_path_exists, mock_makedirs, silscores_fixture, caplog):
    caplog.set_level(logging.INFO)
    
    result = silhouette_score_df(silscores_fixture)
    
    mock_to_csv.assert_called()
    
    mock_displot.assert_called()
    mock_show.assert_called()
    
    assert 'Getting DataFrame...' in caplog.text
    assert 'Saving DataFrame to CSV...' in caplog.text
    assert 'DataFrame saved successfully.' in caplog.text
    
    assert isinstance(result, pd.DataFrame), "The function should return a pandas DataFrame"

    mock_makedirs.assert_called_once()
