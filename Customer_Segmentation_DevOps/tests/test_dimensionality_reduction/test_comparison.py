import pytest
import os
import sys
import pandas as pd
import logging
from unittest.mock import MagicMock, patch
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.dimensionality_reduction.comparison import pca_vs_spectral


@pytest.fixture
def sample_dfpcaf():
    # Create a sample DataFrame that mimics the structure expected by pca_vs_spectral
    return pd.DataFrame({
        'kmeansclustersPCA': [0, 0, 1, 1, 2],
        'sp_clusters': [2, 3, 2, 3, 2],
        'payment_type': ['credit_card', 'credit_card', 'voucher', 'voucher', 'boleto'],
        'customer_state': ['SP', 'RJ', 'SP', 'RJ', 'SP'],
        'product_category_name_english': ['electronics', 'housewares', 'electronics', 'housewares', 'electronics'],
    })

def test_pca_vs_spectral(sample_dfpcaf, caplog):
    caplog.set_level(logging.INFO)
    with patch('matplotlib.pyplot.show') as mock_show:
        pca_vs_spectral(sample_dfpcaf, ['payment_type', 'customer_state', 'product_category_name_english'])

    mock_show.assert_called_once()

    assert 'Starting pca_vs_spectral function...' in caplog.text
    assert 'Finished pca_vs_spectral function and plotted the visualizations.' in caplog.text

    assert 'Input DataFrame' not in caplog.text 
