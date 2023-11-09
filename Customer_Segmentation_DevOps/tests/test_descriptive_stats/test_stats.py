import pytest
from unittest.mock import patch
import pandas as pd
import logging
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.descriptive_stats.stats import describe_dataset, corr

@pytest.fixture
def sample_rfm_dataset():
    return pd.DataFrame({
        'Monetary value': [100, 200, 300],
        'Recency': [10, 20, 30],
        'Frequency': [1, 2, 3],
    })

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'payment_type': [0, 3, 5],
        'payment_installments': [1, 2, 3],
        'payment_value': [100, 200, 300],
    })

def test_describe_dataset(sample_rfm_dataset, caplog):
    caplog.set_level(logging.INFO)
    describe_dataset(sample_rfm_dataset)

    assert 'Starting describe_dataset function...' in caplog.text
    assert 'describe_dataset function completed.' in caplog.text

def test_corr(sample_df, caplog):
    caplog.set_level(logging.INFO)
    with patch('seaborn.pairplot'):
        corr(sample_df)

    assert 'Starting corr function...' in caplog.text
    assert 'Correlation matrix generated for columns: payment_type, payment_installments, payment_value' in caplog.text
    assert 'corr function completed.' in caplog.text
