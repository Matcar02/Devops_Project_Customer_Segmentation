import pytest
import os
import sys
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.dimensionality_reduction.pca import encoding_PCA, pca_preprocessing, pca_ncomponents

@pytest.fixture
def sample_df():
    # Create a sample DataFrame that mimics the structure expected by the functions
    return pd.DataFrame({
        'payment_type': ['credit_card', 'boleto', 'voucher'],
        'customer_city': ['city_a', 'city_b', 'city_c'],
        'product_category_name_english': ['electronics', 'housewares', 'books'],
        'payment_installments': [1, 2, 3],
    })

@pytest.fixture
def sample_rfm_dataset():
    # Create a DataFrame with 30 rows and more than 20 columns
    data = {
        'Monetary value': np.random.rand(30) * 100,
        'Recency': np.random.rand(30) * 100,
        'Frequency': np.random.randint(1, 5, size=30),
    }
    for i in range(17):
        data[f'feature_{i}'] = np.random.rand(30)
    return pd.DataFrame(data)

def test_encoding_PCA(sample_df, sample_rfm_dataset, caplog):
    caplog.set_level(logging.INFO)
    encoded_df, newdf = encoding_PCA(sample_df, sample_rfm_dataset)

    assert not encoded_df.empty, "Encoded DataFrame should not be empty."
    assert 'onehotencoder__payment_type_credit_card' in encoded_df.columns, "Expected encoded columns not found."
    assert 'Encoding and PCA transformation completed.' in caplog.text

def test_pca_preprocessing(sample_rfm_dataset, caplog):
    caplog.set_level(logging.INFO)
    sc_features = pca_preprocessing(sample_rfm_dataset)

    assert not sc_features.empty, "Scaled features DataFrame should not be empty."
    assert sc_features.isnull().sum().sum() == 0, "There should be no null values."
    assert 'PCA preprocessing completed.' in caplog.text

def test_pca_ncomponents(sample_rfm_dataset, caplog):
    caplog.set_level(logging.INFO)
    sc_features = pca_preprocessing(sample_rfm_dataset)

    with patch('matplotlib.pyplot.show'):
        X_ = pca_ncomponents(sc_features)

    assert X_ is not None, "The PCA components should not be None."
    assert 'Determined optimal number of PCA components.' in caplog.text

