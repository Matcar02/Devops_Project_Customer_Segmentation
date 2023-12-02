import pandas as pd
import os
import sys
import pytest
from sklearn.exceptions import NotFittedError

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.data_preparation.encoding import encode_df, get_dummies_df
@pytest.fixture
def sample_encoding_dataframe():
    return pd.DataFrame({
        'order_status': ['delivered', 'canceled', 'delivered'],
        'payment_type': ['credit_card', 'boleto', 'voucher'],
        'customer_city': ['city_a', 'city_b', 'city_c'],
        'customer_state': ['state_a', 'state_b', 'state_c'],
        'seller_city': ['city_d', 'city_e', 'city_f'],
        'seller_state': ['state_d', 'state_e', 'state_f'],
        'product_category_name_english': ['books', 'electronics', 'clothing']
    })

# Tests for encode_df function

def test_encode_df(sample_encoding_dataframe):
    encoded_df = encode_df(sample_encoding_dataframe)
    # Check if the function returns a DataFrame
    assert isinstance(encoded_df, pd.DataFrame)

    expected_column_prefixes = [
        'onehotencoder__order_status',
        'onehotencoder__payment_type',  
        'onehotencoder__customer_city',
        'onehotencoder__customer_state',
        'onehotencoder__seller_city',
        'onehotencoder__seller_state',
        'onehotencoder__product_category_name_english',
    ]
    # Check if the expected columns with the correct prefixes exist
    for prefix in expected_column_prefixes:
        assert any(col.startswith(prefix) for col in encoded_df.columns)

def test_encode_df_exception(sample_encoding_dataframe):
    # Remove a column to test if the function handles exceptions
    sample_encoding_dataframe.drop(columns=['order_status'], inplace=True)
    with pytest.raises(ValueError):
        encode_df(sample_encoding_dataframe)

# Tests for get_dummies_df function

def test_get_dummies_df(sample_encoding_dataframe):
    dummies_df = get_dummies_df(sample_encoding_dataframe)
    # Check if the function returns a DataFrame
    assert isinstance(dummies_df, pd.DataFrame)
    # Check if the function adds new columns for dummy variables
    assert 'order_status_delivered' in dummies_df.columns
