import os
import sys
import pandas as pd
import pytest
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.data_preparation.cleaning import prepare_data, drop_c_id, clean_data

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'order_id': [1, 2, 3],
        'customer_id': [101, 102, 103],
        'customer_unique_id': ['A1', 'B2', 'C3'],
        'order_status': ['delivered', 'canceled', 'delivered'],
        'product_name_lenght': [10, 20, 30],
        'product_description_lenght': [100, 200, 300],
        'shipping_limit_date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'product_category_name': ['books', 'electronics', 'clothing']
    })

@pytest.fixture
def tmp_csv_file(tmpdir, sample_dataframe):
    file_path = tmpdir.join("sample_data.csv")
    sample_dataframe.to_csv(file_path, index=False)
    return file_path

# Tests for prepare_data function

def test_prepare_data(tmp_csv_file):
    df = prepare_data(tmp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert 'product_name_lenght' not in df.columns
    assert 'product_description_lenght' not in df.columns
    assert 'shipping_limit_date' not in df.columns
    assert 'product_category_name' not in df.columns

def test_prepare_data_file_not_found():
    result = prepare_data('non_existent_file.csv')
    assert result is None

# Tests for drop_c_id function

def test_drop_c_id(sample_dataframe):
    df = drop_c_id(sample_dataframe)
    assert 'customer_unique_id' not in df.columns

def test_drop_c_id_without_column(sample_dataframe):
    sample_dataframe.drop('customer_unique_id', axis=1, inplace=True)
    df = drop_c_id(sample_dataframe)
    assert 'customer_unique_id' not in df.columns

# Tests for clean_data function

def test_clean_data(sample_dataframe):
    df = clean_data(sample_dataframe)
    assert isinstance(df, pd.DataFrame)
    assert all(df['order_status'] == 'delivered')


