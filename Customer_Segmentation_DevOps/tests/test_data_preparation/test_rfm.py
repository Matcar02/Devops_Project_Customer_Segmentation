import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch
import logging
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.data_preparation.rfm import get_frequencies, get_recency, get_monetary, concatenate_dataframes_, get_rfm_dataset 
import glob
import os

# Sample data
@pytest.fixture
def sample_rfm_dataframe():
    today = datetime.today()
    return pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 3],
        'order_delivered_customer_date': [today, today - timedelta(days=10), today - timedelta(days=20), today, today],
        'order_purchase_timestamp': [today - timedelta(days=30), today - timedelta(days=40), today - timedelta(days=50), today - timedelta(days=60), today],
        'payment_value': [100, 150, 200, 250, 300]
    })

# Test for get_frequencies function
def test_get_frequencies(sample_rfm_dataframe):
    frequencies = get_frequencies(sample_rfm_dataframe)
    assert isinstance(frequencies, pd.DataFrame)
    assert frequencies['Frequency'].sum() == 5 

# Test for get_recency function
def test_get_recency(sample_rfm_dataframe):
    recency = get_recency(sample_rfm_dataframe)
    assert isinstance(recency, pd.DataFrame)


# Test for get_monetary function
def test_get_monetary(sample_rfm_dataframe):
    monetary = get_monetary(sample_rfm_dataframe)
    assert isinstance(monetary, pd.DataFrame)


# Test for concatenate_dataframes_ function
def test_concatenate_dataframes_(sample_rfm_dataframe):
    frequencies = get_frequencies(sample_rfm_dataframe)
    recency = get_recency(sample_rfm_dataframe)
    monetary = get_monetary(sample_rfm_dataframe)
    rfm_dataset = concatenate_dataframes_(recency, monetary, frequencies)
    assert isinstance(rfm_dataset, pd.DataFrame)




def test_get_rfm_dataset(sample_rfm_dataframe, tmpdir, caplog):
    # Set up a temporary directory for the reports
    reports_path = tmpdir.mkdir("reports").join("dataframes")

    # Check if the directory already exists
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    # Call the actual function to test
    returned_df = get_rfm_dataset(sample_rfm_dataframe)

    print("File path:", reports_path)

    # Assert that the returned object is a DataFrame
    assert isinstance(returned_df, pd.DataFrame), "The function should return a DataFrame."

    # Save the DataFrame as a CSV file
    csv_file_path = os.path.join(reports_path, "rfm_dataset.csv")
    returned_df.to_csv(csv_file_path, index=False)

    # Assert that the CSV file was saved correctly
    assert os.path.isfile(csv_file_path), "The CSV file was not saved correctly."

    # If there was an error logged, let's assert that it shouldn't happen
    for record in caplog.records:
        assert record.levelname != 'ERROR', f"Error in log: {record.message}"
