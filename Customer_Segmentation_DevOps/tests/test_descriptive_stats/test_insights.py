import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

from src.descriptive_stats.insights import segments_insights, kmeans_summary, cluster_summary, installments_analysis, customers_insights, payments_insights, prod_insights, customer_geography, recency

@pytest.fixture
def rfmcopy_fixture():
    data = np.random.rand(1000, 3)
    rfmcopy = pd.DataFrame(data, columns=['Recency', 'Frequency', 'Monetary value'])
    rfmcopy['kmeans_cluster'] = np.random.randint(0, 4, size=1000)
    rfmcopy['sp_clusters'] = np.random.randint(0, 4, size=1000)
    rfmcopy['hc_clusters'] = np.random.randint(0, 4, size=1000)
    rfmcopy['payment_type'] = np.random.choice(['credit_card', 'voucher', 'debit_card', 'boleto'], size=1000)
    rfmcopy['payment_installments'] = np.random.randint(1, 10, size=1000)  
    return rfmcopy

@pytest.fixture
def df_fixture():
    data = {
        'customer_id': np.arange(1000),
        'payment_type': ['credit_card' if i % 2 == 0 else 'voucher' for i in range(1000)],
        'payment_installments': np.random.randint(1, 10, size=1000),
        'customer_state': np.random.choice(['state' + str(i % 20) for i in range(1000)], size=1000),
        'product_category_name_english': ['category' + str(i % 5) for i in range(1000)],
        'payment_value': np.random.rand(1000),
        'kmeans_cluster': np.random.choice([0, 1, 2, 3], size=1000),
        'hc_clusters': np.random.choice([0, 1, 2, 3], size=1000),
        'sp_clusters': np.random.choice([0, 1, 2, 3], size=1000)
    }
    df = pd.DataFrame(data)
    df['Monetary value'] = df['payment_value'] * 100 
    return df

@pytest.fixture
def recency_fixture():
    data = {'Recency': np.random.randint(0, 100, size=100)}
    return pd.DataFrame(data)

@patch('matplotlib.pyplot.show')
@patch('seaborn.displot')
def test_segments_insights(mock_displot, mock_show, rfmcopy_fixture, caplog):
    caplog.set_level(logging.INFO)

    segments_insights(rfmcopy_fixture, nclusterskmeans=4)

    assert mock_displot.call_count == 9
    assert mock_show.call_count == 3
    assert 'Starting analysis on the given rfmcopy data.' in caplog.text
    assert 'Finished analysis. Returning segmented data.' in caplog.text


@patch('pandas.DataFrame.to_csv')
def test_kmeans_summary(mock_to_csv, rfmcopy_fixture, caplog):
    caplog.set_level(logging.INFO)

    summary = kmeans_summary(rfmcopy_fixture, cluster_num=4)

    mock_to_csv.assert_called()
    
    assert 'Input data has 1000 records.' in caplog.text
    assert 'DataFrame saved successfully.' in caplog.text

    assert isinstance(summary, pd.DataFrame), "The function should return a pandas DataFrame"


@patch('pandas.DataFrame.to_csv')
def test_cluster_summary(mock_to_csv, df_fixture, caplog):
    caplog.set_level(logging.INFO)

    column_name = 'Monetary value'
    summary_kmeans, summary_hc, summary_sp = cluster_summary(df_fixture, column_name)

    assert mock_to_csv.call_count == 3

    assert f"Calculating summaries based on the column '{column_name}'." in caplog.text


    assert isinstance(summary_kmeans, pd.DataFrame), "The function should return a pandas DataFrame for kmeans"
    assert isinstance(summary_hc, pd.DataFrame), "The function should return a pandas DataFrame for hc"
    assert isinstance(summary_sp, pd.DataFrame), "The function should return a pandas DataFrame for sp"
    
@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
@patch('pandas.DataFrame.to_csv')
def test_installments_analysis(mock_to_csv, mock_path_exists, mock_makedirs, df_fixture, rfmcopy_fixture, caplog):
    caplog.set_level(logging.INFO)

    paydf = installments_analysis(df_fixture, rfmcopy_fixture)

    mock_to_csv.assert_called()

    assert 'Final \'paydf\' dataframe shape:' in caplog.text

    mock_makedirs.assert_called_once()

    assert not paydf.empty, "The function should not return an empty DataFrame"



@patch('matplotlib.pyplot.show')
@patch('seaborn.histplot')
def test_recency(mock_histplot, mock_show, recency_fixture, caplog):
    caplog.set_level(logging.INFO)

    recency(recency_fixture)

    assert mock_histplot.call_count == 1
    assert mock_show.call_count == 1
    assert 'Starting recency function...' in caplog.text
    assert 'Recency histogram displayed.' in caplog.text


@pytest.fixture
def payments_fixture():
    data = {
        'payment_type': ['credit_card', 'voucher', 'debit_card', 'boleto'] * 25, 
        'payment_value': np.random.rand(100),
        'payment_installments': np.random.randint(1, 10, size=100)
    }
    return pd.DataFrame(data)


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
@patch('seaborn.histplot')
@patch('seaborn.countplot')
def test_payments_insights(mock_countplot, mock_histplot, mock_close, mock_show, payments_fixture, caplog):
    caplog.set_level(logging.INFO)

    paymentdistr = payments_insights(payments_fixture)

    assert mock_histplot.call_count == 1
    assert mock_countplot.call_count == 1
    assert mock_show.call_count == 3
    assert 'Starting payments_insights function...' in caplog.text
    assert 'Payments insights displayed.' in caplog.text


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
def test_prod_insights(mock_savefig, mock_show, df_fixture, caplog):
    caplog.set_level(logging.INFO)

    prod_insights(df_fixture)

    mock_show.assert_called()
    mock_savefig.assert_called()

    assert 'Starting prod_insights function...' in caplog.text
    assert 'Product insights displayed.' in caplog.text

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
def test_customer_geography(mock_savefig, mock_show, df_fixture, caplog):
    caplog.set_level(logging.INFO)

    dfgeo = customer_geography(df_fixture)

    mock_show.assert_called()
    mock_savefig.assert_called()

    assert not dfgeo.empty, "The function should not return an empty DataFrame"

    assert 'Starting customer_geography function...' in caplog.text
    assert 'Customer geography plot displayed.' in caplog.text
