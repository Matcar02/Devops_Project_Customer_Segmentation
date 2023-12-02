import pytest
import pandas as pd
from sklearn.datasets import make_blobs
import sys, os, logging

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)

from src.clustering.pca_methods.insights import pca_insights, pca_insights2

@pytest.fixture
def dfpca_fixture():
    data, _ = make_blobs(n_samples=100, centers=4, n_features=3, random_state=42)
    dfpca = pd.DataFrame(data, columns=['Recency', 'Monetary value', 'Frequency'])
    dfpca['kmeansclustersPCA'] = [i % 4 for i in range(len(dfpca))]
    return dfpca

@pytest.fixture
def df_fixture():
    data = {
        'customer_id': [i for i in range(100)],
        'payment_type': ['credit_card' if i % 2 == 0 else 'voucher' for i in range(100)],
        'payment_installments': [i % 10 for i in range(100)],
        'customer_state': ['state' + str(i % 3) for i in range(100)],
        'product_category_name_english': ['category' + str(i % 5) for i in range(100)],
    }
    return pd.DataFrame(data)

def test_pca_insights(dfpca_fixture, caplog):
    caplog.set_level(logging.INFO)
    descriptions = pca_insights(dfpca_fixture)

    assert len(descriptions) == 4, "Should return descriptions for 4 clusters"
    for desc in descriptions:
        assert isinstance(desc, pd.DataFrame), "Each description should be a pandas DataFrame"
        assert 'mean' in desc.index, "Description should include mean statistics" 
    assert 'Starting PCA insights' in caplog.text
    assert 'Describing Cluster 0' in caplog.text
    assert 'PCA insights completed' in caplog.text


def test_pca_insights2(df_fixture, dfpca_fixture, caplog):
    caplog.set_level(logging.INFO)
    dfpcaf = pca_insights2(df_fixture, dfpca_fixture)

    assert not dfpcaf.empty, "The resulting DataFrame should not be empty"
    assert 'payment_type' in dfpcaf.columns, "DataFrame should include payment_type"
    assert 'payment_installments' in dfpcaf.columns, "DataFrame should include payment_installments"
    assert 'customer_state' in dfpcaf.columns, "DataFrame should include customer_state"
    assert 'product_category_name_english' in dfpcaf.columns, "DataFrame should include product_category_name_english"
    assert 'Starting PCA insights 2' in caplog.text
    assert 'PCA insights 2 completed' in caplog.text
