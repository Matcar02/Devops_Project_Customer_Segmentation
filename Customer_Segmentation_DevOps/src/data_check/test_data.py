import pandas as pd
import numpy as np
import logging
import pytest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_column_names(data, expected_columns):
    logger.info("Testing column names...")
    assert set(expected_columns) == set(data.columns.values), \
        "Column names do not match expected schema."

def test_id_column(data):
    logger.info("Testing ID columns...")
    for col in ['order_id', 'customer_id']:
        assert data[col].dtype == np.int64 and data[col].is_unique and (data[col] > 0).all(), \
            f"ID column {col} does not meet requirements of being unique positive integers."

def test_order_dates(data):
    logger.info("Testing order date columns...")
    date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_columns:
        assert pd.to_datetime(data[col], errors='coerce').notnull().all(), \
            f"Date column {col} contains invalid dates."

def test_payment_columns(data):
    logger.info("Testing payment columns...")
    assert data['payment_type'].isin(['credit_card', 'debit_card', 'voucher', 'boleto']).all(), \
        "Payment type contains invalid values."
    assert (data['payment_installments'] >= 0).all(), \
        "Payment installments contains negative values."
    assert (data['payment_value'] >= 0).all(), \
        "Payment value contains negative values."

def test_customer_information(data):
    logger.info("Testing customer information columns...")
    assert data['customer_unique_id'].apply(lambda x: isinstance(x, str)).all(), \
        "customer_unique_id column contains non-string values."
    assert data['customer_city'].apply(lambda x: isinstance(x, str)).all(), \
        "customer_city column contains non-string values."
    assert data['customer_state'].apply(lambda x: isinstance(x, str)).all(), \
        "customer_state column contains non-string values."

def test_product_information(data):
    logger.info("Testing product information columns...")
    assert (data['price'] >= 0).all(), \
        "Price column contains negative values."
    assert (data['freight_value'] >= 0).all(), \
        "Freight value column contains negative values."

def test_missing_values(data):
    logger.info("Testing for missing values...")
    assert not data.isnull().values.any(), "There are missing values in the data."

def test_data_consistency(data):
    logger.info("Testing data consistency...")
    assert (data['payment_value'] >= data['price']).all(), \
        "Inconsistent data: Payment value is less than product price."
