#optional!

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_df(df):
    """
    Perform one-hot encoding on specified columns of a DataFrame.

    This function applies one-hot encoding to transform categorical data into a format that can be easily used for machine learning models.

    Args:
        df (pandas.DataFrame): The DataFrame to be encoded.

    Returns:
        pandas.DataFrame: A DataFrame with categorical columns one-hot encoded.
    """

    logging.info("Starting one-hot encoding.")
    # Create a transformer for one-hot encoding specified columns
    transformer = make_column_transformer(
        (OneHotEncoder(sparse=False), ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english']),
        remainder='passthrough')

    # Apply the transformer to the DataFrame
    encoded_df = transformer.fit_transform(df)
    # Convert the result back to a DataFrame
    encoded_df = pd.DataFrame(
        encoded_df, 
        columns=transformer.get_feature_names_out()
    )
    logging.info("One-hot encoding completed.")
    return encoded_df

def get_dummies_df(df):
    """
    Encode categorical features using pandas get_dummies method.

    This function encodes categorical columns of the DataFrame using pandas get_dummies, which is another form of one-hot encoding.

    Args:
        df (pandas.DataFrame): The DataFrame whose categorical features are to be encoded.

    Returns:
        pandas.DataFrame: A DataFrame with categorical features encoded.
    """

    logging.info("Starting encoding using get_dummies.")
    # Apply pandas get_dummies for one-hot encoding
    dummies_df = pd.get_dummies(df, columns=['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english'])
    logging.info("Encoding using get_dummies completed.")
    return dummies_df




