#optional!

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import logging
import mlflow
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..')
sys.path.append(src_dir)

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


def main(filepath=None):
    with mlflow.start_run() as run:

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        default_filepath = os.path.join(data_dir, 'cleaned_data.csv')

        # If filepath is "default" or not provided, use the default filepath
        if not filepath or filepath == "default":
            filepath = default_filepath

        df = pd.read_csv(filepath)
        encoded_df = encode_df(df)

        # Save and log the encoded DataFrame
        encoded_csv_path = os.path.join(data_dir, 'encoded_data.csv')
        encoded_df.to_csv(encoded_csv_path, index=False)
        mlflow.log_artifact(encoded_csv_path, artifact_path="encoded_data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run encoding")
    parser.add_argument("--filepath", type=str, help="Path to the cleaned CSV file for encoding", default=None)
    args = parser.parse_args()
    main(filepath=args.filepath)
