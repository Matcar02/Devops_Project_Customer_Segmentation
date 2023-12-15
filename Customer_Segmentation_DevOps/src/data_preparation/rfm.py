import logging
import os
from datetime import datetime
import mlflow
import wandb
import pandas as pd
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_frequencies(df):
    """
    Compute the frequencies of orders for each customer.

    Args:
        df (pandas.DataFrame): The DataFrame containing customer orders.

    Returns:
        pandas.DataFrame: A DataFrame with the frequencies of orders for each customer.
    """

    logging.info("Computing frequencies.")
    # Group the data by customer ID and count the occurrences
    frequencies = df.groupby(by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    return frequencies


def get_recency(df):
    """
    Compute the recency of the last order for each customer.

    Args:
        df (pandas.DataFrame): The DataFrame containing customer orders with timestamps.

    Returns:
        pandas.DataFrame: A DataFrame with the recency of the last order for each customer.
    """

    logging.info("Computing recency.")
    # Convert order purchase timestamps to datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    # Group by customer ID and get the most recent purchase date
    recency = df.groupby(by='customer_id', as_index=False)['order_purchase_timestamp'].max()
    recency.columns = ['Customer ID', 'Latest Purchase']

    # Calculate recency in days
    recent_date = recency['Latest Purchase'].max()
    recency['Recency'] = recency['Latest Purchase'].apply(lambda x: (recent_date - x).days)
    recency.drop(columns=['Latest Purchase'], inplace=True)
    return recency


def get_monetary(df):
    """
    Calculate the total monetary value of orders for each customer.

    Args:
        df (pandas.DataFrame): The DataFrame containing order values.

    Returns:
        pandas.DataFrame: A DataFrame with the total monetary value of orders for each customer.
    """

    logging.info("Computing monetary values.")
    # Group the data by customer ID and sum the payment values
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = ['Monetary Customer ID', 'Monetary value']
    return monetary


def concatenate_dataframes_(recency, monetary, frequencies):
    """
    Concatenate recency, monetary, and frequency DataFrames.

    This function merges the provided recency, monetary, and frequency DataFrames into a single DataFrame.

    Args:
        recency (pandas.DataFrame): The DataFrame containing recency data.
        monetary (pandas.DataFrame): The DataFrame containing monetary data.
        frequencies (pandas.DataFrame): The DataFrame containing frequency data.

    Returns:
        pandas.DataFrame: The concatenated DataFrame.
    """

    logging.info("Concatenating recency, monetary, and frequencies dataframes.")
    # Concatenate the recency, monetary, and frequency DataFrames
    rfm_dataset = pd.concat([recency, monetary['Monetary value'], frequencies['Frequency']], axis=1)

    # Check and drop any rows with missing values
    if rfm_dataset.isnull().sum().any():
        logging.warning("Detected missing values after concatenation. Number of missing values: %d",
                        rfm_dataset.isnull().sum().sum())
    rfm_dataset.dropna(inplace=True)
    logging.info("Dataframes concatenated successfully.")

    return rfm_dataset


def main(args):
    wandb.init(project="customer_segmentation", job_type="rfm_analysis")

    if args.filepath != "default":
        df = pd.read_csv(args.filepath)

    else:

        # Use W&B to load the cleaned data artifact
        artifact = wandb.use_artifact('cleaned_data:latest')
        artifact_dir = artifact.download()
        cleaned_data_filepath = os.path.join(artifact_dir, 'cleaned_data.csv')
        df = pd.read_csv(cleaned_data_filepath)

    frequencies = get_frequencies(df)
    recency = get_recency(df)
    monetary = get_monetary(df)
    rfm_dataset = concatenate_dataframes_(recency, monetary, frequencies)

    # Save the RFM analysis data as a new artifact
    rfm_data_artifact = wandb.Artifact("rfm_data", type="dataset", description="RFM analysis data")
    rfm_csv_path = os.path.join(wandb.run.dir, "rfm_data.csv")
    rfm_dataset.to_csv(rfm_csv_path, index=False)
    rfm_data_artifact.add_file(rfm_csv_path)
    wandb.log_artifact(rfm_data_artifact)

    # Save the recency data as a separate artifact
    recency_data_artifact = wandb.Artifact("recency_data", type="dataset", description="Recency data")
    recency_csv_path = os.path.join(wandb.run.dir, "recency_data.csv")
    recency.to_csv(recency_csv_path, index=False)
    recency_data_artifact.add_file(recency_csv_path)
    wandb.log_artifact(recency_data_artifact)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFM analysis")
    parser.add_argument("--filepath", type=str, default="default", help="Path to the cleaned CSV file for RFM analysis or 'default' to use the latest W&B artifact")
    args = parser.parse_args()
    main(args)