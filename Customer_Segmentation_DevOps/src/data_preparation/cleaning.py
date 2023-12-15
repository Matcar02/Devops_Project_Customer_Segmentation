import pandas as pd
import logging
import os
import random as rand
from datetime import datetime
import pandas as pd 
import mlflow
import argparse
import wandb

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(filepath):
    """
    Load and preprocess data from a CSV file.

    This function loads data from a CSV file, removes duplicate records, and drops specific columns 
    that are not required for further analysis.

    Args:
        filepath (str): The file path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame: A DataFrame with duplicates removed and unnecessary columns dropped.
    """

    logging.info('Preparing data...')

    # Load data from a specified CSV file and handle file not found errors
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error("File not found: %s", filepath)
        return

    # Remove duplicate rows from the DataFrame
    df.drop_duplicates(inplace=True)
    # List of columns to be dropped from the DataFrame
    cat = ['product_name_lenght', 'product_description_lenght', 'shipping_limit_date', 'product_category_name']
    # Drop the specified columns
    df.drop(columns=cat, axis=1, inplace=True)
    logging.debug('Data after removing duplicates and dropping columns:\n%s', df.head())
    logging.info('Data prepared successfully.')
    return df


def drop_c_id(df):
    """
    Drop the 'customer_unique_id' column from the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame from which the 'customer_unique_id' column will be dropped.

    Returns:
        pandas.DataFrame: The DataFrame after the 'customer_unique_id' column has been dropped.
    """
    
    logging.info('Dropping customer id...')
    logging.debug('Number of unique customer ids before dropping: %s', df['customer_id'].nunique())

    if 'customer_unique_id' in df.columns:
        # Drop 'customer_unique_id' column
        df.drop(columns='customer_unique_id', inplace=True)
    else:
        logging.warning("'customer_unique_id' not found in DataFrame.")

    logging.debug('Number of unique customer ids after dropping: %s', df['customer_id'].nunique())
    logging.debug('Data after dropping customer id:\n%s', df.head())
    logging.info('Customer id dropped successfully.')
    return df


def clean_data(df):
    """
    Clean the data by filtering and sampling.

    This function filters the data to include only 'delivered' order status and then samples a fraction of the data
    for further analysis.

    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """

    logging.info('Cleaning data...')

    df = pd.DataFrame(df)
    # Filter DataFrame for rows where order status is 'delivered'
    df = df[df['order_status'] == 'delivered']
    # Randomly sample a fraction of the DataFrame
    df = df.sample(frac=0.1, random_state=rand.randint(0, 1000))
    logging.debug('Data after filtering by order status and sampling:\n%s', df.head())
    logging.info('Data cleaned successfully.')
    return df


def main(args):
    wandb.init(project="customer_segmentation", job_type="data_cleaning")

    # Use the provided filepath or the W&B artifact
    if args.filepath != "default":
        df = pd.read_csv(args.filepath)
    else:
        # Use W&B to load the data artifact
        artifact = wandb.use_artifact('customer_segmentation:latest')
        artifact_dir = artifact.download()
        filepath = os.path.join(artifact_dir, 'customer_segmentation.csv')
        df = prepare_data(filepath)


    if df is not None:
        df = drop_c_id(df)
        df = clean_data(df)

        # Save the cleaned data as a new artifact
        cleaned_data_artifact = wandb.Artifact(
        "cleaned_data", type="dataset",
        description="Cleaned data after preprocessing"
    )
    
        cleaned_csv_path = os.path.join(wandb.run.dir, "cleaned_data.csv")
        df.to_csv(cleaned_csv_path, index=False)
        cleaned_data_artifact.add_file(cleaned_csv_path)

        wandb.log_artifact(cleaned_data_artifact)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Cleaning Process")
    parser.add_argument("--filepath", type=str, default="default", help="Path to the CSV file for cleaning or 'default' to use the latest W&B artifact")
    
    args = parser.parse_args()
    main(args)