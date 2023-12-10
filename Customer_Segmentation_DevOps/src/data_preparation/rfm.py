import logging
import os
from datetime import datetime

import pandas as pd

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


def get_rfm_dataset(rfm_dataset):
    """
    Save the RFM dataset to a CSV file and return the DataFrame.

    Args:
        rfm_dataset (pandas.DataFrame): The RFM dataset to be saved and returned.

    Returns:
        pandas.DataFrame: The RFM dataset after saving it to a CSV file.
    """

    logging.info('Getting DataFrame...')
    # Save the RFM dataset to a CSV file in the 'reports' directory
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports', 'dataframes'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f'rfmdata_{now}.csv'
    try:
        rfm_dataset = pd.DataFrame(rfm_dataset)
        # Construct the complete file path
        file_path = os.path.join(reports_path, filename)  
        # Pass the file path to to_csv function
        rfm_dataset.to_csv(file_path, index=False)  
        logging.info('DataFrame saved successfully.')
    except Exception as e:
        logging.error('Error saving DataFrame to CSV: %s', e)
        return None

    rfm_dataset = pd.DataFrame(rfm_dataset)
    # Return the DataFrame after saving it
    return rfm_dataset  

