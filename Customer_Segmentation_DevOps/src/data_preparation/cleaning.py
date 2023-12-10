import pandas as pd
import logging
import os
import random as rand
from datetime import datetime
import pandas as pd 

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


def get_df(df, output_dir=None):
    """
    Save the DataFrame to a CSV file.

    This function saves the given DataFrame to a CSV file in the specified directory. If the directory is not provided,
    it saves the CSV in a default 'reports' directory.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved as a CSV file.
        output_dir (str, optional): The directory where the CSV file will be saved. Defaults to None, which uses a default directory.

    Returns:
        bool: True if the DataFrame is saved successfully, False otherwise.
    """

    logging.info('Getting DataFrame...')
    # Set the output directory to a default if not provided
    if output_dir is None:
        current_path = os.getcwd()
        output_dir = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info('Saving DataFrame to CSV...')

    # Save the DataFrame as a CSV file in the specified directory
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, 'dataframes')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df.to_csv(os.path.join(output_path, f'initialdata_{now}.csv'), index=False)
        logging.info('DataFrame saved successfully.')
        return True
    except Exception as e:
        logging.error('Error saving DataFrame to CSV: %s', e)
        return False


