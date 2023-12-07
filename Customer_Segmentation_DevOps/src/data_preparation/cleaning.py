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
    Load data from a CSV file, remove duplicates, and drop specific columns.
    """
    logging.info('Preparing data...')

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error("File not found: %s", filepath)
        return

    df.drop_duplicates(inplace=True)
    cat = ['product_name_lenght', 'product_description_lenght', 'shipping_limit_date', 'product_category_name']
    df.drop(columns=cat, axis=1, inplace=True)
    logging.debug('Data after removing duplicates and dropping columns:\n%s', df.head())
    logging.info('Data prepared successfully.')
    return df


def drop_c_id(df):
    """
    Drop the customer_unique_id column.
    """
    logging.info('Dropping customer id...')
    logging.debug('Number of unique customer ids before dropping: %s', df['customer_id'].nunique())

    if 'customer_unique_id' in df.columns:
        df.drop(columns='customer_unique_id', inplace=True)
    else:
        logging.warning("'customer_unique_id' not found in DataFrame.")

    logging.debug('Number of unique customer ids after dropping: %s', df['customer_id'].nunique())
    logging.debug('Data after dropping customer id:\n%s', df.head())
    logging.info('Customer id dropped successfully.')
    return df


def clean_data(df):
    """
    Filter data by order status and sample a fraction.
    """
    logging.info('Cleaning data...')

    df = pd.DataFrame(df)
    df = df[df['order_status'] == 'delivered']
    df = df.sample(frac=0.1, random_state=rand.randint(0, 1000))
    logging.debug('Data after filtering by order status and sampling:\n%s', df.head())
    logging.info('Data cleaned successfully.')
    return df


def get_df(df, output_dir=None):
    """
    Get DataFrame and save it to CSV.
    """
    logging.info('Getting DataFrame...')
    if output_dir is None:
        current_path = os.getcwd()
        output_dir = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info('Saving DataFrame to CSV...')

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


