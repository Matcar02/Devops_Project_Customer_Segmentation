import pandas as pd
import json 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def get_frequencies(df):
    logging.info("Computing frequencies.")
    frequencies = df.groupby(by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    return frequencies

def get_recency(df):
    logging.info("Computing recency.")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    recency = df.groupby(by='customer_id', as_index=False)['order_purchase_timestamp'].max()
    recency.columns = ['Customer ID', 'Latest Purchase']
    recent_date = recency['Latest Purchase'].max()
    recency['Recency'] = recency['Latest Purchase'].apply(lambda x: (recent_date - x).days)
    recency.drop(columns=['Latest Purchase'], inplace=True)
    return recency

def get_monetary(df):
    logging.info("Computing monetary values.")
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = [' Monetary Customer ID', 'Monetary value']
    return monetary 

def concatenate_dataframes_(recency, monetary, frequencies):
    logging.info("Concatenating recency, monetary, and frequencies dataframes.")
    rfm_dataset = pd.concat([recency, monetary['Monetary value'], frequencies['Frequency']], axis=1)
    if rfm_dataset.isnull().sum().any():
        logging.warning(f"Detected missing values after concatenation. Number of missing values: {rfm_dataset.isnull().sum().sum()}")
    rfm_dataset.dropna(inplace=True)
    logging.info("Dataframes concatenated successfully.")
    return rfm_dataset


