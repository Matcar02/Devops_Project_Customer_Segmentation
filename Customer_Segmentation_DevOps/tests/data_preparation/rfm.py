import pandas as pd
from src.data_preparation.cleaning import prepare_data, drop_columns, drop_c_id, clean_data
import json 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#df = prepare_data(r"C:\Users\39339\Documents\University materials\University materials\Erasmus IE\DevOps\Devops_Project\Customer_Segmentation_DevOps\data\external\customer_segmentation.csv")
'''
df = prepare_data(data_path)
df = drop_c_id(df)
df = clean_data(df)
'''

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

def concatenate_dataframes(recency, monetary, frequencies):
    logging.info("Concatenating recency, monetary, and frequencies dataframes.")
    rfm_dataset = pd.concat([recency, monetary['Monetary value'], frequencies['Frequency']], axis=1)
    if rfm_dataset.isnull().sum().any():
        logging.warning(f"Detected missing values after concatenation. Number of missing values: {rfm_dataset.isnull().sum().sum()}")
    rfm_dataset.dropna(inplace=True)
    logging.info("Dataframes concatenated successfully.")
    return rfm_dataset


