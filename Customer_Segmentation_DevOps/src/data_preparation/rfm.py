#testing if they work properly (they do)
from src.data_preparation.cleaning import prepare_data, drop_columns, drop_c_id, clean_data
df = prepare_data(r"C:\Users\39339\Documents\University materials\University materials\Erasmus IE\DevOps\Devops_Project\Customer_Segmentation_DevOps\data\external\customer_segmentation.csv")
df = drop_c_id(df)
df = clean_data(df)



import pandas as pd

def get_frequencies(df):
    #grouping by and getting the total money spent by customer
    frequencies = df.groupby(
        by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    return frequencies



def get_recency(df):
    #using order_purchase_timestamp instead of delivered_carrier_date.
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    recency = df.groupby(by='customer_id',
                            as_index=False)['order_purchase_timestamp'].max()

    recency.columns = ['Customer ID', 'Latest Purchase']

    recent_date = recency['Latest Purchase'].max()

    recency['Recency'] = recency['Latest Purchase'].apply(
        lambda x: (recent_date - x).days)                     
        
    recency.drop(columns=['Latest Purchase'], inplace=True)  #we don't care about the date (we have recency)
    return recency



def get_monetary(df):
    #grouping by and getting the total money spent by customer
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = [' Monetary Customer ID', 'Monetary value']
    return monetary 


def concatenate_dataframes(recency, monetary, frequencies):
    rfm_dataset = pd.concat([recency, monetary['Monetary value'], frequencies['Frequency']], axis=1)
    rfm_dataset.dropna(inplace=True)   #dropping the nulls, if any
    return rfm_dataset





