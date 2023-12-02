#optional!

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_df(df):
    logging.info("Starting one-hot encoding.")
    transformer = make_column_transformer(
        (OneHotEncoder(sparse=False), ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english']),
        remainder='passthrough')

    encoded_df = transformer.fit_transform(df)
    encoded_df = pd.DataFrame(
        encoded_df, 
        columns=transformer.get_feature_names_out()
    )
    logging.info("One-hot encoding completed.")
    return encoded_df

def get_dummies_df(df):
    logging.info("Starting encoding using get_dummies.")
    dummies_df = pd.get_dummies(df, columns=['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english'])
    logging.info("Encoding using get_dummies completed.")
    return dummies_df




