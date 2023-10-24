import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['product_name_lenght', 'product_description_lenght', 'shipping_limit_date', 'product_category_name'], axis=1, inplace=True)
    return df


def drop_columns(df):
    cat =  ['product_name_lenght','product_description_lenght','shipping_limit_date','product_category_name']
    df.drop(columns = cat , axis = 1, inplace= True) 
    return df


def drop_c_id(df):
    df['customer_unique_id'].nunique()
    df['customer_id'].nunique()
    df.drop(columns = 'customer_unique_id',inplace= True)
    df.head()
    return df 

def clean_data(df):
    df = df[df['order_status'] == 'delivered']
    return df




