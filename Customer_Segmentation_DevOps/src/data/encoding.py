import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


def encode_df(df):
    transformer = make_column_transformer(
        (OneHotEncoder(sparse= False), ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english']),
        remainder='passthrough')

    encoded_df = transformer.fit_transform(df)
    encoded_df = pd.DataFrame(
        encoded_df, 
        columns=transformer.get_feature_names_out()
    )
    return encoded_df


def get_dummies_df(df):
    dummies_df = pd.get_dummies(df, columns = ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english'])
    return dummies_df