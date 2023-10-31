import seaborn as sns
import pandas as pd 

def describe_dataset(rfm_dataset):
    print(rfm_dataset.describe())


def corr(df):
    columns=["payment_type","payment_installments","payment_value"]
    sns.pairplot(df[columns])
    corr_matrix = df[columns].corr()
    print(corr_matrix)


