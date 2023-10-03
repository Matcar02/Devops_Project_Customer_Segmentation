import seaborn as sns

def describe_dataset(rfm_dataset):
    print(rfm_dataset.describe())


def corr(df):
    columns=["payment_type","payment_installments","payment_value"]
    sns.pairplot(df[columns])
    df[columns].corr()