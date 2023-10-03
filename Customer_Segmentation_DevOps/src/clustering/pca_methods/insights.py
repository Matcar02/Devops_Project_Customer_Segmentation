import pandas as pd

def pca_insights(dfpca):
    f2 = ['Recency','Monetary value','Frequency']
    first = dfpca[dfpca['kmeansclustersPCA'] == 0][f2]
    sec =  dfpca[dfpca['kmeansclustersPCA'] == 1][f2]
    th = dfpca[dfpca['kmeansclustersPCA'] == 2][f2]
    four = dfpca[dfpca['kmeansclustersPCA'] == 3][f2]
    first.describe()
    sec.describe()
    th.describe()
    four.describe()
    return first, sec, th


def pca_insights2(df, dfpca):
    customer_payment= df.groupby(by='customer_id',
                            as_index=False)['payment_type'].max()
                            
    customer_installments= df.groupby(by='customer_id',
                            as_index=False)['payment_installments'].mean()
    customer_city = df.groupby(by='customer_id',
                            as_index=False)['customer_state'].max()
    product_category= df.groupby(by='customer_id',
                            as_index=False)['product_category_name_english'].max()

    e = customer_payment.iloc[:, [1]]
    e.reset_index(drop=True, inplace=True)
    r = customer_installments.iloc[:,[1]]
    r.reset_index(drop=True, inplace=True)
    q = customer_city.iloc[:,[1]]
    q.reset_index(drop=True, inplace=True)
    t=product_category.iloc[:,[1]]
    t.reset_index(drop=True, inplace=True)

    temp = pd.concat([e,r,q,t], axis=1)

    temp.head()
    temp.reset_index(drop=True, inplace=True)
    dfpcaf= pd.concat([dfpca, temp], axis=1)
    dfpcaf.head()
    return dfpcaf 