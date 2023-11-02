import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def segments_insights(rfmcopy):

    scaledrfm = rfmcopy.copy()

    sns.set_palette("Dark2")
    sns.displot(data = scaledrfm , x = 'Monetary value', hue = "kmeans_cluster", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Frequency" , hue = "kmeans_cluster", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Recency" , hue = "kmeans_cluster", multiple = "stack")

    sns.set_palette("colorblind", 4)
    sns.displot(data = scaledrfm , x = 'Monetary value', hue = "sp_clusters", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Frequency" , hue = "sp_clusters", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Recency" , hue = "sp_clusters", multiple = "stack")

    feat = ['Recency', 'Frequency', 'Monetary value']
    lowspenderskmeans = rfmcopy[rfmcopy['kmeans_cluster'] == 1][feat]
    lowspendershc = rfmcopy[rfmcopy['hc_clusters'] == 2][feat]
    lowspenderssc = rfmcopy[rfmcopy['sp_clusters'] == 0][feat]


    midspenderssp = rfmcopy[(rfmcopy['sp_clusters'] == 1) | (rfmcopy['sp_clusters'] == 3)][feat]
    midspendershc = rfmcopy[rfmcopy['hc_clusters'] == 0][feat]
    midspenderskm = rfmcopy[rfmcopy['kmeans_cluster'] == 0][feat]

    highspenderssc = rfmcopy[ (rfmcopy['sp_clusters'] == 2)][feat]
    highspendershc = rfmcopy[(rfmcopy['hc_clusters'] == 3) | (rfmcopy['hc_clusters'] == 1)][feat]
    highspenderskm = rfmcopy[(rfmcopy['kmeans_cluster'] == 2) | (rfmcopy['kmeans_cluster'] == 3)][feat]

    lowspenderskmeans.describe() 

    lowspenderssc.describe()

    midspenderskm.describe()

    midspenderssp.describe()

    highspenderskm.describe()

    highspenderssc.describe()

    return lowspendershc, lowspenderskmeans, lowspenderssc, midspenderskm, midspenderssp, midspendershc, highspenderskm, highspenderssc, highspendershc 



def kmeans_summary(rfmcopy):

    cl1size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 0])
    cl1sum = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].sum()
    cl1mean = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].mean()
    cl1frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Frequency'].mean()

    cl1fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Frequency'].std()
    cl1sd = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].std()

    cl2size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 1]) 
    cl2sum = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Monetary value'].sum()
    cl2mean = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Monetary value'].mean()
    cl2frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Frequency'].mean()
    cl2fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Frequency'].std()
    cl2sd = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Monetary value'].std()


    cl3size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 2])
    cl3sum =  rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Monetary value'].sum()
    cl3mean = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Monetary value'].mean()
    cl3frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Frequency'].mean()
    cl3fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Frequency'].std()
    cl3sd = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Monetary value'].std()


    cl4size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 3])
    cl4sum = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Monetary value'].sum()
    cl4mean = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Monetary value'].mean()
    cl4frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Frequency'].mean()
    cl4fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Frequency'].std()
    cl4sd = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Monetary value'].std()

    dictio2 = {'Clustersize': [cl1size,cl2size,cl3size, cl4size], 'Total spending by cluster': [cl1sum,cl2sum,cl3sum,cl4sum],
            'Average spending by cluster' : [cl1mean,cl2mean,cl3mean,cl4mean], 'Average frequency by cluster':
            [cl1frequency ,cl2frequency ,cl3frequency ,cl4frequency], 'Frequency std': [cl1fsd,cl2fsd,cl3fsd,cl4fsd], 'Spending sd':
            [cl1sd,cl2sd,cl3sd,cl4sd]}

    Kmeanssummary = pd.DataFrame(dictio2, index = [1,2,3,4]) 
    return Kmeanssummary 


def cluster_summary(df, column_name):

    kmeans_size = df.groupby('kmeans_cluster')[column_name].size()
    kmeans_sum = df.groupby('kmeans_cluster')[column_name].sum()
    kmeans_mean = df.groupby('kmeans_cluster')[column_name].mean()
    kmeans_frequency = df.groupby('kmeans_cluster')['Frequency'].mean()
    kmeans_fsd = df.groupby('kmeans_cluster')['Frequency'].std()
    kmeans_sd = df.groupby('kmeans_cluster')[column_name].std()
    Kmeanssummary = pd.DataFrame({'Clustersize': kmeans_size, 'Total spending by cluster': kmeans_sum,
                                  'Average spending by cluster': kmeans_mean, 'Average frequency by cluster': kmeans_frequency,
                                  'Frequency std': kmeans_fsd, 'Spending sd': kmeans_sd})
    
    hc_size = df.groupby('hc_clusters')[column_name].size()
    hc_sum = df.groupby('hc_clusters')[column_name].sum()
    hc_mean = df.groupby('hc_clusters')[column_name].mean()
    hc_frequency = df.groupby('hc_clusters')['Frequency'].mean()
    hc_fsd = df.groupby('hc_clusters')['Frequency'].std()
    hc_sd = df.groupby('hc_clusters')[column_name].std()
    Hcsummary = pd.DataFrame({'Clustersize': hc_size, 'Total spending by cluster': hc_sum,
                              'Average spending by cluster': hc_mean, 'Average frequency by cluster': hc_frequency,
                              'Frequency std': hc_fsd, 'Spending sd': hc_sd})
    
    sp_size = df.groupby('sp_clusters')[column_name].size()
    sp_sum = df.groupby('sp_clusters')[column_name].sum()
    sp_mean = df.groupby('sp_clusters')[column_name].mean()
    sp_frequency = df.groupby('sp_clusters')['Frequency'].mean()
    sp_fsd = df.groupby('sp_clusters')['Frequency'].std()
    sp_sd = df.groupby('sp_clusters')[column_name].std()
    Spsummary = pd.DataFrame({'Clustersize': sp_size, 'Total spending by cluster': sp_sum,
                              'Average spending by cluster': sp_mean, 'Average frequency by cluster': sp_frequency,
                              'Frequency std': sp_fsd, 'Spending sd': sp_sd})
    
    return Kmeanssummary, Hcsummary, Spsummary



def installments_analysis(df, rfmcopy):

    installments = df.groupby(
        by=['customer_id'], as_index=False)['payment_installments'].mean()

    paymentty = df.groupby(
        by=['customer_id'], as_index=False)['payment_type'].max()

    installments.head()

    w = installments.iloc[:, [1]]
    w.reset_index(drop=True, inplace=True)
    n = paymentty.iloc[:,[1]]
    n.reset_index(drop=True, inplace=True)

    paydf = pd.concat([rfmcopy,w,n], axis=1)
    paydf.head()       

    return paydf 


##to be assessed
def customers_insights(paydf):
    #clusterstype = ['mid-spenders','at-risk customers', 'top-customers','high-spenders']
    paydict = {}
    for i in range(4):
        countpay = paydf[paydf['kmeans_cluster'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['kmeans_cluster'] == i]['payment_installments'].mean()
        paydict[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}
        
        print("")
        print(f"The payment distribution for the cluster made by cluster{[i]} of kmeans is")
        print(countpay)
        print("")
        print(f"The average installments made by customers in cluster{[i]} is {meaninst}")
        print("---------------------------------")

    #customersHc = ['Mid-spenders','top customers','at-risk customers', 'high-spenders']
    paydict2 = {}
    for i in range(4):
        countpay = paydf[paydf['hc_clusters'] == i]['payment_type'].value_counts() 
        meaninst = paydf[paydf['hc_clusters'] == i]['payment_installments'].mean() 
        paydict2[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}
        
        print(f"The payment distribution for the cluster cluster{[i]} of HC is")
        print(countpay)
        print(f"The average installments made by customers in cluster {[i]} is {meaninst}")
        print("---------------------------------")

    #customersSp = ['Low spenders', 'at-risk customers','top customers', 'High spenders']
    paydict3 = {}
    for i in range(4):
        countpay = paydf[paydf['sp_clusters'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['sp_clusters'] == i]['payment_installments'].mean() 
        paydict3[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}
        
        print(f"The payment distribution for the cluster {[i]} of Spectral is")
        print(countpay)
        print(f"The average installments made by customers in cluster {[i]} is {meaninst}")
        print("---------------------------------")

    return paydict, paydict2, paydict3


def recency(recency):
    recencydist=list(recency["Recency"])
    plt.hist(x=recencydist)
    plt.xlabel('days since last purchase')
    plt.ylabel('number of people per period')
    sns.histplot()
    plt.show()


def payments_insights(df):
    sns.histplot(data=df["payment_installments"])
    sns.countplot(x=df["payment_type"])
    paymentdistr = df.groupby(['payment_type'])['payment_value'].mean().reset_index(name='Avg_Spending').sort_values(['Avg_Spending'], ascending = False)
    x=["boleto","credit_card","debit_card","voucher"]
    y=paymentdistr["Avg_Spending"]
    plt.plot(x,y)
    plt.xlabel("Payment Type")
    plt.ylabel("Average Price")
    plt.title("Average Spending Distribution by Payment Type")
    plt.show()
    return paymentdistr 


##to be assessed
def prod_insights(df):
    dfcat = pd.value_counts(df['product_category_name_english']).iloc[:15].index 
    df['product_category_name_english'] = pd.Categorical(df['product_category_name_english'], categories=dfcat, ordered=True)
    ax = sns.countplot(x='product_category_name_english', data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
    plt.show()


def customer_geography(df):
    dfgeo = pd.value_counts(df['customer_state']).iloc[:20]
    dfticks = pd.value_counts(df['customer_state']).iloc[:20].index.to_list() 
    dfgeo.plot()
    plt.title("Customers per state")
    
    l = []
    for i in range(20):
        l.append(df[df["customer_state"] == dfticks[i]]["payment_value"].mean())

    ax = sns.lineplot(x = dfticks, y = l)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()
    return dfgeo

def get_frequencies(df):
    frequencies = df.groupby(
        by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    return frequencies

def get_recency(df):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    recency = df.groupby(by='customer_id',
                            as_index=False)['order_purchase_timestamp'].max()

    recency.columns = ['Customer ID', 'Latest Purchase']

    recent_date = recency['Latest Purchase'].max()

    recency['Recency'] = recency['Latest Purchase'].apply(
        lambda x: (recent_date - x).days)                     
        
    recency.drop(columns=['Latest Purchase'], inplace=True) 
    return recency


def get_monetary(df):
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = [' Monetary Customer ID', 'Monetary value']
    return monetary 


def concatenate_dataframes(recency, monetary, frequencies):
    rfm_dataset = pd.concat([recency, monetary, frequencies], axis=1)
    cols = [3,5]   
    rfm_dataset.drop(columns=rfm_dataset.columns[cols], axis=1, inplace=True)
    rfm_dataset.dropna(inplace=True)  
    return rfm_dataset








