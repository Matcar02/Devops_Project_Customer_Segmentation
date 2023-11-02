import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def segments_insights(rfmcopy):

    logging.info('Starting analysis on the given rfmcopy data.')

    scaledrfm = rfmcopy.copy()
    
    logging.debug('Successfully copied the rfmcopy data to scaledrfm.')

    sns.set_palette("Dark2")
    logging.debug('Plotting kmeans_cluster based segments...')
    sns.displot(data = scaledrfm , x = 'Monetary value', hue = "kmeans_cluster", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Frequency" , hue = "kmeans_cluster", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Recency" , hue = "kmeans_cluster", multiple = "stack")

    sns.set_palette("colorblind", 4)
    logging.debug('Plotting sp_clusters based segments...')
    sns.displot(data = scaledrfm , x = 'Monetary value', hue = "sp_clusters", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Frequency" , hue = "sp_clusters", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Recency" , hue = "sp_clusters", multiple = "stack")

    feat = ['Recency', 'Frequency', 'Monetary value']
    
    logging.debug('Segmenting data based on different cluster categories...')
    lowspenderskmeans = rfmcopy[rfmcopy['kmeans_cluster'] == 1][feat]
    lowspendershc = rfmcopy[rfmcopy['hc_clusters'] == 2][feat]
    lowspenderssc = rfmcopy[rfmcopy['sp_clusters'] == 0][feat]
    midspenderssp = rfmcopy[(rfmcopy['sp_clusters'] == 1) | (rfmcopy['sp_clusters'] == 3)][feat]
    midspendershc = rfmcopy[rfmcopy['hc_clusters'] == 0][feat]
    midspenderskm = rfmcopy[rfmcopy['kmeans_cluster'] == 0][feat]
    highspenderssc = rfmcopy[ (rfmcopy['sp_clusters'] == 2)][feat]
    highspendershc = rfmcopy[(rfmcopy['hc_clusters'] == 3) | (rfmcopy['hc_clusters'] == 1)][feat]
    highspenderskm = rfmcopy[(rfmcopy['kmeans_cluster'] == 2) | (rfmcopy['kmeans_cluster'] == 3)][feat]

    logging.info('Descriptive analysis for each segment...')
    logging.debug('lowspenderskmeans: \n%s', lowspenderskmeans.describe()) 
    logging.debug('lowspenderssc: \n%s', lowspenderssc.describe())
    logging.debug('midspenderskm: \n%s', midspenderskm.describe())
    logging.debug('midspenderssp: \n%s', midspenderssp.describe())
    logging.debug('highspenderskm: \n%s', highspenderskm.describe())
    logging.debug('highspenderssc: \n%s', highspenderssc.describe())

    logging.info('Finished analysis. Returning segmented data.')

    return lowspendershc, lowspenderskmeans, lowspenderssc, midspenderskm, midspenderssp, midspendershc, highspenderskm, highspenderssc, highspendershc 


def kmeans_summary(rfmcopy):
    
    # Log the total size of the input data
    logging.info(f"Input data has {len(rfmcopy)} records.")
    
    # Helper function to calculate and log cluster statistics
    def cluster_stats(cluster_num):
        cluster_data = rfmcopy[rfmcopy['kmeans_cluster'] == cluster_num]
        size = len(cluster_data)
        mvalue_sum = cluster_data['Monetary value'].sum()
        mvalue_mean = cluster_data['Monetary value'].mean()
        frequency_mean = cluster_data['Frequency'].mean()
        frequency_std = cluster_data['Frequency'].std()
        mvalue_std = cluster_data['Monetary value'].std()
        
        # Log calculated stats for the cluster
        logging.info(f"Cluster {cluster_num} - Size: {size}, Total Spending: {mvalue_sum}, "
                     f"Average Spending: {mvalue_mean}, Average Frequency: {frequency_mean}, "
                     f"Frequency Std: {frequency_std}, Spending Std: {mvalue_std}")
        
        return size, mvalue_sum, mvalue_mean, frequency_mean, frequency_std, mvalue_std

    stats = [cluster_stats(i) for i in range(4)]
    
    dictio2 = {
        'Clustersize': [s[0] for s in stats],
        'Total spending by cluster': [s[1] for s in stats],
        'Average spending by cluster': [s[2] for s in stats],
        'Average frequency by cluster': [s[3] for s in stats],
        'Frequency std': [s[4] for s in stats],
        'Spending sd': [s[5] for s in stats]
    }

    Kmeanssummary = pd.DataFrame(dictio2, index=[1, 2, 3, 4])
    return Kmeanssummary 


def cluster_summary(df, column_name):
    # Log the initial size and column details of the input dataframe
    logging.info(f"Input dataframe has {len(df)} rows and {df.shape[1]} columns.")
    logging.info(f"Calculating summaries based on the column '{column_name}'.")

    # Kmeans summary
    kmeans_size = df.groupby('kmeans_cluster')[column_name].size()
    kmeans_sum = df.groupby('kmeans_cluster')[column_name].sum()
    kmeans_mean = df.groupby('kmeans_cluster')[column_name].mean()
    kmeans_frequency = df.groupby('kmeans_cluster')['Frequency'].mean()
    kmeans_fsd = df.groupby('kmeans_cluster')['Frequency'].std()
    kmeans_sd = df.groupby('kmeans_cluster')[column_name].std()
    Kmeanssummary = pd.DataFrame({
        'Clustersize': kmeans_size,
        'Total spending by cluster': kmeans_sum,
        'Average spending by cluster': kmeans_mean,
        'Average frequency by cluster': kmeans_frequency,
        'Frequency std': kmeans_fsd,
        'Spending sd': kmeans_sd
    })
    logging.info(f"KMeans summary processed with {len(Kmeanssummary)} clusters.")

    # HC summary
    hc_size = df.groupby('hc_clusters')[column_name].size()
    hc_sum = df.groupby('hc_clusters')[column_name].sum()
    hc_mean = df.groupby('hc_clusters')[column_name].mean()
    hc_frequency = df.groupby('hc_clusters')['Frequency'].mean()
    hc_fsd = df.groupby('hc_clusters')['Frequency'].std()
    hc_sd = df.groupby('hc_clusters')[column_name].std()
    Hcsummary = pd.DataFrame({
        'Clustersize': hc_size,
        'Total spending by cluster': hc_sum,
        'Average spending by cluster': hc_mean,
        'Average frequency by cluster': hc_frequency,
        'Frequency std': hc_fsd,
        'Spending sd': hc_sd
    })
    logging.info(f"HC summary processed with {len(Hcsummary)} clusters.")

    # SP summary
    sp_size = df.groupby('sp_clusters')[column_name].size()
    sp_sum = df.groupby('sp_clusters')[column_name].sum()
    sp_mean = df.groupby('sp_clusters')[column_name].mean()
    sp_frequency = df.groupby('sp_clusters')['Frequency'].mean()
    sp_fsd = df.groupby('sp_clusters')['Frequency'].std()
    sp_sd = df.groupby('sp_clusters')[column_name].std()
    Spsummary = pd.DataFrame({
        'Clustersize': sp_size,
        'Total spending by cluster': sp_sum,
        'Average spending by cluster': sp_mean,
        'Average frequency by cluster': sp_frequency,
        'Frequency std': sp_fsd,
        'Spending sd': sp_sd
    })
    logging.info(f"SP summary processed with {len(Spsummary)} clusters.")
    
    return Kmeanssummary, Hcsummary, Spsummary



def installments_analysis(df, rfmcopy):
    
    # Log the initial size of the input dataframes
    logging.info(f"Input dataframe 'df' has {len(df)} rows and {df.shape[1]} columns.")
    logging.info(f"Input dataframe 'rfmcopy' has {len(rfmcopy)} rows and {rfmcopy.shape[1]} columns.")
    
    # Analyzing installments
    installments = df.groupby(
        by=['customer_id'], as_index=False)['payment_installments'].mean()
    logging.info(f"Processed 'installments' dataframe shape: {installments.shape}")

    # Analyzing payment types
    paymentty = df.groupby(
        by=['customer_id'], as_index=False)['payment_type'].max()
    logging.info(f"Processed 'paymentty' dataframe shape: {paymentty.shape}")

    w = installments.iloc[:, [1]]
    w.reset_index(drop=True, inplace=True)

    n = paymentty.iloc[:,[1]]
    n.reset_index(drop=True, inplace=True)

    # Concatenating data
    paydf = pd.concat([rfmcopy, w, n], axis=1)

    # Confirming final dataframe characteristics
    logging.info(f"Final 'paydf' dataframe shape: {paydf.shape}")
    logging.info(f"Head of 'paydf':\n{paydf.head()}")
    
    return paydf 


##to be assessed
def customers_insights(paydf):

    logging.debug("Starting customers_insights function...")

    paydict = {}
    for i in range(4):
        countpay = paydf[paydf['kmeans_cluster'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['kmeans_cluster'] == i]['payment_installments'].mean()
        paydict[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}

        logging.info(f"The payment distribution for the cluster made by cluster{i} of kmeans is:\n{countpay}")
        logging.info(f"The average installments made by customers in cluster{i} is {meaninst}")
        logging.debug("---------------------------------")

    paydict2 = {}
    for i in range(4):
        countpay = paydf[paydf['hc_clusters'] == i]['payment_type'].value_counts() 
        meaninst = paydf[paydf['hc_clusters'] == i]['payment_installments'].mean() 
        paydict2[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}

        logging.info(f"The payment distribution for the cluster cluster{i} of HC is:\n{countpay}")
        logging.info(f"The average installments made by customers in cluster{i} is {meaninst}")
        logging.debug("---------------------------------")

    paydict3 = {}
    for i in range(4):
        countpay = paydf[paydf['sp_clusters'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['sp_clusters'] == i]['payment_installments'].mean() 
        paydict3[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}

        logging.info(f"The payment distribution for the cluster{i} of Spectral is:\n{countpay}")
        logging.info(f"The average installments made by customers in cluster{i} is {meaninst}")
        logging.debug("---------------------------------")

    if not (paydict and paydict2 and paydict3):
        logging.warning("One or more of the paydicts is empty.")

    return paydict, paydict2, paydict3


def recency(recency):
    logging.info("Starting recency function...")
    recencydist = list(recency["Recency"])
    
    if not recencydist:
        logging.warning("Recency distribution list is empty.")
        return

    logging.debug("Plotting histogram for recency distribution...")
    plt.hist(x=recencydist)
    plt.xlabel('days since last purchase')
    plt.ylabel('number of people per period')
    sns.histplot()
    plt.show()
    logging.info("Recency histogram displayed.")


def payments_insights(df):
    logging.info("Starting payments_insights function...")
    if df.empty:
        logging.warning("Input DataFrame is empty.")
        return None

    logging.debug("Plotting histogram and countplot for payments insights...")
    sns.histplot(data=df["payment_installments"])
    sns.countplot(x=df["payment_type"])
    paymentdistr = df.groupby(['payment_type'])['payment_value'].mean().reset_index(name='Avg_Spending').sort_values(['Avg_Spending'], ascending=False)
    
    if paymentdistr.empty:
        logging.warning("Payment distribution DataFrame is empty.")
        return None

    x = ["boleto", "credit_card", "debit_card", "voucher"]
    y = paymentdistr["Avg_Spending"]
    plt.plot(x, y)
    plt.xlabel("Payment Type")
    plt.ylabel("Average Price")
    plt.title("Average Spending Distribution by Payment Type")
    plt.show()
    logging.info("Payment insights displayed.")
    
    return paymentdistr 


def prod_insights(df):
    logging.info("Starting prod_insights function...")
    if df.empty:
        logging.warning("Input DataFrame is empty.")
        return

    dfcat = pd.value_counts(df['product_category_name_english']).iloc[:15].index
    df['product_category_name_english'] = pd.Categorical(df['product_category_name_english'], categories=dfcat, ordered=True)
    
    if dfcat.empty:
        logging.warning("Product category DataFrame is empty.")
        return

    logging.debug("Plotting countplot for product insights...")
    ax = sns.countplot(x='product_category_name_english', data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    plt.show()
    logging.info("Product insights displayed.")


def customer_geography(df):
    logging.info("Starting customer_geography function...")
    dfgeo = pd.value_counts(df['customer_state']).iloc[:20]

    if dfgeo.empty:
        logging.error("Geo DataFrame is empty.")
        return None

    dfticks = dfgeo.index.to_list() 
    dfgeo.plot()
    plt.title("Customers per state")

    l = []
    for i in range(20):
        l.append(df[df["customer_state"] == dfticks[i]]["payment_value"].mean())
    
    logging.debug("Generating lineplot for customer geography...")
    ax = sns.lineplot(x=dfticks, y=l)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
    logging.info("Customer geography plot displayed.")
    
    return dfgeo

def get_frequencies(df):
    logging.info("Starting get_frequencies function...")
    frequencies = df.groupby(by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    
    if frequencies.empty:
        logging.warning("Frequencies DataFrame is empty. No customer deliveries recorded.")
        return None
    
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    logging.debug("Finished generating frequencies DataFrame.")
    
    return frequencies

def get_recency(df):
    logging.info("Starting get_recency function...")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    recency = df.groupby(by='customer_id', as_index=False)['order_purchase_timestamp'].max()

    if recency.empty:
        logging.warning("Recency DataFrame is empty. No customer purchase timestamps recorded.")
        return None

    recency.columns = ['Customer ID', 'Latest Purchase']

    recent_date = recency['Latest Purchase'].max()
    
    if pd.isnull(recent_date):
        logging.error("Recent date is NaN. Exiting function.")
        return None

    logging.debug("Calculating Recency for each customer...")
    recency['Recency'] = recency['Latest Purchase'].apply(lambda x: (recent_date - x).days)                     
    recency.drop(columns=['Latest Purchase'], inplace=True)
    logging.debug("Finished generating recency DataFrame.")

    return recency

def get_monetary(df):
    logging.info("Starting get_monetary function...")
    
    if df.empty:
        logging.warning("Input DataFrame is empty. No data to process for monetary value.")
        return pd.DataFrame()
    
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = ['Monetary Customer ID', 'Monetary value']

    logging.debug("Monetary value calculation complete.")
    
    return monetary 

def concatenate_dataframes(recency, monetary, frequencies):
    logging.info("Starting concatenate_dataframes function...")
    
    if recency.empty or monetary.empty or frequencies.empty:
        logging.warning("One or more of the input DataFrames are empty.")
        return pd.DataFrame()

    rfm_dataset = pd.concat([recency, monetary, frequencies], axis=1)

    logging.debug("Concatenation of DataFrames successful.")

    cols = [3, 5]
    rfm_dataset.drop(columns=rfm_dataset.columns[cols], axis=1, inplace=True)
    rfm_dataset.dropna(inplace=True)  

    logging.info("Finished data cleaning and final DataFrame is ready.")
    
    return rfm_dataset








