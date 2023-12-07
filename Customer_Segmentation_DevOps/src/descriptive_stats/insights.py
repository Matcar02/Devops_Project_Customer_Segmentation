import os
import logging
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
def segments_insights(rfmcopy, nclusterskmeans):

    logging.info('Starting analysis on the given rfmcopy data.')

    scaledrfm = rfmcopy.copy()
    
    logging.debug('Successfully copied the rfmcopy data to scaledrfm.')

    sns.set_palette("Dark2")
    logging.debug('Plotting kmeans_cluster based segments...')
    km1 = sns.displot(data = scaledrfm , x = 'Monetary value', hue = "kmeans_cluster", multiple = "stack")
    km2 = sns.displot(data = scaledrfm , x = "Frequency" , hue = "kmeans_cluster", multiple = "stack")
    km3 = sns.displot(data = scaledrfm , x = "Recency" , hue = "kmeans_cluster", multiple = "stack")
    plt.show()

    sns.set_palette("colorblind", 4)
    logging.debug('Plotting sp_clusters based segments...')
    sp1 = sns.displot(data = scaledrfm , x = 'Monetary value', hue = "sp_clusters", multiple = "stack")
    sp2 = sns.displot(data = scaledrfm , x = "Frequency" , hue = "sp_clusters", multiple = "stack")
    sp3 = sns.displot(data = scaledrfm , x = "Recency" , hue = "sp_clusters", multiple = "stack")
    #arrange these plots in a grid and set it as a variable
    plt.show()

    sns.set_palette("bright")
    logging.debug('Plotting hc_clusters based segments...')
    hc1 = sns.displot(data = scaledrfm , x = 'Monetary value', hue = "hc_clusters", multiple = "stack")
    hc2 = sns.displot(data = scaledrfm , x = "Frequency" , hue = "hc_clusters", multiple = "stack")
    hc3 = sns.displot(data = scaledrfm , x = "Recency" , hue = "hc_clusters", multiple = "stack")
    plt.show()
    logging.info('Finished analysis. Returning segmented data.')
    
    return 

def kmeans_summary(rfmcopy, cluster_num):
    """
    Calculate and log cluster statistics based on the input data.

    Args:
        rfmcopy (DataFrame): Input data.

    Returns:
        DataFrame: Summary of cluster statistics.
    """
    # Log the total size of the input data
    logging.info(f"Input data has {len(rfmcopy)} records.")

    # Helper function to calculate and log cluster statistics
    
    cluster_data = rfmcopy[rfmcopy['kmeans_cluster'] == cluster_num]
    size = len(cluster_data)
    mvalue_sum = cluster_data['Monetary value'].sum()
    mvalue_mean = cluster_data['Monetary value'].mean()
    frequency_mean = cluster_data['Frequency'].mean()
    frequency_std = cluster_data['Frequency'].std()
    mvalue_std = cluster_data['Monetary value'].std()

    # Log calculated stats for the cluster
    logging.info("Cluster %d - Size: %d, Total Spending: %f, Average Spending: %f, "
                    "Average Frequency: %f, Frequency Std: %f, Spending Std: %f",
                    cluster_num, size, mvalue_sum, mvalue_mean, frequency_mean, frequency_std, mvalue_std)

    stats = (size, mvalue_sum, mvalue_mean, frequency_mean, frequency_std, mvalue_std)

    dictio2 = {
        'Clustersize': [stats[0]],
        'Total spending by cluster': [stats[1]],
        'Average spending by cluster': [stats[2]],
        'Average frequency by cluster': [stats[3]],
        'Frequency std': [stats[4]],
        'Spending sd': [stats[5]]
    }

    kmeanssummary = pd.DataFrame(dictio2)

    # Saving DataFrame to CSV
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'kmeanssummary_{now}.csv'
        kmeanssummary.to_csv(os.path.join(reports_path, 'dataframes', filename), index=False)

    except Exception as e:
        logging.error('Error saving DataFrame to CSV. %s', e)
        return None

    logging.info('DataFrame saved successfully.')
    return kmeanssummary



def cluster_summary(df, column_name):
    """
    Calculate cluster summaries based on the input dataframe and column name.

    Args:
        df (DataFrame): Input dataframe.
        column_name (str): Name of the column.

    Returns:
        tuple: Kmeanssummary, Hcsummary, Spsummary dataframes.
    """
    # Log the initial size and column details of the input dataframe
    logging.info(f"Input dataframe has {len(df)} rows and {df.shape[1]} columns.")
    logging.info(f"Calculating summaries based on the column '{column_name}'.")

    # Kmeans summary
    kmeans_size = df.groupby('kmeans_cluster')[column_name].size()
    kmeans_sum = df.groupby('kmeans_cluster')[column_name].sum()
    kmeans_mean = df.groupby('kmeans_cluster')[column_name].mean()
    kmeans_sd = df.groupby('kmeans_cluster')[column_name].std()
    kmeanssummary = pd.DataFrame({
        'Clustersize': kmeans_size,
        'Column sum': kmeans_sum,
        f'Average of {column_name}': kmeans_mean,
        f'Column {column_name} sd': kmeans_sd
    })
    logging.info(f"KMeans summary processed with {len(kmeanssummary)} clusters.")

    # HC summary
    hc_size = df.groupby('hc_clusters')[column_name].size()
    hc_sum = df.groupby('hc_clusters')[column_name].sum()
    hc_mean = df.groupby('hc_clusters')[column_name].mean()
    hc_sd = df.groupby('hc_clusters')[column_name].std()
    Hcsummary = pd.DataFrame({
        'Clustersize': hc_size,
        'Column sum': hc_sum,
        f'Average of {column_name}': hc_mean,
        f'Column {column_name} sd': hc_sd
    })
    logging.info(f"HC summary processed with {len(Hcsummary)} clusters.")

    # SP summary
    sp_size = df.groupby('sp_clusters')[column_name].size()
    sp_sum = df.groupby('sp_clusters')[column_name].sum()
    sp_mean = df.groupby('sp_clusters')[column_name].mean()
    sp_sd = df.groupby('sp_clusters')[column_name].std()
    Spsummary = pd.DataFrame({
        'Clustersize': sp_size,
        'Column sum': sp_sum,
        f'Average of {column_name}': sp_mean,
        f'Column {column_name} sd': sp_sd
    })
    logging.info(f"SP summary processed with {len(Spsummary)} clusters.")

    # Getting all the info
    logging.info('Getting DataFrames...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        kmeanssummary.to_csv(os.path.join(reports_path, 'dataframes', f'kmeans_summary_{now}.csv'), index=False)
        Hcsummary.to_csv(os.path.join(reports_path, 'dataframes', f'hc_summary_{now}.csv'), index=False)
        Spsummary.to_csv(os.path.join(reports_path, 'dataframes', f'sp_summary_{now}.csv'), index=False)

    except Exception as e:
        logging.error(f'Error saving DataFrame to CSV: {str(e)}')
        return None

    logging.info('DataFrame saved successfully.')
    return kmeanssummary, Hcsummary, Spsummary


def installments_analysis(df, rfmcopy):
    """
    Analyze installments and payment types based on the input dataframes.

    Args:
        df (DataFrame): Input dataframe.
        rfmcopy (DataFrame): Copy of RFM dataframe.

    Returns:
        DataFrame: Processed 'paydf' dataframe.
    """
    # Log the initial size of the input dataframes
    logging.info(f"Input dataframe 'df' has {len(df)} rows and {df.shape[1]} columns.")
    logging.info(f"Input dataframe 'rfmcopy' has {len(rfmcopy)} rows and {rfmcopy.shape[1]} columns.")

    # Analyzing installments
    installments = df.groupby(
        by=['customer_id'], as_index=False)['payment_installments'].mean()
    logging.info("Processed 'installments' dataframe shape: %s", installments.shape)

    # Analyzing payment types
    paymentty = df.groupby(
        by=['customer_id'], as_index=False)['payment_type'].max()
    logging.info("Processed 'paymentty' dataframe shape: %s", paymentty.shape)

    w = installments.iloc[:, [1]]
    w.reset_index(drop=True, inplace=True)

    n = paymentty.iloc[:, [1]]
    n.reset_index(drop=True, inplace=True)

    # Concatenating data
    paydf = pd.concat([rfmcopy, w, n], axis=1)

    # Confirming final dataframe characteristics
    logging.info("Final 'paydf' dataframe shape: %s", paydf.shape)
    logging.info("Head of 'paydf':\n%s", paydf.head())

    # Saving df in dataframes folder
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        paydf.to_csv(os.path.join(reports_path, 'dataframes', f'paydf_{now}.csv'), index=False)

    except Exception as e:
        logging.error('Error saving DataFrame to CSV: %s', str(e))
        return

    return paydf


def customers_insights(paydf, nclusterskmeans):
    """
    Analyze payment types and average installments based on the input dataframe and number of clusters.

    Args:
        paydf (DataFrame): Input dataframe.
        nclusterskmeans (int): Number of clusters.

    Returns:
        DataFrame: Processed 'paydict', 'paydict2', and 'paydict3' dataframes.
    """
    logging.debug("Starting customers_insights function...")

    paydict = {}
    for i in range(nclusterskmeans):
        countpay = paydf[paydf['kmeans_cluster'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['kmeans_cluster'] == i]['payment_installments'].mean()
        paydict[i+1] = {'cluster'+str(i+1): [countpay, meaninst]}

        logging.info("The payment distribution for the cluster made by cluster%d of kmeans is:\n%s", i, countpay)
        logging.info("The average installments made by customers in cluster%d is %s", i, meaninst)
        logging.debug("---------------------------------")

    paydict2 = {}
    for i in range(nclusterskmeans):
        countpay = paydf[paydf['hc_clusters'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['hc_clusters'] == i]['payment_installments'].mean()
        paydict2[i+1] = {'cluster'+str(i+1): [countpay, meaninst]}

        logging.info("The payment distribution for the cluster cluster%d of HC is:\n%s", i, countpay)
        logging.info("The average installments made by customers in cluster%d is %s", i, meaninst)
        logging.debug("---------------------------------")

    paydict3 = {}
    for i in range(nclusterskmeans):
        countpay = paydf[paydf['sp_clusters'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['sp_clusters'] == i]['payment_installments'].mean()
        paydict3[i+1] = {'cluster'+str(i+1): [countpay, meaninst]}

        logging.info("The payment distribution for the cluster%d of Spectral is:\n%s", i, countpay)
        logging.info("The average installments made by customers in cluster%d is %s", i, meaninst)
        logging.debug("---------------------------------")

    if not (paydict and paydict2 and paydict3):
        logging.warning("One or more of the paydicts is empty.")

    paydict, paydict2, paydict3 = pd.DataFrame(paydict), pd.DataFrame(paydict2), pd.DataFrame(paydict3)
    logging.info("Finished customers_insights function.")

    return paydict, paydict2, paydict3


def recency(recency):
    """
    Analyze the recency distribution and plot a histogram.

    Args:
        recency (DataFrame): Input dataframe containing the recency data.

    Returns:
        None
    """
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
    """
    Analyze payment insights and plot histograms and countplots.

    Args:
        df (DataFrame): Input dataframe.

    Returns:
        DataFrame: Payment distribution dataframe.
    """
    logging.info("Starting payments_insights function...")
    if df.empty:
        logging.warning("Input DataFrame is empty.")
        return None

    logging.debug("Plotting histogram and countplot for payments insights...")
    sns.histplot(data=df["payment_installments"])
    plt.show()
    plt.close()

    sns.countplot(x=df["payment_type"])
    plt.show()
    plt.close()
    logging.info("Payments insights displayed.")

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

    # Saving plot
    logging.info('Getting plot...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports', 'figures'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving plot...')

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(reports_path, f'paymentinsights_{now}.png'))

    except Exception as e:
        logging.error('Error saving Image: %s', str(e))
        return

    logging.info('Image saved successfully.')
    plt.close()

    return paymentdistr


def prod_insights(df):
    """
    Analyze product insights and plot countplot.

    Args:
        df (DataFrame): Input dataframe.

    Returns:
        None
    """
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

    # Saving plot
    logging.info('Getting plot...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving plot...')

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(reports_path, 'figures', f'productinsights_{now}.png'))

    except Exception as e:
        logging.error('Error saving Image: %s', str(e))
        return

    logging.info('Image saved successfully.')
    plt.close()



def customer_geography(df):
    logging.info("Starting customer_geography function...")
    dfgeo = pd.value_counts(df['customer_state']).iloc[:20]

    if dfgeo.empty:
        logging.error("Geo DataFrame is empty.")
        return None

    dfticks = dfgeo.index.to_list() 
    dfgeo.plot()
    plt.title("Customers per state")
    plt.show()

    l = []
    for i in range(20):
        l.append(df[df["customer_state"] == dfticks[i]]["payment_value"].mean())
    
    logging.debug("Generating lineplot for customer geography...")
    ax = sns.lineplot(x=dfticks, y=l)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
    logging.info("Customer geography plot displayed.")

    #saving plot
    logging.info('Getting plot...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving plot...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(reports_path, 'figures', f'customergeography_{now}.png'))

    except Exception as e:
        logging.error(f'Error saving Image: {str(e)}')
        return
        
    logging.info('Image saved successfully.')
    plt.close()
    return dfgeo
