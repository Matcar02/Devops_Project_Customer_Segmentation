import os
import logging
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import mlflow

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def segments_insights(rfmcopy, nclusterskmeans):
    """
    Generate insights by plotting distribution segments using different clustering results.

    Args:
        rfmcopy (pd.DataFrame): The DataFrame containing RFM data with clustering labels.
        nclusterskmeans (int): The number of KMeans clusters.

    Returns:
        None: Displays various distribution plots based on clustering results.
    """

    logging.info('Starting analysis on the given rfmcopy data.')

    # Copy the rfm data for manipulation
    scaledrfm = rfmcopy.copy()
    
    logging.debug('Successfully copied the rfmcopy data to scaledrfm.')

    # Set color palette and plot distribution for KMeans clustering
    sns.set_palette("Dark2")
    # Distribution plots for KMeans clusters
    logging.debug('Plotting kmeans_cluster based segments...')
    km1 = sns.displot(data = scaledrfm , x = 'Monetary value', hue = "kmeans_cluster", multiple = "stack")
    km2 = sns.displot(data = scaledrfm , x = "Frequency" , hue = "kmeans_cluster", multiple = "stack")
    km3 = sns.displot(data = scaledrfm , x = "Recency" , hue = "kmeans_cluster", multiple = "stack")
    plt.show()

    # Save the plot as an image file
    filename_kmeans = "kmeans_clusters.png"
    km1.savefig(filename_kmeans)

    # Repeat process for other clustering techniques (Spectral, Hierarchical)

    sns.set_palette("colorblind", 4)
    logging.debug('Plotting sp_clusters based segments...')
    sp1 = sns.displot(data = scaledrfm , x = 'Monetary value', hue = "sp_clusters", multiple = "stack")
    sp2 = sns.displot(data = scaledrfm , x = "Frequency" , hue = "sp_clusters", multiple = "stack")
    sp3 = sns.displot(data = scaledrfm , x = "Recency" , hue = "sp_clusters", multiple = "stack")
    plt.show()

    filename_spectral = "spectral_clusters.png"
    sp1.savefig(filename_spectral)

    sns.set_palette("bright")
    logging.debug('Plotting hc_clusters based segments...')
    hc1 = sns.displot(data = scaledrfm , x = 'Monetary value', hue = "hc_clusters", multiple = "stack")
    hc2 = sns.displot(data = scaledrfm , x = "Frequency" , hue = "hc_clusters", multiple = "stack")
    hc3 = sns.displot(data = scaledrfm , x = "Recency" , hue = "hc_clusters", multiple = "stack")
    plt.show()

    filename_hierarchical = "hierarchical_clusters.png"
    hc1.savefig(filename_hierarchical)

    logging.info('Finished analysis. Returning segmented data.')
    
    return filename_kmeans, filename_spectral, filename_hierarchical


def kmeans_summary(rfmcopy, cluster_num):
    """
    Calculate and summarize statistics for a specific KMeans cluster.

    Args:
        rfmcopy (pd.DataFrame): The RFM dataset with KMeans cluster labels.
        cluster_num (int): The specific cluster number to analyze.

    Returns:
        pd.DataFrame: A DataFrame summarizing the statistics of the specified cluster.
    """
    logging.info(f"Input data has {len(rfmcopy)} records.")

    # Calculate statistical data for the specified cluster    
    cluster_data = rfmcopy[rfmcopy['kmeans_cluster'] == cluster_num]
    
    # Calculate various statistics such as size, sum, mean, etc.
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
    Generate summaries for specified clusters based on a column.

    Args:
        df (pd.DataFrame): The DataFrame containing clustered data.
        column_name (str): The name of the column for which to generate summaries.

    Returns:
        tuple: A tuple of DataFrames, each summarizing a different clustering approach.
    """
    # Log the initial size and column details of the input dataframe
    logging.info(f"Input dataframe has {len(df)} rows and {df.shape[1]} columns.")
    logging.info(f"Calculating summaries based on the column '{column_name}'.")

    # Generate summaries for different clustering techniques (KMeans, Hierarchical, Spectral)

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
    Analyze the installments and payment types in the data.

    Args:
        df (pd.DataFrame): The original DataFrame containing payment details.
        rfmcopy (pd.DataFrame): A copy of the RFM DataFrame for analysis.

    Returns:
        pd.DataFrame: A DataFrame with combined installment and payment type data.
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
    """
    Analyze and visualize customer distribution by state and their average payment value.

    This function creates two plots: 
    1. A bar plot showing the distribution of customers in the top 20 states.
    2. A line plot showing the average payment value for customers in these states.

    Args:
        df (pd.DataFrame): The DataFrame containing customer and payment data.

    Returns:
        pd.Series: A Series containing the count of customers in the top 20 states.
    """

    logging.info("Starting customer_geography function...")

    # Count the number of customers in each state and select the top 20
    dfgeo = pd.value_counts(df['customer_state']).iloc[:20]

    if dfgeo.empty:
        logging.error("Geo DataFrame is empty.")
        return None

    # Plot the distribution of customers in the top 20 states
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

    # Saving plot
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


def main(filepath, nclusterskmeans, cluster_num, column_name, rfmcopy ):
    # Load your data
    

    # Log the description and correlation matrix as an artifact
    with mlflow.start_run() as run:

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        default_filepath = os.path.join(data_dir, 'rfm_data.csv')

        # If filepath is "default" or not provided, use the default filepath
        if not filepath or filepath == "default":
            filepath = default_filepath

        df = pd.read_csv(filepath)
        rfmcopy = df.copy()

        kmeans_filename, spectral_filename, hierarchical_filename = segments_insights(rfmcopy, nclusterskmeans)
        kmeans_summary_output = kmeans_summary(rfmcopy, cluster_num)
        cluster_summary_output = cluster_summary(df, column_name)
        installments_analysis_output = installments_analysis(df, rfmcopy)
        customers_insights_output = customers_insights(installments_analysis_output, nclusterskmeans)
        recency_output = recency(df)
        payments_insights_output = payments_insights(df)
        prod_insights_output = prod_insights(df)
        customer_geography_output = customer_geography(df)

        mlflow.log_artifact(kmeans_filename)
        mlflow.log_artifact(spectral_filename)
        mlflow.log_artifact(hierarchical_filename)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run insights analysis")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the CSV file for analysis")
    parser.add_argument("--nclusterskmeans", type=int, default=5, help="Number of KMeans clusters")
    parser.add_argument("--cluster_num", type=int, default=0, help="Specific KMeans cluster number to analyze")
    parser.add_argument("--column_name", type=str, default="Monetary value", help="Column name for cluster summary")
    parser.add_argument("--rfmcopy", type=str, default="", help="Path to the RFM copy CSV file for analysis")

    args = parser.parse_args()
    main(args.filepath, args.nclusterskmeans, args.cluster_num, args.column_name, args.rfmcopy)