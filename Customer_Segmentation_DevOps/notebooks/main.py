
#importing libraries

import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.cluster import KMeans 
from sklearn.model_selection import GridSearchCV
import scipy.cluster.hierarchy as sch         
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import logging 
import os
import sys
from datetime import datetime
import random as rand
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score



current_path = os.getcwd()
data_folder = os.path.abspath(os.path.join(current_path, '..', 'data', 'external'))
data_filepath = os.path.join(data_folder, 'customer_segmentation.csv')


# ## Data preparation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(filepath):
    """
    Load data from a CSV file, remove duplicates, and drop specific columns.
    """
    logging.info('Preparing data...')
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error("File not found: {}".format(filepath))
        return
    
    df.drop_duplicates(inplace=True)
    cat =  ['product_name_lenght','product_description_lenght','shipping_limit_date','product_category_name']
    df.drop(columns=cat, axis=1, inplace=True)
    logging.debug('Data after removing duplicates and dropping columns:\n{}'.format(df.head()))
    logging.info('Data prepared successfully.')
    return df


def drop_c_id(df):
    """
    Drop the customer_unique_id column.
    """
    logging.info('Dropping customer id...')
    logging.debug('Number of unique customer ids before dropping: {}'.format(df['customer_id'].nunique()))
    
    if 'customer_unique_id' in df.columns:
        df.drop(columns='customer_unique_id', inplace=True)
    else:
        logging.warning("'customer_unique_id' not found in DataFrame.")
        
    logging.debug('Number of unique customer ids after dropping: {}'.format(df['customer_id'].nunique()))
    logging.debug('Data after dropping customer id:\n{}'.format(df.head()))
    logging.info('Customer id dropped successfully.')
    return df 


def clean_data(df):
    """
    Filter data by order status and sample a fraction.
    """
    logging.info('Cleaning data...')
    
    df = pd.DataFrame(df)
    df = df[df['order_status'] == 'delivered']
    df = df.sample(frac=0.1, random_state= rand.randint(0, 1000))
    logging.debug('Data after filtering by order status and sampling:\n{}'.format(df.head()))
    logging.info('Data cleaned successfully.')
    return df


def get_df(df):
    logging.info('Getting DataFrame...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(os.path.join(reports_path, 'dataframes', f'initialdata_{now}.csv'), index=False)

    except:
        logging.error('Error saving DataFrame to CSV.')
        return
        
    logging.info('DataFrame saved successfully.')

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



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_frequencies(df):
    logging.info("Computing frequencies.")
    frequencies = df.groupby(by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    return frequencies

def get_recency(df):
    logging.info("Computing recency.")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    recency = df.groupby(by='customer_id', as_index=False)['order_purchase_timestamp'].max()
    recency.columns = ['Customer ID', 'Latest Purchase']
    recent_date = recency['Latest Purchase'].max()
    recency['Recency'] = recency['Latest Purchase'].apply(lambda x: (recent_date - x).days)
    recency.drop(columns=['Latest Purchase'], inplace=True)
    return recency

def get_monetary(df):
    logging.info("Computing monetary values.")
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = [' Monetary Customer ID', 'Monetary value']
    return monetary 

def concatenate_dataframes_(recency, monetary, frequencies):
    logging.info("Concatenating recency, monetary, and frequencies dataframes.")
    rfm_dataset = pd.concat([recency, monetary['Monetary value'], frequencies['Frequency']], axis=1)
    if rfm_dataset.isnull().sum().any():
        logging.warning(f"Detected missing values after concatenation. Number of missing values: {rfm_dataset.isnull().sum().sum()}")
    rfm_dataset.dropna(inplace=True)
    logging.info("Dataframes concatenated successfully.")
    
    return rfm_dataset


def get_rfm_dataset(rfm_dataset):
    logging.info('Getting DataFrame...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'rfmdata_{now}.csv'
        rfm_dataset.to_csv(os.path.join(reports_path, 'dataframes', filename), index=False)

    except:
        logging.error('Error saving DataFrame to CSV.')
        return
        
    logging.info('DataFrame saved successfully.')
    return rfm_dataset



#kmeans
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

def elbow_method(rfm_dataset):
    logging.info("Starting Elbow Method")

    features = ['Recency','Monetary value','Frequency']
    wcss = []

    X = rfm_dataset[features]

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    logging.info("Elbow Method completed")
    
    return X, features 


def get_best_kmeans_params(X):
    logging.info("Starting GridSearchCV for KMeans parameters")

    params = {
        'algorithm': ['lloyd', 'elkan'],
        'n_init': [i for i in range(1, 15)],
        'n_clusters': [i for i in range(3, 6)]
    }

    kmeans = KMeans()
    clf = GridSearchCV(estimator=kmeans, param_grid=params).fit(X)

    cv_results = pd.DataFrame(clf.cv_results_)
    logging.info(f"The top parameters to tune into Kmeans are: {clf.best_params_}")

    logging.info("GridSearchCV for KMeans parameters completed")
    
    return clf.best_params_


def silhouette_score_f(X, y, method):
    logging.info(f"Calculating Silhouette Score for {method}")
    
    results = y[method]
    silscores = {}
    silscores[method] = silhouette_score(X, results, metric='euclidean')  
    silsc = silhouette_score(X, results, metric='euclidean')  
    logging.info(f"The silhouette score for {method} is: {silscores[method]}")
    
    return silscores, silsc 


def clustering(clusters1, algorithm1, rand_state, X, df):
    logging.info("Starting clustering")

    kmeans = KMeans(n_clusters=clusters1, init='k-means++', random_state=rand_state, algorithm=algorithm1, n_init=3)
    y_kmeans = kmeans.fit_predict(X)

    rfmcopy = df.copy()
    rfmcopy['kmeans_cluster'] = y_kmeans

    logging.info("Clustering completed")

    return rfmcopy


def choose(rfm_dataset, X):
    logging.info("Starting cluster selection")

    nclusters = [3, 4, 5, 6]
    algo = ["lloyd", "elkan"]
    inp1 = int(input("Please insert the number of clusters you would like to have: "))
    if inp1 not in nclusters:
        logging.warning("Not recommended nclusters. Please insert an integer between 3 and 6 for an optimal result.")
        inp1 = int(input("Please insert the number of clusters you would like to have: "))
    inp2 = str(input("Choose 'lloyd' or 'elkan': "))
    if inp2 not in algo:
        logging.warning("Invalid algorithm choice. Please type either 'lloyd' or 'elkan'.")
        inp2 = str(input("Choose 'lloyd' or 'elkan': "))

    inp3 = int(input("Please insert a random state (integer): "))
    if type(inp3) != int:
        logging.warning("Random state must be an integer. Please reinsert.")
        inp3 = int(input("Reinsert a random integer: "))

    rfmcopy = clustering(inp1, inp2, inp3, X, rfm_dataset)
    logging.info("Cluster selection completed")
    
    return rfmcopy, inp1



#agglomerative
logging.basicConfig(level=logging.INFO)  # Set the desired logging level
def dendrogram(X):
    logging.info("Starting Dendrogram generation")

    Dend = sch.dendrogram(sch.linkage(X, method="ward"))
    plt.title("Dendrogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distances")
    plt.xticks([])  # No ticks are displayed
    plt.show()

    logging.info("Dendrogram generation completed")

    return Dend


def agglomerative_clustering(X, rfmcopy, n_clustersagg):
    logging.info("Starting Agglomerative Clustering")

    hc = AgglomerativeClustering(n_clusters=n_clustersagg, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    
    logging.info("Clustering completed")

    plot2 = go.Figure()
    rfmcopy['hc_clusters'] = y_hc

    n2clusters = sorted(list(rfmcopy['hc_clusters'].unique()))
    for x in n2clusters:
        logging.info(f"Plotting data for Cluster {x+1}")
        plot2.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.hc_clusters == x]['Recency'], 
                                     y=rfmcopy[rfmcopy.hc_clusters == x]['Frequency'],
                                     z=rfmcopy[rfmcopy.hc_clusters == x]['Monetary value'],  
                                     mode='markers', marker_size=8, marker_line_width=1,
                                     name='Cluster ' + str(x+1)
                                     ))
                                                    
    plot2.update_layout(width=800, height=800, autosize=True, showlegend=True,
                       scene=dict(xaxis=dict(title='Recency', titlefont_color='black'),
                                  yaxis=dict(title='Frequency', titlefont_color='black'),
                                  zaxis=dict(title='Monetary value', titlefont_color='black')),
                       font=dict(family="Gilroy", color='black', size=12))

    plot2.show()
    
    logging.info("Agglomerative Clustering completed")
    
    return y_hc


#spectral
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

def show_silscores(silscores):
    logging.info("Displaying silhouette scores")
    
    dfscores = pd.DataFrame(silscores, index=[0])
    print(dfscores)
    
    logging.info("Silhouette scores displayed")


def choose_spectral():
    logging.info("Starting Spectral Clustering selection")

    #instead of nclusters, ask for n_neighbors and affinity
    n_neighbors = [i for i in range(3, 10)]
    affinity = ['nearest_neighbors', 'rbf', 'precomputed']
    inp1 = int(input("Please insert the number of neighbors you would like to have: "))
    if inp1 not in n_neighbors:
        logging.warning("Not recommended n_neighbors. Please insert an integer between 3 and 10 for an optimal result.")
        inp1 = int(input("Please insert the number of neighbors you would like to have: "))
    
    inp2 = str(input("Choose 'nearest_neighbors', 'rbf' or 'precomputed': "))
    if inp2 not in affinity:
        logging.warning("Invalid affinity choice. Please type either 'nearest_neighbors', 'rbf' or 'precomputed'.")
        inp2 = str(input("Choose 'nearest_neighbors', 'rbf' or 'precomputed': "))

    return inp1, inp2 


def spectral_clustering(X, nclusters, affinity, neighbors):
    logging.info("Starting Spectral Clustering")

    spectral = SpectralClustering(n_clusters=nclusters, random_state=42, n_neighbors=neighbors, affinity='nearest_neighbors')
    sp = spectral.fit_predict(X)

    sil_score = silhouette_score(X, sp, metric='euclidean')

    logging.info("Spectral Clustering completed")

    return sp, sil_score


#pca
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

def pca_kmeans(sc_features, scores, nclusterspca):
    logging.info("Starting PCA and K-Means clustering")

    kmeanspca = KMeans(n_clusters=nclusterspca, init="k-means++", random_state=42)
    kmeanspca.fit(scores)

    sc_features2 = sc_features.iloc[:, [0, 1, 2]]
    segmkmeans = pd.concat([sc_features2, pd.DataFrame(scores)], axis=1)
    segmkmeans.columns.values[-3:] = ['component1', 'component2', 'component3']
    segmkmeans['kmeansclusters'] = kmeanspca.labels_

    logging.info("PCA and K-Means clustering completed")

    return segmkmeans, kmeanspca

def pca_components(segmkmeans, kmeanspca, rfmcopy):
    logging.info("Starting PCA components visualization")

    x = segmkmeans['component2']
    y = segmkmeans['component1']
    sns.scatterplot(x=x, y=y, hue=segmkmeans['kmeansclusters'])

    plt.title("Clusters detected by PCA")
    plt.show()

    dfpca = rfmcopy.copy()
    dfpca['kmeansclustersPCA'] = kmeanspca.labels_

    logging.info("PCA components visualization completed")

    return dfpca


def pca_insights(dfpca):
    logging.info("Starting PCA insights")

    f2 = ['Recency', 'Monetary value', 'Frequency']
    first = dfpca[dfpca['kmeansclustersPCA'] == 0][f2]
    sec =  dfpca[dfpca['kmeansclustersPCA'] == 1][f2]
    th = dfpca[dfpca['kmeansclustersPCA'] == 2][f2]
    four = dfpca[dfpca['kmeansclustersPCA'] == 3][f2]
    
    logging.info("Describing Cluster 0")
    first_description = first.describe()
    logging.info("Describing Cluster 1")
    sec_description = sec.describe()
    logging.info("Describing Cluster 2")
    th_description = th.describe()
    logging.info("Describing Cluster 3")
    four_description = four.describe()
    
    logging.info("PCA insights completed")
    
    return first_description, sec_description, th_description, four_description

def pca_insights2(df, dfpca):
    logging.info("Starting PCA insights 2")

    customer_payment = df.groupby(by='customer_id', as_index=False)['payment_type'].max()
    customer_installments = df.groupby(by='customer_id', as_index=False)['payment_installments'].mean()
    customer_city = df.groupby(by='customer_id', as_index=False)['customer_state'].max()
    product_category = df.groupby(by='customer_id', as_index=False)['product_category_name_english'].max()

    e = customer_payment.iloc[:, [1]]
    e.reset_index(drop=True, inplace=True)
    r = customer_installments.iloc[:, [1]]
    r.reset_index(drop=True, inplace=True)
    q = customer_city.iloc[:, [1]]
    q.reset_index(drop=True, inplace=True)
    t = product_category.iloc[:, [1]]
    t.reset_index(drop=True, inplace=True)

    temp = pd.concat([e, r, q, t], axis=1)

    temp.reset_index(drop=True, inplace=True)
    dfpcaf = pd.concat([dfpca, temp], axis=1)
    
    logging.info("PCA insights 2 completed")
    
    return dfpcaf


#silhouette score (performance)

def silhouette_score_df(silscores):
    logging.info('Getting DataFrame...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'silscores_{now}.csv'
        silscores.to_csv(os.path.join(reports_path, 'dataframes', filename), index=False)
        sns.displot(silscores)
        plt.show()
    except:
        logging.error('Error saving DataFrame to CSV.')
        return
        
    logging.info('DataFrame saved successfully.')
    return silscores 


#descriptive stats
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

def kmeans_summary(rfmcopy, nclusterskmeans):
    
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

    #saving df...
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'kmeanssummary_{now}.csv'
        Kmeanssummary.to_csv(os.path.join(reports_path, 'dataframes', filename), index=False)

    except:
        logging.error('Error saving DataFrame to CSV.')
        return
        
    logging.info('DataFrame saved successfully.')
    
    return Kmeanssummary 



def cluster_summary(df, column_name):
    # Log the initial size and column details of the input dataframe
    logging.info(f"Input dataframe has {len(df)} rows and {df.shape[1]} columns.")
    logging.info(f"Calculating summaries based on the column '{column_name}'.")

    # Kmeans summary
    kmeans_size = df.groupby('kmeans_cluster')[column_name].size()
    kmeans_sum = df.groupby('kmeans_cluster')[column_name].sum()
    kmeans_mean = df.groupby('kmeans_cluster')[column_name].mean()
    kmeans_sd = df.groupby('kmeans_cluster')[column_name].std()
    Kmeanssummary = pd.DataFrame({
        'Clustersize': kmeans_size,
        'Column sum': kmeans_sum,
        f'Average of {column_name}': kmeans_mean,
        f'Column {column_name} sd': kmeans_sd
    })
    logging.info(f"KMeans summary processed with {len(Kmeanssummary)} clusters.")

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
    
    #getting all the info
    logging.info('Getting DataFrames...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Kmeanssummary.to_csv(os.path.join(reports_path, 'dataframes', f'Kmeans_summary{now}.csv'), index=False)
        Hcsummary.to_csv(os.path.join(reports_path, 'dataframes', f'Hc_summary{now}.csv'), index=False)
        Spsummary.to_csv(os.path.join(reports_path, 'dataframes', f'Sp_summary{now}.csv'), index=False)
    
    except:
        logging.error('Error saving DataFrame to CSV.')
        return
        
    logging.info('DataFrame saved successfully.')
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


def customers_insights(paydf, nclusterskmeans):

    logging.debug("Starting customers_insights function...")

    paydict = {}
    for i in range(nclusterskmeans):
        countpay = paydf[paydf['kmeans_cluster'] == i]['payment_type'].value_counts()
        meaninst = paydf[paydf['kmeans_cluster'] == i]['payment_installments'].mean()
        paydict[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}

        logging.info(f"The payment distribution for the cluster made by cluster{i} of kmeans is:\n{countpay}")
        logging.info(f"The average installments made by customers in cluster{i} is {meaninst}")
        logging.debug("---------------------------------")

    paydict2 = {}
    for i in range(nclusterskmeans):
        countpay = paydf[paydf['hc_clusters'] == i]['payment_type'].value_counts() 
        meaninst = paydf[paydf['hc_clusters'] == i]['payment_installments'].mean() 
        paydict2[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}

        logging.info(f"The payment distribution for the cluster cluster{i} of HC is:\n{countpay}")
        logging.info(f"The average installments made by customers in cluster{i} is {meaninst}")
        logging.debug("---------------------------------")

    paydict3 = {}
    for i in range(nclusterskmeans):
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
    pltpay = plt.plot(x, y)
    plt.xlabel("Payment Type")
    plt.ylabel("Average Price")
    plt.title("Average Spending Distribution by Payment Type")
    plt.show()
    logging.info("Payment insights displayed.")
    
    #saving plot
    logging.info('Getting plot...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving plot...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(reports_path, 'figures', f'paymentsinsights_{now}.png'))

    except:
        logging.error('Error saving Image.')
        return
        
    logging.info('Image saved successfully.')
    plt.close()
    
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

    #saving plot
    logging.info('Getting plot...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving plot...')
    
    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(reports_path, 'figures', f'productinsights_{now}.png'))

    except:
        logging.error('Error saving Image.')
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

    except:
        logging.error('Error saving Image.')
        return
        
    logging.info('Image saved successfully.')
    plt.close()
    return dfgeo


def describe_dataset(rfm_dataset):
    logging.info('Starting describe_dataset function...')
    
    # You can either log the description or continue to print it.
    description = rfm_dataset.describe()
    logging.info('\n' + str(description))
    # or if you prefer to print: 
    # print(description)

    logging.info('describe_dataset function completed.')

def corr(df):
    logging.info('Starting corr function...')
    
    columns = ["payment_type", "payment_installments", "payment_value"]
    
    logging.debug('Generating pairplot for columns: %s', ', '.join(columns))
    sns.pairplot(df[columns])
    
    corr_matrix = df[columns].corr()
    logging.info('Correlation matrix generated for columns: %s', ', '.join(columns))
    logging.info('\n' + str(corr_matrix))
    # or if you prefer to print:
    # print(corr_matrix)

    logging.info('corr function completed.')



#visualization
def plot_clusters(rfmcopy, clusters1):
    logging.info("Starting plot_clusters function...")
    plot = go.Figure() 

    nclusters = [i for i in range (0, clusters1)]
    for x in nclusters:
        plot.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.kmeans_cluster == x]['Recency'], 
                                    y=rfmcopy[rfmcopy.kmeans_cluster == x]['Frequency'],
                                    z=rfmcopy[rfmcopy.kmeans_cluster == x]['Monetary value'],  
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name='Cluster ' + str(x+1)
                                    ))
    
    logging.debug(f"Added {len(nclusters)} clusters to the plot.")

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                    scene=dict(xaxis=dict(title='Recency', titlefont_color='black'),
                                yaxis=dict(title='Frequency', titlefont_color='black'),
                                zaxis=dict(title='Monetary value', titlefont_color='black')),
                    font=dict(family="Gilroy", color='black', size=12))

    plt.title("K-Means Clustering")
    plot.show()
    logging.info("Cluster plotting completed.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'Kmeans_clusters_{now}.png'
        plot.write_image(os.path.join(reports_path, 'figures', filename))
    except:
        logging.error('Error saving plot.')
        return


def visualize_spectral_clusters(X, sp):
    logging.info("Starting visualize_spectral_clusters function...")
    plot = go.Figure()
    
    rfmcopy = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary value'])
    rfmcopy['sp_clusters'] = sp
    n_clusters = sorted(list(rfmcopy['sp_clusters'].unique()))

    for x in n_clusters:
        plot.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.sp_clusters == x]['Recency'], 
                                    y=rfmcopy[rfmcopy.sp_clusters == x]['Frequency'],
                                    z=rfmcopy[rfmcopy.sp_clusters == x]['Monetary value'],  
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name='Cluster ' + str(x+1)
                                    ))
    
    logging.debug(f"Added {len(n_clusters)} clusters to the plot.")

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                        scene=dict(xaxis=dict(title='Recency', titlefont_color='red'),
                                   yaxis=dict(title='Frequency', titlefont_color='blue'),
                                   zaxis=dict(title='Monetary value', titlefont_color='green')),
                        font=dict(family="Gilroy", color='black', size=12))
    
    plt.title("Spectral Clustering clusters")
    plot.show()
    logging.info("Spectral cluster visualization completed.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'spectral_clusters_{now}.png'
        plot.write_image(os.path.join(reports_path, 'figures', filename))
    except:
        logging.error('Error saving plot.')
        return


def plot_clusters_pca(rfmcopy, clusterspca):
    logging.info("Starting plot_clusters_pca function...")
    plot = go.Figure()

    nclusters = [i for i in range(0, clusterspca)]
    for x in nclusters:
        plot.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.pca_clusters == x]['Recency'], 
                                    y=rfmcopy[rfmcopy.pca_clusters == x]['Frequency'],
                                    z=rfmcopy[rfmcopy.pca_clusters == x]['Monetary value'],  
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name='Cluster ' + str(x+1)
                                    ))
    
    logging.debug(f"Added {len(nclusters)} clusters to the plot.")

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                    scene=dict(xaxis=dict(title='Recency', titlefont_color='black'),
                                yaxis=dict(title='Frequency', titlefont_color='black'),
                                zaxis=dict(title='Monetary value', titlefont_color='black')),
                    font=dict(family="Gilroy", color='black', size=12))

    plt.title("PCA Clustering clusters")
    plot.show()
    logging.info("PCA cluster plotting completed.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'pca_clusters_{now}.png'
        plot.write_image(os.path.join(reports_path, 'figures', filename))
    except:
        logging.error('Error saving plot.')
        return
    
    

def visualize_data(rfm_dataset):
    logging.info("Visualizing data...")
    sns.pairplot(rfm_dataset)
    plt.title("Pairplot")
    plt.show()
    
    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'pairplot_{now}.png'
        #fig1 = plot3.get_figure()
        #fig1.savefig(os.path.join(reports_path, 'figures', filename))
        plt.savefig(os.path.join(reports_path, 'figures', filename)) 
    except:
        logging.error('Error saving plot.')
        return

    plt.close()
        
    
    sns.lineplot(x="Recency", y="Monetary value", data=rfm_dataset.sort_values(by=["Recency"], ascending=False))
    plt.title("Spending by recency")
    plt.show()

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename2 = f'recency_monetary_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename2))
    except:
        logging.error('Error saving plot.')
        return

    plt.close()

    sns.histplot(data=rfm_dataset['Frequency'], discrete=True)
    plt.title("Frequency plot")
    plt.show()

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename3 = f'frequencyplot_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename3))
    except:
        logging.error('Error saving plot.')
        return

    plt.close()
    logging.info("Data visualization complete.")
    return plt  


def plot_average_spending_by_frequency(rfm_dataset):
    logging.info("Plotting average spending by frequency...")
    frd = rfm_dataset.groupby(['Frequency'])['Monetary value'].mean().reset_index(name='Average Spending by frequency')
    sns.scatterplot(data=frd, x="Frequency", y="Average Spending by frequency", s=100, color='red')
    plt.title("Average Spending by Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Average Spending")
    plt.show()
    logging.info("Average spending by frequency plotted.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'average_spending_by_frequency_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except:
        logging.error('Error saving plot.')
        return

    plt.close()


def plot_payment_value_distribution(rfm_dataset):
    logging.info("Plotting payment value distribution...")
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.distplot(rfm_dataset['Monetary value'], kde=False, bins=newbins)
    plt.title("Payment value distribution")
    plt.ylabel("Count")
    plt.show()
    logging.info("Payment value distribution plotted.")

   #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'plot_payment_value_distribution{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except:
        logging.error('Error saving plot.')
        return

def freq(rfm_dataset):
    logging.info("Generating frequency plots...")
    sns.histplot(data=rfm_dataset['Frequency'], discrete=True)
    plt.title("Frequency plot")
    plt.show()
    logging.info("Frequency plots generated.")


def pairplot(rfm_dataset):
    logging.info("Generating pairplot...")
    sns.pairplot(rfm_dataset)
    plt.title("Pairplot")
    plt.show()
    logging.info("Pairplot generated.")


def spending_by_recency(rfm_dataset):
    logging.info("Generating spending by recency plot...")
    timeordered = rfm_dataset.sort_values(by=["Recency"], ascending=False)
    plt1 = sns.lineplot(x="Recency", y="Monetary value", data=timeordered)
    plt.title("Spending by recency")
    plt.show()
    logging.info("Spending by recency plotted.")


def payments_distribution(rfm_dataset):
    logging.info("Plotting payments distribution...")
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.displot(rfm_dataset['Monetary value'], kde=False, bins=newbins)
    plt.title("Payments distribution")
    plt.show()
    logging.info("Payments distribution plotted.")






















