



#!/usr/bin/env python
# coding: utf-8

# # Machine learning project on customer segmentation  
# <h3> Group members: Carucci Matteo, Agudio Tommaso, Natoli Vittorio Alessandro </h3>

# <p style="font-size: 16px" font-family="sans-serif">
# In the first part of the project, we deal with a customers database where customers' orders in Brazil are registered. There are many information stored for each order, including the price spent and also some relevant information about the customer and sellers themselves; Below we will start an explorative data analysis to find out more of the "customer_segmentation" dataset to capture some interesting trends and patterns.

# In[2]:

import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 

import pandas as pd

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['product_name_lenght', 'product_description_lenght', 'shipping_limit_date', 'product_category_name'], axis=1, inplace=True)
    return df
df = pd.read_csv('customer_segmentation.csv')
df.drop_duplicates(inplace = True)
df.columns






# ## Data preparation

# In[3]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def drop_columns(df):
    cat =  ['product_name_lenght','product_description_lenght','shipping_limit_date','product_category_name']
    df.drop(columns = cat , axis = 1, inplace= True) 
    return df


def encode_df(df):
    transformer = make_column_transformer(
        (OneHotEncoder(sparse= False), ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english']),
        remainder='passthrough')

    encoded_df = transformer.fit_transform(df)
    encoded_df = pd.DataFrame(
        encoded_df, 
        columns=transformer.get_feature_names_out()  #getting the feature_names to get it better
    )
    return encoded_df



def get_dummies_df(df):
    dummies_df = pd.get_dummies(df, columns = ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name_english'])
    return dummies_df



def drop_c_id(df):
    df.shape
    df['customer_unique_id'].nunique()    #we will drop this column as it sis not useful for our analysis
    df['customer_id'].nunique()
    df.drop(columns = 'customer_unique_id',inplace= True)

    df.head()

#Checking the features that are categorical
categorical_features = df.select_dtypes(include='O').keys()

#Displaying those features
categorical_features

# Before starting analyzing the dataset, we want to introduce and start working on the real goal of the project; segmenting customers using clusters which identify similarities in consumer behaviour, based on the dataset information.


def clean_data(df):
    #dropping order which have not been delivered is important. Instead of dropping, we consider df as the dataframe having only
    #orders delivered
    df = df[df['order_status'] == 'delivered']
    return df

def get_frequencies(df):
    #grouping by and getting the total money spent by customer
    frequencies = df.groupby(
        by=['customer_id'], as_index=False)['order_delivered_customer_date'].count()
    frequencies.columns = ['Frequencies Customer ID', 'Frequency']
    return frequencies



def get_recency(df):
    #using order_purchase_timestamp instead of delivered_carrier_date.
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    recency = df.groupby(by='customer_id',
                            as_index=False)['order_purchase_timestamp'].max()

    recency.columns = ['Customer ID', 'Latest Purchase']

    recent_date = recency['Latest Purchase'].max()

    recency['Recency'] = recency['Latest Purchase'].apply(
        lambda x: (recent_date - x).days)                     
        
    recency.drop(columns=['Latest Purchase'], inplace=True)  #we don't care about the date (we have recency)
    return recency



def concatenate_dataframes(recency, monetary, frequencies):
    rfm_dataset = pd.concat([recency, monetary, frequencies], axis=1)
    cols = [3,5]    #useless columns
    rfm_dataset.drop(columns=rfm_dataset.columns[cols], axis=1, inplace=True)
    rfm_dataset.dropna(inplace=True)   #dropping the nulls, if any
    return rfm_dataset


#     A frequency histplot can clear up this. In the second part we will also see how recent purchases have been way higher.

    sns.histplot(data = rfm_dataset['Frequency'], discrete= True)


# <i style ="font-size: 15px"> Frequency histplot. There are many people who have only purchased once

# <p style="font-size: 16px" font-family="sans-serif">Let us also see how people spent based on the frequency. That will confirm if people who buy more, spend also more on average.

# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 



def get_monetary(df):
    #grouping by and getting the total money spent by customer
    monetary = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary.columns = [' Monetary Customer ID', 'Monetary value']
    return monetary 



def visualize_data(rfm_dataset):
    #pairplot in new RFm dataset
    sns.pairplot(rfm_dataset)
    plot1 = sns.lineplot(x="Recency", y="Monetary value", data=rfm_dataset.sort_values(by=["Recency"], ascending=False))

    #     A frequency histplot can clear up this. In the second part we will also see how recent purchases have been way higher.
    plot2 = sns.histplot(data = rfm_dataset['Frequency'], discrete= True)
    return plot1, plot2  

def elbow_method(rfm_dataset):
    features = ['Recency','Monetary value','Frequency']
    #X = rfm_dataset[features]
    wcss = []

    X = rfm_dataset[features] #numerical values non-scaled
    #X = sc_features[features] #scaled

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return X, features 

def plot_average_spending_by_frequency(rfm_dataset):
    frd = rfm_dataset.groupby(['Frequency'])['Monetary value'].mean().reset_index(name='Average Spending by frequency')#.sort_values(['Spending by frequency'], ascending = False)
    sns.lineplot(data = frd, x = "Frequency", y = "Average Spending by frequency")

def plot_payment_value_distribution(rfm_dataset):
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()),np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    #applied logarithmic bins to get better visualization
    sns.distplot(rfm_dataset['Monetary value'], kde=False, bins=newbins) 

def describe_dataset(rfm_dataset):
    print(rfm_dataset.describe())
# In[23]:




# <i style ="font-size: 15px"> 
# From the elbow method we can see that either 4 and 5 are the best number of clusters. In order to make sure that we will pick the best among all, we use Gridsearch to better interpret and tune the parameters: number of clusters, clustering algorithm and number of iterations for centroids are to be tuned in K-means!

# In[24]:


from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

def get_best_kmeans_params(X):
    params = {
        'algorithm': ['lloyd', 'elkan'],
        'n_init': [i for i in range(1, 15)],
        'n_clusters': [i for i in range(3, 6)]
    }

    kmeans = KMeans()
    clf = GridSearchCV(estimator=kmeans, param_grid=params).fit(X)

    cv_results = pd.DataFrame(clf.cv_results_)
    print(f"The top parameters to tune into Kmeans are: {clf.best_params_}")
    return clf.best_params_


# As we can see the parameters likely to produce better results are 'elkan' as algorithm and the number of clusters is 5 and the chosen iterations to find centroids set to 3. We can now retry to fit the model with these hypertuned parameters.
# If one wants to see the results choosing the arbitrary but still appropriate parameters, can use the class below which shows the results for all parameters we tune. </p>
# 
# Below, we create a class named Kmeancust which is a customized version of the kmeans. One can choose the hyperparameters arbitrarly to visualize the points separation(clusters) and judge the ideal set of them. </p>

# In[25]:


import plotly.graph_objs as go  #import pylot for 3d objects!

def clustering(clusters1, algorithm1, rand_state, X, df):
    kmeans = KMeans(n_clusters = clusters1, init = 'k-means++', random_state = rand_state, algorithm = algorithm1, n_init = 3)
    y_kmeans = kmeans.fit_predict(X)

    #add respective client's clusters assigned 
    rfmcopy = df.copy()
    rfmcopy['kmeans_cluster'] = y_kmeans

    return rfmcopy

def plot_clusters(rfmcopy, clusters1):
    #in this way we can visualize the data scattered with real values instead of the scaled ones we use in the kmeans!
    plot = go.Figure()    #fig equivalent in plotly

    #a good thing to do is to use the information we stored in the dataframe! 
    #a loop will let us separate data in clusters!
    nclusters = [i for i in range (0,clusters1)]  #basically getting alist containg the ith clusters ---> e.g Let 4 be the number of clusters chosen then nclusters -> [1,2,3,4] that are the  number of clusters
    for x in nclusters:
        plot.add_trace(go.Scatter3d(x = rfmcopy[rfmcopy.kmeans_cluster == x]['Recency'], 
                                    y = rfmcopy[rfmcopy.kmeans_cluster == x]['Frequency'],
                                    z = rfmcopy[rfmcopy.kmeans_cluster == x]['Monetary value'],  
                                    mode='markers', marker_size = 8, marker_line_width = 1,
                                    name = 'Cluster ' + str(x+1)
                                    ))

    # changing the layout!
    plot.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                    scene = dict(xaxis=dict(title = 'Recency', titlefont_color = 'black'),
                                    yaxis=dict(title = 'Frequency', titlefont_color = 'black'),
                                    zaxis=dict(title = 'Monetary value', titlefont_color = 'black')),
                    font = dict(family = "Gilroy", color  = 'black', size = 12))

    plot.show()

def choose(rfm_dataset, X):
    #Let us try all the feasible options:
    nclusters = [3,4,5,6]
    algo = ["lloyd","elkan"]
    inp1 = int(input("please insert the number of clusters you would like to have:"))
    if inp1 not in nclusters:
        print("not reccomended nclusters, insert integer between 3 and 6 for an optimal result")
        inp1 = int(input("please insert the number of clusters you would like to have:"))
    inp2 = str(input("choose lloyd or elkan:"))
    if inp2 not in algo:
        print("Please type correctly the algorithm to use!")        
        inp2 = str(input("choose lloyd or elkan:"))

    inp3 = int(input("please insert a random state(integer!):"))
    if type(inp3) != int:
        print("Random state must be an integer! Please reinsert")
        inp3 = int(input("reinsert an random integer:"))

    #initialization of the object and function calling, X are the scaled features whilst rfm_dataset is the original one    
    rfmcopy = clustering(inp1, inp2, inp3, X, rfm_dataset)
    plot_clusters(rfmcopy, inp1)
    return rfmcopy



# An interactive visualization of clusters. It is easy to identify the segmentations as there are few high spenders, lots of mid spenders and those who are likely to be at-risk customers or occasional ones.

# Sometimes, if not always, hyperparameters tuning does work in theory, but not in practice. As we clearly know, in unsupervised learning we can not quantify the accuracy of the model as we have not any variable to compare to (y variable). What we are doing though, is to compute the silhouette score, which quantifies how well the clusters are separated by taking the intra cluster distance into account. If we separate customers in 5 categories, we obtain a much lower silhouette score with respect to 3 or 4 clusters; also it seems reasonable as in general, customers are segmented in 3/4 main categories: High spenders, Mid spenders, at-risk customers(those who are likely, for random reasons, to leave the service) and low spenders, those who occasionaly buy. After trying multiple solutions we find 4 clusters as a good compromise, even though 3 clusters present an higher silhouette score, 4 segmentations perfectly represent customers' behaviour.

# In[26]:

from sklearn.metrics import silhouette_samples, silhouette_score 

def silhouette_score_f(X, y, method):
    #silhouette score
    results = y[method]
    silscores = {}
    silscores[method] = silhouette_score(X, results, metric='euclidean')  

    print(f"The silhouette score for {method} is: {silscores[method]}")
    return silscores 

'''
silscores['kmeansPCA'] = silhouette_score(X_, kmeanspca.labels_ , metric='euclidean')
silscores['kmeans'] = silhouette_score(X, y_kmeans, metric='euclidean')
silscores['hc'] = silhouette_score(X, y_hc, metric='euclidean')
silscores['spectral'] = silhouette_score(X, sp, metric='euclidean')

'''
    # ## HC clustering:
    # 
  
    # An alternative way to cluster customers is the hierarchical clustering.
    # Hierarchical Clustering creates clusters in a hierarchical tree-like structure (called a Dendrogram, which we will display below). That means, a subset of similar data is created in a tree-like structure in which the root node corresponds to the entire data, and branches are created from the root node to form several clusters. This will show us again what is the right number of clusters we need to tune. </p>

    # In[27]:


import scipy.cluster.hierarchy as sch         
from sklearn.cluster import AgglomerativeClustering

def dendogram(X):
    Dend = sch.dendrogram(sch.linkage(X, method="ward"))
    plt.title("Dendogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distances")
    plt.xticks([])    #no ticks is displayed
    plt.show()
    return Dend 



# It does seem that the doable range of clusters is 3 to 6. We will try all of them and see what the results look like.

# For the sake of consistency, we will choose 4 as the number of clusters, as it then can be comparable with the results we obtained wth the Kmeans algorithm.
# In this case, there are 2 major hyperaparameters to tune, the linkage (ward and average work similarly), that is, the metrics to use to determine the distance betweeen clusters, and the affinity, what type of distance measure to use to compute such linkage.

# In[28]:


#single and complete do not work well, average and Ward are pretty similar
from sklearn.cluster import AgglomerativeClustering

def agglomerative_clustering(X, rfmcopy):
    hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)  #predicting the clusters
    
    #sc_features['hc_clusters'] = y_hc  #in case we want to use scaled rfm_dataset
    plot2 = go.Figure()
    rfmcopy['hc_clusters'] = y_hc #inserting a new column in rfm corresponding to the i-th cluster for each customer

    n2clusters = sorted(list(rfmcopy['hc_clusters'].unique()))   #same as we did before, the y_hc is a numpy array that returns, for each customer, the corresponding cluster assigned.
    for x in n2clusters:
            plot2.add_trace(go.Scatter3d(x = rfmcopy[rfmcopy.hc_clusters == x]['Recency'], 
                                        y = rfmcopy[rfmcopy.hc_clusters == x]['Frequency'],
                                        z = rfmcopy[rfmcopy.hc_clusters == x]['Monetary value'],  
                                        mode='markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(x+1)
                                        ))
                                                    
                                                
    # changing the layout!

    plot2.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                    scene = dict(xaxis=dict(title = 'Recency', titlefont_color = 'black'),
                                    yaxis=dict(title = 'Frequency', titlefont_color = 'black'),
                                    zaxis=dict(title = 'Monetary value', titlefont_color = 'black')),
                    font = dict(family = "Gilroy", color  = 'black', size = 12))

    plot2.show()
    return y_hc 


# <i style ="font-size: 15px"> 
# At this stage, things do not seem to change with Agglomerative clustering: the hierarchical method has found almost the same clusters of the previous method and this is seen with the silhouette score below too, which is slighter less than the kmeans one.

# In[29]:


# ## Spectral clustering
# 
# <p style="font-size: 16px" font-family="sans-serif">
# Another idea, may be to implement the spectral clustering to check if that works better(especially for a lower number of clusters). As we can notice, the final centroids do not describe the clusters well. They are all pretty close even though clusters are distinctly separated as also shown by the silhouette score. The spectral clustering basically creates an affinity matrix, where each datapoint is compared to others by assessing the "similarity", that is, sklearn builds a graph with datapoints as nodes, and uses the number of common neigbors(nearest) to identify specific communities.

# In[30]:


from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# For the sake of consistency, we will choose 4 as the number of clusters, as it then can be comparable with the results we obtained wth the Kmeans algorithm.
# In this case, there are 2 major hyperaparameters to tune, the linkage (ward and average work similarly), that is, the metrics to use to determine the distance betweeen clusters, and the affinity, what type of distance measure to use to compute such linkage.

#single and complete do not work well, average and Ward are pretty similar



# <i style ="font-size: 15px"> 
# At this stage, things do not seem to change with Agglomerative clustering: the hierarchical method has found almost the same clusters of the previous method and this is seen with the silhouette score below too, which is slighter less than the kmeans one.

# In[29]:


# ## Spectral clustering
# 
# <p style="font-size: 16px" font-family="sans-serif">
# Another idea, may be to implement the spectral clustering to check if that works better(especially for a lower number of clusters). As we can notice, the final centroids do not describe the clusters well. They are all pretty close even though clusters are distinctly separated as also shown by the silhouette score. The spectral clustering basically creates an affinity matrix, where each datapoint is compared to others by assessing the "similarity", that is, sklearn builds a graph with datapoints as nodes, and uses the number of common neigbors(nearest) to identify specific communities.

# In[30]:



import plotly.graph_objs as go
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def spectral_clustering(X):
    # Spectral clustering
    spectral = SpectralClustering(n_clusters=4, random_state=42, n_neighbors=8, affinity='nearest_neighbors')
    sp = spectral.fit_predict(X)

    # Silhouette score
    sil_score = silhouette_score(X, sp, metric='euclidean')

    return sp, sil_score

def visualize_spectral_clusters(X, sp):
    # Plotting the results
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

    # Changing the layout
    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                        scene=dict(xaxis=dict(title='Recency', titlefont_color='red'),
                                   yaxis=dict(title='Frequency', titlefont_color='blue'),
                                   zaxis=dict(title='Monetary value', titlefont_color='green')),
                        font=dict(family="Gilroy", color='black', size=12))

    plot.show()



# This method does not work as good as the others; even though the cluster separation is fine, the clustering  identifies too many high spenders and too few risk-customers, our EDA has shown that actually few people have spent a lot, and many spend the same.

# Finally we will display all the silhouette scores for each algorithm.

# In[31]:


def show_silscores(silscores):
    dfscores = pd.DataFrame(silscores, index = [0])
    print(dfscores)


# In[32]:



# ## Final consideration on the three methods:
# 
# 
# <p style="font-size: 16px" font-family="sans-serif">
# As we have seen, HC and KMeans methods outperform the spectral clustering  on clustering visualization and detection, as shown also by the silhouette score.
# In the SC implementation, the algorithm seems not to distinguish between mid and high spenders, making them a unique community. The other 2 algorithms instead are way more indicative as it is easy to understand how they weighted each of the 3 RFM features to cluster; they emphasized mostly on the monetary value and frequency rather than the recency (which is legit by the way). What we want to do now is to collect further infomation on the whole dataset (the original one) and try to find another way to cluster. Let us first of all collect relevant information about each cluster detected by KMeans algorithm and HC.

# In[33]:

def kmeans_summary(rfmcopy):
        #first
    cl1size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 0])
    cl1sum = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].sum()
    cl1mean = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].mean()
    cl1frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Frequency'].mean()
    #cl1sd = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].min()
    #cl1sd = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].max()
    cl1fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Frequency'].std()
    cl1sd = rfmcopy[rfmcopy['kmeans_cluster'] == 0]['Monetary value'].std()

    #second
    cl2size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 1]) 
    cl2sum = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Monetary value'].sum()
    cl2mean = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Monetary value'].mean()
    cl2frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Frequency'].mean()
    cl2fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Frequency'].std()
    cl2sd = rfmcopy[rfmcopy['kmeans_cluster'] == 1]['Monetary value'].std()


    #third
    cl3size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 2])
    cl3sum =  rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Monetary value'].sum()
    cl3mean = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Monetary value'].mean()
    cl3frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Frequency'].mean()
    cl3fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Frequency'].std()
    cl3sd = rfmcopy[rfmcopy['kmeans_cluster'] == 2]['Monetary value'].std()

    #fourth

    cl4size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 3])
    cl4sum = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Monetary value'].sum()
    cl4mean = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Monetary value'].mean()
    cl4frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Frequency'].mean()
    cl4fsd = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Frequency'].std()
    cl4sd = rfmcopy[rfmcopy['kmeans_cluster'] == 3]['Monetary value'].std()

    #fifth(if it exists)
    '''cl5size = len(rfmcopy[rfmcopy['kmeans_cluster'] == 4])
    cl5sum = rfmcopy[rfmcopy['kmeans_cluster'] == 4]['Monetary value'].sum()
    cl5mean = rfmcopy[rfmcopy['kmeans_cluster'] == 4]['Monetary value'].mean()
    cl5frequency = rfmcopy[rfmcopy['kmeans_cluster'] == 4]['Frequency'].mean() '''




    # In[34]:


    dictio2 = {'Clustersize': [cl1size,cl2size,cl3size, cl4size], 'Total spending by cluster': [cl1sum,cl2sum,cl3sum,cl4sum],
            'Average spending by cluster' : [cl1mean,cl2mean,cl3mean,cl4mean], 'Average frequency by cluster':
            [cl1frequency ,cl2frequency ,cl3frequency ,cl4frequency], 'Frequency std': [cl1fsd,cl2fsd,cl3fsd,cl4fsd], 'Spending sd':
            [cl1sd,cl2sd,cl3sd,cl4sd]}

    Kmeanssummary = pd.DataFrame(dictio2, index = [1,2,3,4])  #include 5 if 5 is chosen
    return Kmeanssummary 

# In[35]:


import pandas as pd

def cluster_summary(df, column_name):
    """
    This function takes a dataframe and a column name as input and returns a summary of the clusters detected by Kmeans, Hierarchical and Spectral clustering.
    """
    # Kmeans clustering
    kmeans_size = df.groupby('kmeans_clusters')[column_name].size()
    kmeans_sum = df.groupby('kmeans_clusters')[column_name].sum()
    kmeans_mean = df.groupby('kmeans_clusters')[column_name].mean()
    kmeans_frequency = df.groupby('kmeans_clusters')['Frequency'].mean()
    kmeans_fsd = df.groupby('kmeans_clusters')['Frequency'].std()
    kmeans_sd = df.groupby('kmeans_clusters')[column_name].std()
    Kmeanssummary = pd.DataFrame({'Clustersize': kmeans_size, 'Total spending by cluster': kmeans_sum,
                                  'Average spending by cluster': kmeans_mean, 'Average frequency by cluster': kmeans_frequency,
                                  'Frequency std': kmeans_fsd, 'Spending sd': kmeans_sd})
    
    # Hierarchical clustering
    hc_size = df.groupby('hc_clusters')[column_name].size()
    hc_sum = df.groupby('hc_clusters')[column_name].sum()
    hc_mean = df.groupby('hc_clusters')[column_name].mean()
    hc_frequency = df.groupby('hc_clusters')['Frequency'].mean()
    hc_fsd = df.groupby('hc_clusters')['Frequency'].std()
    hc_sd = df.groupby('hc_clusters')[column_name].std()
    Hcsummary = pd.DataFrame({'Clustersize': hc_size, 'Total spending by cluster': hc_sum,
                              'Average spending by cluster': hc_mean, 'Average frequency by cluster': hc_frequency,
                              'Frequency std': hc_fsd, 'Spending sd': hc_sd})
    
    # Spectral clustering
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

# Example usage
#Kmeanssummary, Hcsummary, Spsummary = cluster_summary(rfmcopy, 'Monetary value')

# <i style ="font-size: 15px">
# This is a surprising result. Even though kmeans and hierarchical seemed better when visualizing the clusters, the spectral algorithm is much more balanced when it comes to size distribution per cluster and standard deviation in monetary value and frequency ; We notice that both algorithm have detected a small number of people who spent much more than the others but detected a group, based on size and average spending is likely to be made of outliers and hence not useful for our customer segmentation. From these dataframes is also clear that there exists some customer in the third cluster of spectral(high-spenders) who buy frequently and spend much more than others. In the spectral, cluster 1 contains people who have bought just once(presumably looking at the average frequency) and they can be considered as "occasional customers" who should be engaged by the subsidiary and more emails must be sent to them. The second cluster instead contains "at-risk customers" who likely purchased once and less than the other categories by far, it is important to send as many emails as possible to retain them. The third cluster has "top/high-spenders" which buy and spend more than others. The fourth cluster instead, contains "mid-spenders or abitual customers" who have purchased more than once; target emails with specific products should be enough to retain them.

# <p style="font-size: 16px" font-family="sans-serif">
# One can also visualizing clearer the results of the different clusters using displots, to see how the customer population is distributed based on the 3 features we have.

# In[42]:

def segments_insights(rfmcopy):
    #displot of kmeans clusters
    #from sklearn.preprocessing import MinMaxScaler
    scaledrfm = rfmcopy.copy()
    #sc = MinMaxScaler()
    #new = sc.fit_transform(scaledrfm['Monetary value'].array.reshape(-1,1)) #getting data less screwed
    #scaledrfm['Monetary value'] = new
    sns.set_palette("Dark2")
    sns.displot(data = scaledrfm , x = 'Monetary value', hue = "kmeans_cluster", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Frequency" , hue = "kmeans_cluster", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Recency" , hue = "kmeans_cluster", multiple = "stack")


    # <p style="font-size: 16px" font-family="sans-serif">
    # As hierarchical and kmeans present similar characteristics, we will see the distribution of the spectral clustering to check for some differences.

    # In[43]:


    #we will do the same with the spectral clusetring that is more balanced
    sns.set_palette("colorblind", 4)
    sns.displot(data = scaledrfm , x = 'Monetary value', hue = "sp_clusters", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Frequency" , hue = "sp_clusters", multiple = "stack")
    sns.displot(data = scaledrfm , x = "Recency" , hue = "sp_clusters", multiple = "stack")


    # <i style ="font-size: 15px">
    # As we observe, it does not seem convenient to display the smallest clusters(top and high spenders) as separate but rather, as a unique cluster as their population is not significant enough.

    # <p style="font-size: 16px" font-family="sans-serif">
    # Let us first analyze the customer behaviour of low spenders. We will compare and see insightful patterns and understand which algorithm better detected this customer segmentation.
    # To get started, we regroup the clusters in 3 categories, as we noticed that some clusters in the 3 algorithms present very similar characteristics and it is hence not useful to leave clusters with only a few customers (these represent top customers that spend much more than others or have purchased expensive products).

    # In[44]:


    #creating one dataframe for each category detected by algorithms
    feat = ['Recency', 'Frequency', 'Monetary value']
    #low/at-risk/occasional customers
    lowspenderskmeans = rfmcopy[rfmcopy['kmeans_cluster'] == 1][feat]  #getting only rfm features
    lowspendershc = rfmcopy[rfmcopy['hc_clusters'] == 2][feat]
    lowspenderssc = rfmcopy[rfmcopy['sp_clusters'] == 0][feat]



    #mid-spenders 
    midspenderssp = rfmcopy[(rfmcopy['sp_clusters'] == 1) | (rfmcopy['sp_clusters'] == 3)][feat]
    midspendershc = rfmcopy[rfmcopy['hc_clusters'] == 0][feat]
    midspenderskm = rfmcopy[rfmcopy['kmeans_cluster'] == 0][feat]

    #high/top spenders in the case of kmeans and hc we will merge them, as there are few outliers which contribute to a tiny cluster
    #for the sake of completeness, also in the spectral the sthird and fourth will be merged
    highspenderssc = rfmcopy[ (rfmcopy['sp_clusters'] == 2)][feat]
    highspendershc = rfmcopy[(rfmcopy['hc_clusters'] == 3) | (rfmcopy['hc_clusters'] == 1)][feat]
    highspenderskm = rfmcopy[(rfmcopy['kmeans_cluster'] == 2) | (rfmcopy['kmeans_cluster'] == 3)][feat]




    # <p style="font-size: 16px" font-family="sans-serif">
    # As hierarchical and kmeans present very similar results we will choose kmeans due to its slightly higher silhouette score. Below, we will compare clusters of spectral with the kmeans ones

    # In[45]:


    lowspenderskmeans.describe() 


    # In[46]:


    lowspenderssc.describe()


    # Let us know see the midspenders in both kmeans and spectral clustering.

    # In[47]:


    midspenderskm.describe()


    # In[48]:


    midspenderssp.describe()


    # In[49]:


    highspenderskm.describe()


    # In[50]:


    highspenderssc.describe()

    return lowspendershc, lowspenderskmeans, lowspenderssc, midspenderskm, midspenderssp, midspendershc, highspenderskm, highspenderssc, highspendershc 
# <i style ="font-size: 15px">
# The two algorithms took a different approach. Whilst kmeans has taken into account the frequency much more, the spectral clustering seems not to distinguish and give importance to that parameter. Also, kmeans do cluster in a more decisive way; we previously saw that there are top clients who spend much more, but spectral has not detected them but instead, merge with others high spenders.

# ## Further data exploration
# #### In this section we will try to find some common patterns and insights in the dataset to confirm or reassess the results we got in clustering.  As we only used the RFM dataframe which contained little if none information on the products and sellers, now we dive into these aspects.

# <p style="font-size: 16px" font-family="sans-serif">
# Is there correlation in payment methods? Do people spend more when making more installments? what product are the most popular?
# We will try to answer all these questions, pointing out differences in customers also relying on clustering results.

# <p style="font-size: 16px" font-family="sans-serif">
# Let us see if there exists some correlation among payment features, that is, understand if particular payments impy more spending or viceversa in the clusters we obtained.

# In[51]:

def corr(df):
    columns=["payment_type","payment_installments","payment_value"]
    sns.pairplot(df[columns])


    # <p style="font-size: 16px" font-family="sans-serif">
    #  As one can easily see from the pairplot, there is an intersting thing. People who spend less per order, are those who make the more installments, whilst people who spend more per order, make less installment to pay (payment_installments < 10).
    #  It would be interesting to see if there exists correlation between them.

    # In[52]:


    df[columns].corr()


# <i style ="font-size: 15px">
# There is a tiny positive correlation between them; this something not remarkable and hence will not make an impact.

# <p style="font-size: 16px" font-family="sans-serif">
# We now want to see, for each method we used, what is the payment method used per cluster, and how many installments were made on average.

# In[53]:

def installments_analysis(df, rfmcopy):
    #we will consider the mean of installments made by the customer
    installments = df.groupby(
        by=['customer_id'], as_index=False)['payment_installments'].mean()

    #most used payment method by customer
    paymentty = df.groupby(
        by=['customer_id'], as_index=False)['payment_type'].max()

    installments.head()


    # In[54]:


    w = installments.iloc[:, [1]]
    w.reset_index(drop=True, inplace=True)
    n = paymentty.iloc[:,[1]]
    n.reset_index(drop=True, inplace=True)


    # In[55]:


    paydf = pd.concat([rfmcopy,w,n], axis=1) #create the df containing all info.
    paydf.head()       

    return paydf 


# #### After this set up we can now start inspecting some payment features insights for each clusters identified by the 3 algorithms

# In[56]:


#some insights about payment features in kmeans clusters
def customers_insights(paydf):
    clusterstype = ['mid-spenders','at-risk customers', 'top-customers','high-spenders']
    paydict = {}
    for i in range(4):
        countpay = paydf[paydf['kmeans_cluster'] == i]['payment_type'].value_counts() #how payment_type is distributed 
        meaninst = paydf[paydf['kmeans_cluster'] == i]['payment_installments'].mean() #average payment installments per cluster
        paydict[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}
        
        print("")
        print(f"The payment distribution for the cluster made by {clusterstype[i]} of kmeans is")
        print(countpay)
        print("")
        print(f"The average installments made by customers in cluster of {clusterstype[i]} is {meaninst}")
        print("---------------------------------")


    # <p style="font-size: 16px" font-family="sans-serif">
    # The same is done with hierarchical and spectral respectively:

    # In[57]:


    #some insight on payment features in hierarchical
    customersHc = ['Mid-spenders','top customers','at-risk customers', 'high-spenders']
    paydict2 = {}
    for i in range(4):
        countpay = paydf[paydf['hc_clusters'] == i]['payment_type'].value_counts() #how payment_type is distributed 
        meaninst = paydf[paydf['hc_clusters'] == i]['payment_installments'].mean() #average payment installments per cluster
        paydict2[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}
        
        print(f"The payment distribution for the cluster {customersHc[i]} of HC is")
        print(countpay)
        print(f"The average installments made by customers in cluster {customersHc[i]} is {meaninst}")
        print("---------------------------------")




# 
# <p style="font-size: 16px" font-family="sans-serif">Ultimately spectral has:

# In[58]:


    #some insight on payment features in spectral
    customersSp = ['Low spenders', 'at-risk customers','top customers', 'High spenders']
    paydict3 = {}
    for i in range(4):
        countpay = paydf[paydf['sp_clusters'] == i]['payment_type'].value_counts() #how payment_type is distributed 
        meaninst = paydf[paydf['sp_clusters'] == i]['payment_installments'].mean() #average payment installments per cluster
        paydict3[i+1] = {'cluster'+str(i+1):[countpay,meaninst]}
        
        print(f"The payment distribution for the cluster {customersSp[i]} of Spectral is")
        print(countpay)
        print(f"The average installments made by customers in cluster {customersSp[i]} is {meaninst}")
        print("---------------------------------")

    return paydict, paydict2, paydict3


# <i style ="font-size: 15px">
# There is a general trend going on above. Those who are considered top and high spenders prefer using credit cards and boleto as payment method, whilst some low, at-risk customers have used vouchers and hence it is likely that they have got the voucher from someone and are not usual customers who buy frequently; Moreover, one can notice that top and mid spenders make less installments than others. These features can be used to use other methods to identify clusters in a more efficient way, as they seem to contribute when it comes to customer segmentation.

# ### We already saw the frequency and the monetary value of each customer, and their distribution, but what about the purchase recency? Below, an histplot lets us see how many customers have purchased per each period of time.

# In[59]:

def recency(recency):
    recencydist=list(recency["Recency"])
    plt.hist(x=recencydist)
    plt.xlabel('days since last purchase')
    plt.ylabel('number of people per period')
    sns.histplot()


# <p style="font-size: 16px" font-family="sans-serif">
# One can also see what is the payment installments distribution; the majority of customers has only 1 installment and a few people have made multiple installments.

# In[60]:

def payments_insights(df):
    sns.histplot(data=df["payment_installments"])
    # Of course something that may be of interest is the distribution of the payment types and therefore see how people purchase products
    sns.countplot(x=df["payment_type"])
    paymentdistr = df.groupby(['payment_type'])['payment_value'].mean().reset_index(name='Avg_Spending').sort_values(['Avg_Spending'], ascending = False)
    x=["boleto","credit_card","debit_card","voucher"]
    y=paymentdistr["Avg_Spending"]
    plt.plot(x,y)
    plt.xlabel("Payment Type")
    plt.ylabel("Average Price")
    plt.title("Average Spending Distribution by Payment Type")
    return paymentdistr 


# Let us now see the top 15 product categories that customers have purchased

# In[62]:

def prod_insights(df):
    dfcat = pd.value_counts(df['product_category_name_english']).iloc[:15].index 

    ax = sns.countplot(df['product_category_name_english'], order= dfcat)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)


# In[63]:





# <i style ="font-size: 15px">
# As we can see, people who have done an order with the voucher tend to spend less due to the promotions(likely not a sure thing). Instead people who use a boleto spend much more. There isn't much difference between the credit card spending and the debit card while they spend a little bit less then users who use a boleto. This could be because there might be a promotion for those who use a credit or debit card. As we have seen before, the credit card is used much more then any other method.

# One can also be interested in where the customers come from. We will regroup the first 15 states and also see their average spending per customer!

# In[131]:

def customer_geography(df):
    dfgeo = pd.value_counts(df['customer_state']).iloc[:20]


    # In[170]:


    dfticks = pd.value_counts(df['customer_state']).iloc[:20].index.to_list() 
    #fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))

    dfgeo.plot()


    l = []
    for i in range(20):
        l.append(df[df["customer_state"] == dfticks[i]]["payment_value"].mean())

    ax = sns.lineplot(x = dfticks, y = l)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    return dfgeo

#ax.set_xticklabels(dfticks, rotation = 60)
#ax.set_title("average spending in the state") '''


# <i style ="font-size: 15px">
# This is a representation of the top 5 states. One can see where people have spent most on average. In blue the number of people buying in that state and in orange the average money spent.

# ## Two alternatives: PCA and Autoencoders

# <p style="font-size: 16px" font-family="sans-serif">
# After finding out intersting features in the dataset, we are eager to clear things out, as previous results may be improved. 
# Since the dataset is quite large, we want to use Principal component analysis to try to detect the most important features and preprocess the data 
# using other features rather than only RFM attributes. There are some features that may be of particular interest, in particular:
#     <ul>
# <li> the payment type and number of installments are an important indicator, as we have seen the total spenditure varies a lot depending on them. </li>
# <li> the customer state, that is, if a customer comes from a rich or poor area (demographics is important) </li>
# <li> the product category, which may underpin some insights. </li>

# In[64]:

from sklearn.compose import make_column_transformer

def encoding_PCA(df, rfm_dataset):
    # one hot encoding for these features
    transformer = make_column_transformer(
        (OneHotEncoder(sparse= False), ['payment_type', 'customer_city', 'product_category_name_english', 'payment_installments']),
        remainder='passthrough') 
    encoded_df = transformer.fit_transform(df.loc(axis=1)['payment_type', 'customer_city', 'product_category_name_english','payment_installments'])


    # In[65]:


    encoded_df = pd.DataFrame(encoded_df,columns=transformer.get_feature_names_out())
    encoded_df.head()
    f = ['Monetary value','Recency','Frequency']
    newdf = pd.concat([rfm_dataset[f], encoded_df], axis=1)
    newdf.head()

    return encoded_df, newdf 


# We then create the dataframe with all the features we want to include in our analysis.

# In[66]:




# ## PCA, the principal component analysis

# <p style="font-size: 16px" font-family="sans-serif">
# As our dataset starts becoming quite large as the number of features has grown after introducing new features, we would like to extract as relevant information as possible by reducing data dmesionality, that is, creating "smaller" components which will be able to describe the dataset with less data. In the Quantitative models for data science course we have been introduced to a powerful technique, the so called "PCA" which is able, with few components (usually 2) to capture most of the variability of the dataset (relevant information). The first thing to do is to check the actual number of components to describe well the dataset without trading off too much information.

# In[67]:

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
def pca_preprocessing(newdf):
#getting started with PCA 
 
    #scaling the data is useful when dealing with PCA.
    sc_features = newdf.copy()
    sc = StandardScaler()
    new = sc.fit_transform(sc_features['Monetary value'].array.reshape(-1,1))
    new2 = sc.fit_transform(sc_features['Recency'].array.reshape(-1,1))
    new3 = sc.fit_transform(sc_features['Frequency'].array.reshape(-1,1))
    sc_features['Monetary value'] = new
    sc_features['Recency'] = new2 
    sc_features['Frequency'] = new3
    sc_features.head() 
    
    sc_features.dropna(inplace = True)   #dropping the nulls!
    sc_features.shape
    return sc_features

# In[68]:




# <p style="font-size: 16px" font-family="sans-serif">
#  What we do now, is to plot the cumulative plot of the principal components, that let us understand how much information(variability) n components provide together.

# In[69]:

def pca_ncomponents(sc_features):
    X_ = sc_features.values #scaled
    #assessing how many components are needed

    pca = PCA(n_components = 20)  #arbitrary
    principalComponents = pca.fit_transform(X_)

    #plotting the results
    features = range(pca.n_components_)
    plt.plot(features, pca.explained_variance_ratio_.cumsum(), marker ="o")   #pca.explained_ratio, displays howmuch information each
    plt.xlabel('PCA components')                                        #each component holds in percentage
    plt.ylabel('variance explained')
    plt.xticks(features)
    return X_ 


# <i style ="font-size: 15px">
# Usually, the accepted percentage of variability explained is 0.8, but in this case we suspect that more than 3 components would just contribute to explain "noise" and not real information needed for our analysis.

# <p style="font-size: 16px" font-family="sans-serif">
# Next, is to choose the number of clusters after that we decided to conìsider to components. We will again use the elbow method.

# In[70]:

def pca(X_):
#getting started with PCA, reducing dimensionality of the original data
    pca = PCA(n_components = 3)
    scores = pca.fit_transform(X_)  #transforming the data

    #applying kmeans, checking ideal number of clusters with elbow method

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X_)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return scores 


# <i style ="font-size: 15px">
# The appropriate number of clusters appears to be 4 again, as 7 clusters are way more than necessary.

# <p style="font-size: 16px" font-family="sans-serif">
# We are now ready to tune this hyperparameter and visualize the results of the PCA which will be ablo to explain 60% of dataset variability using just 3 components! This is a nice and fast way to detect the most important features in the new dataset.

# In[71]:

def pca_kmeans(sc_features, scores):

    kmeanspca = KMeans(n_clusters = 4, init="k-means++", random_state = 42)
    kmeanspca.fit(scores)


    sc_features2 = sc_features.iloc[:,[0,1,2]]  #getting rid of encoded columns
    #create the new dataframe, it will contain components and corresponding cluster labels detected.
    segmkmeans = pd.concat([sc_features2, pd.DataFrame(scores)], axis=1)
    #changing columns names 
    segmkmeans.columns.values[-3:] = ['component1', 'component2', 'component3'] #changing columns name
    segmkmeans['kmeansclusters'] = kmeanspca.labels_


    # In[72]:


    segmkmeans.head()
    return segmkmeans, kmeanspca


# <p style="font-size: 16px" font-family="sans-serif">
# Now we can finally visualize the clusters made by kmeans after the PCA transformation. For the sake of clarity we will see data in 2D as it is easier to see what the differences are between clusters.

# In[73]:

def pca_components(segmkmeans, kmeanspca, rfmcopy):
    x = segmkmeans['component2']
    y = segmkmeans['component1']
    sns.scatterplot(x=x, y=y, hue=segmkmeans['kmeansclusters'])

    plt.title("Clusters detected by PCA")
    plt.show()

    dfpca = rfmcopy.copy()
    dfpca['kmeansclustersPCA'] =  kmeanspca.labels_   #adding clusters labels to the dataframe
    dfpca.head() 

    return dfpca

# <i style ="font-size: 15px">
# It is easy to see that the results we obtained are very similar to the ones before: There are few customers that are more distant (likely to be top spenders) and others who are less sparse (mid, at-risk customers and low spenders). To check that these clusters correspond to the ones before (roughly) we will check the properties of each cluster.

# In[74]:





# <p style="font-size: 16px" font-family="sans-serif">
# After it, we want to check some clusters insights as we did before.

# In[75]:

def pca_insights(dfpca):
    f2 = ['Recency','Monetary value','Frequency']
    first = dfpca[dfpca['kmeansclustersPCA'] == 0][f2]
    sec =  dfpca[dfpca['kmeansclustersPCA'] == 1][f2]
    th = dfpca[dfpca['kmeansclustersPCA'] == 2][f2]
    four = dfpca[dfpca['kmeansclustersPCA'] == 3][f2]


# In[76]:


    first.describe()


# <i style ="font-size: 15px">
# Even though some customers may not be correctly detected (minimum monetary value is 13 which is low), this cluster likely contains high-spenders, people who frequently purchase products in the business.

# In[77]:


    sec.describe()


# <i style ="font-size: 15px">
# In this cluster, there are present customers who are "at risk". Even though they have spent as much as the ones in the next cluster, their recency is much more far away as one can see both at the minimum and maximum(interquartile range)

# In[78]:


    th.describe()


# <i style ="font-size: 15px">
# It is interesting to compare this clusters to the previous one. This third cluster presents customers who can be labelled as "mid-spenders" as their frequency is on average the same, but their recency is much less, which means that they may be usual customers of the brazilian subsidiary.

# In[79]:

 
    four.describe()
    return first, sec, th


# <i style ="font-size: 15px">
# This fourth cluster presents again the top customers. This time, it seems that the algorithm has detected a relatively higher number of them.

# In[80]:

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
                                


# In[155]:




# In[81]:



# ## Autoencoder ANN

# <p style="font-size: 16px" font-family="sans-serif">
# Now we can finally visualize the clusters made by kmeans after the PCA transformation. For the sake of clarity we will see data in 2D as it is easier to see what the differences are between clusters.
# We are now introducing a new concept in our analysis which is very important. As we tried with PCA, we want to reduce the most of dimensionality to capture the best datapoints which make clusters distinct (as we said, one should find something that maximizes the wcss that is the distance among clusters in Kmeans!). For this task, we use an autoencoder, an ANN which is able to reduce the dimensionality of the data. For a n large enough (in our case almost 2000) is it convienient to identify the features underpinning and correlating similar datapoints; in this way we can better represent the clusters. Let us see if this preprocessing step actually works!
# 
# <p style="font-size: 16px" font-family="sans-serif">
# We will use the tensorflow package keras to feed and create the ANN; below the neural network is implemented, Autoencoder works as follows: </p>
# 
# - The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore insignificant data (“noise”).
# 
# - It has 2 main parts:  an encoder that maps the message (data) to a code, and a decoder that reconstructs the message (processed data) from the code, that is the decoder extracts the most relevant patterns and information we want to retrieve. Below an image showing its structure:
# 
# ![autoencoders.png](attachment:autoencoders.png)

# <p style="font-size: 16px" font-family="sans-serif">
# The code below is an autoencoder artificial neural network made up of 4 encoders, the encoder layer, and 2 decoding layers that will output new reducted data!

# In[82]:

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.initializers import GlorotUniform
from keras.optimizers import SGD 

def ann_autoencoder(X_):
    encoding_dim = 7

    # X_.shape is (11127, 1807) 
    input_dim = X_.shape[1] # get the number of columns in X

    input_df = Input(shape=(input_dim,))   #let us define the input layer which will be the number of features in our df!

    # Glorot normal initializer (Xavier normal initializer) draws samples from a truncated normal distribution 
    #assembling the encoder!
    x = Dense(encoding_dim, activation='relu')(input_df)
    x = Dense(500, activation='relu', kernel_initializer = GlorotUniform())   #initializatiion of weights
    x = Dense(500, activation='relu', kernel_initializer = GlorotUniform())(x)
    x = Dense(2000, activation='relu', kernel_initializer = GlorotUniform())(x)

    #encoding layer
    encoded = Dense(10, activation='relu', kernel_initializer = GlorotUniform())(x)


    #decoding layers
    x = Dense(2000, activation='relu', kernel_initializer = GlorotUniform())(encoded)
    x = Dense(500, activation='relu', kernel_initializer = GlorotUniform())(x)

    decoded = Dense(1807, kernel_initializer = GlorotUniform())(x)


    # autoencoder layer
    autoencoder = Model(input_df, decoded)

    #encoder - used for our dimension reduction
    encoder = Model(input_df, encoded)

    autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')

    return autoencoder, encoder, input_df

#adam is a good optimizer and it works as the stochastic gradient descent and the mean_squared error is the cost function we decide to minimize.


# <p style="font-size: 16px" font-family="sans-serif">
# After compiling the autoencoder neural network, we are reay to fit the data.

# In[83]:


#fitting the data
#batch size is set at 128 as it seems a standard in the industry we guess, verbose at 1, not much info needed

def ann_fit_predict(X_, autoencoder, encoder):
    autoencoder.fit(X_,X_, batch_size = 128, epochs = 50, verbose= 1)
    pr = encoder.predict(X_)
    kmeansann = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42, algorithm = "lloyd")
    y2_pr = kmeansann.fit_predict(pr)
    return pr, y2_pr, kmeansann 



# <i style ="font-size: 15px">
# After 50 epochs, the loss is extremely low, looks a good compromise! Below we encode the processed data after the fitting.

# <p style="font-size: 16px" font-family="sans-serif">
# Predicting (transorming) the data:

# In[84]:




# <p style="font-size: 16px" font-family="sans-serif">
# As the data have been encoded and reduced in dimensionality, extracting useful patterns, again we will apply kmeans, the best algorithm tested so far according to silhouette score and also by clustering results using PCA.  <br>
# After the encoding, we are now ready to implement kmeans and plot the resulting clusters! </p>

# In[85]:





# In[86]:


#concatenating results to the dataframe and adding these clusters to the one containing all.
def conc_pca_ann(rfm_dataset, kmeansann, rfmcopy):
    dfann = rfm_dataset.copy()
    rfmcopy['clustersann'] = kmeansann.labels_
    dfann['clustersann'] = kmeansann.labels_
    dfann.head()
    return dfann 


# <p style="font-size: 16px" font-family="sans-serif">
# We are ready to plot the clusters:

# In[87]:


#plotting the results 
def ann_clusters(dfann, clustersann):
    plot4 = go.Figure()

    
    n4clusters = [0,1,2,3]   #same as we did before, the y_hc is a numpy array that returns, for each customer, the corresponding cluster assigned.
    for x in n4clusters:
            plot4.add_trace(go.Scatter3d(x = dfann[dfann.clustersann == x]['Recency'], 
                                        y = dfann[dfann.clustersann == x]['Frequency'],
                                        z = dfann[dfann.clustersann == x]['Monetary value'],  
                                        mode='markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(x+1)
                                        ))
                                                    
                                                

                # changing the layout!

    plot4.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                    scene = dict(xaxis=dict(title = 'Recency', titlefont_color = 'red'),
                                    yaxis=dict(title = 'Frequency', titlefont_color = 'blue'),
                                    zaxis=dict(title = 'Monetary value', titlefont_color = 'green')),
                    font = dict(family = "Gilroy", color  = 'black', size = 12))

    plot4.show()


# <i style ="font-size: 15px">
# The criteria taken into account seem to produce a much different result; The customers are now not only clustered because of RFM features but the ANN has influenced the clustering procedure. 

# In[154]:




# <p style="font-size: 16px" font-family="sans-serif"> 
# Let's see some clusters chaarcteristics:

# In[88]:

def ann_insigths(dfann, dfpcaf):
    dfann[dfann['clustersann'] == 0].describe()


    # <i style ="font-size: 15px">
    # Top customers cluster.

    # <i style ="font-size: 15px">
    # 

    # In[89]:


    dfann[dfann['clustersann'] == 1].describe()


    # <i style ="font-size: 15px">
    # mid-spenders

    # <i style ="font-size: 15px">

    # In[90]:


    dfann[dfann['clustersann'] == 2].describe()


    # In[91]:


    dfann[dfann['clustersann'] == 3].describe()


    # Customers who bought slightly more.

    # <i style ="font-size: 15px">

    # <i style ="font-size: 15px">
    # In the clustering made by kmeans after the autoencoding, Kmeans has given no importance on the rfm features but rather, it gives much more importance on the other features we introduced (payment and geographic information) as it is shown that each of the 4 customer segmentation present the same characteristics.

    # In[92]:


    dfpcaf["clustersann"]=dfann["clustersann"]
    dfpcaf.head()


    # <p style="font-size: 16px" font-family="sans-serif">
    # Let us also see some insights of customers belonging to each cluster, below plots show the payment type, the most popular state where customer belong to and top product categories bought.

    # In[93]:


    insights=["payment_type","payment_installments", "customer_state","product_category_name_english"]


    # In[123]:


    ins1=dfpcaf[dfpcaf["clustersann"]==0][insights]
    #dins1={"payment_type":[],"payment_installments":[], "customer_state":[], "product_category_name_english":[]}
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))



    sns.countplot(x= ins1["payment_type"], ax= ax1)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins1['customer_state']).iloc[:20].index
    sns.countplot(y = ins1["customer_state"], order = ord1, ax= ax2)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins1['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins1['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


    # <i style ="font-size: 15px">
    # Top customers insights.

    # In[129]:


    ins2=dfpcaf[dfpcaf["clustersann"]==1][insights]
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))



    sns.countplot(x= ins2["payment_type"], ax= ax1)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins2['customer_state']).iloc[:20].index
    sns.countplot(y = ins2["customer_state"], order = ord1, ax= ax2)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins2['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins2['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


    # In[128]:


    ins3=dfpcaf[dfpcaf["clustersann"]==2][insights]

    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))



    sns.countplot(x= ins3["payment_type"], ax= ax1)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins3['customer_state']).iloc[:20].index
    sns.countplot(y = ins3["customer_state"], order = ord1, ax= ax2)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins3['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins3['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


    # In[126]:


    ins4=dfpcaf[dfpcaf["clustersann"]==3][insights]

    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))



    sns.countplot(x= ins1["payment_type"], ax= ax1)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins4['customer_state']).iloc[:20].index
    sns.countplot(y = ins1["customer_state"], order = ord1, ax= ax2)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins4['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins4['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


# <i style ="font-size: 15px">
# As we can see customers in these clusters behave almost the same; The majority has bought in SP(San Paolo) and many have used credit cards to buy.

# ## Final considerations

# In our analysis we identified 4 main customer segmentations:
# •	Low spenders/at-risk 
# •	Mid Spenders
# •	High spenders
# •	Top customers
# 
# The way that the various algorithms segmented the data into these four clusters is described as follows:
# 1.	KMeans has detected majority of the customers into a single cluster, the low spenders and those who are likely to leave the business, while the high spenders and top customers are a minority.
# 2.	Hierarchical Cluster had a similar approach to KMeans, but with a key difference. Whilst the low and at-risk customers remain the same, the mid spenders cluster size has increased, while the high and top spenders are unchanged.
# 3.	Spectral Clustering has a more balanced partition, we don’t see very small segmentations like we did in KMeans and Hierarchical Cluster. We see a more even distribution of the customer segmentation; also standard deviations suggest that the datapoints (customers) in each cluster are way more similar to each other than before.
# 4.	For the Principal Component Analysis, we can see that more high spenders have been detected compared to KMeans and HC, but when compared to Spectral, the clusters' size is similar. An interesting difference is the differentiation between those who are at-risk compared to the mid spenders. In fact the PCA gave more importance to the recency compared to the other three algorithms and probably also considered the other factors, such as the payment method, installments and others we have stated before.
# 5.	Autoencoder ANN has provided very similar clusters. Not only does not it take into account the monetary value but also other features such as payment and demographics information do not seem to influence the segments.
# 
# Below, a dataframe showing the silhouette scores.

# In[160]:




# Even though the silhouette score of the PCA kmeans is way less than the ones in rfm kmeans and hierarchical clustering, the algorithm has detected better segmentations; The standard deviations among datapoints is way lower, meaning that the customers belonging to each cluster are generally more similar to each other, while kmeans and hierarchical have identified top customers really well, but disregarded the importance of segmenting mid and at-risk/low spenders properly. 
# The spectral clustering instead has done a great job as one can see both at the segmentations' descriptions and also at the silhouette score, which is way less than its 2 main competitors but still decent.

# ## Final takeaways 
# 
# After a thorough investigation, there are some takeaways that the Brazilian subsidiary can get:
# - Brazilian customers are segmented in 4 categories, low spenders(people who buy occasionaly or just made one single purchase because of vouchers or promotions); at-risk customers, who buy with a slightly higher frequency but spend quite more than the latter; mid-spenders who are usual customers and finally top and high-spenders customers, those who buy frequently (more than 4 times) and spend immensely more than the average.
# 
# - The subsidiary should focus the email campaign on both low and at-risk customers to retain them, by proposing promotions and discounts to those returning/continuing to buy in the business.
# 
# - For midspenders, the firm should introduce new and less-bought products; As we have seen there are products who have been purchased way more than others; it can be useful to increase selling in those who are not purchased as much.
# 
# - For high and top spenders, the firm could as well promote the usual products they buy (as seen in the plots below they buy expensive products such as furnitures, computers exc...) but at the same time propose more general-user products to increase the already large Lifetimevalue they have.

# In[163]:




# In[168]:

def pca_vs_spectral(dfpcaf, insights):
    #high and top spenders identified in PCA and spectral
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))= plt.subplots(2,3, figsize=(30,28))
    ins5 = dfpcaf[(dfpcaf['kmeansclustersPCA'] == 0) | (dfpcaf['kmeansclustersPCA'] == 0)][insights]


    sns.countplot(x= ins5["payment_type"], ax= ax1)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins5['customer_state']).iloc[:20].index
    sns.countplot(y = ins5["customer_state"], order = ord1, ax= ax2)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins5['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins5['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    ins6 = dfpcaf[(dfpcaf['sp_clusters'] == 2 ) | (dfpcaf['sp_clusters'] == 3 )][insights]

    sns.countplot(x= ins6["payment_type"], ax= ax4)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins6['customer_state']).iloc[:20].index
    sns.countplot(y = ins6["customer_state"], order = ord1, ax= ax5)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins6['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins6['product_category_name_english'], order= ord2, ax = ax6)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


# <i style ="font-size: 15px">
# Respectively, on first row, high and top spenders insights identified by Pca and spectral clustering on second. 

    # Below, some insights about low and at risk customers:

    # In[169]:


    #high and top spenders identified in PCA and spectral
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))= plt.subplots(2,3, figsize=(30,28))
    ins7 = dfpcaf[(dfpcaf['kmeansclustersPCA'] == 1) | (dfpcaf['kmeansclustersPCA'] == 2)][insights]


    sns.countplot(x= ins7["payment_type"], ax= ax1)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins7['customer_state']).iloc[:20].index
    sns.countplot(y = ins7["customer_state"], order = ord1, ax= ax2)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins7['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins7['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    ins8 = dfpcaf[dfpcaf['sp_clusters'] == 1][insights]

    sns.countplot(x= ins8["payment_type"], ax= ax4)
    #axs.set_title(title="Most used payment methods")



    ord1 = pd.value_counts(ins8['customer_state']).iloc[:20].index
    sns.countplot(y = ins8["customer_state"], order = ord1, ax= ax5)
    #axs.set_title(title="Customers' states")




    ord2 = pd.value_counts(ins8['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins8['product_category_name_english'], order= ord2, ax = ax6)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


# <i style ="font-size: 15px">
# As we can see low and at-risk customers buy the same type of products (the top three are the same). Targeting emails offering discounts on these products may be a winning strategy to raise revenues and retain/regain these customers, with more who have bought just once and then left the business.




