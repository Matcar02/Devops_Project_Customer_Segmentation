
#importing libraries

import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
import json 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.model_selection import GridSearchCV
import plotly.graph_objs as go  #import pylot for 3d objects!
from sklearn.metrics import silhouette_samples, silhouette_score 
import scipy.cluster.hierarchy as sch         
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import logging 



with open("src/config.json", "r") as config_file:
    config = json.load(config_file)

file_path = config["data_path"]



def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['product_name_lenght', 'product_description_lenght', 'shipping_limit_date', 'product_category_name'], axis=1, inplace=True)
    return df


# ## Data preparation


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



def clean_data(df):

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







# kmeans



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






def clustering(clusters1, algorithm1, rand_state, X, df):
    kmeans = KMeans(n_clusters = clusters1, init = 'k-means++', random_state = rand_state, algorithm = algorithm1, n_init = 3)
    y_kmeans = kmeans.fit_predict(X)

    #add respective client's clusters assigned 
    rfmcopy = df.copy()
    rfmcopy['kmeans_cluster'] = y_kmeans

    return rfmcopy


def plot_clusters(rfmcopy, clusters1):
    plot = go.Figure()    


    nclusters = [i for i in range (0,clusters1)]  
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

    rfmcopy = clustering(inp1, inp2, inp3, X, rfm_dataset)
    plot_clusters(rfmcopy, inp1)
    return rfmcopy



#silhouette score

def silhouette_score_f(X, y, method):
    #silhouette score
    results = y[method]
    silscores = {}
    silscores[method] = silhouette_score(X, results, metric='euclidean')  

    print(f"The silhouette score for {method} is: {silscores[method]}")
    return silscores 


    # In[27]:

#hierarchical clustering



def dendogram(X):
    Dend = sch.dendrogram(sch.linkage(X, method="ward"))
    plt.title("Dendogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distances")
    plt.xticks([])    #no ticks is displayed
    plt.show()
    return Dend




def agglomerative_clustering(X, rfmcopy):
    hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)  #predicting the clusters
    
    #sc_features['hc_clusters'] = y_hc  #in case we want to use scaled rfm_dataset
    plot2 = go.Figure()
    rfmcopy['hc_clusters'] = y_hc #

    n2clusters = sorted(list(rfmcopy['hc_clusters'].unique()))   
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







# Spectral clustering


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





# summary 

def show_silscores(silscores):
    dfscores = pd.DataFrame(silscores, index = [0])
    print(dfscores)




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



    dictio2 = {'Clustersize': [cl1size,cl2size,cl3size, cl4size], 'Total spending by cluster': [cl1sum,cl2sum,cl3sum,cl4sum],
            'Average spending by cluster' : [cl1mean,cl2mean,cl3mean,cl4mean], 'Average frequency by cluster':
            [cl1frequency ,cl2frequency ,cl3frequency ,cl4frequency], 'Frequency std': [cl1fsd,cl2fsd,cl3fsd,cl4fsd], 'Spending sd':
            [cl1sd,cl2sd,cl3sd,cl4sd]}

    Kmeanssummary = pd.DataFrame(dictio2, index = [1,2,3,4])  #include 5 if 5 is chosen
    return Kmeanssummary 




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
    lowspenderskmeans = rfmcopy[rfmcopy['kmeans_cluster'] == 1][feat]  #getting only rfm features
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



def corr(df):
    columns=["payment_type","payment_installments","payment_value"]
    sns.pairplot(df[columns])
    df[columns].corr()




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
    paydf = pd.concat([rfmcopy,w,n], axis=1) #create the df containing all info.
    paydf.head()       

    return paydf 



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




def recency(recency):
    recencydist=list(recency["Recency"])
    plt.hist(x=recencydist)
    plt.xlabel('days since last purchase')
    plt.ylabel('number of people per period')
    sns.histplot()




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



def prod_insights(df):
    dfcat = pd.value_counts(df['product_category_name_english']).iloc[:15].index 
    ax = sns.countplot(df['product_category_name_english'], order= dfcat)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)




def customer_geography(df):
    dfgeo = pd.value_counts(df['customer_state']).iloc[:20]
    dfticks = pd.value_counts(df['customer_state']).iloc[:20].index.to_list() 
    dfgeo.plot()


    l = []
    for i in range(20):
        l.append(df[df["customer_state"] == dfticks[i]]["payment_value"].mean())

    ax = sns.lineplot(x = dfticks, y = l)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    return dfgeo



 


# PCA, the principal component analysis

def encoding_PCA(df, rfm_dataset):
    transformer = make_column_transformer(
        (OneHotEncoder(sparse= False), ['payment_type', 'customer_city', 'product_category_name_english', 'payment_installments']),
        remainder='passthrough') 
    encoded_df = transformer.fit_transform(df.loc(axis=1)['payment_type', 'customer_city', 'product_category_name_english','payment_installments'])

    encoded_df = pd.DataFrame(encoded_df,columns=transformer.get_feature_names_out())
    encoded_df.head()
    f = ['Monetary value','Recency','Frequency']
    newdf = pd.concat([rfm_dataset[f], encoded_df], axis=1)
    newdf.head()

    return encoded_df, newdf

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



def pca(X_):
#getting started with PCA, reducing dimensionality of the original data
    pca = PCA(n_components = 3)
    scores = pca.fit_transform(X_)  #transforming the data

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



def pca_kmeans(sc_features, scores):

    kmeanspca = KMeans(n_clusters = 4, init="k-means++", random_state = 42)
    kmeanspca.fit(scores)


    sc_features2 = sc_features.iloc[:,[0,1,2]]  #getting rid of encoded columns
    segmkmeans = pd.concat([sc_features2, pd.DataFrame(scores)], axis=1)
    segmkmeans.columns.values[-3:] = ['component1', 'component2', 'component3'] #changing columns name
    segmkmeans['kmeansclusters'] = kmeanspca.labels_




    segmkmeans.head()
    return segmkmeans, kmeanspca



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
                                


def general_insights(dfpcaf):
    insights=["payment_type","payment_installments", "customer_state","product_category_name_english"]
    ins1 = dfpcaf[dfpcaf["clustersann"]==0][insights]
    
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))
    sns.countplot(x= ins1["payment_type"], ax= ax1)
    
    ord1 = pd.value_counts(ins1['customer_state']).iloc[:20].index
    sns.countplot(y = ins1["customer_state"], order = ord1, ax= ax2)
    
    ord2 = pd.value_counts(ins1['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins1['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")
    plt.show()


    ins2=dfpcaf[dfpcaf["clustersann"]==1][insights]
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))
    sns.countplot(x= ins2["payment_type"], ax= ax1)



    ord1 = pd.value_counts(ins2['customer_state']).iloc[:20].index
    sns.countplot(y = ins2["customer_state"], order = ord1, ax= ax2)


    ord2 = pd.value_counts(ins2['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins2['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")
    plt.show()



    ins3=dfpcaf[dfpcaf["clustersann"]==2][insights]
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))
    sns.countplot(x= ins3["payment_type"], ax= ax1)


    ord1 = pd.value_counts(ins3['customer_state']).iloc[:20].index
    sns.countplot(y = ins3["customer_state"], order = ord1, ax= ax2)


    ord2 = pd.value_counts(ins3['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins3['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()



    ins4=dfpcaf[dfpcaf["clustersann"]==3][insights]

    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))



    sns.countplot(x= ins1["payment_type"], ax= ax1)

    ord1 = pd.value_counts(ins4['customer_state']).iloc[:20].index
    sns.countplot(y = ins1["customer_state"], order = ord1, ax= ax2)




    ord2 = pd.value_counts(ins4['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins4['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


# to be changed (change it based on the number of clusters)




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



    ord1 = pd.value_counts(ins6['customer_state']).iloc[:20].index
    sns.countplot(y = ins6["customer_state"], order = ord1, ax= ax5)



    ord2 = pd.value_counts(ins6['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins6['product_category_name_english'], order= ord2, ax = ax6)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()



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



