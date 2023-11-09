from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.model_selection import GridSearchCV
import pandas as pd
import logging

# Configure the logging
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

