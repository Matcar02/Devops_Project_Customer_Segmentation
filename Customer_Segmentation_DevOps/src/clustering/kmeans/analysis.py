import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)  # Set the desired logging level


def elbow_method(rfm_dataset):
    """
    Calculate the optimal number of clusters using the Elbow Method.
    """
    logging.info("Starting Elbow Method")

    features = ['Recency', 'Monetary value', 'Frequency']
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
    """
    Find the best parameters for KMeans using GridSearchCV.
    """
    logging.info("Starting GridSearchCV for KMeans parameters")

    params = {
        'algorithm': ['lloyd', 'elkan'],
        'n_init': list(range(1, 15)),
        'n_clusters': list(range(3, 6))
    }

    kmeans = KMeans()
    clf = GridSearchCV(estimator=kmeans, param_grid=params).fit(X)

    logging.info("The top parameters to tune into Kmeans are: %s", clf.best_params_)

    logging.info("GridSearchCV for KMeans parameters completed")

    return clf.best_params_


def silhouette_score_f(X, y, method):
    """
    Calculate the Silhouette Score for a given clustering method.
    """
    logging.info("Calculating Silhouette Score for %s", method)

    results = y[method]
    silsc = silhouette_score(X, results, metric='euclidean')  # Call silhouette_score only once
    silscores = {method: silsc}

    logging.info("The silhouette score for %s is: %s", method, silsc)
    return silscores, silsc

    


