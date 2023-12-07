import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

# Access the imported numpy module
np

def pca_kmeans(sc_features, scores, nclusterspca):
    """
    Perform PCA and K-Means clustering.

    Args:
        sc_features (pd.DataFrame): The scaled features.
        scores (np.ndarray): The PCA scores.
        nclusterspca (int): The number of clusters for K-Means.

    Returns:
        tuple: A tuple containing the segmented data and the K-Means model.
    """
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
    plt.xlabel("Component2")
    plt.ylabel("Component1")
    plt.show()

    dfpca = rfmcopy.copy()
    dfpca['kmeansclustersPCA'] = kmeanspca.labels_

    logging.info("PCA components visualization completed")

    return dfpca
