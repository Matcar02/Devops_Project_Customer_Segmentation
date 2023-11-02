from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

def spectral_clustering(X):
    logging.info("Starting Spectral Clustering")

    spectral = SpectralClustering(n_clusters=4, random_state=42, n_neighbors=8, affinity='nearest_neighbors')
    sp = spectral.fit_predict(X)

    sil_score = silhouette_score(X, sp, metric='euclidean')

    logging.info("Spectral Clustering completed")

    return sp, sil_score
