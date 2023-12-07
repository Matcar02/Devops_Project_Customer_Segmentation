import logging
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 
import logging
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 

# Configure the logging

def choose_spectral():
    """
    Choose the parameters for Spectral Clustering.
    """
    logging.info("Starting Spectral Clustering selection")

    # Instead of nclusters, ask for n_neighbors and affinity
    n_neighbors = list(range(3, 10))
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
    """
    Perform Spectral Clustering on the given data.
    """
    logging.info("Starting Spectral Clustering")

    spectral = SpectralClustering(n_clusters=nclusters, random_state=42, n_neighbors=neighbors, affinity=affinity)
    sp = spectral.fit_predict(X)

    sil_score = silhouette_score(X, sp, metric='euclidean')

    logging.info("Spectral Clustering completed")

    return sp, sil_score
