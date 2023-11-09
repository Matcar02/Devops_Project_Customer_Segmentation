from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level


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
