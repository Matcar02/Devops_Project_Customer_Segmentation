from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 

def spectral_clustering(X):
    spectral = SpectralClustering(n_clusters=4, random_state=42, n_neighbors=8, affinity='nearest_neighbors')
    sp = spectral.fit_predict(X)

    sil_score = silhouette_score(X, sp, metric='euclidean')

    return sp, sil_score