from sklearn.cluster import KMeans
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

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
