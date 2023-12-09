import logging
import wandb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def clustering(clusters, algorithm, rand_state, X, df):
    """
    Perform clustering on the given data.

    Args:
        clusters (int): Number of clusters.
        algorithm (str): Clustering algorithm ('lloyd' or 'elkan').
        rand_state (int): Random state for reproducibility.
        X (array-like): Input data.
        df (pandas.DataFrame): Dataframe to store the clustering results.

    Returns:
        pandas.DataFrame: Dataframe with clustering results.
    """
    logging.info("Starting clustering")

    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=rand_state, algorithm=algorithm, n_init=3)
    y_kmeans = kmeans.fit_predict(X)
    
    rfmcopy = df.copy()
    rfmcopy['kmeans_cluster'] = y_kmeans 

    #Log params 
    wandb.log({"Number of Clusters": clusters, "Clustering Algorithm": algorithm})
    wandb.log({"Silhouette Score": silhouette_score(X, y_kmeans)})
    
    logging.info("Saving Silhouette Score")
    wandb.sklearn.plot_silhouette(kmeans, rfmcopy, [1,2,3,4])
    logging.info("Saved Feature Plot")

    #plot the clusters and save the plot in wandb
    logging.info("Plotting clusters")
    wandb.sklearn.plot_clusterer(kmeans, rfmcopy, list(rfmcopy.columns)[:-1])
    logging.info("Saved Cluster Plot")



    logging.info("Clustering completed")

    return rfmcopy 
   




'''
def choose(rfm_dataset, X):
    """
    Perform cluster selection.

    Args:
        rfm_dataset (pandas.DataFrame): RFM dataset.
        X (array-like): Input data.

    Returns:
        Tuple[pandas.DataFrame, int]: Tuple containing the dataframe with clustering results and the number of clusters.
    """
    logging.info("Starting cluster selection")

    n_clusters = [3, 4, 5, 6]
    algorithms = ["lloyd", "elkan"]

    inp1 = int(input("Please insert the number of clusters you would like to have: "))
    if inp1 not in n_clusters:
        logging.warning("Not recommended nclusters. Please insert an integer between 3 and 6 for an optimal result.")
        inp1 = int(input("Please insert the number of clusters you would like to have: "))

    inp2 = str(input("Choose 'lloyd' or 'elkan': "))
    if inp2 not in algorithms:
        logging.warning("Invalid algorithm choice. Please type either 'lloyd' or 'elkan'.")
        inp2 = str(input("Choose 'lloyd' or 'elkan': "))

    inp3 = int(input("Please insert a random state (integer): "))
    if not isinstance(inp3, int):
        logging.warning("Random state must be an integer. Please reinsert.")
        inp3 = int(input("Reinsert a random integer: "))

    wandb.config.n_cluster_chosen = inp1
    wandb.config.kmeans_algorithm = inp2

    rfm_copy = clustering(inp1, inp2, inp3, X, rfm_dataset)
    logging.info("Cluster selection completed")

    return rfm_copy, inp1

    '''


def choose(rfm_dataset, X):
    """
    Perform cluster selection.

    Args:
        rfm_dataset (pandas.DataFrame): RFM dataset.
        X (array-like): Input data.

    Returns:
        Tuple[pandas.DataFrame, int]: Tuple containing the dataframe with clustering results and the number of clusters.
    """
    logging.info("Starting cluster selection")




    # Retrieve sweep configuration
    wandb.config.update(wandb.config, allow_val_change=True)

    # Access parameters in your code
    inp1 = wandb.config.n_clusters
    inp2 = wandb.config.algorithm
    inp3 = wandb.config.random_state

    rfm_copy = clustering(inp1, inp2, inp3, X, rfm_dataset)
    logging.info("Cluster selection completed")

    return rfm_copy, inp1
