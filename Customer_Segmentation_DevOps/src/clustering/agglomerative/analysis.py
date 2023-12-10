import logging
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Configure the logging
logging.basicConfig(level=logging.INFO) 

def dendrogram(x):
    """
    Generate a dendrogram plot based on the given data.

    Args:
        x (array-like): The input data.

    Returns:
        dict: The dendrogram dictionary.

    """
    logging.info("Starting Dendrogram generation")

    dend = sch.dendrogram(sch.linkage(x, method="ward"))
    plt.title("Dendrogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distances")
    plt.xticks([]) 
    plt.show()

    logging.info("Dendrogram generation completed")

    return dend