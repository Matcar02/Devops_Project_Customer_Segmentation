import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

def dendrogram(X):
    logging.info("Starting Dendrogram generation")

    Dend = sch.dendrogram(sch.linkage(X, method="ward"))
    plt.title("Dendrogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distances")
    plt.xticks([])  # No ticks are displayed
    plt.show()

    logging.info("Dendrogram generation completed")

    return Dend
