import matplotlib as plt
import scipy.cluster.hierarchy as sch      

def dendogram(X):
    Dend = sch.dendrogram(sch.linkage(X, method="ward"))
    plt.title("Dendogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distances")
    plt.xticks([])    #no ticks is displayed
    plt.show()
    return Dend 