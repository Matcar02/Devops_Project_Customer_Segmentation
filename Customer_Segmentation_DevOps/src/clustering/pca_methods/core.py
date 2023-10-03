import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.cluster import KMeans 


def pca_kmeans(sc_features, scores):

    kmeanspca = KMeans(n_clusters = 4, init="k-means++", random_state = 42)
    kmeanspca.fit(scores)

    sc_features2 = sc_features.iloc[:,[0,1,2]]
    segmkmeans = pd.concat([sc_features2, pd.DataFrame(scores)], axis=1)
    segmkmeans.columns.values[-3:] = ['component1', 'component2', 'component3']
    segmkmeans['kmeansclusters'] = kmeanspca.labels_

    segmkmeans.head()
    return segmkmeans, kmeanspca


def pca_components(segmkmeans, kmeanspca, rfmcopy):
    x = segmkmeans['component2']
    y = segmkmeans['component1']
    sns.scatterplot(x=x, y=y, hue=segmkmeans['kmeansclusters'])

    plt.title("Clusters detected by PCA")
    plt.show()

    dfpca = rfmcopy.copy()
    dfpca['kmeansclustersPCA'] =  kmeanspca.labels_ 
    dfpca.head() 

    return dfpca