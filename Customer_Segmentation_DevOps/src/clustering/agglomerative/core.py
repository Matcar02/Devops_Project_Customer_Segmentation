import logging
import plotly.graph_objs as go
from sklearn.cluster import AgglomerativeClustering

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level


def agglomerative_clustering(X, rfmcopy, n_clustersagg):
    """
    Perform Agglomerative Clustering on the given data.

    Args:
        X (array-like): The input data.
        rfmcopy (pandas.DataFrame): The copy of the RFM data.
        n_clustersagg (int): The number of clusters to create.

    Returns:
        array-like: The cluster labels.
    """
    logging.info("Starting Agglomerative Clustering")

    hc = AgglomerativeClustering(n_clusters=n_clustersagg, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)

    logging.info("Clustering completed")

    plot2 = go.Figure()
    rfmcopy['hc_clusters'] = y_hc

    n2clusters = sorted(list(rfmcopy['hc_clusters'].unique()))
    for x in n2clusters:
        logging.info("Plotting data for Cluster %s", x+1)
        plot2.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.hc_clusters == x]['Recency'],
                                     y=rfmcopy[rfmcopy.hc_clusters == x]['Frequency'],
                                     z=rfmcopy[rfmcopy.hc_clusters == x]['Monetary value'],
                                     mode='markers', marker_size=8, marker_line_width=1,
                                     name='Cluster ' + str(x+1)
                                     ))

    plot2.update_layout(width=800, height=800, autosize=True, showlegend=True,
                       scene={"xaxis": {"title": 'Recency', "titlefont_color": 'black'},
                              "yaxis": {"title": 'Frequency', "titlefont_color": 'black'},
                              "zaxis": {"title": 'Monetary value', "titlefont_color": 'black'}},
                       font={"family": "Gilroy", "color": 'black', "size": 12})

    plot2.show()

    logging.info("Agglomerative Clustering completed")

    return y_hc
