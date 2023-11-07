import plotly.graph_objs as go 
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_clusters(rfmcopy, clusters1):
    logging.info("Starting plot_clusters function...")
    plot = go.Figure() 

    nclusters = [i for i in range (0, clusters1)]
    for x in nclusters:
        plot.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.kmeans_cluster == x]['Recency'], 
                                    y=rfmcopy[rfmcopy.kmeans_cluster == x]['Frequency'],
                                    z=rfmcopy[rfmcopy.kmeans_cluster == x]['Monetary value'],  
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name='Cluster ' + str(x+1)
                                    ))
    
    logging.debug(f"Added {len(nclusters)} clusters to the plot.")

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                    scene=dict(xaxis=dict(title='Recency', titlefont_color='black'),
                                yaxis=dict(title='Frequency', titlefont_color='black'),
                                zaxis=dict(title='Monetary value', titlefont_color='black')),
                    font=dict(family="Gilroy", color='black', size=12))

    plot.title("K-Means Clustering")
    plot.show()
    logging.info("Cluster plotting completed.")


def visualize_spectral_clusters(X, sp):
    logging.info("Starting visualize_spectral_clusters function...")
    plot = go.Figure()
    
    rfmcopy = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary value'])
    rfmcopy['sp_clusters'] = sp
    n_clusters = sorted(list(rfmcopy['sp_clusters'].unique()))

    for x in n_clusters:
        plot.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.sp_clusters == x]['Recency'], 
                                    y=rfmcopy[rfmcopy.sp_clusters == x]['Frequency'],
                                    z=rfmcopy[rfmcopy.sp_clusters == x]['Monetary value'],  
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name='Cluster ' + str(x+1)
                                    ))
    
    logging.debug(f"Added {len(n_clusters)} clusters to the plot.")

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                        scene=dict(xaxis=dict(title='Recency', titlefont_color='red'),
                                   yaxis=dict(title='Frequency', titlefont_color='blue'),
                                   zaxis=dict(title='Monetary value', titlefont_color='green')),
                        font=dict(family="Gilroy", color='black', size=12))
    
    plot.title("Spectral Clustering clusters")
    plot.show()
    logging.info("Spectral cluster visualization completed.")


def plot_clusters_pca(rfmcopy, clusterspca):
    logging.info("Starting plot_clusters_pca function...")
    plot = go.Figure()

    nclusters = [i for i in range(0, clusterspca)]
    for x in nclusters:
        plot.add_trace(go.Scatter3d(x=rfmcopy[rfmcopy.pca_clusters == x]['Recency'], 
                                    y=rfmcopy[rfmcopy.pca_clusters == x]['Frequency'],
                                    z=rfmcopy[rfmcopy.pca_clusters == x]['Monetary value'],  
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name='Cluster ' + str(x+1)
                                    ))
    
    logging.debug(f"Added {len(nclusters)} clusters to the plot.")

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                    scene=dict(xaxis=dict(title='Recency', titlefont_color='black'),
                                yaxis=dict(title='Frequency', titlefont_color='black'),
                                zaxis=dict(title='Monetary value', titlefont_color='black')),
                    font=dict(family="Gilroy", color='black', size=12))

    plot.title("PCA Clustering clusters")
    plot.show()
    logging.info("PCA cluster plotting completed.")
