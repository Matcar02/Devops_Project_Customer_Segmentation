import plotly.graph_objs as go 
import plotly.io as pio
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import sys
from datetime import datetime
import wandb 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_clusters(rfmcopy, clusters1):
    """
    Plot clusters using 3D scatter plots for K-Means clustering results.

    Args:
        rfmcopy (pd.DataFrame): The DataFrame containing RFM data with K-Means cluster labels.
        clusters1 (int): The number of clusters to plot.

    Returns:
        None: The function generates and saves 3D scatter plots.
    """

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

    plt.title("K-Means Clustering")
    
    logging.info("Cluster plotting completed.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'Kmeans_clusters_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except Exception as e:
        logging.error(f'Error saving plot. {e}')
        return

    
    plot.show()

        


def visualize_spectral_clusters(X, sp):
    """
    Visualize clusters using 3D scatter plots for Spectral clustering results.

    Args:
        X (array-like): The original data used for clustering.
        sp (array-like): The cluster labels from Spectral clustering.

    Returns:
        None: The function generates and saves 3D scatter plots.
    """

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
    
    plt.title("Spectral Clustering clusters")
    
    logging.info("Spectral cluster visualization completed.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'spectral_clusters_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except Exception as e:
        logging.error(f'Error saving plot. {e}')
        return 
    
    plot.show()


def plot_clusters_pca(rfmcopy, clusterspca):
    """
    Plot clusters using 3D scatter plots for PCA clustering results.

    Args:
        rfmcopy (pd.DataFrame): The DataFrame containing RFM data with PCA cluster labels.
        clusterspca (int): The number of PCA clusters to plot.

    Returns:
        None: The function generates and saves 3D scatter plots.
    """
    
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

    plt.title("PCA Clustering clusters")
    
    logging.info("PCA cluster plotting completed.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'pca_clusters_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except Exception as e:
        logging.error(f'Error saving plot. {e}')
        return
    
    plot.show()
    
    
    

