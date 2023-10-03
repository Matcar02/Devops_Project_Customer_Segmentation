import plotly.graph_objs as go 
import pandas as pd

def plot_clusters(rfmcopy, clusters1):
    plot = go.Figure() 
    
    nclusters = [i for i in range (0,clusters1)]
    for x in nclusters:
        plot.add_trace(go.Scatter3d(x = rfmcopy[rfmcopy.kmeans_cluster == x]['Recency'], 
                                    y = rfmcopy[rfmcopy.kmeans_cluster == x]['Frequency'],
                                    z = rfmcopy[rfmcopy.kmeans_cluster == x]['Monetary value'],  
                                    mode='markers', marker_size = 8, marker_line_width = 1,
                                    name = 'Cluster ' + str(x+1)
                                    ))


    plot.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                    scene = dict(xaxis=dict(title = 'Recency', titlefont_color = 'black'),
                                    yaxis=dict(title = 'Frequency', titlefont_color = 'black'),
                                    zaxis=dict(title = 'Monetary value', titlefont_color = 'black')),
                    font = dict(family = "Gilroy", color  = 'black', size = 12))

    plot.show()


def visualize_spectral_clusters(X, sp):
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

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                        scene=dict(xaxis=dict(title='Recency', titlefont_color='red'),
                                   yaxis=dict(title='Frequency', titlefont_color='blue'),
                                   zaxis=dict(title='Monetary value', titlefont_color='green')),
                        font=dict(family="Gilroy", color='black', size=12))

    plot.show()