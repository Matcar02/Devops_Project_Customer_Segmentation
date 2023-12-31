from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objs as go 



def agglomerative_clustering(X, rfmcopy, n_clustersagg):
    hc = AgglomerativeClustering(n_clusters = n_clustersagg, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    
    plot2 = go.Figure()
    rfmcopy['hc_clusters'] = y_hc

    n2clusters = sorted(list(rfmcopy['hc_clusters'].unique()))
    for x in n2clusters:
            plot2.add_trace(go.Scatter3d(x = rfmcopy[rfmcopy.hc_clusters == x]['Recency'], 
                                        y = rfmcopy[rfmcopy.hc_clusters == x]['Frequency'],
                                        z = rfmcopy[rfmcopy.hc_clusters == x]['Monetary value'],  
                                        mode='markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(x+1)
                                        ))
                                                    
                                                
    plot2.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                    scene = dict(xaxis=dict(title = 'Recency', titlefont_color = 'black'),
                                    yaxis=dict(title = 'Frequency', titlefont_color = 'black'),
                                    zaxis=dict(title = 'Monetary value', titlefont_color = 'black')),
                    font = dict(family = "Gilroy", color  = 'black', size = 12))

    plot2.show()
    return y_hc 