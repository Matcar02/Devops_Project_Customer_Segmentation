from keras.models import Model
from keras.layers import Input, Dense
from sklearn.cluster import KMeans 
import plotly.graph_objs as go 
from keras.initializers import GlorotUniform


def ann_autoencoder(X_):
    encoding_dim = 7

    input_dim = X_.shape[1] 

    input_df = Input(shape=(input_dim,)) 

    x = Dense(encoding_dim, activation='relu')(input_df)
    x = Dense(500, activation='relu', kernel_initializer = GlorotUniform())  
    x = Dense(500, activation='relu', kernel_initializer = GlorotUniform())(x)
    x = Dense(2000, activation='relu', kernel_initializer = GlorotUniform())(x)

    encoded = Dense(10, activation='relu', kernel_initializer = GlorotUniform())(x)

    x = Dense(2000, activation='relu', kernel_initializer = GlorotUniform())(encoded)
    x = Dense(500, activation='relu', kernel_initializer = GlorotUniform())(x)

    decoded = Dense(1807, kernel_initializer = GlorotUniform())(x)

    autoencoder = Model(input_df, decoded)

    encoder = Model(input_df, encoded)

    autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')

    return autoencoder, encoder, input_df


def ann_fit_predict(X_, autoencoder, encoder):
    autoencoder.fit(X_,X_, batch_size = 128, epochs = 50, verbose= 1)
    pr = encoder.predict(X_)
    kmeansann = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42, algorithm = "lloyd")
    y2_pr = kmeansann.fit_predict(pr)
    return pr, y2_pr, kmeansann 

def conc_pca_ann(rfm_dataset, kmeansann, rfmcopy):
    dfann = rfm_dataset.copy()
    rfmcopy['clustersann'] = kmeansann.labels_
    dfann['clustersann'] = kmeansann.labels_
    dfann.head()
    return dfann 


def ann_clusters(dfann, clustersann):
    plot4 = go.Figure()

    
    n4clusters = [0,1,2,3]
    for x in n4clusters:
            plot4.add_trace(go.Scatter3d(x = dfann[dfann.clustersann == x]['Recency'], 
                                        y = dfann[dfann.clustersann == x]['Frequency'],
                                        z = dfann[dfann.clustersann == x]['Monetary value'],  
                                        mode='markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(x+1)
                                        ))
                                                    

    plot4.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                    scene = dict(xaxis=dict(title = 'Recency', titlefont_color = 'red'),
                                    yaxis=dict(title = 'Frequency', titlefont_color = 'blue'),
                                    zaxis=dict(title = 'Monetary value', titlefont_color = 'green')),
                    font = dict(family = "Gilroy", color  = 'black', size = 12))

    plot4.show()