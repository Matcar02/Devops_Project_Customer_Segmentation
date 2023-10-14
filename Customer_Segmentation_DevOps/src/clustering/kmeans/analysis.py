from sklearn.metrics import silhouette_score 
import matplotlib as plt
from sklearn.cluster import KMeans 
from sklearn.model_selection import GridSearchCV
import pandas as pd

####



from data_preparation.cleaning import prepare_data, drop_c_id, clean_data
from data_preparation.rfm import get_frequencies, get_recency, get_monetary, concatenate_dataframes

df = prepare_data('Customer_Segmentation_DevOps\data\external\customer_segmentation.csv')
df = drop_c_id(df)
df = clean_data(df)

frequencies = get_frequencies(df)
recency = get_recency(df)
monetary = get_monetary(df)
rfm_dataset = concatenate_dataframes(recency, monetary, frequencies)

print(rfm_dataset)



def elbow_method(rfm_dataset):
    features = ['Recency','Monetary value','Frequency']
    wcss = []

    X = rfm_dataset[features]

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return X, features 


def get_best_kmeans_params(X):
    params = {
        'algorithm': ['lloyd', 'elkan'],
        'n_init': [i for i in range(1, 15)],
        'n_clusters': [i for i in range(3, 6)]
    }

    kmeans = KMeans()
    clf = GridSearchCV(estimator=kmeans, param_grid=params).fit(X)

    cv_results = pd.DataFrame(clf.cv_results_)
    print(f"The top parameters to tune into Kmeans are: {clf.best_params_}")
    return clf.best_params_


def silhouette_score_f(X, y, method):
    results = y[method]
    silscores = {}
    silscores[method] = silhouette_score(X, results, metric='euclidean')  

    print(f"The silhouette score for {method} is: {silscores[method]}")
    return silscores 




