from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import pandas as pd


def encoding_PCA(df, rfm_dataset):

    transformer = make_column_transformer(
        (OneHotEncoder(sparse= False), ['payment_type', 'customer_city', 'product_category_name_english', 'payment_installments']),
        remainder='passthrough') 
    encoded_df = transformer.fit_transform(df.loc(axis=1)['payment_type', 'customer_city', 'product_category_name_english','payment_installments'])

    encoded_df = pd.DataFrame(encoded_df,columns=transformer.get_feature_names_out())
    encoded_df.head()
    f = ['Monetary value','Recency','Frequency']
    newdf = pd.concat([rfm_dataset[f], encoded_df], axis=1)
    newdf.head()

    return encoded_df, newdf 


def pca_preprocessing(newdf):
 
    sc_features = newdf.copy()
    sc = StandardScaler()
    new = sc.fit_transform(sc_features['Monetary value'].array.reshape(-1,1))
    new2 = sc.fit_transform(sc_features['Recency'].array.reshape(-1,1))
    new3 = sc.fit_transform(sc_features['Frequency'].array.reshape(-1,1))
    sc_features['Monetary value'] = new
    sc_features['Recency'] = new2 
    sc_features['Frequency'] = new3
    sc_features.head() 
    
    sc_features.dropna(inplace = True) 
    sc_features.shape
    return sc_features

def pca_ncomponents(sc_features):
    X_ = sc_features.values

    pca = PCA(n_components = 20) 
    principalComponents = pca.fit_transform(X_)

    features = range(pca.n_components_)
    plt.plot(features, pca.explained_variance_ratio_.cumsum(), marker ="o") 
    plt.xlabel('PCA components')                                   
    plt.ylabel('variance explained')
    plt.xticks(features)
    return X_ 


def pca(X_):
    pca = PCA(n_components = 3)
    scores = pca.fit_transform(X_)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X_)
        wcss.append(kmeans.inertia_)
    
    #return the best number of clusters based on the scores
    

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return scores 