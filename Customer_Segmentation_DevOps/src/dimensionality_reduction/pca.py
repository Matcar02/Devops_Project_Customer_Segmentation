import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encoding_PCA(df, rfm_dataset):
    """
    Perform one-hot encoding on specified columns and prepare data for PCA.

    Args:
        df (pd.DataFrame): The DataFrame containing categorical features for encoding.
        rfm_dataset (pd.DataFrame): The RFM dataset to be combined with encoded features.

    Returns:
        tuple: A tuple containing the encoded DataFrame and the new DataFrame with RFM data.
    """

    logging.info("Starting encoding_PCA function...")
    
    # Apply one-hot encoding to specified columns
    transformer = make_column_transformer(
        (OneHotEncoder(sparse= False), ['payment_type', 'customer_city', 'product_category_name_english', 'payment_installments']),
        remainder='passthrough') 
    encoded_df = transformer.fit_transform(df.loc(axis=1)['payment_type', 'customer_city', 'product_category_name_english','payment_installments'])
    
    encoded_df = pd.DataFrame(encoded_df, columns=transformer.get_feature_names_out())
    logging.debug(f"Encoded dataframe shape: {encoded_df.shape}")
    
    f = ['Monetary value', 'Recency', 'Frequency']
    newdf = pd.concat([rfm_dataset[f], encoded_df], axis=1)

    logging.info("Encoding and PCA transformation completed.")
    return encoded_df, newdf 

def pca_preprocessing(newdf):
    """
    Preprocess the data by scaling features before PCA.

    Args:
        newdf (pd.DataFrame): The DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: A DataFrame with scaled features.
    """

    logging.info("Starting PCA preprocessing...")
    
    # Scale features using StandardScaler
    sc_features = newdf.copy()
    sc = StandardScaler()
    new = sc.fit_transform(sc_features['Monetary value'].array.reshape(-1,1))
    new2 = sc.fit_transform(sc_features['Recency'].array.reshape(-1,1))
    new3 = sc.fit_transform(sc_features['Frequency'].array.reshape(-1,1))
    sc_features['Monetary value'] = new
    sc_features['Recency'] = new2 
    sc_features['Frequency'] = new3
    sc_features.dropna(inplace=True)
    sc_features.drop_duplicates(inplace=True) 

    logging.debug(f"Null values in data: \n{sc_features.isnull().sum()}")
    logging.debug(f"Shape of scaled features: {sc_features.shape}")
    logging.info("PCA preprocessing completed.")
    return sc_features

def pca_ncomponents(sc_features):
    """
    Determine the optimal number of PCA components.

    Args:
        sc_features (pd.DataFrame): The DataFrame with scaled features.

    Returns:
        np.ndarray: The array of PCA-transformed features.
    """

    logging.info("Determining optimal number of PCA components...")

    # Apply PCA and plot the explained variance to find the optimal number of components
    X_ = sc_features.values
    pca = PCA(n_components = 20) 
    principalComponents = pca.fit_transform(X_)

    features = range(pca.n_components_)
    plt.plot(features, pca.explained_variance_ratio_.cumsum(), marker="o") 
    plt.title('Explained variance by components')
    plt.xlabel('PCA components')                                   
    plt.ylabel('variance explained')
    plt.xticks(features)
    plt.show() 
    
    logging.info("Determined optimal number of PCA components.")
    return X_ 

def pca(X_):
    """
    Perform PCA transformation and determine the optimal number of clusters using the Elbow method.

    Args:
        X_ (np.ndarray): The array of features to be transformed using PCA.

    Returns:
        np.ndarray: The PCA-transformed feature scores.
    """

    logging.info("Performing PCA transformation...")
    
    # Apply PCA transformation and use the Elbow method to determine clusters
    pca = PCA(n_components = 3)
    scores = pca.fit_transform(X_)
    mask = np.isnan(scores)
    scores = scores[~mask.any(axis=1)]
    scores = np.unique(scores, axis=0)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    logging.info("PCA transformation completed.")
    return scores
