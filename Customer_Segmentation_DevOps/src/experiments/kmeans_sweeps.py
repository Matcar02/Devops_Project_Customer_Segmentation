# run_experiment.py
import sys
import json  
import os
import wandb


#paths
current_path = os.getcwd()
data_folder = os.path.abspath(os.path.join(current_path, '..', 'data', 'external'))
data_filepath = os.path.join(data_folder, 'customer_segmentation.csv')
sys.path.append(os.path.abspath(os.path.join(current_path, '..')))

#getting modules and functions
from src.data_preparation.cleaning import *
from src.data_preparation.rfm import *
from src.clustering.kmeans.analysis import * 
from src.clustering.kmeans.core import *
from src.visualization.clustering_visualization import *
from src.visualization.data_visualization import * 

'''
# Initialize WandB
wandb.init(project="Customer_Segmentation", name="Kmeans_Experiment")


# Define sweep configuration
sweep_config_path = os.path.join(os.path.dirname(__file__), 'sweep.yaml')
sweep_id = wandb.sweep(sweep=sweep_config_path, project="Customer_Segmentation")


# Data Prep
df = prepare_data(data_filepath)
df = drop_c_id(df)
df = clean_data(df)

# RFM
frequency = get_frequencies(df)
monetary = get_monetary(df)
recency = get_recency(df)
rfm_dataset = concatenate_dataframes_(recency, monetary, frequency)


# KMeans
X = elbow_method(rfm_dataset)[0]
best_params = get_best_kmeans_params(X)

rfmcopy = choose(rfm_dataset, X)[0]
nclusterskmeans = choose(rfm_dataset, X)[1]
plot_clusters(rfmcopy, clusters1=nclusterskmeans)

silscores = {}
silscores['kmeans'] = silhouette_score_f(X, rfmcopy, 'kmeans_cluster')

wandb.finish()

''' 
sweep_config = {
    "name": "kmeans-sweep",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Silhouette Score"},
    "parameters": {
        "n_clusters": {"values": [3, 4, 5, 6]},
        "algorithm": {"values": ["lloyd", "elkan"]},
        "random_state": {"values": [42, 123, 456]}
    }
}

def objective(config):
    # Initialize WandB inside the objective function
    wandb.init(project="Customer_Segmentation", name="Kmeans_Experiment", config=config)

    # Paths
    current_path = os.getcwd()
    data_folder = os.path.abspath(os.path.join(current_path, '..', 'data', 'external'))
    data_filepath = os.path.join(data_folder, 'customer_segmentation.csv')

    # Data Prep
    df = prepare_data(data_filepath)
    df = drop_c_id(df)
    df = clean_data(df)

    # RFM
    frequency = get_frequencies(df)
    monetary = get_monetary(df)
    recency = get_recency(df)
    rfm_dataset = concatenate_dataframes_(recency, monetary, frequency)

    # KMeans
    X = elbow_method(rfm_dataset)[0]
    #best_params = get_best_kmeans_params(X)

    rfmcopy = choose(rfm_dataset, X)[0]
    nclusterskmeans = choose(rfm_dataset, X)[1]
    plot_clusters(rfmcopy, clusters1=nclusterskmeans)

    silscores = {}
    silscores['kmeans'] = silhouette_score_f(X, rfmcopy, 'kmeans_cluster')

    # Log the metrics
    wandb.log({"Silhouette Score": silscores['kmeans']})
    wandb.finish()

# Main function
def main():
    # Define sweep configuration
    sweep_id = wandb.sweep(sweep=sweep_config, project="Customer_Segmentation")

    # Run the sweep
    wandb.agent(sweep_id, function=objective, count=10)

# Run the main function
if __name__ == "__main__":
    main()