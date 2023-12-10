import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Setting up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def pca_vs_spectral(dfpcaf, insights):
    """
    Compare PCA and Spectral Clustering results by visualizing key insights.

    Args:
        dfpcaf (pd.DataFrame): The DataFrame containing PCA and Spectral Clustering results.
        insights (list): The list of columns to be used for insights generation.

    Returns:
        None: Displays comparison plots.
    """

    logging.info("Starting pca_vs_spectral function...")

    # Check if DataFrame is empty
    if dfpcaf.empty:
        logging.warning("Input DataFrame 'dfpcaf' is empty. No data to process.")
        return

    # Create subplots for PCA and Spectral Clustering insights
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30, 28))

    ins5 = dfpcaf[(dfpcaf['kmeansclustersPCA'] == 0) | (dfpcaf['kmeansclustersPCA'] == 0)][insights]

    logging.debug("Filtering data for kmeansclustersPCA...")
    
    sns.countplot(x=ins5["payment_type"], ax=ax1)

    ord1 = pd.value_counts(ins5['customer_state']).iloc[:20].index
    sns.countplot(y=ins5["customer_state"], order=ord1, ax=ax2)

    ord2 = pd.value_counts(ins5['product_category_name_english']).iloc[:15].index
    axs = sns.countplot(ins5['product_category_name_english'], order=ord2, ax=ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
    axs.set_title(label="Top 10 category in cluster")

    logging.debug("Processing data for sp_clusters...")

    ins6 = dfpcaf[(dfpcaf['sp_clusters'] == 2) | (dfpcaf['sp_clusters'] == 3)][insights]

    sns.countplot(x=ins6["payment_type"], ax=ax4)

    ord1 = pd.value_counts(ins6['customer_state']).iloc[:20].index
    sns.countplot(y=ins6["customer_state"], order=ord1, ax=ax5)

    ord2 = pd.value_counts(ins6['product_category_name_english']).iloc[:15].index
    axs = sns.countplot(ins6['product_category_name_english'], order=ord2, ax=ax6)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
    axs.set_title(label="Top 10 category in cluster")

    plt.show()
    logging.info("Finished pca_vs_spectral function and plotted the visualizations.")
