import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
import os 
import sys
from datetime import datetime
import pandas
import mlflow
import argparse
import pandas as pd
import io
import wandb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_data(rfm_dataset):
    """
    Visualize the RFM dataset using various plots including pairplot, lineplot, and histogram.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data for visualization.

    Returns:
        None: The function generates and saves plots without returning any value.
    """
    logging.info("Visualizing data...")

    # Create a figure object with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    sns.pairplot(rfm_dataset, ax=axs[0])
    axs[0].set_title("Pairplot")

    plt.title("Pairplot")
    plt.show()
    
    # Saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'pairplot_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except Exception as e:
        logging.error(f'Error saving plot: {str(e)}')
        return None

    plt.close()
        
    
    sns.lineplot(x="Recency", y="Monetary value", data=rfm_dataset.sort_values(by=["Recency"], ascending=False), ax=axs[1])
    axs[1].set_title("Spending by Recency")
    plt.title("Spending by recency")
    plt.show()

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename2 = f'recency_monetary_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename2))
    except:
        logging.error('Error saving plot.')
        return

    plt.close()

    sns.histplot(data=rfm_dataset['Frequency'], discrete=True, ax=axs[2])
    axs[2].set_title("Frequency Plot")
    plt.title("Frequency plot")
    plt.show()

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename3 = f'frequencyplot_{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename3))
    except:
        logging.error('Error saving plot.')
        return

    plt.close()
    logging.info("Data visualization complete.")
    return fig  


def plot_average_spending_by_frequency(rfm_dataset):
    """
    Plot the average spending by frequency from the RFM dataset.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data.

    Returns:
        None: The function generates and saves a scatter plot.
    """

    logging.info("Plotting average spending by frequency...")

    # Calculations and plot generation for average spending by frequency
    frd = rfm_dataset.groupby(['Frequency'])['Monetary value'].mean().reset_index(name='Average Spending by frequency')
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    sns.scatterplot(data=frd, x="Frequency", y="Average Spending by frequency", s=100, color='red', ax=axs[0])
    axs[0].set_title("Average Spending by Frequency")
    plt.title("Average Spending by Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Average Spending")
    plt.show()
    logging.info("Average spending by frequency plotted.")

    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)


        try:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f'average_spending_by_frequency_{now}.png'
            plt.savefig(os.path.join(reports_path, 'figures', filename))
        except Exception as e:
            logging.error(f'Error saving plot: {str(e)}')
            return

        plt.close()
        return fig


def plot_payment_value_distribution(rfm_dataset):
    """
    Plot the distribution of payment values in the RFM dataset.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data.

    Returns:
        None: The function generates and saves a distribution plot.
    """

    logging.info("Plotting payment value distribution...")

    # Calculations and plot generation for payment value distribution
    log_min, log_max = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    new_bins = np.logspace(log_min, log_max, 4)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    sns.distplot(rfm_dataset['Monetary value'], kde=False, bins=new_bins, ax=axs[0])
    axs[0].set_title("Payment value distribution")
    plt.title("Payment value distribution")
    plt.ylabel("Count")
    plt.show()
    logging.info("Payment value distribution plotted.")

    # Saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'plot_payment_value_distribution_{now}.png'
    plt.savefig(os.path.join(reports_path, 'figures', filename))
    return fig


def freq(rfm_dataset):
    """
    Generate a histogram plot for the 'Frequency' feature in the RFM dataset.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data.

    Returns:
        None: The function displays a histogram plot.
    """
    logging.info("Generating frequency plots...")
    # Generate histogram plot for the Frequency feature
    sns.histplot(data=rfm_dataset['Frequency'], discrete=True)
    plt.title("Frequency plot")
    plt.show()
    logging.info("Frequency plots generated.")


def pairplot(rfm_dataset):
    """
    Generate a pairplot for the RFM dataset to visualize distributions and relationships.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data.

    Returns:
        None: The function displays a pairplot.
    """
    logging.info("Generating pairplot...")
    # Generate pairplot for overall visualization of the dataset
    sns.pairplot(rfm_dataset)
    plt.title("Pairplot")
    plt.show()
    logging.info("Pairplot generated.")


def spending_by_recency(rfm_dataset):
    """
    Generate a line plot to visualize spending by recency in the RFM dataset.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data.

    Returns:
        None: The function displays a line plot.
    """
    logging.info("Generating spending by recency plot...")
    # Sort data by 'Recency' and generate a line plot
    timeordered = rfm_dataset.sort_values(by=["Recency"], ascending=False)
    plt1 = sns.lineplot(x="Recency", y="Monetary value", data=timeordered)
    plt.title("Spending by recency")
    plt.show()
    logging.info("Spending by recency plotted.")


def payments_distribution(rfm_dataset):
    """
    Generate a distribution plot for the 'Monetary value' feature in the RFM dataset.

    Args:
        rfm_dataset (pd.DataFrame): The dataset containing RFM data.

    Returns:
        None: The function displays a distribution plot.
    """
    logging.info("Plotting payments distribution...")
    # Generate distribution plot for the Monetary value feature
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.displot(rfm_dataset['Monetary value'], kde=False, bins=newbins)
    plt.title("Payments distribution")
    plt.show()
    logging.info("Payments distribution plotted.")


def main(args):
    # Initialize a new wandb run
    wandb.init(project="customer_segmentation", job_type="data_visualization")

    # Load the RFM data artifact
    if args.filepath == "default":
        rfm_artifact = wandb.use_artifact('rfm_data:latest')
        rfm_artifact_dir = rfm_artifact.download()
        rfm_data_filepath = os.path.join(rfm_artifact_dir, 'rfm_data.csv')
        rfm_dataset = pd.read_csv(rfm_data_filepath)
    else:
        rfm_dataset = pd.read_csv(args.filepath)

    visualize_data(rfm_dataset)

    plot_average_spending_by_frequency(rfm_dataset)

    plot_payment_value_distribution(rfm_dataset)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Visualization')
    parser.add_argument('--filepath', type=str, default='default', help='Path to the RFM data CSV file or "default" to use the latest W&B artifact')
    args = parser.parse_args()
    main(args)
