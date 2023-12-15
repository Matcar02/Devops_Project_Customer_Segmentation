import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import os
import argparse
import wandb
import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def describe_dataset(rfm_dataset):
    """
    Describe the dataset by providing statistical summaries.

    Args:
        rfm_dataset (pd.DataFrame): The RFM (Recency, Frequency, Monetary) dataset to describe.

    Returns:
        None: Outputs the statistical summary to the logging system.
    """
    logging.info('Starting describe_dataset function...')
    
    # Generate and log statistical description of the dataset
    description = rfm_dataset.describe()
    logging.info('\n' + str(description))

    logging.info('describe_dataset function completed.')
    return description

def corr(df):
    """
    Generate and log a correlation matrix and pairplot for selected columns.

    Args:
        df (pd.DataFrame): The DataFrame from which to calculate correlations.

    Returns:
        None: Outputs the correlation matrix and pairplot visuals.
    """

    logging.info('Starting corr function...')

    # Define columns for correlation analysis
    columns = ["payment_type", "payment_installments", "payment_value"]
    
    logging.debug('Generating pairplot for columns: %s', ', '.join(columns))
    
     # Generate and display a pairplot for the specified columns
    pairplot = sns.pairplot(df[columns])

    # Saving plot
    logging.info('Getting plot...')
     # Save the pairplot image
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports', 'figures'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pairplot_filename = f'pairplot_{now}.png'
    pairplot.savefig(os.path.join(reports_path, pairplot_filename))
    plt.close()
    
    # Compute and log the correlation matrix
    corr_matrix = df[columns].corr()
    logging.info('Correlation matrix generated for columns: %s', ', '.join(columns))
    logging.info('\n' + str(corr_matrix))

    logging.info('corr function completed.')
    return corr_matrix

def main(args):
    wandb.init(project="customer_segmentation", job_type="stats_analysis")

    # Load the RFM data artifact
    if args.rfm == "default":
        rfm_artifact = wandb.use_artifact('rfm_data:latest')
        rfm_artifact_dir = rfm_artifact.download()
        rfm_data_filepath = os.path.join(rfm_artifact_dir, 'rfm_data.csv')
        rfm_data = pd.read_csv(rfm_data_filepath)
    else:
        rfm_data = pd.read_csv(args.rfm)

    # Load the customer segmentation artifact
    if args.customer_segmentation == "default":
        customer_segmentation_artifact = wandb.use_artifact('customer_segmentation:latest')
        customer_segmentation_artifact_dir = customer_segmentation_artifact.download()
        customer_segmentation_filepath = os.path.join(customer_segmentation_artifact_dir, 'customer_segmentation.csv')
        customer_segmentation_data = pd.read_csv(customer_segmentation_filepath)
    else:
        customer_segmentation_data = pd.read_csv(args.customer_segmentation)

    # Perform statistical analysis
    describe_dataset(rfm_data)
    corr_matrix = corr(customer_segmentation_data)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis of data")
    parser.add_argument("--rfm", type=str, default="default", help="Path to the RFM data CSV file or 'default' to use the latest W&B artifact")
    parser.add_argument("--customer_segmentation", type=str, default="default", help="Path to the customer segmentation CSV file or 'default' to use the latest W&B artifact")
    args = parser.parse_args()
    main(args)
