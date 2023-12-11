import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import os
import argparse

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
    sns.pairplot(df[columns])
    plt.savefig("pairplot.png")
    plt.close()
    
    # Compute and log the correlation matrix
    corr_matrix = df[columns].corr()
    logging.info('Correlation matrix generated for columns: %s', ', '.join(columns))
    logging.info('\n' + str(corr_matrix))

    logging.info('corr function completed.')
    return corr_matrix

def main(filepath=None):
    # Load your data
    

    # Log the description and correlation matrix as an artifact
    with mlflow.start_run() as run:

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        default_filepath = os.path.join(data_dir, 'cleaned_data.csv')

        # If filepath is "default" or not provided, use the default filepath
        if not filepath or filepath == "default":
            filepath = default_filepath

        df = pd.read_csv(filepath)

        # Perform your statistical analysis functions
        description = describe_dataset(df)
        corr_matrix = corr(df)


        description.to_csv("dataset_description.csv")
        mlflow.log_artifact("dataset_description.csv", artifact_path="dataset_stats")

        corr_matrix.to_csv("correlation_matrix.csv")
        mlflow.log_artifact("correlation_matrix.csv", artifact_path="dataset_stats")

        mlflow.log_artifact("pairplot.png", artifact_path="visuals")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis of data")
    parser.add_argument("--filepath", type=str, help="Path to the input CSV file", required=True)
    args = parser.parse_args()
    main(filepath=args.filepath)
