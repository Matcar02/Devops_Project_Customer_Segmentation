import logging
import seaborn as sns
import pandas as pd

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
    
    # Compute and log the correlation matrix
    corr_matrix = df[columns].corr()
    logging.info('Correlation matrix generated for columns: %s', ', '.join(columns))
    logging.info('\n' + str(corr_matrix))

    logging.info('corr function completed.')
