import logging
import seaborn as sns
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def describe_dataset(rfm_dataset):
    logging.info('Starting describe_dataset function...')
    
    # You can either log the description or continue to print it.
    description = rfm_dataset.describe()
    logging.info('\n' + str(description))
    # or if you prefer to print: 
    # print(description)

    logging.info('describe_dataset function completed.')

def corr(df):
    logging.info('Starting corr function...')
    
    columns = ["payment_type", "payment_installments", "payment_value"]
    
    logging.debug('Generating pairplot for columns: %s', ', '.join(columns))
    sns.pairplot(df[columns])
    
    corr_matrix = df[columns].corr()
    logging.info('Correlation matrix generated for columns: %s', ', '.join(columns))
    logging.info('\n' + str(corr_matrix))
    # or if you prefer to print:
    # print(corr_matrix)

    logging.info('corr function completed.')
