import logging
import os
import sys
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def silhouette_score_df(silscores):
    """
    Calculate silhouette score and save DataFrame to CSV.
    """
    logging.info('Getting DataFrame...')
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    logging.info('Saving DataFrame to CSV...')

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'silscores_{now}.csv'
        silscores.to_csv(os.path.join(reports_path, 'dataframes', filename), index=False)
        sns.displot(silscores)
        plt.show()
    except Exception as e:
        logging.error('Error saving DataFrame to CSV.')
        logging.error(str(e))
        return None

    logging.info('DataFrame saved successfully.')
    return silscores


