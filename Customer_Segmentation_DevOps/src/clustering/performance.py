import pandas as pd 
import logging 
import os
import sys
from datetime import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
 


def silhouette_score_df(silscores):
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
    except:
        logging.error('Error saving DataFrame to CSV.')
        return
        
    logging.info('DataFrame saved successfully.')
    return silscores 