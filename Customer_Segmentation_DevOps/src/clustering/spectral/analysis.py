import logging
import pandas as pd

# Configure the logging
logging.basicConfig(level=logging.INFO) 

def show_silscores(silscores):
    """
    Displays silhouette scores in a Pandas DataFrame.

    Args:
    silscores (dict): A dictionary containing silhouette scores.
    """

    logging.info("Displaying silhouette scores")
    
    dfscores = pd.DataFrame(silscores, index=[0])
    print(dfscores)
    
    logging.info("Silhouette scores displayed")
