import logging
import pandas as pd

# Configure the logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

def show_silscores(silscores):
    logging.info("Displaying silhouette scores")
    
    dfscores = pd.DataFrame(silscores, index=[0])
    print(dfscores)
    
    logging.info("Silhouette scores displayed")
