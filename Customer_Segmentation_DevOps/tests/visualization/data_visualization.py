import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
import os 
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_data(rfm_dataset):
    logging.info("Visualizing data...")
    sns.pairplot(rfm_dataset)
    plt.title("Pairplot")
    plt.show()
    
    #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'pairplot_{now}.png'
        #fig1 = plot3.get_figure()
        #fig1.savefig(os.path.join(reports_path, 'figures', filename))
        plt.savefig(os.path.join(reports_path, 'figures', filename)) 
    except:
        logging.error('Error saving plot.')
        return

    plt.close()
        
    
    sns.lineplot(x="Recency", y="Monetary value", data=rfm_dataset.sort_values(by=["Recency"], ascending=False))
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

    sns.histplot(data=rfm_dataset['Frequency'], discrete=True)
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
    return plt  


def plot_average_spending_by_frequency(rfm_dataset):
    logging.info("Plotting average spending by frequency...")
    frd = rfm_dataset.groupby(['Frequency'])['Monetary value'].mean().reset_index(name='Average Spending by frequency')
    sns.scatterplot(data=frd, x="Frequency", y="Average Spending by frequency", s=100, color='red')
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
    except:
        logging.error('Error saving plot.')
        return

    plt.close()


def plot_payment_value_distribution(rfm_dataset):
    logging.info("Plotting payment value distribution...")
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.distplot(rfm_dataset['Monetary value'], kde=False, bins=newbins)
    plt.title("Payment value distribution")
    plt.ylabel("Count")
    plt.show()
    logging.info("Payment value distribution plotted.")

   #saving plot
    logging.info("Saving plot...")
    current_path = os.getcwd()
    reports_path = os.path.abspath(os.path.join(current_path, '..', 'reports'))
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    try:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'plot_payment_value_distribution{now}.png'
        plt.savefig(os.path.join(reports_path, 'figures', filename))
    except:
        logging.error('Error saving plot.')
        return

def freq(rfm_dataset):
    logging.info("Generating frequency plots...")
    sns.histplot(data=rfm_dataset['Frequency'], discrete=True)
    plt.title("Frequency plot")
    plt.show()
    logging.info("Frequency plots generated.")


def pairplot(rfm_dataset):
    logging.info("Generating pairplot...")
    sns.pairplot(rfm_dataset)
    plt.title("Pairplot")
    plt.show()
    logging.info("Pairplot generated.")


def spending_by_recency(rfm_dataset):
    logging.info("Generating spending by recency plot...")
    timeordered = rfm_dataset.sort_values(by=["Recency"], ascending=False)
    plt1 = sns.lineplot(x="Recency", y="Monetary value", data=timeordered)
    plt.title("Spending by recency")
    plt.show()
    logging.info("Spending by recency plotted.")


def payments_distribution(rfm_dataset):
    logging.info("Plotting payments distribution...")
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.displot(rfm_dataset['Monetary value'], kde=False, bins=newbins)
    plt.title("Payments distribution")
    plt.show()
    logging.info("Payments distribution plotted.")
