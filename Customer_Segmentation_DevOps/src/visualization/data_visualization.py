import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_data(rfm_dataset):
    logging.info("Visualizing data...")
    plot3 = sns.pairplot(rfm_dataset)
    plt.show()
    plot1 = sns.lineplot(x="Recency", y="Monetary value", data=rfm_dataset.sort_values(by=["Recency"], ascending=False))
    plt.show()
    plot2 = sns.histplot(data=rfm_dataset['Frequency'], discrete=True)
    plt.show()
    logging.info("Data visualization complete.")
    return plot1, plot2, plot3 


def plot_average_spending_by_frequency(rfm_dataset):
    logging.info("Plotting average spending by frequency...")
    frd = rfm_dataset.groupby(['Frequency'])['Monetary value'].mean().reset_index(name='Average Spending by frequency')
    #sns.lineplot(data=frd, x="Frequency", y="Average Spending by frequency")
    sns.scatterplot(data=frd, x="Frequency", y="Average Spending by frequency", s=100, color='red')
    plt.title("Average Spending by Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Average Spending")
    plt.show()
    logging.info("Average spending by frequency plotted.")


def plot_payment_value_distribution(rfm_dataset):
    logging.info("Plotting payment value distribution...")
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()), np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.distplot(rfm_dataset['Monetary value'], kde=False, bins=newbins)
    plt.title("Payment value distribution")
    plt.ylabel("Count")
    plt.show()
    logging.info("Payment value distribution plotted.")


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
    sns.lineplot(x="Recency", y="Monetary value", data=timeordered)
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
