import seaborn as sns
import matplotlib as plt
import numpy as np

def visualize_data(rfm_dataset):
    sns.pairplot(rfm_dataset)
    plot1 = sns.lineplot(x="Recency", y="Monetary value", data=rfm_dataset.sort_values(by=["Recency"], ascending=False))

    plot2 = sns.histplot(data = rfm_dataset['Frequency'], discrete= True)
    return plot1, plot2  


def plot_average_spending_by_frequency(rfm_dataset):
    frd = rfm_dataset.groupby(['Frequency'])['Monetary value'].mean().reset_index(name='Average Spending by frequency')
    sns.lineplot(data = frd, x = "Frequency", y = "Average Spending by frequency")

def plot_payment_value_distribution(rfm_dataset):
    LogMin, LogMax = np.log10(rfm_dataset['Monetary value'].min()),np.log10(rfm_dataset['Monetary value'].max())
    newbins = np.logspace(LogMin, LogMax, 4)
    sns.distplot(rfm_dataset['Monetary value'], kde=False, bins=newbins) 

