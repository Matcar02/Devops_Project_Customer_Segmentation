import seaborn as sns
import matplotlib as plt
import pandas as pd

def pca_vs_spectral(dfpcaf, insights):

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))= plt.subplots(2,3, figsize=(30,28))
    ins5 = dfpcaf[(dfpcaf['kmeansclustersPCA'] == 0) | (dfpcaf['kmeansclustersPCA'] == 0)][insights]

    sns.countplot(x= ins5["payment_type"], ax= ax1)

    ord1 = pd.value_counts(ins5['customer_state']).iloc[:20].index
    sns.countplot(y = ins5["customer_state"], order = ord1, ax= ax2)

    ord2 = pd.value_counts(ins5['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins5['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")

    ins6 = dfpcaf[(dfpcaf['sp_clusters'] == 2 ) | (dfpcaf['sp_clusters'] == 3 )][insights]

    sns.countplot(x= ins6["payment_type"], ax= ax4)

    ord1 = pd.value_counts(ins6['customer_state']).iloc[:20].index
    sns.countplot(y = ins6["customer_state"], order = ord1, ax= ax5)

    ord2 = pd.value_counts(ins6['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins6['product_category_name_english'], order= ord2, ax = ax6)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")

    plt.show()