import seaborn as sns
import pandas as pd
import matplotlib as plt


def ann_insigths(dfann, dfpcaf):
    dfann[dfann['clustersann'] == 0].describe()
    dfann[dfann['clustersann'] == 1].describe()
    dfann[dfann['clustersann'] == 2].describe()
    dfann[dfann['clustersann'] == 3].describe()
    dfpcaf["clustersann"]=dfann["clustersann"]
    dfpcaf.head()

    insights=["payment_type","payment_installments", "customer_state","product_category_name_english"]

    ins1=dfpcaf[dfpcaf["clustersann"]==0][insights]
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))

    sns.countplot(x= ins1["payment_type"], ax= ax1)

    ord1 = pd.value_counts(ins1['customer_state']).iloc[:20].index
    sns.countplot(y = ins1["customer_state"], order = ord1, ax= ax2)

    ord2 = pd.value_counts(ins1['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins1['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")

    plt.show()


    ins2=dfpcaf[dfpcaf["clustersann"]==1][insights]
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))


    sns.countplot(x= ins2["payment_type"], ax= ax1)


    ord1 = pd.value_counts(ins2['customer_state']).iloc[:20].index
    sns.countplot(y = ins2["customer_state"], order = ord1, ax= ax2)


    ord2 = pd.value_counts(ins2['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins2['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()


    ins3=dfpcaf[dfpcaf["clustersann"]==2][insights]

    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))

    sns.countplot(x= ins3["payment_type"], ax= ax1)

    ord1 = pd.value_counts(ins3['customer_state']).iloc[:20].index
    sns.countplot(y = ins3["customer_state"], order = ord1, ax= ax2)

    ord2 = pd.value_counts(ins3['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins3['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")

    plt.show()

    ins4=dfpcaf[dfpcaf["clustersann"]==3][insights]

    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(15,5))

    sns.countplot(x= ins1["payment_type"], ax= ax1)

    ord1 = pd.value_counts(ins4['customer_state']).iloc[:20].index
    sns.countplot(y = ins1["customer_state"], order = ord1, ax= ax2)

    ord2 = pd.value_counts(ins4['product_category_name_english']).iloc[:15].index 
    axs = sns.countplot(ins4['product_category_name_english'], order= ord2, ax = ax3)
    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)
    axs.set_title(label="Top 10 category in cluster")


    plt.show()