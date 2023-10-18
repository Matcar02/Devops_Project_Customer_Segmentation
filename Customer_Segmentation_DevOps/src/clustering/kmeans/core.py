from sklearn.cluster import KMeans 


def clustering(clusters1, algorithm1, rand_state, X, df):
    kmeans = KMeans(n_clusters = clusters1, init = 'k-means++', random_state = rand_state, algorithm = algorithm1, n_init = 3)
    y_kmeans = kmeans.fit_predict(X)

    rfmcopy = df.copy()
    rfmcopy['kmeans_cluster'] = y_kmeans

    return rfmcopy



def choose(rfm_dataset, X):
    nclusters = [3,4,5,6]
    algo = ["lloyd","elkan"]
    inp1 = int(input("please insert the number of clusters you would like to have:"))
    if inp1 not in nclusters:
        print("not reccomended nclusters, insert integer between 3 and 6 for an optimal result")
        inp1 = int(input("please insert the number of clusters you would like to have:"))
    inp2 = str(input("choose lloyd or elkan:"))
    if inp2 not in algo:
        print("Please type correctly the algorithm to use!")        
        inp2 = str(input("choose lloyd or elkan:"))

    inp3 = int(input("please insert a random state(integer!):"))
    if type(inp3) != int:
        print("Random state must be an integer! Please reinsert")
        inp3 = int(input("reinsert an random integer:"))

    rfmcopy = clustering(inp1, inp2, inp3, X, rfm_dataset)
    plot_clusters(rfmcopy, inp1)
    return rfmcopy