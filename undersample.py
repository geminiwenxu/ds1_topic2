import pandas as pd
from sklearn.cluster import KMeans
from numpy import savetxt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors


def read_data(path):
    df = pd.read_csv(path, delimiter=',')
    df1 = df[df.y_cat == 1]
    df2 = df[df.y_cat == 0]
    print("size of positive classs", len(df1.index))
    print("size of negative classs", len(df2.index))
    if len(df1.index) > len(df2.index):
        majority_class = df1
        size_minority = len(df2.index)
        minority_class = df2
    else:
        majority_class = df2
        size_minority = len(df1.index)
        minority_class = df1
    return majority_class, minority_class, size_minority


def centroids(size_minority, majority_class):
    km = KMeans(n_clusters=size_minority, init='k-means++').fit(majority_class)
    centroids = km.cluster_centers_
    savetxt('Data/centroids.csv', centroids, delimiter=',')
    df = pd.read_csv('Data/centroids.csv', delimiter=',')
    print(df.shape)


def random(K, size_minority, majority_class):
    km = KMeans(n_clusters=K, init='k-means++').fit(majority_class)
    clusters_index = km.fit_predict(majority_class)
    majority_class['cluster'] = clusters_index
    majority_class_cluster = majority_class
    size_instance = int(size_minority / K)
    random_sample = pd.DataFrame()
    for i in range(K):
        random_sample = random_sample.append(
            majority_class_cluster[majority_class_cluster.cluster == i].sample(n=size_instance, replace=True),
            ignore_index=True)
    random_sample.to_csv('Data/random.csv')
    df = pd.read_csv('Data/random.csv', delimiter=',')
    print(df.shape)


def top_one(size_minority, majority_class, cols):
    km = KMeans(n_clusters=size_minority, init='k-means++').fit(majority_class)
    centroids = km.cluster_centers_
    arr_majority_class = majority_class.to_numpy()
    # print(len(arr_majority_class[0]))
    closest, _ = pairwise_distances_argmin_min(centroids, arr_majority_class)
    one_neig = pd.DataFrame(columns=cols)
    for index, i in enumerate(closest):
        neighbor = arr_majority_class[i]
        one_neig.loc[index] = neighbor.tolist()
    one_neig.to_csv('Data/one_neig.csv')
    df = pd.read_csv('Data/one_neig.csv', delimiter=',')
    print(df.shape)


def top_n(K, size_minority, majority_class, cols):
    km = KMeans(n_clusters=K, init='k-means++').fit(majority_class)
    centroids = km.cluster_centers_
    size_instance = int(size_minority / K)
    arr_majority_class = majority_class.to_numpy()
    neighbors = NearestNeighbors(n_neighbors=size_instance).fit(majority_class)
    TopN_neigbours = neighbors.kneighbors(centroids, return_distance=False)
    n_neig = pd.DataFrame(columns=cols)
    ls_index = []
    ls_neig = []
    for i in range(K):
        for j in range(size_instance):
            index = TopN_neigbours[i][j]
            ls_index.append(index)
            neighbor = arr_majority_class[index]
            ls_neig.append(neighbor)
    for q in range(size_minority):
        n_neig.loc[q] = ls_neig[q].tolist()
    n_neig.to_csv('Data/n_neig.csv')
    df = pd.read_csv('Data/n_neig.csv', delimiter=',')
    print(df.shape)


if __name__ == '__main__':
    K = 10
    path = "Data/numerical_data.csv"
    majority_class, minority_class, size_minority = read_data(path)
    minority_class.to_csv('Data/minority_class')
    cols = majority_class.columns.tolist()
    print(cols)
    print(len(cols))

    centroids(size_minority, majority_class)
    random(K, size_minority, majority_class)
    top_one(size_minority, majority_class, cols)
    top_n(K, size_minority, majority_class, cols)
