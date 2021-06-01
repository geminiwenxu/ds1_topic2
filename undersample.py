import pandas as pd
from sklearn.cluster import KMeans
from numpy import savetxt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from preprocessing import read_original_data, convert_to_numerical
from config import Config
from classification import logistic_reg


def read_numerical_data(numerical_df):
    df = numerical_df
    # df = pd.read_csv(path, delimiter=',')
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


def centroids(majority_class):
    size_minority = 4640
    km = KMeans(n_clusters=size_minority, init='k-means++').fit(majority_class)
    centroids = km.cluster_centers_
    savetxt('Data/centroids.csv', centroids, delimiter=',')
    df = pd.read_csv('Data/centroids.csv', delimiter=',')
    print(df.shape)


def random(K, majority_class):
    size_minority = 4640
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
    return random_sample
    # random_sample.to_csv('Data/random.csv')
    # df = pd.read_csv('Data/random.csv', delimiter=',')
    # print(df.shape)


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


def top_n(K, majority_class, cols):
    size_minority = 4640
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
    return n_neig
    # n_neig.to_csv('Data/n_neig.csv')
    # df = pd.read_csv('Data/n_neig.csv', delimiter=',')
    # print(df.shape)


def combine_data(k, path, minority, majority_class):
    imbalanced_df = pd.read_csv("/Users/wenxu/PycharmProjects/DS/Data/numerical_data.csv", delimiter=',')
    cols = imbalanced_df.columns.tolist()

    if path == "centroids":
        df = pd.read_csv("/Users/wenxu/PycharmProjects/DS/Data/centroids.csv", delimiter=',')
        df.columns = cols
        total_df = pd.concat([df, minority])
    elif path == "random":
        random_sample = random(k, majority_class)
        df = random_sample
        df.drop(['cluster'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "one_neig":
        df = pd.read_csv('/Users/wenxu/PycharmProjects/DS/Data/one_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "n_neig":
        n_neig_sample = top_n(k, majority_class, cols)
        df = n_neig_sample
        total_df = pd.concat([df, minority])
    else:
        df = pd.read_csv("/Users/wenxu/PycharmProjects/DS/Data/numerical_data.csv", delimiter=',')
        total_df = df
    print(total_df)
    return total_df


if __name__ == '__main__':
    K = None
    path = Config.original_file_path
    df, size_pos, size_neg, ratio = read_original_data(path)
    numerical_df = convert_to_numerical(df)
    majority_class, minority_class, size_minority = read_numerical_data(numerical_df)
    total_df = combine_data(K, "centroids", minority_class, majority_class)
    ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
    print(len(ls_accuracy))
