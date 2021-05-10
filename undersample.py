import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from numpy import savetxt
from sklearn.metrics import pairwise_distances_argmin_min
import csv
from sklearn.neighbors import NearestNeighbors


def read_data(path):
    df = pd.read_csv(path, delimiter=',')
    # print(df.head())
    df1 = df[df.y_cat == 1]
    print("positive class", len(df1.index))
    df2 = df[df.y_cat == 0]
    print("negative class", len(df2.index))
    if len(df1.index) > len(df2.index):
        majority_class = df1
        size_minority = len(df2.index)
        minority_class = df2
    else:
        majority_class = df2
        size_minority = len(df1.index)
        minority_class = df1
    return majority_class, size_minority, minority_class


def clustering(K, majority_class, cols):
    km = KMeans(n_clusters=K, init='k-means++', n_init=10).fit(majority_class)
    clusters_index = km.fit_predict(majority_class)
    majority_class['cluster'] = clusters_index
    majority_class_cluster = majority_class
    return majority_class_cluster, km


def clusters(K, majority_class_cluster):
    x = {}
    for i in range(K):
        x[i] = majority_class_cluster[majority_class_cluster.cluster == i]
    return x


def find_k_closest(centroids, data, TopN):
    neighbors = NearestNeighbors(n_neighbors=TopN).fit(data)
    TopN_neigbours = neighbors.kneighbors(centroids, return_distance=False)
    return TopN_neigbours


if __name__ == '__main__':
    path = "Data/numerical_data.csv"
    majority_class, size_minority, minority_class = read_data(path)

    minority_class.to_csv('minority_class')
    print(size_minority)
    print(len(majority_class.columns))

    # K-means ++ centroids
    # K = size_minority
    # print(majority_class.columns)
    # km = KMeans(n_clusters=K, init='k-means++', n_init=10).fit(majority_class)
    # centers = km.cluster_centers_
    # savetxt('centers.csv', centers, delimiter=',')
    df = pd.read_csv('Data/centers.csv', delimiter=',')
    print(df.shape)

    # K-means ++ random sampling
    # K = 10
    # majority_class_cluster, km = clustering(K, majority_class, majority_class.columns)
    # size_instance = int(size_minority / K)
    # print(size_instance)
    # random_sample = pd.DataFrame()
    # for i in range(K):
    #     random_sample = random_sample.append(
    #         majority_class_cluster[majority_class_cluster.cluster == i].sample(n=size_instance, replace=True),
    #         ignore_index=True)
    # random_sample.to_csv('random.csv')
    # df = pd.read_csv('/Users/wenxu/PycharmProjects/DS/random.csv', delimiter=',')
    # print(df.shape)

    # Top1 centroid's nearest neighbor
    # K = size_minority
    # majority_class_cluster, km = clustering(K, majority_class, majority_class.columns)
    # centers = km.cluster_centers_
    # cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
    #         'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
    #         'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
    #         'job_management', 'job_retired', 'job_self-employed', 'job_services',
    #         'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
    #         'marital_divorced', 'marital_married', 'marital_single',
    #         'marital_unknown', 'contact_cellular', 'contact_telephone',
    #         'default_no', 'default_unknown', 'default_yes', 'housing_no',
    #         'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
    #         'education_cat', 'poutcome_cat', 'y_cat']  # 40 cols
    # arr_majority_class = majority_class_cluster[cols].to_numpy()
    # closest, _ = pairwise_distances_argmin_min(centers,
    #                                            arr_majority_class)  # closest contains the index of the point in majority_class
    # one_neig = pd.DataFrame(columns=cols)
    # for index, i in enumerate(closest):
    #     neighbor = arr_majority_class[i]
    #     one_neig.loc[index] = neighbor.tolist()
    #     # print(neighbor)
    #     # print(len(neighbor))  # 40
    #     # print(type(neighbor))
    # # print(one_neig)
    # one_neig.to_csv('one_neig.csv')
    # df = pd.read_csv('/Users/wenxu/PycharmProjects/DS/one_neig.csv', delimiter=',')
    # print(df.columns)

    # TopN centroid's nearest neighbor
    # K = 10
    # majority_class_cluster, km = clustering(K, majority_class, majority_class.columns)
    # x = clusters(K, majority_class)
    # centers = km.cluster_centers_
    # TopN = int(size_minority / K)
    # cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
    #         'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
    #         'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
    #         'job_management', 'job_retired', 'job_self-employed', 'job_services',
    #         'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
    #         'marital_divorced', 'marital_married', 'marital_single',
    #         'marital_unknown', 'contact_cellular', 'contact_telephone',
    #         'default_no', 'default_unknown', 'default_yes', 'housing_no',
    #         'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
    #         'education_cat', 'poutcome_cat', 'y_cat']
    # arr_majority_class = majority_class_cluster[cols].to_numpy()
    #
    # nns = find_k_closest(centers, arr_majority_class, TopN)
    # n_neig = pd.DataFrame(columns=cols)
    # for i in range(K):
    #     for j in range(TopN):
    #         index = nns[i][j]
    #         neighbor = arr_majority_class[index]
    #         n_neig.loc[index] = neighbor.tolist()
    #         print(neighbor)
    # n_neig.to_csv('n_neig.csv')
    # df = pd.read_csv('/Users/wenxu/PycharmProjects/DS/n_neig.csv', delimiter=',')
    # print(df.shape)
    # print(df.columns)
