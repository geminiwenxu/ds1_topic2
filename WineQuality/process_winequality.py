import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import WineQuality.Methods
import WineQuality.measurement_table
import os
from pathlib import Path

def wine_quality():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    path = Path(__file__).cwd()
    red_wine_dir = os.path.join(path, 'Wine_Data', 'redwine-quality.csv')
    white_wine_dir = os.path.join(path, 'Wine_Data', 'whitewine-quality.csv')
    raw_df, count0 = WineQuality.Methods.combine_raw_data(data_red_dir=red_wine_dir, data_white_dir=white_wine_dir)
    st.header('Wine Quality data set')
    st.dataframe(raw_df)

    st.text('This is the ratio of size_neg vs size_pos is: ')
    f_dist = WineQuality.measurement_table.data_distrbution_concat([count0])
    st.pyplot(f_dist)

    algo_option = st.sidebar.selectbox(
        'Algo selection ',
        ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes', 'Random Forest'])
    st.header('selected algo：' + algo_option)

    num_cluster = st.number_input("Number of cluster")
    num_cluster = np.floor(num_cluster)
    num_cluster = int(num_cluster)
    acc1, df1_plot, train_df1, AUC1 = WineQuality.Methods.imbalance(raw_df, algo_option)
    if num_cluster > 1:
        num_cluster = np.floor(num_cluster)
        num_cluster = int(num_cluster)
        acc2, df2_plot, train_df2, AUC2 = WineQuality.Methods.RS_Kmean(raw_df, n_clusters=num_cluster,
                                                                   algo_name=algo_option)
        acc4, df4_plot, train_df4, AUC4 = WineQuality.Methods.n_near_Kmean(raw_df, n_clusters=num_cluster,
                                                                           n_neighbour='n', algo_name=algo_option)
        acc5, df5_plot, train_df5, AUC5 = WineQuality.Methods.n_near_Kmean(raw_df, n_clusters=num_cluster,
                                                                           n_neighbour=1, algo_name=algo_option)
        acc6, df6_plot, train_df6, AUC6 = WineQuality.Methods.centroid_Kmean(raw_df, num_cluster, algo_name=algo_option)

    acc3, df3_plot, train_df3, AUC3 = WineQuality.Methods.RS(raw_df, algo_option)
    df_acc = pd.DataFrame({'Accuracy': [acc1, acc2, acc3, acc4, acc5, acc6],
                           'Methods':['without undersampling','Random selection and Kmean++', 'Random sampling',
                                      'n near neighbours and Kmean++', 'top 1 near neighbours and Kmean++',
                                      'centroid and Kmean++']})
    option = st.sidebar.selectbox(
        'Metrics selection',
        ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
    st.header('selected metric：' + option)
    if option == 'Accuracy':
        f, ax = WineQuality.measurement_table.make_seaborn_scatter_plot(df_acc, 'Methods', 'Accuracy', 'strip', 'Methods',rotate_xlabel=True, fontsize=15)
        ax.set_xlabel('Methods', fontsize=15)
        ax.set_ylabel('Accuracy', fontsize=15)
        st.pyplot(f)
        pass
    elif option == 'Precision':
        f, ax = plt.subplots(3, 2, figsize=(14, 8))
        ax[0, 0] = WineQuality.measurement_table.bar(df1_plot, 'index', 'precision', ax[0, 0], f, 'Precision', 'Wine Quality', 'imbalance data')
        ax[0, 1] = WineQuality.measurement_table.bar(df2_plot, 'index', 'precision', ax[0, 1], f, 'Precision', 'Wine Quality',
                       'Random selection after Kmean++')
        ax[1, 0] = WineQuality.measurement_table.bar(df3_plot, 'index', 'precision', ax[1, 0], f, 'Precision', 'Wine Quality', 'Random selection')
        ax[1, 1] = WineQuality.measurement_table.bar(df4_plot, 'index', 'precision', ax[1, 1], f, 'Precision', 'Wine Quality',
                       'n neighbors selection after Kmean++')
        ax[2, 0] = WineQuality.measurement_table.bar(df5_plot, 'index', 'precision', ax[2, 0], f, 'Precision', 'Wine Quality',
                       'top 1 neighbors selection after Kmean++')
        ax[2, 1] = WineQuality.measurement_table.bar(df6_plot, 'index', 'precision', ax[2, 1], f, 'Precision', 'Wine Quality', 'centroid')
        f.tight_layout()
        st.pyplot(f)

    elif option == 'Recall':
        f, ax = plt.subplots(3, 2, figsize=(14, 8))
        ax[0, 0] = WineQuality.measurement_table.bar(df1_plot, 'index', 'recall', ax[0, 0], f, 'Recall', 'Wine Quality', 'imbalance data')
        ax[0, 1] = WineQuality.measurement_table.bar(df2_plot, 'index', 'recall', ax[0, 1], f, 'Recall', 'Wine Quality',
                       'Random selection after Kmean++')
        ax[1, 0] = WineQuality.measurement_table.bar(df3_plot, 'index', 'recall', ax[1, 0], f, 'Recall', 'Wine Quality', 'Random selection')
        ax[1, 1] = WineQuality.measurement_table.bar(df4_plot, 'index', 'recall', ax[1, 1], f, 'Recall', 'Wine Quality',
                       'top n neighbors selection after Kmean++')
        ax[2, 0] = WineQuality.measurement_table.bar(df5_plot, 'index', 'recall', ax[2, 0], f, 'Recall', 'Wine Quality',
                       'top 1 neighbors selection after Kmean++')
        ax[2, 1] = WineQuality.measurement_table.bar(df6_plot, 'index', 'recall', ax[2, 1], f, 'Recall', 'Wine Quality', 'centroid')
        f.tight_layout()
        st.pyplot(f)

    elif option == 'F1 score':
        f, ax = plt.subplots(3, 2, figsize=(14, 8))
        ax[0, 0] = WineQuality.measurement_table.bar(df1_plot, 'index', 'f1-score', ax[0, 0], f, 'f1-score', 'Wine Quality', 'imbalance data')
        ax[0, 1] = WineQuality.measurement_table.bar(df2_plot, 'index', 'f1-score', ax[0, 1], f, 'f1-score', 'Wine Quality',
                       'Random selection after Kmean++')
        ax[1, 0] = WineQuality.measurement_table.bar(df3_plot, 'index', 'f1-score', ax[1, 0], f, 'f1-score', 'Wine Quality', 'Random selection')
        ax[1, 1] = WineQuality.measurement_table.bar(df4_plot, 'index', 'f1-score', ax[1, 1], f, 'f1-score', 'Wine Quality',
                       'n neighbors selection after Kmean++')
        ax[2, 0] = WineQuality.measurement_table.bar(df5_plot, 'index', 'f1-score', ax[2, 0], f, 'f1-score', 'Wine Quality',
                       'top 1 neighbors selection after Kmean++')
        ax[2, 1] = WineQuality.measurement_table.bar(df6_plot, 'index', 'f1-score', ax[2, 1], f, 'f1-score', 'Wine Quality', 'centroid')
        f.tight_layout()
        st.pyplot(f)
    else:
        f, ax = plt.subplots(3, 2, figsize=(14, 8))
        ax[0, 0] = WineQuality.measurement_table.bar(AUC1, 'index', 'AUC', ax[0, 0], f, 'AUC', 'Wine Quality', 'imbalance data')
        ax[0, 1] = WineQuality.measurement_table.bar(AUC2, 'index', 'AUC', ax[0, 1], f, 'AUC', 'Wine Quality', 'Random selection after Kmean++')
        ax[1, 0] = WineQuality.measurement_table.bar(AUC3, 'index', 'AUC', ax[1, 0], f, 'AUC', 'Wine Quality', 'Random selection')
        ax[1, 1] = WineQuality.measurement_table.bar(AUC4, 'index', 'AUC', ax[1, 1], f, 'AUC', 'Wine Quality', 'n neighbors selection after Kmean++')
        ax[2, 0] = WineQuality.measurement_table.bar(AUC5, 'index', 'AUC', ax[2, 0], f, 'AUC', 'Wine Quality',
                       'top 1 neighbors selection after Kmean++')
        ax[2, 1] = WineQuality.measurement_table.bar(AUC6, 'index', 'AUC', ax[2, 1], f, 'AUC', 'Wine Quality', 'centroid')
        f.tight_layout()
        st.pyplot(f)