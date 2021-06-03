import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from classifier import combine, random_forest
import numpy as np
from config import Config
from preprocessing import read_original_data, data_ratio, convert_to_numerical
from undersample import combine_data, read_numerical_data


def front_end_rf(numerical_df, minority_class, majority_class):
    option = st.sidebar.selectbox(
        'Metrics selection (k=10) ',
        ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
    st.header('selected metric：' + option)
    cols = numerical_df.columns.tolist()
    ls_paths = ["imbalanced", "centroids", "random",
                "one_neig", "n_neig"]
    if option == 'Accuracy':
        all_ls_accuracy = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            all_ls_accuracy.append(ls_accuracy[0])
        fig, ax = plt.subplots()
        y_pos = np.arange(len(ls_paths))
        ax.bar(y_pos, all_ls_accuracy)
        plt.xticks(y_pos, ls_paths)
        plt.ylabel("accuracy")
        st.pyplot(fig)

    elif option == 'Precision':
        all_ls_precision = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            all_ls_precision.append(ls_precision[0])
        fig, ax = plt.subplots()
        y_pos = np.arange(len(ls_paths))
        ax.bar(y_pos, all_ls_precision)
        plt.xticks(y_pos, ls_paths)
        plt.ylabel("precision")
        st.pyplot(fig)

    elif option == 'Recall':
        all_ls_recall = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            all_ls_recall.append(ls_recall[0])
        fig, ax = plt.subplots()
        y_pos = np.arange(len(ls_paths))
        ax.bar(y_pos, all_ls_recall)
        plt.xticks(y_pos, ls_paths)
        plt.ylabel("recall")
        st.pyplot(fig)

    elif option == 'F1 score':
        all_ls_f1_score = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            all_ls_f1_score.append(ls_f1[0])
        fig, ax = plt.subplots()
        y_pos = np.arange(len(ls_paths))
        ax.bar(y_pos, all_ls_f1_score)
        plt.xticks(y_pos, ls_paths)
        plt.ylabel("f1")
        st.pyplot(fig)

    else:
        all_ls_auc = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            all_ls_auc.append(ls_auc[0])
        fig, ax = plt.subplots()
        y_pos = np.arange(len(ls_paths))
        ax.bar(y_pos, all_ls_auc)
        plt.xticks(y_pos, ls_paths)
        plt.ylabel("accuracy")
        st.pyplot(fig)
    # ---------------------------------------------------------------------------------------
    option = st.sidebar.selectbox(
        'Methods selection',
        ["imbalanced", "centroids", "random", "one_neig", "n_neig"])
    st.header('selected method：' + option)

    if option == "centroids":
        k = None
        total_df = combine_data(k, option, minority_class, majority_class)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
        ls_metrics = [ls_accuracy[0], ls_precision[0], ls_recall[0], ls_f1[0], ls_auc[0]]
        fig, ax = plt.subplots()
        y_pos = np.arange(5)
        ax.bar(y_pos, ls_metrics)
        plt.xticks(y_pos, ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
        plt.ylabel("centroids")
        st.pyplot(fig)

    elif option == "one_neig":
        k = None
        total_df = combine_data(k, option, minority_class, majority_class)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
        ls_metrics = [ls_accuracy[0], ls_precision[0], ls_recall[0], ls_f1[0], ls_auc[0]]
        fig, ax = plt.subplots()
        y_pos = np.arange(5)
        ax.bar(y_pos, ls_metrics)
        plt.xticks(y_pos, ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
        plt.ylabel("one_neig")
        st.pyplot(fig)

    elif option == "random":
        num_cluster = st.number_input("Number of cluster")
        if num_cluster > 1:
            num_cluster = np.floor(num_cluster)
            num_cluster = int(num_cluster)
            total_df = combine_data(num_cluster, option, minority_class, majority_class)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            ls_metrics = [ls_accuracy[0], ls_precision[0], ls_recall[0], ls_f1[0], ls_auc[0]]
            fig, ax = plt.subplots()
            y_pos = np.arange(5)
            ax.bar(y_pos, ls_metrics)
            plt.xticks(y_pos, ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
            plt.ylabel("random")
            st.pyplot(fig)

    elif option == "n_neig":
        num_cluster = st.number_input("Number of cluster")
        if num_cluster > 1:
            num_cluster = np.floor(num_cluster)
            num_cluster = int(num_cluster)
            total_df = combine_data(num_cluster, option, minority_class, majority_class)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
            ls_metrics = [ls_accuracy[0], ls_precision[0], ls_recall[0], ls_f1[0], ls_auc[0]]
            fig, ax = plt.subplots()
            y_pos = np.arange(5)
            ax.bar(y_pos, ls_metrics)
            plt.xticks(y_pos, ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
            plt.ylabel("n_neig")
            st.pyplot(fig)
    else:
        k = None
        total_df = combine_data(k, option, minority_class, majority_class)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = random_forest(total_df)
        ls_metrics = [ls_accuracy[0], ls_precision[0], ls_recall[0], ls_f1[0], ls_auc[0]]
        fig, ax = plt.subplots()
        y_pos = np.arange(5)
        ax.bar(y_pos, ls_metrics)
        plt.xticks(y_pos, ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'])
        plt.ylabel("imbalanced")
        st.pyplot(fig)
