import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from classifier import combine, logistic_reg
import numpy as np
from config import Config
from preprocessing import read_original_data, data_ratio, convert_to_numerical
from undersample import combine_data, read_numerical_data
from classifier import logistic_reg
import altair as alt


def front_end_lg(numerical_df, minority_class, majority_class):
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
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            all_ls_accuracy.append(ls_accuracy)
        chart_data = pd.DataFrame((np.array(all_ls_accuracy)).T, columns=ls_paths)
        st.line_chart(chart_data)

    elif option == 'Precision':
        all_ls_precision = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            all_ls_precision.append(ls_precision)
        chart_data = pd.DataFrame((np.array(all_ls_precision)).T, columns=ls_paths)
        st.line_chart(chart_data)

    elif option == 'Recall':
        all_ls_recall = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            all_ls_recall.append(ls_recall)
        chart_data = pd.DataFrame((np.array(all_ls_recall)).T, columns=ls_paths)
        st.line_chart(chart_data)

    elif option == 'F1 score':
        all_ls_f1_score = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            all_ls_f1_score.append(ls_f1)
        chart_data = pd.DataFrame((np.array(all_ls_f1_score)).T, columns=ls_paths)
        st.line_chart(chart_data)
    else:
        all_ls_auc = []
        for name in ls_paths:
            total_df = combine(name, cols)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            all_ls_auc.append(ls_auc)
        chart_data = pd.DataFrame((np.array(all_ls_auc)).T, columns=ls_paths)
        st.line_chart(chart_data)
    # ---------------------------------------------------------------------------------------
    option = st.sidebar.selectbox(
        'Methods selection',
        ["imbalanced", "centroids", "random", "one_neig", "n_neig"])
    st.header('selected method：' + option)

    if option == "centroids":
        k = None
        total_df = combine_data(k, option, minority_class, majority_class)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
        ls_metrics = np.vstack((np.array(ls_accuracy), np.array(ls_precision), np.array(ls_recall),
                                np.array(ls_f1), np.array(ls_auc)))
        chart_data = pd.DataFrame(ls_metrics.T, columns=["accuracy", "precision", "recall", "f1", "auc"])
        st.bar_chart(chart_data, width=100)

    elif option == "one_neig":
        k = None
        total_df = combine_data(k, option, minority_class, majority_class)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
        ls_metrics = np.vstack((np.array(ls_accuracy), np.array(ls_precision), np.array(ls_recall),
                                np.array(ls_f1), np.array(ls_auc)))
        chart_data = pd.DataFrame(ls_metrics.T, columns=["accuracy", "precision", "recall", "f1", "auc"])

        st.bar_chart(chart_data)

    elif option == "random":
        num_cluster = st.number_input("Number of cluster")
        if num_cluster > 1:
            num_cluster = np.floor(num_cluster)
            num_cluster = int(num_cluster)
            total_df = combine_data(num_cluster, option, minority_class, majority_class)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            ls_metrics = np.vstack((np.array(ls_accuracy), np.array(ls_precision), np.array(ls_recall),
                                    np.array(ls_f1), np.array(ls_auc)))
            chart_data = pd.DataFrame(ls_metrics.T, columns=["accuracy", "precision", "recall", "f1", "auc"])
            st.bar_chart(chart_data)

    elif option == "n_neig":
        num_cluster = st.number_input("Number of cluster")
        if num_cluster > 1:
            num_cluster = np.floor(num_cluster)
            num_cluster = int(num_cluster)
            total_df = combine_data(num_cluster, option, minority_class, majority_class)
            ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
            ls_metrics = np.vstack((np.array(ls_accuracy), np.array(ls_precision), np.array(ls_recall),
                                    np.array(ls_f1), np.array(ls_auc)))
            chart_data = pd.DataFrame(ls_metrics.T, columns=["accuracy", "precision", "recall", "f1", "auc"])
            st.bar_chart(chart_data)
    else:
        k = None
        total_df = combine_data(k, option, minority_class, majority_class)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = logistic_reg(total_df)
        ls_metrics = np.vstack((np.array(ls_accuracy), np.array(ls_precision), np.array(ls_recall),
                                np.array(ls_f1), np.array(ls_auc)))
        chart_data = pd.DataFrame(ls_metrics.T, columns=["accuracy", "precision", "recall", "f1", "auc"])
        st.bar_chart(chart_data)
