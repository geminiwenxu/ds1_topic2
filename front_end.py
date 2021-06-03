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


def bank_marketing():
    imbalanced_data_path = Config.original_file_path
    df, size_pos, size_neg, ratio = read_original_data(imbalanced_data_path)
    st.header('Imbalanced dataset')
    st.dataframe(df)
    st.text('This is the ratio of size_neg vs size_pos is: ')
    st.text(ratio)

    fig = data_ratio(size_pos, size_neg)
    st.pyplot(fig)

    numerical_df = convert_to_numerical(df)
    majority_class, minority_class, size_minority = read_numerical_data(numerical_df)
    st.header('numerical dataset')
    st.dataframe(numerical_df)

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

    st.header('100 iterations and k=10')
    df = pd.DataFrame(np.array([["Original Imbalanced", 0.91, 0.66, 0.4, 0.50, 0.69, ],
                                ["K-means++ and Centroids", 0.73, 0.72, 0.77, 0.74, 0.73],
                                ["K-means++ and Random sampling", 0.68, 0.69, 0.65, 0.67, 0.68],
                                ["K-means++ and Top1 centroids’ nearest neighbor", 0.73, 0.72, 0.76, 0.74, 0.73],
                                ["K-means++ and TopN centroids’ nearest neighbor", 0.71, 0.73, 0.68, 0.70, 0.71]]),
                      columns=(["dataset", "accuracy", "precision", "recall", "f1", "auc"]))
    st.table(df)


if __name__ == '__main__':
    st.title('DS1 T2-1')
    st.text("Group members: Wenxu Li")
    df_option = st.sidebar.selectbox(
        'Data set selection',
        ['Logistic Regression and UCI Bank Marketing dataset', 'Gradient Boosting Decision Tree and Adult Data Set',
         'Balance Scale Data Set with Naive Bayes', 'Wine Quality with SVM'])
    if df_option == 'Logistic Regression and UCI Bank Marketing dataset':
        bank_marketing()
