import BalanceScale.backend as backend
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def bs():
    manager = backend.Manager()
    df = manager.df
    
    #imbalanced_data_path = Config.original_file_path
    #df, size_pos, size_neg, ratio = read_original_data(imbalanced_data_path)
    st.header('Imbalanced dataset')
    st.dataframe(df)
    st.text('Number of members for each class:')
    st.dataframe(df['class'].value_counts())
    #st.text(ratio)

    fig = manager.plot_counts(df)
    st.pyplot(fig)

    ## ---------------------------------------------------------------------------------------
    option = st.sidebar.selectbox(
        'Algo selection ',
        ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes', 'Gradient Boosting'])
    st.header('selected algo：' + option)
    k = st.sidebar.slider('k', 0, 49, 7)

    if option == "Logistic Regression":
        clf = LogisticRegression()
        df_results = manager.random_unbalanced(clf, df, manager.names)
        df_results = manager.random_sampling(clf, df, df_results)
        df_results = manager.centroids(clf, df, df_results)
        df_results = manager.km_random(clf, df, df_results, k)
        st.header('Results for different balancing methods:')
        st.dataframe(df_results.T)
        fig = manager.plot_results(df_results)
        st.pyplot(fig)

    elif option == "Support Vector Machine":
        clf = SVC()
        df_results = manager.random_unbalanced(clf, df, manager.names)
        df_results = manager.random_sampling(clf, df, df_results)
        df_results = manager.centroids(clf, df, df_results)
        df_results = manager.km_random(clf, df, df_results, k)
        st.header('Results for different balancing methods:')
        st.dataframe(df_results.T)
        fig = manager.plot_results(df_results)
        st.pyplot(fig)

    elif option == "Naive Bayes":
        clf = GaussianNB()
        df_results = manager.random_unbalanced(clf, df, manager.names)
        df_results = manager.random_sampling(clf, df, df_results)
        df_results = manager.centroids(clf, df, df_results)
        df_results = manager.km_random(clf, df, df_results, k)
        st.header('Results for different balancing methods:')
        st.dataframe(df_results.T)
        fig = manager.plot_results(df_results)
        st.pyplot(fig)


    ## ---------------------------------------------------------------------------------------
    #st.header('Logistic Regression: 100 iterations and k=10')
    #df = pd.DataFrame(np.array([["Original Imbalanced", 0.91, 0.66, 0.4, 0.50, 0.69, ],
    #                            ["K-means++ and Centroids", 0.73, 0.72, 0.77, 0.74, 0.73],
    #                            ["K-means++ and Random sampling", 0.68, 0.69, 0.65, 0.67, 0.68],
    #                            ["K-means++ and Top1 centroids’ nearest neighbor", 0.73, 0.72, 0.76, 0.74, 0.73],
    #                            ["K-means++ and TopN centroids’ nearest neighbor", 0.71, 0.73, 0.68, 0.70, 0.71]]),
    #                  columns=(["dataset", "accuracy", "precision", "recall", "f1", "auc"]))
    #st.table(df)
