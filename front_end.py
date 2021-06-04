import streamlit as st
import pandas as pd
import numpy as np
from config import Config
from preprocessing import read_original_data, data_ratio, convert_to_numerical
from undersample import combine_data, read_numerical_data
from front_end_lr import front_end_lg
from front_end_svm import front_end_svm
from front_end_nb import front_end_nb
from front_end_db import front_end_db


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
    # ---------------------------------------------------------------------------------------
    option = st.sidebar.selectbox(
        'Algo selection ',
        ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes', 'Gradient Boosting'])
    st.header('selected algo：' + option)

    if option == "Logistic Regression":
        front_end_lg(numerical_df, minority_class, majority_class)

    elif option == "Support Vector Machine":
        front_end_svm(numerical_df, minority_class, majority_class)
    elif option == "Naive Bayes":
        front_end_nb(numerical_df, minority_class, majority_class)
    else:
        front_end_db(numerical_df, minority_class, majority_class)

    # ---------------------------------------------------------------------------------------
    st.header('Logistic Regression: 100 iterations and k=10')
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
    bank_marketing()
