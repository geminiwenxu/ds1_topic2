import pandas as pd


class Config:
    original_file_path = "Data/bank-additional-full.csv"
    numerical_data_path = 'Data/numerical_data.csv'
    numerical_df = pd.read_csv("Data/numerical_data.csv", delimiter=',')
    cols = numerical_df.columns.tolist()  # 40 cols including the y labels
    training_cols = cols[:39]
