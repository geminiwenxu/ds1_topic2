import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from config import Config


def read_original_data(path):
    df = pd.read_csv(path, delimiter=';')
    df1 = df[df.y == 'yes']
    df2 = df[df.y == 'no']
    size_pos = len(df1.index)
    size_neg = len(df2.index)
    if size_neg > size_pos:
        ratio = size_neg / size_pos
    else:
        ratio = size_pos / size_neg
    return df, size_pos, size_neg, ratio


def data_ratio(size_pos, size_neg):
    ls_labels = ["positive samples=4640 ", "negative samples=36548"]
    share = [size_pos, size_neg]
    figureObject, axesObject = plt.subplots()
    axesObject.pie(share, labels=ls_labels, autopct='%1.2f', startangle=120)
    axesObject.axis('equal')
    plt.title("class distribution")
    return figureObject


def convert_to_numerical(df):
    # one-hot encoding
    dum_df = pd.get_dummies(data=df, columns=['job', 'marital', 'contact', 'default', 'housing', 'loan'])
    one_df = dum_df.drop(['month', 'day_of_week'], axis=1)

    # label encoding
    label_encoder = LabelEncoder()
    one_df['education_cat'] = label_encoder.fit_transform(df['education'])
    one_df['poutcome_cat'] = label_encoder.fit_transform(df['poutcome'])
    one_df['y_cat'] = label_encoder.fit_transform(df['y'])
    result = one_df.drop(['education', 'poutcome', 'y'], axis=1)
    # cols = result.columns.tolist()
    # result.to_csv('Data/numerical_data.csv', header=cols, index=False)
    return result


if __name__ == '__main__':
    path = Config.original_file_path
    df, size_pos, size_neg, ratio = read_original_data(path)
    print("size of positive class: ", size_pos, "size of negative class: ", size_neg, "The ratio between two classes",
          ratio)
    data_ratio(size_pos, size_neg)
    result = convert_to_numerical(df)
    print(result)
