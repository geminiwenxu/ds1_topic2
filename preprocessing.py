import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data(path):
    df = pd.read_csv(path, delimiter=';')
    return df


if __name__ == '__main__':
    path = "Data/bank-additional-full.csv"
    df = read_data(path)

    # one-hot encoding
    dum_df = pd.get_dummies(data=df, columns=['job', 'marital', 'contact', 'default', 'housing', 'loan'])
    one_df = dum_df.drop(['month', 'day_of_week'], axis=1)

    # label encoding
    label_encoder = LabelEncoder()
    one_df['education_cat'] = label_encoder.fit_transform(df['education'])
    one_df['poutcome_cat'] = label_encoder.fit_transform(df['poutcome'])
    one_df['y_cat'] = label_encoder.fit_transform(df['y'])
    result = one_df.drop(['education', 'poutcome', 'y'], axis=1)

    # assigning cols
    cols = result.columns.tolist()
    print(cols)
    print(len(cols))
    result.to_csv('numerical_data.csv', header=cols, index=False)


