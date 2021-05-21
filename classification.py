import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def logistic_reg(df):
    X = df[training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    print(accuracy, precision, recall, f1, auc)


def combine(path, cols):
    total_df = pd.DataFrame
    minority = pd.read_csv('Data/minority_class', delimiter=',')
    minority.drop(['Unnamed: 0'], axis=1, inplace=True)
    if path == "centroids_path":
        df = pd.read_csv("Data/centroids.csv", delimiter=',')
        df.columns = cols
        total_df = pd.concat([df, minority])
    elif path == "random_path":
        df = pd.read_csv('Data/random.csv', delimiter=',')
        df.drop(['Unnamed: 0', 'cluster'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "one_neig_path":
        df = pd.read_csv('Data/one_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "n_neig_path":
        df = pd.read_csv('Data/n_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    return total_df


if __name__ == '__main__':
    imbalanced_df = pd.read_csv("Data/numerical_data.csv", delimiter=',')
    print(imbalanced_df.shape)
    cols = imbalanced_df.columns.tolist()
    training_cols = cols[:39]

    print("metrics of imbalanced dataset")
    logistic_reg(imbalanced_df)

    ls_paths = ["centroids_path", "random_path",
                "one_neig_path", "n_neig_path"]
    for name in ls_paths:
        print(name)
        total_df = combine(name, cols)
        print(total_df.shape)
        print("metrics of" + name + "dataset")
        logistic_reg(total_df)
